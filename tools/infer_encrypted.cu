#include "fhe_circuit.h"
#include "keyser.h"

int main(int argc, char** argv) {
    int stage = (argc > 1) ? std::atoi(argv[1]) : 3;
    int nimg  = (argc > 2) ? std::atoi(argv[2]) : 1;
    if (argc > 3) { N = (uint32_t)std::atoi(argv[3]); SL = N / 2; }
    if (argc > 4) g_resident = std::atoi(argv[4]);
    int serround = (argc > 5) ? std::atoi(argv[5]) : 0;
    if (N < 2 * PER) { std::printf("ring too small for period %u\n", PER); return 2; }
    std::printf("ring n=%u  slots=%u  period=%u  log2(QP)=%u  rotation=%s\n",
                N, SL, PER, SIZEQ * 50 + SIZEP * 51,
                g_resident==2 ? "DEVICE" : (g_resident ? "resident" : "host"));

    MLP m;
    if (!mlp_load(m, "models/mlp.txt")) { std::printf("run train_mlp first\n"); return 2; }
    Dataset te;
    if (!load_mnist(te, "data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", 256)) {
        std::printf("run tools/fetch_mnist.sh\n"); return 2;
    }

    Ctx c;
    auto t0 = clk::now();
    setup(c);
    double t_setup = ms(t0);

    // Levels: L5 rotations -> rescale -> L4 relin -> rescale -> L3 rotations
    //         -> rescale -> L2 decrypt.
    Level L5 = make_level(c, 5), L4 = make_level(c, 4), L3 = make_level(c, 3),
          L2 = make_level(c, 2);

    t0 = clk::now();
    std::map<uint32_t, gpufhe::KeySwitchConstants> rk5, rk3;
    for (uint32_t b = 1; b < (uint32_t)BS_N1; b++) {
        rk5[b] = make_rotkey_lv(c, L5, b);
        if (stage >= 3) rk3[b] = make_rotkey_lv(c, L3, b);
    }
    for (uint32_t g = 1; g < (uint32_t)BS_N2; g++) {
        rk5[g*BS_N1] = make_rotkey_lv(c, L5, g*BS_N1);
        if (stage >= 3) rk3[g*BS_N1] = make_rotkey_lv(c, L3, g*BS_N1);
    }
    auto Krl = make_relin(c, L4);

    // BISECT: push every key through the wire format and back. Same process,
    // no sockets. If results move, the serializer is at fault; if they stay
    // bit-identical, the fault is in the demo's plumbing.
    if (serround) {
        size_t total = 0; int bad = 0;
        auto rt = [&](gpufhe::KeySwitchConstants& K) {
            std::vector<uint8_t> b; aegis::ks_serialize(b, K);
            total += b.size();
            gpufhe::KeySwitchConstants K2;
            aegis::Reader r(b.data(), b.size());
            if (!aegis::ks_deserialize(r, K2, 65536, 64)) { bad++; return; }
            if (!aegis::ks_equal(K, K2)) bad++;
            K = std::move(K2);
        };
        for (auto& kv : rk5) rt(kv.second);
        for (auto& kv : rk3) rt(kv.second);
        rt(Krl);
        std::printf("key round trip   %.1f MB, %d bad\n", total / 1048576.0, bad);
    }
    double t_keys = ms(t0);
    std::printf("setup %.0f ms   keys %.0f ms (%zu rot + 1 relin)\n\n",
                t_setup, t_keys, rk5.size() + rk3.size());

    auto C5 = gpufhe::ks_context_create(rk5.begin()->second);
    auto C4 = gpufhe::ks_context_create(Krl);
    gpufhe::DeviceKSContext C3;
    if (stage >= 3) C3 = gpufhe::ks_context_create(rk3.begin()->second);

    size_t TMAX = (size_t)c.sizeQ * c.n;
    uint64_t *dr0, *dr1, *scr, *drp;
    cudaMalloc(&dr0, TMAX*8); cudaMalloc(&dr1, TMAX*8);
    cudaMalloc(&scr, (size_t)c.n*8); cudaMalloc(&drp, (size_t)c.n*8);

    auto D1r = build_diagonals(build_matrix(m.W1, m.hidden, m.features));
    auto D2r = build_diagonals(build_matrix(m.W2, m.classes, m.hidden));
    std::vector<double> b1v(PER, 0.0), b2v(PER, 0.0);
    for (int h = 0; h < m.hidden; h++)  b1v[h] = m.b1[h];
    for (int q = 0; q < m.classes; q++) b2v[q] = m.b2[q];

    const double D50 = std::pow(2.0, 50), D40 = std::pow(2.0, 40);

    // ONE-TIME: encode diagonals. O(n^2) per diagonal, so this dominated the
    // per-image cost at n=16384 until it was hoisted out of the loop.
    t0 = clk::now();
    DiagSet E1 = encode_diags(D1r, D50, c, L5);
    DiagSet E2;
    if (stage >= 3) E2 = encode_diags(D2r, D40, c, L3);
    std::printf("diagonal encode  %.0f ms (one-time, model constants)\n\n", ms(t0));

    DevLayer DL1, DL3;
    if (g_resident == 2) {
        t0 = clk::now();
        devlayer_init(DL1, c, L5, rk5, E1);
        if (stage >= 3) devlayer_init(DL3, c, L3, rk3, E2);
        size_t freeB = 0, totB = 0; cudaMemGetInfo(&freeB, &totB);
        std::printf("device layers    %.0f ms   VRAM used %.0f MB of %.0f MB\n\n",
                    ms(t0), (totB - freeB) / 1048576.0, totB / 1048576.0);
    }

    double worst = 0; int agree = 0, correct = 0;

    for (int img = 0; img < nimg; img++) {
        auto vx = pack_input(te.x[img]);
        std::vector<int64_t> mx;
        gpufhe::encode_host(mx, repl(vx), c.n, D50);
        CtPair ct;
        gpufhe::encrypt_host(ct.c0, ct.c1, mx, c.KPq.pkA, c.KPq.pkB, c.n,
                             c.mod, c.root, NS, SIGMA, 909 + img);

        auto tinf = clk::now();
        ct = (g_resident == 2) ? bsgs_device(ct, E1, c, L5, DL1)
                               : bsgs(ct, E1, c, L5, rk5);
        rescale(ct, c, L5, C5, dr0, dr1, scr, drp);         // -> tw 4, scale 2^50
        add_pt(ct, b1v, D50, c, L4);

        std::vector<double> got;
        if (stage == 1) {
            got = decrypt_slots(ct, c, L4, D50);
        } else {
            ct = square_ct(ct, c, L4, Krl);                 // scale 2^100
            rescale(ct, c, L4, C4, dr0, dr1, scr, drp);     // -> tw 3, scale 2^50
            if (stage == 2) got = decrypt_slots(ct, c, L3, D50);
            else {
                ct = (g_resident == 2) ? bsgs_device(ct, E2, c, L3, DL3)
                                       : bsgs(ct, E2, c, L3, rk3);
                rescale(ct, c, L3, C3, dr0, dr1, scr, drp); // -> tw 2, scale 2^40
                add_pt(ct, b2v, D40, c, L2);
                got = decrypt_slots(ct, c, L2, D40);
            }
        }
        double tms = ms(tinf);

        // plaintext reference for this stage
        std::vector<double> ref(PER, 0.0);
        std::vector<double> z1(m.hidden);
        for (int h = 0; h < m.hidden; h++) {
            double s = m.b1[h];
            for (int i = 0; i < m.features; i++)
                s += (double)m.W1[(size_t)h*m.features+i] * te.x[img][i];
            z1[h] = s;
        }
        int nref = m.hidden;
        if (stage == 1) for (int h = 0; h < m.hidden; h++) ref[h] = z1[h];
        else if (stage == 2) for (int h = 0; h < m.hidden; h++) ref[h] = z1[h]*z1[h];
        else {
            nref = m.classes;
            for (int q = 0; q < m.classes; q++) {
                double s = m.b2[q];
                for (int h = 0; h < m.hidden; h++)
                    s += (double)m.W2[(size_t)q*m.hidden+h] * z1[h]*z1[h];
                ref[q] = s;
            }
        }
        double e = 0;
        for (int i = 0; i < nref; i++) e = std::max(e, std::fabs(got[i] - ref[i]));
        worst = std::max(worst, e);

        if (stage == 3) {
            int ae = 0, ar = 0;
            for (int q = 1; q < m.classes; q++) {
                if (got[q] > got[ae]) ae = q;
                if (ref[q] > ref[ar]) ar = q;
            }
            if (ae == ar) agree++;
            if (ae == te.y[img]) correct++;
            std::printf("img %3d  %7.0f ms  err %.2e  enc=%d ref=%d true=%d%s\n",
                        img, tms, e, ae, ar, te.y[img], ae==ar?"":"   <-- MISMATCH");
        } else {
            std::printf("img %3d  %7.0f ms  err %.2e\n", img, tms, e);
        }
    }

    std::printf("\nSTAGE %d  worst err %.3e", stage, worst);
    if (stage == 3) std::printf("   argmax agreement %d/%d   accuracy %d/%d",
                                agree, nimg, correct, nimg);
    std::printf("\n");
    return 0;
}
