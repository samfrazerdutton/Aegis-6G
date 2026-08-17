// Encrypted MNIST inference on GPU-Resident-Library. Stage 1: layer-1 matvec.
//
// Gated against forward_bsgs (plaintext, already proven == forward_direct).
// Native parameters throughout -- no OpenFHE anywhere in this file.
#include "gpufhe_api.h"
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include "mlp.h"
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <map>
#include <vector>

using namespace aegis;
using clk = std::chrono::steady_clock;
static double ms(clk::time_point t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

static uint64_t mmul(uint64_t a, uint64_t b, uint64_t q) {
    return (uint64_t)(((unsigned __int128)a * b) % q);
}
static uint64_t madd(uint64_t a, uint64_t b, uint64_t q) {
    uint64_t s = a + b; return s >= q ? s - q : s;
}

// ---- parameters ----
static const uint32_t N = 1024, SL = N / 2;   // 512 ciphertext slots
static const uint32_t PER = SLOTS;            // 256-periodic replication
static const uint32_t SIZEQ = 5, SIZEP = 2, NUMPART = 3;
static const uint64_t NS = 1;
static const double   SIGMA = 3.19;

struct Ctx {
    uint32_t n = N, sizeQ = SIZEQ, sizeP = SIZEP, sizeQlP = SIZEQ + SIZEP;
    std::vector<uint64_t> mod, modP, modQP, root, rootQP, PModq_QP;
    gpufhe::KeyPairHost KPqp, KPq;
    std::map<uint32_t, gpufhe::KeySwitchConstants> rotk;   // by rotation amount
};

// Replicate a PER-length real vector across all SL slots.
static std::vector<std::complex<double>> repl(const std::vector<double>& v) {
    std::vector<std::complex<double>> z(SL);
    for (uint32_t i = 0; i < SL; i++) z[i] = {v[i % PER], 0.0};
    return z;
}

static void setup(Ctx& c) {
    gpufhe::set_secret_hamming_weight(0);          // uniform ternary
    gpufhe::native_primes(c.mod,  c.sizeQ, 50, c.n, {});
    gpufhe::native_primes(c.modP, c.sizeP, 51, c.n, c.mod);
    c.modQP = c.mod;
    for (auto p : c.modP) c.modQP.push_back(p);
    c.root.resize(c.sizeQ);
    for (uint32_t i = 0; i < c.sizeQ; i++) c.root[i] = gpufhe::native_root(c.n, c.mod[i]);
    c.rootQP = c.root;
    for (auto p : c.modP) c.rootQP.push_back(gpufhe::native_root(c.n, p));

    c.KPqp = gpufhe::keygen_host(c.n, c.modQP, c.rootQP, NS, SIGMA, 101);
    c.KPq  = gpufhe::keygen_host(c.n, c.mod,   c.root,   NS, SIGMA, 101);

    c.PModq_QP.resize(c.sizeQlP);
    for (uint32_t t = 0; t < c.sizeQlP; t++) {
        uint64_t q = c.modQP[t], P = 1 % q;
        for (uint32_t j = 0; j < c.sizeP; j++) P = mmul(P, c.modQP[c.sizeQ + j] % q, q);
        c.PModq_QP[t] = P;
    }
}

// Per-level QP view: same seed => same ternary secret, reduced mod fewer primes.
struct Level {
    uint32_t tw;
    std::vector<uint64_t> mod, root, modQP, rootQP, PModq, rs1, rs2;
    gpufhe::KeyPairHost KPqp;
};

static Level make_level(Ctx& c, uint32_t tw) {
    Level L; L.tw = tw;
    L.mod.assign(c.mod.begin(), c.mod.begin() + tw);
    L.root.assign(c.root.begin(), c.root.begin() + tw);
    L.modQP = L.mod;  for (auto p : c.modP) L.modQP.push_back(p);
    L.rootQP = L.root;
    for (uint32_t j = 0; j < c.sizeP; j++) L.rootQP.push_back(c.rootQP[c.sizeQ + j]);
    L.KPqp = gpufhe::keygen_host(c.n, L.modQP, L.rootQP, NS, SIGMA, 101);
    uint32_t nqp = tw + c.sizeP;
    L.PModq.resize(nqp);
    for (uint32_t t = 0; t < nqp; t++) {
        uint64_t q = L.modQP[t], P = 1 % q;
        for (uint32_t j = 0; j < c.sizeP; j++) P = mmul(P, L.modQP[tw + j] % q, q);
        L.PModq[t] = P;
    }
    if (tw >= 2) gpufhe::native_rescale_consts(L.rs1, L.rs2, L.mod, tw - 1);
    return L;
}

static void fill_roots(gpufhe::KeySwitchConstants& K, Ctx& c, const Level& L) {
    for (uint32_t i = 0; i < L.tw; i++) { K.rootModList.push_back(L.mod[i]); K.rootValList.push_back(L.root[i]); }
    for (uint32_t j = 0; j < c.sizeP; j++) { K.rootModList.push_back(L.modQP[L.tw+j]); K.rootValList.push_back(L.rootQP[L.tw+j]); }
}

static uint32_t npart_for(uint32_t tw) { uint32_t p = (tw + 1) / 2; return p ? p : 1; }

// Relinearisation key (s^2) at this level.
static gpufhe::KeySwitchConstants make_relin(Ctx& c, const Level& L) {
    gpufhe::KeySwitchConstants K; K.n = c.n;
    gpufhe::compute_keyswitch_constants(K, L.mod, c.modP, npart_for(L.tw));
    fill_roots(K, c, L);
    gpufhe::evalkeygen_host(K, L.KPqp.s, L.KPqp.pkA, L.KPqp.pkB, L.PModq,
                            L.modQP, L.rootQP, NS, SIGMA, 202);
    return K;
}

// Rotation key for slot rotation `r` at level L.
static gpufhe::KeySwitchConstants make_rotkey_lv(Ctx& c, const Level& L, uint32_t r) {
    const uint32_t M = 2 * c.n;
    uint64_t k = 1;
    for (uint32_t i = 0; i < r; i++) k = (k * 5) % M;
    std::vector<uint64_t> sA = L.KPqp.s;
    gpufhe::automorphism_eval_host(sA, L.tw + c.sizeP, c.n, (uint32_t)k, L.modQP, L.rootQP);
    gpufhe::KeySwitchConstants K; K.n = c.n;
    gpufhe::compute_keyswitch_constants(K, L.mod, c.modP, npart_for(L.tw));
    fill_roots(K, c, L);
    gpufhe::evalkeygen_host_sold(K, L.KPqp.s, sA, L.KPqp.pkA, L.KPqp.pkB,
                                 L.PModq, L.modQP, L.rootQP, NS, SIGMA, 800 + r);
    return K;
}

static gpufhe::KeySwitchConstants make_rotkey(Ctx& c, uint32_t r) {
    const uint32_t M = 2 * c.n;
    uint64_t k = 1;
    for (uint32_t i = 0; i < r; i++) k = (k * 5) % M;

    std::vector<uint64_t> sA = c.KPqp.s;
    gpufhe::automorphism_eval_host(sA, c.sizeQlP, c.n, (uint32_t)k, c.modQP, c.rootQP);

    gpufhe::KeySwitchConstants K; K.n = c.n;
    gpufhe::compute_keyswitch_constants(K, c.mod, c.modP, NUMPART);
    for (uint32_t i = 0; i < c.sizeQ; i++) { K.rootModList.push_back(c.mod[i]);  K.rootValList.push_back(c.root[i]); }
    for (uint32_t j = 0; j < c.sizeP; j++) { K.rootModList.push_back(c.modQP[c.sizeQ+j]); K.rootValList.push_back(c.rootQP[c.sizeQ+j]); }
    gpufhe::evalkeygen_host_sold(K, c.KPqp.s, sA, c.KPqp.pkA, c.KPqp.pkB,
                                 c.PModq_QP, c.modQP, c.rootQP, NS, SIGMA, 800 + r);
    return K;
}

static uint32_t autok(uint32_t n, uint32_t r) {
    const uint32_t M = 2 * n; uint64_t k = 1;
    for (uint32_t i = 0; i < r; i++) k = (k * 5) % M;
    return (uint32_t)k;
}


// ---------- circuit helpers ----------
struct CtPair { std::vector<uint64_t> c0, c1; };

static void add_pt(CtPair& ct, const std::vector<double>& v, double scale,
                   Ctx& c, const Level& L) {
    std::vector<int64_t> mb;
    gpufhe::encode_host(mb, repl(v), c.n, scale);
    std::vector<uint64_t> ev;
    gpufhe::pt_to_eval_host(ev, mb, L.tw, c.n, L.mod, L.root);
    for (uint32_t t = 0; t < L.tw; t++) {
        uint64_t q = L.mod[t];
        for (uint32_t k = 0; k < c.n; k++) {
            size_t x = (size_t)t * c.n + k;
            ct.c0[x] = madd(ct.c0[x], ev[x], q);
        }
    }
}

// One rescale step: tw -> tw-1, scale /= mod[tw-1].
static void rescale(CtPair& ct, Ctx& c, const Level& L,
                    gpufhe::DeviceKSContext& C,
                    uint64_t* dr0, uint64_t* dr1, uint64_t* scr, uint64_t* drp) {
    size_t T = (size_t)L.tw * c.n;
    cudaMemcpy(dr0, ct.c0.data(), T * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dr1, ct.c1.data(), T * 8, cudaMemcpyHostToDevice);
    gpufhe::rescale_resident_raw(dr0, L.tw, C, L.rs1, L.rs2, scr, drp, 0);
    gpufhe::rescale_resident_raw(dr1, L.tw, C, L.rs1, L.rs2, scr, drp, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(ct.c0.data(), dr0, T * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(ct.c1.data(), dr1, T * 8, cudaMemcpyDeviceToHost);
    size_t Tn = (size_t)(L.tw - 1) * c.n;
    ct.c0.resize(Tn); ct.c1.resize(Tn);
}

// y = sum_k diag_k (x) rot_k(x), BSGS factored.
static CtPair bsgs(const CtPair& in, const std::vector<std::vector<double>>& D,
                   double dscale, Ctx& c, const Level& L,
                   std::map<uint32_t, gpufhe::KeySwitchConstants>& rk) {
    size_t T = (size_t)L.tw * c.n;
    std::vector<CtPair> B(BS_N1);
    B[0] = in;
    for (int b = 1; b < BS_N1; b++) {
        B[b] = in;
        gpufhe::rotate_ct_host(B[b].c0, B[b].c1, autok(c.n, b), rk[b],
                               L.tw, c.n, L.mod, L.root);
    }
    CtPair acc; acc.c0.assign(T, 0); acc.c1.assign(T, 0);
    bool got = false;
    for (int g = 0; g < BS_N2; g++) {
        int shift = g * BS_N1;
        CtPair inner; bool any = false;
        for (int b = 0; b < BS_N1; b++) {
            int k = shift + b;
            bool nz = false;
            for (int i = 0; i < SLOTS; i++) if (D[k][i] != 0.0) { nz = true; break; }
            if (!nz) continue;
            std::vector<double> dp(PER);
            for (int i = 0; i < (int)PER; i++) {
                int t = i - shift; if (t < 0) t += SLOTS;
                dp[i] = D[k][t];
            }
            std::vector<int64_t> md;
            gpufhe::encode_host(md, repl(dp), c.n, dscale);
            std::vector<uint64_t> ev;
            gpufhe::pt_to_eval_host(ev, md, L.tw, c.n, L.mod, L.root);
            CtPair p = B[b];
            gpufhe::ct_mul_pt_host(p.c0, p.c1, ev, L.tw, c.n, L.mod);
            if (!any) { inner = p; any = true; }
            else gpufhe::ct_add_ct_host(inner.c0, inner.c1, p.c0, p.c1, L.tw, c.n, L.mod);
        }
        if (!any) continue;
        if (shift) gpufhe::rotate_ct_host(inner.c0, inner.c1, autok(c.n, shift),
                                          rk[shift], L.tw, c.n, L.mod, L.root);
        if (!got) { acc = inner; got = true; }
        else gpufhe::ct_add_ct_host(acc.c0, acc.c1, inner.c0, inner.c1, L.tw, c.n, L.mod);
    }
    return acc;
}

// Square: tensor + relinearise. Scale becomes s^2; caller rescales.
static CtPair square_ct(const CtPair& a, Ctx& c, const Level& L,
                        const gpufhe::KeySwitchConstants& Krl) {
    size_t T = (size_t)L.tw * c.n;
    std::vector<uint64_t> t0(T), t1(T), t2(T);
    for (uint32_t t = 0; t < L.tw; t++) {
        uint64_t q = L.mod[t];
        for (uint32_t k = 0; k < c.n; k++) {
            size_t x = (size_t)t * c.n + k;
            t0[x] = mmul(a.c0[x], a.c0[x], q);
            t1[x] = madd(mmul(a.c0[x], a.c1[x], q), mmul(a.c1[x], a.c0[x], q), q);
            t2[x] = mmul(a.c1[x], a.c1[x], q);
        }
    }
    auto R = gpufhe::keyswitch_core_resident(t2, Krl);
    CtPair r; r.c0.resize(T); r.c1.resize(T);
    for (uint32_t t = 0; t < L.tw; t++) {
        uint64_t q = L.mod[t];
        for (uint32_t k = 0; k < c.n; k++) {
            size_t x = (size_t)t * c.n + k;
            r.c0[x] = madd(t0[x], R.ba0[x], q);
            r.c1[x] = madd(t1[x], R.ba1[x], q);
        }
    }
    return r;
}

static std::vector<double> decrypt_slots(const CtPair& ct, Ctx& c,
                                         const Level& L, double scale) {
    auto KPr = gpufhe::keygen_host(c.n, L.mod, L.root, NS, SIGMA, 101);
    std::vector<int64_t> dec;
    gpufhe::decrypt_host(dec, ct.c0, ct.c1, KPr.s, c.n, L.mod, L.root);
    std::vector<std::complex<double>> y;
    gpufhe::decode_host(y, dec, c.n, scale);
    std::vector<double> o(PER);
    for (uint32_t i = 0; i < PER; i++) o[i] = y[i].real();
    return o;
}

int main(int argc, char** argv) {
    int stage = (argc > 1) ? std::atoi(argv[1]) : 3;
    int nimg  = (argc > 2) ? std::atoi(argv[2]) : 1;

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

    auto D1 = build_diagonals(build_matrix(m.W1, m.hidden, m.features));
    auto D2 = build_diagonals(build_matrix(m.W2, m.classes, m.hidden));
    std::vector<double> b1v(PER, 0.0), b2v(PER, 0.0);
    for (int h = 0; h < m.hidden; h++)  b1v[h] = m.b1[h];
    for (int q = 0; q < m.classes; q++) b2v[q] = m.b2[q];

    const double D50 = std::pow(2.0, 50), D40 = std::pow(2.0, 40);
    double worst = 0; int agree = 0, correct = 0;

    for (int img = 0; img < nimg; img++) {
        auto vx = pack_input(te.x[img]);
        std::vector<int64_t> mx;
        gpufhe::encode_host(mx, repl(vx), c.n, D50);
        CtPair ct;
        gpufhe::encrypt_host(ct.c0, ct.c1, mx, c.KPq.pkA, c.KPq.pkB, c.n,
                             c.mod, c.root, NS, SIGMA, 909 + img);

        auto tinf = clk::now();
        ct = bsgs(ct, D1, D50, c, L5, rk5);
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
                ct = bsgs(ct, D2, D40, c, L3, rk3);         // scale 2^90
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
