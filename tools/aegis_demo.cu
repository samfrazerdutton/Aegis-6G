// Aegis-6G end-to-end: encrypted inference over a post-quantum channel.
//
// THE PROPERTY THIS DEMONSTRATES:
//   - the CLIENT holds the CKKS secret key; the server never receives it
//   - the SERVER holds the model weights; the client never receives them
//   - the server evaluates on ciphertext and returns ciphertext
//   - everything on the wire is ML-KEM-768 + ChaCha20-Poly1305
// The server therefore never observes a plaintext image or a plaintext logit.
#include "fhe_circuit.h"
#include "transport.h"
#include "keyser.h"
#include <unistd.h>
#include <sys/wait.h>

// Client and server run as separate PROCESSES, not threads. GPU-Resident-
// Library keeps process-global caches (device root tables keyed by (n,q),
// keyswitch constants) that were never written for concurrent access -- two
// threads touching them corrupts results in a way that looks like a crypto
// bug. fork() must happen BEFORE any CUDA call, since a CUDA context does
// not survive fork.

using namespace aegis;

enum : uint32_t { MSG_PARAMS = 1, MSG_EVALKEY = 2, MSG_IMAGE = 3,
                  MSG_RESULT = 4, MSG_DONE = 5 };

static void put_hdr(std::vector<uint8_t>& b, uint32_t kind, uint32_t a, uint32_t c2) {
    put_u32(b, kind); put_u32(b, a); put_u32(b, c2);
}

static void put_ct(std::vector<uint8_t>& b, const CtPair& ct) {
    put_v64(b, ct.c0); put_v64(b, ct.c1);
}
static bool get_ct(Reader& r, CtPair& ct, size_t maxlen) {
    return get_v64(r, ct.c0, maxlen) && get_v64(r, ct.c1, maxlen);
}

static const uint16_t PORT = 35777;

// ---------------- server ----------------
struct ServerStats { double keytime = 0, evaltime = 0; size_t keybytes = 0; int nimg = 0; };

static void run_server(int lfd, const MLP& m) {
    ServerStats st;
    int fd = tcp_accept(lfd);
    if (fd < 0) return;
    tcp_nodelay(fd);

    uint8_t d[32], z[32], dk[mlkem768::DK_BYTES], ek[mlkem768::EK_BYTES];
    for (int i = 0; i < 32; i++) { d[i] = (uint8_t)(i * 3 + 1); z[i] = (uint8_t)(i * 5 + 2); }
    mlkem768::keygen_internal(ek, dk, d, z);

    Session s;
    if (!server_handshake(fd, s, dk)) { tcp_close(fd); return; }

    Ctx c;
    setup(c, false);                       // NO secret key material server-side
    Level L5 = make_level(c, 5, false), L4 = make_level(c, 4, false),
          L3 = make_level(c, 3, false), L2 = make_level(c, 2, false);

    auto D1r = build_diagonals(build_matrix(m.W1, m.hidden, m.features));
    auto D2r = build_diagonals(build_matrix(m.W2, m.classes, m.hidden));
    const double D50 = std::pow(2.0, 50), D40 = std::pow(2.0, 40);
    DiagSet E1 = encode_diags(D1r, D50, c, L5);
    DiagSet E2 = encode_diags(D2r, D40, c, L3);
    std::vector<double> b1v(PER, 0.0), b2v(PER, 0.0);
    for (int h = 0; h < m.hidden; h++)  b1v[h] = m.b1[h];
    for (int q = 0; q < m.classes; q++) b2v[q] = m.b2[q];

    std::map<uint32_t, gpufhe::KeySwitchConstants> rk5, rk3;
    gpufhe::KeySwitchConstants Krl;
    DevLayer DL1, DL3;
    bool layers_ready = false;
    auto tkey = clk::now();

    auto C4 = gpufhe::ks_context_create;   // silence unused warnings on some paths
    (void)C4;

    std::vector<uint8_t> rec;
    while (true) {
        if (!recv_record(fd, s, rec)) break;
        Reader r(rec.data(), rec.size());
        uint32_t kind = get_u32(r), a1 = get_u32(r), a2 = get_u32(r);
        if (!r.ok) break;

        if (kind == MSG_PARAMS) {
            std::vector<uint64_t> cm;
            if (!get_v64(r, cm, 64) || cm.size() != c.mod.size() ||
                !std::equal(cm.begin(), cm.end(), c.mod.begin())) {
                std::printf("[server] client parameters do not match; refusing\n");
                break;
            }
        } else if (kind == MSG_EVALKEY) {
            gpufhe::KeySwitchConstants K;
            if (!ks_deserialize(r, K, 65536, 64)) {
                std::printf("[server] rejected malformed eval key\n"); break;
            }
            st.keybytes += rec.size();
            if (a1 == 5)      rk5[a2] = std::move(K);
            else if (a1 == 3) rk3[a2] = std::move(K);
            else              Krl = std::move(K);
        } else if (kind == MSG_IMAGE) {
            if (!layers_ready) {
                devlayer_init(DL1, c, L5, rk5, E1);
                devlayer_init(DL3, c, L3, rk3, E2);
                layers_ready = true;
                st.keytime = ms(tkey);
            }
            CtPair ct;
            if (!get_ct(r, ct, (size_t)64 * c.n)) break;

            auto t0 = clk::now();
            auto C5 = gpufhe::ks_context_create(rk5.begin()->second);
            auto CR = gpufhe::ks_context_create(Krl);
            auto C3 = gpufhe::ks_context_create(rk3.begin()->second);
            size_t TM = (size_t)c.sizeQ * c.n;
            uint64_t *dr0, *dr1, *scr, *drp;
            cudaMalloc(&dr0, TM*8); cudaMalloc(&dr1, TM*8);
            cudaMalloc(&scr, (size_t)c.n*8); cudaMalloc(&drp, (size_t)c.n*8);

            ct = bsgs_device(ct, E1, c, L5, DL1);
            rescale(ct, c, L5, C5, dr0, dr1, scr, drp);
            add_pt(ct, b1v, D50, c, L4);
            ct = square_ct(ct, c, L4, Krl);
            rescale(ct, c, L4, CR, dr0, dr1, scr, drp);
            ct = bsgs_device(ct, E2, c, L3, DL3);
            rescale(ct, c, L3, C3, dr0, dr1, scr, drp);
            add_pt(ct, b2v, D40, c, L2);
            st.evaltime += ms(t0); st.nimg++;

            cudaFree(dr0); cudaFree(dr1); cudaFree(scr); cudaFree(drp);
            gpufhe::ks_context_destroy(C5); gpufhe::ks_context_destroy(CR);
            gpufhe::ks_context_destroy(C3);

            std::vector<uint8_t> out;
            put_hdr(out, MSG_RESULT, a1, 0);
            put_ct(out, ct);
            if (!send_record(fd, s, out.data(), out.size())) break;
        } else if (kind == MSG_DONE) break;
    }
    s.close(); tcp_close(fd);
    std::printf("[server] %.1f MB of keys, %.1f s to build device layers, "
                "%.0f ms/image evaluating\n",
                st.keybytes / 1048576.0, st.keytime / 1000.0,
                st.evaltime / (double)(st.nimg ? st.nimg : 1));
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    int nimg = (argc > 1) ? std::atoi(argv[1]) : 3;
    if (argc > 2) { N = (uint32_t)std::atoi(argv[2]); SL = N / 2; }
    g_resident = 2;

    MLP m;
    if (!mlp_load(m, "models/mlp.txt")) { std::printf("run train_mlp first\n"); return 2; }
    Dataset te;
    if (!load_mnist(te, "data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", 64)) {
        std::printf("run tools/fetch_mnist.sh\n"); return 2;
    }
    std::printf("Aegis-6G end-to-end demo   ring n=%u   log2(QP)=%u\n\n",
                N, SIZEQ * 50 + SIZEP * 51);

    int lfd = tcp_listen(PORT);
    if (lfd < 0) { std::printf("listen failed\n"); return 1; }

    pid_t pid = fork();                 // no CUDA has been touched yet
    if (pid < 0) { std::printf("fork failed\n"); return 1; }
    if (pid == 0) { run_server(lfd, m); tcp_close(lfd); _exit(0); }
    tcp_close(lfd);                     // parent is the client only

    // ---------------- client ----------------
    uint8_t d[32], z[32], mm32[32], ek[mlkem768::EK_BYTES], dk[mlkem768::DK_BYTES];
    for (int i = 0; i < 32; i++) { d[i] = (uint8_t)(i * 3 + 1); z[i] = (uint8_t)(i * 5 + 2); mm32[i] = (uint8_t)(i + 9); }
    mlkem768::keygen_internal(ek, dk, d, z);   // client has the server's ek pinned

    int fd = -1;
    for (int a = 0; a < 200 && fd < 0; a++) {
        fd = tcp_connect("127.0.0.1", PORT);
        if (fd < 0) usleep(20000);
    }
    if (fd < 0) { std::printf("connect failed\n"); return 1; }
    tcp_nodelay(fd);
    Session cs;
    if (!client_handshake(fd, cs, ek, mm32)) { std::printf("handshake failed\n"); return 1; }
    std::printf("[client] ML-KEM-768 session established\n");

    Ctx c; setup(c, true);                     // CLIENT holds the secret key
    Level L5 = make_level(c, 5), L4 = make_level(c, 4),
          L3 = make_level(c, 3), L2 = make_level(c, 2);

    { std::vector<uint8_t> b; put_hdr(b, MSG_PARAMS, 0, 0); put_v64(b, c.mod);
      send_record(fd, cs, b.data(), b.size()); }

    auto tk = clk::now();
    size_t sent = 0;
    auto ship = [&](uint32_t lvl, uint32_t amt, const gpufhe::KeySwitchConstants& K) {
        std::vector<uint8_t> b; put_hdr(b, MSG_EVALKEY, lvl, amt);
        ks_serialize(b, K);
        sent += b.size();
        send_record(fd, cs, b.data(), b.size());
    };
    for (uint32_t b2 = 1; b2 < (uint32_t)BS_N1; b2++) {
        ship(5, b2, make_rotkey_lv(c, L5, b2));
        ship(3, b2, make_rotkey_lv(c, L3, b2));
    }
    for (uint32_t g = 1; g < (uint32_t)BS_N2; g++) {
        ship(5, g*BS_N1, make_rotkey_lv(c, L5, g*BS_N1));
        ship(3, g*BS_N1, make_rotkey_lv(c, L3, g*BS_N1));
    }
    ship(4, 0, make_relin(c, L4));
    std::printf("[client] provisioned 61 evaluation keys, %.1f MB, %.1f s\n\n",
                sent / 1048576.0, ms(tk) / 1000.0);

    const double D50 = std::pow(2.0, 50), D40 = std::pow(2.0, 40);
    int agree = 0; double worst = 0;
    for (int img = 0; img < nimg; img++) {
        auto vx = pack_input(te.x[img]);
        std::vector<int64_t> mx;
        gpufhe::encode_host(mx, repl(vx), c.n, D50);
        CtPair ct;
        gpufhe::encrypt_host(ct.c0, ct.c1, mx, c.KPq.pkA, c.KPq.pkB, c.n,
                             c.mod, c.root, NS, SIGMA, 909 + img);

        std::vector<uint8_t> b; put_hdr(b, MSG_IMAGE, (uint32_t)img, 0);
        put_ct(b, ct);
        auto t0 = clk::now();
        send_record(fd, cs, b.data(), b.size());
        std::vector<uint8_t> back;
        if (!recv_record(fd, cs, back)) { std::printf("no result\n"); break; }
        double rtt = ms(t0);

        Reader r(back.data(), back.size());
        get_u32(r); get_u32(r); get_u32(r);
        CtPair res;
        if (!get_ct(r, res, (size_t)64 * c.n)) break;

        auto got = decrypt_slots(res, c, L2, D40);
        std::vector<double> ref(CLASSES);
        forward_direct(m, te.x[img], &ref);
        int ae = 0, ar = 0;
        for (int q = 1; q < CLASSES; q++) { if (got[q] > got[ae]) ae = q; if (ref[q] > ref[ar]) ar = q; }
        double e = 0;
        for (int q = 0; q < CLASSES; q++) e = std::max(e, std::fabs(got[q] - ref[q]));
        worst = std::max(worst, e);
        if (ae == ar) agree++;
        std::printf("img %2d  round trip %7.0f ms  err %.2e  enc=%d plain=%d true=%d%s\n",
                    img, rtt, e, ae, ar, te.y[img], ae == ar ? "" : "  <-- MISMATCH");
    }

    { std::vector<uint8_t> b; put_hdr(b, MSG_DONE, 0, 0);
      send_record(fd, cs, b.data(), b.size()); }
    cs.close(); tcp_close(fd);
    int status = 0; waitpid(pid, &status, 0);

    std::printf("\nargmax agreement %d/%d   worst logit err %.3e\n", agree, nimg, worst);
    std::printf("\nThe server never held the secret key, the image, or the logits.\n");
    return 0;
}
