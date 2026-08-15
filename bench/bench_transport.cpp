// Transport cost vs FHE compute cost.
//
// MEASURED here: AEAD throughput, loopback echo RTT, handshake cost. These
// are SOFTWARE costs -- loopback has no propagation delay and no bandwidth
// ceiling, so it is a floor, not a network.
// COMPUTED here: link serialization at nominal rates (bytes/rate + RTT).
// Reporting a loopback number as a 6G measurement would be dishonest; the
// two are printed in separate tables and labelled.
#include "transport.h"
#include "mlkem_sample.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

using namespace aegis;
using namespace aegis::mlkem768;
using clk = std::chrono::steady_clock;

static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

struct Stat { double lo, med, hi; };
static Stat summarize(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return {v.front(), v[v.size() / 2], v.back()};
}

static std::mt19937 rng(11);
static void rnd(uint8_t* b, size_t n) {
    for (size_t i = 0; i < n; i++) b[i] = (uint8_t)(rng() & 0xff);
}

// CKKS ciphertext = 2 polys * towers * n coefficients * 8 bytes.
struct Shape { const char* label; int n; int towers; };
static size_t ct_bytes(const Shape& s) {
    return (size_t)2 * s.towers * s.n * 8;
}

static const uint16_t PORT = 34611;
static const int REPS = 9;          // odd, median-reported

int main() {
    mlkem::ntt_init();

    std::printf("Aegis-6G transport benchmark\n");
    std::printf("REPS=%d, median reported, min/max as spread.\n", REPS);
    std::printf("NOTE: this machine shows ~30%% single-shot spread (WSL);\n"
                "      single timings are not trustworthy here.\n\n");

    // ---- handshake cost (one-time per session) ----
    {
        uint8_t d[32], z[32], m[32];
        uint8_t ek[EK_BYTES], dk[DK_BYTES], ct[CT_BYTES], ss[32];
        rnd(d, 32); rnd(z, 32); rnd(m, 32);

        std::vector<double> kg, en, de;
        for (int r = 0; r < REPS; r++) {
            auto t0 = clk::now();
            keygen_internal(ek, dk, d, z);
            kg.push_back(ms_since(t0));

            t0 = clk::now();
            encaps_internal(ss, ct, ek, m);
            en.push_back(ms_since(t0));

            t0 = clk::now();
            decaps_internal(ss, dk, ct);
            de.push_back(ms_since(t0));
        }
        auto k = summarize(kg), e = summarize(en), dd = summarize(de);
        std::printf("== MEASURED: ML-KEM-768 handshake (one-time per session)\n");
        std::printf("  keygen   %7.3f ms  [%.3f-%.3f]\n", k.med, k.lo, k.hi);
        std::printf("  encaps   %7.3f ms  [%.3f-%.3f]\n", e.med, e.lo, e.hi);
        std::printf("  decaps   %7.3f ms  [%.3f-%.3f]\n", dd.med, dd.lo, dd.hi);
        std::printf("  wire     %zu bytes (ct)\n\n", (size_t)CT_BYTES);
    }

    const Shape shapes[] = {
        {"n=1024  tw=1  (bootstrap in)",  1024,  1},
        {"n=1024  tw=30 (bootstrap ctx)", 1024, 30},
        {"n=8192  tw=30",                 8192, 30},
        {"n=32768 tw=4  (post-rescale)", 32768,  4},
        {"n=32768 tw=30 (full ladder)",  32768, 30},
    };

    // ---- AEAD throughput, no sockets ----
    std::printf("== MEASURED: record seal (AEAD only, no I/O)\n");
    std::printf("  %-30s %10s %10s %10s\n", "ciphertext shape", "size", "seal ms", "GB/s");
    std::vector<double> seal_ms_for;
    for (const Shape& sh : shapes) {
        size_t n = ct_bytes(sh);
        std::vector<uint8_t> pt(n), rec(n + REC_OVERHEAD);
        rnd(pt.data(), n);

        uint8_t d[32], z[32], m[32], ek[EK_BYTES], dk[DK_BYTES], c[CT_BYTES];
        rnd(d, 32); rnd(z, 32); rnd(m, 32);
        keygen_internal(ek, dk, d, z);
        Session s;
        session_init(s, c, ek, m);

        record_seal(s, rec.data(), pt.data(), n);      // warm

        std::vector<double> v;
        for (int r = 0; r < REPS; r++) {
            auto t0 = clk::now();
            record_seal(s, rec.data(), pt.data(), n);
            v.push_back(ms_since(t0));
        }
        auto st = summarize(v);
        double gbs = (double)n / (st.med * 1e-3) / 1e9;
        std::printf("  %-30s %8.2f MB %10.3f %10.2f\n",
                    sh.label, n / 1048576.0, st.med, gbs);
        seal_ms_for.push_back(st.med);
        s.close();
    }
    std::printf("\n");

    // ---- loopback echo RTT (software floor: AEAD + syscalls + copies) ----
    std::printf("== MEASURED: loopback echo RTT (SOFTWARE FLOOR, not a link)\n");
    std::printf("  %-30s %10s %12s\n", "ciphertext shape", "size", "rtt ms");

    std::vector<double> loop_med;
    for (const Shape& sh : shapes) {
        size_t n = ct_bytes(sh);
        std::vector<uint8_t> pt(n);
        rnd(pt.data(), n);

        uint8_t d[32], z[32], m[32], ek[EK_BYTES], dk[DK_BYTES];
        rnd(d, 32); rnd(z, 32); rnd(m, 32);
        keygen_internal(ek, dk, d, z);

        int lfd = tcp_listen(PORT);
        if (lfd < 0) { std::printf("  listen failed\n"); return 1; }

        std::thread srv([&] {
            int cfd = tcp_accept(lfd);
            if (cfd < 0) return;
            tcp_nodelay(cfd);
            Session s;
            if (!server_handshake(cfd, s, dk)) { tcp_close(cfd); return; }
            std::vector<uint8_t> got;
            for (int r = 0; r < REPS + 1; r++) {
                if (!recv_record(cfd, s, got)) break;
                if (!send_record(cfd, s, got.data(), got.size())) break;
            }
            s.close();
            tcp_close(cfd);
        });

        int fd = tcp_connect("127.0.0.1", PORT);
        tcp_nodelay(fd);
        Session c;
        client_handshake(fd, c, ek, m);

        std::vector<uint8_t> back;
        send_record(fd, c, pt.data(), n);              // warm
        recv_record(fd, c, back);

        std::vector<double> v;
        for (int r = 0; r < REPS; r++) {
            auto t0 = clk::now();
            send_record(fd, c, pt.data(), n);
            recv_record(fd, c, back);
            v.push_back(ms_since(t0));
        }
        auto st = summarize(v);
        std::printf("  %-30s %8.2f MB %8.3f  [%.3f-%.3f]\n",
                    sh.label, n / 1048576.0, st.med, st.lo, st.hi);
        loop_med.push_back(st.med);

        c.close();
        tcp_close(fd);
        srv.join();
        tcp_close(lfd);
    }
    std::printf("\n");

    // ---- COMPUTED link serialization ----
    std::printf("== COMPUTED: one-way link time = bytes/rate (propagation extra)\n");
    std::printf("  %-30s %10s %10s %10s %10s\n",
                "ciphertext shape", "size", "1 Gb/s", "10 Gb/s", "100 Gb/s");
    for (const Shape& sh : shapes) {
        size_t n = ct_bytes(sh);
        double bits = (double)n * 8.0;
        std::printf("  %-30s %8.2f MB %8.2f ms %8.2f ms %8.2f ms\n",
                    sh.label, n / 1048576.0,
                    bits / 1e9 * 1e3, bits / 1e10 * 1e3, bits / 1e11 * 1e3);
    }

    std::printf("\n== REFERENCE: measured FHE compute on this GPU (prior runs)\n");
    std::printf("  batched EvalMult (n=32768, tw=4)        ~4.2   ms/op\n");
    std::printf("  full pipeline w/ rescale + CUDA graphs  ~3.87  ms/op\n");
    std::printf("  bootstrap n=1024                        ~23300 ms\n");
    std::printf("  bootstrap n=8192                       ~244000 ms\n");
    std::printf("\n  Compare the COMPUTED column against these.\n");

    return 0;
}
