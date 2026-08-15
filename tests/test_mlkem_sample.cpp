// Gates for FIPS 203 sampling + serialization.
#include "mlkem_sample.h"
#include "keccak.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>

using namespace aegis;
using namespace aegis::mlkem;

static int fails = 0;
static void check(const char* name, bool ok, const char* detail = nullptr) {
    if (!ok) fails++;
    std::printf("%-38s %s\n", name, ok ? "ok" : "FAIL");
    if (!ok && detail) std::printf("    %s\n", detail);
}

static std::mt19937 rng(999);

int main() {
    ntt_init();
    uint8_t rho[32];
    for (int i = 0; i < 32; i++) rho[i] = (uint8_t)(i * 7 + 1);

    // 1. sample_ntt: range + determinism + index sensitivity.
    {
        Poly a, b, c;
        sample_ntt(a, rho, 1, 2);
        sample_ntt(b, rho, 1, 2);
        sample_ntt(c, rho, 2, 1);   // transposed indices must differ
        bool range = true;
        for (int i = 0; i < N; i++)
            if (a.c[i] < 0 || a.c[i] >= Q) { range = false; break; }
        check("sample_ntt coeffs in [0,q)", range);
        check("sample_ntt deterministic",
              std::memcmp(a.c, b.c, sizeof a.c) == 0);
        check("sample_ntt A[i][j] != A[j][i]",
              std::memcmp(a.c, c.c, sizeof a.c) != 0);
    }

    // 2. sample_ntt uniformity: mean should sit near (q-1)/2 = 1664.
    //    A byte-order or shift bug skews this hard even when range passes.
    {
        double sum = 0; int cnt = 0;
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++) {
                Poly a; sample_ntt(a, rho, (uint8_t)i, (uint8_t)j);
                for (int k = 0; k < N; k++) { sum += a.c[k]; cnt++; }
            }
        double mean = sum / cnt;
        char buf[80];
        std::snprintf(buf, sizeof buf, "mean %.1f, want ~1664", mean);
        check("sample_ntt mean ~ (q-1)/2", std::fabs(mean - 1664.0) < 40.0, buf);
    }

    // 3. CBD: support must be exactly [-eta, eta] and the distribution
    //    binomial. eta=2 -> P(0)=6/16, P(+-1)=4/16, P(+-2)=1/16.
    for (int eta = 2; eta <= 3; eta++) {
        std::vector<uint8_t> buf(64 * eta);
        long hist[9] = {0};
        long total = 0;
        for (int t = 0; t < 200; t++) {
            for (auto& x : buf) x = (uint8_t)(rng() & 0xff);
            Poly a;
            sample_cbd(a, buf.data(), eta);
            for (int i = 0; i < N; i++) {
                int v = a.c[i] >= Q - eta ? a.c[i] - Q : a.c[i];
                if (v < -eta || v > eta) { hist[8] = -1; break; }
                hist[v + 4]++;
                total++;
            }
        }
        char nm[64];
        std::snprintf(nm, sizeof nm, "cbd eta=%d support in [-eta,eta]", eta);
        check(nm, hist[8] != -1);

        // Chi-square-ish: compare P(0) against the binomial expectation.
        double p0 = (double)hist[4] / total;
        double want = (eta == 2) ? 6.0 / 16.0 : 20.0 / 64.0;
        std::snprintf(nm, sizeof nm, "cbd eta=%d P(0) ~ binomial", eta);
        char det[80];
        std::snprintf(det, sizeof det, "got %.4f, want %.4f", p0, want);
        check(nm, std::fabs(p0 - want) < 0.02, det);
    }

    // 4. byte_encode/decode round trip at every d ML-KEM uses.
    {
        int ds[] = {1, 4, 5, 10, 11, 12};
        for (int d : ds) {
            Poly a, b;
            int mod = (d == 12) ? Q : (1 << d);
            std::uniform_int_distribution<int> dist(0, mod - 1);
            for (int i = 0; i < N; i++) a.c[i] = (int16_t)dist(rng);
            std::vector<uint8_t> buf(32 * d);
            byte_encode(buf.data(), a, d);
            byte_decode(b, buf.data(), d);
            char nm[64];
            std::snprintf(nm, sizeof nm, "byte_encode/decode d=%d round trip", d);
            check(nm, std::memcmp(a.c, b.c, sizeof a.c) == 0);
        }
    }

    // 5. Compression error bound: FIPS 203 guarantees
    //    |Decompress(Compress(x)) - x| mod+- q  <=  ceil(q / 2^(d+1)).
    //    This is the bound the whole decryption-failure analysis rests on.
    {
        int ds[] = {1, 4, 5, 10, 11};
        for (int d : ds) {
            int bound = (Q + (1 << (d + 1)) - 1) / (1 << (d + 1));
            int worst = 0;
            for (int x = 0; x < Q; x++) {
                Poly p; p.c[0] = (int16_t)x;
                poly_compress(p, d);
                poly_decompress(p, d);
                int diff = (int)p.c[0] - x;
                if (diff > Q / 2)  diff -= Q;
                if (diff < -Q / 2) diff += Q;
                if (std::abs(diff) > worst) worst = std::abs(diff);
            }
            char nm[64], det[80];
            std::snprintf(nm, sizeof nm, "compress d=%d error <= ceil(q/2^(d+1))", d);
            std::snprintf(det, sizeof det, "worst %d, bound %d", worst, bound);
            check(nm, worst <= bound, det);
        }
    }

    std::printf("\n%s (%d failing)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
