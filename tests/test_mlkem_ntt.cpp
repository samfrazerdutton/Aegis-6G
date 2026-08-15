// Gates for the ML-KEM ring arithmetic. All exact -- no tolerance.
#include "mlkem.h"
#include <cstdio>
#include <cstring>
#include <random>

using namespace aegis::mlkem;

static int fails = 0;
static void check(const char* name, bool ok, const char* detail = nullptr) {
    if (!ok) fails++;
    std::printf("%-34s %s\n", name, ok ? "ok" : "FAIL");
    if (!ok && detail) std::printf("    %s\n", detail);
}

static std::mt19937 rng(12345);
static void randpoly(Poly& p) {
    std::uniform_int_distribution<int> d(0, Q - 1);
    for (int i = 0; i < N; i++) p.c[i] = (int16_t)d(rng);
}

int main() {
    ntt_init();

    // 1. Root order: zeta^128 == -1, so zeta has order 256 and X^256+1 splits
    //    only down to quadratics. This is WHY basemul exists.
    {
        int32_t r = 1;
        for (int i = 0; i < 128; i++) r = (r * ZETA) % Q;
        char buf[64];
        std::snprintf(buf, sizeof buf, "zeta^128 = %d, want %d", (int)r, Q - 1);
        check("zeta^128 == -1 mod q", r == Q - 1, buf);
    }

    // 2. Round trip.
    {
        bool ok = true;
        for (int t = 0; t < 64 && ok; t++) {
            Poly a, b;
            randpoly(a);
            b = a;
            ntt(b);
            invntt(b);
            if (std::memcmp(a.c, b.c, sizeof a.c) != 0) ok = false;
        }
        check("invntt(ntt(x)) == x", ok);
    }

    // 3. THE structural gate: NTT-domain basemul must equal schoolbook
    //    negacyclic multiplication. A wrong zeta index, a wrong sign on the
    //    odd half, or a bad bit-reversal all survive gate 2 and die here.
    {
        bool ok = true;
        int  bad = -1;
        for (int t = 0; t < 32 && ok; t++) {
            Poly a, b, ref, got, na, nb;
            randpoly(a);
            randpoly(b);
            poly_mul_schoolbook(ref, a, b);
            na = a; nb = b;
            ntt(na); ntt(nb);
            basemul(got, na, nb);
            invntt(got);
            for (int i = 0; i < N; i++)
                if (ref.c[i] != got.c[i]) { ok = false; bad = i; break; }
        }
        char buf[64];
        std::snprintf(buf, sizeof buf, "first mismatch at coeff %d", bad);
        check("basemul == schoolbook negacyclic", ok, buf);
    }

    // 4. Linearity: NTT is a ring hom on addition, so ntt(a+b)==ntt(a)+ntt(b).
    {
        Poly a, b, s, na, nb, ns;
        randpoly(a); randpoly(b);
        poly_add(s, a, b);
        na = a; nb = b; ns = s;
        ntt(na); ntt(nb); ntt(ns);
        Poly sum;
        poly_add(sum, na, nb);
        check("ntt(a+b) == ntt(a)+ntt(b)",
              std::memcmp(sum.c, ns.c, sizeof sum.c) == 0);
    }

    // 5. Range invariant: every routine must leave coeffs in [0, Q).
    {
        Poly a, b, r;
        randpoly(a); randpoly(b);
        ntt(a); ntt(b);
        basemul(r, a, b);
        invntt(r);
        bool ok = true;
        for (int i = 0; i < N; i++)
            if (r.c[i] < 0 || r.c[i] >= Q) { ok = false; break; }
        check("coeffs stay in [0,q)", ok);
    }

    std::printf("\n%s (%d failing)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
