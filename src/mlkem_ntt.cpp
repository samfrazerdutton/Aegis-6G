#include "mlkem.h"

namespace aegis {
namespace mlkem {

static int16_t zetas[128];
static bool    inited = false;

static inline int16_t addq(int16_t a, int16_t b) {
    int32_t t = (int32_t)a + b;
    return (int16_t)(t >= Q ? t - Q : t);
}
static inline int16_t subq(int16_t a, int16_t b) {
    int32_t t = (int32_t)a - b;
    return (int16_t)(t < 0 ? t + Q : t);
}
static inline int16_t mulq(int16_t a, int16_t b) {
    return (int16_t)(((int32_t)a * (int32_t)b) % Q);
}

static int16_t powq(int16_t base, int e) {
    int32_t r = 1, b = base;
    while (e > 0) {
        if (e & 1) r = (r * b) % Q;
        b = (b * b) % Q;
        e >>= 1;
    }
    return (int16_t)r;
}

// Reverse the low 7 bits.
static int brv7(int i) {
    int r = 0;
    for (int b = 0; b < 7; b++) if (i & (1 << b)) r |= 1 << (6 - b);
    return r;
}

void ntt_init() {
    if (inited) return;
    for (int i = 0; i < 128; i++) zetas[i] = powq(ZETA, brv7(i));
    inited = true;
}

void ntt(Poly& p) {
    int k = 1;
    for (int len = 128; len >= 2; len >>= 1) {
        for (int start = 0; start < N; start += 2 * len) {
            int16_t z = zetas[k++];
            for (int j = start; j < start + len; j++) {
                int16_t t = mulq(z, p.c[j + len]);
                p.c[j + len] = subq(p.c[j], t);
                p.c[j]       = addq(p.c[j], t);
            }
        }
    }
}

void invntt(Poly& p) {
    int k = 127;
    for (int len = 2; len <= 128; len <<= 1) {
        for (int start = 0; start < N; start += 2 * len) {
            int16_t z = zetas[k--];
            for (int j = start; j < start + len; j++) {
                int16_t t   = p.c[j];
                p.c[j]      = addq(t, p.c[j + len]);
                p.c[j+len]  = mulq(z, subq(p.c[j + len], t));
            }
        }
    }
    const int16_t ninv = powq(128, Q - 2);   // 128^-1 mod Q = 3303
    for (int i = 0; i < N; i++) p.c[i] = mulq(p.c[i], ninv);
}

// (a0 + a1 X)(b0 + b1 X) mod (X^2 - zeta)
static inline void bm2(int16_t r[2], const int16_t a[2], const int16_t b[2],
                       int16_t zeta) {
    r[0] = addq(mulq(mulq(a[1], b[1]), zeta), mulq(a[0], b[0]));
    r[1] = addq(mulq(a[0], b[1]), mulq(a[1], b[0]));
}

void basemul(Poly& r, const Poly& a, const Poly& b) {
    for (int i = 0; i < N / 4; i++) {
        int16_t z = zetas[64 + i];
        bm2(&r.c[4*i],     &a.c[4*i],     &b.c[4*i],     z);
        bm2(&r.c[4*i + 2], &a.c[4*i + 2], &b.c[4*i + 2], (int16_t)(Q - z));
    }
}

void poly_add(Poly& r, const Poly& a, const Poly& b) {
    for (int i = 0; i < N; i++) r.c[i] = addq(a.c[i], b.c[i]);
}
void poly_sub(Poly& r, const Poly& a, const Poly& b) {
    for (int i = 0; i < N; i++) r.c[i] = subq(a.c[i], b.c[i]);
}

void poly_mul_schoolbook(Poly& r, const Poly& a, const Poly& b) {
    int32_t acc[2 * N] = {0};
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            acc[i + j] = (acc[i + j] + (int32_t)a.c[i] * b.c[j]) % Q;
    for (int i = 0; i < N; i++) {
        int32_t v = (acc[i] - acc[i + N]) % Q;   // X^N = -1
        r.c[i] = (int16_t)(v < 0 ? v + Q : v);
    }
}

} // namespace mlkem
} // namespace aegis
