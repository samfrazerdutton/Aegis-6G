#include "mlkem_sample.h"
#include "keccak.h"
#include <cstring>

namespace aegis {
namespace mlkem {

void prf(uint8_t* out, size_t outlen, const uint8_t s[32], uint8_t b) {
    uint8_t in[33];
    std::memcpy(in, s, 32);
    in[32] = b;
    shake256(out, outlen, in, 33);
}

void sample_ntt(Poly& a, const uint8_t rho[32], uint8_t i, uint8_t j) {
    uint8_t seed[34];
    std::memcpy(seed, rho, 32);
    seed[32] = j;          // FIPS 203: rho || j || i
    seed[33] = i;

    Xof xof;
    xof.init128(seed, 34);

    int n = 0;
    while (n < N) {
        uint8_t b[3];
        xof.squeeze(b, 3);
        int d1 = (int)b[0] + 256 * ((int)b[1] & 0x0F);
        int d2 = ((int)b[1] >> 4) + 16 * (int)b[2];
        if (d1 < Q && n < N) a.c[n++] = (int16_t)d1;
        if (d2 < Q && n < N) a.c[n++] = (int16_t)d2;
    }
}

void sample_cbd(Poly& a, const uint8_t* buf, int eta) {
    // Read 2*eta bits per coefficient: x = sum of first eta, y = sum of next.
    const int total = 2 * eta * N;
    for (int i = 0; i < N; i++) {
        int x = 0, y = 0;
        for (int k = 0; k < eta; k++) {
            int p = 2 * eta * i + k;
            x += (buf[p >> 3] >> (p & 7)) & 1;
        }
        for (int k = 0; k < eta; k++) {
            int p = 2 * eta * i + eta + k;
            y += (buf[p >> 3] >> (p & 7)) & 1;
        }
        int v = x - y;
        a.c[i] = (int16_t)(v < 0 ? v + Q : v);
    }
    (void)total;
}

void byte_encode(uint8_t* out, const Poly& a, int d) {
    std::memset(out, 0, 32 * d);
    for (int i = 0; i < N; i++) {
        uint32_t v = (uint32_t)a.c[i];
        for (int k = 0; k < d; k++) {
            int p = i * d + k;
            if ((v >> k) & 1u) out[p >> 3] |= (uint8_t)(1u << (p & 7));
        }
    }
}

void byte_decode(Poly& a, const uint8_t* in, int d) {
    for (int i = 0; i < N; i++) {
        uint32_t v = 0;
        for (int k = 0; k < d; k++) {
            int p = i * d + k;
            if ((in[p >> 3] >> (p & 7)) & 1u) v |= (1u << k);
        }
        a.c[i] = (int16_t)(d == 12 ? (v % Q) : v);
    }
}

void poly_compress(Poly& a, int d) {
    const uint32_t mask = (1u << d) - 1u;
    for (int i = 0; i < N; i++) {
        // round(2^d * x / q) with integer arithmetic, then reduce mod 2^d.
        uint32_t x = (uint32_t)a.c[i];
        uint32_t t = (uint32_t)(((uint64_t)x * (1u << d) + Q / 2) / Q);
        a.c[i] = (int16_t)(t & mask);
    }
}

void poly_decompress(Poly& a, int d) {
    for (int i = 0; i < N; i++) {
        uint32_t y = (uint32_t)a.c[i];
        uint32_t t = (uint32_t)(((uint64_t)y * Q + (1u << (d - 1))) >> d);
        a.c[i] = (int16_t)t;
    }
}

} // namespace mlkem
} // namespace aegis
