#pragma once
#include <cstdint>

// ML-KEM (FIPS 203) arithmetic layer.
// Ring: Z_q[X]/(X^256 + 1), q = 3329.
//
// CONSTANT-TIME STATUS: NOT constant time. Reduction uses '%' and the
// arithmetic is written for auditability, not for timing. Secret key
// material flows through these routines. A constant-time pass (Montgomery +
// Barrett, branch-free) is required before this is anything but a research
// artifact. See docs/security.md.
namespace aegis {
namespace mlkem {

constexpr int  N = 256;
constexpr int  Q = 3329;
constexpr int  ZETA = 17;   // primitive 256th root of unity mod Q

struct Poly {
    int16_t c[N];           // coefficients in [0, Q)
};

void ntt_init();            // build the zeta table; idempotent, call once

void ntt(Poly& p);          // in place, normal -> NTT domain
void invntt(Poly& p);       // in place, NTT domain -> normal (incl. n^-1)

// Pointwise multiply in the NTT domain: 128 independent degree-1 products,
// each modulo X^2 - zeta_i. NOT a plain elementwise product.
void basemul(Poly& r, const Poly& a, const Poly& b);

void poly_add(Poly& r, const Poly& a, const Poly& b);
void poly_sub(Poly& r, const Poly& a, const Poly& b);

// Reference negacyclic multiply, schoolbook, O(N^2). Test oracle only.
void poly_mul_schoolbook(Poly& r, const Poly& a, const Poly& b);

} // namespace mlkem
} // namespace aegis
