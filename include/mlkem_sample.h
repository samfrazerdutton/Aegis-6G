#pragma once
#include "mlkem.h"
#include <cstdint>
#include <cstddef>

// FIPS 203 sampling and serialization (Algorithms 6-9, 4-5).
namespace aegis {
namespace mlkem {

// Alg 7: rejection-sample a uniform poly in the NTT domain from SHAKE128.
// XOF is seeded with rho || j || i  (note the byte order: column then row).
void sample_ntt(Poly& a, const uint8_t rho[32], uint8_t i, uint8_t j);

// Alg 8: centered binomial distribution, eta in {2,3}, from 64*eta bytes.
void sample_cbd(Poly& a, const uint8_t* buf, int eta);

// Alg 5/6: pack/unpack d-bit coefficients, little-endian bit order.
// Output is 32*d bytes. d=12 decodes modulo q; d<12 modulo 2^d.
void byte_encode(uint8_t* out, const Poly& a, int d);
void byte_decode(Poly& a, const uint8_t* in, int d);

// Alg 4/5 helpers: lossy compression to d bits.
void poly_compress(Poly& a, int d);     // in place, coeffs -> [0, 2^d)
void poly_decompress(Poly& a, int d);   // in place, coeffs -> [0, q)

// PRF (Alg 3): SHAKE256(s || b), 64*eta output bytes.
void prf(uint8_t* out, size_t outlen, const uint8_t s[32], uint8_t b);

} // namespace mlkem
} // namespace aegis
