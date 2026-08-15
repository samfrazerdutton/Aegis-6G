#pragma once
#include <cstdint>

// ML-KEM-768 (FIPS 203). Parameter set: k=3, eta1=eta2=2, du=10, dv=4.
namespace aegis {
namespace mlkem768 {

constexpr int K       = 3;
constexpr int ETA1    = 2;
constexpr int ETA2    = 2;
constexpr int DU      = 10;
constexpr int DV      = 4;

constexpr int EK_BYTES = 384 * K + 32;              // 1184
constexpr int DK_BYTES = 768 * K + 96;              // 2400
constexpr int CT_BYTES = 32 * (DU * K + DV);        // 1088
constexpr int SS_BYTES = 32;

// Deterministic internals (FIPS 203 *_internal). Exposed because KAT vectors
// supply the randomness explicitly -- a KEM whose only API self-seeds cannot
// be tested against known answers.
void keygen_internal(uint8_t ek[EK_BYTES], uint8_t dk[DK_BYTES],
                     const uint8_t d[32], const uint8_t z[32]);

void encaps_internal(uint8_t ss[SS_BYTES], uint8_t ct[CT_BYTES],
                     const uint8_t ek[EK_BYTES], const uint8_t m[32]);

// Never fails: a bad ciphertext yields an unrelated but deterministic key
// (implicit rejection), which is what makes the FO transform CCA-secure.
void decaps_internal(uint8_t ss[SS_BYTES],
                     const uint8_t dk[DK_BYTES], const uint8_t ct[CT_BYTES]);

// Modulus check from FIPS 203 6.2: ek must re-encode to itself.
bool ek_valid(const uint8_t ek[EK_BYTES]);

} // namespace mlkem768
} // namespace aegis
