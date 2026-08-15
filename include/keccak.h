#pragma once
#include <cstdint>
#include <cstddef>

// Keccak-f[1600] + FIPS 202 sponge functions.
// Zero external dependencies. Assumes a little-endian host (x86-64).
namespace aegis {

void keccakf(uint64_t st[25]);

void sha3_256(const uint8_t* in, size_t inlen, uint8_t out[32]);
void sha3_512(const uint8_t* in, size_t inlen, uint8_t out[64]);
void shake128(uint8_t* out, size_t outlen, const uint8_t* in, size_t inlen);
void shake256(uint8_t* out, size_t outlen, const uint8_t* in, size_t inlen);

// Incremental XOF. ML-KEM's rejection sampler needs to pull bytes until it
// has enough accepted coefficients, so a one-shot API is not sufficient.
struct Xof {
    uint64_t st[25];
    size_t   rate;
    size_t   pos;

    void init(const uint8_t* in, size_t inlen, size_t rate_bytes, uint8_t pad);
    void init128(const uint8_t* in, size_t inlen) { init(in, inlen, 168, 0x1f); }
    void init256(const uint8_t* in, size_t inlen) { init(in, inlen, 136, 0x1f); }
    void squeeze(uint8_t* out, size_t outlen);
};

} // namespace aegis
