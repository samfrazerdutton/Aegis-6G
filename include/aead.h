#pragma once
#include <cstdint>
#include <cstddef>

// ChaCha20-Poly1305 AEAD (RFC 8439).
//
// Chosen over AES-GCM deliberately: no S-box tables and no key-dependent
// branches, so this layer is constant-time by construction rather than by
// later repair. (The ML-KEM ring arithmetic is NOT -- see docs/security.md.)
namespace aegis {

void chacha20_xor(uint8_t* out, const uint8_t* in, size_t len,
                  const uint8_t key[32], uint32_t counter,
                  const uint8_t nonce[12]);

void poly1305(uint8_t mac[16], const uint8_t* m, size_t len,
              const uint8_t key[32]);

void aead_seal(uint8_t* ct, uint8_t tag[16],
               const uint8_t* pt, size_t ptlen,
               const uint8_t* aad, size_t aadlen,
               const uint8_t key[32], const uint8_t nonce[12]);

// Returns false on authentication failure. On failure the plaintext buffer
// contents are unspecified and MUST NOT be used or released to a caller.
bool aead_open(uint8_t* pt,
               const uint8_t* ct, size_t ctlen, const uint8_t tag[16],
               const uint8_t* aad, size_t aadlen,
               const uint8_t key[32], const uint8_t nonce[12]);

} // namespace aegis
