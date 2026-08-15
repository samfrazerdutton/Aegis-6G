#pragma once
#include "mlkem768.h"
#include <cstdint>
#include <cstddef>

// Aegis-6G secure session: ML-KEM-768 key agreement + ChaCha20-Poly1305
// record protection.
//
// THREAT MODEL -- READ BEFORE USING.
// This is UNAUTHENTICATED key agreement. Encapsulating to a peer's ek proves
// the responder holds the matching dk. It proves NOTHING about who that is.
// Security therefore rests entirely on the initiator having the CORRECT ek,
// pinned out of band (e.g. provisioned at manufacture). An adversary who can
// substitute an ek during provisioning is a full MITM. Authenticating the
// peer requires signatures (ML-DSA) or a PKI; neither is implemented here.
namespace aegis {

constexpr size_t REC_HDR      = 4;
constexpr size_t REC_TAG      = 16;
constexpr size_t REC_OVERHEAD = REC_HDR + REC_TAG;
constexpr size_t REC_MAX      = 1u << 26;   // 64 MiB; CKKS ciphertexts are big

struct Session {
    uint8_t  tx_key[32];
    uint8_t  rx_key[32];
    uint8_t  id[32];          // session binding value, safe to log
    uint64_t tx_seq;
    uint64_t rx_seq;
    bool     established;

    void close();             // zeroize key material
};

// Initiator. Writes the 1088-byte KEM ciphertext to send to the peer.
// m is the 32 bytes of encapsulation randomness.
void session_init(Session& s, uint8_t ct_out[mlkem768::CT_BYTES],
                  const uint8_t peer_ek[mlkem768::EK_BYTES],
                  const uint8_t m[32]);

// Responder. NOTE: this CANNOT fail. ML-KEM implicit rejection means a forged
// or corrupted ct yields an unrelated but well-formed key. The mismatch
// surfaces at the first record's tag check, not here. That is correct FO
// behaviour -- failing at the handshake would leak a decryption oracle.
void session_accept(Session& s, const uint8_t dk[mlkem768::DK_BYTES],
                    const uint8_t ct[mlkem768::CT_BYTES]);

// Seal one record. out needs ptlen + REC_OVERHEAD bytes. Returns bytes written.
size_t record_seal(Session& s, uint8_t* out, const uint8_t* pt, size_t ptlen);

// Open one record. Returns plaintext length, or -1 on failure.
// On failure the receive counter is NOT advanced and the session should be
// torn down: an AEAD stream cannot resynchronise after a rejected record.
long record_open(Session& s, uint8_t* pt_out, const uint8_t* rec, size_t reclen);

} // namespace aegis
