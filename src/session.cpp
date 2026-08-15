#include "session.h"
#include "aead.h"
#include "keccak.h"
#include <cstring>

namespace aegis {

using namespace mlkem768;

static const char LABEL[] = "AEGIS6G-session-v1";

static inline void st32le(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static inline uint32_t ld32le(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline void st64le(uint8_t* p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}

// One SHAKE256 pass over the full transcript -> 96 bytes, split three ways.
// Binding H(ek) and H(ct) means a session key is tied to the exact peer key
// and the exact handshake message, so a ct replayed against a different ek
// cannot produce the same session.
static void derive(Session& s, const uint8_t ss[32],
                   const uint8_t ek[EK_BYTES], const uint8_t ct[CT_BYTES],
                   bool initiator) {
    uint8_t hek[32], hct[32];
    sha3_256(ek, EK_BYTES, hek);
    sha3_256(ct, CT_BYTES, hct);

    uint8_t in[sizeof(LABEL) - 1 + 32 + 32 + 32];
    size_t o = 0;
    std::memcpy(in + o, LABEL, sizeof(LABEL) - 1); o += sizeof(LABEL) - 1;
    std::memcpy(in + o, ss, 32);  o += 32;
    std::memcpy(in + o, hek, 32); o += 32;
    std::memcpy(in + o, hct, 32); o += 32;

    uint8_t okm[96];
    shake256(okm, 96, in, o);

    // okm[0..32)  = initiator -> responder
    // okm[32..64) = responder -> initiator
    // Distinct per direction, so a record reflected back at its sender fails.
    if (initiator) {
        std::memcpy(s.tx_key, okm,      32);
        std::memcpy(s.rx_key, okm + 32, 32);
    } else {
        std::memcpy(s.tx_key, okm + 32, 32);
        std::memcpy(s.rx_key, okm,      32);
    }
    std::memcpy(s.id, okm + 64, 32);

    s.tx_seq = 0;
    s.rx_seq = 0;
    s.established = true;

    volatile uint8_t* z = okm;
    for (size_t i = 0; i < sizeof okm; i++) z[i] = 0;
}

void Session::close() {
    volatile uint8_t* a = tx_key;
    volatile uint8_t* b = rx_key;
    for (int i = 0; i < 32; i++) { a[i] = 0; b[i] = 0; }
    tx_seq = rx_seq = 0;
    established = false;
}

void session_init(Session& s, uint8_t ct_out[CT_BYTES],
                  const uint8_t peer_ek[EK_BYTES], const uint8_t m[32]) {
    uint8_t ss[32];
    encaps_internal(ss, ct_out, peer_ek, m);
    derive(s, ss, peer_ek, ct_out, true);
    volatile uint8_t* z = ss;
    for (int i = 0; i < 32; i++) z[i] = 0;
}

void session_accept(Session& s, const uint8_t dk[DK_BYTES],
                    const uint8_t ct[CT_BYTES]) {
    uint8_t ss[32];
    decaps_internal(ss, dk, ct);
    const uint8_t* ek = dk + 384 * K;   // dk carries a copy of ek
    derive(s, ss, ek, ct, false);
    volatile uint8_t* z = ss;
    for (int i = 0; i < 32; i++) z[i] = 0;
}

// AAD = header || local sequence number. The sequence is NOT transmitted:
// the receiver uses its own counter, so a replayed or reordered record is
// authenticated against the wrong seq and fails.
static void build_aad(uint8_t aad[12], const uint8_t hdr[4], uint64_t seq) {
    std::memcpy(aad, hdr, 4);
    st64le(aad + 4, seq);
}
static void build_nonce(uint8_t nonce[12], uint64_t seq) {
    st32le(nonce, 0);
    st64le(nonce + 4, seq);
}

size_t record_seal(Session& s, uint8_t* out, const uint8_t* pt, size_t ptlen) {
    if (!s.established || ptlen > REC_MAX) return 0;

    uint8_t hdr[4], aad[12], nonce[12];
    st32le(hdr, (uint32_t)ptlen);
    build_aad(aad, hdr, s.tx_seq);
    build_nonce(nonce, s.tx_seq);

    std::memcpy(out, hdr, 4);
    aead_seal(out + REC_HDR, out + REC_HDR + ptlen,
              pt, ptlen, aad, 12, s.tx_key, nonce);

    s.tx_seq++;              // never reset -> nonce reuse is impossible
    return ptlen + REC_OVERHEAD;
}

long record_open(Session& s, uint8_t* pt_out, const uint8_t* rec,
                 size_t reclen) {
    if (!s.established || reclen < REC_OVERHEAD) return -1;

    uint32_t len = ld32le(rec);
    if (len > REC_MAX) return -1;
    if ((size_t)len + REC_OVERHEAD != reclen) return -1;

    uint8_t aad[12], nonce[12];
    build_aad(aad, rec, s.rx_seq);
    build_nonce(nonce, s.rx_seq);

    if (!aead_open(pt_out, rec + REC_HDR, len, rec + REC_HDR + len,
                   aad, 12, s.rx_key, nonce))
        return -1;

    s.rx_seq++;
    return (long)len;
}

} // namespace aegis
