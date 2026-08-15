#include "aead.h"
#include <cstring>

namespace aegis {

// ---------- ChaCha20 (RFC 8439 section 2) ----------

static inline uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}
static inline uint32_t ld32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline void st32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

#define QR(a, b, c, d)                        \
    a += b; d ^= a; d = rotl32(d, 16);        \
    c += d; b ^= c; b = rotl32(b, 12);        \
    a += b; d ^= a; d = rotl32(d, 8);         \
    c += d; b ^= c; b = rotl32(b, 7);

static void chacha20_block(uint8_t out[64], const uint8_t key[32],
                           uint32_t counter, const uint8_t nonce[12]) {
    uint32_t s[16], x[16];
    s[0] = 0x61707865; s[1] = 0x3320646e;
    s[2] = 0x79622d32; s[3] = 0x6b206574;
    for (int i = 0; i < 8; i++) s[4 + i] = ld32(key + 4 * i);
    s[12] = counter;
    for (int i = 0; i < 3; i++) s[13 + i] = ld32(nonce + 4 * i);

    std::memcpy(x, s, sizeof s);
    for (int i = 0; i < 10; i++) {
        QR(x[0], x[4], x[ 8], x[12])
        QR(x[1], x[5], x[ 9], x[13])
        QR(x[2], x[6], x[10], x[14])
        QR(x[3], x[7], x[11], x[15])
        QR(x[0], x[5], x[10], x[15])
        QR(x[1], x[6], x[11], x[12])
        QR(x[2], x[7], x[ 8], x[13])
        QR(x[3], x[4], x[ 9], x[14])
    }
    for (int i = 0; i < 16; i++) st32(out + 4 * i, x[i] + s[i]);
}

void chacha20_xor(uint8_t* out, const uint8_t* in, size_t len,
                  const uint8_t key[32], uint32_t counter,
                  const uint8_t nonce[12]) {
    uint8_t blk[64];
    while (len > 0) {
        chacha20_block(blk, key, counter++, nonce);
        size_t n = len < 64 ? len : 64;
        for (size_t i = 0; i < n; i++) out[i] = in[i] ^ blk[i];
        out += n; in += n; len -= n;
    }
}

// ---------- Poly1305 (RFC 8439 section 2.5) ----------
// 3 x 44-bit limb representation ("donna64" structure): h and r held mod
// 2^130-5 with lazy carries, __uint128_t for the products.

struct Poly {
    uint64_t r[3], h[3], pad[2];
    uint8_t  buf[16];
    size_t   leftover;
    int      done;

    void init(const uint8_t key[32]);
    void blocks(const uint8_t* m, size_t bytes);
    void update(const uint8_t* m, size_t bytes);
    void finish(uint8_t mac[16]);
};

static inline uint64_t ld64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}
static inline void st64(uint8_t* p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}

void Poly::init(const uint8_t key[32]) {
    uint64_t t0 = ld64(key), t1 = ld64(key + 8);
    r[0] = ( t0                    ) & 0x0ffc0fffffffULL;
    r[1] = ((t0 >> 44) | (t1 << 20)) & 0x0fffffc0ffffULL;
    r[2] = ( t1 >> 24              ) & 0x00ffffffc0fULL;
    h[0] = h[1] = h[2] = 0;
    pad[0] = ld64(key + 16);
    pad[1] = ld64(key + 24);
    leftover = 0;
    done = 0;
}

void Poly::blocks(const uint8_t* m, size_t bytes) {
    const uint64_t hibit = done ? 0 : ((uint64_t)1 << 40);
    const uint64_t r0 = r[0], r1 = r[1], r2 = r[2];
    const uint64_t s1 = r1 * 20, s2 = r2 * 20;
    uint64_t h0 = h[0], h1 = h[1], h2 = h[2], c;

    while (bytes >= 16) {
        uint64_t t0 = ld64(m), t1 = ld64(m + 8);
        h0 += ( t0                    ) & 0xfffffffffffULL;
        h1 += ((t0 >> 44) | (t1 << 20)) & 0xfffffffffffULL;
        h2 += (((t1 >> 24) & 0x3ffffffffffULL) | hibit);

        __uint128_t d0 = (__uint128_t)h0 * r0 + (__uint128_t)h1 * s2 +
                         (__uint128_t)h2 * s1;
        __uint128_t d1 = (__uint128_t)h0 * r1 + (__uint128_t)h1 * r0 +
                         (__uint128_t)h2 * s2;
        __uint128_t d2 = (__uint128_t)h0 * r2 + (__uint128_t)h1 * r1 +
                         (__uint128_t)h2 * r0;

        c = (uint64_t)(d0 >> 44); h0 = (uint64_t)d0 & 0xfffffffffffULL;
        d1 += c;
        c = (uint64_t)(d1 >> 44); h1 = (uint64_t)d1 & 0xfffffffffffULL;
        d2 += c;
        c = (uint64_t)(d2 >> 42); h2 = (uint64_t)d2 & 0x3ffffffffffULL;
        h0 += c * 5;
        c = h0 >> 44;             h0 &= 0xfffffffffffULL;
        h1 += c;

        m += 16; bytes -= 16;
    }
    h[0] = h0; h[1] = h1; h[2] = h2;
}

void Poly::update(const uint8_t* m, size_t bytes) {
    if (leftover) {
        size_t want = 16 - leftover;
        if (want > bytes) want = bytes;
        std::memcpy(buf + leftover, m, want);
        leftover += want; m += want; bytes -= want;
        if (leftover < 16) return;
        blocks(buf, 16);
        leftover = 0;
    }
    if (bytes >= 16) {
        size_t want = bytes & ~(size_t)15;
        blocks(m, want);
        m += want; bytes -= want;
    }
    if (bytes) { std::memcpy(buf + leftover, m, bytes); leftover += bytes; }
}

void Poly::finish(uint8_t mac[16]) {
    if (leftover) {
        buf[leftover++] = 1;
        while (leftover < 16) buf[leftover++] = 0;
        done = 1;
        blocks(buf, 16);
    }
    uint64_t h0 = h[0], h1 = h[1], h2 = h[2], c;
    c = h1 >> 44; h1 &= 0xfffffffffffULL;
    h2 += c; c = h2 >> 42; h2 &= 0x3ffffffffffULL;
    h0 += c * 5; c = h0 >> 44; h0 &= 0xfffffffffffULL;
    h1 += c; c = h1 >> 44; h1 &= 0xfffffffffffULL;
    h2 += c; c = h2 >> 42; h2 &= 0x3ffffffffffULL;
    h0 += c * 5; c = h0 >> 44; h0 &= 0xfffffffffffULL;
    h1 += c;

    uint64_t g0 = h0 + 5;  c = g0 >> 44; g0 &= 0xfffffffffffULL;
    uint64_t g1 = h1 + c;  c = g1 >> 44; g1 &= 0xfffffffffffULL;
    uint64_t g2 = h2 + c - ((uint64_t)1 << 42);

    // Branch-free select: take g if h >= p, else h.
    c = (g2 >> 63) - 1;
    g0 &= c; g1 &= c; g2 &= c;
    c = ~c;
    h0 = (h0 & c) | g0;
    h1 = (h1 & c) | g1;
    h2 = (h2 & c) | g2;

    uint64_t t0 = pad[0], t1 = pad[1];
    h0 += ( t0                    ) & 0xfffffffffffULL;
    c = h0 >> 44; h0 &= 0xfffffffffffULL;
    h1 += (((t0 >> 44) | (t1 << 20)) & 0xfffffffffffULL) + c;
    c = h1 >> 44; h1 &= 0xfffffffffffULL;
    h2 += (((t1 >> 24) & 0x3ffffffffffULL)) + c;
    h2 &= 0x3ffffffffffULL;

    st64(mac,     (h0      ) | (h1 << 44));
    st64(mac + 8, (h1 >> 20) | (h2 << 24));
}

void poly1305(uint8_t mac[16], const uint8_t* m, size_t len,
              const uint8_t key[32]) {
    Poly p;
    p.init(key);
    p.update(m, len);
    p.finish(mac);
}

// ---------- AEAD (RFC 8439 section 2.8) ----------

static void pad16(Poly& p, size_t len) {
    static const uint8_t z[16] = {0};
    size_t r = len % 16;
    if (r) p.update(z, 16 - r);
}

static void aead_tag(uint8_t tag[16], const uint8_t* ct, size_t ctlen,
                     const uint8_t* aad, size_t aadlen,
                     const uint8_t key[32], const uint8_t nonce[12]) {
    uint8_t pk[64] = {0}, zeros[64] = {0};
    chacha20_xor(pk, zeros, 64, key, 0, nonce);   // counter 0 -> Poly key

    Poly p;
    p.init(pk);
    p.update(aad, aadlen);  pad16(p, aadlen);
    p.update(ct, ctlen);    pad16(p, ctlen);
    uint8_t lens[16];
    st64(lens, (uint64_t)aadlen);
    st64(lens + 8, (uint64_t)ctlen);
    p.update(lens, 16);
    p.finish(tag);
}

void aead_seal(uint8_t* ct, uint8_t tag[16],
               const uint8_t* pt, size_t ptlen,
               const uint8_t* aad, size_t aadlen,
               const uint8_t key[32], const uint8_t nonce[12]) {
    chacha20_xor(ct, pt, ptlen, key, 1, nonce);   // counter 1 -> data
    aead_tag(tag, ct, ptlen, aad, aadlen, key, nonce);
}

bool aead_open(uint8_t* pt,
               const uint8_t* ct, size_t ctlen, const uint8_t tag[16],
               const uint8_t* aad, size_t aadlen,
               const uint8_t key[32], const uint8_t nonce[12]) {
    uint8_t want[16];
    aead_tag(want, ct, ctlen, aad, aadlen, key, nonce);

    uint8_t diff = 0;
    for (int i = 0; i < 16; i++) diff |= (uint8_t)(want[i] ^ tag[i]);
    if (diff) return false;                       // verify BEFORE decrypting

    chacha20_xor(pt, ct, ctlen, key, 1, nonce);
    return true;
}

} // namespace aegis
