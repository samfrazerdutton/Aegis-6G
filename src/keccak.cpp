#include "keccak.h"
#include <cstring>

namespace aegis {

#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))

static const uint64_t RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};

static const int ROTC[24] = {1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
                             27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44};

static const int PILN[24] = {10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
                             15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1};

void keccakf(uint64_t st[25]) {
    uint64_t t, bc[5];
    for (int r = 0; r < 24; r++) {
        // theta
        for (int i = 0; i < 5; i++)
            bc[i] = st[i] ^ st[i+5] ^ st[i+10] ^ st[i+15] ^ st[i+20];
        for (int i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ ROTL64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
        }
        // rho + pi
        t = st[1];
        for (int i = 0; i < 24; i++) {
            int j = PILN[i];
            bc[0] = st[j];
            st[j] = ROTL64(t, ROTC[i]);
            t = bc[0];
        }
        // chi
        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; i++) bc[i] = st[j + i];
            for (int i = 0; i < 5; i++)
                st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }
        // iota
        st[0] ^= RC[r];
    }
}

// Absorb the whole input and apply the pad. Leaves the state absorbed but
// NOT yet permuted, so the caller controls the first squeeze permutation.
static void absorb(uint64_t st[25], size_t rate, const uint8_t* in,
                   size_t inlen, uint8_t pad) {
    uint8_t* s = reinterpret_cast<uint8_t*>(st);
    std::memset(st, 0, 25 * sizeof(uint64_t));
    while (inlen >= rate) {
        for (size_t i = 0; i < rate; i++) s[i] ^= in[i];
        keccakf(st);
        in += rate;
        inlen -= rate;
    }
    uint8_t blk[200];
    std::memset(blk, 0, rate);
    std::memcpy(blk, in, inlen);
    blk[inlen] = pad;
    blk[rate - 1] |= 0x80;
    for (size_t i = 0; i < rate; i++) s[i] ^= blk[i];
}

static void squeeze_all(uint8_t* out, size_t outlen, uint64_t st[25],
                        size_t rate) {
    while (outlen > 0) {
        keccakf(st);
        size_t n = outlen < rate ? outlen : rate;
        std::memcpy(out, reinterpret_cast<uint8_t*>(st), n);
        out += n;
        outlen -= n;
    }
}

void sha3_256(const uint8_t* in, size_t inlen, uint8_t out[32]) {
    uint64_t st[25];
    absorb(st, 136, in, inlen, 0x06);
    squeeze_all(out, 32, st, 136);
}

void sha3_512(const uint8_t* in, size_t inlen, uint8_t out[64]) {
    uint64_t st[25];
    absorb(st, 72, in, inlen, 0x06);
    squeeze_all(out, 64, st, 72);
}

void shake128(uint8_t* out, size_t outlen, const uint8_t* in, size_t inlen) {
    uint64_t st[25];
    absorb(st, 168, in, inlen, 0x1f);
    squeeze_all(out, outlen, st, 168);
}

void shake256(uint8_t* out, size_t outlen, const uint8_t* in, size_t inlen) {
    uint64_t st[25];
    absorb(st, 136, in, inlen, 0x1f);
    squeeze_all(out, outlen, st, 136);
}

void Xof::init(const uint8_t* in, size_t inlen, size_t rate_bytes, uint8_t pad) {
    rate = rate_bytes;
    absorb(st, rate, in, inlen, pad);
    pos = rate;  // forces a permutation on the first squeeze
}

void Xof::squeeze(uint8_t* out, size_t outlen) {
    const uint8_t* s = reinterpret_cast<const uint8_t*>(st);
    while (outlen > 0) {
        if (pos == rate) {
            keccakf(st);
            pos = 0;
        }
        size_t avail = rate - pos;
        size_t n = outlen < avail ? outlen : avail;
        std::memcpy(out, s + pos, n);
        out += n;
        pos += n;
        outlen -= n;
    }
}

} // namespace aegis
