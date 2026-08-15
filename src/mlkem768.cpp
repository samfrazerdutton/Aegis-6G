#include "mlkem768.h"
#include "mlkem_sample.h"
#include "keccak.h"
#include <cstring>

namespace aegis {
namespace mlkem768 {

using mlkem::Poly;
using mlkem::N;
using mlkem::Q;

struct Vec { Poly p[K]; };

static void G(uint8_t out[64], const uint8_t* in, size_t inlen) {
    sha3_512(in, inlen, out);
}
static void H(uint8_t out[32], const uint8_t* in, size_t inlen) {
    sha3_256(in, inlen, out);
}
static void J(uint8_t out[32], const uint8_t* in, size_t inlen) {
    shake256(out, 32, in, inlen);
}

static void vec_ntt(Vec& v)    { for (int i = 0; i < K; i++) mlkem::ntt(v.p[i]); }
static void vec_invntt(Vec& v) { for (int i = 0; i < K; i++) mlkem::invntt(v.p[i]); }

// acc = sum_i a[i] * b[i], all in NTT domain.
static void dot(Poly& acc, const Poly a[K], const Poly b[K]) {
    Poly t;
    mlkem::basemul(acc, a[0], b[0]);
    for (int i = 1; i < K; i++) {
        mlkem::basemul(t, a[i], b[i]);
        mlkem::poly_add(acc, acc, t);
    }
}

// K-PKE.KeyGen. Returns ek and the PKE secret key.
static void kpke_keygen(uint8_t ek[EK_BYTES], uint8_t dkpke[384 * K],
                        const uint8_t d[32]) {
    uint8_t gin[33], g[64];
    std::memcpy(gin, d, 32);
    gin[32] = (uint8_t)K;              // FIPS 203: G(d || k)
    G(g, gin, 33);
    const uint8_t* rho = g;
    const uint8_t* sig = g + 32;

    Poly A[K][K];
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++)
            mlkem::sample_ntt(A[i][j], rho, (uint8_t)i, (uint8_t)j);

    uint8_t buf[64 * ETA1];
    Vec s, e;
    uint8_t nn = 0;
    for (int i = 0; i < K; i++) {
        mlkem::prf(buf, 64 * ETA1, sig, nn++);
        mlkem::sample_cbd(s.p[i], buf, ETA1);
    }
    for (int i = 0; i < K; i++) {
        mlkem::prf(buf, 64 * ETA1, sig, nn++);
        mlkem::sample_cbd(e.p[i], buf, ETA1);
    }
    vec_ntt(s);
    vec_ntt(e);

    // t = A*s + e
    Vec t;
    for (int i = 0; i < K; i++) {
        dot(t.p[i], A[i], s.p);
        mlkem::poly_add(t.p[i], t.p[i], e.p[i]);
    }

    for (int i = 0; i < K; i++) mlkem::byte_encode(ek + 384 * i, t.p[i], 12);
    std::memcpy(ek + 384 * K, rho, 32);
    for (int i = 0; i < K; i++) mlkem::byte_encode(dkpke + 384 * i, s.p[i], 12);
}

static void kpke_encrypt(uint8_t ct[CT_BYTES], const uint8_t ek[EK_BYTES],
                         const uint8_t m[32], const uint8_t r[32]) {
    Vec t;
    for (int i = 0; i < K; i++) mlkem::byte_decode(t.p[i], ek + 384 * i, 12);
    const uint8_t* rho = ek + 384 * K;

    Poly A[K][K];
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++)
            mlkem::sample_ntt(A[i][j], rho, (uint8_t)i, (uint8_t)j);

    uint8_t buf[64 * ETA1];
    Vec y, e1;
    Poly e2;
    uint8_t nn = 0;
    for (int i = 0; i < K; i++) {
        mlkem::prf(buf, 64 * ETA1, r, nn++);
        mlkem::sample_cbd(y.p[i], buf, ETA1);
    }
    for (int i = 0; i < K; i++) {
        mlkem::prf(buf, 64 * ETA2, r, nn++);
        mlkem::sample_cbd(e1.p[i], buf, ETA2);
    }
    mlkem::prf(buf, 64 * ETA2, r, nn++);
    mlkem::sample_cbd(e2, buf, ETA2);

    vec_ntt(y);

    // u = invNTT(A^T * y) + e1   -- note the TRANSPOSE vs keygen
    Vec u;
    for (int i = 0; i < K; i++) {
        Poly col[K];
        for (int j = 0; j < K; j++) col[j] = A[j][i];
        dot(u.p[i], col, y.p);
    }
    vec_invntt(u);
    for (int i = 0; i < K; i++) mlkem::poly_add(u.p[i], u.p[i], e1.p[i]);

    // v = invNTT(t^T * y) + e2 + Decompress_1(mu)
    Poly v, mu;
    dot(v, t.p, y.p);
    mlkem::invntt(v);
    mlkem::poly_add(v, v, e2);
    mlkem::byte_decode(mu, m, 1);
    mlkem::poly_decompress(mu, 1);
    mlkem::poly_add(v, v, mu);

    for (int i = 0; i < K; i++) {
        mlkem::poly_compress(u.p[i], DU);
        mlkem::byte_encode(ct + 32 * DU * i, u.p[i], DU);
    }
    mlkem::poly_compress(v, DV);
    mlkem::byte_encode(ct + 32 * DU * K, v, DV);
}

static void kpke_decrypt(uint8_t m[32], const uint8_t dkpke[384 * K],
                         const uint8_t ct[CT_BYTES]) {
    Vec u;
    for (int i = 0; i < K; i++) {
        mlkem::byte_decode(u.p[i], ct + 32 * DU * i, DU);
        mlkem::poly_decompress(u.p[i], DU);
    }
    Poly v;
    mlkem::byte_decode(v, ct + 32 * DU * K, DV);
    mlkem::poly_decompress(v, DV);

    Vec s;
    for (int i = 0; i < K; i++) mlkem::byte_decode(s.p[i], dkpke + 384 * i, 12);

    vec_ntt(u);
    Poly w;
    dot(w, s.p, u.p);
    mlkem::invntt(w);
    mlkem::poly_sub(w, v, w);

    mlkem::poly_compress(w, 1);
    mlkem::byte_encode(m, w, 1);
}

void keygen_internal(uint8_t ek[EK_BYTES], uint8_t dk[DK_BYTES],
                     const uint8_t d[32], const uint8_t z[32]) {
    uint8_t dkpke[384 * K];
    kpke_keygen(ek, dkpke, d);
    std::memcpy(dk, dkpke, 384 * K);
    std::memcpy(dk + 384 * K, ek, EK_BYTES);
    H(dk + 384 * K + EK_BYTES, ek, EK_BYTES);
    std::memcpy(dk + 384 * K + EK_BYTES + 32, z, 32);
}

void encaps_internal(uint8_t ss[SS_BYTES], uint8_t ct[CT_BYTES],
                     const uint8_t ek[EK_BYTES], const uint8_t m[32]) {
    uint8_t gin[64], g[64];
    std::memcpy(gin, m, 32);
    H(gin + 32, ek, EK_BYTES);
    G(g, gin, 64);
    std::memcpy(ss, g, 32);            // K
    kpke_encrypt(ct, ek, m, g + 32);   // r
}

void decaps_internal(uint8_t ss[SS_BYTES],
                     const uint8_t dk[DK_BYTES], const uint8_t ct[CT_BYTES]) {
    const uint8_t* dkpke = dk;
    const uint8_t* ek    = dk + 384 * K;
    const uint8_t* h     = dk + 384 * K + EK_BYTES;
    const uint8_t* z     = dk + 384 * K + EK_BYTES + 32;

    uint8_t mp[32];
    kpke_decrypt(mp, dkpke, ct);

    uint8_t gin[64], g[64];
    std::memcpy(gin, mp, 32);
    std::memcpy(gin + 32, h, 32);
    G(g, gin, 64);

    uint8_t kbar[32], jin[32 + CT_BYTES];
    std::memcpy(jin, z, 32);
    std::memcpy(jin + 32, ct, CT_BYTES);
    J(kbar, jin, 32 + CT_BYTES);

    uint8_t ctp[CT_BYTES];
    kpke_encrypt(ctp, ek, mp, g + 32);

    // Accumulating compare -- no early exit. NOT verified constant time;
    // the compiler is free to undo this. See docs/security.md.
    uint8_t diff = 0;
    for (int i = 0; i < CT_BYTES; i++) diff |= (uint8_t)(ct[i] ^ ctp[i]);
    uint8_t mask = (uint8_t)(((uint32_t)diff + 0xFF) >> 8);  // 0 if equal else 1
    mask = (uint8_t)(0 - mask);                              // 0x00 or 0xFF
    for (int i = 0; i < 32; i++)
        ss[i] = (uint8_t)((g[i] & ~mask) | (kbar[i] & mask));
}

bool ek_valid(const uint8_t ek[EK_BYTES]) {
    for (int i = 0; i < K; i++) {
        Poly p;
        uint8_t re[384];
        mlkem::byte_decode(p, ek + 384 * i, 12);
        mlkem::byte_encode(re, p, 12);
        if (std::memcmp(re, ek + 384 * i, 384) != 0) return false;
    }
    return true;
}

} // namespace mlkem768
} // namespace aegis
