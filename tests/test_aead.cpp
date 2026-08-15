// RFC 8439 known-answer vectors + tamper-detection properties.
#include "aead.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace aegis;

static int fails = 0;
static void check(const char* name, bool ok, const char* detail = nullptr) {
    if (!ok) fails++;
    std::printf("%-40s %s\n", name, ok ? "ok" : "FAIL");
    if (!ok && detail) std::printf("    %s\n", detail);
}
static std::string hex(const uint8_t* b, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; i++) { s += d[b[i] >> 4]; s += d[b[i] & 15]; }
    return s;
}
static std::vector<uint8_t> unhex(const std::string& s) {
    std::vector<uint8_t> v(s.size() / 2);
    for (size_t i = 0; i < v.size(); i++)
        v[i] = (uint8_t)std::stoul(s.substr(2 * i, 2), nullptr, 16);
    return v;
}

int main() {
    // RFC 8439 2.4.2 -- ChaCha20 encryption.
    {
        uint8_t key[32], nonce[12] = {0,0,0,0, 0,0,0,0x4a, 0,0,0,0};
        for (int i = 0; i < 32; i++) key[i] = (uint8_t)i;
        const char* pt = "Ladies and Gentlemen of the class of '99: If I could"
                         " offer you only one tip for the future, sunscreen wo"
                         "uld be it.";
        size_t n = std::strlen(pt);
        std::vector<uint8_t> ct(n);
        chacha20_xor(ct.data(), (const uint8_t*)pt, n, key, 1, nonce);
        check("RFC8439 2.4.2 chacha20 encrypt", hex(ct.data(), n) ==
            "6e2e359a2568f98041ba0728dd0d6981e97e7aec1d4360c20a27afccfd9fae0b"
            "f91b65c5524733ab8f593dabcd62b3571639d624e65152ab8f530c359f0861d8"
            "07ca0dbf500d6a6156a38e088a22b65e52bc514d16ccf806818ce91ab7793736"
            "5af90bbf74a35be6b40b8eedf2785e42874d");

        // Round trip must restore the plaintext.
        std::vector<uint8_t> back(n);
        chacha20_xor(back.data(), ct.data(), n, key, 1, nonce);
        check("chacha20 round trip",
              std::memcmp(back.data(), pt, n) == 0);
    }

    // RFC 8439 2.5.2 -- Poly1305.
    {
        auto key = unhex("85d6be7857556d337f4452fe42d506a8"
                         "0103808afb0db2fd4abff6af4149f51b");
        const char* m = "Cryptographic Forum Research Group";
        uint8_t mac[16];
        poly1305(mac, (const uint8_t*)m, std::strlen(m), key.data());
        check("RFC8439 2.5.2 poly1305",
              hex(mac, 16) == "a8061dc1305136c6c22b8baf0c0127a9");
    }

    // RFC 8439 2.8.2 -- AEAD seal.
    const char* pt = "Ladies and Gentlemen of the class of '99: If I could"
                     " offer you only one tip for the future, sunscreen wo"
                     "uld be it.";
    size_t ptlen = std::strlen(pt);
    uint8_t key[32], nonce[12] = {0x07,0,0,0, 0x40,0x41,0x42,0x43,
                                  0x44,0x45,0x46,0x47};
    for (int i = 0; i < 32; i++) key[i] = (uint8_t)(0x80 + i);
    uint8_t aad[12] = {0x50,0x51,0x52,0x53, 0xc0,0xc1,0xc2,0xc3,
                       0xc4,0xc5,0xc6,0xc7};
    std::vector<uint8_t> ct(ptlen);
    uint8_t tag[16];
    aead_seal(ct.data(), tag, (const uint8_t*)pt, ptlen, aad, 12, key, nonce);

    check("RFC8439 2.8.2 aead ciphertext", hex(ct.data(), ptlen) ==
        "d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d6"
        "3dbea45e8ca9671282fafb69da92728b1a71de0a9e060b2905d6a5b67ecd3b36"
        "92ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7bc"
        "3ff4def08e4b7a9de576d26586cec64b6116");
    check("RFC8439 2.8.2 aead tag",
          hex(tag, 16) == "1ae10b594f09e26a7e902ecbd0600691");

    // Open must succeed and restore plaintext.
    {
        std::vector<uint8_t> out(ptlen);
        bool ok = aead_open(out.data(), ct.data(), ptlen, tag, aad, 12,
                            key, nonce);
        check("aead_open accepts genuine",
              ok && std::memcmp(out.data(), pt, ptlen) == 0);
    }

    // Tamper matrix: every input class must cause rejection.
    {
        std::vector<uint8_t> out(ptlen);
        auto try_open = [&](const uint8_t* c, const uint8_t* t,
                            const uint8_t* a, const uint8_t* k,
                            const uint8_t* nn) {
            return aead_open(out.data(), c, ptlen, t, a, 12, k, nn);
        };

        std::vector<uint8_t> c2 = ct; c2[0] ^= 1;
        check("tampered ciphertext rejected",
              !try_open(c2.data(), tag, aad, key, nonce));

        c2 = ct; c2[ptlen - 1] ^= 0x80;
        check("tampered last ct byte rejected",
              !try_open(c2.data(), tag, aad, key, nonce));

        uint8_t t2[16]; std::memcpy(t2, tag, 16); t2[15] ^= 1;
        check("tampered tag rejected",
              !try_open(ct.data(), t2, aad, key, nonce));

        uint8_t a2[12]; std::memcpy(a2, aad, 12); a2[3] ^= 1;
        check("tampered aad rejected",
              !try_open(ct.data(), tag, a2, key, nonce));

        uint8_t k2[32]; std::memcpy(k2, key, 32); k2[31] ^= 1;
        check("wrong key rejected",
              !try_open(ct.data(), tag, aad, k2, nonce));

        uint8_t n2[12]; std::memcpy(n2, nonce, 12); n2[11] ^= 1;
        check("wrong nonce rejected",
              !try_open(ct.data(), tag, aad, key, n2));
    }

    // Edge cases the session layer will actually hit: empty AAD, empty
    // plaintext (keepalive frames), and a length crossing the 64-byte
    // ChaCha block plus the 16-byte Poly block.
    {
        bool ok = true;
        for (size_t n : {(size_t)0, (size_t)1, (size_t)15, (size_t)16,
                         (size_t)63, (size_t)64, (size_t)65, (size_t)1000}) {
            std::vector<uint8_t> p(n), c(n), o(n);
            for (size_t i = 0; i < n; i++) p[i] = (uint8_t)(i * 31 + 7);
            uint8_t tg[16];
            aead_seal(c.data(), tg, p.data(), n, nullptr, 0, key, nonce);
            if (!aead_open(o.data(), c.data(), n, tg, nullptr, 0, key, nonce))
                ok = false;
            else if (n && std::memcmp(o.data(), p.data(), n) != 0) ok = false;
        }
        check("empty aad + length edge cases", ok);
    }

    std::printf("\n%s (%d failing)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
