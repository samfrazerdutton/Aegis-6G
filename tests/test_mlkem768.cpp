#include "mlkem768.h"
#include "mlkem_sample.h"
#include "keccak.h"
#include <cstdio>
#include <cstring>
#include <random>

using namespace aegis;
using namespace aegis::mlkem768;

static int fails = 0;
static void check(const char* name, bool ok, const char* detail = nullptr) {
    if (!ok) fails++;
    std::printf("%-38s %s\n", name, ok ? "ok" : "FAIL");
    if (!ok && detail) std::printf("    %s\n", detail);
}

static std::mt19937 rng(2024);
static void rand32(uint8_t b[32]) { for (int i = 0; i < 32; i++) b[i] = (uint8_t)(rng() & 0xff); }

int main() {
    mlkem::ntt_init();

    check("sizes ek/dk/ct = 1184/2400/1088",
          EK_BYTES == 1184 && DK_BYTES == 2400 && CT_BYTES == 1088);

    uint8_t ek[EK_BYTES], dk[DK_BYTES], ct[CT_BYTES], ss1[32], ss2[32];
    uint8_t d[32], z[32], m[32];

    // 1. Correctness: encaps/decaps agree. Any single-bit error anywhere in
    //    the stack breaks this, so it is the broadest gate available.
    {
        bool ok = true;
        int trials = 200;
        for (int t = 0; t < trials && ok; t++) {
            rand32(d); rand32(z); rand32(m);
            keygen_internal(ek, dk, d, z);
            encaps_internal(ss1, ct, ek, m);
            decaps_internal(ss2, dk, ct);
            if (std::memcmp(ss1, ss2, 32) != 0) ok = false;
        }
        char det[64];
        std::snprintf(det, sizeof det, "%d trials", trials);
        check("encaps/decaps agree", ok, det);
    }

    // 2. Determinism of the *_internal API -- required for KAT testing.
    {
        uint8_t ek2[EK_BYTES], dk2[DK_BYTES];
        rand32(d); rand32(z);
        keygen_internal(ek, dk, d, z);
        keygen_internal(ek2, dk2, d, z);
        check("keygen_internal deterministic",
              std::memcmp(ek, ek2, EK_BYTES) == 0 &&
              std::memcmp(dk, dk2, DK_BYTES) == 0);
    }

    // 3. Implicit rejection: a corrupted ciphertext must NOT fail and must
    //    NOT yield the real key -- it yields J(z||c'). This is the FO
    //    property; returning an error here would leak a decryption oracle.
    {
        rand32(d); rand32(z); rand32(m);
        keygen_internal(ek, dk, d, z);
        encaps_internal(ss1, ct, ek, m);

        uint8_t bad[CT_BYTES];
        std::memcpy(bad, ct, CT_BYTES);
        bad[7] ^= 0x01;

        uint8_t got[32];
        decaps_internal(got, dk, bad);
        check("corrupt ct -> different key", std::memcmp(got, ss1, 32) != 0);

        uint8_t jin[32 + CT_BYTES], want[32];
        std::memcpy(jin, dk + 384 * K + EK_BYTES + 32, 32);
        std::memcpy(jin + 32, bad, CT_BYTES);
        shake256(want, 32, jin, 32 + CT_BYTES);
        check("corrupt ct -> J(z||c) exactly", std::memcmp(got, want, 32) == 0);

        uint8_t again[32];
        decaps_internal(again, dk, bad);
        check("rejection key deterministic", std::memcmp(got, again, 32) == 0);
    }

    // 4. Wrong key decaps must not collide with the right shared secret.
    {
        uint8_t ekB[EK_BYTES], dkB[DK_BYTES], ssB[32];
        rand32(d); rand32(z); rand32(m);
        keygen_internal(ek, dk, d, z);
        rand32(d); rand32(z);
        keygen_internal(ekB, dkB, d, z);
        encaps_internal(ss1, ct, ek, m);
        decaps_internal(ssB, dkB, ct);
        check("wrong dk -> different ss", std::memcmp(ss1, ssB, 32) != 0);
    }

    // 5. Modulus check: a genuine ek validates; a coefficient forced to
    //    q + small must be rejected (non-canonical 12-bit encoding).
    {
        rand32(d); rand32(z);
        keygen_internal(ek, dk, d, z);
        check("ek_valid on genuine ek", ek_valid(ek));

        uint8_t bad[EK_BYTES];
        std::memcpy(bad, ek, EK_BYTES);
        bad[0] = 0x01; bad[1] = (uint8_t)((bad[1] & 0xF0) | 0x0D);  // 3329 = 0xD01
        check("ek_valid rejects non-canonical", !ek_valid(bad));
    }

    std::printf("\n%s (%d failing)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
