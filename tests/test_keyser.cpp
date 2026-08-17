// Evaluation-key serialization: exact round trip + rejection of hostile input.
// The server parses key material from an untrusted client, so the rejection
// half matters as much as the round-trip half.
#include "keyser.h"
#include "gpufhe_api.h"
#include "keygen.h"
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

using namespace aegis;

static int fails = 0;
static void check(const char* name, bool ok, const char* detail = nullptr) {
    if (!ok) fails++;
    std::printf("%-46s %s\n", name, ok ? "ok" : "FAIL");
    if (!ok && detail) std::printf("    %s\n", detail);
}

int main() {
    const uint32_t n = 1024, sizeQ = 5, sizeP = 2, numPart = 3;
    const uint64_t ns = 1; const double sigma = 3.19;

    std::vector<uint64_t> mod, modP, root, rootQP, modQP;
    gpufhe::native_primes(mod, sizeQ, 50, n, {});
    gpufhe::native_primes(modP, sizeP, 51, n, mod);
    modQP = mod; for (auto p : modP) modQP.push_back(p);
    root.resize(sizeQ);
    for (uint32_t i = 0; i < sizeQ; i++) root[i] = gpufhe::native_root(n, mod[i]);
    rootQP = root; for (auto p : modP) rootQP.push_back(gpufhe::native_root(n, p));

    gpufhe::set_secret_hamming_weight(0);
    auto KP = gpufhe::keygen_host(n, modQP, rootQP, ns, sigma, 101);
    std::vector<uint64_t> PModq(sizeQ + sizeP);
    for (uint32_t t = 0; t < sizeQ + sizeP; t++) {
        uint64_t q = modQP[t], P = 1 % q;
        for (uint32_t j = 0; j < sizeP; j++)
            P = (uint64_t)(((unsigned __int128)P * (modQP[sizeQ+j] % q)) % q);
        PModq[t] = P;
    }

    gpufhe::KeySwitchConstants K; K.n = n;
    gpufhe::compute_keyswitch_constants(K, mod, modP, numPart);
    for (uint32_t i = 0; i < sizeQ; i++) { K.rootModList.push_back(mod[i]); K.rootValList.push_back(root[i]); }
    for (uint32_t j = 0; j < sizeP; j++) { K.rootModList.push_back(modQP[sizeQ+j]); K.rootValList.push_back(rootQP[sizeQ+j]); }
    gpufhe::evalkeygen_host(K, KP.s, KP.pkA, KP.pkB, PModq, modQP, rootQP, ns, sigma, 202);

    std::vector<uint8_t> blob;
    ks_serialize(blob, K);
    std::printf("\nserialized rotation key: %zu bytes (%.2f MB) at n=%u\n",
                blob.size(), blob.size() / 1048576.0, n);
    std::printf("60 keys would be %.1f MB at n=%u, %.1f MB at n=16384\n\n",
                blob.size() * 60 / 1048576.0, n, blob.size() * 60 * 16.0 / 1048576.0);

    // 1. Exact round trip.
    {
        gpufhe::KeySwitchConstants K2;
        Reader r(blob.data(), blob.size());
        bool ok = ks_deserialize(r, K2, 65536, 64);
        check("deserialize succeeds", ok);
        check("consumed the whole blob", r.o == blob.size());
        check("round trip is field-for-field exact", ok && ks_equal(K, K2));
    }

    // 2. Truncation at EVERY length must fail cleanly, never crash or
    //    over-read. This is the property a length-prefixed parser gets
    //    wrong most often.
    {
        bool ok = true; size_t accepted = 0;
        for (size_t len = 0; len < blob.size(); len += 97) {
            gpufhe::KeySwitchConstants K2;
            Reader r(blob.data(), len);
            if (ks_deserialize(r, K2, 65536, 64)) { accepted++; ok = false; }
        }
        char d[64]; std::snprintf(d, sizeof d, "%zu truncations accepted", accepted);
        check("every truncated blob rejected", ok, d);
    }

    // 3. Corrupted headers must be rejected on their face.
    {
        auto hdr_attack = [&](size_t off, uint32_t val, const char* name) {
            std::vector<uint8_t> b = blob;
            for (int i = 0; i < 4; i++) b[off + i] = (uint8_t)(val >> (8 * i));
            gpufhe::KeySwitchConstants K2;
            Reader r(b.data(), b.size());
            check(name, !ks_deserialize(r, K2, 65536, 64));
        };
        hdr_attack(0,  0xDEADBEEF, "wrong magic rejected");
        hdr_attack(4,  0x7FFFFFFF, "absurd n rejected");
        hdr_attack(4,  1000,       "non-power-of-two n rejected");
        hdr_attack(8,  0x7FFFFFFF, "absurd sizeQl rejected");
        hdr_attack(16, 0x7FFFFFFF, "absurd numPart rejected");
        hdr_attack(16, 0,          "zero numPart rejected");
        hdr_attack(24, 0x7FFFFFFF, "absurd fullQ rejected");
        hdr_attack(24, 2,           "fullQ < sizeQl rejected");
        hdr_attack(28, 0x7FFFFFFF, "absurd evalKeyTowers rejected");
        hdr_attack(28, 3,           "evalKeyTowers < fullQ+sizeP rejected");
    }

    // 4. A blob whose declared array lengths disagree with its header must
    //    fail the cross-check, not be quietly accepted.
    {
        gpufhe::KeySwitchConstants Kbad = K;
        Kbad.av.pop_back();                    // numPart says 3, av has 2
        std::vector<uint8_t> b; ks_serialize(b, Kbad);
        gpufhe::KeySwitchConstants K2;
        Reader r(b.data(), b.size());
        check("av count != numPart rejected", !ks_deserialize(r, K2, 65536, 64));
    }

    // 5. Server-side limits must bind even on a well-formed blob.
    {
        gpufhe::KeySwitchConstants K2;
        Reader r(blob.data(), blob.size());
        check("n above server limit rejected",
              !ks_deserialize(r, K2, 512, 64));
    }

    // 6. Random byte corruption must never crash. Value corruption may well
    //    deserialize fine -- that is expected and is what the AEAD tag is
    //    for; what matters is that the parser stays in bounds.
    {
        std::mt19937 rng(5);
        for (int t = 0; t < 400; t++) {
            std::vector<uint8_t> b = blob;
            for (int k = 0; k < 8; k++) b[rng() % b.size()] ^= (uint8_t)(1 + rng() % 255);
            gpufhe::KeySwitchConstants K2;
            Reader r(b.data(), b.size());
            ks_deserialize(r, K2, 65536, 64);   // must not crash
        }
        check("400 corrupted blobs parsed without crashing", true);
    }

    std::printf("\n%s (%d failing)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
