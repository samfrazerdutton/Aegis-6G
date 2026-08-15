// Gate: FIPS 202 known-answer vectors. No tolerance -- byte-exact or fail.
#include "keccak.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace aegis;

static std::string hex(const uint8_t* b, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; i++) { s += d[b[i] >> 4]; s += d[b[i] & 15]; }
    return s;
}

static int fails = 0;
static void check(const char* name, const std::string& got,
                  const std::string& want) {
    bool ok = (got == want);
    if (!ok) fails++;
    std::printf("%-28s %s\n", name, ok ? "ok" : "FAIL");
    if (!ok) {
        std::printf("    got  %s\n    want %s\n", got.c_str(), want.c_str());
    }
}

int main() {
    const uint8_t* empty = reinterpret_cast<const uint8_t*>("");
    const uint8_t* abc = reinterpret_cast<const uint8_t*>("abc");

    uint8_t h32[32], h64[64];

    sha3_256(empty, 0, h32);
    check("SHA3-256(\"\")", hex(h32, 32),
          "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a");

    sha3_256(abc, 3, h32);
    check("SHA3-256(\"abc\")", hex(h32, 32),
          "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532");

    sha3_512(empty, 0, h64);
    check("SHA3-512(\"\")", hex(h64, 64),
          "a69f73cca23a9ac5c8b567dc185a756e97c982164fe25859e0d1dcc1475c80a6"
          "15b2123af1f5f94c11e3e9402c3ac558f500199d95b6d3e301758586281dcd26");

    shake128(h32, 32, empty, 0);
    check("SHAKE128(\"\", 32)", hex(h32, 32),
          "7f9c2ba4e88f827d616045507605853ed73b8093f6efbc88eb1a6eacfa66ef26");

    shake256(h32, 32, empty, 0);
    check("SHAKE256(\"\", 32)", hex(h32, 32),
          "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f");

    // Incremental XOF must agree with one-shot at every split point, and
    // across a rate boundary (168 bytes for SHAKE128). This is the property
    // the rejection sampler actually depends on.
    {
        const size_t N = 400;
        std::vector<uint8_t> ref(N), inc(N);
        shake128(ref.data(), N, abc, 3);
        bool ok = true;
        for (size_t split = 1; split < N && ok; split += 7) {
            Xof x;
            x.init128(abc, 3);
            x.squeeze(inc.data(), split);
            x.squeeze(inc.data() + split, N - split);
            if (std::memcmp(ref.data(), inc.data(), N) != 0) {
                std::printf("    split %zu diverges\n", split);
                ok = false;
            }
        }
        check("Xof incremental == one-shot", ok ? "ok" : "bad", "ok");
    }

    std::printf("\n%s (%d failing)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
