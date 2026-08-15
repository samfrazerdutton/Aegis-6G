// THE conformance gate: NIST ACVP vectors for ML-KEM-768 (FIPS 203).
// Everything before this proves the construction is self-consistent.
// Only this proves it is ML-KEM.
#include "mlkem768.h"
#include "mlkem_sample.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace aegis;
using namespace aegis::mlkem768;

static int fails = 0;
static void report(const char* name, int bad, int total) {
    if (bad) fails++;
    std::printf("%-40s %s  (%d/%d)\n", name, bad ? "FAIL" : "ok",
                total - bad, total);
}

static std::vector<uint8_t> unhex(const std::string& s) {
    std::vector<uint8_t> v(s.size() / 2);
    for (size_t i = 0; i < v.size(); i++)
        v[i] = (uint8_t)std::stoul(s.substr(2 * i, 2), nullptr, 16);
    return v;
}
static std::string hex(const uint8_t* b, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; i++) { s += d[b[i] >> 4]; s += d[b[i] & 15]; }
    return s;
}
static std::string lower(std::string s) {
    for (auto& c : s) if (c >= 'A' && c <= 'F') c += 32;
    return s;
}

static std::vector<std::vector<std::string>> load(const char* path) {
    std::ifstream f(path);
    if (!f) {
        std::printf("MISSING %s -- run tools/fetch_kat.sh && tools/extract_kat.py\n",
                    path);
        std::exit(2);
    }
    int n; f >> n; f.ignore();
    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (rows.size() < (size_t)n && std::getline(f, line)) {
        std::istringstream ss(line);
        std::vector<std::string> cols;
        std::string tok;
        while (ss >> tok) cols.push_back(tok);
        if (!cols.empty()) rows.push_back(cols);
    }
    return rows;
}

int main() {
    mlkem::ntt_init();

    // keygen: d,z -> ek,dk must match byte for byte.
    {
        auto rows = load("tests/kat/keygen.txt");
        int badek = 0, baddk = 0;
        for (auto& r : rows) {
            auto d = unhex(r[1]), z = unhex(r[2]);
            uint8_t ek[EK_BYTES], dk[DK_BYTES];
            keygen_internal(ek, dk, d.data(), z.data());
            if (hex(ek, EK_BYTES) != lower(r[3])) badek++;
            if (hex(dk, DK_BYTES) != lower(r[4])) baddk++;
        }
        report("ACVP keyGen: ek matches", badek, (int)rows.size());
        report("ACVP keyGen: dk matches", baddk, (int)rows.size());
    }

    // encaps: ek,m -> c,k must match byte for byte.
    {
        auto rows = load("tests/kat/encaps.txt");
        int badc = 0, badk = 0;
        for (auto& r : rows) {
            auto ek = unhex(r[1]), m = unhex(r[2]);
            uint8_t ct[CT_BYTES], ss[SS_BYTES];
            encaps_internal(ss, ct, ek.data(), m.data());
            if (hex(ct, CT_BYTES) != lower(r[3])) badc++;
            if (hex(ss, SS_BYTES) != lower(r[4])) badk++;
        }
        report("ACVP encaps: ciphertext matches", badc, (int)rows.size());
        report("ACVP encaps: shared secret matches", badk, (int)rows.size());
    }

    // decaps: includes 5 MODIFIED-ciphertext cases whose expected k is the
    // implicit-rejection value. NIST checking our rejection output is far
    // stronger than our own self-consistency check.
    {
        auto rows = load("tests/kat/decaps.txt");
        int bad = 0, badmod = 0, nmod = 0;
        for (auto& r : rows) {
            auto dk = unhex(r[1]), c = unhex(r[2]);
            uint8_t ss[SS_BYTES];
            decaps_internal(ss, dk.data(), c.data());
            bool ok = (hex(ss, SS_BYTES) == lower(r[3]));
            bool mod = r[4].find("modified") != std::string::npos;
            if (mod) { nmod++; if (!ok) badmod++; }
            if (!ok) bad++;
        }
        report("ACVP decaps: shared secret matches", bad, (int)rows.size());
        report("ACVP decaps: implicit rejection exact", badmod, nmod);
    }

    // encapsulation key check: 5 valid, 5 with coefficients >= q.
    {
        auto rows = load("tests/kat/ekcheck.txt");
        int bad = 0;
        for (auto& r : rows) {
            auto ek = unhex(r[1]);
            bool want = (r[2] == "1");
            if (ek_valid(ek.data()) != want) bad++;
        }
        report("ACVP ek modulus check", bad, (int)rows.size());
    }

    std::printf("\n%s (%d failing gates)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
