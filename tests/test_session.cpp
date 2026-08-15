// Session state machine gates. No I/O -- these exercise adversarial cases
// a loopback socket cannot easily produce.
#include "session.h"
#include "mlkem_sample.h"
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace aegis;
using namespace aegis::mlkem768;

static int fails = 0;
static void check(const char* name, bool ok, const char* detail = nullptr) {
    if (!ok) fails++;
    std::printf("%-44s %s\n", name, ok ? "ok" : "FAIL");
    if (!ok && detail) std::printf("    %s\n", detail);
}

static std::mt19937 rng(7);
static void rnd(uint8_t* b, size_t n) {
    for (size_t i = 0; i < n; i++) b[i] = (uint8_t)(rng() & 0xff);
}

struct Pair { Session a, b; uint8_t ek[EK_BYTES], dk[DK_BYTES]; };

static void handshake(Pair& p) {
    uint8_t d[32], z[32], m[32], ct[CT_BYTES];
    rnd(d, 32); rnd(z, 32); rnd(m, 32);
    keygen_internal(p.ek, p.dk, d, z);
    session_init(p.a, ct, p.ek, m);
    session_accept(p.b, p.dk, ct);
}

int main() {
    mlkem::ntt_init();

    // 1. Both sides agree, and the directions are mirrored.
    {
        Pair p; handshake(p);
        check("session ids match",
              std::memcmp(p.a.id, p.b.id, 32) == 0);
        check("initiator tx == responder rx",
              std::memcmp(p.a.tx_key, p.b.rx_key, 32) == 0);
        check("initiator rx == responder tx",
              std::memcmp(p.a.rx_key, p.b.tx_key, 32) == 0);
        check("directions use different keys",
              std::memcmp(p.a.tx_key, p.a.rx_key, 32) != 0);
    }

    // 2. Bidirectional traffic, many records, varied sizes.
    {
        Pair p; handshake(p);
        bool ok = true;
        for (int i = 0; i < 300 && ok; i++) {
            size_t n = (size_t)(rng() % 4000);
            std::vector<uint8_t> pt(n), rec(n + REC_OVERHEAD), out(n);
            rnd(pt.data(), n);
            Session& tx = (i & 1) ? p.b : p.a;
            Session& rx = (i & 1) ? p.a : p.b;
            size_t w = record_seal(tx, rec.data(), pt.data(), n);
            long r = record_open(rx, out.data(), rec.data(), w);
            if (r != (long)n) ok = false;
            else if (n && std::memcmp(out.data(), pt.data(), n) != 0) ok = false;
        }
        check("300 bidirectional records round trip", ok);
    }

    // 3. Replay: re-delivering a record fails because rx_seq advanced.
    {
        Pair p; handshake(p);
        uint8_t pt[64], rec[64 + REC_OVERHEAD], out[64];
        rnd(pt, 64);
        size_t w = record_seal(p.a, rec, pt, 64);
        check("first delivery accepted",
              record_open(p.b, out, rec, w) == 64);
        check("replayed record rejected",
              record_open(p.b, out, rec, w) == -1);
    }

    // 4. Reflection: a record from A must not verify at A.
    {
        Pair p; handshake(p);
        uint8_t pt[64], rec[64 + REC_OVERHEAD], out[64];
        rnd(pt, 64);
        size_t w = record_seal(p.a, rec, pt, 64);
        check("reflected record rejected at sender",
              record_open(p.a, out, rec, w) == -1);
    }

    // 5. Dropped record desynchronises and is detected.
    {
        Pair p; handshake(p);
        uint8_t pt[32], r1[32 + REC_OVERHEAD], r2[32 + REC_OVERHEAD], out[32];
        rnd(pt, 32);
        record_seal(p.a, r1, pt, 32);                 // seq 0, dropped
        size_t w2 = record_seal(p.a, r2, pt, 32);     // seq 1
        check("record delivered out of order rejected",
              record_open(p.b, out, r2, w2) == -1);
    }

    // 6. Tampering, and the receive counter must not advance on failure.
    {
        Pair p; handshake(p);
        uint8_t pt[100], rec[100 + REC_OVERHEAD], out[100];
        rnd(pt, 100);
        size_t w = record_seal(p.a, rec, pt, 100);
        uint8_t bad[100 + REC_OVERHEAD];
        std::memcpy(bad, rec, w);
        bad[REC_HDR + 50] ^= 0x20;
        uint64_t before = p.b.rx_seq;
        check("tampered payload rejected",
              record_open(p.b, out, bad, w) == -1);
        check("rx_seq not advanced on failure", p.b.rx_seq == before);
        check("genuine record still accepted after",
              record_open(p.b, out, rec, w) == 100);

        // Length field is authenticated too.
        std::memcpy(bad, rec, w);
        bad[0] ^= 1;
        check("tampered length header rejected",
              record_open(p.b, out, bad, w) == -1);
    }

    // 7. THE FO GATE. Initiator encapsulates to the WRONG ek (a MITM's key).
    //    session_accept must NOT fail -- implicit rejection gives the
    //    responder an unrelated key -- and the mismatch must surface at the
    //    first record instead.
    {
        uint8_t d[32], z[32], m[32];
        uint8_t ekR[EK_BYTES], dkR[DK_BYTES];
        uint8_t ekM[EK_BYTES], dkM[DK_BYTES];
        rnd(d, 32); rnd(z, 32); keygen_internal(ekR, dkR, d, z);
        rnd(d, 32); rnd(z, 32); keygen_internal(ekM, dkM, d, z);

        Session a, b;
        uint8_t ct[CT_BYTES];
        rnd(m, 32);
        session_init(a, ct, ekM, m);       // initiator targets attacker key
        session_accept(b, dkR, ct);        // real responder decapsulates

        check("handshake to wrong ek does not fail", b.established);
        check("mismatched sessions derive different ids",
              std::memcmp(a.id, b.id, 32) != 0);

        uint8_t pt[48], rec[48 + REC_OVERHEAD], out[48];
        rnd(pt, 48);
        size_t w = record_seal(a, rec, pt, 48);
        check("mismatch surfaces at first record",
              record_open(b, out, rec, w) == -1);
    }

    // 8. Nonce discipline: sequence numbers strictly increase and never
    //    reset, so a (key, nonce) pair is never reused within a session.
    {
        Pair p; handshake(p);
        uint8_t pt[16], rec[16 + REC_OVERHEAD];
        rnd(pt, 16);
        bool ok = true;
        for (uint64_t i = 0; i < 1000; i++) {
            if (p.a.tx_seq != i) { ok = false; break; }
            record_seal(p.a, rec, pt, 16);
        }
        check("tx_seq strictly increasing, no reset",
              ok && p.a.tx_seq == 1000);
    }

    // 9. close() zeroizes.
    {
        Pair p; handshake(p);
        p.a.close();
        uint8_t zero[32] = {0};
        check("close() zeroizes key material",
              std::memcmp(p.a.tx_key, zero, 32) == 0 &&
              std::memcmp(p.a.rx_key, zero, 32) == 0 && !p.a.established);
    }

    std::printf("\n%s (%d failing)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
