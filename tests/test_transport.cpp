// Loopback integration: two real sockets, one thread each.
#include "transport.h"
#include "mlkem_sample.h"
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>
#include <atomic>

using namespace aegis;
using namespace aegis::mlkem768;

static int fails = 0;
static void check(const char* name, bool ok, const char* detail = nullptr) {
    if (!ok) fails++;
    std::printf("%-44s %s\n", name, ok ? "ok" : "FAIL");
    if (!ok && detail) std::printf("    %s\n", detail);
}

static std::mt19937 rng(31337);
static void rnd(uint8_t* b, size_t n) {
    for (size_t i = 0; i < n; i++) b[i] = (uint8_t)(rng() & 0xff);
}

static const uint16_t PORT = 34567;

int main() {
    mlkem::ntt_init();

    uint8_t d[32], z[32], m[32];
    uint8_t ek[EK_BYTES], dk[DK_BYTES];
    rnd(d, 32); rnd(z, 32); rnd(m, 32);
    keygen_internal(ek, dk, d, z);

    // Sizes chosen to straddle the socket buffer: a 4 MiB record forces
    // many partial reads, which is exactly what naive recv() gets wrong.
    std::vector<size_t> sizes = {0, 1, 64, 1500, 65536, 1 << 20, 4 << 20};
    std::vector<std::vector<uint8_t>> payloads;
    for (size_t n : sizes) {
        std::vector<uint8_t> v(n);
        rnd(v.data(), n);
        payloads.push_back(std::move(v));
    }

    int lfd = tcp_listen(PORT);
    check("listen on loopback", lfd >= 0);
    if (lfd < 0) return 1;

    std::atomic<int> srv_ok{0}, srv_echoed{0};

    std::thread server([&] {
        int cfd = tcp_accept(lfd);
        if (cfd < 0) return;
        tcp_nodelay(cfd);
        Session s;
        if (!server_handshake(cfd, s, dk)) { tcp_close(cfd); return; }
        srv_ok = 1;
        for (size_t i = 0; i < sizes.size(); i++) {
            std::vector<uint8_t> got;
            if (!recv_record(cfd, s, got)) break;
            if (got.size() != payloads[i].size()) break;
            if (got.size() && std::memcmp(got.data(), payloads[i].data(),
                                          got.size()) != 0) break;
            if (!send_record(cfd, s, got.data(), got.size())) break;
            srv_echoed++;
        }
        s.close();
        tcp_close(cfd);
    });

    int fd = tcp_connect("127.0.0.1", PORT);
    check("connect", fd >= 0);
    tcp_nodelay(fd);

    Session c;
    check("client handshake", client_handshake(fd, c, ek, m));

    bool echo_ok = true;
    for (size_t i = 0; i < sizes.size(); i++) {
        if (!send_record(fd, c, payloads[i].data(), payloads[i].size())) {
            echo_ok = false; break;
        }
        std::vector<uint8_t> back;
        if (!recv_record(fd, c, back)) { echo_ok = false; break; }
        if (back.size() != payloads[i].size()) { echo_ok = false; break; }
        if (back.size() && std::memcmp(back.data(), payloads[i].data(),
                                       back.size()) != 0) {
            echo_ok = false; break;
        }
    }
    check("echo round trip, 0 B to 4 MiB", echo_ok);

    server.join();
    check("server established session", srv_ok == 1);
    char det[64];
    std::snprintf(det, sizeof det, "%d/%zu", srv_echoed.load(), sizes.size());
    check("server echoed every record", srv_echoed == (int)sizes.size(), det);

    c.close();
    tcp_close(fd);
    tcp_close(lfd);

    std::printf("\n%s (%d failing)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
