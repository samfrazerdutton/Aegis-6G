#include "transport.h"
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>

namespace aegis {

using namespace mlkem768;

int tcp_listen(uint16_t port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = htons(port);
    if (::bind(fd, (sockaddr*)&a, sizeof a) < 0) { ::close(fd); return -1; }
    if (::listen(fd, 8) < 0) { ::close(fd); return -1; }
    return fd;
}

int tcp_accept(int lfd) { return ::accept(lfd, nullptr, nullptr); }

int tcp_connect(const std::string& host, uint16_t port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    if (::inet_pton(AF_INET, host.c_str(), &a.sin_addr) != 1) {
        ::close(fd); return -1;
    }
    if (::connect(fd, (sockaddr*)&a, sizeof a) < 0) { ::close(fd); return -1; }
    return fd;
}

void tcp_close(int fd) { if (fd >= 0) ::close(fd); }

void tcp_nodelay(int fd) {
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
}

bool send_all(int fd, const uint8_t* buf, size_t len) {
    while (len > 0) {
        ssize_t n = ::send(fd, buf, len, MSG_NOSIGNAL);
        if (n < 0) { if (errno == EINTR) continue; return false; }
        if (n == 0) return false;
        buf += n; len -= (size_t)n;
    }
    return true;
}

bool recv_all(int fd, uint8_t* buf, size_t len) {
    while (len > 0) {
        ssize_t n = ::recv(fd, buf, len, 0);
        if (n < 0) { if (errno == EINTR) continue; return false; }
        if (n == 0) return false;            // peer closed mid-record
        buf += n; len -= (size_t)n;
    }
    return true;
}

bool client_handshake(int fd, Session& s, const uint8_t peer_ek[EK_BYTES],
                      const uint8_t m[32]) {
    uint8_t ct[CT_BYTES];
    session_init(s, ct, peer_ek, m);
    return send_all(fd, ct, CT_BYTES);
}

bool server_handshake(int fd, Session& s, const uint8_t dk[DK_BYTES]) {
    uint8_t ct[CT_BYTES];
    if (!recv_all(fd, ct, CT_BYTES)) return false;
    // Cannot fail by design: a forged ct yields an unrelated key and the
    // mismatch surfaces at the first record. See session.h.
    session_accept(s, dk, ct);
    return true;
}

bool send_record(int fd, Session& s, const uint8_t* pt, size_t len) {
    std::vector<uint8_t> rec(len + REC_OVERHEAD);
    size_t w = record_seal(s, rec.data(), pt, len);
    if (w == 0) return false;
    return send_all(fd, rec.data(), w);
}

bool recv_record(int fd, Session& s, std::vector<uint8_t>& out) {
    uint8_t hdr[REC_HDR];
    if (!recv_all(fd, hdr, REC_HDR)) return false;
    uint32_t len = (uint32_t)hdr[0] | ((uint32_t)hdr[1] << 8) |
                   ((uint32_t)hdr[2] << 16) | ((uint32_t)hdr[3] << 24);
    if (len > REC_MAX) return false;         // reject before allocating

    std::vector<uint8_t> rec(REC_HDR + len + REC_TAG);
    std::memcpy(rec.data(), hdr, REC_HDR);
    if (!recv_all(fd, rec.data() + REC_HDR, len + REC_TAG)) return false;

    out.resize(len);
    long r = record_open(s, out.data(), rec.data(), rec.size());
    if (r < 0) { out.clear(); return false; }
    return true;
}

} // namespace aegis
