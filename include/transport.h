#pragma once
#include "session.h"
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

// Blocking TCP transport for Aegis-6G sessions.
// Thin by design: the crypto is gated elsewhere; this layer only has to
// move bytes without losing or reordering them.
namespace aegis {

// Handshake wire format: initiator sends CT_BYTES raw. No negotiation, no
// version byte on the wire -- the version is bound into the key schedule
// label instead, so a mismatched build fails at the first record.

int  tcp_listen(uint16_t port);                 // returns listening fd, -1 err
int  tcp_accept(int lfd);                       // returns conn fd, -1 err
int  tcp_connect(const std::string& host, uint16_t port);
void tcp_close(int fd);
void tcp_nodelay(int fd);                       // disable Nagle for latency

// Full-buffer helpers: partial writes and short reads are the normal case on
// a socket, and a naive send()/recv() pair silently truncates large records.
bool send_all(int fd, const uint8_t* buf, size_t len);
bool recv_all(int fd, uint8_t* buf, size_t len);

// Handshake.
bool client_handshake(int fd, Session& s,
                      const uint8_t peer_ek[mlkem768::EK_BYTES],
                      const uint8_t m[32]);
bool server_handshake(int fd, Session& s,
                      const uint8_t dk[mlkem768::DK_BYTES]);

// Records.
bool send_record(int fd, Session& s, const uint8_t* pt, size_t len);
// Reads one record. Returns false on I/O error, peer close, or auth failure.
// out is resized to the plaintext length.
bool recv_record(int fd, Session& s, std::vector<uint8_t>& out);

} // namespace aegis
