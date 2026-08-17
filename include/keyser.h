#pragma once
#include "keyswitch.h"
#include <cstdint>
#include <vector>

// Wire format for CKKS evaluation keys.
//
// THREAT NOTE: the server deserializes key material supplied by a client it
// does not trust. Every read is bounds-checked and every declared length is
// checked against limits derived from the server's own parameters. A blob
// claiming numPart = 2^31 must fail cleanly, not allocate.
namespace aegis {

struct Reader {
    const uint8_t* p; size_t n; size_t o = 0; bool ok = true;
    Reader(const uint8_t* b, size_t len) : p(b), n(len) {}
    bool need(size_t k) { if (!ok || o + k > n) { ok = false; return false; } return true; }
};

void put_u32(std::vector<uint8_t>& b, uint32_t v);
void put_u64(std::vector<uint8_t>& b, uint64_t v);
void put_v64(std::vector<uint8_t>& b, const std::vector<uint64_t>& v);
void put_v32(std::vector<uint8_t>& b, const std::vector<uint32_t>& v);
void put_vv64(std::vector<uint8_t>& b, const std::vector<std::vector<uint64_t>>& v);

uint32_t get_u32(Reader& r);
uint64_t get_u64(Reader& r);
bool get_v64(Reader& r, std::vector<uint64_t>& out, size_t maxlen);
bool get_v32(Reader& r, std::vector<uint32_t>& out, size_t maxlen);
bool get_vv64(Reader& r, std::vector<std::vector<uint64_t>>& out,
              size_t maxouter, size_t maxinner);

void ks_serialize(std::vector<uint8_t>& b, const gpufhe::KeySwitchConstants& K);

// maxn / maxtowers bound what the server is willing to accept.
bool ks_deserialize(Reader& r, gpufhe::KeySwitchConstants& K,
                    uint32_t maxn, uint32_t maxtowers);

bool ks_equal(const gpufhe::KeySwitchConstants& a,
              const gpufhe::KeySwitchConstants& b);

} // namespace aegis
