#include "keyser.h"
#include <cstring>

namespace aegis {

// Hard ceilings, independent of anything the peer claims.
static const size_t MAX_ELEMS = 64u * 1024u * 1024u;   // 512 MB per vector
static const uint32_t MAX_PART = 64;

void put_u32(std::vector<uint8_t>& b, uint32_t v) {
    for (int i = 0; i < 4; i++) b.push_back((uint8_t)(v >> (8 * i)));
}
void put_u64(std::vector<uint8_t>& b, uint64_t v) {
    for (int i = 0; i < 8; i++) b.push_back((uint8_t)(v >> (8 * i)));
}
void put_v64(std::vector<uint8_t>& b, const std::vector<uint64_t>& v) {
    put_u64(b, (uint64_t)v.size());
    size_t off = b.size();
    b.resize(off + v.size() * 8);
    std::memcpy(b.data() + off, v.data(), v.size() * 8);   // host is LE
}
void put_v32(std::vector<uint8_t>& b, const std::vector<uint32_t>& v) {
    put_u64(b, (uint64_t)v.size());
    for (uint32_t x : v) put_u32(b, x);
}
void put_vv64(std::vector<uint8_t>& b, const std::vector<std::vector<uint64_t>>& v) {
    put_u64(b, (uint64_t)v.size());
    for (const auto& in : v) put_v64(b, in);
}

uint32_t get_u32(Reader& r) {
    if (!r.need(4)) return 0;
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) v |= (uint32_t)r.p[r.o + i] << (8 * i);
    r.o += 4; return v;
}
uint64_t get_u64(Reader& r) {
    if (!r.need(8)) return 0;
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)r.p[r.o + i] << (8 * i);
    r.o += 8; return v;
}
bool get_v64(Reader& r, std::vector<uint64_t>& out, size_t maxlen) {
    uint64_t len = get_u64(r);
    if (!r.ok) return false;
    if (len > maxlen || len > MAX_ELEMS) { r.ok = false; return false; }
    if (!r.need((size_t)len * 8)) return false;
    out.resize((size_t)len);
    if (len) std::memcpy(out.data(), r.p + r.o, (size_t)len * 8);
    r.o += (size_t)len * 8;
    return true;
}
bool get_v32(Reader& r, std::vector<uint32_t>& out, size_t maxlen) {
    uint64_t len = get_u64(r);
    if (!r.ok) return false;
    if (len > maxlen || len > MAX_ELEMS) { r.ok = false; return false; }
    if (!r.need((size_t)len * 4)) return false;
    out.resize((size_t)len);
    for (size_t i = 0; i < out.size(); i++) out[i] = get_u32(r);
    return r.ok;
}
bool get_vv64(Reader& r, std::vector<std::vector<uint64_t>>& out,
              size_t maxouter, size_t maxinner) {
    uint64_t cnt = get_u64(r);
    if (!r.ok) return false;
    if (cnt > maxouter) { r.ok = false; return false; }
    out.assign((size_t)cnt, {});
    for (size_t i = 0; i < out.size(); i++)
        if (!get_v64(r, out[i], maxinner)) return false;
    return true;
}

void ks_serialize(std::vector<uint8_t>& b, const gpufhe::KeySwitchConstants& K) {
    put_u32(b, 0x4B534131);              // "KSA1"
    put_u32(b, K.n); put_u32(b, K.sizeQl); put_u32(b, K.sizeP);
    put_u32(b, K.numPart); put_u32(b, K.alpha); put_u32(b, K.fullQ);
    put_u32(b, K.evalKeyTowers);
    put_v64(b, K.qMod); put_v64(b, K.pMod);
    put_v32(b, K.sizePart); put_v32(b, K.sizeCompl); put_v32(b, K.startIdx);
    put_vv64(b, K.partQHatInv);   put_vv64(b, K.partQHatInvPrec);
    put_vv64(b, K.partSrcMod);    put_vv64(b, K.partQHatModp);
    put_vv64(b, K.partComplMod);  put_vv64(b, K.partBMuLo);
    put_vv64(b, K.partBMuHi);
    put_vv64(b, K.av); put_vv64(b, K.bv);
    put_v64(b, K.pHatInv); put_v64(b, K.pHatInvPrec); put_v64(b, K.pHatModq);
    put_v64(b, K.mdBMuLo); put_v64(b, K.mdBMuHi); put_v64(b, K.pInvModq);
    put_v64(b, K.rootModList); put_v64(b, K.rootValList);
}

bool ks_deserialize(Reader& r, gpufhe::KeySwitchConstants& K,
                    uint32_t maxn, uint32_t maxtowers) {
    if (get_u32(r) != 0x4B534131) { r.ok = false; return false; }
    K.n = get_u32(r); K.sizeQl = get_u32(r); K.sizeP = get_u32(r);
    K.numPart = get_u32(r); K.alpha = get_u32(r); K.fullQ = get_u32(r);
    K.evalKeyTowers = get_u32(r);
    if (!r.ok) return false;

    // Reject implausible headers BEFORE any sizing decision depends on them.
    if (K.n == 0 || K.n > maxn || (K.n & (K.n - 1)) != 0) { r.ok = false; return false; }
    if (K.sizeQl == 0 || K.sizeQl > maxtowers) { r.ok = false; return false; }
    if (K.sizeP == 0 || K.sizeP > maxtowers) { r.ok = false; return false; }
    if (K.numPart == 0 || K.numPart > MAX_PART) { r.ok = false; return false; }
    if (K.evalKeyTowers > 2 * maxtowers) { r.ok = false; return false; }
    // fullQ is NOT inert: the eval-key index is (i>=sizeQl)? i+delta : i with
    // delta = fullQ - sizeQl, so an inflated fullQ reads past the end of the
    // key. The largest index touched is fullQ + sizeP - 1.
    if (K.fullQ < K.sizeQl || K.fullQ > maxtowers) { r.ok = false; return false; }
    if ((size_t)K.fullQ + K.sizeP > K.evalKeyTowers) { r.ok = false; return false; }

    const size_t tw = (size_t)maxtowers;
    const size_t keylen = (size_t)K.evalKeyTowers * K.n;

    if (!get_v64(r, K.qMod, tw) || !get_v64(r, K.pMod, tw)) return false;
    if (!get_v32(r, K.sizePart, MAX_PART) ||
        !get_v32(r, K.sizeCompl, MAX_PART) ||
        !get_v32(r, K.startIdx, MAX_PART)) return false;
    if (!get_vv64(r, K.partQHatInv,     MAX_PART, tw) ||
        !get_vv64(r, K.partQHatInvPrec, MAX_PART, tw) ||
        !get_vv64(r, K.partSrcMod,      MAX_PART, tw) ||
        !get_vv64(r, K.partQHatModp,    MAX_PART, tw * tw * 4) ||
        !get_vv64(r, K.partComplMod,    MAX_PART, tw * 2) ||
        !get_vv64(r, K.partBMuLo,       MAX_PART, tw * 2) ||
        !get_vv64(r, K.partBMuHi,       MAX_PART, tw * 2)) return false;
    if (!get_vv64(r, K.av, MAX_PART, keylen) ||
        !get_vv64(r, K.bv, MAX_PART, keylen)) return false;
    if (!get_v64(r, K.pHatInv, tw) || !get_v64(r, K.pHatInvPrec, tw) ||
        !get_v64(r, K.pHatModq, tw * tw * 4) ||
        !get_v64(r, K.mdBMuLo, tw) || !get_v64(r, K.mdBMuHi, tw) ||
        !get_v64(r, K.pInvModq, tw)) return false;
    if (!get_v64(r, K.rootModList, tw * 2) ||
        !get_v64(r, K.rootValList, tw * 2)) return false;

    // Cross-checks: the arrays must agree with the header, not just fit.
    if (K.av.size() != K.numPart || K.bv.size() != K.numPart) { r.ok = false; return false; }
    if (K.sizePart.size() != K.numPart) { r.ok = false; return false; }
    if (K.rootModList.size() != K.rootValList.size()) { r.ok = false; return false; }
    for (const auto& v : K.av) if (v.size() != keylen) { r.ok = false; return false; }
    for (const auto& v : K.bv) if (v.size() != keylen) { r.ok = false; return false; }
    return r.ok;
}

bool ks_equal(const gpufhe::KeySwitchConstants& a,
              const gpufhe::KeySwitchConstants& b) {
    return a.n == b.n && a.sizeQl == b.sizeQl && a.sizeP == b.sizeP &&
           a.numPart == b.numPart && a.alpha == b.alpha && a.fullQ == b.fullQ &&
           a.evalKeyTowers == b.evalKeyTowers &&
           a.qMod == b.qMod && a.pMod == b.pMod &&
           a.sizePart == b.sizePart && a.sizeCompl == b.sizeCompl &&
           a.startIdx == b.startIdx &&
           a.partQHatInv == b.partQHatInv &&
           a.partQHatInvPrec == b.partQHatInvPrec &&
           a.partSrcMod == b.partSrcMod && a.partQHatModp == b.partQHatModp &&
           a.partComplMod == b.partComplMod &&
           a.partBMuLo == b.partBMuLo && a.partBMuHi == b.partBMuHi &&
           a.av == b.av && a.bv == b.bv &&
           a.pHatInv == b.pHatInv && a.pHatInvPrec == b.pHatInvPrec &&
           a.pHatModq == b.pHatModq && a.mdBMuLo == b.mdBMuLo &&
           a.mdBMuHi == b.mdBMuHi && a.pInvModq == b.pInvModq &&
           a.rootModList == b.rootModList && a.rootValList == b.rootValList;
}

} // namespace aegis
