#pragma once
// Encrypted MNIST inference on GPU-Resident-Library. Stage 1: layer-1 matvec.
//
// Gated against forward_bsgs (plaintext, already proven == forward_direct).
// Native parameters throughout -- no OpenFHE anywhere in this file.
#include "gpufhe_api.h"
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include "stage_acc.h"
#include "mlp.h"
// Shared by infer_encrypted (local benchmark) and aegis_demo (client/server).
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <map>
#include <vector>

using namespace aegis;
using clk = std::chrono::steady_clock;
static double ms(clk::time_point t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

static uint64_t mmul(uint64_t a, uint64_t b, uint64_t q) {
    return (uint64_t)(((unsigned __int128)a * b) % q);
}
static uint64_t madd(uint64_t a, uint64_t b, uint64_t q) {
    uint64_t s = a + b; return s >= q ? s - q : s;
}

// ---- parameters ----
static uint32_t N = 1024, SL = 512;           // ring / slots, set from argv
static int g_resident = 0;                    // 0 = host path, 1 = resident

// Same signature both ways, so this is a pure dispatch. The resident path is
// gated bit-exact against the host path inside the library.
static void rot(std::vector<uint64_t>& c0, std::vector<uint64_t>& c1, uint32_t k,
                const gpufhe::KeySwitchConstants& K, uint32_t tw, uint32_t n,
                const std::vector<uint64_t>& mod, const std::vector<uint64_t>& root) {
    if (g_resident) gpufhe::rotate_ct_resident(c0, c1, k, K, tw, n, mod, root);
    else            gpufhe::rotate_ct_host(c0, c1, k, K, tw, n, mod, root);
}
static const uint32_t PER = SLOTS;            // 256-periodic replication
static const uint32_t SIZEQ = 5, SIZEP = 2, NUMPART = 3;
static const uint64_t NS = 1;
static const double   SIGMA = 3.19;

struct Ctx {
    uint32_t n = 0, sizeQ = SIZEQ, sizeP = SIZEP, sizeQlP = SIZEQ + SIZEP;
    std::vector<uint64_t> mod, modP, modQP, root, rootQP, PModq_QP;
    gpufhe::KeyPairHost KPqp, KPq;
    std::map<uint32_t, gpufhe::KeySwitchConstants> rotk;   // by rotation amount
};

// Replicate a PER-length real vector across all SL slots.
static std::vector<std::complex<double>> repl(const std::vector<double>& v) {
    std::vector<std::complex<double>> z(SL);
    for (uint32_t i = 0; i < SL; i++) z[i] = {v[i % PER], 0.0};
    return z;
}

static void setup(Ctx& c, bool with_keys = true) {
    c.n = N;
    gpufhe::set_secret_hamming_weight(0);          // uniform ternary
    gpufhe::native_primes(c.mod,  c.sizeQ, 50, c.n, {});
    gpufhe::native_primes(c.modP, c.sizeP, 51, c.n, c.mod);
    c.modQP = c.mod;
    for (auto p : c.modP) c.modQP.push_back(p);
    c.root.resize(c.sizeQ);
    for (uint32_t i = 0; i < c.sizeQ; i++) c.root[i] = gpufhe::native_root(c.n, c.mod[i]);
    c.rootQP = c.root;
    for (auto p : c.modP) c.rootQP.push_back(gpufhe::native_root(c.n, p));

    if (with_keys) {
        c.KPqp = gpufhe::keygen_host(c.n, c.modQP, c.rootQP, NS, SIGMA, 101);
        c.KPq  = gpufhe::keygen_host(c.n, c.mod,   c.root,   NS, SIGMA, 101);
    }

    c.PModq_QP.resize(c.sizeQlP);
    for (uint32_t t = 0; t < c.sizeQlP; t++) {
        uint64_t q = c.modQP[t], P = 1 % q;
        for (uint32_t j = 0; j < c.sizeP; j++) P = mmul(P, c.modQP[c.sizeQ + j] % q, q);
        c.PModq_QP[t] = P;
    }
}

// Per-level QP view: same seed => same ternary secret, reduced mod fewer primes.
struct Level {
    uint32_t tw;
    std::vector<uint64_t> mod, root, modQP, rootQP, PModq, rs1, rs2;
    gpufhe::KeyPairHost KPqp;
};

static Level make_level(Ctx& c, uint32_t tw, bool with_keys = true) {
    Level L; L.tw = tw;
    L.mod.assign(c.mod.begin(), c.mod.begin() + tw);
    L.root.assign(c.root.begin(), c.root.begin() + tw);
    L.modQP = L.mod;  for (auto p : c.modP) L.modQP.push_back(p);
    L.rootQP = L.root;
    for (uint32_t j = 0; j < c.sizeP; j++) L.rootQP.push_back(c.rootQP[c.sizeQ + j]);
    if (with_keys) L.KPqp = gpufhe::keygen_host(c.n, L.modQP, L.rootQP, NS, SIGMA, 101);
    uint32_t nqp = tw + c.sizeP;
    L.PModq.resize(nqp);
    for (uint32_t t = 0; t < nqp; t++) {
        uint64_t q = L.modQP[t], P = 1 % q;
        for (uint32_t j = 0; j < c.sizeP; j++) P = mmul(P, L.modQP[tw + j] % q, q);
        L.PModq[t] = P;
    }
    if (tw >= 2) gpufhe::native_rescale_consts(L.rs1, L.rs2, L.mod, tw - 1);
    return L;
}

static void fill_roots(gpufhe::KeySwitchConstants& K, Ctx& c, const Level& L) {
    for (uint32_t i = 0; i < L.tw; i++) { K.rootModList.push_back(L.mod[i]); K.rootValList.push_back(L.root[i]); }
    for (uint32_t j = 0; j < c.sizeP; j++) { K.rootModList.push_back(L.modQP[L.tw+j]); K.rootValList.push_back(L.rootQP[L.tw+j]); }
}

static uint32_t npart_for(uint32_t tw) { uint32_t p = (tw + 1) / 2; return p ? p : 1; }

// Relinearisation key (s^2) at this level.
static gpufhe::KeySwitchConstants make_relin(Ctx& c, const Level& L) {
    gpufhe::KeySwitchConstants K; K.n = c.n;
    gpufhe::compute_keyswitch_constants(K, L.mod, c.modP, npart_for(L.tw));
    fill_roots(K, c, L);
    gpufhe::evalkeygen_host(K, L.KPqp.s, L.KPqp.pkA, L.KPqp.pkB, L.PModq,
                            L.modQP, L.rootQP, NS, SIGMA, 202);
    return K;
}

// Rotation key for slot rotation `r` at level L.
static gpufhe::KeySwitchConstants make_rotkey_lv(Ctx& c, const Level& L, uint32_t r) {
    const uint32_t M = 2 * c.n;
    uint64_t k = 1;
    for (uint32_t i = 0; i < r; i++) k = (k * 5) % M;
    std::vector<uint64_t> sA = L.KPqp.s;
    gpufhe::automorphism_eval_host(sA, L.tw + c.sizeP, c.n, (uint32_t)k, L.modQP, L.rootQP);
    gpufhe::KeySwitchConstants K; K.n = c.n;
    gpufhe::compute_keyswitch_constants(K, L.mod, c.modP, npart_for(L.tw));
    fill_roots(K, c, L);
    gpufhe::evalkeygen_host_sold(K, L.KPqp.s, sA, L.KPqp.pkA, L.KPqp.pkB,
                                 L.PModq, L.modQP, L.rootQP, NS, SIGMA, 800 + r);
    return K;
}

static gpufhe::KeySwitchConstants make_rotkey(Ctx& c, uint32_t r) {
    const uint32_t M = 2 * c.n;
    uint64_t k = 1;
    for (uint32_t i = 0; i < r; i++) k = (k * 5) % M;

    std::vector<uint64_t> sA = c.KPqp.s;
    gpufhe::automorphism_eval_host(sA, c.sizeQlP, c.n, (uint32_t)k, c.modQP, c.rootQP);

    gpufhe::KeySwitchConstants K; K.n = c.n;
    gpufhe::compute_keyswitch_constants(K, c.mod, c.modP, NUMPART);
    for (uint32_t i = 0; i < c.sizeQ; i++) { K.rootModList.push_back(c.mod[i]);  K.rootValList.push_back(c.root[i]); }
    for (uint32_t j = 0; j < c.sizeP; j++) { K.rootModList.push_back(c.modQP[c.sizeQ+j]); K.rootValList.push_back(c.rootQP[c.sizeQ+j]); }
    gpufhe::evalkeygen_host_sold(K, c.KPqp.s, sA, c.KPqp.pkA, c.KPqp.pkB,
                                 c.PModq_QP, c.modQP, c.rootQP, NS, SIGMA, 800 + r);
    return K;
}

static uint32_t autok(uint32_t n, uint32_t r) {
    const uint32_t M = 2 * n; uint64_t k = 1;
    for (uint32_t i = 0; i < r; i++) k = (k * 5) % M;
    return (uint32_t)k;
}


// ---------- circuit helpers ----------
struct CtPair { std::vector<uint64_t> c0, c1; };

static void add_pt(CtPair& ct, const std::vector<double>& v, double scale,
                   Ctx& c, const Level& L) {
    std::vector<int64_t> mb;
    gpufhe::encode_host(mb, repl(v), c.n, scale);
    std::vector<uint64_t> ev;
    gpufhe::pt_to_eval_host(ev, mb, L.tw, c.n, L.mod, L.root);
    for (uint32_t t = 0; t < L.tw; t++) {
        uint64_t q = L.mod[t];
        for (uint32_t k = 0; k < c.n; k++) {
            size_t x = (size_t)t * c.n + k;
            ct.c0[x] = madd(ct.c0[x], ev[x], q);
        }
    }
}

// One rescale step: tw -> tw-1, scale /= mod[tw-1].
static void rescale(CtPair& ct, Ctx& c, const Level& L,
                    gpufhe::DeviceKSContext& C,
                    uint64_t* dr0, uint64_t* dr1, uint64_t* scr, uint64_t* drp) {
    size_t T = (size_t)L.tw * c.n;
    cudaMemcpy(dr0, ct.c0.data(), T * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dr1, ct.c1.data(), T * 8, cudaMemcpyHostToDevice);
    gpufhe::rescale_resident_raw(dr0, L.tw, C, L.rs1, L.rs2, scr, drp, 0);
    gpufhe::rescale_resident_raw(dr1, L.tw, C, L.rs1, L.rs2, scr, drp, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(ct.c0.data(), dr0, T * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(ct.c1.data(), dr1, T * 8, cudaMemcpyDeviceToHost);
    size_t Tn = (size_t)(L.tw - 1) * c.n;
    ct.c0.resize(Tn); ct.c1.resize(Tn);
}

// Device-resident BSGS state: one KS context per rotation amount, plus every
// diagonal already in VRAM. Nothing crosses PCIe inside a layer.
struct DevLayer {
    std::map<uint32_t, gpufhe::DeviceKSContext> ctx;
    gpufhe::DeviceKSWork W{};
    std::vector<uint64_t*> d_ev;          // per k, null if zero diagonal
    uint64_t *d_x0=nullptr, *d_x1=nullptr, *d_a0=nullptr, *d_a1=nullptr;
    uint64_t *d_i0=nullptr, *d_i1=nullptr, *d_b0=nullptr, *d_b1=nullptr;
    uint64_t *d_sc=nullptr;
    std::vector<uint64_t*> d_B0, d_B1;    // baby rotations
    uint32_t tw=0;
};

// Precomputed BSGS diagonals in eval form. Model constants -- encoded ONCE.
struct DiagSet {
    std::vector<std::vector<uint64_t>> ev;   // per k, empty if zero diagonal
};

static DiagSet encode_diags(const std::vector<std::vector<double>>& D,
                            double dscale, Ctx& c, const Level& L) {
    DiagSet ds; ds.ev.resize(SLOTS);
    for (int k = 0; k < SLOTS; k++) {
        bool nz = false;
        for (int i = 0; i < SLOTS; i++) if (D[k][i] != 0.0) { nz = true; break; }
        if (!nz) continue;
        int shift = (k / BS_N1) * BS_N1;      // giant-step shift for this k
        std::vector<double> dp(PER);
        for (int i = 0; i < (int)PER; i++) {
            int t = i - shift; if (t < 0) t += SLOTS;
            dp[i] = D[k][t];
        }
        std::vector<int64_t> md;
        gpufhe::encode_host(md, repl(dp), c.n, dscale);
        gpufhe::pt_to_eval_host(ds.ev[k], md, L.tw, c.n, L.mod, L.root);
    }
    return ds;
}

static void devlayer_init(DevLayer& D, Ctx& c, const Level& L,
                          std::map<uint32_t, gpufhe::KeySwitchConstants>& rk,
                          const DiagSet& ds) {
    D.tw = L.tw;
    size_t T = (size_t)L.tw * c.n, B = T * 8;
    for (auto& kv : rk) D.ctx[kv.first] = gpufhe::ks_context_create(kv.second);
    D.W = gpufhe::ks_work_create(D.ctx.begin()->second);
    D.d_ev.assign(SLOTS, nullptr);
    for (int k = 0; k < SLOTS; k++) {
        if (ds.ev[k].empty()) continue;
        cudaMalloc(&D.d_ev[k], B);
        cudaMemcpy(D.d_ev[k], ds.ev[k].data(), B, cudaMemcpyHostToDevice);
    }
    cudaMalloc(&D.d_x0, B); cudaMalloc(&D.d_x1, B);
    cudaMalloc(&D.d_a0, B); cudaMalloc(&D.d_a1, B);
    cudaMalloc(&D.d_i0, B); cudaMalloc(&D.d_i1, B);
    cudaMalloc(&D.d_b0, B); cudaMalloc(&D.d_b1, B);
    cudaMalloc(&D.d_sc, (size_t)c.n * 8);
    D.d_B0.resize(BS_N1); D.d_B1.resize(BS_N1);
    for (int b = 0; b < BS_N1; b++) {
        cudaMalloc(&D.d_B0[b], B); cudaMalloc(&D.d_B1[b], B);
    }
}

// Fully device-resident BSGS. Host ciphertext in, host ciphertext out; every
// rotation, multiply and accumulation between those points stays in VRAM.
static CtPair bsgs_device(const CtPair& in, const DiagSet& ds, Ctx& c,
                          const Level& L, DevLayer& D) {
    size_t T = (size_t)L.tw * c.n, B = T * 8;
    cudaMemcpy(D.d_x0, in.c0.data(), B, cudaMemcpyHostToDevice);
    cudaMemcpy(D.d_x1, in.c1.data(), B, cudaMemcpyHostToDevice);

    for (int b = 0; b < BS_N1; b++) {
        cudaMemcpy(D.d_B0[b], D.d_x0, B, cudaMemcpyDeviceToDevice);
        cudaMemcpy(D.d_B1[b], D.d_x1, B, cudaMemcpyDeviceToDevice);
        if (b) gpufhe::rotate_ct_device(D.d_B0[b], D.d_B1[b], autok(c.n, b),
                                        D.ctx[b], D.W, L.tw, c.n, L.mod, L.root,
                                        D.d_sc, D.d_b0, D.d_b1, 0);
    }

    cudaMemset(D.d_a0, 0, B); cudaMemset(D.d_a1, 0, B);
    for (int g = 0; g < BS_N2; g++) {
        int shift = g * BS_N1;
        cudaMemset(D.d_i0, 0, B); cudaMemset(D.d_i1, 0, B);
        bool any = false;
        for (int b = 0; b < BS_N1; b++) {
            int k = shift + b;
            if (!D.d_ev[k]) continue;
            for (uint32_t t = 0; t < L.tw; t++) {
                size_t o = (size_t)t * c.n;
                LaunchPtMulAcc(D.d_i0 + o, D.d_i1 + o,
                               D.d_B0[b] + o, D.d_B1[b] + o,
                               D.d_ev[k] + o, L.mod[t], c.n, 0);
            }
            any = true;
        }
        if (!any) continue;
        if (shift) gpufhe::rotate_ct_device(D.d_i0, D.d_i1, autok(c.n, shift),
                                            D.ctx[shift], D.W, L.tw, c.n,
                                            L.mod, L.root, D.d_sc, D.d_b0, D.d_b1, 0);
        for (uint32_t t = 0; t < L.tw; t++) {
            size_t o = (size_t)t * c.n;
            LaunchAddInto(D.d_a0 + o, D.d_i0 + o, L.mod[t], c.n, 0);
            LaunchAddInto(D.d_a1 + o, D.d_i1 + o, L.mod[t], c.n, 0);
        }
    }
    cudaDeviceSynchronize();
    CtPair out; out.c0.resize(T); out.c1.resize(T);
    cudaMemcpy(out.c0.data(), D.d_a0, B, cudaMemcpyDeviceToHost);
    cudaMemcpy(out.c1.data(), D.d_a1, B, cudaMemcpyDeviceToHost);
    return out;
}

// y = sum_k diag_k (x) rot_k(x), BSGS factored.
static CtPair bsgs(const CtPair& in, const DiagSet& D,
                   Ctx& c, const Level& L,
                   std::map<uint32_t, gpufhe::KeySwitchConstants>& rk) {
    size_t T = (size_t)L.tw * c.n;
    std::vector<CtPair> B(BS_N1);
    B[0] = in;
    for (int b = 1; b < BS_N1; b++) {
        B[b] = in;
        rot(B[b].c0, B[b].c1, autok(c.n, b), rk[b], L.tw, c.n, L.mod, L.root);
    }
    CtPair acc; acc.c0.assign(T, 0); acc.c1.assign(T, 0);
    bool got = false;
    for (int g = 0; g < BS_N2; g++) {
        int shift = g * BS_N1;
        CtPair inner; bool any = false;
        for (int b = 0; b < BS_N1; b++) {
            int k = shift + b;
            if (D.ev[k].empty()) continue;
            CtPair p = B[b];
            gpufhe::ct_mul_pt_host(p.c0, p.c1, D.ev[k], L.tw, c.n, L.mod);
            if (!any) { inner = p; any = true; }
            else gpufhe::ct_add_ct_host(inner.c0, inner.c1, p.c0, p.c1, L.tw, c.n, L.mod);
        }
        if (!any) continue;
        if (shift) rot(inner.c0, inner.c1, autok(c.n, shift), rk[shift],
                       L.tw, c.n, L.mod, L.root);
        if (!got) { acc = inner; got = true; }
        else gpufhe::ct_add_ct_host(acc.c0, acc.c1, inner.c0, inner.c1, L.tw, c.n, L.mod);
    }
    return acc;
}

// Square: tensor + relinearise. Scale becomes s^2; caller rescales.
static CtPair square_ct(const CtPair& a, Ctx& c, const Level& L,
                        const gpufhe::KeySwitchConstants& Krl) {
    size_t T = (size_t)L.tw * c.n;
    std::vector<uint64_t> t0(T), t1(T), t2(T);
    for (uint32_t t = 0; t < L.tw; t++) {
        uint64_t q = L.mod[t];
        for (uint32_t k = 0; k < c.n; k++) {
            size_t x = (size_t)t * c.n + k;
            t0[x] = mmul(a.c0[x], a.c0[x], q);
            t1[x] = madd(mmul(a.c0[x], a.c1[x], q), mmul(a.c1[x], a.c0[x], q), q);
            t2[x] = mmul(a.c1[x], a.c1[x], q);
        }
    }
    auto R = gpufhe::keyswitch_core_resident(t2, Krl);
    CtPair r; r.c0.resize(T); r.c1.resize(T);
    for (uint32_t t = 0; t < L.tw; t++) {
        uint64_t q = L.mod[t];
        for (uint32_t k = 0; k < c.n; k++) {
            size_t x = (size_t)t * c.n + k;
            r.c0[x] = madd(t0[x], R.ba0[x], q);
            r.c1[x] = madd(t1[x], R.ba1[x], q);
        }
    }
    return r;
}

static std::vector<double> decrypt_slots(const CtPair& ct, Ctx& c,
                                         const Level& L, double scale) {
    auto KPr = gpufhe::keygen_host(c.n, L.mod, L.root, NS, SIGMA, 101);
    std::vector<int64_t> dec;
    gpufhe::decrypt_host(dec, ct.c0, ct.c1, KPr.s, c.n, L.mod, L.root);
    std::vector<std::complex<double>> y;
    gpufhe::decode_host(y, dec, c.n, scale);
    std::vector<double> o(PER);
    for (uint32_t i = 0; i < PER; i++) o[i] = y[i].real();
    return o;
}

