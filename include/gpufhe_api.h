#pragma once
// Declarations for GPU-Resident-Library functions with no public header.
//
// Defined in the library, declared only locally by its own tests. Keeping them
// here rather than patching the submodule means third_party/gpufhe stays
// exactly as pinned, and the coupling surface is visible in one file.
//
// Copied VERBATIM from the definitions -- do not retype from call sites.
//   encode_host, decode_host                 src/encode.cpp
//   native_primes, native_root,
//   native_rescale_consts                    src/params.cpp
//   pt_to_eval_host, rotate_ct_host,
//   automorphism_eval_host                   src/rotate.cpp
//   ct_add_ct_host, ct_mul_pt_host           src/ptops.cpp
#include "keyswitch.h"          // KeySwitchConstants
#include <complex>
#include <cstdint>
#include <vector>

namespace gpufhe {

void encode_host(std::vector<int64_t>&,
                 const std::vector<std::complex<double>>&, uint32_t, double);
void decode_host(std::vector<std::complex<double>>&,
                 const std::vector<int64_t>&, uint32_t, double);

// out, count, bits, n, avoid -- primes q = 1 mod 2n, descending from 2^bits
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t,
                   const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
// s1, s2, mod, lv
void native_rescale_consts(std::vector<uint64_t>&, std::vector<uint64_t>&,
                           const std::vector<uint64_t>&, uint32_t);

void pt_to_eval_host(std::vector<uint64_t>& out, const std::vector<int64_t>& m,
                     uint32_t towers, uint32_t n,
                     const std::vector<uint64_t>& mod,
                     const std::vector<uint64_t>& root);

// In-place sigma_k on an eval-form poly (INTT -> permute -> NTT).
void automorphism_eval_host(std::vector<uint64_t>& v, uint32_t towers,
                            uint32_t n, uint32_t k,
                            const std::vector<uint64_t>& mod,
                            const std::vector<uint64_t>& root);

void rotate_ct_host(std::vector<uint64_t>& c0, std::vector<uint64_t>& c1,
                    uint32_t k, const KeySwitchConstants& Krot,
                    uint32_t towers, uint32_t n,
                    const std::vector<uint64_t>& mod,
                    const std::vector<uint64_t>& root);

void ct_add_ct_host(std::vector<uint64_t>& c0, std::vector<uint64_t>& c1,
                    const std::vector<uint64_t>& d0,
                    const std::vector<uint64_t>& d1,
                    uint32_t towers, uint32_t n,
                    const std::vector<uint64_t>& mod);

void ct_mul_pt_host(std::vector<uint64_t>& c0, std::vector<uint64_t>& c1,
                    const std::vector<uint64_t>& ptEval,
                    uint32_t towers, uint32_t n,
                    const std::vector<uint64_t>& mod);

} // namespace gpufhe
