# Aegis-6G

A from-scratch C++/CUDA framework for privacy-preserving inference on edge
devices: post-quantum key agreement, authenticated encrypted transport, and
homomorphic neural network evaluation on a consumer GPU (RTX 2060, 6 GB).

Zero runtime dependencies. Everything below is implemented directly against
its specification and gated against external test vectors where they exist.

## Status

| Component | State |
|---|---|
| Post-quantum transport (ML-KEM-768, FIPS 203) | **Done — all NIST ACVP vectors pass** |
| AEAD + session + TCP transport | **Done — RFC 8439 vectors + 18 protocol gates** |
| Network cost characterisation | **Done — see `docs/findings.md`** |
| GPU CKKS engine | Provided by [GPU-Resident-Library](https://github.com/samfrazerdutton/GPU-Resident-Library) (submodule) |
| Encrypted MNIST inference | **Done — 4.8 s/image at 163-bit security, matches plaintext decisions** |

## Headline results

**Encrypted MNIST inference at 163-bit classical / 151-bit quantum security
in 4.8 s/image on an RTX 2060 Max-Q (6 GB).** A 196->64->10 square-activation
MLP (97.15% plaintext accuracy) evaluated entirely under CKKS -- no
bootstrapping, no decryption mid-circuit -- reproducing the plaintext model's
predictions exactly. Security estimated with a primal-uSVP tool re-validated
against the published 128-bit table.


Benchmarking the transport layer against measured FHE compute costs
**falsifies the premise that link bandwidth is the enabling factor** for
encrypted multi-agent inference:

- FHE compute dominates transport by 3+ orders of magnitude
  (bootstrap: 23.3 s at n=1024; a 15 MB ciphertext at 10 Gb/s: 12.6 ms).
- Where transport costs anything, the cost is **local symmetric crypto, not
  the network**: 92% of a measured loopback round-trip is AEAD.
- Consequence: send post-rescale ciphertexts (2 MB), never full-ladder
  (15 MB) — 7.5x on both axes simultaneously.

Full numbers and method in `docs/findings.md`.

## Security status

**This is research code. It is not deployable.** Read `docs/security.md`
before drawing any conclusion from the word "quantum-resistant":

- ML-KEM-768 is functionally conformant to FIPS 203 but hand-written,
  unaudited, and **not constant-time** (the ring arithmetic uses `%` on
  secret data). ChaCha20-Poly1305 *is* constant-time by construction.
- Key agreement is **unauthenticated**. It assumes peer public keys are
  pinned out of band. Peer authentication needs ML-DSA or a PKI; neither
  is implemented.
- The CKKS parameters used in tests are sandbox parameters at roughly
  44-bit security. A 128-bit-secure configuration needs a far larger ring.

## Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
Optional data fetches (not needed for `ctest`, which ships its vectors):
./tools/fetch_kat.sh && ./tools/extract_kat.py # NIST ACVP ML-KEM vectors
./tools/fetch_mnist.sh # MNIST
./build/train_mlp 20 0.03 # trains to ~97.1%
## Testing philosophy

Every layer is gated against an **external** oracle wherever one exists —
NIST ACVP for ML-KEM, RFC 8439 for the AEAD, FIPS 202 for Keccak, and a
schoolbook implementation for the NTT. Self-consistency tests are not
evidence of correctness: an implementation with a matrix transposed in both
directions round-trips perfectly and interoperates with nothing.

Two bugs found this way are documented in the commit history: a BSGS
giant-step sign error that produced exactly-random 10-class accuracy, and
the compression-bound check that underpins ML-KEM's decryption-failure rate.

## Licence

MIT.
