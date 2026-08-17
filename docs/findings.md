# Measured findings

## F1 — At 6G-class link rates the network is not the constraint; the
##      encryption path is.

Measured on RTX 2060 laptop / WSL2, ChaCha20-Poly1305 scalar, loopback TCP.
Link times are COMPUTED (bytes/rate), not measured -- loopback has no
propagation delay or bandwidth ceiling.

CKKS ciphertext = 2 * towers * n * 8 bytes.

| shape                  | size    | seal   | link@10Gb/s | EvalMult |
|------------------------|---------|--------|-------------|----------|
| n=32768 tw=4  (rescaled)| 2.00 MB | 4.60ms | 1.68 ms     | ~4.2 ms  |
| n=32768 tw=30 (full)   | 15.00 MB| 36.1ms | 12.58 ms    | ~4.2 ms  |

AEAD throughput is flat at ~0.46 GB/s across all sizes.
Loopback echo RTT at 15 MB is 157.7 ms, of which 4 AEAD passes account for
144.6 ms (92%); the ~13 ms residual is 60 MB of socket copies (~4.6 GB/s).

Therefore the loopback RTT measures the AEAD, not the transport.
Against bootstrapping (23.3 s at n=1024, 244 s at n=8192) all transport
costs are three orders of magnitude below the noise floor.

### Consequences
1. Send POST-RESCALE ciphertexts at low tower counts (2 MB), never
   full-ladder (15 MB): 7.5x on AEAD and link simultaneously.
2. Where the protocol allows, move plaintext-sized queries rather than
   ciphertexts at all.
3. The one transport optimization with a measurement behind it is the AEAD
   itself (~0.46 GB/s scalar; SIMD implementations reach several GB/s).
   NOT the network layer.

### What this falsifies
The premise that a high-bandwidth 6G link is the enabling factor for
encrypted multi-agent inference. It is not. FHE compute dominates by
3+ orders of magnitude, and where transport does cost anything, the cost
is local symmetric crypto rather than the link.

## F2 — Encrypted inference at 128-bit-plus security is practical on a
##      6 GB consumer GPU.

Model: 196 -> 64 -> 10 MLP, square activation, 97.15% plaintext test accuracy.
Circuit: BSGS matvec -> square (tensor+relin) -> BSGS matvec. 3 levels, no
bootstrapping. Hardware: RTX 2060 Max-Q (6 GB), WSL2.

Security from tools/lwe_security.py (primal-uSVP, BDGL16), re-validated
in-session against the HES 2018 128-bit table. log2(QP) = 352
(sizeQ=5 x 50-bit, sizeP=2 x 51-bit), uniform ternary secret.

| ring   | classical | quantum | per image | worst logit err |
|--------|-----------|---------|-----------|-----------------|
| 1024   |  44.3     |  43.0   |  0.12 s   | 1.5e-08         |
| 16384  | 163.1     | 151.2   |  2.46 s   | 4.8e-07         |

n=8192 gives only 77.7 bits, so n=16384 is the SMALLEST secure ring here --
2.46 s is the cost of security, not a conservative choice.
Argmax agreement with the plaintext model: 3/3 at n=16384, 20/20 at n=1024.
One-time per model: keys 4.0 s, diagonal encoding 223 s.

### The optimisation that mattered
Per-image cost at n=16384 was initially 240 s -- 160x the n=1024 time for a
16x ring. Superlinear scaling localised it immediately: encode_host is O(N^2)
and the BSGS re-encoded all 256 plaintext diagonals per image. Diagonals are
model constants, so encoding belongs in setup. Hoisting gave 50x with
BIT-IDENTICAL output (4.849e-07 before and after), which is what distinguishes
a correctness-preserving speedup from a different computation that happens
to pass.

### Second optimisation: resident rotation (done)
Switching rotations to the library's device-resident keyswitch gave 1.30x
(4.8 -> 3.65 s/image), again bit-identical. Less than the 1.58x measured at
n=8192/tw=30, because rotate_ct_resident still builds a DeviceKSContext on
every call: at tw=5 the kernel work it saves shrank while that fixed setup
did not. Cumulative: 240 -> 3.65 s/image = 65x.

### Third optimisation: device-resident BSGS (done)
Caching one DeviceKSContext per rotation amount and keeping every diagonal
in VRAM lets a whole BSGS layer run without crossing PCIe: 3.65 -> 2.46
s/image (1.48x), bit-identical. At n=1024 the gain is 2.5x, since per-call
setup was a larger share of a smaller rotation.

**Cumulative: 240 -> 2.46 s/image = 98x, every step bit-identical.**

VRAM: 1569 MB of 6144 at n=16384 -- but 1071 MB of that is already present
at n=1024, so ~1 GB is fixed per-context overhead independent of ring size.
The binding constraint on context caching is the NUMBER of rotation keys,
not the ring dimension.

### Remaining levers (measured, not speculative)
1. encode_host is O(N^2). The one-time diagonal encode is 223 s, now ~96%
   of a 3-image run's wall time. An FFT encode would cut it to seconds.
   Per-model, not per-image -- so it changes the setup story, not latency.
2. encode_host is O(N^2); an FFT encode would cut the 223 s setup to seconds.
3. Batching 32 images across 8192 slots would amortise per-image cost, but
   naive block packing breaks under cyclic rotation -- needs masking or
   block-aligned rotations, which cost a level.

### Contrast with bootstrapping
Bootstrapping needs log2(QP) ~ 2050, forcing n=131072 and far more than 6 GB.
A 3-level inference circuit needs 352 bits and fits comfortably. Shallow
encrypted inference is deployable on consumer hardware today; bootstrapped
FHE is not.

## F3 — Ciphertext transport is free; KEY provisioning is not.

F1 found link time irrelevant next to FHE compute. Closing the loop into a
client/server system inverts that for one specific step.

The client must hold the CKKS secret key, so the server needs evaluation
keys: 60 rotation keys plus one relinearisation key.

| ring   | per rotation key | 60 keys | AEAD seal time @0.46 GB/s |
|--------|------------------|---------|---------------------------|
| 1024   | 0.33 MB          |  20 MB  | ~43 s                     |
| 16384  | 5.3 MB           | 316 MB  | ~11 min                   |

Against a 2 MB inference ciphertext (4.6 ms to seal), provisioning is four
orders of magnitude larger. So:

1. Provision once per client/server pair and keep the session alive. Do NOT
   re-handshake per inference.
2. The AEAD throughput lever (F1, item 3) is worth ~11 minutes of wall clock
   here, not the milliseconds it was worth for ciphertexts.

### Parsing untrusted key material
The server deserialises evaluation keys supplied by a client it does not
trust. Every read is bounds-checked; declared lengths are checked against
server-side limits before any allocation.

A corruption test aimed at the wrong byte offset found a genuine
out-of-bounds read: the `fullQ` field was unvalidated, and the eval-key index
`(i >= sizeQl) ? i + delta : i` with `delta = fullQ - sizeQl` reads past the
end of the key for inflated values. Now bounded by
`sizeQl <= fullQ <= maxtowers` and `evalKeyTowers >= fullQ + sizeP`.

Gate (tests/test_keyser.cpp, 17 checks): exact field-for-field round trip,
rejection at every truncation length, header-corruption rejection, header/
array cross-checks, and 400 randomly corrupted blobs parsed without crashing.
Corrupted-value blobs may deserialise successfully -- that is correct, since
integrity is the AEAD tag's job, not the parser's.
