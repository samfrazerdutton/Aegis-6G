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
