# Security status — READ FIRST

Aegis-6G is a research framework. The parameters shipped in tests are
SANDBOX parameters and are NOT cryptographically secure.

## FHE layer (CKKS)
Estimated classical security at the sandbox ring (n=1024, log2 QP = 2050):
~44 bits. Not deployable. A 128-bit-secure configuration at this modulus
requires n=131072 with hamming weight h=256 (est. 197 bits, depth 21).
See tools/ for the primal-uSVP and hybrid dual+MITM estimators.

## Transport layer (ML-KEM-768)
Hand-written implementation to FIPS 203. Gated against NIST KAT vectors.
NOT independently audited. NOT constant-time-verified. Do not deploy.

## What is claimed
Correctness (bit-exact / bounded-error gates) and throughput on consumer
hardware. NOT deployment-grade security.

## Conformance status (2026-08-15)
Transport layer passes the full NIST ACVP vector set for ML-KEM-768
(FIPS 203): keyGen 25/25 ek + 25/25 dk, encaps 25/25 ct + 25/25 ss,
decaps 10/10 including 5/5 implicit-rejection cases, ek modulus check 10/10.
Source: usnistgov/ACVP-Server, ML-KEM-{keyGen,encapDecap}-FIPS203.

This establishes FUNCTIONAL CONFORMANCE only. It does NOT establish:
  - constant-time execution (mulq uses %, decaps compare is not verified)
  - side-channel resistance of any kind
  - freedom from memory-safety defects under adversarial input
Reproduce: ./tools/fetch_kat.sh && ./tools/extract_kat.py && ./build/test_mlkem_kat
