#!/usr/bin/env bash
# Fetch NIST ACVP ML-KEM vectors (FIPS 203). Run once; output is gitignored.
set -euo pipefail
BASE="https://raw.githubusercontent.com/usnistgov/ACVP-Server/master/gen-val/json-files"
OUT="$(dirname "$0")/../tests/kat"
mkdir -p "$OUT"
curl -fsSL -o "$OUT/keygen.json"     "$BASE/ML-KEM-keyGen-FIPS203/internalProjection.json"
curl -fsSL -o "$OUT/encapdecap.json" "$BASE/ML-KEM-encapDecap-FIPS203/internalProjection.json"
ls -l "$OUT"/*.json
