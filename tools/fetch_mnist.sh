#!/usr/bin/env bash
# Fetch MNIST. Gunzip is a system tool, not a library dependency.
set -euo pipefail
B="https://raw.githubusercontent.com/fgnt/mnist/master"
D="$(dirname "$0")/../data"
mkdir -p "$D"
for f in train-images-idx3-ubyte train-labels-idx1-ubyte \
         t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  if [ ! -f "$D/$f" ]; then
    curl -fsSL -o "$D/$f.gz" "$B/$f.gz"
    gunzip -f "$D/$f.gz"
  fi
done
ls -l "$D"
