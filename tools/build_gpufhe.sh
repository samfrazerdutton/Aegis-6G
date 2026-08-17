#!/usr/bin/env bash
# Build ONLY the gpufhe core library, in its own build tree.
#
# Not add_subdirectory: GPU-Resident-Library calls enable_testing() and
# registers 56 OpenFHE-linked test targets, which would land in Aegis-6G's
# ctest and break a dependency-free verification run.
#
# PATH is stripped of /mnt/* first. Under WSL the Windows PATH is inherited,
# and CMake's find_library probes it over the 9p filesystem -- FindCUDAToolkit
# makes hundreds of probes, turning a 4-second configure into minutes.
set -euo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '^/mnt/' | paste -sd:)"
cmake -S "$R/third_party/gpufhe" -B "$R/build-gpufhe" \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DCMAKE_DISABLE_FIND_PACKAGE_OpenFHE=ON
cmake --build "$R/build-gpufhe" -j4 --target gpufhe
ls -l "$R/build-gpufhe/libgpufhe.a"
