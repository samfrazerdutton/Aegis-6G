#pragma once
#include <cstdint>
#include <string>
#include <vector>

// MNIST loader + the preprocessing the encrypted model requires.
namespace aegis {

constexpr int IMG_DIM  = 28;
constexpr int POOL_DIM = 14;
constexpr int FEATURES = POOL_DIM * POOL_DIM;   // 196
constexpr int SLOTS    = 256;                   // padded, power of two for BSGS
constexpr int CLASSES  = 10;

struct Dataset {
    std::vector<std::vector<float>> x;   // FEATURES each, in [0,1]
    std::vector<int>                y;
    size_t size() const { return y.size(); }
};

// 2x2 average pool 28x28 -> 14x14, scaled to [0,1]. Applied identically at
// train time and inference time; the encrypted path sees the SAME vector.
bool load_mnist(Dataset& d, const std::string& img_path,
                const std::string& lbl_path, size_t limit = 0);

} // namespace aegis
