#pragma once
#include "mnist.h"
#include <string>
#include <vector>

// The MNIST model, evaluated three ways that must all agree:
//   forward_direct  -- plain loops, the training-time semantics
//   forward_diag    -- 256-slot packed, Halevi-Shoup diagonals
//   forward_bsgs    -- same, factored baby-step/giant-step
// The encrypted path mirrors forward_bsgs exactly, so any indexing error is
// caught here at microseconds per iteration rather than under encryption.
namespace aegis {

constexpr int HIDDEN = 64;
constexpr int BS_N1  = 16;          // baby steps; SLOTS = BS_N1 * BS_N2
constexpr int BS_N2  = SLOTS / BS_N1;

struct MLP {
    int   features = FEATURES, hidden = HIDDEN, classes = CLASSES;
    float max_z1 = 0.f, train_acc = 0.f;
    std::vector<float> W1, b1, W2, b2;
};

bool mlp_load(MLP& m, const std::string& path);

// Pad a FEATURES-length input into SLOTS slots.
std::vector<double> pack_input(const std::vector<float>& x);

// M is SLOTS x SLOTS, row-major, zero-padded from the real weight matrix.
std::vector<double> build_matrix(const std::vector<float>& W,
                                 int rows, int cols);

// diag_d[i] = M[i][(i+d) mod SLOTS]; pairs with LEFT rotation rot_d(x)[i] =
// x[(i+d) mod SLOTS], which is the direction the CKKS automorphism gives.
std::vector<std::vector<double>> build_diagonals(const std::vector<double>& M);

std::vector<double> rot_left(const std::vector<double>& v, int k);

int forward_direct(const MLP& m, const std::vector<float>& x,
                   std::vector<double>* logits = nullptr);
int forward_diag(const MLP& m, const std::vector<float>& x,
                 std::vector<double>* logits = nullptr);
int forward_bsgs(const MLP& m, const std::vector<float>& x,
                 std::vector<double>* logits = nullptr);

} // namespace aegis
