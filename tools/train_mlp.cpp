// Train 196 -> 64 -> 10 with SQUARE activation, for homomorphic evaluation.
// No external libraries. Reports the pre-activation range, which is the
// number that decides whether the encrypted version is feasible at all.
#include "mnist.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

using namespace aegis;

constexpr int H = 64;

struct Model {
    std::vector<float> W1, b1, W2, b2;   // W1 H x FEATURES, W2 CLASSES x H
    Model() : W1(H * FEATURES), b1(H, 0.f),
              W2(CLASSES * H), b2(CLASSES, 0.f) {}
};

// Forward. z1 = W1 x + b1 ; a1 = z1^2 ; z2 = W2 a1 + b2
static void forward(const Model& m, const float* x,
                    float* z1, float* a1, float* z2) {
    for (int h = 0; h < H; h++) {
        float s = m.b1[h];
        const float* w = &m.W1[(size_t)h * FEATURES];
        for (int i = 0; i < FEATURES; i++) s += w[i] * x[i];
        z1[h] = s;
        a1[h] = s * s;
    }
    for (int c = 0; c < CLASSES; c++) {
        float s = m.b2[c];
        const float* w = &m.W2[(size_t)c * H];
        for (int h = 0; h < H; h++) s += w[h] * a1[h];
        z2[c] = s;
    }
}

static float eval_acc(const Model& m, const Dataset& d, float* max_abs_z1) {
    std::vector<float> z1(H), a1(H), z2(CLASSES);
    size_t correct = 0;
    float mx = 0.f;
    for (size_t i = 0; i < d.size(); i++) {
        forward(m, d.x[i].data(), z1.data(), a1.data(), z2.data());
        for (int h = 0; h < H; h++) mx = std::max(mx, std::fabs(z1[h]));
        int arg = 0;
        for (int c = 1; c < CLASSES; c++) if (z2[c] > z2[arg]) arg = c;
        if (arg == d.y[i]) correct++;
    }
    if (max_abs_z1) *max_abs_z1 = mx;
    return (float)correct / (float)d.size();
}

int main(int argc, char** argv) {
    int epochs = (argc > 1) ? std::atoi(argv[1]) : 20;
    float lr    = (argc > 2) ? (float)std::atof(argv[2]) : 0.03f;

    Dataset tr, te;
    if (!load_mnist(tr, "data/train-images-idx3-ubyte",
                    "data/train-labels-idx1-ubyte") ||
        !load_mnist(te, "data/t10k-images-idx3-ubyte",
                    "data/t10k-labels-idx1-ubyte")) {
        std::printf("MNIST not found -- run tools/fetch_mnist.sh\n");
        return 2;
    }
    std::printf("train %zu  test %zu  features %d  hidden %d\n\n",
                tr.size(), te.size(), FEATURES, H);

    std::mt19937 rng(42);
    Model m;
    // Deliberately small init: with a square activation, standard He init
    // makes z1^2 explode and training diverges in the first epoch.
    std::normal_distribution<float> n1(0.f, 0.5f / std::sqrt((float)FEATURES));
    std::normal_distribution<float> n2(0.f, 1.0f / std::sqrt((float)H));
    for (auto& v : m.W1) v = n1(rng);
    for (auto& v : m.W2) v = n2(rng);

    Model g, vel;
    for (auto* p : {&vel.W1, &vel.b1, &vel.W2, &vel.b2})
        std::fill(p->begin(), p->end(), 0.f);

    const int B = 64;
    const float mom = 0.9f;
    std::vector<size_t> idx(tr.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::vector<float> z1(H), a1(H), z2(CLASSES), p(CLASSES);
    std::vector<float> dz1(H), da1(H), dz2(CLASSES);

    for (int ep = 1; ep <= epochs; ep++) {
        std::shuffle(idx.begin(), idx.end(), rng);
        double loss = 0;
        for (size_t off = 0; off + B <= tr.size(); off += B) {
            std::fill(g.W1.begin(), g.W1.end(), 0.f);
            std::fill(g.b1.begin(), g.b1.end(), 0.f);
            std::fill(g.W2.begin(), g.W2.end(), 0.f);
            std::fill(g.b2.begin(), g.b2.end(), 0.f);

            for (int b = 0; b < B; b++) {
                size_t s = idx[off + b];
                const float* x = tr.x[s].data();
                forward(m, x, z1.data(), a1.data(), z2.data());

                float mxz = z2[0];
                for (int c = 1; c < CLASSES; c++) mxz = std::max(mxz, z2[c]);
                float sum = 0;
                for (int c = 0; c < CLASSES; c++) {
                    p[c] = std::exp(z2[c] - mxz);
                    sum += p[c];
                }
                for (int c = 0; c < CLASSES; c++) p[c] /= sum;
                loss -= std::log(std::max(p[tr.y[s]], 1e-12f));

                for (int c = 0; c < CLASSES; c++)
                    dz2[c] = p[c] - (c == tr.y[s] ? 1.f : 0.f);
                for (int h = 0; h < H; h++) da1[h] = 0.f;
                for (int c = 0; c < CLASSES; c++) {
                    g.b2[c] += dz2[c];
                    for (int h = 0; h < H; h++) {
                        g.W2[(size_t)c * H + h] += dz2[c] * a1[h];
                        da1[h] += dz2[c] * m.W2[(size_t)c * H + h];
                    }
                }
                for (int h = 0; h < H; h++) dz1[h] = da1[h] * 2.f * z1[h];
                for (int h = 0; h < H; h++) {
                    g.b1[h] += dz1[h];
                    float* gw = &g.W1[(size_t)h * FEATURES];
                    for (int i = 0; i < FEATURES; i++) gw[i] += dz1[h] * x[i];
                }
            }

            const float s = lr / (float)B;
            auto step = [&](std::vector<float>& w, std::vector<float>& gv,
                            std::vector<float>& v) {
                for (size_t i = 0; i < w.size(); i++) {
                    v[i] = mom * v[i] - s * gv[i];
                    w[i] += v[i];
                }
            };
            step(m.W1, g.W1, vel.W1); step(m.b1, g.b1, vel.b1);
            step(m.W2, g.W2, vel.W2); step(m.b2, g.b2, vel.b2);
        }
        float mz;
        float acc = eval_acc(m, te, &mz);
        std::printf("epoch %2d  loss %7.4f  test acc %6.2f%%  max|z1| %6.3f\n",
                    ep, loss / (tr.size() / B * B), acc * 100.f, mz);
    }

    float mz;
    float acc = eval_acc(m, te, &mz);
    std::printf("\nFINAL test acc %.2f%%   max|z1| %.4f   max|a1| %.4f\n",
                acc * 100.f, mz, mz * mz);
    std::printf("These ranges set the CKKS scale budget for the encrypted "
                "evaluation.\n");

    std::FILE* f = std::fopen("models/mlp.txt", "w");
    std::fprintf(f, "%d %d %d\n", FEATURES, H, CLASSES);
    std::fprintf(f, "%.8f %.8f\n", mz, acc);
    for (float v : m.W1) std::fprintf(f, "%.8f\n", v);
    for (float v : m.b1) std::fprintf(f, "%.8f\n", v);
    for (float v : m.W2) std::fprintf(f, "%.8f\n", v);
    for (float v : m.b2) std::fprintf(f, "%.8f\n", v);
    std::fclose(f);
    std::printf("wrote models/mlp.txt\n");
    return 0;
}
