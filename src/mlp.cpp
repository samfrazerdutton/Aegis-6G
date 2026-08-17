#include "mlp.h"
#include <cstdio>
#include <cmath>

namespace aegis {

bool mlp_load(MLP& m, const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return false;
    if (std::fscanf(f, "%d %d %d", &m.features, &m.hidden, &m.classes) != 3) {
        std::fclose(f); return false;
    }
    if (std::fscanf(f, "%f %f", &m.max_z1, &m.train_acc) != 2) {
        std::fclose(f); return false;
    }
    auto rd = [&](std::vector<float>& v, size_t n) {
        v.resize(n);
        for (size_t i = 0; i < n; i++)
            if (std::fscanf(f, "%f", &v[i]) != 1) return false;
        return true;
    };
    bool ok = rd(m.W1, (size_t)m.hidden * m.features) &&
              rd(m.b1, (size_t)m.hidden) &&
              rd(m.W2, (size_t)m.classes * m.hidden) &&
              rd(m.b2, (size_t)m.classes);
    std::fclose(f);
    return ok;
}

std::vector<double> pack_input(const std::vector<float>& x) {
    std::vector<double> v(SLOTS, 0.0);
    for (size_t i = 0; i < x.size() && i < SLOTS; i++) v[i] = x[i];
    return v;
}

std::vector<double> build_matrix(const std::vector<float>& W,
                                 int rows, int cols) {
    std::vector<double> M((size_t)SLOTS * SLOTS, 0.0);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            M[(size_t)r * SLOTS + c] = W[(size_t)r * cols + c];
    return M;
}

std::vector<std::vector<double>> build_diagonals(const std::vector<double>& M) {
    std::vector<std::vector<double>> d(SLOTS, std::vector<double>(SLOTS, 0.0));
    for (int k = 0; k < SLOTS; k++)
        for (int i = 0; i < SLOTS; i++)
            d[k][i] = M[(size_t)i * SLOTS + ((i + k) % SLOTS)];
    return d;
}

std::vector<double> rot_left(const std::vector<double>& v, int k) {
    std::vector<double> o(SLOTS);
    for (int i = 0; i < SLOTS; i++) o[i] = v[(i + k) % SLOTS];
    return o;
}

// Dense diagonal matvec: y = sum_k diag_k * rot_k(x)
static std::vector<double> matvec_diag(
        const std::vector<std::vector<double>>& diags,
        const std::vector<double>& x) {
    std::vector<double> y(SLOTS, 0.0);
    for (int k = 0; k < SLOTS; k++) {
        bool nz = false;
        for (int i = 0; i < SLOTS; i++) if (diags[k][i] != 0.0) { nz = true; break; }
        if (!nz) continue;
        std::vector<double> r = rot_left(x, k);
        for (int i = 0; i < SLOTS; i++) y[i] += diags[k][i] * r[i];
    }
    return y;
}

// BSGS: k = g*N1 + b
//   sum_k diag_k * rot_k(x)
//     = sum_g rot_{g*N1}( sum_b diag'_{g,b} * rot_b(x) )
// with diag'_{g,b}[i] = diag_{g*N1+b}[(i - g*N1) mod SLOTS]   (MINUS)
// Rotations needed: N1 baby steps + N2 giant steps instead of SLOTS.
static std::vector<double> matvec_bsgs(
        const std::vector<std::vector<double>>& diags,
        const std::vector<double>& x) {
    std::vector<std::vector<double>> baby(BS_N1);
    for (int b = 0; b < BS_N1; b++) baby[b] = rot_left(x, b);

    std::vector<double> y(SLOTS, 0.0);
    for (int g = 0; g < BS_N2; g++) {
        std::vector<double> inner(SLOTS, 0.0);
        int shift = g * BS_N1;
        for (int b = 0; b < BS_N1; b++) {
            const std::vector<double>& dk = diags[shift + b];
            for (int i = 0; i < SLOTS; i++)
                { int t = i - shift; if (t < 0) t += SLOTS;
                  inner[i] += dk[t] * baby[b][i]; }
        }
        std::vector<double> r = rot_left(inner, shift);
        for (int i = 0; i < SLOTS; i++) y[i] += r[i];
    }
    return y;
}

static int argmax10(const std::vector<double>& v) {
    int a = 0;
    for (int c = 1; c < CLASSES; c++) if (v[c] > v[a]) a = c;
    return a;
}

int forward_direct(const MLP& m, const std::vector<float>& x,
                   std::vector<double>* logits) {
    std::vector<double> a1(m.hidden), z2(m.classes);
    for (int h = 0; h < m.hidden; h++) {
        double s = m.b1[h];
        for (int i = 0; i < m.features; i++)
            s += (double)m.W1[(size_t)h * m.features + i] * x[i];
        a1[h] = s * s;
    }
    for (int c = 0; c < m.classes; c++) {
        double s = m.b2[c];
        for (int h = 0; h < m.hidden; h++)
            s += (double)m.W2[(size_t)c * m.hidden + h] * a1[h];
        z2[c] = s;
    }
    if (logits) *logits = z2;
    return argmax10(z2);
}

// Shared packed pipeline; matvec is the only thing that differs.
template <typename F>
static int forward_packed(const MLP& m, const std::vector<float>& x,
                          std::vector<double>* logits, F matvec) {
    static std::vector<std::vector<double>> D1, D2;
    static bool built = false;
    if (!built) {
        D1 = build_diagonals(build_matrix(m.W1, m.hidden, m.features));
        D2 = build_diagonals(build_matrix(m.W2, m.classes, m.hidden));
        built = true;
    }
    std::vector<double> v = pack_input(x);
    v = matvec(D1, v);
    for (int h = 0; h < m.hidden; h++) v[h] += m.b1[h];
    for (int i = 0; i < SLOTS; i++) v[i] = v[i] * v[i];   // square activation
    // Slots >= hidden hold squared garbage from padded rows; the second
    // matrix has zero columns there, so they cannot contribute. Zeroing is
    // NOT required and would cost a homomorphic mask.
    v = matvec(D2, v);
    std::vector<double> z2(CLASSES);
    for (int c = 0; c < CLASSES; c++) z2[c] = v[c] + m.b2[c];
    if (logits) *logits = z2;
    return argmax10(z2);
}

int forward_diag(const MLP& m, const std::vector<float>& x,
                 std::vector<double>* logits) {
    return forward_packed(m, x, logits, matvec_diag);
}
int forward_bsgs(const MLP& m, const std::vector<float>& x,
                 std::vector<double>* logits) {
    return forward_packed(m, x, logits, matvec_bsgs);
}

} // namespace aegis
