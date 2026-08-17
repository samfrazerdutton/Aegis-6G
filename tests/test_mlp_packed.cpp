// The packed/diagonal/BSGS pipeline must reproduce the training-time model
// exactly. This is the oracle the encrypted evaluation will be gated against.
#include "mlp.h"
#include <cmath>
#include <cstdio>

using namespace aegis;

static int fails = 0;
static void check(const char* name, bool ok, const char* detail = nullptr) {
    if (!ok) fails++;
    std::printf("%-44s %s\n", name, ok ? "ok" : "FAIL");
    if (!ok && detail) std::printf("    %s\n", detail);
}

int main() {
    MLP m;
    if (!mlp_load(m, "models/mlp.txt")) {
        std::printf("models/mlp.txt missing -- run train_mlp first\n");
        return 2;
    }
    Dataset te;
    if (!load_mnist(te, "data/t10k-images-idx3-ubyte",
                    "data/t10k-labels-idx1-ubyte")) {
        std::printf("MNIST missing -- run tools/fetch_mnist.sh\n");
        return 2;
    }
    std::printf("model %d->%d->%d  max|z1| %.3f\n\n",
                m.features, m.hidden, m.classes, m.max_z1);

    // Rotation identity, since everything below depends on the direction.
    {
        std::vector<double> v(SLOTS);
        for (int i = 0; i < SLOTS; i++) v[i] = i;
        auto r = rot_left(v, 3);
        check("rot_left(v,3)[0] == v[3]", r[0] == 3.0 && r[SLOTS-1] == 2.0);
    }

    double worst_diag = 0, worst_bsgs = 0;
    int dis_diag = 0, dis_bsgs = 0;
    size_t correct_direct = 0, correct_bsgs = 0;

    std::vector<double> ld, lg, lb;
    for (size_t i = 0; i < te.size(); i++) {
        int a = forward_direct(m, te.x[i], &ld);
        int b = forward_diag(m, te.x[i], &lg);
        int c = forward_bsgs(m, te.x[i], &lb);
        for (int k = 0; k < CLASSES; k++) {
            worst_diag = std::max(worst_diag, std::fabs(ld[k] - lg[k]));
            worst_bsgs = std::max(worst_bsgs, std::fabs(ld[k] - lb[k]));
        }
        if (a != b) dis_diag++;
        if (a != c) dis_bsgs++;
        if (a == te.y[i]) correct_direct++;
        if (c == te.y[i]) correct_bsgs++;
    }

    char det[96];
    std::snprintf(det, sizeof det, "max logit diff %.3e", worst_diag);
    check("dense diagonal == direct (logits)", worst_diag < 1e-9, det);
    std::snprintf(det, sizeof det, "max logit diff %.3e", worst_bsgs);
    check("BSGS == direct (logits)", worst_bsgs < 1e-9, det);

    std::snprintf(det, sizeof det, "%d disagreements", dis_diag);
    check("dense diagonal == direct (argmax)", dis_diag == 0, det);
    std::snprintf(det, sizeof det, "%d disagreements", dis_bsgs);
    check("BSGS == direct (argmax)", dis_bsgs == 0, det);

    double ad = (double)correct_direct / te.size() * 100.0;
    double ab = (double)correct_bsgs / te.size() * 100.0;
    std::snprintf(det, sizeof det, "direct %.2f%% vs bsgs %.2f%%", ad, ab);
    check("packed pipeline preserves accuracy",
          correct_direct == correct_bsgs, det);

    std::printf("\ntest accuracy %.2f%%\n", ad);
    std::printf("rotations: dense %d, BSGS %d (%d baby + %d giant)\n",
                SLOTS, BS_N1 + BS_N2, BS_N1, BS_N2);
    std::printf("\n%s (%d failing)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
