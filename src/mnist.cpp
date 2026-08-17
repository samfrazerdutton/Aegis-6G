#include "mnist.h"
#include <cstdio>
#include <cstring>

namespace aegis {

static bool rd_u32(std::FILE* f, uint32_t& v) {
    uint8_t b[4];
    if (std::fread(b, 1, 4, f) != 4) return false;
    v = ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) |
        ((uint32_t)b[2] << 8) | (uint32_t)b[3];      // IDX is big-endian
    return true;
}

bool load_mnist(Dataset& d, const std::string& img_path,
                const std::string& lbl_path, size_t limit) {
    std::FILE* fi = std::fopen(img_path.c_str(), "rb");
    std::FILE* fl = std::fopen(lbl_path.c_str(), "rb");
    if (!fi || !fl) {
        if (fi) std::fclose(fi);
        if (fl) std::fclose(fl);
        return false;
    }
    uint32_t mi, ni, rows, cols, ml, nl;
    bool ok = rd_u32(fi, mi) && rd_u32(fi, ni) && rd_u32(fi, rows) &&
              rd_u32(fi, cols) && rd_u32(fl, ml) && rd_u32(fl, nl);
    if (!ok || mi != 0x803 || ml != 0x801 || ni != nl ||
        rows != IMG_DIM || cols != IMG_DIM) {
        std::fclose(fi); std::fclose(fl);
        return false;
    }
    size_t n = ni;
    if (limit && limit < n) n = limit;

    d.x.assign(n, std::vector<float>(FEATURES, 0.f));
    d.y.assign(n, 0);

    std::vector<uint8_t> raw(IMG_DIM * IMG_DIM);
    for (size_t i = 0; i < n; i++) {
        if (std::fread(raw.data(), 1, raw.size(), fi) != raw.size()) return false;
        int lb = std::fgetc(fl);
        if (lb < 0) return false;
        d.y[i] = lb;
        for (int r = 0; r < POOL_DIM; r++)
            for (int c = 0; c < POOL_DIM; c++) {
                int s = raw[(2*r) * IMG_DIM + (2*c)] +
                        raw[(2*r) * IMG_DIM + (2*c+1)] +
                        raw[(2*r+1) * IMG_DIM + (2*c)] +
                        raw[(2*r+1) * IMG_DIM + (2*c+1)];
                d.x[i][r * POOL_DIM + c] = (float)s / (4.0f * 255.0f);
            }
    }
    std::fclose(fi); std::fclose(fl);
    return true;
}

} // namespace aegis
