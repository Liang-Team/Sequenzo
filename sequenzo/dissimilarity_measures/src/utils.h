#ifndef CC_CODE_UTILS_H
#define CC_CODE_UTILS_H

#include <cmath>

static constexpr double EPS = 1e-10;

static inline double normalize_distance(double rawdist, double maxdist, double l1, double l2, int norm){
    if (std::fabs(rawdist) < EPS) return 0.0;
    switch (norm) {
        case 0:
            return rawdist;
        case 1:
            return l1 > l2 ? rawdist / l1 : l2 > 0.0 ? rawdist / l2 : 0.0;
        case 2:
            /* gmean: when l1*l2==0, match TraMineR: l1!=l2 => 1 (one zero), l1==l2 => 0 (both zero) */
            if (std::fabs(l1 * l2) < EPS) {
                return std::fabs(l1 - l2) < EPS ? 0.0 : 1.0;
            }
            return 1.0 - ((maxdist - rawdist) / (2.0 * std::sqrt(l1) * std::sqrt(l2)));
        case 3:
            return std::fabs(maxdist) < EPS ? 1.0 : rawdist / maxdist;
        case 4:
            return std::fabs(maxdist) < EPS ? 1.0 : (2.0 * rawdist) / (rawdist + maxdist);
        default:
            return rawdist;
    }
}

#endif //CC_CODE_UTILS_H
