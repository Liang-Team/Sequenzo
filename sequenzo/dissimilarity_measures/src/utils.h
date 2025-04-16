#ifndef CC_CODE_UTILS_H
#define CC_CODE_UTILS_H

#include <cmath>

static inline double normalize_distance(double rawdist, double maxdist, double l1, double l2, int norm){
    if (rawdist == 0.0) return 0.0;
    switch (norm) {
        case 0:
            return rawdist;
        case 1:
            return l1 > l2 ? rawdist / l1 : l2 > 0.0 ? rawdist / l2 : 0.0;
        case 2:
            return (l1 * l2 == 0.0) ? (l1 != l2 ? 1.0 : 0.0)
                                    : 1.0 - ((maxdist - rawdist) / (2.0 * std::sqrt(l1) * std::sqrt(l2)));
        case 3:
            return maxdist == 0.0 ? 1.0 : rawdist / maxdist;
        case 4:
            return maxdist == 0.0 ? 1.0 : (2.0 * rawdist) / (rawdist + maxdist);
        default:
            return rawdist;
    }
}

#endif //CC_CODE_UTILS_H
