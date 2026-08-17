#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

struct PamCandidate {
    double score = std::numeric_limits<double>::infinity();
    int h = -1;
    int k_slot = -1;
};

template <typename Callback>
inline void pam_for_each_distance_in_row(
    int n,
    int row,
    const double* full,
    const double* condensed,
    bool use_condensed,
    Callback&& callback) {
    if (!use_condensed) {
        const double* row_data = full + static_cast<std::size_t>(row) * n;
        for (int j = 0; j < n; ++j) callback(j, row_data[j]);
        return;
    }

    std::size_t index = row > 0
        ? static_cast<std::size_t>(row - 1)
        : 0;
    for (int j = 0; j < row; ++j) {
        callback(j, condensed[index]);
        index += static_cast<std::size_t>(n - j - 2);
    }

    callback(row, 0.0);
    index = static_cast<std::size_t>(row) * (2 * n - row - 1) / 2;
    for (int j = row + 1; j < n; ++j) {
        callback(j, condensed[index]);
        ++index;
    }
}

inline bool pam_candidate_better(double score, int h, int k_slot,
                                 const PamCandidate& best) {
    if (score < best.score) return true;
    if (score > best.score || best.h < 0) return false;
    return h < best.h || (h == best.h && k_slot < best.k_slot);
}

inline void pam_consider_candidate(PamCandidate& best, double score,
                                   int h, int k_slot) {
    if (pam_candidate_better(score, h, k_slot, best)) {
        best = {score, h, k_slot};
    }
}

template <typename WeightAccessor, typename DistanceAccessor,
          typename RowVisitor>
inline double pam_classic_swap_score(
    int h,
    int k_slot,
    const int* medoids,
    const int* nearest_slot,
    const double* nearest_distance,
    const double* second_distance,
    const WeightAccessor& weight_at,
    const DistanceAccessor& get_dist,
    const RowVisitor& for_each_distance) {
    const int removed_medoid = medoids[k_slot];
    double score = 0.0;

    for_each_distance(h, [&](int j, double candidate_distance) {
        double contribution = 0.0;

        if (nearest_slot[j] == k_slot ||
            get_dist(removed_medoid, j) == nearest_distance[j]) {
            const double replacement =
                second_distance[j] > candidate_distance
                    ? candidate_distance
                    : second_distance[j];
            contribution =
                weight_at(j) * (-nearest_distance[j] + replacement);
        } else if (candidate_distance < nearest_distance[j]) {
            contribution =
                weight_at(j) * (-nearest_distance[j] + candidate_distance);
        }
        score += contribution;
    });
    return score;
}

inline double pam_upward_add(double left, double right) {
    return std::nextafter(left + right,
                          std::numeric_limits<double>::infinity());
}

inline double pam_upward_multiply(double left, double right) {
    return std::nextafter(left * right,
                          std::numeric_limits<double>::infinity());
}

inline double pam_gamma_bound(std::size_t operations) {
    if (operations == 0) return 0.0;
    constexpr double unit_roundoff =
        std::numeric_limits<double>::epsilon() / 2.0;
    const double product = pam_upward_multiply(
        static_cast<double>(operations), unit_roundoff);
    if (product >= 1.0) {
        return std::numeric_limits<double>::infinity();
    }
    return std::nextafter(product / (1.0 - product),
                          std::numeric_limits<double>::infinity());
}

inline double pam_upper_abs_sum(double rounded_sum, std::size_t operations) {
    const double gamma = pam_gamma_bound(operations);
    if (!std::isfinite(rounded_sum) || gamma >= 1.0) {
        return std::numeric_limits<double>::infinity();
    }
    return std::nextafter(
        rounded_sum / (1.0 - gamma),
        std::numeric_limits<double>::infinity());
}
