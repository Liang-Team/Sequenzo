/*
 * cluster_quality_pipeline.cpp
 *
 * Consolidated pipeline for ClusterQuality:
 *   - Path A: from pre-computed cluster data (condensed + linkage + weights)
 *   - Path B: from raw distance matrix (prep -> linkage -> CQI)
 *
 * All CQI scores, summary (opt k, raw value, z-score), and range table
 * are computed in a single C++ call to minimise Python-C++ round-trips.
 */

#include "cluster_quality_pipeline.h"
#include "cluster_quality.h"
#include "cluster_core.h"
#include "distance_prep_utils.h"
#include "fastcluster_linkage.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

// Metric index order matches Python's metric_order list:
//   PBC, HG, HGSD, ASW, ASWw, CH, R2, CHsq, R2sq, HC
static constexpr int METRIC_IDX[] = {
    ClusterQualHPG,  ClusterQualHG,   ClusterQualHGSD,
    ClusterQualASWi, ClusterQualASWw, ClusterQualF,
    ClusterQualR,    ClusterQualF2,   ClusterQualR2,
    ClusterQualHC
};

// IEEE-754 NaN test that works even under -ffast-math.
inline bool is_nan_portable(double x) {
    uint64_t bits = 0;
    std::memcpy(&bits, &x, sizeof(double));
    const uint64_t exp_mask = 0x7ff0000000000000ULL;
    const uint64_t man_mask = 0x000fffffffffffffULL;
    return ((bits & exp_mask) == exp_mask) && ((bits & man_mask) != 0ULL);
}

// Compute summary (opt_clusters, raw_values, z_scores) from score arrays.
void compute_summary(
    const double* const* score_ptrs,
    int k_count,
    int k_start,
    double* opt_out,
    double* raw_out,
    double* z_out
) {
    const double nan = std::numeric_limits<double>::quiet_NaN();

    for (int m = 0; m < CQ_NUM_METRICS; ++m) {
        const double* arr = score_ptrs[m];

        double sum = 0.0, sumsq = 0.0;
        int cnt = 0;
        double best_val = -std::numeric_limits<double>::infinity();
        int best_idx = -1;

        for (int j = 0; j < k_count; ++j) {
            const double v = arr[j];
            if (is_nan_portable(v)) continue;
            sum += v;
            sumsq += v * v;
            ++cnt;
            if (best_idx < 0 || v > best_val) {
                best_val = v;
                best_idx = j;
            }
        }

        if (best_idx < 0 || cnt == 0) {
            opt_out[m] = nan;
            raw_out[m] = nan;
            z_out[m] = nan;
            continue;
        }

        const double mean = sum / static_cast<double>(cnt);
        const double var = std::max(0.0, (sumsq / static_cast<double>(cnt)) - mean * mean);
        const double stddev = std::sqrt(var);
        const double raw = arr[best_idx];

        opt_out[m] = static_cast<double>(k_start + best_idx);
        raw_out[m] = raw;
        z_out[m] = (stddev > 0.0) ? ((raw - mean) / stddev) : raw;
    }
}

// Build the range table (k_count x CQ_NUM_METRICS, row-major).
void build_range_table(
    const double* const* score_ptrs,
    int k_count,
    double* out
) {
    for (int r = 0; r < k_count; ++r) {
        for (int c = 0; c < CQ_NUM_METRICS; ++c) {
            out[r * CQ_NUM_METRICS + c] = score_ptrs[c][r];
        }
    }
}

// Core: given condensed distances, linkage, weights, compute all CQI
// results and populate the result struct's score/summary/range fields.
void compute_cqi_core(
    const double* condensed,
    const double* linkage,
    const double* weights,
    int n,
    int k_min,
    int k_max,
    ClusterQualityPipelineResult& result
) {
    const int k_count = k_max - k_min + 1;
    result.k_min = k_min;
    result.k_count = k_count;

    // Allocate per-metric score arrays.
    result.pbc.resize(static_cast<size_t>(k_count));
    result.hg.resize(static_cast<size_t>(k_count));
    result.hgsd.resize(static_cast<size_t>(k_count));
    result.asw.resize(static_cast<size_t>(k_count));
    result.asww.resize(static_cast<size_t>(k_count));
    result.ch.resize(static_cast<size_t>(k_count));
    result.r2.resize(static_cast<size_t>(k_count));
    result.chsq.resize(static_cast<size_t>(k_count));
    result.r2sq.resize(static_cast<size_t>(k_count));
    result.hc.resize(static_cast<size_t>(k_count));

    // Pointers for easy indexed access (same order as METRIC_IDX).
    double* metric_ptrs[CQ_NUM_METRICS] = {
        result.pbc.data(),  result.hg.data(),   result.hgsd.data(),
        result.asw.data(),  result.asww.data(), result.ch.data(),
        result.r2.data(),   result.chsq.data(), result.r2sq.data(),
        result.hc.data()
    };

    // Temporary labels buffer.
    std::vector<int> labels(static_cast<size_t>(n), 1);
    std::vector<double> stats(ClusterQualNumStat);

    for (int idx = 0; idx < k_count; ++idx) {
        const int k = k_min + idx;

        // Cut the linkage tree.
        compute_labels_from_linkage(linkage, n, k, labels.data());

        // Compute quality metrics.
        std::vector<double> asw_cluster(static_cast<size_t>(2 * k));
        KendallTree kendall;
        clusterquality_dist(
            condensed, labels.data(), weights, n,
            stats.data(), k, asw_cluster.data(), kendall
        );
        finalizeKendall(kendall);

        // Scatter stats into per-metric arrays.
        for (int m = 0; m < CQ_NUM_METRICS; ++m) {
            metric_ptrs[m][idx] = stats[METRIC_IDX[m]];
        }
    }

    // Compute summary.
    result.opt_clusters.resize(CQ_NUM_METRICS);
    result.raw_values.resize(CQ_NUM_METRICS);
    result.z_scores.resize(CQ_NUM_METRICS);

    const double* const_ptrs[CQ_NUM_METRICS];
    for (int m = 0; m < CQ_NUM_METRICS; ++m) {
        const_ptrs[m] = metric_ptrs[m];
    }

    compute_summary(
        const_ptrs, k_count, k_min,
        result.opt_clusters.data(),
        result.raw_values.data(),
        result.z_scores.data()
    );

    // Build range table.
    result.range_table.resize(static_cast<size_t>(k_count) * CQ_NUM_METRICS);
    build_range_table(const_ptrs, k_count, result.range_table.data());
}

}  // namespace

// ============================================================================
// Path A: from pre-computed cluster data
// ============================================================================

ClusterQualityPipelineResult cluster_quality_from_cluster_data(
    const double* condensed, int condensed_len,
    const double* linkage, int n,
    const double* weights,
    int k_min, int k_max
) {
    if (n < 2) {
        throw std::runtime_error("Need at least 2 data points for cluster quality.");
    }
    const int expected_condensed = n * (n - 1) / 2;
    if (condensed_len != expected_condensed) {
        throw std::runtime_error("Condensed distance array size mismatch.");
    }
    if (k_min < 2 || k_max < k_min || k_max > n) {
        throw std::runtime_error("Require 2 <= k_min <= k_max <= n.");
    }

    ClusterQualityPipelineResult result;
    compute_cqi_core(condensed, linkage, weights, n, k_min, k_max, result);
    return result;
}

// ============================================================================
// Path B: from raw distance matrix
// ============================================================================

ClusterQualityPipelineResult cluster_quality_from_matrix(
    const double* matrix, int N,
    const std::string& raw_method,
    int max_clusters,
    const double* weights
) {
    if (N < 2) {
        throw std::runtime_error("Need at least 2 data points for cluster quality.");
    }

    // Reuse cluster_from_matrix for matrix prep + linkage.
    ClusterCoreResult core = cluster_from_matrix(matrix, N, raw_method, /*fast_path=*/false);

    const int k_max = std::min(max_clusters, N);
    if (k_max < 2) {
        throw std::runtime_error("max_clusters must be at least 2.");
    }

    ClusterQualityPipelineResult result;

    // Transfer linkage/matrix data from core result.
    result.linkage_matrix = std::move(core.linkage_matrix);
    result.condensed_matrix = std::move(core.condensed_matrix);
    result.full_matrix = std::move(core.full_matrix);
    result.warning_flags = core.warning_flags;
    result.euclidean_compatible = core.euclidean_compatible;

    // Compute CQI using the prepared data.
    compute_cqi_core(
        result.condensed_matrix.data(),
        result.linkage_matrix.data(),
        weights, N, 2, k_max, result
    );

    return result;
}
