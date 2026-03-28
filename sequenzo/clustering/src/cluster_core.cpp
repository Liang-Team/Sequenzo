/*
 * cluster_core.cpp
 *
 * Complete hierarchical clustering pipeline in C++.
 * Replaces the Python Cluster.__init__ logic for both the
 * distance-matrix path and the feature-vector path.
 */

#include "cluster_core.h"
#include "fastcluster_linkage.h"
#include "distance_prep_utils.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _MSC_VER
typedef ptrdiff_t ssize_t;
#endif

namespace {

// Canonical method name normalisation.
// "ward" -> "ward_d" for backward compat; everything else lowercase passthrough.
std::string normalise_method(const std::string& raw) {
    if (raw == "ward") return "ward_d";
    return raw;
}

bool is_ward_method(const std::string& method) {
    return method == "ward_d" || method == "ward_d2";
}

// Apply Ward-D correction: divide linkage distances (col 2) by 2.
void apply_ward_d_correction(std::vector<double>& Z, int N) {
    const int rows = N - 1;
    for (int i = 0; i < rows; ++i) {
        Z[static_cast<size_t>(i) * 4 + 2] /= 2.0;
    }
}

// Map normalised method name to the fastcluster-internal method code
// used for the condensed-distance path.
// This resolves "ward_d" -> METHOD_METR_WARD and keeps fastcluster semantics.
int condensed_method_code(const std::string& method) {
    // For the condensed path, both ward_d and ward_d2 use the same core
    // algorithm (ward_d2 / NN_chain_core<WARD_D2>).  We differentiate
    // them only after linkage via the Ward-D correction.
    if (method == "ward_d") return 4;   // METHOD_METR_WARD
    if (method == "ward_d2") return 7;  // METHOD_METR_WARD_D2
    return method_string_to_code(method);
}

// Map normalised method name to the fastcluster-internal method code
// used for the feature-vector path.
int vector_method_code(const std::string& method) {
    if (method == "ward_d") return 4;   // METHOD_METR_WARD
    if (method == "ward_d2") return 7;  // METHOD_METR_WARD_D2
    throw std::runtime_error(
        "cluster_from_features only supports ward_d / ward_d2, got '" +
        method + "'.");
}

}  // namespace

// ============================================================================
// cluster_from_matrix
// ============================================================================

ClusterCoreResult cluster_from_matrix(
    const double* matrix, int N,
    const std::string& raw_method,
    bool fast_path,
    bool retain_condensed)
{
    if (N < 1) {
        throw std::runtime_error("At least one element is needed for clustering.");
    }

    const std::string method = normalise_method(raw_method);

    // Validate method name early.
    condensed_method_code(method);

    ClusterCoreResult result;

    if (N == 1) {
        result.warning_flags = 0;
        result.euclidean_compatible = true;
        return result;
    }

    const auto n = static_cast<ssize_t>(N);
    const int mcode = condensed_method_code(method);

    if (fast_path) {
        // ---- FAST PATH ----
        // Single-pass extraction: square matrix → condensed.
        // No full-matrix allocation, no symmetry check, no Euclidean check.
        PreparedCondensedData prep = prepare_matrix_to_condensed_fast(
            matrix, n, /*replacement_quantile=*/0.95);

        result.warning_flags = prep.warning_flags;

        result.linkage_matrix.resize(static_cast<size_t>(N - 1) * 4);

        if (retain_condensed) {
            // Copy condensed before linkage corrupts it.
            result.condensed_matrix = prep.condensed.to_vector();
        }

        compute_linkage_condensed(
            prep.condensed.data(), N, mcode,
            result.linkage_matrix.data()
        );

        if (method == "ward_d") {
            apply_ward_d_correction(result.linkage_matrix, N);
        }
    } else {
        // ---- FULL PATH (optimized: fused single-pass extraction) ----
        // Uses prepare_matrix_to_condensed_fused to combine symmetry check,
        // validation, and condensed extraction into one upper-triangle pass.
        // Never allocates a full N×N copy.
        PreparedCondensedData prep = prepare_matrix_to_condensed_fused(
            matrix, n,
            /*replacement_quantile=*/0.95,
            /*check_symmetry=*/true,
            /*rtol=*/1e-5,
            /*atol=*/1e-8
        );

        result.warning_flags = prep.warning_flags;

        // Euclidean compatibility check for Ward methods.
        // Read directly from the input matrix pointer (still alive in our scope).
        if (is_ward_method(method)) {
            EuclideanCheckResult eu = check_euclidean_compatibility_pure(
                matrix, n, method);
            result.euclidean_compatible = eu.compatible;
            if (!eu.compatible) {
                result.warning_flags |= WARN_WARD_NON_EUCLIDEAN;
            }
        }

        // full_matrix is no longer stored — Python can lazily reconstruct
        // from condensed_matrix if needed.

        result.linkage_matrix.resize(static_cast<size_t>(N - 1) * 4);

        if (retain_condensed) {
            // All methods modify condensed in-place during linkage
            // (Ward/centroid/median square distances; single/complete/average/
            // weighted update distances via f_single/f_complete/etc).
            // Copy before linkage to preserve the original.
            result.condensed_matrix = prep.condensed.to_vector();
        }

        compute_linkage_condensed(
            prep.condensed.data(), N, mcode,
            result.linkage_matrix.data()
        );

        if (method == "ward_d") {
            apply_ward_d_correction(result.linkage_matrix, N);
        }
    }

    return result;
}

// ============================================================================
// cluster_from_condensed
// ============================================================================

ClusterCoreResult cluster_from_condensed(
    const double* condensed, int N,
    const std::string& raw_method,
    bool fast_path,
    bool retain_condensed)
{
    if (N < 1) {
        throw std::runtime_error("At least one element is needed for clustering.");
    }

    const std::string method = normalise_method(raw_method);
    condensed_method_code(method);

    ClusterCoreResult result;

    if (N == 1) {
        result.warning_flags = 0;
        result.euclidean_compatible = true;
        return result;
    }

    const auto n = static_cast<ssize_t>(N);
    const auto condensed_len = static_cast<ssize_t>(N) * (N - 1) / 2;

    PreparedCondensedData prep = prepare_distance_condensed_impl(
        condensed, condensed_len, n, /*replacement_quantile=*/0.95
    );

    result.warning_flags = prep.warning_flags;

    const int mcode = condensed_method_code(method);
    result.linkage_matrix.resize(static_cast<size_t>(N - 1) * 4);

    if (fast_path) {
        // ---- FAST PATH ----
        // No Euclidean check, no condensed copy — compute linkage directly.
        if (retain_condensed) {
            result.condensed_matrix = prep.condensed.to_vector();
        }
        compute_linkage_condensed(
            prep.condensed.data(), N, mcode,
            result.linkage_matrix.data()
        );
    } else {
        // ---- FULL PATH ----
        // Ward euclidean check requires a full matrix for sampling.
        if (is_ward_method(method)) {
            const int sample_cap = std::min(N, 50);
            std::vector<double> sample_full(static_cast<size_t>(sample_cap) * sample_cap, 0.0);
            for (int i = 0; i < sample_cap; ++i) {
                for (int j = i + 1; j < sample_cap; ++j) {
                    const auto idx = static_cast<size_t>(
                        static_cast<ssize_t>(i) * (2 * n - i - 1) / 2 + (j - i - 1));
                    const double d = prep.condensed[idx];
                    sample_full[i * sample_cap + j] = d;
                    sample_full[j * sample_cap + i] = d;
                }
            }
            EuclideanCheckResult eu = check_euclidean_compatibility_pure(
                sample_full.data(), static_cast<ssize_t>(sample_cap), method);
            result.euclidean_compatible = eu.compatible;
            if (!eu.compatible) {
                result.warning_flags |= WARN_WARD_NON_EUCLIDEAN;
            }
        }

        if (retain_condensed) {
            // All methods corrupt condensed during linkage — copy to preserve.
            result.condensed_matrix = prep.condensed.to_vector();
        }

        compute_linkage_condensed(
            prep.condensed.data(), N, mcode,
            result.linkage_matrix.data()
        );
    }

    if (method == "ward_d") {
        apply_ward_d_correction(result.linkage_matrix, N);
    }

    // full_matrix left empty — lazily reconstructed in Python if needed.
    return result;
}

// ============================================================================
// cluster_from_features
// ============================================================================

ClusterCoreResult cluster_from_features(
    const double* X, int N, int D,
    const std::string& raw_method)
{
    if (N < 1) {
        throw std::runtime_error("At least one element is needed for clustering.");
    }
    if (D < 1) {
        throw std::runtime_error("Invalid dimension of the data set.");
    }

    const std::string method = normalise_method(raw_method);
    const int mcode = vector_method_code(method);

    ClusterCoreResult result;

    if (N == 1) {
        result.warning_flags = 0;
        result.euclidean_compatible = true;
        return result;
    }

    // compute_linkage_vector_euclidean modifies X in-place (merge_inplace),
    // so we must work on a copy.
    std::vector<double> X_work(static_cast<size_t>(N) * static_cast<size_t>(D));
    std::memcpy(X_work.data(), X,
                static_cast<size_t>(N) * static_cast<size_t>(D) * sizeof(double));

    result.linkage_matrix.resize(static_cast<size_t>(N - 1) * 4);

    compute_linkage_vector_euclidean(
        X_work.data(), N, D, mcode,
        result.linkage_matrix.data()
    );

    // Ward-D correction: linkage_vector with METHOD_METR_WARD produces
    // ward_d2-style distances; divide by 2 for ward_d.
    if (method == "ward_d") {
        apply_ward_d_correction(result.linkage_matrix, N);
    }

    // condensed_matrix and full_matrix are left empty for the vector path;
    // Python can lazily compute them from the feature matrix if needed.
    result.euclidean_compatible = true;
    result.warning_flags = 0;

    return result;
}
