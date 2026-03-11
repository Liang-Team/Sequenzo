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
    bool fast_path)
{
    if (N < 1) {
        throw std::runtime_error("At least one element is needed for clustering.");
    }

    const std::string method = normalise_method(raw_method);

    // Validate method name early.
    condensed_method_code(method);

    ClusterCoreResult result;

    if (N == 1) {
        // Trivial: no merges.
        result.warning_flags = 0;
        result.euclidean_compatible = true;
        return result;
    }

    // --- Distance matrix preparation (reuse existing C++ impl) ---
    const auto n = static_cast<ssize_t>(N);
    PreparedMatrixData prep = prepare_distance_matrix_impl(
        matrix, n,
        /*enforce_symmetry=*/true,
        /*rtol=*/1e-5,
        /*atol=*/1e-8,
        /*replacement_quantile=*/0.95
    );

    result.warning_flags = prep.warning_flags;

    // Euclidean compatibility check for Ward methods.
    if (!fast_path && is_ward_method(method)) {
        EuclideanCheckResult eu = check_euclidean_compatibility_impl(
            prep.full.data(), n, method);
        result.euclidean_compatible = eu.compatible;
        if (!eu.compatible) {
            result.warning_flags |= WARN_WARD_NON_EUCLIDEAN;
        }
    }

    // Keep full matrix when not in fast_path.
    if (!fast_path) {
        result.full_matrix = std::move(prep.full);
    }

    // --- Linkage computation ---
    // compute_linkage_condensed consumes the condensed array, which is fine
    // because we move it into result afterward (after copying for linkage).
    // We need the condensed data both for linkage and for the result, so copy.
    result.condensed_matrix = prep.condensed;  // copy for later use
    std::vector<double> condensed_work = std::move(prep.condensed);

    const int mcode = condensed_method_code(method);
    result.linkage_matrix.resize(static_cast<size_t>(N - 1) * 4);

    compute_linkage_condensed(
        condensed_work.data(), N, mcode,
        result.linkage_matrix.data()
    );

    // Ward-D correction: divide distances by 2.
    if (method == "ward_d") {
        apply_ward_d_correction(result.linkage_matrix, N);
    }

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
