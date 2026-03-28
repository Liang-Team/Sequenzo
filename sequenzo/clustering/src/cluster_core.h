#pragma once

#include <string>
#include <vector>

struct ClusterCoreResult {
    std::vector<double> linkage_matrix;    // (N-1)*4, flat row-major
    std::vector<double> condensed_matrix;  // N*(N-1)/2  (empty unless retain_condensed)
    std::vector<double> full_matrix;       // always empty (kept for API compat)
    int warning_flags = 0;
    bool euclidean_compatible = true;
};

// Full pipeline: distance matrix -> prepared matrix -> linkage.
// matrix: N x N square distance matrix (row-major).
// method: "ward", "ward_d", "ward_d2", "single", "complete", "average",
//         "weighted", "centroid", "median"
// fast_path: skip euclidean check and full_matrix retention.
// retain_condensed: if true, preserve the original condensed array in the result
//                   (costs an extra N*(N-1)/2 copy since linkage modifies in-place).
//                   Default false for performance.
ClusterCoreResult cluster_from_matrix(
    const double* matrix, int N,
    const std::string& method,
    bool fast_path,
    bool retain_condensed = false
);

// Fast pipeline: condensed distance array -> linkage.
// condensed: N*(N-1)/2 doubles (upper triangle, SciPy pdist order).
// retain_condensed: see cluster_from_matrix.
ClusterCoreResult cluster_from_condensed(
    const double* condensed, int N,
    const std::string& method,
    bool fast_path,
    bool retain_condensed = false
);

// Full pipeline: feature matrix -> linkage_vector.
// X: N x D feature matrix (row-major, will be copied internally).
// method: "ward", "ward_d", "ward_d2" only.
ClusterCoreResult cluster_from_features(
    const double* X, int N, int D,
    const std::string& method
);
