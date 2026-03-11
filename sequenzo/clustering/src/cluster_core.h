#pragma once

#include <string>
#include <vector>

struct ClusterCoreResult {
    std::vector<double> linkage_matrix;    // (N-1)*4, flat row-major
    std::vector<double> condensed_matrix;  // N*(N-1)/2  (empty for vector path)
    std::vector<double> full_matrix;       // N*N        (empty when fast_path)
    int warning_flags = 0;
    bool euclidean_compatible = true;
};

// Full pipeline: distance matrix -> prepared matrix -> linkage.
// matrix: N x N square distance matrix (row-major, will be copied internally).
// method: "ward", "ward_d", "ward_d2", "single", "complete", "average",
//         "weighted", "centroid", "median"
// fast_path: skip euclidean check and full_matrix retention.
ClusterCoreResult cluster_from_matrix(
    const double* matrix, int N,
    const std::string& method,
    bool fast_path
);

// Full pipeline: feature matrix -> linkage_vector.
// X: N x D feature matrix (row-major, will be copied internally).
// method: "ward", "ward_d", "ward_d2" only.
ClusterCoreResult cluster_from_features(
    const double* X, int N, int D,
    const std::string& method
);
