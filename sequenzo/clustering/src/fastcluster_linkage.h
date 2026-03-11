#pragma once

#include <vector>
#include <string>

// Compute linkage from a condensed distance matrix (N*(N-1)/2 doubles).
// method_code follows fastcluster conventions:
//   0=single, 1=complete, 2=average, 3=weighted,
//   4=ward (ward_d), 5=centroid, 6=median, 7=ward_d2
// condensed is CONSUMED (modified in-place) for methods that square distances.
// Z_out must be pre-allocated with (N-1)*4 doubles.
void compute_linkage_condensed(
    double* condensed, int N, int method_code,
    double* Z_out
);

// Compute linkage from a feature matrix (N x D doubles, row-major).
// Only supports Ward and Ward-D2 with Euclidean metric.
// method_code: 4=ward (ward_d), 7=ward_d2
// X is CONSUMED (modified in-place for merge operations).
// Z_out must be pre-allocated with (N-1)*4 doubles.
void compute_linkage_vector_euclidean(
    double* X, int N, int D, int method_code,
    double* Z_out
);

// Map method string to fastcluster method_code.
// Throws std::runtime_error on unknown method.
int method_string_to_code(const std::string& method);
