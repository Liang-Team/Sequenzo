#pragma once

#include <cstddef>
#include <vector>

struct FannyCoreResult {
    std::vector<double> membership;  // n * k, row-major
    std::vector<int> clustering;     // n, 0-based crisp labels after caddy
    double objective = 0.0;
    double partition_coefficient = 0.0;
    double normalized_coefficient = 0.0;
    int iterations = 0;              // -1 if not converged (R convention)
    int k_crisp = 0;
    bool converged = false;
};

// Square distance matrix diss[n,n] row-major (diss[i*n+j]).
// If ini_membership is empty, R default initialization is used; otherwise
// ini_membership must have length n*k row-major with rows summing to 1.
FannyCoreResult fanny_from_diss(
    const double* diss,
    int n,
    int k,
    double memb_exp,
    int max_iter,
    double tol,
    const double* ini_membership,
    bool reorder_columns
);
