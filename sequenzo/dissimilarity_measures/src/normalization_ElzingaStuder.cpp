/**
 * @Author  : Yuqi Liang 梁彧祺
 * @File    : normalization_ElzingaStuder.cpp
 * @Time    : 2026/2/10 7:17
 * @Desc    : 
 * @brief Reference-based normalization according to Elzinga & Studer (2019)
 * 
 * Implements equation (9) from the paper:
 * D_r(x,y) = d(x,y) / ((d(x,y) + d(x,r) + d(y,r)) / 2)
 * 
 * This normalization projects all objects onto a unit sphere centered at reference r.
 * Reference: Elzinga, C. H., & Studer, M. (2019). Normalization of Distance and 
 * Similarity in Sequence Analysis. Sociological Methods & Research, 48(4), 877-904.
 * 
 * We apply a theoretical normalization following Elzinga & Studer (2019),
 * dividing distances by their theoretical maxima to ensure comparability across measures.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

/**
 * Normalize a distance matrix using reference-based normalization (equation 9).
 * 
 * @param distance_matrix Input distance matrix (n x n), must be symmetric
 * @param reference_index Index of the reference object (0-based)
 * @return Normalized distance matrix (n x n)
 * 
 * Formula: D_r(x,y) = d(x,y) / ((d(x,y) + d(x,r) + d(y,r)) / 2)
 * 
 * Properties:
 * - D_r(x,x) = 0 for all x
 * - D_r(x,r) = 1 for all x != r
 * - 0 < D_r(x,y) <= 1 for all x != y
 */
py::array_t<double> normalize_distance_matrix_ElzingaStuder(
    py::array_t<double> distance_matrix,
    int reference_index
) {
    // Get buffer info
    auto buf = distance_matrix.unchecked<2>();
    int n = static_cast<int>(buf.shape(0));
    int m = static_cast<int>(buf.shape(1));
    
    if (n != m) {
        throw std::invalid_argument("distance_matrix must be square (n x n)");
    }
    
    if (reference_index < 0 || reference_index >= n) {
        throw std::invalid_argument("reference_index out of range");
    }
    
    // Allocate output array
    auto result = py::array_t<double>({n, n});
    auto result_buf = result.mutable_unchecked<2>();
    
    // Check input matrix symmetry
    // If symmetric, we can skip symmetry enforcement; if not, we'll average symmetric positions
    const double SYMMETRY_TOLERANCE = 1e-10;
    bool is_symmetric = true;
    for (int i = 0; i < n && is_symmetric; i++) {
        for (int j = i + 1; j < n; j++) {
            if (std::abs(buf(i, j) - buf(j, i)) > SYMMETRY_TOLERANCE) {
                is_symmetric = false;
                break;
            }
        }
    }
    
    // Compute normalized distances using equation (9)
    // D_r(x,y) = d(x,y) / ((d(x,y) + d(x,r) + d(y,r)) / 2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                // D1: D_r(x,x) = 0 for all x
                result_buf(i, j) = 0.0;
            } else {
                double d_xy = buf(i, j);
                double d_xr = buf(i, reference_index);
                double d_yr = buf(j, reference_index);
                
                // Denominator: (d(x,y) + d(x,r) + d(y,r)) / 2
                double denominator = (d_xy + d_xr + d_yr) / 2.0;
                
                if (denominator == 0.0) {
                    // Edge case: all distances are zero
                    result_buf(i, j) = 0.0;
                } else {
                    // Apply equation (9): D_r(x,y) = d(x,y) / denominator
                    result_buf(i, j) = d_xy / denominator;
                }
            }
        }
    }
    
    // If input matrix was not symmetric, enforce symmetry by averaging symmetric positions
    // This ensures the output satisfies distance axioms even if input had minor asymmetries
    if (!is_symmetric) {
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double avg = (result_buf(i, j) + result_buf(j, i)) / 2.0;
                result_buf(i, j) = avg;
                result_buf(j, i) = avg;
            }
        }
    }
    
    // Verify key property: D_r(x,r) = 1 for all x != r
    const double VERIFICATION_TOLERANCE = 1e-10;
    for (int i = 0; i < n; i++) {
        if (i != reference_index) {
            double d_r_value = result_buf(i, reference_index);
            if (std::abs(d_r_value - 1.0) > VERIFICATION_TOLERANCE) {
                throw std::runtime_error(
                    "Verification failed: D_r(" + std::to_string(i) + ", r) = " 
                    + std::to_string(d_r_value) + " != 1.0. "
                    "This indicates a bug in the normalization implementation."
                );
            }
        }
    }
    
    return result;
}

/**
 * Convert normalized distance matrix to normalized similarity matrix.
 * 
 * Formula: S_r(x,y) = 1 - D_r(x,y) (equation 11)
 * 
 * @param normalized_distance_matrix Normalized distance matrix from normalize_distance_matrix_ElzingaStuder
 * @return Normalized similarity matrix
 */
py::array_t<double> normalize_similarity_from_distance_ElzingaStuder(
    py::array_t<double> normalized_distance_matrix
) {
    auto buf = normalized_distance_matrix.unchecked<2>();
    int n = static_cast<int>(buf.shape(0));
    int m = static_cast<int>(buf.shape(1));
    
    if (n != m) {
        throw std::invalid_argument("normalized_distance_matrix must be square (n x n)");
    }
    
    auto result = py::array_t<double>({n, n});
    auto result_buf = result.mutable_unchecked<2>();
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result_buf(i, j) = 1.0 - buf(i, j);
        }
    }
    
    return result;
}
