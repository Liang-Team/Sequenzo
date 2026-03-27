#include "distance_prep_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr double kWardTol = 1e-10;
constexpr double kWardViolationThreshold = 0.1;
constexpr int kWardSampleCap = 50;
constexpr int kEigenCheckCap = 100;

double percentile_linear(std::vector<double>& values, double q) {
    if (values.empty()) {
        throw std::runtime_error("Cannot compute percentile on empty values");
    }
    if (q <= 0.0) {
        return *std::min_element(values.begin(), values.end());
    }
    if (q >= 1.0) {
        return *std::max_element(values.begin(), values.end());
    }

    std::sort(values.begin(), values.end());
    const double pos = q * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) {
        return values[lo];
    }
    const double alpha = pos - static_cast<double>(lo);
    return values[lo] * (1.0 - alpha) + values[hi] * alpha;
}

// Use std::isfinite directly — this TU is compiled WITHOUT -ffast-math,
// so std::isfinite is IEEE-correct and generates better code than
// the manual bit-masking approach.
inline bool is_finite_val(double x) {
    return std::isfinite(x);
}

// Jacobi eigenvalue algorithm for symmetric matrices (max size ~100×100).
// Computes eigenvalues in-place. Returns them sorted ascending in eigenvalues_out.
void jacobi_eigenvalues(const double* matrix_in, int s, double* eigenvalues_out) {
    const int max_iter = 100;
    const double eps = 1e-15;

    std::vector<double> A(static_cast<size_t>(s) * s);
    std::copy(matrix_in, matrix_in + s * s, A.begin());

    for (int iter = 0; iter < max_iter; ++iter) {
        double off_diag = 0.0;
        for (int i = 0; i < s; ++i) {
            for (int j = i + 1; j < s; ++j) {
                off_diag += A[i * s + j] * A[i * s + j];
            }
        }
        if (off_diag < eps) break;

        for (int p = 0; p < s - 1; ++p) {
            for (int q = p + 1; q < s; ++q) {
                const double apq = A[p * s + q];
                if (std::abs(apq) < eps) continue;

                const double app = A[p * s + p];
                const double aqq = A[q * s + q];
                const double tau = (aqq - app) / (2.0 * apq);
                double t;
                if (tau >= 0.0) {
                    t = 1.0 / (tau + std::sqrt(1.0 + tau * tau));
                } else {
                    t = -1.0 / (-tau + std::sqrt(1.0 + tau * tau));
                }
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double sn = t * c;

                A[p * s + p] -= t * apq;
                A[q * s + q] += t * apq;
                A[p * s + q] = 0.0;
                A[q * s + p] = 0.0;

                for (int r = 0; r < s; ++r) {
                    if (r == p || r == q) continue;
                    const double arp = A[r * s + p];
                    const double arq = A[r * s + q];
                    A[r * s + p] = c * arp - sn * arq;
                    A[p * s + r] = A[r * s + p];
                    A[r * s + q] = sn * arp + c * arq;
                    A[q * s + r] = A[r * s + q];
                }
            }
        }
    }

    for (int i = 0; i < s; ++i) {
        eigenvalues_out[i] = A[i * s + i];
    }
    std::sort(eigenvalues_out, eigenvalues_out + s);
}

}  // namespace

PreparedMatrixData prepare_distance_matrix_impl(
    const double* in_ptr,
    std::ptrdiff_t n,
    bool enforce_symmetry,
    double rtol,
    double atol,
    double replacement_quantile
) {
    PreparedMatrixData out;
    out.n = n;
    const std::ptrdiff_t nn = n * n;
    out.full.resize(static_cast<size_t>(nn));

    int had_nonfinite_flag = 0;
    int had_negative_flag = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(|:had_nonfinite_flag,had_negative_flag) if(nn > 4096)
#endif
    for (std::ptrdiff_t i = 0; i < nn; ++i) {
        const double v = in_ptr[static_cast<size_t>(i)];
        out.full[static_cast<size_t>(i)] = v;
        if (!is_finite_val(v)) {
            had_nonfinite_flag = 1;
        } else if (v < 0.0) {
            had_negative_flag = 1;
        }
    }
    out.had_nonfinite = (had_nonfinite_flag != 0);
    out.had_negative = (had_negative_flag != 0);

    if (out.had_nonfinite) {
        out.warning_flags |= WARN_NONFINITE;
        std::vector<double> finite_vals;
        finite_vals.reserve(static_cast<size_t>(nn));
        for (std::ptrdiff_t i = 0; i < nn; ++i) {
            const double v = out.full[static_cast<size_t>(i)];
            if (is_finite_val(v)) {
                finite_vals.push_back(v);
            }
        }
        if (!finite_vals.empty()) {
            out.replacement_value = percentile_linear(finite_vals, replacement_quantile);
        } else {
            out.replacement_value = 1.0;
        }
#ifdef _OPENMP
#pragma omp parallel for if(nn > 4096)
#endif
        for (std::ptrdiff_t i = 0; i < nn; ++i) {
            if (!is_finite_val(out.full[static_cast<size_t>(i)])) {
                out.full[static_cast<size_t>(i)] = out.replacement_value;
            }
        }
    }

    for (std::ptrdiff_t i = 0; i < n; ++i) {
        out.full[static_cast<size_t>(i * n + i)] = 0.0;
    }
    if (out.had_negative) {
        out.warning_flags |= WARN_NEGATIVE;
#ifdef _OPENMP
#pragma omp parallel for if(nn > 4096)
#endif
        for (std::ptrdiff_t i = 0; i < nn; ++i) {
            if (out.full[static_cast<size_t>(i)] < 0.0) {
                out.full[static_cast<size_t>(i)] = 0.0;
            }
        }
    }

    if (enforce_symmetry) {
        int asymmetric_flag = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(|:asymmetric_flag) if(n > 256)
#endif
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            for (std::ptrdiff_t j = i + 1; j < n; ++j) {
                const double a = out.full[static_cast<size_t>(i * n + j)];
                const double b = out.full[static_cast<size_t>(j * n + i)];
                const double tol = atol + rtol * std::abs(b);
                if (std::abs(a - b) > tol) {
                    asymmetric_flag = 1;
                }
            }
        }
        if (asymmetric_flag) {
            out.was_symmetrized = true;
            out.warning_flags |= WARN_SYMMETRIZED;
#ifdef _OPENMP
#pragma omp parallel for if(n > 256)
#endif
            for (std::ptrdiff_t i = 0; i < n; ++i) {
                for (std::ptrdiff_t j = i + 1; j < n; ++j) {
                    const double avg = 0.5 * (
                        out.full[static_cast<size_t>(i * n + j)] +
                        out.full[static_cast<size_t>(j * n + i)]
                    );
                    out.full[static_cast<size_t>(i * n + j)] = avg;
                    out.full[static_cast<size_t>(j * n + i)] = avg;
                }
            }
        }
    }

    const std::ptrdiff_t condensed_size = n * (n - 1) / 2;
    out.condensed.resize(static_cast<size_t>(condensed_size));
#ifdef _OPENMP
#pragma omp parallel for if(n > 256)
#endif
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const std::ptrdiff_t start = (i * (2 * n - i - 1)) / 2;
        for (std::ptrdiff_t j = i + 1; j < n; ++j) {
            const std::ptrdiff_t local = j - i - 1;
            out.condensed[static_cast<size_t>(start + local)] =
                out.full[static_cast<size_t>(i * n + j)];
        }
    }
    return out;
}

// check_euclidean_compatibility_impl (NumPy-dependent) has been removed.
// Use check_euclidean_compatibility_pure() instead — pure C++ with Jacobi eigenvalues.

py::array_t<double> vector_to_pyarray_2d(std::vector<double>&& data, py::ssize_t rows, py::ssize_t cols) {
    auto* heap_vec = new std::vector<double>(std::move(data));
    py::capsule free_when_done(heap_vec, [](void* p) {
        delete reinterpret_cast<std::vector<double>*>(p);
    });
    return py::array_t<double>(
        {rows, cols},
        {static_cast<py::ssize_t>(sizeof(double) * cols), static_cast<py::ssize_t>(sizeof(double))},
        heap_vec->data(),
        free_when_done
    );
}

py::array_t<double> vector_to_pyarray_1d(std::vector<double>&& data) {
    auto* heap_vec = new std::vector<double>(std::move(data));
    py::capsule free_when_done(heap_vec, [](void* p) {
        delete reinterpret_cast<std::vector<double>*>(p);
    });
    return py::array_t<double>(
        {static_cast<py::ssize_t>(heap_vec->size())},
        {static_cast<py::ssize_t>(sizeof(double))},
        heap_vec->data(),
        free_when_done
    );
}

// ============================================================================
// prepare_distance_condensed_impl — fast path for condensed input
// ============================================================================

PreparedCondensedData prepare_distance_condensed_impl(
    const double* in_ptr,
    std::ptrdiff_t condensed_len,
    std::ptrdiff_t n,
    double replacement_quantile
) {
    PreparedCondensedData out;
    out.n = n;

    const std::ptrdiff_t expected = n * (n - 1) / 2;
    if (condensed_len != expected) {
        throw std::runtime_error(
            "Condensed array length mismatch: expected " +
            std::to_string(expected) + ", got " + std::to_string(condensed_len));
    }

    out.condensed.resize(static_cast<size_t>(condensed_len));

    int had_nonfinite_flag = 0;
    int had_negative_flag = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(|:had_nonfinite_flag,had_negative_flag) if(condensed_len > 4096)
#endif
    for (std::ptrdiff_t i = 0; i < condensed_len; ++i) {
        const double v = in_ptr[static_cast<size_t>(i)];
        out.condensed[static_cast<size_t>(i)] = v;
        if (!is_finite_val(v)) {
            had_nonfinite_flag = 1;
        } else if (v < 0.0) {
            had_negative_flag = 1;
        }
    }
    out.had_nonfinite = (had_nonfinite_flag != 0);
    out.had_negative = (had_negative_flag != 0);

    if (out.had_nonfinite) {
        out.warning_flags |= WARN_NONFINITE;
        std::vector<double> finite_vals;
        finite_vals.reserve(static_cast<size_t>(condensed_len));
        for (std::ptrdiff_t i = 0; i < condensed_len; ++i) {
            const double v = out.condensed[static_cast<size_t>(i)];
            if (is_finite_val(v)) {
                finite_vals.push_back(v);
            }
        }
        if (!finite_vals.empty()) {
            out.replacement_value = percentile_linear(finite_vals, replacement_quantile);
        } else {
            out.replacement_value = 1.0;
        }
#ifdef _OPENMP
#pragma omp parallel for if(condensed_len > 4096)
#endif
        for (std::ptrdiff_t i = 0; i < condensed_len; ++i) {
            if (!is_finite_val(out.condensed[static_cast<size_t>(i)])) {
                out.condensed[static_cast<size_t>(i)] = out.replacement_value;
            }
        }
    }

    if (out.had_negative) {
        out.warning_flags |= WARN_NEGATIVE;
#ifdef _OPENMP
#pragma omp parallel for if(condensed_len > 4096)
#endif
        for (std::ptrdiff_t i = 0; i < condensed_len; ++i) {
            if (out.condensed[static_cast<size_t>(i)] < 0.0) {
                out.condensed[static_cast<size_t>(i)] = 0.0;
            }
        }
    }

    return out;
}

// ============================================================================
// prepare_matrix_to_condensed_fast — single-pass extraction for fast_path
// ============================================================================

PreparedCondensedData prepare_matrix_to_condensed_fast(
    const double* in_ptr,
    std::ptrdiff_t n,
    double replacement_quantile
) {
    PreparedCondensedData out;
    out.n = n;

    const std::ptrdiff_t condensed_len = n * (n - 1) / 2;
    out.condensed.resize(static_cast<size_t>(condensed_len));

    // --- Pass 1: extract upper triangle, symmetrize, detect problems ---
    int had_nonfinite_flag = 0;
    int had_negative_flag = 0;

#ifdef _OPENMP
#pragma omp parallel for reduction(|:had_nonfinite_flag,had_negative_flag) if(n > 256)
#endif
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const std::ptrdiff_t row_start = (i * (2 * n - i - 1)) / 2;
        for (std::ptrdiff_t j = i + 1; j < n; ++j) {
            const double a = in_ptr[static_cast<size_t>(i * n + j)];
            const double b = in_ptr[static_cast<size_t>(j * n + i)];

            // Symmetrize inline: average of (i,j) and (j,i).
            // For symmetric matrices this is a no-op; for asymmetric ones
            // it avoids the need for a separate symmetry-check pass.
            const double v = (a + b) * 0.5;

            const std::ptrdiff_t local = j - i - 1;
            const auto idx = static_cast<size_t>(row_start + local);

            if (!is_finite_val(v)) {
                had_nonfinite_flag = 1;
                out.condensed[idx] = v;  // placeholder, fixed in pass 2
            } else if (v < 0.0) {
                had_negative_flag = 1;
                out.condensed[idx] = 0.0;  // clamp inline
            } else {
                out.condensed[idx] = v;
            }
        }
    }
    out.had_nonfinite = (had_nonfinite_flag != 0);
    out.had_negative = (had_negative_flag != 0);

    if (out.had_negative) {
        out.warning_flags |= WARN_NEGATIVE;
    }

    // --- Pass 2 (rare): replace non-finite values ---
    if (out.had_nonfinite) {
        out.warning_flags |= WARN_NONFINITE;
        std::vector<double> finite_vals;
        finite_vals.reserve(static_cast<size_t>(condensed_len));
        for (std::ptrdiff_t i = 0; i < condensed_len; ++i) {
            const double v = out.condensed[static_cast<size_t>(i)];
            if (is_finite_val(v)) {
                finite_vals.push_back(v);
            }
        }
        if (!finite_vals.empty()) {
            out.replacement_value = percentile_linear(finite_vals, replacement_quantile);
        } else {
            out.replacement_value = 1.0;
        }
#ifdef _OPENMP
#pragma omp parallel for if(condensed_len > 4096)
#endif
        for (std::ptrdiff_t i = 0; i < condensed_len; ++i) {
            if (!is_finite_val(out.condensed[static_cast<size_t>(i)])) {
                out.condensed[static_cast<size_t>(i)] = out.replacement_value;
            }
        }
    }

    return out;
}

// ============================================================================
// prepare_matrix_to_condensed_fused — replaces multi-pass full path
// ============================================================================

PreparedCondensedData prepare_matrix_to_condensed_fused(
    const double* in_ptr,
    std::ptrdiff_t n,
    double replacement_quantile,
    bool check_symmetry,
    double rtol,
    double atol
) {
    PreparedCondensedData out;
    out.n = n;

    const std::ptrdiff_t condensed_len = n * (n - 1) / 2;
    out.condensed.resize(static_cast<size_t>(condensed_len));

    // --- Single fused pass: extract upper triangle, symmetrize, validate ---
    int had_nonfinite_flag = 0;
    int had_negative_flag = 0;
    int had_asymmetry_flag = 0;

#ifdef _OPENMP
#pragma omp parallel for reduction(|:had_nonfinite_flag,had_negative_flag,had_asymmetry_flag) if(n > 256)
#endif
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const std::ptrdiff_t row_start = (i * (2 * n - i - 1)) / 2;
        for (std::ptrdiff_t j = i + 1; j < n; ++j) {
            const double a = in_ptr[static_cast<size_t>(i * n + j)];
            const double b = in_ptr[static_cast<size_t>(j * n + i)];

            // Optional symmetry detection
            if (check_symmetry) {
                const double tol = atol + rtol * std::abs(b);
                if (std::abs(a - b) > tol) {
                    had_asymmetry_flag = 1;
                }
            }

            // Symmetrize inline: average of (i,j) and (j,i).
            const double v = (a + b) * 0.5;

            const std::ptrdiff_t local = j - i - 1;
            const auto idx = static_cast<size_t>(row_start + local);

            if (!is_finite_val(v)) {
                had_nonfinite_flag = 1;
                out.condensed[idx] = v;  // placeholder, fixed in pass 2
            } else if (v < 0.0) {
                had_negative_flag = 1;
                out.condensed[idx] = 0.0;  // clamp inline
            } else {
                out.condensed[idx] = v;
            }
        }
    }
    out.had_nonfinite = (had_nonfinite_flag != 0);
    out.had_negative = (had_negative_flag != 0);

    if (out.had_negative) {
        out.warning_flags |= WARN_NEGATIVE;
    }
    if (had_asymmetry_flag) {
        out.warning_flags |= WARN_SYMMETRIZED;
    }

    // --- Pass 2 (rare): replace non-finite values ---
    if (out.had_nonfinite) {
        out.warning_flags |= WARN_NONFINITE;
        std::vector<double> finite_vals;
        finite_vals.reserve(static_cast<size_t>(condensed_len));
        for (std::ptrdiff_t i = 0; i < condensed_len; ++i) {
            const double v = out.condensed[static_cast<size_t>(i)];
            if (is_finite_val(v)) {
                finite_vals.push_back(v);
            }
        }
        if (!finite_vals.empty()) {
            out.replacement_value = percentile_linear(finite_vals, replacement_quantile);
        } else {
            out.replacement_value = 1.0;
        }
#ifdef _OPENMP
#pragma omp parallel for if(condensed_len > 4096)
#endif
        for (std::ptrdiff_t i = 0; i < condensed_len; ++i) {
            if (!is_finite_val(out.condensed[static_cast<size_t>(i)])) {
                out.condensed[static_cast<size_t>(i)] = out.replacement_value;
            }
        }
    }

    return out;
}

// ============================================================================
// check_euclidean_compatibility_pure — no NumPy dependency
// ============================================================================

EuclideanCheckResult check_euclidean_compatibility_pure(
    const double* matrix_ptr,
    std::ptrdiff_t n,
    const std::string& method
) {
    EuclideanCheckResult out;
    const std::string m = method;
    if (m != "ward" && m != "ward_d" && m != "ward_d2") {
        out.compatible = true;
        return out;
    }

    const int sample_size = static_cast<int>(std::min<std::ptrdiff_t>(kWardSampleCap, n));
    std::vector<int> indices(static_cast<size_t>(n));
    std::iota(indices.begin(), indices.end(), 0);
    if (n > sample_size) {
        std::mt19937 gen(5489U + static_cast<uint32_t>(n));
        std::shuffle(indices.begin(), indices.end(), gen);
        indices.resize(static_cast<size_t>(sample_size));
    }
    out.sample_n = static_cast<int>(indices.size());
    const int s = out.sample_n;

    std::vector<double> sample(static_cast<size_t>(s * s), 0.0);
#ifdef _OPENMP
#pragma omp parallel for if(s > 16)
#endif
    for (int i = 0; i < s; ++i) {
        for (int j = 0; j < s; ++j) {
            sample[static_cast<size_t>(i * s + j)] =
                matrix_ptr[static_cast<size_t>(indices[static_cast<size_t>(i)] * n + indices[static_cast<size_t>(j)])];
        }
    }

    long long violations = 0;
    long long total_checks = 0;
    for (int i = 0; i < s; ++i) {
        for (int j = i + 1; j < s; ++j) {
            for (int k = j + 1; k < s; ++k) {
                const double dij = sample[static_cast<size_t>(i * s + j)];
                const double dik = sample[static_cast<size_t>(i * s + k)];
                const double djk = sample[static_cast<size_t>(j * s + k)];
                if (dik > dij + djk + kWardTol || dij > dik + djk + kWardTol || djk > dij + dik + kWardTol) {
                    ++violations;
                }
                ++total_checks;
            }
        }
    }
    if (total_checks > 0) {
        out.violation_rate = static_cast<double>(violations) / static_cast<double>(total_checks);
        if (out.violation_rate > kWardViolationThreshold) {
            out.compatible = false;
            return out;
        }
    }

    if (s <= kEigenCheckCap) {
        // Build centering matrix H = I - 1/s * ones, compute B = -0.5 * H * D² * H
        // Then find eigenvalues with Jacobi method (pure C++, no NumPy).
        std::vector<double> sample_for_eigen = sample;
        double max_abs_sample = 0.0;
        for (double v : sample_for_eigen) {
            if (is_finite_val(v)) {
                max_abs_sample = std::max(max_abs_sample, std::abs(v));
            }
        }
        if (max_abs_sample > 0.0) {
            const double inv_scale = 1.0 / max_abs_sample;
            for (double& v : sample_for_eigen) {
                v *= inv_scale;
            }
        }

        // sq = element-wise square of distance matrix
        std::vector<double> sq(static_cast<size_t>(s * s));
        for (int i = 0; i < s * s; ++i) {
            sq[i] = sample_for_eigen[i] * sample_for_eigen[i];
        }

        // B = -0.5 * H * sq * H  where H = I - (1/s)*ones
        // Compute tmp = sq * H first, then B = H * tmp, then scale by -0.5
        const double inv_s = 1.0 / static_cast<double>(s);

        // tmp = sq * H = sq - sq * ones * (1/s)
        // (sq * ones)[i][j] = sum_k sq[i][k], so (sq * ones * 1/s)[i][j] = row_sum[i] / s
        std::vector<double> row_sums(s, 0.0);
        for (int i = 0; i < s; ++i) {
            for (int k = 0; k < s; ++k) {
                row_sums[i] += sq[i * s + k];
            }
        }

        std::vector<double> tmp(static_cast<size_t>(s * s));
        for (int i = 0; i < s; ++i) {
            for (int j = 0; j < s; ++j) {
                tmp[i * s + j] = sq[i * s + j] - row_sums[i] * inv_s;
            }
        }

        // B = H * tmp = tmp - ones * (1/s) * tmp
        // (ones * (1/s) * tmp)[i][j] = (1/s) * sum_k tmp[k][j] = col_sum[j] / s
        std::vector<double> col_sums(s, 0.0);
        for (int j = 0; j < s; ++j) {
            for (int k = 0; k < s; ++k) {
                col_sums[j] += tmp[k * s + j];
            }
        }

        std::vector<double> B(static_cast<size_t>(s * s));
        for (int i = 0; i < s; ++i) {
            for (int j = 0; j < s; ++j) {
                B[i * s + j] = -0.5 * (tmp[i * s + j] - col_sums[j] * inv_s);
            }
        }

        // Symmetrize B to guard against floating-point asymmetry
        for (int i = 0; i < s; ++i) {
            for (int j = i + 1; j < s; ++j) {
                double avg = 0.5 * (B[i * s + j] + B[j * s + i]);
                B[i * s + j] = avg;
                B[j * s + i] = avg;
            }
        }

        std::vector<double> eigenvals(s);
        jacobi_eigenvalues(B.data(), s, eigenvals.data());

        double neg_energy = 0.0;
        double total_energy = 0.0;
        for (int i = 0; i < s; ++i) {
            const double ev = eigenvals[i];
            if (ev < -kWardTol) {
                neg_energy += -ev;
            }
            total_energy += std::abs(ev);
        }
        if (total_energy > 0.0) {
            out.neg_energy_ratio = neg_energy / total_energy;
            if (out.neg_energy_ratio > kWardViolationThreshold) {
                out.compatible = false;
                return out;
            }
        }
    }

    out.compatible = true;
    return out;
}
