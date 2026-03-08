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

bool is_finite_ieee(double x) {
    uint64_t bits = 0;
    std::memcpy(&bits, &x, sizeof(double));
    return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}

}  // namespace

PreparedMatrixData prepare_distance_matrix_impl(
    const double* in_ptr,
    py::ssize_t n,
    bool enforce_symmetry,
    double rtol,
    double atol,
    double replacement_quantile
) {
    PreparedMatrixData out;
    out.n = n;
    const py::ssize_t nn = n * n;
    out.full.resize(static_cast<size_t>(nn));

    int had_nonfinite_flag = 0;
    int had_negative_flag = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(|:had_nonfinite_flag,had_negative_flag) if(nn > 4096)
#endif
    for (py::ssize_t i = 0; i < nn; ++i) {
        const double v = in_ptr[static_cast<size_t>(i)];
        out.full[static_cast<size_t>(i)] = v;
        if (!is_finite_ieee(v)) {
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
        for (py::ssize_t i = 0; i < nn; ++i) {
            const double v = out.full[static_cast<size_t>(i)];
            if (is_finite_ieee(v)) {
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
        for (py::ssize_t i = 0; i < nn; ++i) {
            if (!is_finite_ieee(out.full[static_cast<size_t>(i)])) {
                out.full[static_cast<size_t>(i)] = out.replacement_value;
            }
        }
    }

    for (py::ssize_t i = 0; i < n; ++i) {
        out.full[static_cast<size_t>(i * n + i)] = 0.0;
    }
    if (out.had_negative) {
        out.warning_flags |= WARN_NEGATIVE;
#ifdef _OPENMP
#pragma omp parallel for if(nn > 4096)
#endif
        for (py::ssize_t i = 0; i < nn; ++i) {
            if (out.full[static_cast<size_t>(i)] < 0.0) {
                out.full[static_cast<size_t>(i)] = 0.0;
            }
        }
    }

    if (enforce_symmetry) {
        bool is_symmetric = true;
        for (py::ssize_t i = 0; i < n && is_symmetric; ++i) {
            for (py::ssize_t j = i + 1; j < n; ++j) {
                const double a = out.full[static_cast<size_t>(i * n + j)];
                const double b = out.full[static_cast<size_t>(j * n + i)];
                const double tol = atol + rtol * std::abs(b);
                if (std::abs(a - b) > tol) {
                    is_symmetric = false;
                    break;
                }
            }
        }
        if (!is_symmetric) {
            out.was_symmetrized = true;
            out.warning_flags |= WARN_SYMMETRIZED;
            for (py::ssize_t i = 0; i < n; ++i) {
                for (py::ssize_t j = i + 1; j < n; ++j) {
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

    const py::ssize_t condensed_size = n * (n - 1) / 2;
    out.condensed.resize(static_cast<size_t>(condensed_size));
#ifdef _OPENMP
#pragma omp parallel for if(n > 256)
#endif
    for (py::ssize_t i = 0; i < n; ++i) {
        const py::ssize_t start = (i * (2 * n - i - 1)) / 2;
        for (py::ssize_t j = i + 1; j < n; ++j) {
            const py::ssize_t local = j - i - 1;
            out.condensed[static_cast<size_t>(start + local)] =
                out.full[static_cast<size_t>(i * n + j)];
        }
    }
    return out;
}

EuclideanCheckResult check_euclidean_compatibility_impl(
    const double* matrix_ptr,
    py::ssize_t n,
    const std::string& method
) {
    EuclideanCheckResult out;
    const std::string m = method;
    if (m != "ward" && m != "ward_d" && m != "ward_d2") {
        out.compatible = true;
        return out;
    }

    const int sample_size = static_cast<int>(std::min<py::ssize_t>(kWardSampleCap, n));
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
        try {
            // Normalize sampled distances before squaring to avoid overflow/invalid
            // in NumPy matmul for very large but finite distance values.
            std::vector<double> sample_for_eigen = sample;
            double max_abs_sample = 0.0;
            for (double v : sample_for_eigen) {
                if (is_finite_ieee(v)) {
                    max_abs_sample = std::max(max_abs_sample, std::abs(v));
                }
            }
            if (max_abs_sample > 0.0) {
                const double inv_scale = 1.0 / max_abs_sample;
                for (double& v : sample_for_eigen) {
                    v *= inv_scale;
                }
            }

            auto sample_arr = py::array_t<double>({s, s});
            auto sample_buf = sample_arr.request();
            auto* sample_ptr = static_cast<double*>(sample_buf.ptr);
            std::copy(sample_for_eigen.begin(), sample_for_eigen.end(), sample_ptr);

            py::module_ np = py::module_::import("numpy");
            py::module_ linalg = py::module_::import("numpy.linalg");
            py::object H = np.attr("eye")(s) - (np.attr("ones")(py::make_tuple(s, s)) / py::float_(static_cast<double>(s)));
            py::object sq = np.attr("square")(sample_arr);
            py::object B = py::float_(-0.5) * H.attr("__matmul__")(sq.attr("__matmul__")(H));
            py::array eigenvals = linalg.attr("eigvalsh")(B).cast<py::array>();
            auto eig_buf = eigenvals.request();
            auto* eig_ptr = static_cast<double*>(eig_buf.ptr);

            double neg_energy = 0.0;
            double total_energy = 0.0;
            for (py::ssize_t i = 0; i < eig_buf.size; ++i) {
                const double ev = eig_ptr[i];
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
        } catch (const py::error_already_set&) {
            // Keep compatibility as-is if eig computation fails, same as Python behavior.
        }
    }

    out.compatible = true;
    return out;
}

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
