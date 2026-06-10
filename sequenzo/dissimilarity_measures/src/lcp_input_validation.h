#ifndef LCP_INPUT_VALIDATION_H
#define LCP_INPUT_VALIDATION_H

#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace lcp_input {

constexpr double TOTALDUR_REL_TOL = 1e-10;
constexpr double NORM_CLAMP_EPS = 1e-12;

struct RefseqConfig {
    bool has_refseq = false;
    int nseq = 0;
    int rseq1 = -1;
    int rseq2 = -1;
};

inline void require_2d(const py::array& arr, const char* name) {
    if (arr.ndim() != 2) {
        throw std::invalid_argument(std::string(name) + " must be a 2D array");
    }
}

inline void require_1d(const py::array& arr, const char* name) {
    if (arr.ndim() != 1) {
        throw std::invalid_argument(std::string(name) + " must be a 1D array");
    }
}

inline void require_refseqS(const py::array_t<int>& refseqS) {
    if (refseqS.ndim() != 1 || refseqS.shape(0) != 2) {
        throw std::invalid_argument("refseqS must be a length-2 int vector");
    }
}

inline void require_sign(int sign) {
    if (sign != -1 && sign != 1) {
        throw std::invalid_argument("sign must be -1 or 1");
    }
}

inline void require_norm(int norm) {
    if (norm < 0 || norm > 4) {
        throw std::invalid_argument("norm must be one of 0, 1, 2, 3, or 4");
    }
}

inline void require_lcpspell_norm(int norm) {
    require_norm(norm);
    if (norm != 0 && norm != 3) {
        throw std::invalid_argument(
            "LCPspell/RLCPspell support only norm=none or norm=maxdist");
    }
}

inline void require_lcpmst_norm(int norm) {
    require_norm(norm);
    if (norm == 1) {
        throw std::invalid_argument(
            "LCPmst/RLCPmst do not support norm=maxlength");
    }
}

inline void require_lcpprod_norm(int norm) {
    require_norm(norm);
    if (norm != 0) {
        throw std::invalid_argument(
            "LCPprod/RLCPprod support only raw distances with norm=none");
    }
}

inline void require_pair_indices(int i, int j, int upper) {
    if (i < 0 || i >= upper || j < 0 || j >= upper) {
        throw std::out_of_range("sequence index is out of bounds");
    }
}

inline double clamp_unit_interval_or_throw(double d, const char* context) {
    if (!std::isfinite(d)) {
        throw std::runtime_error(std::string(context) + " is not finite");
    }
    if (d < -NORM_CLAMP_EPS || d > 1.0 + NORM_CLAMP_EPS) {
        throw std::runtime_error(std::string(context) + " fell outside [0, 1]");
    }
    return std::max(0.0, std::min(1.0, d));
}

inline RefseqConfig parse_refseq(const py::array_t<int>& refseqS, int original_nseq) {
    require_refseqS(refseqS);

    RefseqConfig cfg;
    const int a = refseqS.at(0);
    const int b = refseqS.at(1);

    if (a == -1 && b == -1) {
        cfg.has_refseq = false;
        cfg.nseq = original_nseq;
        cfg.rseq1 = -1;
        cfg.rseq2 = -1;
        return cfg;
    }

    if (a < b) {
        if (a <= 0 || b != original_nseq) {
            throw std::invalid_argument(
                "refseqS subset mode requires 0 < a < b == number of input sequences");
        }
        cfg.has_refseq = true;
        cfg.rseq1 = a;
        cfg.rseq2 = b;
        cfg.nseq = a;
        return cfg;
    }

    if (a == b) {
        if (a < 1 || a > original_nseq) {
            throw std::invalid_argument("refseqS reference index is out of bounds");
        }
        cfg.has_refseq = true;
        cfg.rseq1 = a - 1;
        cfg.rseq2 = a;
        cfg.nseq = original_nseq;
        return cfg;
    }

    throw std::invalid_argument("Invalid refseqS convention");
}

inline void require_refseq_for_compute(const RefseqConfig& cfg) {
    if (!cfg.has_refseq) {
        throw std::invalid_argument(
            "compute_refseq_distances requires a valid reference sequence");
    }
}

inline void validate_distinct_successive_states(
    const py::array& sequences,
    const py::array& seqlength) {
    auto seq_arr = py::array_t<int>(sequences);
    auto len_arr = py::array_t<int>(seqlength);
    auto ptr_seq = seq_arr.unchecked<2>();
    auto ptr_len = len_arr.unchecked<1>();

    for (py::ssize_t i = 0; i < seqlength.shape(0); ++i) {
        for (int k = 1; k < ptr_len(i); ++k) {
            if (ptr_seq(i, k) == ptr_seq(i, k - 1)) {
                throw std::invalid_argument(
                    "active spell states must be distinct from adjacent spell states");
            }
        }
    }
}

inline void validate_spell_distance_inputs(
    const py::array& sequences,
    const py::array& durations,
    const py::array& seqlength,
    bool require_canonical_spells = true) {
    require_2d(sequences, "sequences");
    require_2d(durations, "durations");
    require_1d(seqlength, "seqlength");
    if (sequences.shape(0) != durations.shape(0)) {
        throw std::invalid_argument("sequences and durations must have the same number of rows");
    }
    if (durations.shape(1) < sequences.shape(1)) {
        throw std::invalid_argument(
            "durations must have at least as many columns as sequences");
    }
    if (seqlength.shape(0) != sequences.shape(0)) {
        throw std::invalid_argument("seqlength length must match number of sequences");
    }

    auto len_arr = py::array_t<int>(seqlength);
    auto dur_arr = py::array_t<double>(durations);
    auto ptr_len = len_arr.unchecked<1>();
    auto ptr_dur = dur_arr.unchecked<2>();

    const int ncols = static_cast<int>(sequences.shape(1));
    for (py::ssize_t i = 0; i < seqlength.shape(0); ++i) {
        const int li = ptr_len(i);
        if (li < 0 || li > ncols) {
            throw std::invalid_argument(
                "seqlength values must satisfy 0 <= seqlength[i] <= n_columns");
        }
        for (int k = 0; k < li; ++k) {
            const double d = ptr_dur(i, k);
            if (!std::isfinite(d) || d <= 0.0) {
                throw std::invalid_argument(
                    "active spell durations must be finite and strictly positive");
            }
        }
    }

    if (require_canonical_spells) {
        validate_distinct_successive_states(sequences, seqlength);
    }
}

inline void validate_totaldur(
    const py::array& totaldur,
    const py::array& durations,
    const py::array& seqlength,
    py::ssize_t nseq) {
    require_1d(totaldur, "totaldur");
    if (totaldur.shape(0) != nseq) {
        throw std::invalid_argument("totaldur length must match number of sequences");
    }

    auto total_arr = py::array_t<double>(totaldur);
    auto dur_arr = py::array_t<double>(durations);
    auto len_arr = py::array_t<int>(seqlength);

    auto ptr_total = total_arr.unchecked<1>();
    auto ptr_dur = dur_arr.unchecked<2>();
    auto ptr_len = len_arr.unchecked<1>();

    for (py::ssize_t i = 0; i < nseq; ++i) {
        const double supplied = ptr_total(i);
        if (!std::isfinite(supplied) || supplied < 0.0) {
            throw std::invalid_argument("totaldur must be finite and non-negative");
        }

        double expected = 0.0;
        for (int k = 0; k < ptr_len(i); ++k) {
            expected += ptr_dur(i, k);
        }

        if (!std::isfinite(expected)) {
            throw std::invalid_argument(
                "sum of active spell durations must be finite");
        }

        const double tol = TOTALDUR_REL_TOL *
            std::max({1.0, std::fabs(supplied), std::fabs(expected)});
        if (std::fabs(supplied - expected) > tol) {
            throw std::invalid_argument(
                "totaldur must equal the sum of active spell durations");
        }
    }
}

inline void validate_lcpspell_maxdist_duration_ref(
    const py::array_t<double>& seqdur,
    const py::array_t<int>& seqlength,
    int norm,
    double timecost,
    double duration_ref) {
    if (norm != 3 || timecost <= 0.0) {
        return;
    }

    auto dur_arr = seqdur;
    auto len_arr = seqlength;
    auto ptr_dur = dur_arr.unchecked<2>();
    auto ptr_len = len_arr.unchecked<1>();

    double max_obs_dur = 0.0;
    for (py::ssize_t i = 0; i < seqdur.shape(0); ++i) {
        for (int k = 0; k < ptr_len(i); ++k) {
            max_obs_dur = std::max(max_obs_dur, ptr_dur(i, k));
        }
    }

    if (max_obs_dur > duration_ref) {
        throw std::invalid_argument(
            "For normalized LCPspell/RLCPspell, duration_ref must be at least "
            "the largest active spell duration");
    }
}

inline void validate_lcpspell_inputs(
    const py::array& sequences,
    const py::array& seqdur,
    const py::array& seqlength,
    int norm,
    double timecost,
    double duration_ref,
    int sign) {
    require_lcpspell_norm(norm);
    validate_spell_distance_inputs(sequences, seqdur, seqlength, true);
    require_sign(sign);
    if (!std::isfinite(timecost) || timecost < 0.0) {
        throw std::invalid_argument("timecost must be finite and non-negative");
    }
    if (!std::isfinite(duration_ref) || duration_ref <= 0.0) {
        throw std::invalid_argument("duration_ref must be finite and positive");
    }

    auto dur_typed = py::array_t<double>(seqdur);
    auto len_typed = py::array_t<int>(seqlength);
    validate_lcpspell_maxdist_duration_ref(
        dur_typed, len_typed, norm, timecost, duration_ref);
}

}  // namespace lcp_input

#endif
