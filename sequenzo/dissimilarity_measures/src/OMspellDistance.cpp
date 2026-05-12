/*
 * OMspellDistance: Optimal Matching on spell sequences (duration-weighted).
 *
 * Optimizations vs original:
 *   [OPT-1] Removed pseudo-SIMD. xsimd batch loaded B cells for del/sub but insertion
 *           depends on curr[j-1] → sequential fixup negated SIMD benefit.
 *   [OPT-2] Manual 3-way min replaces std::min({a,b,c}) (avoids initializer_list heap alloc).
 *   [OPT-3] Cache dur_j once per j-iteration (original tail loop called ptr_dur(js,j-1) twice).
 *
 * @Author  : Yuqi Liang 梁彧祺, Yapeng Wei 卫亚鹏
 * @File    : OMspellDistance.cpp
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <iostream>
#include "utils.h"
#include "dp_utils.h"
#ifdef _OPENMP
    #include <omp.h>
#endif

namespace py = pybind11;

class OMspellDistance {
public:
    OMspellDistance(py::array_t<int> sequences, py::array_t<double> sm, double indel, int norm, py::array_t<int> refseqS,
                    double timecost, py::array_t<double> seqdur, py::array_t<double> indellist, py::array_t<int> seqlength,
                    py::array_t<int> norm_seqlength)
            : indel(indel), norm(norm), timecost(timecost) {

        py::print("[>] Starting Optimal Matching with spell(OMspell)...");
        std::cout << std::flush;

        try {
            this->sequences = sequences;
            this->sm = sm;
            this->seqdur = seqdur;
            this->indellist = indellist;
            this->seqlength = seqlength;
            this->norm_seqlength = norm_seqlength;

            auto seq_shape = sequences.shape();
            nseq = seq_shape[0];
            len = seq_shape[1];
            seq_ptr = sequences.data();
            seq_cols = static_cast<int>(sequences.shape(1));

            dist_matrix = py::array_t<double>({nseq, nseq});
            fmatsize = len + 1;

            auto sm_shape = sm.shape();
            alphasize = sm_shape[0];
            sm_ptr = sm.data();
            sm_cols = static_cast<int>(sm.shape(1));
            dur_ptr = seqdur.data();
            dur_cols = static_cast<int>(seqdur.shape(1));
            indel_ptr = indellist.data();
            len_ptr = seqlength.data();
            norm_len_ptr = norm_seqlength.data();

            cost_cols = seq_cols;
            dur_cost_buf.resize(static_cast<size_t>(nseq) * static_cast<size_t>(cost_cols));
            indel_dur_cost_buf.resize(static_cast<size_t>(nseq) * static_cast<size_t>(cost_cols));
            for (int row = 0; row < nseq; row++) {
                const int* seq_row = seq_ptr + static_cast<ptrdiff_t>(row) * seq_cols;
                const double* dur_row = dur_ptr + static_cast<ptrdiff_t>(row) * dur_cols;
                double* dur_cost_row = dur_cost_buf.data() + static_cast<ptrdiff_t>(row) * cost_cols;
                double* indel_dur_cost_row = indel_dur_cost_buf.data() + static_cast<ptrdiff_t>(row) * cost_cols;
                for (int col = 0; col < cost_cols; col++) {
                    const int state = seq_row[col];
                    const double dur_cost = timecost * dur_row[col];
                    dur_cost_row[col] = dur_cost;
                    indel_dur_cost_row[col] = indel_ptr[state] + dur_cost;
                }
            }
            dur_cost_ptr = dur_cost_buf.data();
            indel_dur_cost_ptr = indel_dur_cost_buf.data();

            mismatch_sub_dominated = true;
            for (int i = 1; i < alphasize; i++) {
                for (int j = 1; j < alphasize; j++) {
                    if (i == j) continue;
                    const double sub_base = sm_ptr[static_cast<ptrdiff_t>(i) * sm_cols + j];
                    if (sub_base + 1e-12 < indel_ptr[i] + indel_ptr[j]) {
                        mismatch_sub_dominated = false;
                        break;
                    }
                }
                if (!mismatch_sub_dominated) break;
            }

            auto ptr = sm.mutable_unchecked<2>();
            if (norm == 4) {
                maxscost = 2 * indel;
            } else {
                for (int i = 0; i < alphasize; i++)
                    for (int j = i + 1; j < alphasize; j++)
                        if (ptr(i, j) > maxscost)
                            maxscost = ptr(i, j);
                maxscost = std::min(maxscost, 2 * indel);
            }

            nans = nseq;
            rseq1 = refseqS.at(0);
            rseq2 = refseqS.at(1);
            if (rseq1 < rseq2) {
                nseq = rseq1;
                nans = nseq * (rseq2 - rseq1);
            } else {
                rseq1 = rseq1 - 1;
            }
            refdist_matrix = py::array_t<double>({nseq, (rseq2 - rseq1)});
        } catch (const std::exception& e) {
            py::print("Error in constructor: ", e.what());
            throw;
        }
    }

    double getIndel(int i, int j, int state) {
        auto ptr_indel = indellist.mutable_unchecked<1>();
        auto ptr_dur = seqdur.mutable_unchecked<2>();
        return ptr_indel(state) + timecost * ptr_dur(i, j);
    }

    double getSubCost(int i_state, int j_state, int i_x, int i_y, int j_x, int j_y) {
        auto ptr_dur = seqdur.mutable_unchecked<2>();
        if (i_state == j_state) {
            return abs(timecost * (ptr_dur(i_x, i_y) - ptr_dur(j_x, j_y)));
        }
        auto ptr_sm = sm.mutable_unchecked<2>();
        return ptr_sm(i_state, j_state) + (ptr_dur(i_x, i_y) + ptr_dur(j_x, j_y)) * timecost;
    }

    double compute_distance(int is, int js, double* prev, double* curr) {
        try {
            const int* seq_i = seq_ptr + static_cast<ptrdiff_t>(is) * seq_cols;
            const int* seq_j = seq_ptr + static_cast<ptrdiff_t>(js) * seq_cols;
            const double* dur_i_cost_row = dur_cost_ptr + static_cast<ptrdiff_t>(is) * cost_cols;
            const double* dur_j_cost_row = dur_cost_ptr + static_cast<ptrdiff_t>(js) * cost_cols;
            const double* indel_i_cost_row = indel_dur_cost_ptr + static_cast<ptrdiff_t>(is) * cost_cols;
            const double* indel_j_cost_row = indel_dur_cost_ptr + static_cast<ptrdiff_t>(js) * cost_cols;

            int mm = len_ptr[is];
            int nn = len_ptr[js];
            int mSuf = mm + 1;
            int nSuf = nn + 1;

            prev[0] = 0;
            for (int jj = 1; jj < nSuf; jj++) {
                prev[jj] = prev[jj - 1] + indel_j_cost_row[jj - 1];
            }

            // [OPT-1] Pure scalar DP — pseudo-SIMD removed.
            for (int i = 1; i < mSuf; i++) {
                int i_state = seq_i[i - 1];
                double dur_i_cost = dur_i_cost_row[i - 1];
                double del_cost_i = indel_i_cost_row[i - 1];
                const double* sm_i = mismatch_sub_dominated ? nullptr : sm_ptr + static_cast<ptrdiff_t>(i_state) * sm_cols;

                curr[0] = prev[0] + del_cost_i;

                for (int j = 1; j < nSuf; j++) {
                    int j_state = seq_j[j - 1];
                    double dur_j_cost = dur_j_cost_row[j - 1];  // [OPT-3] cached once

                    double del_cost = prev[j] + del_cost_i;
                    double ins_cost = curr[j - 1] + indel_j_cost_row[j - 1];

                    // [OPT-2] Manual 3-way min.
                    double best = del_cost;
                    if (ins_cost < best) best = ins_cost;
                    if (i_state == j_state) {
                        const double sub_cost = prev[j - 1] + std::abs(dur_i_cost - dur_j_cost);
                        if (sub_cost < best) best = sub_cost;
                    } else if (!mismatch_sub_dominated) {
                        const double sub_cost = prev[j - 1] + sm_i[j_state] + dur_i_cost + dur_j_cost;
                        if (sub_cost < best) best = sub_cost;
                    }
                    curr[j] = best;
                }
                std::swap(prev, curr);
            }

            int mm_norm = norm_len_ptr[is];
            int nn_norm = norm_len_ptr[js];
            double maxpossiblecost = std::abs(nn_norm - mm_norm) * indel + maxscost * std::min(mm_norm, nn_norm);
            double ml = double(mm_norm) * indel;
            double nl = double(nn_norm) * indel;
            return normalize_distance(prev[nSuf - 1], maxpossiblecost, ml, nl, norm);
        } catch (const std::exception& e) {
            py::print("Error in compute_distance: ", e.what());
            throw;
        }
    }

    double compute_distance_full_fmat(int is, int js, double* fmat) {
        try {
            const int* seq_i = seq_ptr + static_cast<ptrdiff_t>(is) * seq_cols;
            const int* seq_j = seq_ptr + static_cast<ptrdiff_t>(js) * seq_cols;
            const double* dur_i_cost_row = dur_cost_ptr + static_cast<ptrdiff_t>(is) * cost_cols;
            const double* dur_j_cost_row = dur_cost_ptr + static_cast<ptrdiff_t>(js) * cost_cols;
            const double* indel_i_cost_row = indel_dur_cost_ptr + static_cast<ptrdiff_t>(is) * cost_cols;
            const double* indel_j_cost_row = indel_dur_cost_ptr + static_cast<ptrdiff_t>(js) * cost_cols;

            int mm = len_ptr[is];
            int nn = len_ptr[js];
            int mSuf = mm + 1;
            int nSuf = nn + 1;

            fmat[0] = 0.0;
            for (int i = 1; i < mSuf; i++) {
                fmat[i] = fmat[i - 1] + indel_i_cost_row[i - 1];
            }
            for (int j = 1; j < nSuf; j++) {
                fmat[static_cast<ptrdiff_t>(j) * fmatsize] =
                    fmat[static_cast<ptrdiff_t>(j - 1) * fmatsize] +
                    indel_j_cost_row[j - 1];
            }

            for (int j = 1; j < nSuf; j++) {
                int j_state = seq_j[j - 1];
                double dur_j_cost = dur_j_cost_row[j - 1];
                double ins_cost_j = indel_j_cost_row[j - 1];
                const ptrdiff_t col = static_cast<ptrdiff_t>(j) * fmatsize;
                const ptrdiff_t prev_col = static_cast<ptrdiff_t>(j - 1) * fmatsize;

                for (int i = 1; i < mSuf; i++) {
                    int i_state = seq_i[i - 1];
                    double dur_i_cost = dur_i_cost_row[i - 1];
                    double del_cost_i = indel_i_cost_row[i - 1];
                    const double* sm_i = mismatch_sub_dominated ? nullptr : sm_ptr + static_cast<ptrdiff_t>(i_state) * sm_cols;

                    double del_cost = fmat[i + prev_col] + ins_cost_j;
                    double ins_cost = fmat[(i - 1) + col] + del_cost_i;

                    double best = del_cost;
                    if (ins_cost < best) best = ins_cost;
                    if (i_state == j_state) {
                        const double sub_cost = fmat[(i - 1) + prev_col] + std::abs(dur_i_cost - dur_j_cost);
                        if (sub_cost < best) best = sub_cost;
                    } else if (!mismatch_sub_dominated) {
                        const double sub_cost = fmat[(i - 1) + prev_col] + sm_i[j_state] + dur_i_cost + dur_j_cost;
                        if (sub_cost < best) best = sub_cost;
                    }
                    fmat[i + col] = best;
                }
            }

            int mm_norm = norm_len_ptr[is];
            int nn_norm = norm_len_ptr[js];
            double maxpossiblecost = std::abs(nn_norm - mm_norm) * indel + maxscost * std::min(mm_norm, nn_norm);
            double ml = double(mm_norm) * indel;
            double nl = double(nn_norm) * indel;
            return normalize_distance(fmat[(mSuf - 1) + static_cast<ptrdiff_t>(nSuf - 1) * fmatsize],
                                      maxpossiblecost, ml, nl, norm);
        } catch (const std::exception& e) {
            py::print("Error in compute_distance_full_fmat: ", e.what());
            throw;
        }
    }

    double compute_distance_column_rolling(int is, int js, double* prev, double* curr) {
        try {
            const int* seq_i = seq_ptr + static_cast<ptrdiff_t>(is) * seq_cols;
            const int* seq_j = seq_ptr + static_cast<ptrdiff_t>(js) * seq_cols;
            const double* dur_i_cost_row = dur_cost_ptr + static_cast<ptrdiff_t>(is) * cost_cols;
            const double* dur_j_cost_row = dur_cost_ptr + static_cast<ptrdiff_t>(js) * cost_cols;
            const double* indel_i_cost_row = indel_dur_cost_ptr + static_cast<ptrdiff_t>(is) * cost_cols;
            const double* indel_j_cost_row = indel_dur_cost_ptr + static_cast<ptrdiff_t>(js) * cost_cols;

            int mm = len_ptr[is];
            int nn = len_ptr[js];
            int mSuf = mm + 1;
            int nSuf = nn + 1;

            prev[0] = 0.0;
            for (int i = 1; i < mSuf; i++) {
                prev[i] = prev[i - 1] + indel_i_cost_row[i - 1];
            }

            for (int j = 1; j < nSuf; j++) {
                int j_state = seq_j[j - 1];
                double dur_j_cost = dur_j_cost_row[j - 1];
                double ins_cost_j = indel_j_cost_row[j - 1];

                curr[0] = prev[0] + ins_cost_j;

                for (int i = 1; i < mSuf; i++) {
                    int i_state = seq_i[i - 1];
                    double dur_i_cost = dur_i_cost_row[i - 1];
                    double del_cost_i = indel_i_cost_row[i - 1];
                    const double* sm_i = mismatch_sub_dominated ? nullptr : sm_ptr + static_cast<ptrdiff_t>(i_state) * sm_cols;

                    double ins_cost = prev[i] + ins_cost_j;
                    double del_cost = curr[i - 1] + del_cost_i;

                    double best = ins_cost;
                    if (del_cost < best) best = del_cost;
                    if (i_state == j_state) {
                        const double sub_cost = prev[i - 1] + std::abs(dur_i_cost - dur_j_cost);
                        if (sub_cost < best) best = sub_cost;
                    } else if (!mismatch_sub_dominated) {
                        const double sub_cost = prev[i - 1] + sm_i[j_state] + dur_i_cost + dur_j_cost;
                        if (sub_cost < best) best = sub_cost;
                    }
                    curr[i] = best;
                }
                std::swap(prev, curr);
            }

            int mm_norm = norm_len_ptr[is];
            int nn_norm = norm_len_ptr[js];
            double maxpossiblecost = std::abs(nn_norm - mm_norm) * indel + maxscost * std::min(mm_norm, nn_norm);
            double ml = double(mm_norm) * indel;
            double nl = double(nn_norm) * indel;
            return normalize_distance(prev[mSuf - 1], maxpossiblecost, ml, nl, norm);
        } catch (const std::exception& e) {
            py::print("Error in compute_distance_column_rolling: ", e.what());
            throw;
        }
    }

    double compute_distance_column_rolling_dominated(int is, int js, double* prev, double* curr) {
        try {
            const int* seq_i = seq_ptr + static_cast<ptrdiff_t>(is) * seq_cols;
            const int* seq_j = seq_ptr + static_cast<ptrdiff_t>(js) * seq_cols;
            const double* dur_i_cost_row = dur_cost_ptr + static_cast<ptrdiff_t>(is) * cost_cols;
            const double* dur_j_cost_row = dur_cost_ptr + static_cast<ptrdiff_t>(js) * cost_cols;
            const double* indel_i_cost_row = indel_dur_cost_ptr + static_cast<ptrdiff_t>(is) * cost_cols;
            const double* indel_j_cost_row = indel_dur_cost_ptr + static_cast<ptrdiff_t>(js) * cost_cols;

            int mm = len_ptr[is];
            int nn = len_ptr[js];
            int mSuf = mm + 1;
            int nSuf = nn + 1;

            prev[0] = 0.0;
            for (int i = 1; i < mSuf; i++) {
                prev[i] = prev[i - 1] + indel_i_cost_row[i - 1];
            }

            for (int j = 1; j < nSuf; j++) {
                const int j_state = seq_j[j - 1];
                const double dur_j_cost = dur_j_cost_row[j - 1];
                const double ins_cost_j = indel_j_cost_row[j - 1];

                curr[0] = prev[0] + ins_cost_j;

                for (int i = 1; i < mSuf; i++) {
                    const int i_state = seq_i[i - 1];
                    const double ins_cost = prev[i] + ins_cost_j;
                    const double del_cost = curr[i - 1] + indel_i_cost_row[i - 1];

                    double best = ins_cost;
                    if (del_cost < best) best = del_cost;

                    if (i_state == j_state) {
                        const double sub_cost = prev[i - 1] + std::abs(dur_i_cost_row[i - 1] - dur_j_cost);
                        if (sub_cost < best) best = sub_cost;
                    }

                    curr[i] = best;
                }
                std::swap(prev, curr);
            }

            int mm_norm = norm_len_ptr[is];
            int nn_norm = norm_len_ptr[js];
            double maxpossiblecost = std::abs(nn_norm - mm_norm) * indel + maxscost * std::min(mm_norm, nn_norm);
            double ml = double(mm_norm) * indel;
            double nl = double(nn_norm) * indel;
            return normalize_distance(prev[mSuf - 1], maxpossiblecost, ml, nl, norm);
        } catch (const std::exception& e) {
            py::print("Error in compute_distance_column_rolling_dominated: ", e.what());
            throw;
        }
    }

    double compute_distance_column_rolling_dominated_low_state(int is, int js, double* prev, double* curr) {
        try {
            const int* seq_i = seq_ptr + static_cast<ptrdiff_t>(is) * seq_cols;
            const int* seq_j = seq_ptr + static_cast<ptrdiff_t>(js) * seq_cols;
            const double* dur_i_cost_row = dur_cost_ptr + static_cast<ptrdiff_t>(is) * cost_cols;
            const double* dur_j_cost_row = dur_cost_ptr + static_cast<ptrdiff_t>(js) * cost_cols;
            const double* indel_i_cost_row = indel_dur_cost_ptr + static_cast<ptrdiff_t>(is) * cost_cols;
            const double* indel_j_cost_row = indel_dur_cost_ptr + static_cast<ptrdiff_t>(js) * cost_cols;

            int mm = len_ptr[is];
            int nn = len_ptr[js];
            int mSuf = mm + 1;
            int nSuf = nn + 1;

            prev[0] = 0.0;
            for (int i = 1; i < mSuf; i++) {
                prev[i] = prev[i - 1] + indel_i_cost_row[i - 1];
            }

            for (int j = 1; j < nSuf; j++) {
                const int j_state = seq_j[j - 1];
                const double dur_j_cost = dur_j_cost_row[j - 1];
                const double ins_cost_j = indel_j_cost_row[j - 1];

                curr[0] = prev[0] + ins_cost_j;

                for (int i = 1; i < mSuf; i++) {
                    const double ins_cost = prev[i] + ins_cost_j;
                    const double del_cost = curr[i - 1] + indel_i_cost_row[i - 1];
                    double best = ins_cost;
                    if (del_cost < best) best = del_cost;

                    const double sub_cost = prev[i - 1] + std::abs(dur_i_cost_row[i - 1] - dur_j_cost);
                    const bool take_sub = (seq_i[i - 1] == j_state) && (sub_cost < best);
                    best = take_sub ? sub_cost : best;

                    curr[i] = best;
                }
                std::swap(prev, curr);
            }

            int mm_norm = norm_len_ptr[is];
            int nn_norm = norm_len_ptr[js];
            double maxpossiblecost = std::abs(nn_norm - mm_norm) * indel + maxscost * std::min(mm_norm, nn_norm);
            double ml = double(mm_norm) * indel;
            double nl = double(nn_norm) * indel;
            return normalize_distance(prev[mSuf - 1], maxpossiblecost, ml, nl, norm);
        } catch (const std::exception& e) {
            py::print("Error in compute_distance_column_rolling_dominated_low_state: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_all_distances() {
        try {
            if (mismatch_sub_dominated) {
                return dp_utils::compute_all_distances(
                    nseq, fmatsize, dist_matrix,
                    [this](int i, int j, double* prev, double* curr) {
                        return this->compute_distance_column_rolling_dominated(i, j, prev, curr);
                    });
            }
            return dp_utils::compute_all_distances(
                nseq, fmatsize, dist_matrix,
                [this](int i, int j, double* prev, double* curr) {
                    return this->compute_distance_column_rolling(i, j, prev, curr);
                });
        } catch (const std::exception& e) {
            py::print("Error in compute_all_distances: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_all_distances_single_thread() {
        try {
            auto buffer = dist_matrix.mutable_unchecked<2>();
            double* prev = dp_utils::aligned_alloc_double(static_cast<size_t>(fmatsize));
            double* curr = dp_utils::aligned_alloc_double(static_cast<size_t>(fmatsize));

            for (int i = 0; i < nseq; i++) {
                buffer(i, i) = 0.0;
                for (int j = i + 1; j < nseq; j++) {
                    const double dist = mismatch_sub_dominated
                        ? compute_distance_column_rolling_dominated(i, j, prev, curr)
                        : compute_distance_column_rolling(i, j, prev, curr);
                    buffer(i, j) = dist;
                    buffer(j, i) = dist;
                }
            }

            dp_utils::aligned_free_double(prev);
            dp_utils::aligned_free_double(curr);
            return dist_matrix;
        } catch (const std::exception& e) {
            py::print("Error in compute_all_distances_single_thread: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_all_distances_serial() {
        try {
            auto serial_matrix = py::array_t<double>({nseq, nseq});
            auto buffer = serial_matrix.mutable_unchecked<2>();
            double* prev = dp_utils::aligned_alloc_double(static_cast<size_t>(fmatsize));
            double* curr = dp_utils::aligned_alloc_double(static_cast<size_t>(fmatsize));

            for (int i = 0; i < nseq; i++) {
                buffer(i, i) = 0.0;
                for (int j = i + 1; j < nseq; j++) {
                    buffer(i, j) = compute_distance(i, j, prev, curr);
                }
            }

            for (int i = 0; i < nseq; i++) {
                for (int j = i + 1; j < nseq; j++) {
                    buffer(j, i) = buffer(i, j);
                }
            }

            dp_utils::aligned_free_double(prev);
            dp_utils::aligned_free_double(curr);
            return serial_matrix;
        } catch (const std::exception& e) {
            py::print("Error in compute_all_distances_serial: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_unique_condensed_serial() {
        try {
            const long long condensed_len = static_cast<long long>(nseq) * (nseq - 1) / 2;
            auto condensed = py::array_t<double>(condensed_len);
            double* out = static_cast<double*>(condensed.request().ptr);
            double* prev = dp_utils::aligned_alloc_double(static_cast<size_t>(fmatsize));
            double* curr = dp_utils::aligned_alloc_double(static_cast<size_t>(fmatsize));

            for (int i = 0; i < nseq; i++) {
                const long long row_start = static_cast<long long>(i) * (2 * nseq - i - 1) / 2;
                for (int j = i + 1; j < nseq; j++) {
                    out[row_start + (j - i - 1)] = compute_distance(i, j, prev, curr);
                }
            }

            dp_utils::aligned_free_double(prev);
            dp_utils::aligned_free_double(curr);
            return condensed;
        } catch (const std::exception& e) {
            py::print("Error in compute_unique_condensed_serial: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_unique_condensed_column_rolling_serial() {
        try {
            const long long condensed_len = static_cast<long long>(nseq) * (nseq - 1) / 2;
            auto condensed = py::array_t<double>(condensed_len);
            double* out = static_cast<double*>(condensed.request().ptr);
            double* prev = dp_utils::aligned_alloc_double(static_cast<size_t>(fmatsize));
            double* curr = dp_utils::aligned_alloc_double(static_cast<size_t>(fmatsize));

            for (int i = 0; i < nseq; i++) {
                const long long row_start = static_cast<long long>(i) * (2 * nseq - i - 1) / 2;
                for (int j = i + 1; j < nseq; j++) {
                    out[row_start + (j - i - 1)] = compute_distance_column_rolling(i, j, prev, curr);
                }
            }

            dp_utils::aligned_free_double(prev);
            dp_utils::aligned_free_double(curr);
            return condensed;
        } catch (const std::exception& e) {
            py::print("Error in compute_unique_condensed_column_rolling_serial: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_unique_condensed_full_fmat_serial() {
        try {
            const long long condensed_len = static_cast<long long>(nseq) * (nseq - 1) / 2;
            auto condensed = py::array_t<double>(condensed_len);
            double* out = static_cast<double*>(condensed.request().ptr);
            double* fmat = dp_utils::aligned_alloc_double(static_cast<size_t>(fmatsize) * static_cast<size_t>(fmatsize));

            for (int i = 0; i < nseq; i++) {
                const long long row_start = static_cast<long long>(i) * (2 * nseq - i - 1) / 2;
                for (int j = i + 1; j < nseq; j++) {
                    out[row_start + (j - i - 1)] = compute_distance_full_fmat(i, j, fmat);
                }
            }

            dp_utils::aligned_free_double(fmat);
            return condensed;
        } catch (const std::exception& e) {
            py::print("Error in compute_unique_condensed_full_fmat_serial: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_original_condensed_full_fmat(py::array_t<int> original_didxs, int original_nseq) {
        try {
            const long long unique_len = static_cast<long long>(nseq) * (nseq - 1) / 2;
            auto unique_condensed = py::array_t<double>(unique_len);
            double* unique_out = static_cast<double*>(unique_condensed.request().ptr);

            #pragma omp parallel
            {
                double* fmat = dp_utils::aligned_alloc_double(static_cast<size_t>(fmatsize) * static_cast<size_t>(fmatsize));

                #pragma omp for schedule(dynamic, 16)
                for (int i = 0; i < nseq; i++) {
                    const long long row_start = static_cast<long long>(i) * (2 * nseq - i - 1) / 2;
                    for (int j = i + 1; j < nseq; j++) {
                        unique_out[row_start + (j - i - 1)] = compute_distance_full_fmat(i, j, fmat);
                    }
                }

                dp_utils::aligned_free_double(fmat);
            }

            const long long original_len = static_cast<long long>(original_nseq) * (original_nseq - 1) / 2;
            auto original_condensed = py::array_t<double>(original_len);
            double* original_out = static_cast<double*>(original_condensed.request().ptr);
            const int* didxs = original_didxs.data();

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < original_nseq; i++) {
                const int ui = didxs[i];
                const long long out_row_start = static_cast<long long>(i) * (2 * original_nseq - i - 1) / 2;
                for (int j = i + 1; j < original_nseq; j++) {
                    const int uj = didxs[j];
                    if (ui == uj) {
                        original_out[out_row_start + (j - i - 1)] = 0.0;
                    } else {
                        const int a = ui < uj ? ui : uj;
                        const int b = ui < uj ? uj : ui;
                        const long long unique_idx = static_cast<long long>(a) * (2 * nseq - a - 1) / 2 + (b - a - 1);
                        original_out[out_row_start + (j - i - 1)] = unique_out[unique_idx];
                    }
                }
            }

            return original_condensed;
        } catch (const std::exception& e) {
            py::print("Error in compute_original_condensed_full_fmat: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_original_condensed_column_rolling(py::array_t<int> original_didxs, int original_nseq) {
        try {
            const long long unique_len = static_cast<long long>(nseq) * (nseq - 1) / 2;
            auto unique_condensed = py::array_t<double>(unique_len);
            double* unique_out = static_cast<double*>(unique_condensed.request().ptr);

            #pragma omp parallel
            {
                double* prev = dp_utils::aligned_alloc_double(static_cast<size_t>(fmatsize));
                double* curr = dp_utils::aligned_alloc_double(static_cast<size_t>(fmatsize));

                #pragma omp for schedule(dynamic, 16)
                for (int i = 0; i < nseq; i++) {
                    const long long row_start = static_cast<long long>(i) * (2 * nseq - i - 1) / 2;
                    for (int j = i + 1; j < nseq; j++) {
                        unique_out[row_start + (j - i - 1)] = mismatch_sub_dominated
                            ? compute_distance_column_rolling_dominated(i, j, prev, curr)
                            : compute_distance_column_rolling(i, j, prev, curr);
                    }
                }

                dp_utils::aligned_free_double(prev);
                dp_utils::aligned_free_double(curr);
            }

            const long long original_len = static_cast<long long>(original_nseq) * (original_nseq - 1) / 2;
            auto original_condensed = py::array_t<double>(original_len);
            double* original_out = static_cast<double*>(original_condensed.request().ptr);
            const int* didxs = original_didxs.data();

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < original_nseq; i++) {
                const int ui = didxs[i];
                const long long out_row_start = static_cast<long long>(i) * (2 * original_nseq - i - 1) / 2;
                for (int j = i + 1; j < original_nseq; j++) {
                    const int uj = didxs[j];
                    if (ui == uj) {
                        original_out[out_row_start + (j - i - 1)] = 0.0;
                    } else {
                        const int a = ui < uj ? ui : uj;
                        const int b = ui < uj ? uj : ui;
                        const long long unique_idx = static_cast<long long>(a) * (2 * nseq - a - 1) / 2 + (b - a - 1);
                        original_out[out_row_start + (j - i - 1)] = unique_out[unique_idx];
                    }
                }
            }

            return original_condensed;
        } catch (const std::exception& e) {
            py::print("Error in compute_original_condensed_column_rolling: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_refseq_distances() {
        try {
            auto buffer = refdist_matrix.mutable_unchecked<2>();
            #pragma omp parallel
            {
                double* prev = dp_utils::aligned_alloc_double(static_cast<size_t>(fmatsize));
                double* curr = dp_utils::aligned_alloc_double(static_cast<size_t>(fmatsize));
                #pragma omp for schedule(static)
                for (int rseq = rseq1; rseq < rseq2; rseq++) {
                    for (int is = 0; is < nseq; is++) {
                        buffer(is, rseq - rseq1) = (is != rseq)
                            ? (
                                mismatch_sub_dominated
                                ? compute_distance_column_rolling_dominated(is, rseq, prev, curr)
                                : compute_distance_column_rolling(is, rseq, prev, curr)
                              )
                            : 0.0;
                    }
                }
                dp_utils::aligned_free_double(prev);
                dp_utils::aligned_free_double(curr);
            }
            return refdist_matrix;
        } catch (const std::exception& e) {
            py::print("Error in compute_refseq_distances: ", e.what());
            throw;
        }
    }

private:
    py::array_t<int> sequences;
    py::array_t<int> seqlength;
    py::array_t<int> norm_seqlength;
    py::array_t<double> sm;
    double indel;
    int norm;
    int nseq;
    int len;
    int alphasize;
    int fmatsize;
    py::array_t<double> dist_matrix;
    double maxscost = 0.0;
    const int* seq_ptr = nullptr;
    const int* len_ptr = nullptr;
    const int* norm_len_ptr = nullptr;
    const double* sm_ptr = nullptr;
    const double* dur_ptr = nullptr;
    const double* indel_ptr = nullptr;
    int seq_cols = 0;
    int sm_cols = 0;
    int dur_cols = 0;
    int cost_cols = 0;
    std::vector<double> dur_cost_buf;
    std::vector<double> indel_dur_cost_buf;
    const double* dur_cost_ptr = nullptr;
    const double* indel_dur_cost_ptr = nullptr;
    bool mismatch_sub_dominated = false;

    double timecost;
    py::array_t<double> seqdur;
    py::array_t<double> indellist;

    int nans = -1;
    int rseq1 = -1;
    int rseq2 = -1;
    py::array_t<double> refdist_matrix;
};
