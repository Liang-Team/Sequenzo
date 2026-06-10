/*
 * SVRspellDistance: Spell-based distance with duration and soft matching (proximity).
 *
 * Port of TraMineR's NMSDURSoftdistance (seqdist method "SVRspell").
 * Uses spell sequences (state + duration per spell), a softmatch (prox) matrix for
 * state similarity, and kweights for aggregating over subsequence lengths. The distance
 * is the square root of (Ival + Jval - 2*Aval) where Aval is the weighted sum of
 * minimal shared time over common subsequences, and Ival/Jval are the self-terms.
 * TraMineR applies sqrt to the raw value in R. For the standard branch, the
 * C++ implementation returns the raw value and Python applies sqrt for parity.
 *
 * @Author  : Yuqi Liang 梁彧祺
 * @File    : SVRspellDistance.cpp
 * @Time    : 2026/2/6 9:27
 * @Desc    : 
 * Reference: TraMineR src/NMSDURSoftdistance.cpp, NMSdistance.cpp (SUBSEQdistance::distance).
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <exception>
#include <vector>
#include <limits>
#include <iostream>
#include "utils.h"
#include "dp_utils.h"

namespace py = pybind11;

// Row-major index for (i, j) in a matrix with ncols columns
#define IDX(i, j, ncols) ((i) * (ncols) + (j))

class SVRspellDistance {
    struct Workspace {
        std::vector<double> e1;
        std::vector<double> e;
        std::vector<double> t_i;
        std::vector<double> t_j;
        std::vector<double> t_ij;
        std::vector<double> kvect;
        std::vector<int> match_i;
        std::vector<int> match_j;
        std::vector<double> match_ti;
        std::vector<double> match_tj;
        std::vector<double> cur_e;
        std::vector<double> cur_ti;
        std::vector<double> cur_tj;
        std::vector<double> cur_tij;
        std::vector<double> next_e;
        std::vector<double> next_ti;
        std::vector<double> next_tj;
        std::vector<double> next_tij;
        std::vector<double> bit_e;
        std::vector<double> bit_ti;
        std::vector<double> bit_tj;
        std::vector<double> bit_tij;

        explicit Workspace(int maxlen = 0) : kvect(static_cast<size_t>(maxlen), 0.0) {}

        void reset_kvect(int maxlen) {
            if (static_cast<int>(kvect.size()) != maxlen) {
                kvect.resize(static_cast<size_t>(maxlen));
            }
            std::fill(kvect.begin(), kvect.end(), 0.0);
        }

        void reset(int maxlen) {
            const int rowsize = maxlen + 1;
            const size_t matsize = static_cast<size_t>(rowsize) * static_cast<size_t>(rowsize);
            if (e1.size() != matsize) {
                e1.resize(matsize);
                e.resize(matsize);
                t_i.resize(matsize);
                t_j.resize(matsize);
                t_ij.resize(matsize);
            }
            std::fill(e1.begin(), e1.end(), 0.0);
            std::fill(e.begin(), e.end(), 0.0);
            std::fill(t_i.begin(), t_i.end(), 0.0);
            std::fill(t_j.begin(), t_j.end(), 0.0);
            std::fill(t_ij.begin(), t_ij.end(), 0.0);
            if (static_cast<int>(kvect.size()) != maxlen) {
                kvect.resize(static_cast<size_t>(maxlen));
            }
            std::fill(kvect.begin(), kvect.end(), 0.0);
        }

        void prepare_sparse(size_t count, int ncols) {
            match_i.clear();
            match_j.clear();
            match_ti.clear();
            match_tj.clear();
            cur_e.clear();
            cur_ti.clear();
            cur_tj.clear();
            cur_tij.clear();
            match_i.reserve(count);
            match_j.reserve(count);
            match_ti.reserve(count);
            match_tj.reserve(count);
            cur_e.reserve(count);
            cur_ti.reserve(count);
            cur_tj.reserve(count);
            cur_tij.reserve(count);
            next_e.assign(count, 0.0);
            next_ti.assign(count, 0.0);
            next_tj.assign(count, 0.0);
            next_tij.assign(count, 0.0);
            bit_e.assign(static_cast<size_t>(ncols) + 1, 0.0);
            bit_ti.assign(static_cast<size_t>(ncols) + 1, 0.0);
            bit_tj.assign(static_cast<size_t>(ncols) + 1, 0.0);
            bit_tij.assign(static_cast<size_t>(ncols) + 1, 0.0);
        }
    };

public:
    /*
     * Constructor.
     * - sequences: spell states, shape (nseq, max_spells).
     * - seqdur: spell durations, shape (nseq, max_spells).
     * - seqlength: number of spells per sequence, shape (nseq,).
     * - softmatch: proximity matrix (alphasize x alphasize); softmatch[i,j] = similarity between state i and j.
     * - kweights: weights for subsequence lengths, length >= max_spells (extra ignored).
     * - norm: normalization index (TraMineR uses YujianBo/4 for SVRspell when auto).
     * - refseqS: [rseq1, rseq2).
     */
    SVRspellDistance(py::array_t<int> sequences,
                    py::array_t<double> seqdur,
                    py::array_t<int> seqlength,
                    py::array_t<double> softmatch,
                    py::array_t<double> kweights,
                    int norm,
                    py::array_t<int> refseqS)
            : norm(norm) {
        py::print("[>] Starting SVRspell (spell duration + soft matching)...");
        std::cout << std::flush;

        try {
            this->sequences = sequences;
            this->seqdur = seqdur;
            this->seqlength = seqlength;
            this->softmatch = softmatch;
            this->kweights = kweights;

            auto seq_shape = sequences.shape();
            nseq = static_cast<int>(seq_shape[0]);
            maxlen = static_cast<int>(seq_shape[1]);
            alphasize = static_cast<int>(softmatch.shape()[0]);
            prox_is_identity = is_identity_prox();

            // kweights length; use at most maxlen
            int kw_len = static_cast<int>(kweights.shape()[0]);
            kweights_len = (kw_len < maxlen) ? kw_len : maxlen;

            // Precompute self-terms: for each sequence is, computeattr(is, is) and store kvect in selfmatvect
            selfmatvect.resize(static_cast<size_t>(nseq) * static_cast<size_t>(maxlen), 0.0);
            std::exception_ptr first_exception;
            std::atomic<bool> has_exception(false);
            const int row_chunk = row_schedule_chunk();

            #pragma omp parallel
            {
                Workspace workspace(maxlen);

                #pragma omp for schedule(dynamic, row_chunk)
                for (int is = 0; is < nseq; is++) {
                    if (has_exception.load()) continue;
                    try {
                        computeattr(is, is, workspace);
                        for (int k = 0; k < maxlen; k++) {
                            selfmatvect[static_cast<size_t>(is) * static_cast<size_t>(maxlen) + static_cast<size_t>(k)] = workspace.kvect[static_cast<size_t>(k)];
                        }
                    } catch (...) {
                        dp_utils::record_first_exception(first_exception, has_exception);
                    }
                }
            }

            if (first_exception) {
                std::rethrow_exception(first_exception);
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
        } catch (const std::exception& e) {
            py::print("Error in SVRspellDistance constructor: ", e.what());
            throw;
        }
    }

    /*
     * NMSDURSoftdistance::computeattr(is, js): fill workspace.kvect with tot_t_ij for k=0,1,...,min(m,n)-1.
     */
    void computeattr(int is, int js, Workspace& work) {
        auto ptr_seq = sequences.unchecked<2>();
        auto ptr_dur = seqdur.unchecked<2>();
        auto ptr_len = seqlength.unchecked<1>();
        auto ptr_sm = softmatch.unchecked<2>();
        int m = ptr_len(is);
        int n = ptr_len(js);
        if (prox_is_identity) {
            computeattr_identity_sparse(is, js, work);
            return;
        }
        work.reset(maxlen);
        if (m == 0 || n == 0) return;

        int mrows = m + 1;
        int ncols = n + 1;
        int rowsize = maxlen + 1;
        auto& e1 = work.e1;
        auto& e = work.e;
        auto& t_i = work.t_i;
        auto& t_j = work.t_j;
        auto& t_ij = work.t_ij;

        double tot_t_ij = 0.0;
        for (int i = 0; i < m; i++) {
            int seqi = ptr_seq(is, i);
            double ti = ptr_dur(is, i);
            for (int j = 0; j < n; j++) {
                int ij = IDX(i, j, rowsize);
                int seqj = ptr_seq(js, j);
                double sf = prox_is_identity ? (seqi == seqj ? 1.0 : 0.0) : ptr_sm(seqi - 1, seqj - 1);
                if (sf == 0.0) continue;
                double tj = ptr_dur(js, j);
                e1[ij] = sf;
                e[ij] = sf;
                t_i[ij] = sf * ti;
                t_j[ij] = sf * tj;
                t_ij[ij] = sf * ti * tj;
                tot_t_ij += t_ij[ij];
                if (tot_t_ij >= std::numeric_limits<double>::max()) {
                    throw std::runtime_error("[!] Number of subsequences is getting too big");
                }
            }
        }

        for (int i = 0; i < m; i++) {
            int ij = IDX(i, ncols - 1, rowsize);
            e1[ij] = e[ij] = t_i[ij] = t_j[ij] = t_ij[ij] = 0.0;
        }
        for (int j = 0; j < ncols; j++) {
            int ij = IDX(mrows - 1, j, rowsize);
            e1[ij] = e[ij] = t_i[ij] = t_j[ij] = t_ij[ij] = 0.0;
        }

        int k = 0;
        work.kvect[static_cast<size_t>(k)] = tot_t_ij;
        if (tot_t_ij == 0.0) return;

        while (mrows != 0 && ncols != 0) {
            k++;
            double sum_e, sum_t_i, sum_t_j, sum_t_ij;
            double temp_e, temp_t_i, temp_t_j, temp_t_ij;

            for (int irow = 0; irow < mrows; irow++) {
                sum_e = sum_t_i = sum_t_j = sum_t_ij = 0.0;
                for (int jcol = ncols - 1; jcol >= 0; jcol--) {
                    int ij = IDX(irow, jcol, rowsize);
                    temp_e = sum_e; sum_e += e[ij]; e[ij] = temp_e;
                    temp_t_i = sum_t_i; sum_t_i += t_i[ij]; t_i[ij] = temp_t_i;
                    temp_t_j = sum_t_j; sum_t_j += t_j[ij]; t_j[ij] = temp_t_j;
                    temp_t_ij = sum_t_ij; sum_t_ij += t_ij[ij]; t_ij[ij] = temp_t_ij;
                }
            }

            double tot_e = 0.0;
            for (int jcol = 0; jcol < ncols; jcol++) {
                sum_e = sum_t_i = sum_t_j = sum_t_ij = 0.0;
                for (int irow = mrows - 1; irow >= 0; irow--) {
                    int ij = IDX(irow, jcol, rowsize);
                    temp_e = sum_e; sum_e += e[ij]; e[ij] = temp_e;
                    temp_t_i = sum_t_i; sum_t_i += t_i[ij]; t_i[ij] = temp_t_i;
                    temp_t_j = sum_t_j; sum_t_j += t_j[ij]; t_j[ij] = temp_t_j;
                    temp_t_ij = sum_t_ij; sum_t_ij += t_ij[ij]; t_ij[ij] = temp_t_ij;
                    tot_e += e[ij];
                }
            }

            if (tot_e == 0.0) return;

            tot_t_ij = 0.0;
            for (int irow = 0; irow < mrows; irow++) {
                double ti = (irow < m) ? ptr_dur(is, irow) : 0.0;
                for (int jcol = 0; jcol < ncols; jcol++) {
                    int ij = IDX(irow, jcol, rowsize);
                    double tj = (jcol < n) ? ptr_dur(js, jcol) : 0.0;
                    double sf = e1[ij];
                    if (sf == 0.0) {
                        e[ij] = t_i[ij] = t_j[ij] = t_ij[ij] = 0.0;
                        continue;
                    }
                    e[ij] = sf * e[ij];
                    t_ij[ij] = sf * (t_ij[ij] + ti * t_j[ij] + tj * t_i[ij] + e[ij] * ti * tj);
                    t_i[ij] = sf * (t_i[ij] + ti * e[ij]);
                    t_j[ij] = sf * (t_j[ij] + tj * e[ij]);
                    tot_t_ij += t_ij[ij];
                }
            }

            if (k < maxlen) work.kvect[static_cast<size_t>(k)] = tot_t_ij;
            if (tot_t_ij >= std::numeric_limits<double>::max()) {
                throw std::runtime_error("[!] Number of subsequences is getting too big");
            }
            mrows--;
            ncols--;
        }
    }

    /*
     * distMethod == 2: raw distance = Ival + Jval - 2*Aval.
     * For the standard branch, return the raw value; Python applies sqrt for parity.
     */
    double compute_distance(int is, int js, Workspace& work) {
        try {
            auto ptr_kw = kweights.unchecked<1>();
            computeattr(is, js, work);

            double Aval = 0.0, Ival = 0.0, Jval = 0.0;
            int minimum = maxlen;
            for (int i = 0; i < minimum; i++) {
                if (i < kweights_len && ptr_kw(i) != 0.0) {
                    Aval += ptr_kw(i) * work.kvect[static_cast<size_t>(i)];
                    Ival += ptr_kw(i) * selfmatvect[static_cast<size_t>(is) * static_cast<size_t>(maxlen) + static_cast<size_t>(i)];
                    Jval += ptr_kw(i) * selfmatvect[static_cast<size_t>(js) * static_cast<size_t>(maxlen) + static_cast<size_t>(i)];
                }
            }

            double dist = Ival + Jval - 2.0 * Aval;
            double maxdist = Ival + Jval;
            if (dist < 0.0) dist = 0.0;  // numerical safety
            // Return pre-square-root distance; Python applies sqrt once, as in TraMineR.
            return normalize_distance(dist, maxdist, Ival, Jval, norm);
        } catch (const std::exception& e) {
            py::print("Error in SVRspellDistance::compute_distance: ", e.what());
            throw;
        }
    }

    double compute_distance(int is, int js) {
        Workspace workspace(maxlen);
        return compute_distance(is, js, workspace);
    }

    py::array_t<double> compute_all_distances() {
        try {
            auto dist_matrix = py::array_t<double>({nseq, nseq});
            auto buffer = dist_matrix.mutable_unchecked<2>();
            std::exception_ptr first_exception;
            std::atomic<bool> has_exception(false);
            const int row_chunk = row_schedule_chunk();

            #pragma omp parallel
            {
                Workspace workspace(maxlen);

                #pragma omp for schedule(dynamic, row_chunk)
                for (int i = 0; i < nseq; i++) {
                    if (has_exception.load()) continue;
                    buffer(i, i) = 0.0;
                    for (int j = i + 1; j < nseq; j++) {
                        if (has_exception.load()) break;
                        try {
                            buffer(i, j) = compute_distance(i, j, workspace);
                        } catch (...) {
                            dp_utils::record_first_exception(first_exception, has_exception);
                            break;
                        }
                    }
                }
            }

            if (first_exception) {
                std::rethrow_exception(first_exception);
            }

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nseq; i++) {
                for (int j = i + 1; j < nseq; j++) {
                    buffer(j, i) = buffer(i, j);
                }
            }
            return dist_matrix;
        } catch (const std::exception& e) {
            py::print("Error in SVRspellDistance::compute_all_distances: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_condensed_distances() {
        try {
            const long long condensed_len = static_cast<long long>(nseq) * (nseq - 1) / 2;
            py::array_t<double> condensed(
                std::array<py::ssize_t, 1>{static_cast<py::ssize_t>(condensed_len)}
            );
            auto buffer = condensed.mutable_unchecked<1>();
            std::exception_ptr first_exception;
            std::atomic<bool> has_exception(false);
            const int row_chunk = row_schedule_chunk();

            #pragma omp parallel
            {
                Workspace workspace(maxlen);

                #pragma omp for schedule(dynamic, row_chunk)
                for (int i = 0; i < nseq - 1; i++) {
                    if (has_exception.load()) continue;
                    for (int j = i + 1; j < nseq; j++) {
                        if (has_exception.load()) break;
                        try {
                            buffer(dp_utils::condensed_index(nseq, i, j)) = compute_distance(i, j, workspace);
                        } catch (...) {
                            dp_utils::record_first_exception(first_exception, has_exception);
                            break;
                        }
                    }
                }
            }

            if (first_exception) {
                std::rethrow_exception(first_exception);
            }

            return condensed;
        } catch (const std::exception& e) {
            py::print("Error in SVRspellDistance::compute_condensed_distances: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_refseq_distances() {
        try {
            auto refdist_matrix = py::array_t<double>({nseq, (rseq2 - rseq1)});
            auto buffer = refdist_matrix.mutable_unchecked<2>();
            const int nref = rseq2 - rseq1;
            const long long total = static_cast<long long>(nseq) * static_cast<long long>(nref);
            std::exception_ptr first_exception;
            std::atomic<bool> has_exception(false);

            #pragma omp parallel
            {
                Workspace workspace(maxlen);

                #pragma omp for schedule(dynamic, 16)
                for (long long idx = 0; idx < total; idx++) {
                    if (has_exception.load()) continue;
                    const int is = static_cast<int>(idx / nref);
                    const int col = static_cast<int>(idx % nref);
                    const int rseq = rseq1 + col;
                    if (is == rseq) {
                        buffer(is, col) = 0.0;
                        continue;
                    }
                    try {
                        buffer(is, col) = (is <= rseq)
                                          ? compute_distance(is, rseq, workspace)
                                          : compute_distance(rseq, is, workspace);
                    } catch (...) {
                        dp_utils::record_first_exception(first_exception, has_exception);
                    }
                }
            }

            if (first_exception) {
                std::rethrow_exception(first_exception);
            }

            return refdist_matrix;
        } catch (const std::exception& e) {
            py::print("Error in SVRspellDistance::compute_refseq_distances: ", e.what());
            throw;
        }
    }

private:
    static void bit_add(std::vector<double>& bit, int index, double value) {
        for (int idx = index + 1; idx < static_cast<int>(bit.size()); idx += idx & -idx) {
            bit[static_cast<size_t>(idx)] += value;
        }
    }

    static double bit_prefix(const std::vector<double>& bit, int index) {
        double total = 0.0;
        for (int idx = index + 1; idx > 0; idx -= idx & -idx) {
            total += bit[static_cast<size_t>(idx)];
        }
        return total;
    }

    static double bit_suffix_gt(const std::vector<double>& bit, int index, double total) {
        return total - bit_prefix(bit, index);
    }

    void computeattr_identity_sparse(int is, int js, Workspace& work) {
        auto ptr_seq = sequences.unchecked<2>();
        auto ptr_dur = seqdur.unchecked<2>();
        auto ptr_len = seqlength.unchecked<1>();
        int m = ptr_len(is);
        int n = ptr_len(js);
        work.reset_kvect(maxlen);
        if (m == 0 || n == 0) return;

        const size_t reserve_count = static_cast<size_t>(m) * static_cast<size_t>(n) / static_cast<size_t>(std::max(1, alphasize)) + 1;
        work.prepare_sparse(reserve_count, n);

        double tot_t_ij = 0.0;
        for (int i = 0; i < m; i++) {
            const int seqi = ptr_seq(is, i);
            const double ti = ptr_dur(is, i);
            for (int j = 0; j < n; j++) {
                if (seqi != ptr_seq(js, j)) continue;
                const double tj = ptr_dur(js, j);
                work.match_i.push_back(i);
                work.match_j.push_back(j);
                work.match_ti.push_back(ti);
                work.match_tj.push_back(tj);
                work.cur_e.push_back(1.0);
                work.cur_ti.push_back(ti);
                work.cur_tj.push_back(tj);
                work.cur_tij.push_back(ti * tj);
                tot_t_ij += ti * tj;
                if (tot_t_ij >= std::numeric_limits<double>::max()) {
                    throw std::runtime_error("[!] Number of subsequences is getting too big");
                }
            }
        }

        const size_t match_count = work.match_i.size();
        if (match_count == 0) return;
        work.next_e.resize(match_count);
        work.next_ti.resize(match_count);
        work.next_tj.resize(match_count);
        work.next_tij.resize(match_count);
        work.kvect[0] = tot_t_ij;

        const int max_k = std::min(std::min(m, n), maxlen);
        for (int k = 1; k < max_k; k++) {
            std::fill(work.next_e.begin(), work.next_e.end(), 0.0);
            std::fill(work.next_ti.begin(), work.next_ti.end(), 0.0);
            std::fill(work.next_tj.begin(), work.next_tj.end(), 0.0);
            std::fill(work.next_tij.begin(), work.next_tij.end(), 0.0);
            std::fill(work.bit_e.begin(), work.bit_e.end(), 0.0);
            std::fill(work.bit_ti.begin(), work.bit_ti.end(), 0.0);
            std::fill(work.bit_tj.begin(), work.bit_tj.end(), 0.0);
            std::fill(work.bit_tij.begin(), work.bit_tij.end(), 0.0);

            double total_e = 0.0;
            double total_ti = 0.0;
            double total_tj = 0.0;
            double total_tij = 0.0;
            tot_t_ij = 0.0;

            int pos = static_cast<int>(match_count) - 1;
            while (pos >= 0) {
                const int current_i = work.match_i[static_cast<size_t>(pos)];
                int group_start = pos;
                while (group_start >= 0 && work.match_i[static_cast<size_t>(group_start)] == current_i) {
                    group_start--;
                }

                for (int idx = group_start + 1; idx <= pos; idx++) {
                    const size_t sidx = static_cast<size_t>(idx);
                    const int j = work.match_j[sidx];
                    const double e_sum = bit_suffix_gt(work.bit_e, j, total_e);
                    const double ti_sum = bit_suffix_gt(work.bit_ti, j, total_ti);
                    const double tj_sum = bit_suffix_gt(work.bit_tj, j, total_tj);
                    const double tij_sum = bit_suffix_gt(work.bit_tij, j, total_tij);
                    if (e_sum == 0.0 && ti_sum == 0.0 && tj_sum == 0.0 && tij_sum == 0.0) continue;

                    const double ti = work.match_ti[sidx];
                    const double tj = work.match_tj[sidx];
                    work.next_e[sidx] = e_sum;
                    work.next_ti[sidx] = ti_sum + ti * e_sum;
                    work.next_tj[sidx] = tj_sum + tj * e_sum;
                    work.next_tij[sidx] = tij_sum + ti * tj_sum + tj * ti_sum + e_sum * ti * tj;
                    tot_t_ij += work.next_tij[sidx];
                    if (tot_t_ij >= std::numeric_limits<double>::max()) {
                        throw std::runtime_error("[!] Number of subsequences is getting too big");
                    }
                }

                for (int idx = group_start + 1; idx <= pos; idx++) {
                    const size_t sidx = static_cast<size_t>(idx);
                    const int j = work.match_j[sidx];
                    const double e = work.cur_e[sidx];
                    const double ti = work.cur_ti[sidx];
                    const double tj = work.cur_tj[sidx];
                    const double tij = work.cur_tij[sidx];
                    if (e != 0.0) {
                        bit_add(work.bit_e, j, e);
                        total_e += e;
                    }
                    if (ti != 0.0) {
                        bit_add(work.bit_ti, j, ti);
                        total_ti += ti;
                    }
                    if (tj != 0.0) {
                        bit_add(work.bit_tj, j, tj);
                        total_tj += tj;
                    }
                    if (tij != 0.0) {
                        bit_add(work.bit_tij, j, tij);
                        total_tij += tij;
                    }
                }

                pos = group_start;
            }

            if (tot_t_ij == 0.0) return;
            work.kvect[static_cast<size_t>(k)] = tot_t_ij;
            work.cur_e.swap(work.next_e);
            work.cur_ti.swap(work.next_ti);
            work.cur_tj.swap(work.next_tj);
            work.cur_tij.swap(work.next_tij);
        }
    }

    bool is_identity_prox() const {
        if (softmatch.ndim() != 2 || softmatch.shape()[0] != softmatch.shape()[1]) return false;
        auto ptr_sm = softmatch.unchecked<2>();
        for (py::ssize_t i = 0; i < softmatch.shape()[0]; i++) {
            for (py::ssize_t j = 0; j < softmatch.shape()[1]; j++) {
                const double expected = (i == j) ? 1.0 : 0.0;
                if (ptr_sm(i, j) != expected) return false;
            }
        }
        return true;
    }

    int row_schedule_chunk() const {
        return nseq >= 2000 ? 4 : 16;
    }

    py::array_t<int> sequences;
    py::array_t<double> seqdur;
    py::array_t<int> seqlength;
    py::array_t<double> softmatch;
    py::array_t<double> kweights;
    int norm;
    int nseq;
    int maxlen;
    int alphasize;
    int kweights_len;
    bool prox_is_identity;
    std::vector<double> selfmatvect;

    int nans;
    int rseq1;
    int rseq2;
};
