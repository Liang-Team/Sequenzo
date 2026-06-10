/*
 * NMSMSTdistance: Number of Matching Subsequences with Minimal Shared Time (spell duration).
 *
 * Port of TraMineR's NMSMSTdistance (seqdist method "NMSMST").
 * Same as NMS but uses spell sequences with durations: e[i,j]=1 if same state else 0,
 * t[i,j]=min(dur_i, dur_j) at matches. kvect[k] = total minimal shared time over common
 * subsequences of length k. Distance formula is the same as NMS (distMethod==2).
 * TraMineR applies sqrt to the result in R; we return the normalized value so Python
 * can apply sqrt to match TraMineR output.
 *
 * @Author  : Yuqi Liang 梁彧祺
 * @File    : NMSMSTdistance.cpp
 * @Time    : 2026/2/7 14:55
 * @Desc    : 
 * Reference: TraMineR src/NMSMSTdistance.cpp, NMSMSTdistance.h.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>
#include <iostream>
#include "utils.h"
#include "dp_utils.h"

namespace py = pybind11;

#define IDX(i, j, ncols) ((i) * (ncols) + (j))

class NMSMSTdistance {
public:
    struct Workspace {
        std::vector<double> e1;
        std::vector<double> e;
        std::vector<double> t1;
        std::vector<double> t;
        std::vector<double> kvect;
    };

    /*
     * Constructor.
     * - sequences: spell states, shape (nseq, max_spells).
     * - seqdur: spell durations, shape (nseq, max_spells).
     * - seqlength: number of spells per sequence, shape (nseq,).
     * - kweights: weights for subsequence lengths.
     * - norm: normalization index.
     * - refseqS: [rseq1, rseq2).
     */
    NMSMSTdistance(py::array_t<int> sequences,
                   py::array_t<double> seqdur,
                   py::array_t<int> seqlength,
                   py::array_t<double> kweights,
                   int norm,
                   py::array_t<int> refseqS)
            : norm(norm) {
        py::print("[>] Starting NMSMST (NMS with Minimal Shared Time)...");
        std::cout << std::flush;

        try {
            this->sequences = sequences;
            this->seqdur = seqdur;
            this->seqlength = seqlength;
            this->kweights = kweights;

            auto seq_shape = sequences.shape();
            nseq = static_cast<int>(seq_shape[0]);
            maxlen = static_cast<int>(seq_shape[1]);

            int kw_len = static_cast<int>(kweights.shape()[0]);
            kweights_len = (kw_len < maxlen) ? kw_len : maxlen;

            rowsize = maxlen + 1;
            selfmatvect.resize(static_cast<size_t>(nseq) * static_cast<size_t>(maxlen), 0.0);
            std::exception_ptr first_exception;
            std::atomic<bool> has_exception(false);

            #pragma omp parallel
            {
                Workspace workspace;
                prepare_workspace(workspace);

                #pragma omp for schedule(dynamic, 16)
                for (int is = 0; is < nseq; is++) {
                    if (has_exception.load()) continue;
                    try {
                        reset_kvect(workspace);
                        computeattr(is, is, workspace);
                        for (int k = 0; k < maxlen; k++)
                            selfmatvect[static_cast<size_t>(is) * static_cast<size_t>(maxlen) + static_cast<size_t>(k)] = workspace.kvect[static_cast<size_t>(k)];
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
            py::print("Error in NMSMSTdistance constructor: ", e.what());
            throw;
        }
    }

    void prepare_workspace(Workspace& workspace) const {
        const size_t matsize = static_cast<size_t>(rowsize) * static_cast<size_t>(rowsize);
        const size_t vecsize = static_cast<size_t>(maxlen);
        if (workspace.e1.size() != matsize) workspace.e1.resize(matsize, 0.0);
        if (workspace.e.size() != matsize) workspace.e.resize(matsize, 0.0);
        if (workspace.t1.size() != matsize) workspace.t1.resize(matsize, 0.0);
        if (workspace.t.size() != matsize) workspace.t.resize(matsize, 0.0);
        if (workspace.kvect.size() != vecsize) workspace.kvect.resize(vecsize, 0.0);
    }

    void reset_kvect(Workspace& workspace) const {
        for (size_t k = 0; k < workspace.kvect.size(); k++) workspace.kvect[k] = 0.0;
    }

    /*
     * NMSMSTdistance::computeattr(is, js): e1[i,j]=1 if same state else 0, t1[i,j]=min(ti,tj).
     * Iterate with cumulative sums and update e, t; kvect[k] = total minimal shared time.
     */
    void computeattr(int is, int js, Workspace& workspace) {
        auto ptr_seq = sequences.unchecked<2>();
        auto ptr_dur = seqdur.unchecked<2>();
        auto ptr_len = seqlength.unchecked<1>();

        int m = ptr_len(is);
        int n = ptr_len(js);
        if (m == 0 || n == 0) return;

        int mrows = m + 1;
        int ncols = n + 1;
        double st = 0.0;

        for (int i = 0; i < m; i++) {
            int seqi = ptr_seq(is, i);
            double ti = ptr_dur(is, i);
            for (int j = 0; j < n; j++) {
                int ij = IDX(i, j, rowsize);
                if (seqi == ptr_seq(js, j)) {
                    double tj = ptr_dur(js, j);
                    workspace.e1[ij] = 1.0;
                    workspace.e[ij] = 1.0;
                    workspace.t1[ij] = std::min(ti, tj);
                    workspace.t[ij] = workspace.t1[ij];
                    st += workspace.t[ij];
                    if (st >= std::numeric_limits<double>::max())
                        throw std::runtime_error("[!] Number of subsequences is getting too big");
                } else {
                    workspace.e1[ij] = workspace.e[ij] = workspace.t1[ij] = workspace.t[ij] = 0.0;
                }
            }
        }

        for (int i = 0; i < m; i++) {
            int ij = IDX(i, ncols - 1, rowsize);
            workspace.e1[ij] = workspace.e[ij] = workspace.t1[ij] = workspace.t[ij] = 0.0;
        }
        for (int j = 0; j < ncols; j++) {
            int ij = IDX(mrows - 1, j, rowsize);
            workspace.e1[ij] = workspace.e[ij] = workspace.t1[ij] = workspace.t[ij] = 0.0;
        }

        int k = 0;
        workspace.kvect[static_cast<size_t>(k)] = st;
        if (st == 0.0) return;

        while (mrows != 0 && ncols != 0) {
            k++;
            double sum = 0.0, sumt = 0.0, temp = 0.0, tempt = 0.0, temps = 0.0, temptt = 0.0;

            for (int irow = 0; irow < mrows; irow++) {
                sum = sumt = 0.0;
                for (int jcol = ncols - 1; jcol >= 0; jcol--) {
                    int ij = IDX(irow, jcol, rowsize);
                    temp = sum;
                    tempt = sumt;
                    sum += workspace.e[ij];
                    sumt += workspace.t[ij];
                    workspace.e[ij] = temp;
                    workspace.t[ij] = tempt;
                }
            }

            for (int jcol = 0; jcol < ncols; jcol++) {
                sum = 0.0;
                sumt = 0.0;
                for (int irow = mrows - 1; irow >= 0; irow--) {
                    int ij = IDX(irow, jcol, rowsize);
                    temp = sum;
                    tempt = sumt;
                    sum += workspace.e[ij];
                    sumt += workspace.t[ij];
                    workspace.e[ij] = temp * workspace.e1[ij];
                    workspace.t[ij] = workspace.e1[ij] * (workspace.e[ij] * workspace.t1[ij] + tempt);
                    temps += workspace.e[ij];
                    temptt += workspace.t[ij];
                }
            }

            if (temps == 0.0) return;
            if (k < maxlen) workspace.kvect[static_cast<size_t>(k)] = temptt;
            if (temptt >= std::numeric_limits<double>::max())
                throw std::runtime_error("[!] Number of subsequences is getting too big");
            mrows--;
            ncols--;
        }
    }

    double compute_distance(int is, int js) {
        try {
            auto ptr_kw = kweights.unchecked<1>();
            thread_local Workspace workspace;
            prepare_workspace(workspace);
            reset_kvect(workspace);
            computeattr(is, js, workspace);

            double Aval = 0.0, Ival = 0.0, Jval = 0.0;
            for (int i = 0; i < maxlen; i++) {
                if (i < kweights_len && ptr_kw(i) != 0.0) {
                    Aval += ptr_kw(i) * workspace.kvect[static_cast<size_t>(i)];
                    Ival += ptr_kw(i) * selfmatvect[static_cast<size_t>(is) * static_cast<size_t>(maxlen) + static_cast<size_t>(i)];
                    Jval += ptr_kw(i) * selfmatvect[static_cast<size_t>(js) * static_cast<size_t>(maxlen) + static_cast<size_t>(i)];
                }
            }

            double dist = Ival + Jval - 2.0 * Aval;
            double maxdist = Ival + Jval;
            if (dist < 0.0) dist = 0.0;
            return normalize_distance(dist, maxdist, Ival, Jval, norm);
        } catch (const std::exception& e) {
            py::print("Error in NMSMSTdistance::compute_distance: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_all_distances() {
        try {
            auto dist_matrix = py::array_t<double>({nseq, nseq});
            return dp_utils::compute_all_distances_simple(
                    nseq, dist_matrix,
                    [this](int i, int j) { return this->compute_distance(i, j); });
        } catch (const std::exception& e) {
            py::print("Error in NMSMSTdistance::compute_all_distances: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_condensed_distances() {
        try {
            return dp_utils::compute_condensed_distances_simple(
                    nseq,
                    [this](int i, int j) { return this->compute_distance(i, j); });
        } catch (const std::exception& e) {
            py::print("Error in NMSMSTdistance::compute_condensed_distances: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_refseq_distances() {
        try {
            auto refdist_matrix = py::array_t<double>({nseq, (rseq2 - rseq1)});
            return dp_utils::compute_refseq_distances_simple(
                    nseq, rseq1, rseq2, refdist_matrix,
                    [this](int is, int rseq) { return this->compute_distance(is, rseq); });
        } catch (const std::exception& e) {
            py::print("Error in NMSMSTdistance::compute_refseq_distances: ", e.what());
            throw;
        }
    }

private:
    py::array_t<int> sequences;
    py::array_t<double> seqdur;
    py::array_t<int> seqlength;
    py::array_t<double> kweights;
    int norm;
    int nseq;
    int maxlen;
    int rowsize;
    int kweights_len;
    std::vector<double> selfmatvect;
    int nans;
    int rseq1;
    int rseq2;
};
