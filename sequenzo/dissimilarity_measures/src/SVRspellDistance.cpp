/*
 * SVRspellDistance: Spell-based distance with duration and soft matching (proximity).
 *
 * Port of TraMineR's NMSDURSoftdistance (seqdist method "SVRspell").
 * Uses spell sequences (state + duration per spell), a softmatch (prox) matrix for
 * state similarity, and kweights for aggregating over subsequence lengths. The distance
 * is the square root of (Ival + Jval - 2*Aval) where Aval is the weighted sum of
 * minimal shared time over common subsequences, and Ival/Jval are the self-terms.
 * TraMineR applies sqrt to the raw value in R; we return the same final value so
 * results match.
 *
 * @Author  : Yuqi Liang 梁彧祺
 * @File    : SVRspellDistance.cpp
 * @Time    : 2026/2/6 9:27
 * @Desc    : 
 * Reference: TraMineR src/NMSDURSoftdistance.cpp, NMSdistance.cpp (SUBSEQdistance::distance).
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <limits>
#include <iostream>
#include "utils.h"
#include "dp_utils.h"

namespace py = pybind11;

// Row-major index for (i, j) in a matrix with ncols columns
#define IDX(i, j, ncols) ((i) * (ncols) + (j))

class SVRspellDistance {
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

            dist_matrix = py::array_t<double>({nseq, nseq});

            // kweights length; use at most maxlen
            int kw_len = static_cast<int>(kweights.shape()[0]);
            kweights_len = (kw_len < maxlen) ? kw_len : maxlen;

            // Precompute self-terms: for each sequence is, computeattr(is, is) and store kvect in selfmatvect
            selfmatvect.resize(static_cast<size_t>(nseq) * static_cast<size_t>(maxlen), 0.0);
            kvect_work.resize(static_cast<size_t>(maxlen), 0.0);

            for (int is = 0; is < nseq; is++) {
                reset_kvect();
                computeattr(is, is);
                for (int k = 0; k < maxlen; k++) {
                    selfmatvect[static_cast<size_t>(is) * static_cast<size_t>(maxlen) + static_cast<size_t>(k)] = kvect_work[static_cast<size_t>(k)];
                }
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
            py::print("Error in SVRspellDistance constructor: ", e.what());
            throw;
        }
    }

    void reset_kvect() {
        for (size_t k = 0; k < kvect_work.size(); k++) kvect_work[k] = 0.0;
    }

    /*
     * NMSDURSoftdistance::computeattr(is, js): fill kvect_work with tot_t_ij for k=0,1,...,min(m,n)-1.
     */
    void computeattr(int is, int js) {
        auto ptr_seq = sequences.unchecked<2>();
        auto ptr_dur = seqdur.unchecked<2>();
        auto ptr_len = seqlength.unchecked<1>();
        auto ptr_sm = softmatch.unchecked<2>();
        int m = ptr_len(is);
        int n = ptr_len(js);
        if (m == 0 || n == 0) return;

        int mrows = m + 1;
        int ncols = n + 1;
        int rowsize = maxlen + 1;
        size_t matsize = static_cast<size_t>(rowsize) * static_cast<size_t>(rowsize);

        std::vector<double> e1(matsize, 0.0), e(matsize, 0.0);
        std::vector<double> t_i(matsize, 0.0), t_j(matsize, 0.0), t_ij(matsize, 0.0);

        double tot_t_ij = 0.0;
        for (int i = 0; i < m; i++) {
            int seqi = ptr_seq(is, i);
            double ti = ptr_dur(is, i);
            for (int j = 0; j < n; j++) {
                int ij = IDX(i, j, rowsize);
                int seqj = ptr_seq(js, j);
                double sf = ptr_sm(seqi, seqj);
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
        kvect_work[static_cast<size_t>(k)] = tot_t_ij;
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
                    e[ij] = sf * e[ij];
                    t_ij[ij] = sf * (t_ij[ij] + ti * t_j[ij] + tj * t_i[ij] + e[ij] * ti * tj);
                    t_i[ij] = sf * (t_i[ij] + ti * e[ij]);
                    t_j[ij] = sf * (t_j[ij] + tj * e[ij]);
                    tot_t_ij += t_ij[ij];
                }
            }

            if (k < maxlen) kvect_work[static_cast<size_t>(k)] = tot_t_ij;
            if (tot_t_ij >= std::numeric_limits<double>::max()) {
                throw std::runtime_error("[!] Number of subsequences is getting too big");
            }
            mrows--;
            ncols--;
        }
    }

    /*
     * distMethod == 2: dist = Ival + Jval - 2*Aval; return sqrt(dist) then normalize (TraMineR applies sqrt in R).
     */
    double compute_distance(int is, int js) {
        try {
            auto ptr_kw = kweights.unchecked<1>();
            reset_kvect();
            computeattr(is, js);

            double Aval = 0.0, Ival = 0.0, Jval = 0.0;
            int minimum = maxlen;
            for (int i = 0; i < minimum; i++) {
                if (i < kweights_len && ptr_kw(i) != 0.0) {
                    Aval += ptr_kw(i) * kvect_work[static_cast<size_t>(i)];
                    Ival += ptr_kw(i) * selfmatvect[static_cast<size_t>(is) * static_cast<size_t>(maxlen) + static_cast<size_t>(i)];
                    Jval += ptr_kw(i) * selfmatvect[static_cast<size_t>(js) * static_cast<size_t>(maxlen) + static_cast<size_t>(i)];
                }
            }

            double dist = Ival + Jval - 2.0 * Aval;
            double maxdist = Ival + Jval;
            if (dist < 0.0) dist = 0.0;  // numerical safety
            double raw_sqrt = std::sqrt(dist);
            double maxdist_sqrt = (maxdist > 0.0) ? std::sqrt(maxdist) : 0.0;
            return normalize_distance(raw_sqrt, maxdist_sqrt, Ival, Jval, norm);
        } catch (const std::exception& e) {
            py::print("Error in SVRspellDistance::compute_distance: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_all_distances() {
        try {
            return dp_utils::compute_all_distances_simple(
                    nseq,
                    dist_matrix,
                    [this](int i, int j) { return this->compute_distance(i, j); }
            );
        } catch (const std::exception& e) {
            py::print("Error in SVRspellDistance::compute_all_distances: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_refseq_distances() {
        try {
            return dp_utils::compute_refseq_distances_simple(
                    nseq,
                    rseq1,
                    rseq2,
                    refdist_matrix,
                    [this](int is, int rseq) { return this->compute_distance(is, rseq); }
            );
        } catch (const std::exception& e) {
            py::print("Error in SVRspellDistance::compute_refseq_distances: ", e.what());
            throw;
        }
    }

private:
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
    std::vector<double> selfmatvect;
    std::vector<double> kvect_work;
    py::array_t<double> dist_matrix;

    int nans;
    int rseq1;
    int rseq2;
    py::array_t<double> refdist_matrix;
};
