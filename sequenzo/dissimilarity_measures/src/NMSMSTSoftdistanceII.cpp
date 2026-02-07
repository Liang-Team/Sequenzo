/*
 * NMSMSTSoftdistanceII: NMS with soft matching (proximity matrix).
 *
 * Port of TraMineR's NMSMSTSoftdistanceII (seqdist method "NMS" when prox is supplied).
 * Position-based sequences (no spell duration). e1[i,j] = softmatch[seqi, seqj];
 * kvect[k] = weighted count of common subsequences of length k. Same distance formula
 * as NMS (distMethod==2). TraMineR applies sqrt to the result in R; we return the
 * normalized value so Python can apply sqrt to match TraMineR output.
 *
 * @Author  : Yuqi Liang 梁彧祺
 * @File    : NMSMSTSoftdistanceII.cpp
 * @Time    : 2026/2/7 16:29
 * @Desc    : 
 * Reference: TraMineR src/NMSMSTSoftdistanceII.cpp, NMSMSTSoftdistanceII.h.
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

#define IDX(i, j, ncols) ((i) * (ncols) + (j))

class NMSMSTSoftdistanceII {
public:
    /*
     * Constructor.
     * - sequences: state sequences, shape (nseq, maxlen); only positions 0..seqlength(i)-1 valid.
     * - seqlength: number of valid positions per sequence, shape (nseq,).
     * - softmatch: proximity matrix (alphasize x alphasize); softmatch[i,j] = similarity between state i and j.
     * - kweights: weights for subsequence lengths.
     * - norm: normalization index.
     * - refseqS: [rseq1, rseq2).
     */
    NMSMSTSoftdistanceII(py::array_t<int> sequences,
                         py::array_t<int> seqlength,
                         py::array_t<double> softmatch,
                         py::array_t<double> kweights,
                         int norm,
                         py::array_t<int> refseqS)
            : norm(norm) {
        py::print("[>] Starting NMS with soft matching (prox)...");
        std::cout << std::flush;

        try {
            this->sequences = sequences;
            this->seqlength = seqlength;
            this->softmatch = softmatch;
            this->kweights = kweights;

            auto seq_shape = sequences.shape();
            nseq = static_cast<int>(seq_shape[0]);
            maxlen = static_cast<int>(seq_shape[1]);
            alphasize = static_cast<int>(softmatch.shape()[0]);

            int kw_len = static_cast<int>(kweights.shape()[0]);
            kweights_len = (kw_len < maxlen) ? kw_len : maxlen;

            dist_matrix = py::array_t<double>({nseq, nseq});

            rowsize = maxlen + 1;
            size_t matsize = static_cast<size_t>(rowsize) * static_cast<size_t>(rowsize);
            e1.resize(matsize, 0.0);
            e.resize(matsize, 0.0);

            selfmatvect.resize(static_cast<size_t>(nseq) * static_cast<size_t>(maxlen), 0.0);
            kvect_work.resize(static_cast<size_t>(maxlen), 0.0);

            for (int is = 0; is < nseq; is++) {
                reset_kvect();
                computeattr(is, is);
                for (int k = 0; k < maxlen; k++)
                    selfmatvect[static_cast<size_t>(is) * static_cast<size_t>(maxlen) + static_cast<size_t>(k)] = kvect_work[static_cast<size_t>(k)];
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
            py::print("Error in NMSMSTSoftdistanceII constructor: ", e.what());
            throw;
        }
    }

    void reset_kvect() {
        for (size_t k = 0; k < kvect_work.size(); k++) kvect_work[k] = 0.0;
    }

    /*
     * NMSMSTSoftdistanceII::computeattr(is, js): e1[i,j] = softmatch[seqi, seqj], e[i,j] = same.
     * kvect[0] = s + 1 (TraMineR convention). Then iterate with cumulative sums; kvect[k] = phi_k.
     */
    void computeattr(int is, int js) {
        auto ptr_seq = sequences.unchecked<2>();
        auto ptr_len = seqlength.unchecked<1>();
        auto ptr_sm = softmatch.unchecked<2>();

        int m = ptr_len(is);
        int n = ptr_len(js);
        if (m == 0 || n == 0) return;

        int mrows = m + 1;
        int ncols = n + 1;
        double s = 0.0;

        for (int i = 0; i < m; i++) {
            int seqi = ptr_seq(is, i);
            for (int j = 0; j < n; j++) {
                int ij = IDX(i, j, rowsize);
                int seqj = ptr_seq(js, j);
                double sf = ptr_sm(seqi, seqj);
                e1[ij] = sf;
                e[ij] = sf;
                s += sf;
                if (s >= std::numeric_limits<double>::max())
                    throw std::runtime_error("[!] Number of subsequences is getting too big");
            }
        }

        for (int i = 0; i < m; i++) {
            int ij = IDX(i, ncols - 1, rowsize);
            e1[ij] = e[ij] = 0.0;
        }
        for (int j = 0; j < ncols; j++) {
            int ij = IDX(mrows - 1, j, rowsize);
            e1[ij] = e[ij] = 0.0;
        }

        int k = 0;
        this->kvect_work[static_cast<size_t>(k)] = s + 1.0;  // TraMineR: kvect[k] = s + 1
        if (s == 0.0) return;

        while (mrows != 0 && ncols != 0) {
            k++;
            double phi_k = 0.0;

            for (int irow = 0; irow < mrows; irow++) {
                s = 0.0;
                for (int jcol = ncols - 1; jcol >= 0; jcol--) {
                    int ij = IDX(irow, jcol, rowsize);
                    double t = s;
                    s += e[ij];
                    e[ij] = t;
                }
            }

            for (int jcol = 0; jcol < ncols; jcol++) {
                s = 0.0;
                for (int irow = mrows - 1; irow >= 0; irow--) {
                    int ij = IDX(irow, jcol, rowsize);
                    double t = s;
                    s += e[ij];
                    e[ij] = t * e1[ij];
                    phi_k += e[ij];
                }
            }

            if (phi_k == 0.0) return;
            if (k < maxlen) kvect_work[static_cast<size_t>(k)] = phi_k;
            if (phi_k >= std::numeric_limits<double>::max())
                throw std::runtime_error("[!] Number of subsequences is getting too big");
            mrows--;
            ncols--;
        }
    }

    double compute_distance(int is, int js) {
        try {
            auto ptr_kw = kweights.unchecked<1>();
            reset_kvect();
            computeattr(is, js);

            double Aval = 0.0, Ival = 0.0, Jval = 0.0;
            for (int i = 0; i < maxlen; i++) {
                if (i < kweights_len && ptr_kw(i) != 0.0) {
                    Aval += ptr_kw(i) * kvect_work[static_cast<size_t>(i)];
                    Ival += ptr_kw(i) * selfmatvect[static_cast<size_t>(is) * static_cast<size_t>(maxlen) + static_cast<size_t>(i)];
                    Jval += ptr_kw(i) * selfmatvect[static_cast<size_t>(js) * static_cast<size_t>(maxlen) + static_cast<size_t>(i)];
                }
            }

            double dist = Ival + Jval - 2.0 * Aval;
            double maxdist = Ival + Jval;
            if (dist < 0.0) dist = 0.0;
            if (norm == 4)
                return normalize_distance(std::sqrt(dist), (maxdist > 0.0 ? std::sqrt(maxdist) : 0.0), Ival, Jval, norm);
            return normalize_distance(dist, maxdist, Ival, Jval, norm);
        } catch (const std::exception& e) {
            py::print("Error in NMSMSTSoftdistanceII::compute_distance: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_all_distances() {
        try {
            return dp_utils::compute_all_distances_simple(
                    nseq, dist_matrix,
                    [this](int i, int j) { return this->compute_distance(i, j); });
        } catch (const std::exception& e) {
            py::print("Error in NMSMSTSoftdistanceII::compute_all_distances: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_refseq_distances() {
        try {
            return dp_utils::compute_refseq_distances_simple(
                    nseq, rseq1, rseq2, refdist_matrix,
                    [this](int is, int rseq) { return this->compute_distance(is, rseq); });
        } catch (const std::exception& e) {
            py::print("Error in NMSMSTSoftdistanceII::compute_refseq_distances: ", e.what());
            throw;
        }
    }

private:
    py::array_t<int> sequences;
    py::array_t<int> seqlength;
    py::array_t<double> softmatch;
    py::array_t<double> kweights;
    int norm;
    int nseq;
    int maxlen;
    int alphasize;
    int rowsize;
    int kweights_len;
    std::vector<double> e1;
    std::vector<double> e;
    std::vector<double> selfmatvect;
    std::vector<double> kvect_work;
    py::array_t<double> dist_matrix;
    int nans;
    int rseq1;
    int rseq2;
    py::array_t<double> refdist_matrix;
};
