/*
 * NMSdistance: Number of Matching Subsequences distance.
 *
 * Port of TraMineR's NMSdistance (seqdist method "NMS").
 * For each pair of sequences we count the number of common subsequences of each length k,
 * store in kvect[k]. Distance uses distMethod==2: dist = Ival + Jval - 2*Aval where
 * Aval = sum kweights[k]*kvect[k], Ival/Jval = sum kweights[k]*selfmatvect[is/js, k].
 * TraMineR then applies sqrt to the result in R; we return the normalized value so
 * Python can apply sqrt to match TraMineR output.
 *
 * @Author  : Yuqi Liang 梁彧祺
 * @File    : NMSdistance.cpp
 * @Time    : 2026/2/7 13:43
 * @Desc    : 
 * Reference: TraMineR src/NMSdistance.cpp, NMSdistance.h.
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

// Row-major index: (i, j) in matrix with ncols columns
#define IDX(i, j, ncols) ((i) * (ncols) + (j))

class NMSdistance {
public:
    /*
     * Constructor.
     * - sequences: state sequences, shape (nseq, maxlen); only positions 0..seqlength(i)-1 valid.
     * - seqlength: number of valid positions per sequence, shape (nseq,).
     * - kweights: weights for subsequence lengths, length >= maxlen (extra ignored).
     * - norm: normalization index (TraMineR uses YujianBo/4 for NMS when auto).
     * - refseqS: [rseq1, rseq2).
     */
    NMSdistance(py::array_t<int> sequences,
                py::array_t<int> seqlength,
                py::array_t<double> kweights,
                int norm,
                py::array_t<int> refseqS)
            : norm(norm) {
        py::print("[>] Starting NMS (Number of Matching Subsequences)...");
        std::cout << std::flush;

        try {
            this->sequences = sequences;
            this->seqlength = seqlength;
            this->kweights = kweights;

            auto seq_shape = sequences.shape();
            nseq = static_cast<int>(seq_shape[0]);
            maxlen = static_cast<int>(seq_shape[1]);

            int kw_len = static_cast<int>(kweights.shape()[0]);
            kweights_len = (kw_len < maxlen) ? kw_len : maxlen;

            dist_matrix = py::array_t<double>({nseq, nseq});

            // Precompute self-terms: for each sequence is, computeattr(is, is) -> kvect, store in selfmatvect
            selfmatvect.resize(static_cast<size_t>(nseq) * static_cast<size_t>(maxlen), 0.0);
            kvect_work.resize(static_cast<size_t>(maxlen), 0.0);
            hmat.resize(static_cast<size_t>(maxlen) * static_cast<size_t>(maxlen), 0.0);
            vmat.resize(static_cast<size_t>(maxlen) * static_cast<size_t>(maxlen), 0.0);
            zmat_i.resize(static_cast<size_t>(maxlen) * static_cast<size_t>(maxlen), 0);
            zmat_j.resize(static_cast<size_t>(maxlen) * static_cast<size_t>(maxlen), 0);

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
            py::print("Error in NMSdistance constructor: ", e.what());
            throw;
        }
    }

    void reset_kvect() {
        for (size_t k = 0; k < kvect_work.size(); k++) kvect_work[k] = 0.0;
    }

    /*
     * NMSdistance::computeattr(is, js): build zmat of matching (i,j), hmat, then iterate
     * with vmat recurrence to get kvect[k] = number of common subsequences of length k.
     * Uses row-major indexing: (i,j) -> i*maxlen+j.
     */
    void computeattr(int is, int js) {
        auto ptr_seq = sequences.unchecked<2>();
        auto ptr_len = seqlength.unchecked<1>();

        int m = ptr_len(is);
        int n = ptr_len(js);
        if (m == 0 || n == 0) return;

        int ksize = (m < n) ? m : n;
        int zsize = 0;

        // Build list of matching positions (i, j) where seq_is[i] == seq_js[j]
        for (int i = 0; i < m; i++) {
            int si = ptr_seq(is, i);
            for (int j = 0; j < n; j++) {
                if (si == ptr_seq(js, j)) {
                    size_t idx = static_cast<size_t>(zsize);
                    if (idx < zmat_i.size()) {
                        zmat_i[idx] = i;
                        zmat_j[idx] = j;
                    }
                    zsize++;
                }
            }
        }

        // Initialize vmat border (last row and last column) to 0
        for (int j = 0; j < n; j++) vmat[IDX(m - 1, j, maxlen)] = 0.0;
        for (int i = 0; i < m; i++) vmat[IDX(i, n - 1, maxlen)] = 0.0;

        // hmat: 1 at match positions, 0 elsewhere; vmat zeroed in interior (filled in loop)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int ij = IDX(i, j, maxlen);
                hmat[ij] = 0.0;
                vmat[ij] = 0.0;
            }
        }
        int zindex = 0;
        double htot = 0.0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int ij = IDX(i, j, maxlen);
                if (zindex < zsize && zmat_i[zindex] == i && zmat_j[zindex] == j) {
                    hmat[ij] = 1.0;
                    htot += 1.0;
                    zindex++;
                }
            }
        }

        int k = 0;
        if (k < maxlen) kvect_work[static_cast<size_t>(k)] = htot;
        k++;

        if (m > 1 && n > 1) {
            while (k < ksize && htot > 0) {
                if (htot >= std::numeric_limits<double>::max())
                    throw std::runtime_error("[!] Number of subsequences is getting too big");

                // vmat[i,j] = vmat[i+1,j] + vmat[i,j+1] - vmat[i+1,j+1] + hmat[i+1,j+1] (row-major)
                for (int j = n - 2; j >= 0; j--) {
                    for (int i = m - 2; i >= 0; i--) {
                        int ij = IDX(i, j, maxlen);
                        vmat[ij] = vmat[IDX(i + 1, j, maxlen)] + vmat[IDX(i, j + 1, maxlen)]
                                   - vmat[IDX(i + 1, j + 1, maxlen)] + hmat[IDX(i + 1, j + 1, maxlen)];
                    }
                }

                if (vmat[0] == 0.0) {
                    if (k < maxlen) kvect_work[static_cast<size_t>(k)] = 0.0;
                    break;
                }

                htot = 0.0;
                for (zindex = 0; zindex < zsize; zindex++) {
                    int i = zmat_i[zindex], j = zmat_j[zindex];
                    int ij = IDX(i, j, maxlen);
                    hmat[ij] = vmat[ij];
                    htot += vmat[ij];
                }
                if (k < maxlen) kvect_work[static_cast<size_t>(k)] = htot;
                k++;
            }
        }
        for (; k < maxlen; k++) kvect_work[static_cast<size_t>(k)] = 0.0;
    }

    /*
     * distMethod == 2: dist = Ival + Jval - 2*Aval; return normalized (TraMineR applies sqrt in R).
     */
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
            py::print("Error in NMSdistance::compute_distance: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_all_distances() {
        try {
            return dp_utils::compute_all_distances_simple(
                    nseq, dist_matrix,
                    [this](int i, int j) { return this->compute_distance(i, j); });
        } catch (const std::exception& e) {
            py::print("Error in NMSdistance::compute_all_distances: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_refseq_distances() {
        try {
            return dp_utils::compute_refseq_distances_simple(
                    nseq, rseq1, rseq2, refdist_matrix,
                    [this](int is, int rseq) { return this->compute_distance(is, rseq); });
        } catch (const std::exception& e) {
            py::print("Error in NMSdistance::compute_refseq_distances: ", e.what());
            throw;
        }
    }

private:
    py::array_t<int> sequences;
    py::array_t<int> seqlength;
    py::array_t<double> kweights;
    int norm;
    int nseq;
    int maxlen;
    int kweights_len;
    std::vector<double> selfmatvect;
    std::vector<double> kvect_work;
    std::vector<double> hmat;
    std::vector<double> vmat;
    std::vector<int> zmat_i;
    std::vector<int> zmat_j;
    py::array_t<double> dist_matrix;
    int nans;
    int rseq1;
    int rseq2;
    py::array_t<double> refdist_matrix;
};
