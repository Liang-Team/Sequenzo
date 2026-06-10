#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstring>
#include "utils.h"
#include "dp_utils.h"
#ifdef _OPENMP
    #include <omp.h>
#endif

namespace py = pybind11;

class OMdistance {
public:
    OMdistance(py::array_t<int> sequences, py::array_t<double> sm, double indel, int norm, py::array_t<int> seqlength, py::array_t<int> refseqS,
               py::array_t<double> indellist = py::array_t<double>())
            : indel(indel), norm(norm) {

        py::print("[>] Starting Optimal Matching(OM)...");
        std::cout << std::flush;

        this->sequences = sequences;
        this->sm = sm;
        this->seqlength = seqlength;

        auto seq_shape = sequences.shape();
        nseq = seq_shape[0];
        seqlen = seq_shape[1];
        alphasize = sm.shape()[0];

        use_indellist = (indellist.size() == static_cast<py::ssize_t>(alphasize) || indellist.size() == static_cast<py::ssize_t>(alphasize - 1));
        indellist_0based = (indellist.size() == static_cast<py::ssize_t>(alphasize - 1));
        if (use_indellist)
            this->indellist = indellist;

        // Cache raw pointers. Original creates pybind11 unchecked<>() accessors
        // inside compute_distance, which is called ~n^2/2 times. Each accessor has overhead
        // from bounds-check setup and stride computation. Raw pointers eliminate this.
        seq_ptr = sequences.data();
        len_ptr = seqlength.data();
        if (use_indellist)
            indel_ptr = indellist.data();

        // Keep diagonal substitution costs at zero for direct table lookup.
        sm_buf.resize(alphasize * alphasize);
        std::memcpy(sm_buf.data(), sm.data(), alphasize * alphasize * sizeof(double));
        for (int i = 0; i < alphasize; ++i)
            sm_buf[i * alphasize + i] = 0.0;
        sm_local = sm_buf.data();

        fmatsize = use_indellist ? (seqlen + 2) : (seqlen + 1);

        if (norm == 4) {
            maxscost = 2 * indel;
        } else {
            for (int i = 0; i < alphasize; i++)
                for (int j = i + 1; j < alphasize; j++)
                    maxscost = std::max(maxscost, sm_local[i * alphasize + j]);
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
    }

    inline double get_indel_fast(int state) const {
        if (use_indellist) {
            int idx = indellist_0based ? (state - 1) : state;
            return indel_ptr[idx];
        }
        return indel;
    }

    double compute_distance(int is, int js, double* RESTRICT prev, double* RESTRICT curr) {
        const int m_full = len_ptr[is];
        const int n_full = len_ptr[js];
        int mSuf = m_full + 1, nSuf = n_full + 1;
        int prefix = 0;

        const int* seq_i = seq_ptr + static_cast<ptrdiff_t>(is) * seqlen;
        const int* seq_j = seq_ptr + static_cast<ptrdiff_t>(js) * seqlen;

        // Prefix/suffix trimming. Disabled for vector indel per TraMineR OMVIdistance.
        if (!use_indellist) {
            if (m_full > 0 && n_full > 0 && seq_i[0] == seq_j[0]) {
                prefix = 1;
                const int lim = std::min(m_full, n_full);
                int ii = 1;
                while (ii < lim && seq_i[ii] == seq_j[ii]) {
                    ii++; prefix++;
                }
            }
            while (mSuf > prefix + 1 && nSuf > prefix + 1 && seq_i[mSuf - 2] == seq_j[nSuf - 2]) {
                mSuf--; nSuf--;
            }
        }

        const int m = mSuf - prefix - 1;
        const int n = nSuf - prefix - 1;

        // Pre-compute normalization constants (all exit paths use these).
        const double ml = double(m_full) * indel;
        const double nl = double(n_full) * indel;
        const double maxpossiblecost = std::abs(n_full - m_full) * indel + maxscost * std::min(m_full, n_full);

        if (m == 0 && n == 0)
            return normalize_distance(0.0, 0.0, 0.0, 0.0, norm);
        if (m == 0) {
            double cost = 0;
            for (int x = 0; x < n; ++x)
                cost += get_indel_fast(seq_j[prefix + x]);
            return normalize_distance(cost, maxpossiblecost, ml, nl, norm);
        }
        if (n == 0) {
            double cost = 0;
            for (int x = 0; x < m; ++x)
                cost += get_indel_fast(seq_i[prefix + x]);
            return normalize_distance(cost, maxpossiblecost, ml, nl, norm);
        }

        // First row: cumulative insertion costs along seq_j segment.
        prev[0] = 0;
        for (int x = 1; x <= n; ++x)
            prev[x] = prev[x - 1] + get_indel_fast(seq_j[prefix + x - 1]);

        if (!use_indellist) {
            const int* seg_j = seq_j + prefix;
            const double indel_val = indel;

            for (int i = 1; i <= m; ++i) {
                const int ai = seq_i[prefix + i - 1];
                curr[0] = indel_val * double(i);
                const double* sm_row = sm_local + ai * alphasize;

                for (int j = 1; j <= n; ++j) {
                    const double subcost = sm_row[seg_j[j - 1]];
                    const double del_cost = prev[j] + indel_val;
                    const double ins_cost = curr[j - 1] + indel_val;
                    const double sub_cost = prev[j - 1] + subcost;
                    double best = del_cost;
                    if (ins_cost < best) best = ins_cost;
                    if (sub_cost < best) best = sub_cost;
                    curr[j] = best;
                }
                std::swap(prev, curr);
            }
        } else {
            double cum_indel_i = 0;
            for (int i = 1; i <= m; ++i) {
                const int ai = seq_i[prefix + i - 1];
                cum_indel_i += get_indel_fast(ai);
                curr[0] = cum_indel_i;

                const double* sm_row = sm_local + ai * alphasize;
                const double indel_ai = get_indel_fast(ai);

                for (int j = 1; j <= n; ++j) {
                    const int bj = seq_j[prefix + j - 1];
                    const double delcost = prev[j] + indel_ai;
                    const double inscost = curr[j - 1] + get_indel_fast(bj);
                    const double subval = prev[j - 1] + sm_row[bj];
                    double best = delcost;
                    if (inscost < best) best = inscost;
                    if (subval < best) best = subval;
                    curr[j] = best;
                }
                std::swap(prev, curr);
            }
        }

        return normalize_distance(prev[n], maxpossiblecost, ml, nl, norm);
    }


    py::array_t<double> compute_all_distances() {
        auto dist_matrix = py::array_t<double>({nseq, nseq});
        return dp_utils::compute_all_distances(
            nseq,
            fmatsize,
            dist_matrix,
            [this](int i, int j, double* prev, double* curr) {
                return this->compute_distance(i, j, prev, curr);
            }
        );
    }

    py::array_t<double> compute_condensed_distances() {
        return dp_utils::compute_condensed_distances(
            nseq,
            fmatsize,
            [this](int i, int j, double* prev, double* curr) {
                return this->compute_distance(i, j, prev, curr);
            }
        );
    }

    py::array_t<double> compute_refseq_distances() {
        auto refdist_matrix = py::array_t<double>({nseq, (rseq2 - rseq1)});
        auto buffer = refdist_matrix.mutable_unchecked<2>();

        #pragma omp parallel
        {
            double* prev = dp_utils::aligned_alloc_double(static_cast<size_t>(fmatsize));
            double* curr = dp_utils::aligned_alloc_double(static_cast<size_t>(fmatsize));

            #pragma omp for schedule(dynamic, 4)
            for (int rseq = rseq1; rseq < rseq2; rseq ++) {
                for (int is = 0; is < nseq; is ++) {
                    double cmpres = (is == rseq) ? 0.0 : compute_distance(is, rseq, prev, curr);
                    buffer(is, rseq - rseq1) = cmpres;
                }
            }
            dp_utils::aligned_free_double(prev);
            dp_utils::aligned_free_double(curr);
        }

        return refdist_matrix;
    }

private:
    py::array_t<int> sequences;
    py::array_t<double> sm;
    double indel;
    int norm;
    bool use_indellist = false;
    bool indellist_0based = false;
    py::array_t<double> indellist;
    int nseq;
    int seqlen;
    int alphasize;
    int fmatsize;
    py::array_t<int> seqlength;
    double maxscost = 0.0;

    const int* seq_ptr = nullptr;
    const int* len_ptr = nullptr;
    const double* indel_ptr = nullptr;
    std::vector<double> sm_buf;
    const double* sm_local = nullptr;

    int nans = -1;
    int rseq1 = -1;
    int rseq2 = -1;
};
