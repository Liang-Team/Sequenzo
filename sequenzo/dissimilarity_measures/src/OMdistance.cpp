#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <iostream>

namespace py = pybind11;

class OMdistance {
public:
    OMdistance(py::array_t<int> sequences, py::array_t<double> sm, double indel, int norm, py::array_t<int> seqlength,py::array_t<int> refseqS)
            : indel(indel), norm(norm) {

        py::print("[>] Starting Optimal Matching(OM)...");
        std::cout << std::flush;

        try {
            // =========================
            // parameter : sequences, sm
            // =========================
            this->sequences = sequences;
            this->sm = sm;
            this->seqlength = seqlength;

            auto seq_shape = sequences.shape();
            nseq = seq_shape[0];
            seqlen = seq_shape[1];
            alphasize = sm.shape()[0];

            dist_matrix = py::array_t<double>({nseq, nseq});

            fmatsize = seqlen + 1;
            fmat.resize(fmatsize);
            for (int i = 0; i < fmatsize; ++i)
                fmat[i].resize(fmatsize, 0);

            // ==================
            // initialize maxcost
            // ==================
            if(norm == 4){
                maxscost = 2 * indel;
            }else{
                auto ptr = sm.mutable_unchecked<2>();
                for(int i = 0; i < alphasize; i++){
                    for(int j = i+1; j < alphasize; j++){
                        if(ptr(i, j) > maxscost){
                            maxscost = ptr(i, j);
                        }
                    }
                }
                maxscost = std::min(maxscost, 2 * indel);
            }

            // about reference sequences :
            nans = nseq;

            rseq1 = refseqS.at(0);
            rseq2 = refseqS.at(1);
            if(rseq1 < rseq2){
                nseq = rseq1;
                nans = nseq * (rseq2 - rseq1);
            }else{
                rseq1 = rseq1 - 1;
            }
            refdist_matrix = py::array_t<double>({nseq, (rseq2-rseq1)});
        } catch (const std::exception& e) {
            py::print("Error in constructor: ", e.what());
            throw;
        }
    }

    double normalize_distance(double rawdist, double maxdist, double l1, double l2) const {
        if (rawdist == 0.0) return 0.0;
        switch (norm) {
            case 0:
                return rawdist;
            case 1:
                return l1 > l2 ? rawdist / l1 : l2 > 0.0 ? rawdist / l2 : 0.0;
            case 2:
                return (l1 * l2 == 0.0) ? (l1 != l2 ? 1.0 : 0.0)
                                        : 1.0 - ((maxdist - rawdist) / (2.0 * std::sqrt(l1) * std::sqrt(l2)));
            case 3:
                return maxdist == 0.0 ? 1.0 : rawdist / maxdist;
            case 4:
                return maxdist == 0.0 ? 1.0 : (2.0 * rawdist) / (rawdist + maxdist);
            default: return rawdist;
        }
    }

    double compute_distance(int is, int js) {
        try {
            auto ptr_len = seqlength.unchecked<1>();
            double maxpossiblecost;
            int m = ptr_len(is);
            int n = ptr_len(js);
            int mSuf = m+1, nSuf = n+1;
            int prefix = 0;

            auto ptr_seq = sequences.unchecked<2>();
            auto ptr_sm = sm.unchecked<2>();

//            Skipping common prefix
            int i = 1;
            int j = 1;
            while (i < mSuf && j < nSuf &&
                   ptr_seq(is, i-1) == ptr_seq(js, j-1)){
                i++;
                j++;
                prefix++;
            }
//            Skipping common suffix
            while (mSuf > i && nSuf > j
                   && ptr_seq(is, mSuf - 2) == ptr_seq(js, nSuf - 2)) {
                mSuf--;
                nSuf--;
            }

            m = mSuf - prefix;
            n = nSuf - prefix;
            for(int i = 0; i < m; i ++)
                fmat[i][0]  = i * indel;
            for(int i = 0; i < n; i ++)
                fmat[0][i] = i * indel;

            for(int i = prefix+1; i < mSuf; i ++){
                for(int j = prefix+1; j < nSuf; j ++){
                    double minimum = fmat[i-1-prefix][j-prefix] + indel;
                    double j_indel = fmat[i-prefix][j-1-prefix] + indel;

                    double sub = 0;
                    if(ptr_seq(is, i-1) == ptr_seq(js, j-1)){
                        sub = fmat[i-1-prefix][j-1-prefix];
                    }else{
                        sub = fmat[i-1-prefix][j-1-prefix] + ptr_sm(ptr_seq(is, i-1), ptr_seq(js, j-1));
                    }

                    fmat[i-prefix][j-prefix] = std::min({minimum, j_indel, sub});
                }
            }

            maxpossiblecost = abs(n-m) * indel + maxscost * std::min(m, n);

            double ml = double(m) * indel;
            double nl = double(n) * indel;

            return normalize_distance(fmat[mSuf-1-prefix][nSuf-1-prefix],maxpossiblecost, ml, nl);
        } catch (const std::exception& e) {
            py::print("Error in compute_distance: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_all_distances() {
        try {
            auto buffer = dist_matrix.mutable_unchecked<2>();

            for (int i = 0; i < nseq; i++) {
                for (int j = i; j < nseq; j++) {
                    double dist = compute_distance(i, j);
                    buffer(i, j) = dist;
                    buffer(j, i) = dist;
                }
            }
            return dist_matrix;
        } catch (const std::exception& e) {
            py::print("Error in compute_all_distances: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_refseq_distances() {
        try {
            auto buffer = refdist_matrix.mutable_unchecked<2>();

            double cmpres = 0;
            for (int rseq = rseq1; rseq < rseq2; rseq ++) {
                for (int is = 0; is < nseq; is ++) {
                    if(is == rseq){
                        cmpres = 0;
                    }else{
                        cmpres = compute_distance(is, rseq);
                    }

                    buffer(is, rseq-rseq1) = cmpres;
                }
            }

            return refdist_matrix;
        } catch (const std::exception& e) {
            py::print("Error in compute_all_distances: ", e.what());
            throw;
        }
    }

private:
    py::array_t<int> sequences;
    py::array_t<double> sm;
    double indel;
    int norm;
    int nseq;
    int seqlen;
    int alphasize;
    int fmatsize;
    py::array_t<int> seqlength;
    std::vector<std::vector<double>> fmat;
    py::array_t<double> dist_matrix;
    double maxscost;

    // about reference sequences :
    int nans = -1;
    int rseq1 = -1;
    int rseq2 = -1;
    py::array_t<double> refdist_matrix;
};