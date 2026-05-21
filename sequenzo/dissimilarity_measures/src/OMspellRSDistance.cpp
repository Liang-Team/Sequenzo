/*
 * OMspellRSDistance: Reference-scaled Optimal Matching on spells.
 *
 * Same DP as OMspell. Duration terms are divided by a fixed reference scale tau
 * (duration_ref, passed from Python) before multiplying by timecost (lambda / expcost).
 * tau is fixed before distance computation (not dataset- or pair-dependent).
 *
 * Spell-level costs (timecost = lambda), corrected OMspell then scaled by tau:
 *   Indel/del of (a_k, d_k):  c_indel(a_k) + lambda * (d_k - 1) / tau
 *   Sub same state:           lambda * |d_i - d_j| / tau
 *   Sub different state:      sigma(i,j) + lambda * (d_i + d_j - 2) / tau
 *
 * Normalization (optional, separate from reference-scaled costs): structural upper bound
 *   |n_s - m_s| * max(c_indel) + max(n_s, m_s) * max(sigma)
 * using maxindel from indellist and maxscost from sm. Duration expansion is not added
 * (OMspell minimum-cost alignment; unlike LCPspell).
 * ml/nl: sum of state-specific c_indel over spells (same convention as OMspell/OMtspell).
 *
 * Parameter timecost (Python expcost) is the linear duration_weight, not exponential.
 * @Author  : Yuqi Liang 梁彧祺
 * @File    : OMspellRSDistance.cpp
 * @Time    : 2026/1/31 22:12
 */

 #include <pybind11/pybind11.h>
 #include <pybind11/numpy.h>
 #include <vector>
 #include <cmath>
 #include <iostream>
 #include <stdexcept>
 #include "utils.h"
 #include "dp_utils.h"
 #ifdef _OPENMP
     #include <omp.h>
 #endif
 #include <xsimd/xsimd.hpp>
 
 namespace py = pybind11;
 
 class OMspellRSDistance {
 public:
     OMspellRSDistance(py::array_t<int> sequences, py::array_t<double> sm, double indel, int norm, py::array_t<int> refseqS,
                               double timecost, double duration_ref, py::array_t<double> seqdur, py::array_t<double> indellist,
                               py::array_t<int> seqlength)
             : indel(indel), norm(norm), timecost(timecost), duration_ref(duration_ref) {
 
         py::print("[>] Starting Optimal Matching with spell (OMspellRS, reference-scaled duration)...");
         std::cout << std::flush;
 
         if (duration_ref <= 0.0) {
             throw std::invalid_argument("duration_ref must be positive for OMspellRS.");
         }
 
         try {
             this->sequences = sequences;
             this->sm = sm;
             this->seqdur = seqdur;
             this->indellist = indellist;
             this->seqlength = seqlength;
 
             auto seq_shape = sequences.shape();
             nseq = static_cast<int>(seq_shape[0]);
             len = static_cast<int>(seq_shape[1]);
 
             dist_matrix = py::array_t<double>({nseq, nseq});
             fmatsize = len + 1;
 
             auto sm_shape = sm.shape();
             alphasize = static_cast<int>(sm_shape[0]);
 
             maxindel = indel;
             if (indellist.size() > 0) {
                 auto ptr_indel_init = indellist.unchecked<1>();
                 maxindel = 0.0;
                 for (py::ssize_t i = 0; i < indellist.shape(0); ++i) {
                     if (ptr_indel_init(i) > maxindel) maxindel = ptr_indel_init(i);
                 }
             }
 
             auto ptr_sm = sm.mutable_unchecked<2>();
             if (norm == 4) {
                 maxscost = 2 * maxindel;
             } else {
                 maxscost = 0.0;
                 for (int i = 0; i < alphasize; i++) {
                     for (int j = i + 1; j < alphasize; j++) {
                         if (ptr_sm(i, j) > maxscost) maxscost = ptr_sm(i, j);
                     }
                 }
                 maxscost = std::min(maxscost, 2 * maxindel);
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
             py::print("Error in OMspellRSDistance constructor: ", e.what());
             throw;
         }
     }
 
     double compute_distance(int is, int js, double* prev, double* curr) {
         try {
             auto ptr_seq = sequences.unchecked<2>();
             auto ptr_len = seqlength.unchecked<1>();
             auto ptr_sm = sm.unchecked<2>();
             auto ptr_dur = seqdur.unchecked<2>();
             auto ptr_indel = indellist.unchecked<1>();
 
             int mm = ptr_len(is);
             int nn = ptr_len(js);
             int mSuf = mm + 1;
             int nSuf = nn + 1;
 
             const double dur_scale = 1.0 / duration_ref;
 
             prev[0] = 0;
             curr[0] = 0;
 
             for (int jj = 1; jj < nSuf; jj++) {
                 int bj = ptr_seq(js, jj - 1);
                 double ins_cost = ptr_indel(bj) + timecost * ((ptr_dur(js, jj - 1) - 1.0) * dur_scale);
                 prev[jj] = prev[jj - 1] + ins_cost;
             }
 
             using batch_t = xsimd::batch<double>;
             constexpr std::size_t B = batch_t::size;
 
             for (int i = 1; i < mSuf; i++) {
                 int i_state = ptr_seq(is, i - 1);
                 double dur_i = ptr_dur(is, i - 1);
                 double del_cost_i = ptr_indel(i_state) + timecost * ((dur_i - 1.0) * dur_scale);
 
                 curr[0] = prev[0] + del_cost_i;
 
                 int j = 1;
                 for (; j + (int)B <= nSuf; j += (int)B) {
                     const double* prev_ptr = prev + j;
                     const double* prevm1_ptr = prev + (j - 1);
                     batch_t prevj = batch_t::load_unaligned(prev_ptr);
                     batch_t prevjm1 = batch_t::load_unaligned(prevm1_ptr);
 
                     alignas(64) double subs[B];
                     alignas(64) double ins[B];
                     for (std::size_t b = 0; b < B; ++b) {
                         int jj_idx = j + (int)b - 1;
                         int bj = ptr_seq(js, jj_idx);
                         double dur_j = ptr_dur(js, jj_idx);
                         if (i_state == bj) {
                             subs[b] = timecost * std::fabs(dur_i - dur_j) * dur_scale;
                         } else {
                            subs[b] = ptr_sm(i_state, bj) + timecost * (dur_i + dur_j - 2.0) * dur_scale;
                        }
                        ins[b] = ptr_indel(bj) + timecost * ((dur_j - 1.0) * dur_scale);
                     }
 
                     batch_t sub_batch = batch_t::load_unaligned(subs);
                     batch_t cand_del = prevj + batch_t(del_cost_i);
                     batch_t cand_sub = prevjm1 + sub_batch;
                     batch_t vert = xsimd::min(cand_del, cand_sub);
 
                     double running = curr[j - 1] + ins[0];
                     for (std::size_t b = 0; b < B; ++b) {
                         double v = vert.get(b);
                         double c = std::min(v, running);
                         curr[j + (int)b] = c;
                         if (b + 1 < B) running = c + ins[b + 1];
                     }
                 }
 
                 for (; j < nSuf; ++j) {
                     int j_state = ptr_seq(js, j - 1);
                     double dur_j = ptr_dur(js, j - 1);
                     double minimum = prev[j] + del_cost_i;
                    double j_indel = curr[j - 1] + (ptr_indel(j_state) + timecost * ((dur_j - 1.0) * dur_scale));
                    double sub = prev[j - 1] + (
                        (i_state == j_state)
                        ? (timecost * std::fabs(dur_i - dur_j) * dur_scale)
                        : (ptr_sm(i_state, j_state) + timecost * (dur_i + dur_j - 2.0) * dur_scale)
                    );
                     curr[j] = std::min({ minimum, j_indel, sub });
                 }
 
                 std::swap(prev, curr);
             }
 
             // Structural upper bound: |n_s-m_s|*max(c_indel) + max(n_s,m_s)*max(sigma); no duration term.
             const int max_nm = (mm > nn) ? mm : nn;
            double maxpossiblecost =
                std::abs(nn - mm) * maxindel + static_cast<double>(max_nm) * maxscost;

            double ml = 0.0;
            for (int spell_i = 0; spell_i < mm; ++spell_i) {
                ml += ptr_indel(ptr_seq(is, spell_i));
            }

            double nl = 0.0;
            for (int spell_j = 0; spell_j < nn; ++spell_j) {
                nl += ptr_indel(ptr_seq(js, spell_j));
            }
            return normalize_distance(prev[nSuf - 1], maxpossiblecost, ml, nl, norm);
         } catch (const std::exception& e) {
             py::print("Error in OMspellRSDistance::compute_distance: ", e.what());
             throw;
         }
     }
 
     py::array_t<double> compute_all_distances() {
         try {
             return dp_utils::compute_all_distances(
                 nseq,
                 fmatsize,
                 dist_matrix,
                 [this](int i, int j, double* prev, double* curr) {
                     return this->compute_distance(i, j, prev, curr);
                 }
             );
         } catch (const std::exception& e) {
             py::print("Error in OMspellRSDistance::compute_all_distances: ", e.what());
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
                         buffer(is, rseq - rseq1) = (is != rseq) ? compute_distance(is, rseq, prev, curr) : 0.0;
                     }
                 }
                 dp_utils::aligned_free_double(prev);
                 dp_utils::aligned_free_double(curr);
             }
             return refdist_matrix;
         } catch (const std::exception& e) {
             py::print("Error in OMspellRSDistance::compute_refseq_distances: ", e.what());
             throw;
         }
     }
 
 private:
     py::array_t<int> sequences;
     py::array_t<int> seqlength;
     py::array_t<double> sm;
     double indel;
     int norm;
     int nseq;
     int len;
     int alphasize;
     int fmatsize;
     py::array_t<double> dist_matrix;
     double maxscost;
     double maxindel;
     double timecost;
     double duration_ref;
     py::array_t<double> seqdur;
     py::array_t<double> indellist;
     int nans;
     int rseq1;
     int rseq2;
     py::array_t<double> refdist_matrix;
 };