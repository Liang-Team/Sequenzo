/*
 * LCPspellDistance: Spell-based Longest Common Prefix distance.
 *
 * Unlike position-wise LCP (which compares state at the same time index),
 * LCPspell compares sequences spell-by-spell: the k-th spell of sequence A
 * is compared with the k-th spell of sequence B. Two spells "match" if they
 * have the same state; we do not require the same start time.
 *
 * Raw distance:
 *   d_raw = (n + m - 2*L) + timecost * duration_penalty_norm
 * where:
 *   - n, m = number of spells in sequence A and B.
 *   - L = number of matched spells (prefix or suffix per sign).
 *   - duration_penalty_norm = sum over matched spells of |dur_A(k) - dur_B(k)| / tau
 *     (tau = duration_ref, fixed before computation; not dataset- or pair-dependent).
 *   - timecost (Python expcost): weight on relative duration differences; 0 ignores duration.
 *
 * Normalization (optional): maxdist = n + m + timecost * min(n, m), provided tau is chosen
 * as a maximum possible spell duration so that |dur_A(k)-dur_B(k)|/tau <= 1 for each matched
 * spell (unlike OMspell, LCPspell has no alternative edit path around duration penalty).
 *
 * @Author  : Yuqi Liang 梁彧祺
 * @File    : LCPspellDistance.cpp
 * @Time    : 2026/1/29 22:42
 */

 #include <pybind11/pybind11.h>
 #include <pybind11/numpy.h>
 #include <cmath>
 #include <iostream>
 #include <stdexcept>
 #include "utils.h"
 #include "dp_utils.h"
 
 namespace py = pybind11;
 
 class LCPspellDistance {
 public:
     LCPspellDistance(py::array_t<int> sequences,
                      py::array_t<double> seqdur,
                      py::array_t<int> seqlength,
                      int norm,
                      int sign,
                      py::array_t<int> refseqS,
                      double timecost,
                      double duration_ref)
             : norm(norm), sign(sign), timecost(timecost), duration_ref(duration_ref) {
         py::print("[>] Starting (Reverse) Longest Common Prefix on spells (LCPspell/RLCPspell)...");
         std::cout << std::flush;
 
         if (duration_ref <= 0.0) {
             throw std::invalid_argument("duration_ref must be positive for LCPspell/RLCPspell.");
         }
 
         try {
             this->sequences = sequences;
             this->seqdur = seqdur;
             this->seqlength = seqlength;
 
             auto seq_shape = sequences.shape();
             nseq = static_cast<int>(seq_shape[0]);
             max_spells = static_cast<int>(seq_shape[1]);
 
             double obs_max_dur = 0.0;
             auto ptr_dur = seqdur.unchecked<2>();
             auto ptr_len = seqlength.unchecked<1>();
             for (int i = 0; i < nseq; ++i) {
                 int li = ptr_len(i);
                 for (int k = 0; k < li && k < max_spells; ++k) {
                     double d = ptr_dur(i, k);
                     if (d > obs_max_dur) obs_max_dur = d;
                 }
             }
             if (obs_max_dur > duration_ref) {
                 py::print("[!] Warning: duration_ref is smaller than an observed spell duration. "
                           "Normalized LCPspell distances may exceed the stated bound.");
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
             py::print("Error in LCPspellDistance constructor: ", e.what());
             throw;
         }
     }
 
     double compute_distance(int is, int js) {
         try {
             auto ptr_seq = sequences.unchecked<2>();
             auto ptr_dur = seqdur.unchecked<2>();
             auto ptr_len = seqlength.unchecked<1>();
 
             int n = ptr_len(is);
             int m = ptr_len(js);
             int min_nm = (n < m) ? n : m;
 
             const double inv_tau = 1.0 / duration_ref;
 
             if (min_nm == 0) {
                 double raw = static_cast<double>(n + m);
                 double maxdist = raw;
                 return normalize_distance(raw, maxdist, static_cast<double>(n), static_cast<double>(m), norm);
             }
 
             int L = 0;
             double duration_penalty_sum = 0.0;
 
             if (sign > 0) {
                 while (L < min_nm && ptr_seq(is, L) == ptr_seq(js, L)) {
                     duration_penalty_sum += std::fabs(ptr_dur(is, L) - ptr_dur(js, L));
                     L++;
                 }
             } else {
                 while (L < min_nm && ptr_seq(is, n - 1 - L) == ptr_seq(js, m - 1 - L)) {
                     duration_penalty_sum += std::fabs(ptr_dur(is, n - 1 - L) - ptr_dur(js, m - 1 - L));
                     L++;
                 }
             }
 
             double duration_penalty_norm = duration_penalty_sum * inv_tau;
             double raw = (n + m - 2.0 * L) + timecost * duration_penalty_norm;
             double maxdist = static_cast<double>(n + m) + timecost * static_cast<double>(min_nm);
             return normalize_distance(raw, maxdist, static_cast<double>(n), static_cast<double>(m), norm);
         } catch (const std::exception& e) {
             py::print("Error in LCPspellDistance::compute_distance: ", e.what());
             throw;
         }
     }
 
     py::array_t<double> compute_all_distances() {
         try {
             auto dist_matrix = py::array_t<double>({nseq, nseq});
             return dp_utils::compute_all_distances_simple(
                     nseq,
                     dist_matrix,
                     [this](int i, int j) { return this->compute_distance(i, j); }
             );
         } catch (const std::exception& e) {
             py::print("Error in LCPspellDistance::compute_all_distances: ", e.what());
             throw;
         }
     }
 
     py::array_t<double> compute_condensed_distances() {
         try {
             return dp_utils::compute_condensed_distances_simple(
                     nseq,
                     [this](int i, int j) { return this->compute_distance(i, j); }
             );
         } catch (const std::exception& e) {
             py::print("Error in LCPspellDistance::compute_condensed_distances: ", e.what());
             throw;
         }
     }

     py::array_t<double> compute_refseq_distances() {
         try {
             auto refdist_matrix = py::array_t<double>({nseq, (rseq2 - rseq1)});
             return dp_utils::compute_refseq_distances_simple(
                     nseq,
                     rseq1,
                     rseq2,
                     refdist_matrix,
                     [this](int is, int rseq) { return this->compute_distance(is, rseq); }
             );
         } catch (const std::exception& e) {
             py::print("Error in LCPspellDistance::compute_refseq_distances: ", e.what());
             throw;
         }
     }
 
 private:
     py::array_t<int> sequences;
     py::array_t<double> seqdur;
     py::array_t<int> seqlength;
     int norm;
     int sign;
     double timecost;
     double duration_ref;
     int nseq;
     int max_spells;
 
     int nans;
     int rseq1;
     int rseq2;
 };
