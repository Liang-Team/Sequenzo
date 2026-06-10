/*
 * LCPspellDistance: Spell-based Longest Common Prefix distance.
 *
 * Normalization (optional): maxdist = n + m + timecost * min(n, m) is a conservative
 * method-specific upper bound (not necessarily the tightest theoretical maximum).
 */

 #include <pybind11/pybind11.h>
 #include <pybind11/numpy.h>
 #include <cmath>
 #include <iostream>
 #include <stdexcept>
 #include "utils.h"
 #include "dp_utils.h"
 #include "lcp_input_validation.h"
 
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
 
         lcp_input::validate_lcpspell_inputs(
             sequences, seqdur, seqlength, norm, timecost, duration_ref, sign);

         this->sequences = sequences;
         this->seqdur = seqdur;
         this->seqlength = seqlength;

         auto seq_shape = sequences.shape();
         original_nseq_ = static_cast<int>(seq_shape[0]);
         max_spells = static_cast<int>(seq_shape[1]);

         const auto refseq_cfg = lcp_input::parse_refseq(refseqS, original_nseq_);
         has_refseq = refseq_cfg.has_refseq;
         nseq = refseq_cfg.nseq;
         rseq1 = refseq_cfg.rseq1;
         rseq2 = refseq_cfg.rseq2;
     }

     double compute_distance(int is, int js) {
         lcp_input::require_pair_indices(is, js, original_nseq_);
         return compute_distance_unchecked(is, js);
     }
 
     py::array_t<double> compute_all_distances() {
         auto dist_matrix = py::array_t<double>({nseq, nseq});
         return dp_utils::compute_all_distances_simple(
                 nseq,
                 dist_matrix,
                 [this](int i, int j) { return this->compute_distance_unchecked(i, j); }
         );
     }
 
     py::array_t<double> compute_condensed_distances() {
         return dp_utils::compute_condensed_distances_simple(
                 nseq,
                 [this](int i, int j) { return this->compute_distance_unchecked(i, j); }
         );
     }

     py::array_t<double> compute_refseq_distances() {
         lcp_input::require_refseq_for_compute({has_refseq, nseq, rseq1, rseq2});
         auto refdist_matrix = py::array_t<double>({nseq, (rseq2 - rseq1)});
         return dp_utils::compute_refseq_distances_simple(
                 nseq,
                 rseq1,
                 rseq2,
                 refdist_matrix,
                 [this](int is, int rseq) { return this->compute_distance_unchecked(is, rseq); }
         );
     }
 
 private:
     double normalize_lcpspell_or_return_raw(
         double raw,
         double maxdist,
         double n,
         double m) const {
         if (norm == 0) {
             return raw;
         }
         const double d = normalize_distance(raw, maxdist, n, m, norm);
         return lcp_input::clamp_unit_interval_or_throw(
             d, "normalized LCPspell distance");
     }

     double compute_distance_unchecked(int is, int js) {
         auto ptr_seq = sequences.unchecked<2>();
         auto ptr_dur = seqdur.unchecked<2>();
         auto ptr_len = seqlength.unchecked<1>();

         int n = ptr_len(is);
         int m = ptr_len(js);
         int min_nm = (n < m) ? n : m;

         const double n_d = static_cast<double>(n);
         const double m_d = static_cast<double>(m);
         const double inv_tau = 1.0 / duration_ref;

         if (min_nm == 0) {
             const double raw = n_d + m_d;
             if (norm == 0 || raw == 0.0) {
                 return raw;
             }
             return normalize_lcpspell_or_return_raw(raw, raw, n_d, m_d);
         }

         int L = 0;
         double duration_penalty_norm = 0.0;

         if (sign > 0) {
             while (L < min_nm && ptr_seq(is, L) == ptr_seq(js, L)) {
                 duration_penalty_norm +=
                     std::fabs(ptr_dur(is, L) - ptr_dur(js, L)) * inv_tau;
                 L++;
             }
         } else {
             while (L < min_nm && ptr_seq(is, n - 1 - L) == ptr_seq(js, m - 1 - L)) {
                 duration_penalty_norm +=
                     std::fabs(ptr_dur(is, n - 1 - L) - ptr_dur(js, m - 1 - L)) * inv_tau;
                 L++;
             }
         }
         const double raw = (n_d + m_d - 2.0 * L) + timecost * duration_penalty_norm;
         const double maxdist = n_d + m_d + timecost * static_cast<double>(min_nm);
         return normalize_lcpspell_or_return_raw(raw, maxdist, n_d, m_d);
     }

     py::array_t<int> sequences;
     py::array_t<double> seqdur;
     py::array_t<int> seqlength;
     int norm;
     int sign;
     double timecost;
     double duration_ref;
     int nseq;
     int original_nseq_;
     int max_spells;

     bool has_refseq = false;
     int rseq1;
     int rseq2;
 };
