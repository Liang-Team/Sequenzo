/*
 * Differences from WeightedCluster 2.0
 * wcKMedoids(method = "PAMonce"):
 * - strict PAM stopping and zero-distance swap candidates
 * - FastPAM screening with classic PAM rescoring
 * - condensed/codebook input, bounded workspaces, and optional OpenMP
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "pam_common.h"

namespace py = pybind11;

struct PamScreenedCandidate {
    double lower_bound = 0.0;
    int h = -1;
    int k_slot = -1;
};

constexpr std::size_t PAM_SCREENED_BUFFER_CAPACITY = 64;

class PAMonce {
public:
    PAMonce(int nelements, py::array diss,
            PamIntArray centroids, int npass,
            PamDoubleArray weights, int requested_threads = 0,
            std::size_t memory_budget_bytes = 0,
            PamIntArray build_tie_keys = PamIntArray(),
            PamDoubleArray distance_codebook = PamDoubleArray())
        : nelements(nelements),
          diss(std::move(diss)),
          centroids(std::move(centroids)),
          npass(npass),
          weights(std::move(weights)),
          build_tie_keys(std::move(build_tie_keys)),
          distance_codebook(std::move(distance_codebook)),
          nclusters(static_cast<int>(this->centroids.size())),
          clusterid(nelements),
          dysma(nelements),
          dysmb(nelements),
          second_slot(nelements),
          is_medoid(nelements, 0) {
        unit_weights = this->weights.size() == 0;
        wt_ptr = unit_weights ? nullptr : this->weights.data();
        if (this->build_tie_keys.size() == nelements) {
            build_tie_ptr = this->build_tie_keys.data();
        }
        if (this->distance_codebook.size() > 0) {
            distance_codebook_ptr = this->distance_codebook.data();
            distance_codebook_size = static_cast<std::size_t>(
                this->distance_codebook.size());
        }
        initializeDistanceStorage();
        maxdist = std::numeric_limits<double>::infinity();

        sum_abs_weights_bound = static_cast<double>(nelements);
        if (!unit_weights) {
            double rounded_weight_sum = 0.0;
            for (int i = 0; i < nelements; ++i) {
                rounded_weight_sum += std::abs(wt_ptr[i]);
                if (!std::isfinite(wt_ptr[i]) || wt_ptr[i] < 0.0) {
                    weights_are_nonnegative = false;
                }
                if (!std::isfinite(wt_ptr[i]) || wt_ptr[i] < 0.0 ||
                    wt_ptr[i] != std::trunc(wt_ptr[i])) {
                    weights_are_nonnegative_integers = false;
                }
            }
            sum_abs_weights_bound = pam_upper_abs_sum(
                rounded_weight_sum,
                nelements > 0 ? static_cast<std::size_t>(nelements - 1) : 0);
        }
        const double sum_gamma = pam_gamma_bound(
            nelements > 0 ? static_cast<std::size_t>(nelements - 1) : 0);
        sign_aware_abs_sum_scale = sum_gamma < 1.0
            ? std::nextafter(
                1.0 / (1.0 - sum_gamma),
                std::numeric_limits<double>::infinity())
            : std::numeric_limits<double>::infinity();
        sign_aware_sum_gamma = pam_gamma_bound(
            static_cast<std::size_t>(nelements));
        sign_aware_underflow_bound = pam_upward_multiply(
            4.0 * static_cast<double>(2 * nelements + 1),
            std::numeric_limits<double>::denorm_min());
#ifdef _OPENMP
        const int available_threads = requested_threads > 0
            ? requested_threads
            : omp_get_max_threads();
        worker_count = std::max(
            1, std::min(available_threads, nelements - nclusters));
#else
        worker_count = 1;
#endif
        const std::size_t bytes_per_worker =
            static_cast<std::size_t>(nclusters) *
                (2 * sizeof(double) + sizeof(std::uint32_t))
            + sizeof(PamCandidate) + 128
            + PAM_SCREENED_BUFFER_CAPACITY * sizeof(PamScreenedCandidate);
        if (memory_budget_bytes > 0 && bytes_per_worker > 0) {
            if (memory_budget_bytes < bytes_per_worker) {
                throw py::value_error(
                    "memory_budget_mb is too small for one PAMonce thread workspace.");
            }
            const std::size_t affordable =
                memory_budget_bytes / bytes_per_worker;
            worker_count = std::min(
                worker_count, static_cast<int>(std::min<std::size_t>(
                    affordable, static_cast<std::size_t>(worker_count))));
        }
        workspaces.reserve(worker_count);
        thread_upper_bounds.resize(
            worker_count, 0.0);
        thread_overlap_counts.resize(worker_count, 0);
        thread_best.resize(worker_count);
        screened_buffers.resize(worker_count);
        for (int i = 0; i < worker_count; ++i) {
            workspaces.emplace_back(nclusters);
            screened_buffers[i].resize(PAM_SCREENED_BUFFER_CAPACITY);
        }
        bounded_candidate_buffer_bytes =
            static_cast<std::size_t>(worker_count) *
            PAM_SCREENED_BUFFER_CAPACITY * sizeof(PamScreenedCandidate);
        thread_workspace_bytes =
            static_cast<std::size_t>(worker_count) * bytes_per_worker;
        workspace_peak_bytes =
            static_cast<std::size_t>(nelements) *
                (2 * sizeof(double) + 2 * sizeof(int) + sizeof(std::uint8_t))
            + thread_workspace_bytes;
    }

    py::array_t<int> runclusterloop() {
        return runclusterloopImpl(0);
    }

    py::array_t<int> runclusterloop_one_based() {
        return runclusterloopImpl(1);
    }

    double objective() const {
        return final_objective;
    }

    void set_collect_diagnostics(bool enabled) {
        collect_diagnostics = enabled;
    }

    py::array_t<int> build_initial_medoids() {
        py::array_t<int> result(nclusters);
        int* output = result.mutable_data();
        int* cent_ptr = centroids.mutable_data();
        {
            py::gil_scoped_release release;
            maxdist = worker_count == 1 ? computeMaxDistSerial() : computeMaxDist();
            if (unit_weights) {
                if (worker_count == 1) {
                    buildInitialCentroidsSerial<true>(cent_ptr);
                } else {
                    buildInitialCentroids<true>(cent_ptr);
                }
            } else {
                if (worker_count == 1) {
                    buildInitialCentroidsSerial<false>(cent_ptr);
                } else {
                    buildInitialCentroids<false>(cent_ptr);
                }
            }
            std::copy(cent_ptr, cent_ptr + nclusters, output);
        }
        return result;
    }

    py::dict diagnostics() const {
        py::dict result;
        result["fast_score_evaluations"] = fast_score_evaluations;
        result["classic_score_evaluations"] = classic_score_evaluations;
        result["swap_rounds"] = swap_rounds;
        result["accepted_swaps"] = accepted_swaps;
        result["swap_trace"] = swapTrace();
        result["worker_count"] = worker_count;
        result["candidate_storage_bytes"] = 0;
        result["bounded_candidate_buffer_bytes"] = bounded_candidate_buffer_bytes;
        result["thread_workspace_bytes"] = thread_workspace_bytes;
        result["workspace_peak_bytes"] = workspace_peak_bytes;
        result["distance_storage_bytes"] =
            diss.nbytes() + distance_codebook.nbytes();
        result["distance_dtype"] = distanceDtypeName();
        result["distance_codebook_size"] = distance_codebook_size;
        result["distance_properties_precomputed"] =
            input_distance_properties_known;
        result["screened_candidate_highwater"] = screened_candidate_highwater;
        result["adaptive_fallback_rounds"] = adaptive_fallback_rounds;
        result["two_pass_recovery_rounds"] = two_pass_recovery_rounds;
        result["small_k_fused_rounds"] = small_k_fused_rounds;
        result["small_k_fused_seconds"] = small_k_fused_seconds;
        result["small_k_reynolds_rounds"] = small_k_reynolds_rounds;
        result["exact_integer_rounds"] = exact_integer_rounds;
        result["exact_fixed_point_rounds"] = exact_fixed_point_rounds;
        result["sign_aware_verified_rounds"] =
            sign_aware_verified_rounds;
        result["execution_path"] = executionPath();
        result["max_distance_seconds"] = max_distance_seconds;
        result["build_seconds"] = build_seconds;
        result["assignment_seconds"] = assignment_seconds;
        result["fast_screen_seconds"] = fast_screen_seconds;
        result["exact_arbitration_seconds"] = exact_arbitration_seconds;
        result["reynolds_fallback_seconds"] = reynolds_fallback_seconds;
        result["two_pass_recovery_seconds"] = two_pass_recovery_seconds;
        result["swap_update_seconds"] = swap_update_seconds;
        result["full_assignment_rounds"] = full_assignment_rounds;
        result["incremental_update_rounds"] = incremental_update_rounds;
        result["incremental_rescanned_rows"] = incremental_rescanned_rows;
        result["incremental_update_seconds"] = incremental_update_seconds;
        result["final_assignment_seconds"] = final_assignment_seconds;
        result["objective"] = final_objective;
        return result;
    }

private:
    using Clock = std::chrono::steady_clock;

    enum class DistanceStorage {
        Float64,
        UInt8,
        UInt16,
        UInt32,
    };

    void initializeDistanceStorage() {
        if (diss.ndim() != 1 && diss.ndim() != 2) {
            throw py::value_error("PAMonce distance input must be 1-D or 2-D.");
        }
        if ((diss.flags() & py::array::c_style) == 0) {
            throw py::type_error("PAMonce distance input must be C-contiguous.");
        }
        use_condensed = diss.ndim() == 1;
        if (py::dtype::of<double>().is(diss.dtype())) {
            if (distance_codebook_size > 0) {
                throw py::value_error(
                    "A distance codebook requires an unsigned integer code array.");
            }
            distance_storage = DistanceStorage::Float64;
            distance_f64 = static_cast<const double*>(diss.data());
        } else if (py::dtype::of<std::uint8_t>().is(diss.dtype())) {
            distance_storage = DistanceStorage::UInt8;
            distance_u8 = static_cast<const std::uint8_t*>(diss.data());
        } else if (py::dtype::of<std::uint16_t>().is(diss.dtype())) {
            distance_storage = DistanceStorage::UInt16;
            distance_u16 = static_cast<const std::uint16_t*>(diss.data());
        } else if (py::dtype::of<std::uint32_t>().is(diss.dtype())) {
            distance_storage = DistanceStorage::UInt32;
            distance_u32 = static_cast<const std::uint32_t*>(diss.data());
        } else {
            throw py::type_error(
                "PAMonce distance input must use float64, uint8, uint16, or uint32.");
        }
    }

    const char* distanceDtypeName() const {
        switch (distance_storage) {
            case DistanceStorage::Float64: return "float64";
            case DistanceStorage::UInt8: return "uint8";
            case DistanceStorage::UInt16: return "uint16";
            case DistanceStorage::UInt32: return "uint32";
        }
        return "unknown";
    }

    inline double distanceAt(std::size_t index) const {
        std::size_t code = 0;
        switch (distance_storage) {
            case DistanceStorage::Float64: return distance_f64[index];
            case DistanceStorage::UInt8: code = distance_u8[index]; break;
            case DistanceStorage::UInt16: code = distance_u16[index]; break;
            case DistanceStorage::UInt32: code = distance_u32[index]; break;
        }
        return distance_codebook_size == 0
            ? static_cast<double>(code)
            : distance_codebook_ptr[code];
    }

    Clock::time_point stageStarted() const {
        return collect_diagnostics ? Clock::now() : Clock::time_point{};
    }

    double elapsedSeconds(const Clock::time_point& started) const {
        if (!collect_diagnostics) return 0.0;
        return std::chrono::duration<double>(Clock::now() - started).count();
    }

    std::string executionPath() const {
        if (small_k_fused_rounds > 0 && fast_score_evaluations == 0) {
            return "fused_exact_k2_shared_ties";
        }
        if (small_k_reynolds_rounds > 0 && fast_score_evaluations == 0) {
            return "reynolds_small_k";
        }
        if (adaptive_fallback_rounds == 0) return "fastpam";
        if (fast_score_evaluations == 0) return "reynolds";
        return "mixed";
    }

    py::list swapTrace() const {
        py::list trace;
        for (std::size_t index = 0; index < swap_entering.size(); ++index) {
            trace.append(py::make_tuple(
                swap_removed[index], swap_entering[index], swap_slots[index]));
        }
        return trace;
    }

    py::array_t<int> runclusterloopImpl(int output_offset) {
        if (worker_count == 1) {
            return runclusterloopSerialImpl(output_offset);
        }
        return runclusterloopParallelImpl(output_offset);
    }

    py::array_t<int> runclusterloopParallelImpl(int output_offset) {
        int* cent_ptr = centroids.mutable_data();
        int* assignment = clusterid.mutable_data();
        fast_score_evaluations = 0;
        classic_score_evaluations = 0;
        swap_rounds = 0;
        accepted_swaps = 0;
        adaptive_fallback_rounds = 0;
        two_pass_recovery_rounds = 0;
        small_k_fused_rounds = 0;
        small_k_fused_seconds = 0.0;
        small_k_reynolds_rounds = 0;
        exact_integer_rounds = 0;
        exact_fixed_point_rounds = 0;
        sign_aware_verified_rounds = 0;
        screened_candidate_highwater = 0;
        max_distance_seconds = 0.0;
        build_seconds = 0.0;
        assignment_seconds = 0.0;
        fast_screen_seconds = 0.0;
        exact_arbitration_seconds = 0.0;
        reynolds_fallback_seconds = 0.0;
        two_pass_recovery_seconds = 0.0;
        swap_update_seconds = 0.0;
        full_assignment_rounds = 0;
        incremental_update_rounds = 0;
        incremental_rescanned_rows = 0;
        incremental_update_seconds = 0.0;
        final_assignment_seconds = 0.0;
        swap_removed.clear();
        swap_entering.clear();
        swap_slots.clear();

        {
            py::gil_scoped_release release;
            if (npass > 0) {
                const auto max_started = stageStarted();
                maxdist = computeMaxDist();
                max_distance_seconds += elapsedSeconds(max_started);
                const auto build_started = stageStarted();
                if (unit_weights) {
                    buildInitialCentroids<true>(cent_ptr);
                } else {
                    buildInitialCentroids<false>(cent_ptr);
                }
                build_seconds += elapsedSeconds(build_started);
            } else {
                initializeMedoidFlags(cent_ptr);
            }

            PamCandidate best;
            bool accept_swap = false;
            bool assignments_ready = false;
            do {
                ++swap_rounds;
                if (!assignments_ready) {
                    const auto assignment_started = stageStarted();
                    assignToNearestMedoids(cent_ptr, assignment);
                    assignment_seconds += elapsedSeconds(assignment_started);
                    ++full_assignment_rounds;
                    assignments_ready = true;
                }

                if (nclusters == 2) {
                    ++small_k_fused_rounds;
                    const auto fused_start = stageStarted();
                    best = unit_weights
                        ? findBestFusedTwoMedoidSwap<true>(
                            assignment, is_medoid)
                        : findBestFusedTwoMedoidSwap<false>(
                            assignment, is_medoid);
                    small_k_fused_seconds += elapsedSeconds(fused_start);
                } else {
                    best = unit_weights
                        ? findBestVerifiedFastSwap<true>(
                            cent_ptr, assignment, is_medoid)
                        : findBestVerifiedFastSwap<false>(
                            cent_ptr, assignment, is_medoid);
                }
                accept_swap = pam_is_significant_improvement(best.score);
                if (accept_swap) {
                    const auto update_started = stageStarted();
                    const int removed = cent_ptr[best.k_slot];
                    if (collect_diagnostics) {
                        swap_removed.push_back(removed);
                        swap_entering.push_back(best.h);
                        swap_slots.push_back(best.k_slot);
                    }
                    cent_ptr[best.k_slot] = best.h;
                    bool removed_is_still_used = false;
                    for (int k = 0; k < nclusters; ++k) {
                        if (cent_ptr[k] == removed) {
                            removed_is_still_used = true;
                            break;
                        }
                    }
                    if (!removed_is_still_used) is_medoid[removed] = 0;
                    is_medoid[best.h] = 1;
                    ++accepted_swaps;
                    swap_update_seconds += elapsedSeconds(update_started);

                    const auto incremental_started = stageStarted();
                    incrementalUpdateNearestMedoids(
                        cent_ptr, best.k_slot, assignment);
                    incremental_update_seconds += elapsedSeconds(
                        incremental_started);
                    ++incremental_update_rounds;
                }
            } while (accept_swap);

            final_objective = unit_weights
                ? currentObjective<true>()
                : currentObjective<false>();

            const auto output_started = stageStarted();
#ifdef _OPENMP
#pragma omp parallel for num_threads(worker_count) if(worker_count > 1) schedule(static)
#endif
            for (int i = 0; i < nelements; ++i) {
                assignment[i] = cent_ptr[assignment[i]] + output_offset;
            }
            final_assignment_seconds += elapsedSeconds(output_started);
        }
        return clusterid;
    }

    py::array_t<int> runclusterloopSerialImpl(int output_offset) {
        int* cent_ptr = centroids.mutable_data();
        int* assignment = clusterid.mutable_data();
        fast_score_evaluations = 0;
        classic_score_evaluations = 0;
        swap_rounds = 0;
        accepted_swaps = 0;
        adaptive_fallback_rounds = 0;
        two_pass_recovery_rounds = 0;
        small_k_fused_rounds = 0;
        small_k_fused_seconds = 0.0;
        small_k_reynolds_rounds = 0;
        exact_integer_rounds = 0;
        exact_fixed_point_rounds = 0;
        sign_aware_verified_rounds = 0;
        screened_candidate_highwater = 0;
        max_distance_seconds = 0.0;
        build_seconds = 0.0;
        assignment_seconds = 0.0;
        fast_screen_seconds = 0.0;
        exact_arbitration_seconds = 0.0;
        reynolds_fallback_seconds = 0.0;
        two_pass_recovery_seconds = 0.0;
        swap_update_seconds = 0.0;
        full_assignment_rounds = 0;
        incremental_update_rounds = 0;
        incremental_rescanned_rows = 0;
        incremental_update_seconds = 0.0;
        final_assignment_seconds = 0.0;
        swap_removed.clear();
        swap_entering.clear();
        swap_slots.clear();

        {
            py::gil_scoped_release release;
            if (npass > 0) {
                const auto max_started = stageStarted();
                maxdist = computeMaxDistSerial();
                max_distance_seconds += elapsedSeconds(max_started);
                const auto build_started = stageStarted();
                if (unit_weights) {
                    buildInitialCentroidsSerial<true>(cent_ptr);
                } else {
                    buildInitialCentroidsSerial<false>(cent_ptr);
                }
                build_seconds += elapsedSeconds(build_started);
            } else {
                initializeMedoidFlags(cent_ptr);
            }

            bool accept_swap = false;
            bool assignments_ready = false;
            do {
                ++swap_rounds;
                if (!assignments_ready) {
                    const auto assignment_started = stageStarted();
                    assignToNearestMedoidsSerial(cent_ptr, assignment);
                    assignment_seconds += elapsedSeconds(assignment_started);
                    ++full_assignment_rounds;
                    assignments_ready = true;
                }

                PamCandidate best;
                if (nclusters == 2) {
                    ++small_k_fused_rounds;
                    const auto fused_start = stageStarted();
                    best = unit_weights
                        ? findBestFusedTwoMedoidSwapSerial<true>(
                            assignment, is_medoid)
                        : findBestFusedTwoMedoidSwapSerial<false>(
                            assignment, is_medoid);
                    small_k_fused_seconds += elapsedSeconds(fused_start);
                } else {
                    best = unit_weights
                        ? findBestVerifiedFastSwapSerial<true>(
                            cent_ptr, assignment, is_medoid)
                        : findBestVerifiedFastSwapSerial<false>(
                            cent_ptr, assignment, is_medoid);
                }
                accept_swap = pam_is_significant_improvement(best.score);
                if (accept_swap) {
                    const auto update_started = stageStarted();
                    const int removed = cent_ptr[best.k_slot];
                    if (collect_diagnostics) {
                        swap_removed.push_back(removed);
                        swap_entering.push_back(best.h);
                        swap_slots.push_back(best.k_slot);
                    }
                    cent_ptr[best.k_slot] = best.h;
                    bool removed_is_still_used = false;
                    for (int k = 0; k < nclusters; ++k) {
                        if (cent_ptr[k] == removed) {
                            removed_is_still_used = true;
                            break;
                        }
                    }
                    if (!removed_is_still_used) is_medoid[removed] = 0;
                    is_medoid[best.h] = 1;
                    ++accepted_swaps;
                    swap_update_seconds += elapsedSeconds(update_started);

                    const auto incremental_started = stageStarted();
                    incrementalUpdateNearestMedoidsSerial(
                        cent_ptr, best.k_slot, assignment);
                    incremental_update_seconds += elapsedSeconds(
                        incremental_started);
                    ++incremental_update_rounds;
                }
            } while (accept_swap);

            final_objective = unit_weights
                ? currentObjective<true>()
                : currentObjective<false>();
            const auto output_started = stageStarted();
            for (int i = 0; i < nelements; ++i) {
                assignment[i] = cent_ptr[assignment[i]] + output_offset;
            }
            final_assignment_seconds += elapsedSeconds(output_started);
        }
        return clusterid;
    }

    struct alignas(64) FastWorkspace {
        explicit FastWorkspace(int k)
            : delta(k) {}

        void reset() {
            if (abs_delta.size() != delta.size()) {
                abs_delta.resize(delta.size());
                delta_count.resize(delta.size());
            }
            base = 0.0;
            abs_base = 0.0;
            base_count = 0;
            max_candidate_distance = 0.0;
            minimum_quantum_exponent = INT_MAX;
            all_distances_are_nonnegative_integers = true;
            all_distances_are_finite_nonnegative = true;
            std::fill(delta.begin(), delta.end(), 0.0);
            std::fill(abs_delta.begin(), abs_delta.end(), 0.0);
            std::fill(delta_count.begin(), delta_count.end(), 0);
        }

        void resetSignAware() {
            base = 0.0;
            sign_aware_abs_base = 0.0;
            std::fill(delta.begin(), delta.end(), 0.0);
        }

        double base = 0.0;
        double abs_base = 0.0;
        double sign_aware_abs_base = 0.0;
        std::uint32_t base_count = 0;
        double max_candidate_distance = 0.0;
        int minimum_quantum_exponent = INT_MAX;
        bool all_distances_are_nonnegative_integers = true;
        bool all_distances_are_finite_nonnegative = true;
        std::vector<double> delta;
        std::vector<double> abs_delta;
        std::vector<std::uint32_t> delta_count;
    };

    template <bool UnitWeights>
    inline double weightAt(int index) const {
        if constexpr (UnitWeights) return 1.0;
        return wt_ptr[index];
    }

    template <bool UnitWeights>
    double currentObjective() const {
        double objective = 0.0;
        for (int i = 0; i < nelements; ++i) {
            objective += weightAt<UnitWeights>(i) * dysma[i];
        }
        return objective;
    }

    inline double get_dist(int i, int j) const {
        if (!use_condensed)
            return distanceAt(static_cast<std::size_t>(i) * nelements + j);
        if (i == j) return 0.0;
        int a = i;
        int b = j;
        if (a > b) std::swap(a, b);
        return distanceAt(
            static_cast<std::size_t>(a) * (2 * nelements - a - 1) / 2
            + (b - a - 1));
    }

    template <typename Callback>
    inline void forEachDistanceInRow(int row, Callback&& callback) const {
        if (distance_storage == DistanceStorage::Float64) {
            pam_for_each_distance_in_row(
                nelements, row,
                use_condensed ? nullptr : distance_f64,
                use_condensed ? distance_f64 : nullptr,
                use_condensed,
                std::forward<Callback>(callback));
            return;
        }
        if (!use_condensed) {
            const std::size_t offset = static_cast<std::size_t>(row) * nelements;
            for (int j = 0; j < nelements; ++j)
                callback(j, distanceAt(offset + j));
            return;
        }
        std::size_t index = row > 0 ? static_cast<std::size_t>(row - 1) : 0;
        for (int j = 0; j < row; ++j) {
            callback(j, distanceAt(index));
            index += static_cast<std::size_t>(nelements - j - 2);
        }
        callback(row, 0.0);
        index = static_cast<std::size_t>(row) *
            (2 * nelements - row - 1) / 2;
        for (int j = row + 1; j < nelements; ++j) {
            callback(j, distanceAt(index));
            ++index;
        }
    }

    double computeMaxDistSerial() {
        double maximum = 0.0;
        bool distances_are_nonnegative_integers = true;
        bool distances_are_finite_nonnegative = true;
        bool fixed_point_may_be_exact = true;
        int minimum_quantum_exponent = INT_MAX;
        const auto inspect = [&](double value) {
            maximum = std::max(maximum, value);
            const bool finite_nonnegative =
                std::isfinite(value) && value >= 0.0;
            const bool integer_value =
                finite_nonnegative && value == std::trunc(value);
            if (!integer_value) {
                distances_are_nonnegative_integers = false;
            }
            if (!finite_nonnegative) {
                distances_are_finite_nonnegative = false;
                fixed_point_may_be_exact = false;
            } else if (!integer_value && fixed_point_may_be_exact) {
                minimum_quantum_exponent = std::min(
                    minimum_quantum_exponent,
                    pam_binary_quantum_exponent(value));
            }
            if (fixed_point_may_be_exact &&
                minimum_quantum_exponent != INT_MAX &&
                !fixedPointScoresAreExact(
                    maximum, minimum_quantum_exponent)) {
                fixed_point_may_be_exact = false;
                minimum_quantum_exponent =
                    std::numeric_limits<double>::min_exponent -
                    std::numeric_limits<double>::digits;
            }
        };

        if (use_condensed) {
            const std::size_t pair_count = diss.size();
            for (std::size_t index = 0; index < pair_count; ++index) {
                inspect(distanceAt(index));
            }
        } else {
            for (int i = 0; i < nelements; ++i) {
                for (int j = i + 1; j < nelements; ++j) {
                    inspect(get_dist(i, j));
                }
            }
        }

        exact_integer_scores = distances_are_nonnegative_integers &&
            weights_are_nonnegative_integers &&
            pam_upward_multiply(
                2.0 * maximum, sum_abs_weights_bound) <
                std::ldexp(1.0, 52);
        exact_fixed_point_scores = fixed_point_may_be_exact &&
            distances_are_finite_nonnegative &&
            fixedPointScoresAreExact(maximum, minimum_quantum_exponent);
        input_distance_properties_known = true;
        input_distances_are_nonnegative_integers =
            distances_are_nonnegative_integers;
        input_distances_are_finite_nonnegative =
            distances_are_finite_nonnegative;
        minimum_input_quantum_exponent = minimum_quantum_exponent;
        maximum_input_distance = maximum;
        return 1.1 * maximum + 1.0;
    }

    double computeMaxDist() {
        double maximum = 0.0;
        int distances_are_nonnegative_integers = 1;
        int distances_are_finite_nonnegative = 1;
        int minimum_quantum_exponent = INT_MAX;
        if (use_condensed) {
            const std::ptrdiff_t pair_count =
                static_cast<std::ptrdiff_t>(diss.size());
#ifdef _OPENMP
#pragma omp parallel for num_threads(worker_count) if(worker_count > 1) reduction(max:maximum) reduction(&:distances_are_nonnegative_integers,distances_are_finite_nonnegative) reduction(min:minimum_quantum_exponent) schedule(static)
#endif
            for (std::ptrdiff_t index = 0; index < pair_count; ++index) {
                const double value = distanceAt(static_cast<std::size_t>(index));
                maximum = std::max(maximum, value);
                if (!std::isfinite(value) || value < 0.0 ||
                    value != std::trunc(value)) {
                    distances_are_nonnegative_integers = 0;
                }
                if (!std::isfinite(value) || value < 0.0) {
                    distances_are_finite_nonnegative = 0;
                } else if (value != std::trunc(value)) {
                    minimum_quantum_exponent = std::min(
                        minimum_quantum_exponent,
                        pam_binary_quantum_exponent(value));
                }
            }
            exact_integer_scores = distances_are_nonnegative_integers != 0 &&
                weights_are_nonnegative_integers &&
                pam_upward_multiply(
                    2.0 * maximum, sum_abs_weights_bound) <
                    std::ldexp(1.0, 52);
            exact_fixed_point_scores = distances_are_finite_nonnegative != 0 &&
                fixedPointScoresAreExact(maximum, minimum_quantum_exponent);
            input_distance_properties_known = true;
            input_distances_are_nonnegative_integers =
                distances_are_nonnegative_integers != 0;
            input_distances_are_finite_nonnegative =
                distances_are_finite_nonnegative != 0;
            minimum_input_quantum_exponent = minimum_quantum_exponent;
            maximum_input_distance = maximum;
            return 1.1 * maximum + 1.0;
        }
#ifdef _OPENMP
#pragma omp parallel for num_threads(worker_count) if(worker_count > 1) reduction(max:maximum) reduction(&:distances_are_nonnegative_integers,distances_are_finite_nonnegative) reduction(min:minimum_quantum_exponent) schedule(static)
#endif
        for (int i = 0; i < nelements; ++i) {
            for (int j = i + 1; j < nelements; ++j) {
                const double value = get_dist(i, j);
                maximum = std::max(maximum, value);
                if (!std::isfinite(value) || value < 0.0 ||
                    value != std::trunc(value)) {
                    distances_are_nonnegative_integers = 0;
                }
                if (!std::isfinite(value) || value < 0.0) {
                    distances_are_finite_nonnegative = 0;
                } else if (value != std::trunc(value)) {
                    minimum_quantum_exponent = std::min(
                        minimum_quantum_exponent,
                        pam_binary_quantum_exponent(value));
                }
            }
        }
        exact_integer_scores = distances_are_nonnegative_integers != 0 &&
            weights_are_nonnegative_integers &&
            pam_upward_multiply(
                2.0 * maximum, sum_abs_weights_bound) <
                std::ldexp(1.0, 52);
        exact_fixed_point_scores = distances_are_finite_nonnegative != 0 &&
            fixedPointScoresAreExact(maximum, minimum_quantum_exponent);
        input_distance_properties_known = true;
        input_distances_are_nonnegative_integers =
            distances_are_nonnegative_integers != 0;
        input_distances_are_finite_nonnegative =
            distances_are_finite_nonnegative != 0;
        minimum_input_quantum_exponent = minimum_quantum_exponent;
        maximum_input_distance = maximum;
        return 1.1 * maximum + 1.0;
    }

    bool fixedPointScoresAreExact(
        double maximum_distance, int minimum_quantum_exponent) const {
        if (!weights_are_nonnegative_integers ||
            !std::isfinite(maximum_distance) || maximum_distance < 0.0) {
            return false;
        }
        if (maximum_distance == 0.0) return true;
        if (minimum_quantum_exponent == INT_MAX) return false;
        const long double scaled_bound = std::ldexp(
            2.0L * static_cast<long double>(maximum_distance) *
                static_cast<long double>(sum_abs_weights_bound),
            -minimum_quantum_exponent);
        return std::isfinite(scaled_bound) &&
            scaled_bound < std::ldexp(1.0L, 52);
    }

    template <bool UnitWeights>
    double buildGainFloat64(int row) const {
        double gain = 0.0;
        const auto accumulate = [&](int j, double distance) {
            gain += weightAt<UnitWeights>(j) *
                std::max(0.0, dysma[j] - distance);
        };
        if (!use_condensed) {
            const double* row_data = distance_f64 +
                static_cast<std::size_t>(row) * nelements;
            for (int j = 0; j < nelements; ++j) {
                accumulate(j, row_data[j]);
            }
            return gain;
        }

        std::size_t index = row > 0
            ? static_cast<std::size_t>(row - 1)
            : 0;
        for (int j = 0; j < row; ++j) {
            accumulate(j, distance_f64[index]);
            index += static_cast<std::size_t>(nelements - j - 2);
        }
        accumulate(row, 0.0);
        index = static_cast<std::size_t>(row) *
            (2 * nelements - row - 1) / 2;
        for (int j = row + 1; j < nelements; ++j) {
            accumulate(j, distance_f64[index]);
            ++index;
        }
        return gain;
    }

    void updateBuildNearestFloat64(int row) {
        const auto update = [&](int j, double distance) {
            dysma[j] = std::min(dysma[j], distance);
        };
        if (!use_condensed) {
            const double* row_data = distance_f64 +
                static_cast<std::size_t>(row) * nelements;
            for (int j = 0; j < nelements; ++j) update(j, row_data[j]);
            return;
        }

        std::size_t index = row > 0
            ? static_cast<std::size_t>(row - 1)
            : 0;
        for (int j = 0; j < row; ++j) {
            update(j, distance_f64[index]);
            index += static_cast<std::size_t>(nelements - j - 2);
        }
        update(row, 0.0);
        index = static_cast<std::size_t>(row) *
            (2 * nelements - row - 1) / 2;
        for (int j = row + 1; j < nelements; ++j) {
            update(j, distance_f64[index]);
            ++index;
        }
    }

    template <bool UnitWeights>
    void buildInitialCentroidsSerial(int* cent_ptr) {
        std::fill(is_medoid.begin(), is_medoid.end(), 0);
        std::fill(dysma.begin(), dysma.end(), maxdist);

        for (int selected = 0; selected < nclusters; ++selected) {
            double best_gain = -std::numeric_limits<double>::infinity();
            int best_index = -1;
            for (int i = 0; i < nelements; ++i) {
                if (is_medoid[i]) continue;
                double gain = 0.0;
                if (distance_storage == DistanceStorage::Float64) {
                    gain = buildGainFloat64<UnitWeights>(i);
                } else {
                    forEachDistanceInRow(i, [&](int j, double distance) {
                        const double improvement = dysma[j] - distance;
                        gain += weightAt<UnitWeights>(j) *
                            std::max(0.0, improvement);
                    });
                }
                if (gain > best_gain ||
                    (gain == best_gain &&
                     buildTieKey(i) > buildTieKey(best_index))) {
                    best_gain = gain;
                    best_index = i;
                }
            }
            is_medoid[best_index] = 1;
            cent_ptr[selected] = best_index;
            if (distance_storage == DistanceStorage::Float64) {
                updateBuildNearestFloat64(best_index);
            } else {
                forEachDistanceInRow(
                    best_index,
                    [&](int j, double distance) {
                        dysma[j] = std::min(dysma[j], distance);
                    });
            }
        }
    }

    template <bool UnitWeights>
    void buildInitialCentroids(int* cent_ptr) {
        std::fill(is_medoid.begin(), is_medoid.end(), 0);
        std::fill(dysma.begin(), dysma.end(), maxdist);

        for (int selected = 0; selected < nclusters; ++selected) {
            double best_gain = -std::numeric_limits<double>::infinity();
            int best_index = -1;
#ifdef _OPENMP
#pragma omp parallel num_threads(worker_count) if(worker_count > 1)
            {
                double local_gain = -std::numeric_limits<double>::infinity();
                int local_index = -1;
#pragma omp for schedule(static) nowait
                for (int i = 0; i < nelements; ++i) {
                    if (is_medoid[i]) continue;
                    double gain = 0.0;
                    if (distance_storage == DistanceStorage::Float64) {
                        gain = buildGainFloat64<UnitWeights>(i);
                    } else {
                        forEachDistanceInRow(i, [&](int j, double distance) {
                            const double improvement = dysma[j] - distance;
                            gain += weightAt<UnitWeights>(j) *
                                std::max(0.0, improvement);
                        });
                    }
                    if (gain > local_gain ||
                        (gain == local_gain &&
                         buildTieKey(i) > buildTieKey(local_index))) {
                        local_gain = gain;
                        local_index = i;
                    }
                }
#pragma omp critical
                {
                    if (local_index >= 0 &&
                        (local_gain > best_gain ||
                         (local_gain == best_gain &&
                          buildTieKey(local_index) > buildTieKey(best_index)))) {
                        best_gain = local_gain;
                        best_index = local_index;
                    }
                }
            }
#else
            for (int i = 0; i < nelements; ++i) {
                if (is_medoid[i]) continue;
                double gain = 0.0;
                forEachDistanceInRow(i, [&](int j, double distance) {
                    const double improvement = dysma[j] - distance;
                    gain += weightAt<UnitWeights>(j) *
                        std::max(0.0, improvement);
                });
                if (gain > best_gain ||
                    (gain == best_gain &&
                     buildTieKey(i) > buildTieKey(best_index))) {
                    best_gain = gain;
                    best_index = i;
                }
            }
#endif
            is_medoid[best_index] = 1;
            cent_ptr[selected] = best_index;
            if (distance_storage == DistanceStorage::Float64) {
                updateBuildNearestFloat64(best_index);
            } else {
                forEachDistanceInRow(
                    best_index,
                    [&](int j, double distance) {
                        dysma[j] = std::min(dysma[j], distance);
                    });
            }
        }
    }

    int buildTieKey(int index) const {
        if (index < 0) return -1;
        return build_tie_ptr == nullptr ? index : build_tie_ptr[index];
    }

    void initializeMedoidFlags(const int* cent_ptr) {
        std::fill(is_medoid.begin(), is_medoid.end(), 0);
        for (int k = 0; k < nclusters; ++k) {
            is_medoid[cent_ptr[k]] = 1;
        }
    }

    void assignToNearestMedoidsSerial(
        const int* cent_ptr, int* assignment) {
        double maximum_assignment_distance = 0.0;
        bool assignment_distances_are_integers =
            input_distance_properties_known
                ? input_distances_are_nonnegative_integers
                : true;
        bool assignment_distances_are_valid =
            input_distance_properties_known
                ? input_distances_are_finite_nonnegative
                : true;
        int minimum_assignment_quantum = input_distance_properties_known
            ? minimum_input_quantum_exponent
            : INT_MAX;
        for (int i = 0; i < nelements; ++i) {
            double nearest = maxdist;
            double second = maxdist;
            int nearest_slot = -1;
            int next_slot = -1;
            for (int k = 0; k < nclusters; ++k) {
                const double distance = get_dist(i, cent_ptr[k]);
                maximum_assignment_distance = std::max(
                    maximum_assignment_distance, distance);
                if (!input_distance_properties_known) {
                    if (!std::isfinite(distance) || distance < 0.0 ||
                        distance != std::trunc(distance)) {
                        assignment_distances_are_integers = false;
                    }
                    if (!std::isfinite(distance) || distance < 0.0) {
                        assignment_distances_are_valid = false;
                    } else if (distance != std::trunc(distance)) {
                        minimum_assignment_quantum = std::min(
                            minimum_assignment_quantum,
                            pam_binary_quantum_exponent(distance));
                    }
                }
                if (distance < nearest ||
                    (distance == nearest &&
                     (nearest_slot < 0 || k < nearest_slot))) {
                    second = nearest;
                    next_slot = nearest_slot;
                    nearest = distance;
                    nearest_slot = k;
                } else if (distance < second ||
                           (distance == second &&
                            (next_slot < 0 || k < next_slot))) {
                    second = distance;
                    next_slot = k;
                }
            }
            dysma[i] = nearest;
            dysmb[i] = second;
            assignment[i] = nearest_slot;
            second_slot[i] = next_slot;
        }
        max_assignment_distance = maximum_assignment_distance;
        assignment_distances_are_nonnegative_integers =
            assignment_distances_are_integers;
        assignment_distances_are_finite_nonnegative =
            assignment_distances_are_valid;
        minimum_assignment_quantum_exponent = minimum_assignment_quantum;
    }

    void assignToNearestMedoids(const int* cent_ptr, int* assignment) {
        double maximum_assignment_distance = 0.0;
        int assignment_distances_are_integers =
            input_distance_properties_known
                ? static_cast<int>(input_distances_are_nonnegative_integers)
                : 1;
        int assignment_distances_are_valid =
            input_distance_properties_known
                ? static_cast<int>(input_distances_are_finite_nonnegative)
                : 1;
        int minimum_assignment_quantum = input_distance_properties_known
            ? minimum_input_quantum_exponent
            : INT_MAX;
#ifdef _OPENMP
#pragma omp parallel for num_threads(worker_count) if(worker_count > 1) reduction(max:maximum_assignment_distance) reduction(&:assignment_distances_are_integers,assignment_distances_are_valid) reduction(min:minimum_assignment_quantum) schedule(static)
#endif
        for (int i = 0; i < nelements; ++i) {
            double nearest = maxdist;
            double second = maxdist;
            int nearest_slot = -1;
            int next_slot = -1;
            for (int k = 0; k < nclusters; ++k) {
                const double distance = get_dist(i, cent_ptr[k]);
                maximum_assignment_distance = std::max(
                    maximum_assignment_distance, distance);
                if (!input_distance_properties_known) {
                    if (!std::isfinite(distance) || distance < 0.0 ||
                        distance != std::trunc(distance)) {
                        assignment_distances_are_integers = 0;
                    }
                    if (!std::isfinite(distance) || distance < 0.0) {
                        assignment_distances_are_valid = 0;
                    } else if (distance != std::trunc(distance)) {
                        minimum_assignment_quantum = std::min(
                            minimum_assignment_quantum,
                            pam_binary_quantum_exponent(distance));
                    }
                }
                if (distance < nearest ||
                    (distance == nearest &&
                     (nearest_slot < 0 || k < nearest_slot))) {
                    second = nearest;
                    next_slot = nearest_slot;
                    nearest = distance;
                    nearest_slot = k;
                } else if (distance < second ||
                           (distance == second &&
                            (next_slot < 0 || k < next_slot))) {
                    second = distance;
                    next_slot = k;
                }
            }
            dysma[i] = nearest;
            dysmb[i] = second;
            assignment[i] = nearest_slot;
            second_slot[i] = next_slot;
        }
        max_assignment_distance = maximum_assignment_distance;
        assignment_distances_are_nonnegative_integers =
            assignment_distances_are_integers != 0;
        assignment_distances_are_finite_nonnegative =
            assignment_distances_are_valid != 0;
        minimum_assignment_quantum_exponent = minimum_assignment_quantum;
    }

    void incrementalUpdateNearestMedoids(
        const int* cent_ptr, int replaced_slot, int* assignment) {
        std::size_t rescanned_rows = 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(worker_count) if(worker_count > 1) reduction(+:rescanned_rows) schedule(static)
#endif
        for (int i = 0; i < nelements; ++i) {
            double nearest = dysma[i];
            double second = dysmb[i];
            int nearest_slot = assignment[i];
            int next_slot = second_slot[i];

            if (nearest_slot == replaced_slot || next_slot == replaced_slot) {
                nearest = maxdist;
                second = maxdist;
                nearest_slot = -1;
                next_slot = -1;
                for (int k = 0; k < nclusters; ++k) {
                    const double distance = get_dist(i, cent_ptr[k]);
                    if (distance < nearest ||
                        (distance == nearest &&
                         (nearest_slot < 0 || k < nearest_slot))) {
                        second = nearest;
                        next_slot = nearest_slot;
                        nearest = distance;
                        nearest_slot = k;
                    } else if (distance < second ||
                               (distance == second &&
                                (next_slot < 0 || k < next_slot))) {
                        second = distance;
                        next_slot = k;
                    }
                }
                ++rescanned_rows;
            } else {
                const double distance = get_dist(i, cent_ptr[replaced_slot]);
                if (distance < nearest ||
                    (distance == nearest && replaced_slot < nearest_slot)) {
                    second = nearest;
                    next_slot = nearest_slot;
                    nearest = distance;
                    nearest_slot = replaced_slot;
                } else if (distance < second ||
                           (distance == second && replaced_slot < next_slot)) {
                    second = distance;
                    next_slot = replaced_slot;
                }
            }

            dysma[i] = nearest;
            dysmb[i] = second;
            assignment[i] = nearest_slot;
            second_slot[i] = next_slot;
        }
        incremental_rescanned_rows += rescanned_rows;
    }

    void incrementalUpdateNearestMedoidsSerial(
        const int* cent_ptr, int replaced_slot, int* assignment) {
        std::size_t rescanned_rows = 0;
        for (int i = 0; i < nelements; ++i) {
            double nearest = dysma[i];
            double second = dysmb[i];
            int nearest_slot = assignment[i];
            int next_slot = second_slot[i];

            if (nearest_slot == replaced_slot || next_slot == replaced_slot) {
                nearest = maxdist;
                second = maxdist;
                nearest_slot = -1;
                next_slot = -1;
                for (int k = 0; k < nclusters; ++k) {
                    const double distance = get_dist(i, cent_ptr[k]);
                    if (distance < nearest ||
                        (distance == nearest &&
                         (nearest_slot < 0 || k < nearest_slot))) {
                        second = nearest;
                        next_slot = nearest_slot;
                        nearest = distance;
                        nearest_slot = k;
                    } else if (distance < second ||
                               (distance == second &&
                                (next_slot < 0 || k < next_slot))) {
                        second = distance;
                        next_slot = k;
                    }
                }
                ++rescanned_rows;
            } else {
                const double distance = get_dist(i, cent_ptr[replaced_slot]);
                if (distance < nearest ||
                    (distance == nearest && replaced_slot < nearest_slot)) {
                    second = nearest;
                    next_slot = nearest_slot;
                    nearest = distance;
                    nearest_slot = replaced_slot;
                } else if (distance < second ||
                           (distance == second && replaced_slot < next_slot)) {
                    second = distance;
                    next_slot = replaced_slot;
                }
            }

            dysma[i] = nearest;
            dysmb[i] = second;
            assignment[i] = nearest_slot;
            second_slot[i] = next_slot;
        }
        incremental_rescanned_rows += rescanned_rows;
    }

    template <bool UnitWeights>
    void computeSignAwareFastScores(
        int h, const int* assignment, FastWorkspace& workspace) const {
        workspace.resetSignAware();
        forEachDistanceInRow(h, [&](int j, double candidate_distance) {
            const double nearest = dysma[j];
            const double weight = weightAt<UnitWeights>(j);
            if (candidate_distance < nearest) {
                workspace.base += weight * (candidate_distance - nearest);
            } else {
                workspace.delta[assignment[j]] += weight *
                    (std::min(dysmb[j], candidate_distance) - nearest);
            }
        });
        workspace.sign_aware_abs_base = pam_upward_multiply(
            -workspace.base, sign_aware_abs_sum_scale);
    }

    double signAwareScoreErrorBound(
        const FastWorkspace& workspace, int k_slot) const {
        const double correction = workspace.delta[k_slot];
        if (!std::isfinite(workspace.sign_aware_abs_base) ||
            !std::isfinite(correction) || correction < 0.0) {
            return std::numeric_limits<double>::infinity();
        }
        const double abs_delta = pam_upward_multiply(
            correction, sign_aware_abs_sum_scale);
        const double total_abs = pam_upward_add(
            workspace.sign_aware_abs_base, abs_delta);
        const double classic_error = pam_upward_multiply(
            sign_aware_sum_gamma, total_abs);
        const double fast_error = pam_upward_multiply(
            sign_aware_sum_gamma, total_abs);
        const double final_add_error = pam_upward_multiply(
            pam_gamma_bound(1),
            pam_upward_add(std::abs(workspace.base), correction));
        return pam_upward_add(
            pam_upward_add(classic_error, fast_error),
            pam_upward_add(final_add_error, sign_aware_underflow_bound));
    }

    template <bool UnitWeights>
    void computeFastScores(int h, const int* assignment,
                           FastWorkspace& workspace) const {
        workspace.reset();
        if (input_distance_properties_known) {
            workspace.all_distances_are_nonnegative_integers =
                input_distances_are_nonnegative_integers;
            workspace.all_distances_are_finite_nonnegative =
                input_distances_are_finite_nonnegative;
            workspace.minimum_quantum_exponent =
                minimum_input_quantum_exponent;
            workspace.max_candidate_distance = maximum_input_distance;
        }
        forEachDistanceInRow(h, [&](int j, double candidate_distance) {
            const double nearest = dysma[j];
            const double weight = weightAt<UnitWeights>(j);
            if (candidate_distance < nearest) {
                const double contribution =
                    weight * (candidate_distance - nearest);
                workspace.base += contribution;
                if (contribution != 0.0) {
                    workspace.abs_base += std::abs(contribution);
                    ++workspace.base_count;
                }
            } else {
                const int slot = assignment[j];
                const double contribution = weight *
                    (std::min(dysmb[j], candidate_distance) - nearest);
                workspace.delta[slot] += contribution;
                if (contribution != 0.0) {
                    workspace.abs_delta[slot] += std::abs(contribution);
                    ++workspace.delta_count[slot];
                }
            }
            if (!input_distance_properties_known) {
                workspace.max_candidate_distance = std::max(
                    workspace.max_candidate_distance,
                    std::abs(candidate_distance));
                if (!std::isfinite(candidate_distance) ||
                    candidate_distance < 0.0 ||
                    candidate_distance != std::trunc(candidate_distance)) {
                    workspace.all_distances_are_nonnegative_integers = false;
                }
                if (!std::isfinite(candidate_distance) ||
                    candidate_distance < 0.0) {
                    workspace.all_distances_are_finite_nonnegative = false;
                } else if (candidate_distance != std::trunc(candidate_distance)) {
                    workspace.minimum_quantum_exponent = std::min(
                        workspace.minimum_quantum_exponent,
                        pam_binary_quantum_exponent(candidate_distance));
                }
            }
        });
    }

    double fastScoreErrorBound(
        const FastWorkspace& workspace, int k_slot) const {
        if (exact_integer_scores || exact_fixed_point_scores) return 0.0;
        const std::size_t base_count = workspace.base_count;
        const std::size_t delta_count = workspace.delta_count[k_slot];
        const double abs_base = pam_upper_abs_sum(
            workspace.abs_base,
            base_count > 0 ? base_count - 1 : 0);
        const double abs_delta = pam_upper_abs_sum(
            workspace.abs_delta[k_slot],
            delta_count > 0 ? delta_count - 1 : 0);
        if (!std::isfinite(abs_base) || !std::isfinite(abs_delta) ||
            !std::isfinite(workspace.base) ||
            !std::isfinite(workspace.delta[k_slot])) {
            return std::numeric_limits<double>::infinity();
        }

        const std::size_t total_count = base_count + delta_count;
        if (total_count == 0) return 0.0;

        const double total_abs = pam_upward_add(abs_base, abs_delta);
        const double classic_error = pam_upward_multiply(
            pam_gamma_bound(total_count), total_abs);
        const double fast_base_error = pam_upward_multiply(
            pam_gamma_bound(base_count), abs_base);
        const double fast_delta_error = pam_upward_multiply(
            pam_gamma_bound(delta_count), abs_delta);
        const double final_operand_abs = pam_upward_add(
            std::abs(workspace.base),
            std::abs(workspace.delta[k_slot]));
        const double final_add_error = pam_upward_multiply(
            pam_gamma_bound(1), final_operand_abs);
        const double underflow_error = pam_upward_multiply(
            4.0 * static_cast<double>(total_count + 1),
            std::numeric_limits<double>::denorm_min());
        return pam_upward_add(
            pam_upward_add(classic_error, fast_base_error),
            pam_upward_add(
                fast_delta_error,
                pam_upward_add(final_add_error, underflow_error)));
    }

    template <bool UnitWeights>
    double classicSwapScore(int h, int k, const int* cent_ptr,
                            const int* assignment) const {
        return pam_classic_swap_score(
            h, k, cent_ptr, assignment,
            dysma.data(), dysmb.data(),
            [this](int index) {
                return weightAt<UnitWeights>(index);
            },
            [this](int i, int j) { return get_dist(i, j); },
            [self = this](int row, const auto& callback) {
                self->forEachDistanceInRow(row, callback);
            });
    }

    template <bool UnitWeights>
    PamCandidate findBestClassicSwapSerial(
        const int* cent_ptr,
        const int* assignment,
        const std::vector<std::uint8_t>& medoid_flags) {
        PamCandidate best;
        for (int h = 0; h < nelements; ++h) {
            if (medoid_flags[h]) continue;
            for (int k = 0; k < nclusters; ++k) {
                const double score = classicSwapScore<UnitWeights>(
                    h, k, cent_ptr, assignment);
                ++classic_score_evaluations;
                pam_consider_candidate(best, score, h, k);
            }
        }
        return best;
    }

    template <bool UnitWeights>
    void fusedTwoMedoidSwapScores(
        int h,
        const int* assignment,
        double& score_zero,
        double& score_one) const {
        score_zero = 0.0;
        score_one = 0.0;
        forEachDistanceInRow(h, [&](int j, double candidate_distance) {
            const double nearest = dysma[j];
            const double second = dysmb[j];
            const double weight = weightAt<UnitWeights>(j);
            const bool nearest_is_tied = second == nearest;
            if (nearest_is_tied) {
                const double replacement = second > candidate_distance
                    ? candidate_distance
                    : second;
                const double contribution =
                    weight * (-nearest + replacement);
                score_zero += contribution;
                score_one += contribution;
                return;
            }

            double contribution_zero = 0.0;
            if (assignment[j] == 0) {
                const double replacement = second > candidate_distance
                    ? candidate_distance
                    : second;
                contribution_zero = weight * (-nearest + replacement);
            } else if (candidate_distance < nearest) {
                contribution_zero = weight * (-nearest + candidate_distance);
            }
            score_zero += contribution_zero;

            double contribution_one = 0.0;
            if (assignment[j] == 1) {
                const double replacement = second > candidate_distance
                    ? candidate_distance
                    : second;
                contribution_one = weight * (-nearest + replacement);
            } else if (candidate_distance < nearest) {
                contribution_one = weight * (-nearest + candidate_distance);
            }
            score_one += contribution_one;
        });
    }

    template <bool UnitWeights>
    PamCandidate findBestFusedTwoMedoidSwapSerial(
        const int* assignment,
        const std::vector<std::uint8_t>& medoid_flags) {
        PamCandidate best;
        for (int h = 0; h < nelements; ++h) {
            if (medoid_flags[h]) continue;
            double score_zero = 0.0;
            double score_one = 0.0;
            fusedTwoMedoidSwapScores<UnitWeights>(
                h, assignment, score_zero, score_one);
            classic_score_evaluations += 2;
            pam_consider_candidate(best, score_zero, h, 0);
            pam_consider_candidate(best, score_one, h, 1);
        }
        return best;
    }

    template <bool UnitWeights>
    PamCandidate findBestFusedTwoMedoidSwap(
        const int* assignment,
        const std::vector<std::uint8_t>& medoid_flags) {
        PamCandidate best;
#ifdef _OPENMP
        std::size_t exact_count = 0;
        std::fill(thread_best.begin(), thread_best.end(), PamCandidate{});
#pragma omp parallel num_threads(worker_count) if(worker_count > 1) reduction(+:exact_count)
        {
            PamCandidate local;
#pragma omp for schedule(static) nowait
            for (int h = 0; h < nelements; ++h) {
                if (medoid_flags[h]) continue;
                double score_zero = 0.0;
                double score_one = 0.0;
                fusedTwoMedoidSwapScores<UnitWeights>(
                    h, assignment, score_zero, score_one);
                exact_count += 2;
                pam_consider_candidate(local, score_zero, h, 0);
                pam_consider_candidate(local, score_one, h, 1);
            }
            thread_best[omp_get_thread_num()] = local;
        }
        classic_score_evaluations += exact_count;
        for (const PamCandidate& local : thread_best) {
            if (local.h >= 0) {
                pam_consider_candidate(
                    best, local.score, local.h, local.k_slot);
            }
        }
#else
        return findBestFusedTwoMedoidSwapSerial<UnitWeights>(
            assignment, medoid_flags);
#endif
        return best;
    }

    template <bool UnitWeights>
    PamCandidate findBestClassicSwap(
        const int* cent_ptr,
        const int* assignment,
        const std::vector<std::uint8_t>& is_medoid) {
        PamCandidate best;
#ifdef _OPENMP
        std::size_t exact_count = 0;
        std::fill(thread_best.begin(), thread_best.end(), PamCandidate{});
#pragma omp parallel num_threads(worker_count) if(worker_count > 1) reduction(+:exact_count)
        {
            PamCandidate local;
#pragma omp for schedule(static) nowait
            for (int h = 0; h < nelements; ++h) {
                if (is_medoid[h]) continue;
                for (int k = 0; k < nclusters; ++k) {
                    const double score = classicSwapScore<UnitWeights>(
                        h, k, cent_ptr, assignment);
                    ++exact_count;
                    pam_consider_candidate(local, score, h, k);
                }
            }
            thread_best[omp_get_thread_num()] = local;
        }
        classic_score_evaluations += exact_count;
        for (const PamCandidate& local : thread_best) {
            if (local.h >= 0) {
                pam_consider_candidate(
                    best, local.score, local.h, local.k_slot);
            }
        }
#else
        for (int h = 0; h < nelements; ++h) {
            if (is_medoid[h]) continue;
            for (int k = 0; k < nclusters; ++k) {
                const double score = classicSwapScore<UnitWeights>(
                    h, k, cent_ptr, assignment);
                ++classic_score_evaluations;
                pam_consider_candidate(best, score, h, k);
            }
        }
#endif
        return best;
    }

    template <bool UnitWeights>
    PamCandidate findBestVerifiedSecondPassSerial(
        const int* cent_ptr,
        const int* assignment,
        const std::vector<std::uint8_t>& medoid_flags,
        double global_upper) {
        const auto started = stageStarted();
        ++two_pass_recovery_rounds;
        PamCandidate best;
        FastWorkspace& workspace = workspaces[0];

        for (int h = 0; h < nelements; ++h) {
            if (medoid_flags[h]) continue;
            computeFastScores<UnitWeights>(h, assignment, workspace);
            for (int k = 0; k < nclusters; ++k) {
                const double fast_score = workspace.base + workspace.delta[k];
                const double error = fastScoreErrorBound(workspace, k);
                const double lower_bound = std::isfinite(error)
                    ? std::nextafter(
                        fast_score - error,
                        -std::numeric_limits<double>::infinity())
                    : -std::numeric_limits<double>::infinity();
                if (lower_bound >= 0.0 || lower_bound > global_upper) continue;
                const double classic_score = classicSwapScore<UnitWeights>(
                    h, k, cent_ptr, assignment);
                ++classic_score_evaluations;
                pam_consider_candidate(best, classic_score, h, k);
            }
        }
        fast_score_evaluations += static_cast<std::size_t>(
            nelements - nclusters) * static_cast<std::size_t>(nclusters);
        two_pass_recovery_seconds += elapsedSeconds(started);
        return best;
    }

    template <bool UnitWeights>
    PamCandidate findBestVerifiedSecondPass(
        const int* cent_ptr,
        const int* assignment,
        const std::vector<std::uint8_t>& medoid_flags,
        double global_upper) {
        const auto started = stageStarted();
        ++two_pass_recovery_rounds;
        PamCandidate best;
#ifdef _OPENMP
        std::size_t exact_count = 0;
        std::fill(thread_best.begin(), thread_best.end(), PamCandidate{});
#pragma omp parallel num_threads(worker_count) if(worker_count > 1) reduction(+:exact_count)
        {
            const int thread_id = omp_get_thread_num();
            FastWorkspace& workspace = workspaces[thread_id];
            PamCandidate local;
#pragma omp for schedule(static) nowait
            for (int h = 0; h < nelements; ++h) {
                if (medoid_flags[h]) continue;
                computeFastScores<UnitWeights>(h, assignment, workspace);
                for (int k = 0; k < nclusters; ++k) {
                    const double fast_score =
                        workspace.base + workspace.delta[k];
                    const double error = fastScoreErrorBound(workspace, k);
                    const double lower_bound = std::isfinite(error)
                        ? std::nextafter(
                            fast_score - error,
                            -std::numeric_limits<double>::infinity())
                        : -std::numeric_limits<double>::infinity();
                    if (lower_bound >= 0.0 ||
                        lower_bound > global_upper) continue;
                    const double classic_score = classicSwapScore<UnitWeights>(
                        h, k, cent_ptr, assignment);
                    ++exact_count;
                    pam_consider_candidate(local, classic_score, h, k);
                }
            }
            thread_best[thread_id] = local;
        }
        classic_score_evaluations += exact_count;
        for (const PamCandidate& local : thread_best) {
            if (local.h >= 0) {
                pam_consider_candidate(
                    best, local.score, local.h, local.k_slot);
            }
        }
#else
        --two_pass_recovery_rounds;
        return findBestVerifiedSecondPassSerial<UnitWeights>(
            cent_ptr, assignment, medoid_flags, global_upper);
#endif
        fast_score_evaluations += static_cast<std::size_t>(
            nelements - nclusters) * static_cast<std::size_t>(nclusters);
        two_pass_recovery_seconds += elapsedSeconds(started);
        return best;
    }

    template <bool UnitWeights>
    PamCandidate findBestVerifiedFastSwapSerial(
        const int* cent_ptr,
        const int* assignment,
        const std::vector<std::uint8_t>& medoid_flags) {
        FastWorkspace& workspace = workspaces[0];
        double best_upper = 0.0;
        std::size_t overlap_count = 0;
        PamCandidate fast_best;
        bool invalid_fast_scores = false;
        bool candidate_distances_are_integers = true;
        bool candidate_distances_are_valid = true;
        int minimum_candidate_quantum = INT_MAX;
        double maximum_candidate_distance = 0.0;
        bool screened_buffer_overflow = false;
        const bool sign_aware =
            weights_are_nonnegative && input_distance_properties_known;
        const bool exact_scores =
            exact_integer_scores || exact_fixed_point_scores;
        const bool track_fast_best =
            exact_scores || !input_distance_properties_known;
        if (sign_aware) ++sign_aware_verified_rounds;
        const auto screen_started = stageStarted();

        for (int h = 0; h < nelements; ++h) {
            if (medoid_flags[h]) continue;
            if (sign_aware) {
                computeSignAwareFastScores<UnitWeights>(
                    h, assignment, workspace);
            } else {
                computeFastScores<UnitWeights>(h, assignment, workspace);
                maximum_candidate_distance = std::max(
                    maximum_candidate_distance,
                    workspace.max_candidate_distance);
                if (!workspace.all_distances_are_nonnegative_integers) {
                    candidate_distances_are_integers = false;
                }
                if (!workspace.all_distances_are_finite_nonnegative) {
                    candidate_distances_are_valid = false;
                }
                minimum_candidate_quantum = std::min(
                    minimum_candidate_quantum,
                    workspace.minimum_quantum_exponent);
            }
            if (!std::isfinite(workspace.base)) {
                invalid_fast_scores = true;
                continue;
            }

            if (track_fast_best) {
                for (int k = 0; k < nclusters; ++k) {
                    pam_consider_candidate(
                        fast_best,
                        workspace.base + workspace.delta[k], h, k);
                }
            }
            if (exact_scores) {
                continue;
            }

            double shared_error = 0.0;
            if (sign_aware) {
                const int max_slot = static_cast<int>(std::distance(
                    workspace.delta.begin(),
                    std::max_element(
                        workspace.delta.begin(), workspace.delta.end())));
                shared_error = signAwareScoreErrorBound(workspace, max_slot);
                if (!std::isfinite(shared_error)) {
                    invalid_fast_scores = true;
                    continue;
                }
            }
            for (int k = 0; k < nclusters; ++k) {
                const double fast_score = workspace.base + workspace.delta[k];
                const double error = sign_aware
                    ? shared_error
                    : fastScoreErrorBound(workspace, k);
                if (!std::isfinite(error)) {
                    invalid_fast_scores = true;
                    continue;
                }
                const double lower_bound = std::nextafter(
                    fast_score - error,
                    -std::numeric_limits<double>::infinity());
                const double upper_bound = std::nextafter(
                    fast_score + error,
                    std::numeric_limits<double>::infinity());
                if (upper_bound < best_upper) {
                    best_upper = upper_bound;
                }
                if (lower_bound < 0.0 && lower_bound <= best_upper) {
                    if (overlap_count < PAM_SCREENED_BUFFER_CAPACITY) {
                        screened_buffers[0][overlap_count] = {
                            lower_bound, h, k};
                    } else {
                        screened_buffer_overflow = true;
                    }
                    ++overlap_count;
                }
            }
        }

        fast_score_evaluations += static_cast<std::size_t>(
            nelements - nclusters) * static_cast<std::size_t>(nclusters);
        fast_screen_seconds += elapsedSeconds(screen_started);

        if (!sign_aware && !exact_integer_scores &&
            assignment_distances_are_nonnegative_integers &&
            candidate_distances_are_integers &&
            weights_are_nonnegative_integers) {
            const double maximum_distance = std::max(
                max_assignment_distance, maximum_candidate_distance);
            exact_integer_scores = pam_upward_multiply(
                pam_upward_multiply(2.0, maximum_distance),
                sum_abs_weights_bound) < std::ldexp(1.0, 52);
        }

        if (!sign_aware && !exact_fixed_point_scores &&
            assignment_distances_are_finite_nonnegative &&
            candidate_distances_are_valid) {
            const double maximum_distance = std::max(
                max_assignment_distance, maximum_candidate_distance);
            const int minimum_quantum = std::min(
                minimum_assignment_quantum_exponent,
                minimum_candidate_quantum);
            exact_fixed_point_scores = fixedPointScoresAreExact(
                maximum_distance, minimum_quantum);
        }

        if (exact_integer_scores || exact_fixed_point_scores) {
            if (exact_integer_scores) {
                ++exact_integer_rounds;
            } else {
                ++exact_fixed_point_rounds;
            }
            return fast_best;
        }

        screened_candidate_highwater = std::max(
            screened_candidate_highwater, overlap_count);
        if (invalid_fast_scores) {
            const auto fallback_started = stageStarted();
            ++adaptive_fallback_rounds;
            PamCandidate fallback = findBestClassicSwapSerial<UnitWeights>(
                cent_ptr, assignment, medoid_flags);
            reynolds_fallback_seconds += elapsedSeconds(fallback_started);
            return fallback;
        }
        if (screened_buffer_overflow) {
            return findBestVerifiedSecondPassSerial<UnitWeights>(
                cent_ptr, assignment, medoid_flags, best_upper);
        }

        PamCandidate best;
        const auto arbitration_started = stageStarted();
        const std::size_t count = std::min(
            overlap_count, PAM_SCREENED_BUFFER_CAPACITY);
        for (std::size_t index = 0; index < count; ++index) {
            const PamScreenedCandidate& candidate =
                screened_buffers[0][index];
            if (candidate.lower_bound > best_upper) continue;
            const double classic_score = classicSwapScore<UnitWeights>(
                candidate.h, candidate.k_slot, cent_ptr, assignment);
            ++classic_score_evaluations;
            pam_consider_candidate(
                best, classic_score, candidate.h, candidate.k_slot);
        }
        exact_arbitration_seconds += elapsedSeconds(arbitration_started);
        return best;
    }

    template <bool UnitWeights>
    PamCandidate findBestVerifiedFastSwap(
        const int* cent_ptr,
        const int* assignment,
        const std::vector<std::uint8_t>& is_medoid) {
        std::fill(
            thread_upper_bounds.begin(), thread_upper_bounds.end(),
            0.0);
        std::fill(
            thread_overlap_counts.begin(), thread_overlap_counts.end(),
            0);
        std::fill(thread_best.begin(), thread_best.end(), PamCandidate{});
        int invalid_fast_scores = 0;
        int candidate_distances_are_integers = 1;
        int candidate_distances_are_valid = 1;
        int minimum_candidate_quantum = INT_MAX;
        double maximum_candidate_distance = 0.0;
        int screened_buffer_overflow = 0;
        const bool sign_aware =
            weights_are_nonnegative && input_distance_properties_known;
        const bool exact_scores =
            exact_integer_scores || exact_fixed_point_scores;
        const bool track_fast_best =
            exact_scores || !input_distance_properties_known;
        if (sign_aware) ++sign_aware_verified_rounds;
        const auto screen_started = stageStarted();

#ifdef _OPENMP
#pragma omp parallel num_threads(worker_count) if(worker_count > 1) reduction(|:invalid_fast_scores,screened_buffer_overflow) reduction(&:candidate_distances_are_integers,candidate_distances_are_valid) reduction(min:minimum_candidate_quantum) reduction(max:maximum_candidate_distance)
        {
            const int thread_id = omp_get_thread_num();
            FastWorkspace& workspace = workspaces[thread_id];
            double best_upper = 0.0;
            std::size_t overlap_count = 0;
            PamCandidate local_best;

#pragma omp for schedule(static) nowait
            for (int h = 0; h < nelements; ++h) {
                if (is_medoid[h]) continue;
                if (sign_aware) {
                    computeSignAwareFastScores<UnitWeights>(
                        h, assignment, workspace);
                } else {
                    computeFastScores<UnitWeights>(h, assignment, workspace);
                    maximum_candidate_distance = std::max(
                        maximum_candidate_distance,
                        workspace.max_candidate_distance);
                    if (!workspace.all_distances_are_nonnegative_integers) {
                        candidate_distances_are_integers = 0;
                    }
                    if (!workspace.all_distances_are_finite_nonnegative) {
                        candidate_distances_are_valid = 0;
                    }
                    minimum_candidate_quantum = std::min(
                        minimum_candidate_quantum,
                        workspace.minimum_quantum_exponent);
                }
                if (!std::isfinite(workspace.base)) {
                    invalid_fast_scores = 1;
                    continue;
                }

                if (track_fast_best) {
                    for (int k = 0; k < nclusters; ++k) {
                        pam_consider_candidate(
                            local_best,
                            workspace.base + workspace.delta[k], h, k);
                    }
                }
                if (exact_scores) {
                    continue;
                }

                double shared_error = 0.0;
                if (sign_aware) {
                    const int max_slot = static_cast<int>(std::distance(
                        workspace.delta.begin(),
                        std::max_element(
                            workspace.delta.begin(), workspace.delta.end())));
                    shared_error = signAwareScoreErrorBound(
                        workspace, max_slot);
                    if (!std::isfinite(shared_error)) {
                        invalid_fast_scores = 1;
                        continue;
                    }
                }
                for (int k = 0; k < nclusters; ++k) {
                    const double fast_score = workspace.base + workspace.delta[k];
                    const double error = sign_aware
                        ? shared_error
                        : fastScoreErrorBound(workspace, k);
                    if (!std::isfinite(error)) {
                        invalid_fast_scores = 1;
                        continue;
                    }
                    const double lower_bound = std::nextafter(
                        fast_score - error,
                        -std::numeric_limits<double>::infinity());
                    const double upper_bound = std::nextafter(
                        fast_score + error,
                        std::numeric_limits<double>::infinity());
                    if (upper_bound < best_upper) {
                        best_upper = upper_bound;
                    }
                    if (lower_bound < 0.0 && lower_bound <= best_upper) {
                        if (overlap_count < PAM_SCREENED_BUFFER_CAPACITY) {
                            screened_buffers[thread_id][overlap_count] = {
                                lower_bound, h, k};
                        } else {
                            screened_buffer_overflow = 1;
                        }
                        ++overlap_count;
                    }
                }
            }
            thread_upper_bounds[thread_id] = best_upper;
            thread_overlap_counts[thread_id] = overlap_count;
            thread_best[thread_id] = local_best;
        }
#else
        FastWorkspace& workspace = workspaces[0];
        double best_upper = 0.0;
        std::size_t overlap_count = 0;
        PamCandidate local_best;
        for (int h = 0; h < nelements; ++h) {
            if (is_medoid[h]) continue;
            if (sign_aware) {
                computeSignAwareFastScores<UnitWeights>(
                    h, assignment, workspace);
            } else {
                computeFastScores<UnitWeights>(h, assignment, workspace);
                maximum_candidate_distance = std::max(
                    maximum_candidate_distance,
                    workspace.max_candidate_distance);
                if (!workspace.all_distances_are_nonnegative_integers) {
                    candidate_distances_are_integers = 0;
                }
                if (!workspace.all_distances_are_finite_nonnegative) {
                    candidate_distances_are_valid = 0;
                }
                minimum_candidate_quantum = std::min(
                    minimum_candidate_quantum,
                    workspace.minimum_quantum_exponent);
            }
            if (!std::isfinite(workspace.base)) {
                invalid_fast_scores = 1;
                continue;
            }

            if (track_fast_best) {
                for (int k = 0; k < nclusters; ++k) {
                    pam_consider_candidate(
                        local_best,
                        workspace.base + workspace.delta[k], h, k);
                }
            }
            if (exact_scores) {
                continue;
            }

            double shared_error = 0.0;
            if (sign_aware) {
                const int max_slot = static_cast<int>(std::distance(
                    workspace.delta.begin(),
                    std::max_element(
                        workspace.delta.begin(), workspace.delta.end())));
                shared_error = signAwareScoreErrorBound(workspace, max_slot);
                if (!std::isfinite(shared_error)) {
                    invalid_fast_scores = 1;
                    continue;
                }
            }
            for (int k = 0; k < nclusters; ++k) {
                const double fast_score = workspace.base + workspace.delta[k];
                const double error = sign_aware
                    ? shared_error
                    : fastScoreErrorBound(workspace, k);
                if (!std::isfinite(error)) {
                    invalid_fast_scores = 1;
                    continue;
                }
                const double lower_bound = std::nextafter(
                    fast_score - error,
                    -std::numeric_limits<double>::infinity());
                const double upper_bound = std::nextafter(
                    fast_score + error,
                    std::numeric_limits<double>::infinity());
                if (upper_bound < best_upper) {
                    best_upper = upper_bound;
                }
                if (lower_bound < 0.0 && lower_bound <= best_upper) {
                    if (overlap_count < PAM_SCREENED_BUFFER_CAPACITY) {
                        screened_buffers[0][overlap_count] = {
                            lower_bound, h, k};
                    } else {
                        screened_buffer_overflow = 1;
                    }
                    ++overlap_count;
                }
            }
        }
        thread_upper_bounds[0] = best_upper;
        thread_overlap_counts[0] = overlap_count;
        thread_best[0] = local_best;
#endif

        fast_score_evaluations += static_cast<std::size_t>(
            nelements - nclusters) * static_cast<std::size_t>(nclusters);
        fast_screen_seconds += elapsedSeconds(screen_started);

        if (!sign_aware && !exact_integer_scores &&
            assignment_distances_are_nonnegative_integers &&
            candidate_distances_are_integers != 0 &&
            weights_are_nonnegative_integers) {
            const double maximum_distance = std::max(
                max_assignment_distance, maximum_candidate_distance);
            exact_integer_scores = pam_upward_multiply(
                pam_upward_multiply(2.0, maximum_distance),
                sum_abs_weights_bound) < std::ldexp(1.0, 52);
        }

        if (!sign_aware && !exact_fixed_point_scores &&
            assignment_distances_are_finite_nonnegative &&
            candidate_distances_are_valid != 0) {
            const double maximum_distance = std::max(
                max_assignment_distance, maximum_candidate_distance);
            const int minimum_quantum = std::min(
                minimum_assignment_quantum_exponent,
                minimum_candidate_quantum);
            exact_fixed_point_scores = fixedPointScoresAreExact(
                maximum_distance, minimum_quantum);
        }

        if (exact_integer_scores || exact_fixed_point_scores) {
            PamCandidate best;
            for (const PamCandidate& local : thread_best) {
                if (local.h >= 0) {
                    pam_consider_candidate(
                        best, local.score, local.h, local.k_slot);
                }
            }
            if (exact_integer_scores) {
                ++exact_integer_rounds;
            } else {
                ++exact_fixed_point_rounds;
            }
            return best;
        }

        if (invalid_fast_scores != 0) {
            const auto fallback_started = stageStarted();
            ++adaptive_fallback_rounds;
            PamCandidate fallback = findBestClassicSwap<UnitWeights>(
                cent_ptr, assignment, is_medoid);
            reynolds_fallback_seconds += elapsedSeconds(fallback_started);
            return fallback;
        }

        const double global_upper = *std::min_element(
            thread_upper_bounds.begin(), thread_upper_bounds.end());
        std::size_t overlap_count = 0;
        for (std::size_t count : thread_overlap_counts) {
            overlap_count += count;
        }
        screened_candidate_highwater = std::max(
            screened_candidate_highwater, overlap_count);

        if (screened_buffer_overflow != 0) {
            return findBestVerifiedSecondPass<UnitWeights>(
                cent_ptr, assignment, is_medoid, global_upper);
        }

        PamCandidate best;
        const auto arbitration_started = stageStarted();
#ifdef _OPENMP
        std::size_t exact_count = 0;
        std::fill(thread_best.begin(), thread_best.end(), PamCandidate{});
#pragma omp parallel for num_threads(worker_count) if(worker_count > 1) reduction(+:exact_count) schedule(static)
        for (int thread_id = 0; thread_id < worker_count; ++thread_id) {
            PamCandidate local;
            const std::size_t count = std::min(
                thread_overlap_counts[thread_id],
                PAM_SCREENED_BUFFER_CAPACITY);
            for (std::size_t index = 0; index < count; ++index) {
                const PamScreenedCandidate& candidate =
                    screened_buffers[thread_id][index];
                if (candidate.lower_bound > global_upper) continue;
                const double classic_score = classicSwapScore<UnitWeights>(
                    candidate.h, candidate.k_slot, cent_ptr, assignment);
                ++exact_count;
                pam_consider_candidate(
                    local, classic_score, candidate.h, candidate.k_slot);
            }
            thread_best[thread_id] = local;
        }
        classic_score_evaluations += exact_count;
        for (const PamCandidate& local : thread_best) {
            if (local.h >= 0) {
                pam_consider_candidate(
                    best, local.score, local.h, local.k_slot);
            }
        }
#else
        const std::size_t count = std::min(
            thread_overlap_counts[0], PAM_SCREENED_BUFFER_CAPACITY);
        for (std::size_t index = 0; index < count; ++index) {
            const PamScreenedCandidate& candidate = screened_buffers[0][index];
            if (candidate.lower_bound > global_upper) continue;
            const double classic_score = classicSwapScore<UnitWeights>(
                candidate.h, candidate.k_slot, cent_ptr, assignment);
            ++classic_score_evaluations;
            pam_consider_candidate(
                best, classic_score, candidate.h, candidate.k_slot);
        }
#endif

        exact_arbitration_seconds += elapsedSeconds(arbitration_started);
        return best;
    }

    int nelements;
    py::array diss;
    PamIntArray centroids;
    int npass;
    PamDoubleArray weights;
    PamIntArray build_tie_keys;
    PamDoubleArray distance_codebook;
    int nclusters;
    py::array_t<int> clusterid;
    std::vector<double> dysma;
    std::vector<double> dysmb;
    std::vector<int> second_slot;
    std::vector<std::uint8_t> is_medoid;
    std::vector<FastWorkspace> workspaces;
    std::vector<double> thread_upper_bounds;
    std::vector<std::size_t> thread_overlap_counts;
    std::vector<PamCandidate> thread_best;
    std::vector<std::vector<PamScreenedCandidate>> screened_buffers;
    std::vector<int> swap_removed;
    std::vector<int> swap_entering;
    std::vector<int> swap_slots;
    double maxdist;
    double final_objective = std::numeric_limits<double>::infinity();
    double sum_abs_weights_bound = 0.0;
    double sign_aware_abs_sum_scale = 0.0;
    double sign_aware_sum_gamma = 0.0;
    double sign_aware_underflow_bound = 0.0;
    double max_assignment_distance = 0.0;
    DistanceStorage distance_storage = DistanceStorage::Float64;
    const double* distance_f64 = nullptr;
    const std::uint8_t* distance_u8 = nullptr;
    const std::uint16_t* distance_u16 = nullptr;
    const std::uint32_t* distance_u32 = nullptr;
    const double* distance_codebook_ptr = nullptr;
    std::size_t distance_codebook_size = 0;
    const double* wt_ptr = nullptr;
    const int* build_tie_ptr = nullptr;
    bool use_condensed = false;
    bool unit_weights = false;
    bool weights_are_nonnegative = true;
    bool weights_are_nonnegative_integers = true;
    bool collect_diagnostics = false;
    bool exact_integer_scores = false;
    bool exact_fixed_point_scores = false;
    bool input_distance_properties_known = false;
    bool input_distances_are_nonnegative_integers = false;
    bool input_distances_are_finite_nonnegative = false;
    bool assignment_distances_are_nonnegative_integers = false;
    bool assignment_distances_are_finite_nonnegative = false;
    int minimum_input_quantum_exponent = INT_MAX;
    int minimum_assignment_quantum_exponent = INT_MAX;
    double maximum_input_distance = 0.0;
    std::size_t fast_score_evaluations = 0;
    std::size_t classic_score_evaluations = 0;
    std::size_t swap_rounds = 0;
    std::size_t accepted_swaps = 0;
    std::size_t screened_candidate_highwater = 0;
    std::size_t adaptive_fallback_rounds = 0;
    std::size_t two_pass_recovery_rounds = 0;
    std::size_t small_k_fused_rounds = 0;
    double small_k_fused_seconds = 0.0;
    std::size_t small_k_reynolds_rounds = 0;
    std::size_t exact_integer_rounds = 0;
    std::size_t exact_fixed_point_rounds = 0;
    std::size_t sign_aware_verified_rounds = 0;
    std::size_t workspace_peak_bytes = 0;
    std::size_t bounded_candidate_buffer_bytes = 0;
    std::size_t thread_workspace_bytes = 0;
    std::size_t full_assignment_rounds = 0;
    std::size_t incremental_update_rounds = 0;
    std::size_t incremental_rescanned_rows = 0;
    double max_distance_seconds = 0.0;
    double build_seconds = 0.0;
    double assignment_seconds = 0.0;
    double fast_screen_seconds = 0.0;
    double exact_arbitration_seconds = 0.0;
    double reynolds_fallback_seconds = 0.0;
    double two_pass_recovery_seconds = 0.0;
    double swap_update_seconds = 0.0;
    double incremental_update_seconds = 0.0;
    double final_assignment_seconds = 0.0;
    int worker_count = 1;
};
