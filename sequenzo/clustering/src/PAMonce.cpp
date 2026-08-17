#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "pam_common.h"

namespace py = pybind11;

class PAMonce {
public:
    PAMonce(int nelements, PamDoubleArray diss,
            PamIntArray centroids, int npass,
            PamDoubleArray weights)
        : nelements(nelements),
          diss(std::move(diss)),
          centroids(std::move(centroids)),
          npass(npass),
          weights(std::move(weights)),
          nclusters(static_cast<int>(this->centroids.size())),
          clusterid(nelements),
          dysma(nelements),
          dysmb(nelements),
          is_medoid(nelements, 0) {
        unit_weights = this->weights.size() == 0;
        wt_ptr = unit_weights ? nullptr : this->weights.data();
        if (this->diss.ndim() == 1) {
            use_condensed = true;
            cond_ptr = this->diss.data();
        } else {
            diss_ptr = this->diss.data();
        }
        maxdist = std::numeric_limits<double>::infinity();

        sum_abs_weights_bound = static_cast<double>(nelements);
        if (!unit_weights) {
            double rounded_weight_sum = 0.0;
            for (int i = 0; i < nelements; ++i) {
                rounded_weight_sum += std::abs(wt_ptr[i]);
            }
            sum_abs_weights_bound = pam_upper_abs_sum(
                rounded_weight_sum,
                nelements > 0 ? static_cast<std::size_t>(nelements - 1) : 0);
        }
        rounding_scale = pam_upward_multiply(
            64.0,
            pam_upward_multiply(
                pam_gamma_bound(
                    static_cast<std::size_t>(2 * nelements + 16)),
                sum_abs_weights_bound));
        underflow_bound = pam_upward_multiply(
            64.0 * static_cast<double>(nelements + 1),
            std::numeric_limits<double>::denorm_min());
        const double score_and_error_weight_scale = pam_upward_add(
            sum_abs_weights_bound, rounding_scale);
        if (std::isfinite(score_and_error_weight_scale) &&
            score_and_error_weight_scale > 0.0) {
            max_safe_distance_scale = std::nextafter(
                (std::numeric_limits<double>::max() - underflow_bound) /
                    score_and_error_weight_scale,
                0.0);
        }

#ifdef _OPENMP
        worker_count = std::max(
            1, std::min(omp_get_max_threads(), nelements - nclusters));
#else
        worker_count = 1;
#endif
        workspaces.reserve(worker_count);
        screened_candidates.resize(worker_count);
        thread_upper_bounds.resize(
            worker_count, 0.0);
        thread_best.resize(worker_count);
        const int candidate_rows = std::max(0, nelements - nclusters);
        const int rows_per_worker = candidate_rows == 0
            ? 0
            : (candidate_rows + worker_count - 1) / worker_count;
        const int reserved_rows = std::min(4, rows_per_worker);
        for (int i = 0; i < worker_count; ++i) {
            workspaces.emplace_back(nclusters);
            screened_candidates[i].values.reserve(
                static_cast<std::size_t>(nclusters) * reserved_rows);
        }
    }

    py::array_t<int> runclusterloop() {
        return runclusterloopImpl(0);
    }

    py::array_t<int> runclusterloop_one_based() {
        return runclusterloopImpl(1);
    }

    py::dict diagnostics() const {
        py::dict result;
        result["fast_score_evaluations"] = fast_score_evaluations;
        result["classic_score_evaluations"] = classic_score_evaluations;
        result["swap_rounds"] = swap_rounds;
        return result;
    }

private:
    py::array_t<int> runclusterloopImpl(int output_offset) {
        int* cent_ptr = centroids.mutable_data();
        int* assignment = clusterid.mutable_data();
        fast_score_evaluations = 0;
        classic_score_evaluations = 0;
        swap_rounds = 0;

        {
            py::gil_scoped_release release;
            if (npass > 0) {
                maxdist = computeMaxDist();
                if (unit_weights) {
                    buildInitialCentroids<true>(cent_ptr);
                } else {
                    buildInitialCentroids<false>(cent_ptr);
                }
            } else {
                initializeMedoidFlags(cent_ptr);
            }

            PamCandidate best;
            do {
                ++swap_rounds;
                assignToNearestMedoids(cent_ptr, assignment);

                best = unit_weights
                    ? findBestVerifiedFastSwap<true>(
                        cent_ptr, assignment, is_medoid)
                    : findBestVerifiedFastSwap<false>(
                        cent_ptr, assignment, is_medoid);
                if (best.score < 0.0) {
                    const int removed = cent_ptr[best.k_slot];
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
                }
            } while (best.score < 0.0);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < nelements; ++i) {
                assignment[i] = cent_ptr[assignment[i]] + output_offset;
            }
        }
        return clusterid;
    }

    struct alignas(64) FastWorkspace {
        explicit FastWorkspace(int k)
            : delta(k) {}

        void reset() {
            base = 0.0;
            max_candidate_distance = 0.0;
            std::fill(delta.begin(), delta.end(), 0.0);
        }

        double base = 0.0;
        double max_candidate_distance = 0.0;
        std::vector<double> delta;
    };

    struct ScreenedCandidate {
        double lower_bound;
        int h;
        int k_slot;
    };

    struct alignas(64) CandidateBuffer {
        std::vector<ScreenedCandidate> values;
    };

    template <bool UnitWeights>
    inline double weightAt(int index) const {
        if constexpr (UnitWeights) return 1.0;
        return wt_ptr[index];
    }

    inline double get_dist(int i, int j) const {
        if (!use_condensed) {
            return diss_ptr[static_cast<std::size_t>(i) * nelements + j];
        }
        if (i == j) return 0.0;
        int a = i;
        int b = j;
        if (a > b) std::swap(a, b);
        return cond_ptr[
            static_cast<std::size_t>(a) * (2 * nelements - a - 1) / 2
            + (b - a - 1)];
    }

    template <typename Callback>
    inline void forEachDistanceInRow(int row, Callback&& callback) const {
        pam_for_each_distance_in_row(
            nelements, row, diss_ptr, cond_ptr, use_condensed,
            std::forward<Callback>(callback));
    }

    double computeMaxDist() const {
        double maximum = 0.0;
        if (use_condensed) {
            const std::ptrdiff_t pair_count =
                static_cast<std::ptrdiff_t>(diss.size());
#ifdef _OPENMP
#pragma omp parallel for reduction(max:maximum) schedule(static)
#endif
            for (std::ptrdiff_t index = 0; index < pair_count; ++index) {
                maximum = std::max(maximum, cond_ptr[index]);
            }
            return 1.1 * maximum + 1.0;
        }
#ifdef _OPENMP
#pragma omp parallel for reduction(max:maximum) schedule(static)
#endif
        for (int i = 0; i < nelements; ++i) {
            for (int j = i + 1; j < nelements; ++j) {
                maximum = std::max(maximum, get_dist(i, j));
            }
        }
        return 1.1 * maximum + 1.0;
    }

    template <bool UnitWeights>
    void buildInitialCentroids(int* cent_ptr) {
        std::fill(is_medoid.begin(), is_medoid.end(), 0);
        std::fill(dysma.begin(), dysma.end(), maxdist);

        for (int selected = 0; selected < nclusters; ++selected) {
            double best_gain = -std::numeric_limits<double>::infinity();
            int best_index = -1;
#ifdef _OPENMP
#pragma omp parallel
            {
                double local_gain = -std::numeric_limits<double>::infinity();
                int local_index = -1;
#pragma omp for schedule(static) nowait
                for (int i = 0; i < nelements; ++i) {
                    if (is_medoid[i]) continue;
                    double gain = 0.0;
                    forEachDistanceInRow(i, [&](int j, double distance) {
                        const double improvement = dysma[j] - distance;
                        gain += weightAt<UnitWeights>(j) *
                            std::max(0.0, improvement);
                    });
                    if (local_gain <= gain) {
                        local_gain = gain;
                        local_index = i;
                    }
                }
#pragma omp critical
                {
                    if (local_index >= 0 &&
                        (local_gain > best_gain ||
                         (local_gain == best_gain && local_index > best_index))) {
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
                if (best_gain <= gain) {
                    best_gain = gain;
                    best_index = i;
                }
            }
#endif
            is_medoid[best_index] = 1;
            cent_ptr[selected] = best_index;
            forEachDistanceInRow(
                best_index,
                [&](int j, double distance) {
                    dysma[j] = std::min(dysma[j], distance);
                });
        }
    }

    void initializeMedoidFlags(const int* cent_ptr) {
        std::fill(is_medoid.begin(), is_medoid.end(), 0);
        for (int k = 0; k < nclusters; ++k) {
            is_medoid[cent_ptr[k]] = 1;
        }
    }

    void assignToNearestMedoids(const int* cent_ptr, int* assignment) {
        double maximum_nearest = 0.0;
        double maximum_second = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(max:maximum_nearest,maximum_second) schedule(static)
#endif
        for (int i = 0; i < nelements; ++i) {
            double nearest = maxdist;
            double second = maxdist;
            int nearest_slot = 0;
            for (int k = 0; k < nclusters; ++k) {
                const double distance = get_dist(i, cent_ptr[k]);
                if (nearest > distance) {
                    second = nearest;
                    nearest = distance;
                    nearest_slot = k;
                } else if (second > distance) {
                    second = distance;
                }
            }
            dysma[i] = nearest;
            dysmb[i] = second;
            assignment[i] = nearest_slot;
            maximum_nearest = std::max(maximum_nearest, std::abs(nearest));
            maximum_second = std::max(maximum_second, std::abs(second));
        }
        max_nearest_distance = maximum_nearest;
        max_second_distance = maximum_second;
    }

    template <bool UnitWeights>
    void computeFastScores(int h, const int* assignment,
                           FastWorkspace& workspace) const {
        workspace.reset();
        forEachDistanceInRow(h, [&](int j, double candidate_distance) {
            const double nearest = dysma[j];
            const double weight = weightAt<UnitWeights>(j);
            if (candidate_distance < nearest) {
                workspace.base += weight * (candidate_distance - nearest);
            } else {
                workspace.delta[assignment[j]] += weight *
                    (std::min(dysmb[j], candidate_distance) - nearest);
            }
            workspace.max_candidate_distance = std::max(
                workspace.max_candidate_distance,
                std::abs(candidate_distance));
        });
    }

    double fastScoreErrorBound(const FastWorkspace& workspace) const {
        const double distance_scale = pam_upward_add(
            workspace.max_candidate_distance,
            pam_upward_add(max_nearest_distance, max_second_distance));
        if (!std::isfinite(distance_scale) ||
            !std::isfinite(rounding_scale)) {
            return std::numeric_limits<double>::infinity();
        }

        if (distance_scale > max_safe_distance_scale) {
            return std::numeric_limits<double>::infinity();
        }

        // Both scoring paths form weighted distance differences and then sum
        // at most N terms. 64 * gamma(2N + 16) times the absolute input scale
        // covers term formation, base/correction accumulation, the final add,
        // the candidate interval endpoints, and the canonical sequential sum.
        // The denormal term covers gradual underflow without branches in the
        // N^2 loop.
        const double rounding = pam_upward_multiply(
            rounding_scale, distance_scale);
        return pam_upward_add(rounding, underflow_bound);
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
    PamCandidate findBestClassicSwap(
        const int* cent_ptr,
        const int* assignment,
        const std::vector<std::uint8_t>& is_medoid) {
        PamCandidate best;
#ifdef _OPENMP
        std::size_t exact_count = 0;
        std::fill(thread_best.begin(), thread_best.end(), PamCandidate{});
#pragma omp parallel num_threads(worker_count) reduction(+:exact_count)
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
    PamCandidate findBestVerifiedFastSwap(
        const int* cent_ptr,
        const int* assignment,
        const std::vector<std::uint8_t>& is_medoid) {
        std::fill(
            thread_upper_bounds.begin(), thread_upper_bounds.end(),
            0.0);
        for (auto& buffer : screened_candidates) buffer.values.clear();
        int invalid_fast_scores = 0;

#ifdef _OPENMP
#pragma omp parallel num_threads(worker_count) reduction(|:invalid_fast_scores)
        {
            const int thread_id = omp_get_thread_num();
            FastWorkspace& workspace = workspaces[thread_id];
            std::vector<ScreenedCandidate>& candidates =
                screened_candidates[thread_id].values;
            double best_upper = 0.0;

#pragma omp for schedule(static) nowait
            for (int h = 0; h < nelements; ++h) {
                if (is_medoid[h]) continue;
                computeFastScores<UnitWeights>(h, assignment, workspace);
                const double error = fastScoreErrorBound(workspace);
                if (!std::isfinite(workspace.base) || !std::isfinite(error)) {
                    invalid_fast_scores = 1;
                    continue;
                }

                for (int k = 0; k < nclusters; ++k) {
                    const double fast_score = workspace.base + workspace.delta[k];
                    const double lower_bound = fast_score - error;
                    const double upper_bound = fast_score + error;
                    if (upper_bound < best_upper) {
                        best_upper = upper_bound;
                        candidates.erase(
                            std::remove_if(
                                candidates.begin(), candidates.end(),
                                [best_upper](const ScreenedCandidate& candidate) {
                                    return candidate.lower_bound > best_upper;
                                }),
                            candidates.end());
                    }
                    if (lower_bound < 0.0 && lower_bound <= best_upper) {
                        candidates.push_back({lower_bound, h, k});
                    }
                }
            }
            thread_upper_bounds[thread_id] = best_upper;
        }
#else
        FastWorkspace& workspace = workspaces[0];
        std::vector<ScreenedCandidate>& candidates =
            screened_candidates[0].values;
        double best_upper = 0.0;
        for (int h = 0; h < nelements; ++h) {
            if (is_medoid[h]) continue;
            computeFastScores<UnitWeights>(h, assignment, workspace);
            const double error = fastScoreErrorBound(workspace);
            if (!std::isfinite(workspace.base) || !std::isfinite(error)) {
                invalid_fast_scores = 1;
                continue;
            }

            for (int k = 0; k < nclusters; ++k) {
                const double fast_score = workspace.base + workspace.delta[k];
                const double lower_bound = fast_score - error;
                const double upper_bound = fast_score + error;
                if (upper_bound < best_upper) {
                    best_upper = upper_bound;
                    candidates.erase(
                        std::remove_if(
                            candidates.begin(), candidates.end(),
                            [best_upper](const ScreenedCandidate& candidate) {
                                return candidate.lower_bound > best_upper;
                            }),
                        candidates.end());
                }
                if (lower_bound < 0.0 && lower_bound <= best_upper) {
                    candidates.push_back({lower_bound, h, k});
                }
            }
        }
        thread_upper_bounds[0] = best_upper;
#endif

        fast_score_evaluations += static_cast<std::size_t>(
            nelements - nclusters) * static_cast<std::size_t>(nclusters);

        if (invalid_fast_scores != 0) {
            return findBestClassicSwap<UnitWeights>(
                cent_ptr, assignment, is_medoid);
        }

        const double global_upper = *std::min_element(
            thread_upper_bounds.begin(), thread_upper_bounds.end());
        std::size_t finalist_count = 0;
        ScreenedCandidate first_finalist{};
        for (const auto& buffer : screened_candidates) {
            for (const ScreenedCandidate& candidate : buffer.values) {
                if (candidate.lower_bound <= global_upper) {
                    if (finalist_count == 0) first_finalist = candidate;
                    ++finalist_count;
                }
            }
        }

        if (finalist_count == 0) return PamCandidate{};
        if (finalist_count == 1) {
            if (global_upper < 0.0 ||
                first_finalist.lower_bound >= 0.0) {
                return {global_upper < 0.0
                            ? global_upper
                            : first_finalist.lower_bound,
                        first_finalist.h,
                        first_finalist.k_slot};
            }
        }

        PamCandidate best;
#ifdef _OPENMP
        std::size_t exact_count = 0;
        std::fill(thread_best.begin(), thread_best.end(), PamCandidate{});
#pragma omp parallel for num_threads(worker_count) schedule(static, 1) reduction(+:exact_count)
        for (std::size_t thread_index = 0;
             thread_index < screened_candidates.size(); ++thread_index) {
            PamCandidate local;
            for (const ScreenedCandidate& candidate :
                 screened_candidates[thread_index].values) {
                if (candidate.lower_bound > global_upper) continue;
                const double classic_score = classicSwapScore<UnitWeights>(
                    candidate.h, candidate.k_slot, cent_ptr, assignment);
                ++exact_count;
                pam_consider_candidate(
                    local, classic_score, candidate.h, candidate.k_slot);
            }
            thread_best[thread_index] = local;
        }
        classic_score_evaluations += exact_count;
        for (const PamCandidate& local : thread_best) {
            if (local.h >= 0) {
                pam_consider_candidate(
                    best, local.score, local.h, local.k_slot);
            }
        }
#else
        for (const auto& buffer : screened_candidates) {
            for (const ScreenedCandidate& candidate : buffer.values) {
                if (candidate.lower_bound > global_upper) continue;
                const double classic_score = classicSwapScore<UnitWeights>(
                    candidate.h, candidate.k_slot, cent_ptr, assignment);
                ++classic_score_evaluations;
                pam_consider_candidate(
                    best, classic_score, candidate.h, candidate.k_slot);
            }
        }
#endif

        return best;
    }

    int nelements;
    PamDoubleArray diss;
    PamIntArray centroids;
    int npass;
    PamDoubleArray weights;
    int nclusters;
    py::array_t<int> clusterid;
    std::vector<double> dysma;
    std::vector<double> dysmb;
    std::vector<std::uint8_t> is_medoid;
    std::vector<FastWorkspace> workspaces;
    std::vector<CandidateBuffer> screened_candidates;
    std::vector<double> thread_upper_bounds;
    std::vector<PamCandidate> thread_best;
    double maxdist;
    double sum_abs_weights_bound = 0.0;
    double rounding_scale = 0.0;
    double underflow_bound = 0.0;
    double max_safe_distance_scale = 0.0;
    double max_nearest_distance = 0.0;
    double max_second_distance = 0.0;
    const double* diss_ptr = nullptr;
    const double* cond_ptr = nullptr;
    const double* wt_ptr = nullptr;
    bool use_condensed = false;
    bool unit_weights = false;
    std::size_t fast_score_evaluations = 0;
    std::size_t classic_score_evaluations = 0;
    std::size_t swap_rounds = 0;
    int worker_count = 1;
};
