#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "pam_common.h"

namespace py = pybind11;

class PAM {
public:
    PAM(int nelements, PamDoubleArray diss,
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
#ifdef _OPENMP
        candidate_threads = std::max(
            1, std::min(omp_get_max_threads(), nelements - nclusters));
        thread_best.resize(candidate_threads);
#endif
    }

    py::array_t<int> runclusterloop() {
        return runclusterloopImpl(0);
    }

    py::array_t<int> runclusterloop_one_based() {
        return runclusterloopImpl(1);
    }

private:
    py::array_t<int> runclusterloopImpl(int output_offset) {
        int* cent_ptr = centroids.mutable_data();
        int* assignment = clusterid.mutable_data();

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
                assignToNearestMedoids(cent_ptr, assignment);

                best = unit_weights
                    ? findBestClassicSwap<true>(
                        cent_ptr, assignment, is_medoid)
                    : findBestClassicSwap<false>(
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
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
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
        }
    }

    template <bool UnitWeights>
    PamCandidate findBestClassicSwap(
        const int* cent_ptr,
        const int* assignment,
        const std::vector<std::uint8_t>& is_medoid) {
        PamCandidate best;
#ifdef _OPENMP
        std::fill(thread_best.begin(), thread_best.end(), PamCandidate{});
#pragma omp parallel num_threads(candidate_threads)
        {
            PamCandidate local;
#pragma omp for schedule(static) nowait
            for (int h = 0; h < nelements; ++h) {
                if (is_medoid[h]) continue;
                for (int k = 0; k < nclusters; ++k) {
                    const double score = classicSwapScore<UnitWeights>(
                        h, k, cent_ptr, assignment);
                    pam_consider_candidate(local, score, h, k);
                }
            }
            thread_best[omp_get_thread_num()] = local;
        }
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
                pam_consider_candidate(best, score, h, k);
            }
        }
#endif
        return best;
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
    std::vector<PamCandidate> thread_best;
    double maxdist;
    const double* diss_ptr = nullptr;
    const double* cond_ptr = nullptr;
    const double* wt_ptr = nullptr;
    bool use_condensed = false;
    bool unit_weights = false;
    int candidate_threads = 1;
};
