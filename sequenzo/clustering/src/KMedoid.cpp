#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <random>
#include <cfloat>
#include <climits>
#include <cmath>

using namespace std;
namespace py = pybind11;

class KMedoid {
protected:
    int nelements;        // Number of elements (data points)
    int nclusters;        // Number of clusters (medoids)
    int npass;            // Maximum number of iterations

    vector<int> tclusterid;        // Temporary cluster assignment for each element
    vector<int> saved;             // Saved cluster assignments to check for convergence
    vector<int> clusterMembership; // Cluster membership indices (flattened 2D: nclusters x nelements)
    vector<int> clusterSize;       // Size of each cluster

    py::array_t<double> diss;      // Distance matrix (2D numpy array)
    py::array_t<int> centroids;    // Medoid indices (1D numpy array)
    py::array_t<double> weights;   // Weights of elements (1D numpy array)

public:
    // Constructor initializes members and allocates necessary storage
    KMedoid(int nelements, py::array_t<double> diss,
            py::array_t<int> centroids, int npass,
            py::array_t<double> weights)
        : nelements(nelements),
          diss(diss),
          centroids(centroids),
          npass(npass),
          weights(weights),
          nclusters(static_cast<int>(centroids.size())) {
        // 注释掉信息性打印，避免在并行环境（如 CLARA）中降低性能
        // py::print("[>] Starting KMedoids...");

        tclusterid.resize(nelements);
        saved.resize(nelements);
        clusterMembership.resize(nelements * nclusters);
        clusterSize.resize(nclusters);
        fill(clusterSize.begin(), clusterSize.end(), 0);
    }

    // Initialize medoids using a k-means++ style seeding method for better starting points
    void init_medoids() {
        auto ptr_diss = diss.unchecked<2>();
        auto ptr_centroids = centroids.mutable_unchecked<1>();
        auto ptr_weights = weights.unchecked<1>();

        vector<int> selected;                             // Indices of selected medoids
        vector<double> min_dists(nelements, DBL_MAX);     // Minimum distance to selected medoids

        mt19937 rng(random_device{}()); // 3. Random number generator initialization (non-deterministic seed)
        uniform_int_distribution<> dist(0, nelements - 1);

        // Randomly choose the first medoid
        int first = dist(rng);
        selected.push_back(first);
        ptr_centroids[0] = first;

        for (int k = 1; k < nclusters; ++k) {
            int last = selected.back();

            // Update min_dists using only the last selected medoid
            for (int i = 0; i < nelements; ++i) {
                double d = ptr_diss(i, last);
                if (d < min_dists[i]) min_dists[i] = d;
            }

            // Compute weighted total distance
            double total_weight = 0.0;
            for (int i = 0; i < nelements; ++i) {
                total_weight += min_dists[i] * ptr_weights[i];
            }

            // Handle degenerate case
            if (total_weight <= 1e-10) {
                int fallback = dist(rng);
                selected.push_back(fallback);
                ptr_centroids[k] = fallback;
                continue;
            }

            // Select next medoid using weighted probability
            uniform_real_distribution<double> rdist(0, total_weight);
            double r = rdist(rng), accumulator = 0.0;
            int next = -1;

            for (int i = 0; i < nelements; ++i) {
                accumulator += min_dists[i] * ptr_weights[i];
                if (accumulator >= r) {
                    next = i;
                    break;
                }
            }

            if (next == -1) next = dist(rng);  // fallback again just in case
            selected.push_back(next);
            ptr_centroids[k] = next;
        }
    }


    // Update medoids by selecting the element minimizing the sum of weighted distances to all other elements in the cluster
    void getclustermedoids() {
        auto ptr_weights = weights.unchecked<1>();
        auto ptr_diss = diss.unchecked<2>();
        auto ptr_centroids = centroids.mutable_unchecked<1>();

        #pragma omp parallel for schedule(dynamic)
        for (int k = 0; k < nclusters; ++k) {
            int size = clusterSize[k];
            double best = DBL_MAX;
            int bestID = 0;

            // Iterate over all members of cluster k to find the best medoid
            for (int i = 0; i < size; ++i) {
                int ii = clusterMembership[k * nelements + i];
                double current = 0;

                // Sum weighted distances from candidate medoid ii to all other members
                for (int j = 0; j < size; ++j) {
                    if (i == j) continue;
                    int jj = clusterMembership[k * nelements + j];
                    current += ptr_weights[jj] * ptr_diss(ii, jj);
                    if (current >= best) break;  // Early stop if worse than current best
                }

                if (current < best) {
                    best = current;
                    bestID = ii;
                }
            }

            ptr_centroids[k] = bestID;  // Assign best medoid for cluster k
        }
    }

    // Main loop to run the clustering process until convergence or max iterations
    py::array_t<int> runclusterloop() {
        auto ptr_weights = weights.unchecked<1>();
        auto ptr_diss = diss.unchecked<2>();
        auto ptr_centroids = centroids.mutable_unchecked<1>();

        double total = DBL_MAX;
        int counter = 0;
        int period = 10;  // Frequency to save cluster assignments for convergence checking

        while (counter <= npass) {
            PyErr_CheckSignals();  // Allow Python interruption

            double prev = total;
            total = 0;

            if (counter > 0) getclustermedoids();

            // Periodically save cluster assignment to check for convergence
            if (counter % period == 0) {
                for (int i = 0; i < nelements; ++i)
                    saved[i] = tclusterid[i];

                if (period < INT_MAX / 2) period *= 2;  // Exponentially increase period
            }

            counter++;

            vector<vector<int>> localMembers(nclusters);

            // Parallel assignment of elements to closest medoid
            #pragma omp parallel
            {
                vector<vector<int>> threadLocal(nclusters);

                #pragma omp for reduction(+:total) schedule(static)
                for (int i = 0; i < nelements; ++i) {
                    double dist = DBL_MAX;
                    int assign = 0;

                    // Find nearest medoid
                    for (int k = 0; k < nclusters; ++k) {
                        int j = ptr_centroids[k];
                        double tdistance = ptr_diss(i, j);

                        if (tdistance < dist) {
                            dist = tdistance;
                            assign = k;
                        }
                    }

                    tclusterid[i] = assign;
                    threadLocal[assign].push_back(i);
                    total += ptr_weights[i] * dist;
                }

                // Merge thread local cluster memberships into shared vector safely
                #pragma omp critical
                {
                    for (int k = 0; k < nclusters; ++k) {
                        localMembers[k].insert(
                            localMembers[k].end(),
                            threadLocal[k].begin(),
                            threadLocal[k].end()
                        );
                    }
                }
            }

            // Update cluster membership and sizes
            for (int k = 0; k < nclusters; ++k) {
                clusterSize[k] = static_cast<int>(localMembers[k].size());

                // If a cluster is empty, reinitialize medoids and restart
                if (clusterSize[k] == 0) {
                    init_medoids();
                    counter = 0;
                    break;
                }

                for (int i = 0; i < clusterSize[k]; ++i) {
                    clusterMembership[k * nelements + i] = localMembers[k][i];
                }
            }

            // Convergence check based on total cost change
            if (abs(total - prev) < 1e-6) break;

            // Check if cluster assignments are unchanged from last saved
            bool same = true;
            for (int i = 0; i < nelements; ++i) {
                if (saved[i] != tclusterid[i]) {
                    same = false;
                    break;
                }
            }

            if (same) break;
        }

        return getResultArray();
    }

    // Construct and return the final array of medoid assignments for each element
    py::array_t<int> getResultArray() const {
        py::array_t<int> result(nelements);
        auto results = result.mutable_unchecked<1>();
        auto centroid = centroids.unchecked<1>();

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nelements; ++i) {
            results(i) = centroid(tclusterid[i]);
        }

        return result;
    }
};
