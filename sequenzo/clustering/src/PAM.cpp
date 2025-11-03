#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cfloat>
#include <climits>
#include <cmath>
#define WEIGHTED_CLUST_TOL -1e-10
using namespace std;
namespace py = pybind11;

class PAM {
public:
    // Constructor: Initializes the PAM algorithm with required parameters.
    PAM(int nelements, py::array_t<double> diss,
        py::array_t<int> centroids, int npass, py::array_t<double> weights) {
        // 注释掉信息性打印，避免在并行环境（如 CLARA）中降低性能
        // py::print("[>] Starting Partitioning Around Medoids (PAM)...");

        try {
            this->nelements = nelements;
            this->centroids = centroids;
            this->npass = npass;
            this->weights = weights;
            this->diss = diss;
            this->maxdist = 0.0;
            this->nclusters = static_cast<int>(centroids.size()); // Number of clusters
            this->tclusterid.resize(nelements); // Initialize cluster id vector
            this->computeMaxDist(); // Compute the maximum distance for use later

            // Initialize dysma and dysmb with maxdist
            dysma.resize(nelements, maxdist);
            dysmb.resize(nelements, maxdist);
        } catch (const exception &e) {
            py::print("Error: ", e.what()); // Error handling
        }
    }

    // Computes the maximum distance between any two elements in the distance matrix.
    void computeMaxDist() {
        auto ptr_diss = diss.unchecked<2>();

        // The manual array collects the thread maxima
        int nthreads = 1;
        #ifdef _OPENMP
        #pragma omp parallel
        {
            #pragma omp single
            nthreads = omp_get_num_threads();
        }
        #endif

        std::vector<double> thread_max(nthreads, 0.0);

        #ifdef _OPENMP
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
        #else
        {
            int tid = 0;
        #endif
            double local = 0.0;

            #ifdef _OPENMP
            #pragma omp for schedule(static)
            #endif
            for (int i = 0; i < nelements; ++i) {
                for (int j = i + 1; j < nelements; ++j) {
                    double val = ptr_diss(i, j);
                    if (val > local) local = val;
                }
            }

            thread_max[tid] = local;
        }

        // Final reduction (serial, fast)
        double max_val = 0.0;
        for (double val : thread_max) {
            if (val > max_val) max_val = val;
        }

        maxdist = 1.1 * max_val + 1.0;
    }


    // Runs the PAM clustering loop, repeatedly updating centroids and assigning elements to clusters.
    py::array_t<int> runclusterloop() {
        auto ptr_weights = weights.unchecked<1>(); // Access to the weights
        auto ptr_diss = diss.unchecked<2>(); // Access to the distance matrix
        auto ptr_centroids = centroids.mutable_unchecked<1>(); // Access to the centroids

        double dzsky;
        int ipass = 0;
        int hbest = -1;
        int nbest = -1;
        int k, icluster, h;
        double total = -1.0;
        int nclusters = static_cast<int>(centroids.size());

        do {
            // Parallel loop to update dysma and dysmb based on current centroids
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nelements; i++) {
                dysmb[i] = maxdist;
                dysma[i] = maxdist;

                // Update dysma and dysmb values based on the distance to centroids
                for (int k = 0; k < nclusters; k++) {
                    int icluster = ptr_centroids(k);
                    double dist = ptr_diss(i, icluster);

                    if (dysma[i] > dist) {
                        dysmb[i] = dysma[i];
                        dysma[i] = dist;
                        tclusterid[i] = k; // Assign element to the current cluster
                    } else if (dysmb[i] > dist) {
                        dysmb[i] = dist;
                    }
                }
            }

            // If total hasn't been calculated yet, calculate it
            if (total < 0) {
                total = 0;

                // Parallel loop to calculate the total weighted distance
                #pragma omp parallel for reduction(+:total) schedule(static)
                for (int i = 0; i < nelements; i++) {
                    total += ptr_weights[i] * dysma[i];
                }
            }

            dzsky = 1; // Initialize dzsky to 1 for the change cost comparison

            // Parallel loop to compute the cost of switching elements' medoids
            #pragma omp parallel for schedule(dynamic)
            for (int h = 0; h < nelements; h++) {
                bool is_current_medoid = false;
                for (int k = 0; k < nclusters; k++) {
                    if (h == ptr_centroids[k]) {
                        is_current_medoid = true;
                        break;
                    }
                }

                if (is_current_medoid) // Skip if the element is already a medoid
                    continue;

                double local_dzsky = dzsky;
                int local_hbest = -1;
                int local_nbest = -1;

                // Evaluate the change cost for switching each element with a new medoid
                for (int k = 0; k < nclusters; k++) {
                    int i = ptr_centroids[k];
                    double dz = 0.0;

                    for (int j = 0; j < nelements; j++) {
                        if (ptr_diss(i, j) == dysma[j]) {
                            double small = (dysmb[j] > ptr_diss(h, j)) ? ptr_diss(h, j) : dysmb[j];
                            dz += ptr_weights[j] * (-dysma[j] + small); // Update change cost
                        } else if (ptr_diss(h, j) < dysma[j]) {
                            dz += ptr_weights[j] * (-dysma[j] + ptr_diss(h, j));
                        }
                    }

                    // Keep track of the best change
                    if (dz < local_dzsky) {
                        local_dzsky = dz;
                        local_hbest = h;
                        local_nbest = i;
                    }
                }

                // Critical section to update dzsky with the best change
                #pragma omp critical
                {
                    if (local_dzsky < dzsky) {
                        dzsky = local_dzsky;
                        hbest = local_hbest;
                        nbest = local_nbest;
                    }
                }
            }

            // If there was an improvement in the total cost, update the centroids
            if (dzsky < 0) {
                for (k = 0; k < nclusters; k++) {
                    if (ptr_centroids[k] == nbest) {
                        ptr_centroids[k] = hbest; // Swap the medoids
                    }
                }

                total += dzsky; // Update the total cost
            }

            ipass++; // Increment pass count
            if (ipass >= npass) {
                break; // Break if max passes reached
            }
        } while (dzsky < 0); // Repeat until no improvement

        return getResultArray(); // Return the final cluster assignments
    }

    // Returns an array of cluster assignments for each element
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


protected:
    int nelements;               // Number of elements to cluster
    py::array_t<double> diss;    // Distance matrix
    py::array_t<int> centroids;  // Initial centroids
    int npass;                   // Number of passes for the algorithm
    py::array_t<double> weights; // Element weights
    vector<int> tclusterid;      // Cluster IDs for each element
    vector<double> dysmb;        // Temporary variable for distances
    int nclusters;               // Number of clusters
    double maxdist;              // Maximum distance value
    vector<double> dysma;        // Temporary variable for distances
};
