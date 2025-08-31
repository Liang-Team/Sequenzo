#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>
#define WEIGHTED_CLUST_TOL -1e-10

namespace py = pybind11;

class PAMonce {
public:
    PAMonce(int nelement, py::array_t<double> diss, py::array_t<int> centroids, int npass, py::array_t<double> weights){
        py::print("[>] Starting Partitioning Around Medoids with a Once-Only Swap Pass (PAMonce)...");
        std::cout << std::flush;

        try {
            this->nelement = nelement;
            this->diss = diss;
            this->centroids = centroids;
            this->npass = npass;
            this->weights = weights;

            clusterid = py::array_t<int>(nelement);
            tclusterid.resize(nelement, -1);

            maxdist = find_max_value(diss);
            dysma.resize(nelement, maxdist);
            dysmb.resize(nelement, maxdist);

            fvect.resize(nelement, 0);
            nclusters = centroids.size();
        } catch (const std::exception& e){
            py::print("Error in constructor: ", e.what());
            throw;
        }
    }

    double find_max_value(py::array_t<double> diss) {
        auto buf_info = diss.shape();
        auto ptr = diss.unchecked<2>();

        int rows = buf_info[0];
        int cols = buf_info[1];

        double max_val = -1;
        for (int i = 0; i < rows ; ++i){
            for (int j = i; j < cols; ++j){
                max_val = std::max(max_val, ptr(i, j));
            }
        }

//        double max_val = -std::numeric_limits<double>::infinity();
//
//        #pragma omp parallel
//        {
//            double thread_max = -std::numeric_limits<double>::infinity();
//            #pragma omp for nowait
//            for (int i = 0; i < rows; ++i) {
//                for (int j = 0; j < cols; ++j) {
//                    thread_max = std::max(thread_max, ptr(i, j));
//                }
//            }
//
//            #pragma omp critical
//            {
//                max_val = std::max(max_val, thread_max);
//            }
//        }

        return max_val;
    }

    py::array_t<int> runclusterloop() {
        auto ptr_diss = diss.unchecked<2>();
        auto ptr_weights = weights.unchecked<1>();
        auto ptr_centroids = centroids.mutable_data();
        auto ptr_clusterid = clusterid.mutable_data();

//        for (int i = 0; i < nelement; i++) {
            ptr_clusterid = keep;
//        }

        double dzsky = 1;
        int hbest = -1, nbest = -1;
        double total = -1;

        do {
            for (int i = 0; i < nelement; i++) {
                dysma[i] = maxdist;
                dysmb[i] = maxdist;
                for (int k = 0; k < nclusters; k++) {
                    int i_cluster = ptr_centroids[k];
                    double dist = ptr_diss(i, i_cluster);

                    if (dysma[i] > dist) {
                        dysmb[i] = dysma[i];
                        dysma[i] = dist;

                        tclusterid[i] = k;
                    } else if (dysmb[i] > dist) {
                        dysmb[i] = dist;
                    }
                }
            }

            py::print("=== 1 ===");

            if (total < 0) {
                total = 0;
//                #pragma omp parallel for reduction(+:total) schedule(static)
                for (int i = 0; i < nelement; i++) {
                    total += ptr_weights(i) * dysma[i];
                }
            }

            py::print("=== 2 ===");

            dzsky = 1;
            hbest = -1;
            nbest = -1;

            // 遍历每个聚类中心 i，寻找替换中心 h 的可能性
            for (int k = 0; k < nclusters; k++) {
                int i = ptr_centroids[k];
                double removeCost = 0;

                py::print("=== 3 ===");

                // 计算移除该 medoid 的成本
//                #pragma omp parallel for reduction(+:removeCost) schedule(static)
                for (int j = 0; j < nelement; j++) {
                    if (tclusterid[j] == k) {
                        removeCost += ptr_weights(j) * (dysmb[j] - dysma[j]);
                        fvect[j] = dysmb[j];
                    } else {
                        fvect[j] = dysma[j];
                    }
                }

                py::print("=== 4 ===");

                // 查找最优的新 medoid h
                for (int h = 0; h < nelement; h++) {
                    if (ptr_diss(h, i) > 0) {
                        double addGain = removeCost;
                        py::print("=== 5 ===");
                        for (int j = 0; j < nelement; j++) {
                            if (ptr_diss(h, j) < fvect[j]) {
                                addGain += ptr_weights(j) * (ptr_diss(h, j) - fvect[j]);
                            }
                        }
                        py::print("addGain = ", addGain);
                        py::print("=== 6 ===");

                        if (dzsky > addGain) {
                            dzsky = addGain;
                            hbest = h;
                            nbest = i;
                        }
                    }
                }
            }

            // 更新 medoids
            py::print("dzsky = ", dzsky);
            if (dzsky < WEIGHTED_CLUST_TOL) {
                py::print("=== 7 ===");
                for (int k = 0; k < nclusters; k++) {
                    if (ptr_centroids[k] == nbest) {
                        ptr_centroids[k] = hbest;
                    }
                }
                py::print("=== 8 ===");
                total += dzsky;
            }
        } while (dzsky < WEIGHTED_CLUST_TOL);

        py::print("=== 9 ===");

        // 更新最终聚类分配
        int init = ptr_centroids[0];
//        #pragma omp parallel for
        for (int j = 0; j < nelement; j++) {
            if (tclusterid[j] != -1){
                ptr_clusterid[j] = ptr_centroids[tclusterid[j]];
            } else{
                ptr_clusterid[j] = init;
            }
        }
        py::print("=== 10 ===");

        return clusterid;
    }


private:
    int nelement;
    py::array_t<double> diss;
    py::array_t<int> centroids;
    int npass;
    py::array_t<double> weights;

    py::array_t<int> clusterid;
    std::vector<int> tclusterid;

    double maxdist;
    std::vector<double> dysma;
    std::vector<double> dysmb;

    std::vector<double> fvect;
    int nclusters;
};