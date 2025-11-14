#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <algorithm>
#define WEIGHTED_CLUST_TOL -1e-10
#include <cfloat>
#include <cmath>
#include <limits>

namespace py = pybind11;

class PAMonce {
public:
    PAMonce(int nelement, py::array_t<double> diss, py::array_t<int> centroids, int npass, py::array_t<double> weights){
        // 注释掉信息性打印，避免在并行环境（如 CLARA）中降低性能
        // py::print("[>] Starting Partitioning Around Medoids with a Once-Only Swap Pass (PAMonce)...");
        // std::cout << std::flush;

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

    // 小工具：把 vector 打成一行文本
//    static void debug_print_vec(const char* name,
//                                const std::vector<int>& v,
//                                std::size_t maxn = 50) {
//        std::ostringstream oss;
//        oss << name << " (n=" << v.size() << ") [";
//        std::size_t n = std::min<std::size_t>(v.size(), maxn);
//        for (std::size_t i = 0; i < n; ++i) {
//            if (i) oss << ", ";
//            oss << v[i];
//        }
//        if (v.size() > n) oss << ", ...";
//        oss << "]\n";
//        // 用 cerr 更容易立刻看到，并避免与 Python 输出缓冲混在一起
//        std::cerr << oss.str() << std::flush;
//    }

    double find_max_value(py::array_t<double> diss) {
        auto buf_info = diss.shape();
        auto ptr = diss.unchecked<2>();

        int rows = buf_info[0];
        int cols = buf_info[1];

        double max_val = -std::numeric_limits<double>::infinity();

        #pragma omp parallel
        {
            double thread_max = -std::numeric_limits<double>::infinity();
            #pragma omp for nowait
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    double val = ptr(i, j);
                    if (!std::isfinite(val)) {
                        continue;
                    }
                    thread_max = std::max(thread_max, val);
                }
            }

            #pragma omp critical
            {
                max_val = std::max(max_val, thread_max);
            }
        }

        if (!std::isfinite(max_val) || max_val <= 0.0) {
            max_val = 1.0;
        }

        return max_val;
    }

    py::array_t<int> runclusterloop() {
        auto ptr_diss = diss.unchecked<2>();
        auto ptr_weights = weights.unchecked<1>();
        auto ptr_centroids = centroids.mutable_data();
        auto ptr_clusterid = clusterid.mutable_data();

        for (int i = 0; i < nelement; i++) {
            ptr_clusterid[i] = -1;
        }

        double dzsky = 1;
        int hbest = -1, nbest = -1;
        double total = -1;

        do {
            // 为每个点寻找距离它最近和次近的中心点
        for (int i = 0; i < nelement; i++) {
            dysma[i] = maxdist;
            dysmb[i] = maxdist;
            for (int k = 0; k < nclusters; k++) {
                int i_cluster = ptr_centroids[k];
                double dist = ptr_diss(i, i_cluster);
                if (!std::isfinite(dist)) {
                    dist = maxdist;
                }

                if (dysma[i] >= dist) {
                    // 原码是‘>’，现改为‘>=’。
                    // 因为如果当前点 i 与所有 medoids 的距离都是 maxdist，那么将无法进入这个分支
                    dysmb[i] = dysma[i];
                    dysma[i] = dist;

                    tclusterid[i] = k;  // tclusterid 是中间变量，如果没有进入这个分支，那么将采用初始值-1
                } else if (dysmb[i] > dist) {
                    dysmb[i] = dist;
                }
            }

            if (tclusterid[i] < 0) {
                tclusterid[i] = 0;
                dysma[i] = maxdist;
            }
        }

            if (total < 0) {
                total = 0;
                #pragma omp parallel for reduction(+:total) schedule(static)
                for (int i = 0; i < nelement; i++) {
                    total += ptr_weights(i) * dysma[i];
                }
            }

            dzsky = 1;
            hbest = -1;
            nbest = -1;

            // 遍历每个聚类中心 i，寻找替换中心 h 的可能性
            for (int k = 0; k < nclusters; k++) {
                int i = ptr_centroids[k];
                double removeCost = 0;

                // 计算移除该 medoid 的成本
                #pragma omp parallel for reduction(+:removeCost) schedule(static)
                for (int j = 0; j < nelement; j++) {
                    if (tclusterid[j] == k) {
                        removeCost += ptr_weights(j) * (dysmb[j] - dysma[j]);
                        fvect[j] = dysmb[j];
                    } else {
                        fvect[j] = dysma[j];
                    }
                }

                // 查找最优的新 medoid h
                #pragma omp parallel
                {
                    double local_dzsky = 1;
                    int local_hbest = -1, local_nbest = -1;

                    #pragma omp for schedule(static)
                    for (int h = 0; h < nelement; h++) {
                        double dist_hi = ptr_diss(h, i);
                        if (!std::isfinite(dist_hi) || dist_hi <= 0) {
                            continue;
                        }
                        if (dist_hi > 0) {
                            double addGain = removeCost;
                            for (int j = 0; j < nelement; j++) {
                                double dist_hj = ptr_diss(h, j);
                                if (!std::isfinite(dist_hj)) {
                                    continue;
                                }
                                if (dist_hj < fvect[j]) {
                                    addGain += ptr_weights(j) * (dist_hj - fvect[j]);
                                }
                            }

                            if (local_dzsky > addGain) {
                                local_dzsky = addGain;
                                local_hbest = h;
                                local_nbest = i;
                            }
                        }
                    }

                    // 合并线程局部结果
                    #pragma omp critical
                    {
                        if (dzsky > local_dzsky) {
                            dzsky = local_dzsky;
                            hbest = local_hbest;
                            nbest = local_nbest;
                        }
                    }
                }
            }

            // 更新 medoids
            if (dzsky < WEIGHTED_CLUST_TOL) {
                for (int k = 0; k < nclusters; k++) {
                    if (ptr_centroids[k] == nbest) {
                        ptr_centroids[k] = hbest;
                    }
                }
                total += dzsky;
            }
        } while (dzsky < WEIGHTED_CLUST_TOL);

        // ---- 安全打印（无 pybind11，线程/进程均可）----
//        {
//            // 复制指针区的数据，形成可打印的 vector
//            std::vector<int> centroids_vec(ptr_centroids, ptr_centroids + nclusters);
//
//            // 注意：如果你用了 OpenMP，可选地只让一个线程打印，避免多线程交叉输出
//            // #pragma omp single
//            debug_print_vec("tclusterid", tclusterid);
//            debug_print_vec("ptr_centroids", centroids_vec);
//        }

        // 更新最终聚类分配
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < nelement; j++) {
            ptr_clusterid[j] = ptr_centroids[tclusterid[j]];
        }

//        {
//            std::vector<int> centroids_vec(ptr_clusterid, ptr_clusterid + nelement);
//            debug_print_vec("ptr_clusterid", centroids_vec);
//        }

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