#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>
#include <xsimd/xsimd.hpp>
#define WEIGHTED_CLUST_TOL -1e-10
#ifdef _OPENMP
    #include <omp.h>
#endif

namespace py = pybind11;

class PAMonce {
public:
    PAMonce(int nelement, py::array_t<double> diss, py::array_t<int> centroids, int npass, py::array_t<double> weights){
        py::print("[>] Starting Partitioning Around Medoids (PAMonce)");
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

        double max_val = -std::numeric_limits<double>::infinity(); // 初始化为负无穷

        constexpr int batch_size = xsimd::batch<double>::size;

        #pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j += batch_size) {
                xsimd::batch<double> batch_vals = xsimd::load_unaligned(&ptr(i, j));
                double batch_max = xsimd::reduce_max(batch_vals);
                max_val = std::max(max_val, batch_max);
            }
        }

        return max_val;
    }

    py::array_t<int> runclusterloop(){
        auto ptr_diss = diss.unchecked<2>();
        auto ptr_weights = weights.unchecked<1>();
        auto ptr_centroids = centroids.mutable_unchecked<1>();

        auto ptr_clusterid = clusterid.mutable_unchecked<1>();

        for(int i=0; i<nelement; i++){
            ptr_clusterid(i) = -1;
        }

        double dzsky = 1;
        int hbest = -1, nbest = -1;
        double total = -1;

        xsimd::batch<double> max_val(std::numeric_limits<double>::infinity()); // 默认是正无穷

        do {
            #pragma omp parallel for
            for(int i=0; i<nelement; i++){
                for(int k=0; k<nclusters; k++){
                    int i_cluster = ptr_centroids(k);

                    xsimd::batch<double> diss_val(ptr_diss(i, i_cluster));
                    if(dysma[i] > diss_val.get(0)){
                        dysmb[i] = dysma[i];
                        dysma[i] = diss_val.get(0);
                        tclusterid[i] = k;
                    }else if(dysmb[i] > diss_val.get(0)){
                        dysmb[i] = diss_val.get(0);
                    }
                }
            }

            // 计算 total
            if(total < 0){
                total = 0;
                xsimd::batch<double> batch_total(0.0);
                for(int i = 0; i < nelement; i++){
                    xsimd::batch<double> weight = xsimd::load_unaligned(&ptr_weights(i));
                    xsimd::batch<double> diss_val(dysma[i]);
                    batch_total += weight * diss_val;
                }

                total = xsimd::reduce_add(batch_total);
            }

            dzsky = 1;

            // 计算 removeCost
            #pragma omp parallel for
            for(int k=0; k<nclusters; k++){
                int i = ptr_centroids(k);
                double removeCost = 0;

                xsimd::batch<double> batch_removeCost(0.0);
                for(int j = 0; j < nelement; j++){
                    if(tclusterid[j] == k){
                        xsimd::batch<double> weight = xsimd::load_unaligned(&ptr_weights(j));
                        xsimd::batch<double> diff(dysmb[j] - dysma[j]);
                        batch_removeCost += weight * diff;
                        fvect[j] = dysmb[j];
                    }else{
                        fvect[j] = dysma[j];
                    }
                }

                removeCost = xsimd::reduce_add(batch_removeCost);

                // 计算 addGain
                for(int h = 0; h < nelement; h++){
                    if(ptr_diss(h, i) > 0){
                        double addGain = removeCost;

                        xsimd::batch<double> batch_addGain(addGain);

                        for(int j = 0; j < nelement; j++){
                            if(ptr_diss(h, j) < fvect[j]){
                                xsimd::batch<double> weight = xsimd::load_unaligned(&ptr_weights(j));
                                xsimd::batch<double> diss_val(ptr_diss(h, j));
                                xsimd::batch<double> fvect_val(fvect[j]);
                                batch_addGain += weight * (diss_val - fvect_val);
                            }
                        }

                        if(dzsky > batch_addGain.get(0)){
                            dzsky = batch_addGain.get(0);
                            hbest = h;
                            nbest = i;
                        }
                    }
                }
            }

            // 更新 centroids
            if(dzsky < WEIGHTED_CLUST_TOL){
                for(int k=0; k<nclusters; k++){
                    if(ptr_centroids(k) == nbest){
                        ptr_centroids(k) = hbest;
                    }
                }

                total += dzsky;
            }
        } while (dzsky < WEIGHTED_CLUST_TOL);

        // 更新 clusterid
        for(int j = 0; j < nelement; j++){
            ptr_clusterid(j) = ptr_centroids[tclusterid[j]];
        }

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