#include "PAM.cpp"
#include "KMedoid.cpp"
#include "PAMonce.cpp"
#include "weightedinertia.cpp"
#include "cluster_quality.cpp"

namespace py = pybind11;

PYBIND11_MODULE(clustering_c_code, m) {
    py::class_<PAM>(m, "PAM")
            .def(py::init<int, py::array_t<double>, py::array_t<int>, int, py::array_t<double>>())
            .def("runclusterloop", &PAM::runclusterloop);

    py::class_<KMedoid>(m, "KMedoid")
            .def(py::init<int, py::array_t<double>, py::array_t<int>, int, py::array_t<double>>())
            .def("runclusterloop", &KMedoid::runclusterloop);

    py::class_<PAMonce>(m, "PAMonce")
            .def(py::init<int, py::array_t<double>, py::array_t<int>, int, py::array_t<double>>())
            .def("runclusterloop", &PAMonce::runclusterloop);

    py::class_<weightedinertia>(m, "weightedinertia")
            .def(py::init<py::array_t<double>, py::array_t<int>, py::array_t<double>>())
            .def("tmrWeightedInertiaContrib", &weightedinertia::tmrWeightedInertiaContrib);

    // Cluster Quality functions
    m.def("cluster_quality", [](py::array_t<double> diss_matrix, 
                               py::array_t<int> cluster_labels,
                               py::array_t<double> weights,
                               int nclusters) -> py::dict {
        auto diss_buf = diss_matrix.request();
        auto cluster_buf = cluster_labels.request();
        auto weights_buf = weights.request();
        
        if (diss_buf.ndim != 2 || diss_buf.shape[0] != diss_buf.shape[1]) {
            throw std::runtime_error("Distance matrix must be square");
        }
        
        int n = diss_buf.shape[0];
        
        if (cluster_buf.size != n || weights_buf.size != n) {
            throw std::runtime_error("Cluster labels and weights must have same length as matrix dimension");
        }
        
        double* diss_ptr = static_cast<double*>(diss_buf.ptr);
        int* cluster_ptr = static_cast<int*>(cluster_buf.ptr);
        double* weights_ptr = static_cast<double*>(weights_buf.ptr);
        
        // Prepare output arrays
        std::vector<double> stats(ClusterQualNumStat);
        std::vector<double> asw(2 * nclusters);
        
        // Create Kendall tree for caching
        KendallTree kendall;
        
        // Call core function
        clusterquality(diss_ptr, cluster_ptr, weights_ptr, n, 
                      stats.data(), nclusters, asw.data(), kendall);
        
        // Clean up Kendall tree
        finalizeKendall(kendall);
        
        // Return results as dictionary
        py::dict result;
        result["PBC"] = stats[ClusterQualHPG];  // PBC is stored in HPG position
        result["HG"] = stats[ClusterQualHG];
        result["HGSD"] = stats[ClusterQualHGSD];
        result["ASW"] = stats[ClusterQualASWi];
        result["ASWw"] = stats[ClusterQualASWw];
        result["CH"] = stats[ClusterQualF];
        result["R2"] = stats[ClusterQualR];
        result["CHsq"] = stats[ClusterQualF2];
        result["R2sq"] = stats[ClusterQualR2];
        result["HC"] = stats[ClusterQualHC];
        
        // Convert ASW array to numpy array
        auto asw_array = py::array_t<double>(2 * nclusters);
        auto asw_buf = asw_array.request();
        double* asw_out = static_cast<double*>(asw_buf.ptr);
        std::copy(asw.begin(), asw.end(), asw_out);
        result["cluster_asw"] = asw_array;
        
        return result;
    }, "Compute cluster quality indicators for full distance matrix");

    m.def("cluster_quality_condensed", [](py::array_t<double> diss_condensed,
                                         py::array_t<int> cluster_labels,
                                         py::array_t<double> weights,
                                         int n, int nclusters) -> py::dict {
        auto diss_buf = diss_condensed.request();
        auto cluster_buf = cluster_labels.request();
        auto weights_buf = weights.request();
        
        int expected_size = n * (n - 1) / 2;
        if (diss_buf.size != expected_size) {
            throw std::runtime_error("Condensed distance array size mismatch");
        }
        
        if (cluster_buf.size != n || weights_buf.size != n) {
            throw std::runtime_error("Cluster labels and weights must have length n");
        }
        
        double* diss_ptr = static_cast<double*>(diss_buf.ptr);
        int* cluster_ptr = static_cast<int*>(cluster_buf.ptr);
        double* weights_ptr = static_cast<double*>(weights_buf.ptr);
        
        // Prepare output arrays
        std::vector<double> stats(ClusterQualNumStat);
        std::vector<double> asw(2 * nclusters);
        
        // Create Kendall tree for caching
        KendallTree kendall;
        
        // Call core function
        clusterquality_dist(diss_ptr, cluster_ptr, weights_ptr, n,
                           stats.data(), nclusters, asw.data(), kendall);
        
        // Clean up Kendall tree
        finalizeKendall(kendall);
        
        // Return results as dictionary
        py::dict result;
        result["PBC"] = stats[ClusterQualHPG];  // PBC is stored in HPG position
        result["HG"] = stats[ClusterQualHG];
        result["HGSD"] = stats[ClusterQualHGSD];
        result["ASW"] = stats[ClusterQualASWi];
        result["ASWw"] = stats[ClusterQualASWw];
        result["CH"] = stats[ClusterQualF];
        result["R2"] = stats[ClusterQualR];
        result["CHsq"] = stats[ClusterQualF2];
        result["R2sq"] = stats[ClusterQualR2];
        result["HC"] = stats[ClusterQualHC];
        
        // Convert ASW array to numpy array
        auto asw_array = py::array_t<double>(2 * nclusters);
        auto asw_buf = asw_array.request();
        double* asw_out = static_cast<double*>(asw_buf.ptr);
        std::copy(asw.begin(), asw.end(), asw_out);
        result["cluster_asw"] = asw_array;
        
        return result;
    }, "Compute cluster quality indicators for condensed distance array");

    m.def("individual_asw", [](py::array_t<double> diss_matrix,
                              py::array_t<int> cluster_labels,
                              py::array_t<double> weights,
                              int nclusters) -> py::dict {
        auto diss_buf = diss_matrix.request();
        auto cluster_buf = cluster_labels.request();
        auto weights_buf = weights.request();
        
        if (diss_buf.ndim != 2 || diss_buf.shape[0] != diss_buf.shape[1]) {
            throw std::runtime_error("Distance matrix must be square");
        }
        
        int n = diss_buf.shape[0];
        
        if (cluster_buf.size != n || weights_buf.size != n) {
            throw std::runtime_error("Cluster labels and weights must have same length as matrix dimension");
        }
        
        double* diss_ptr = static_cast<double*>(diss_buf.ptr);
        int* cluster_ptr = static_cast<int*>(cluster_buf.ptr);
        double* weights_ptr = static_cast<double*>(weights_buf.ptr);
        
        // Prepare output arrays
        auto asw_i = py::array_t<double>(n);
        auto asw_w = py::array_t<double>(n);
        
        auto asw_i_buf = asw_i.request();
        auto asw_w_buf = asw_w.request();
        
        double* asw_i_ptr = static_cast<double*>(asw_i_buf.ptr);
        double* asw_w_ptr = static_cast<double*>(asw_w_buf.ptr);
        
        // Call core function
        indiv_asw(diss_ptr, cluster_ptr, weights_ptr, n, nclusters, asw_i_ptr, asw_w_ptr);
        
        // Return results as dictionary
        py::dict result;
        result["asw_individual"] = asw_i;
        result["asw_weighted"] = asw_w;
        
        return result;
    }, "Compute individual ASW scores for all samples");

    m.def("individual_asw_condensed", [](py::array_t<double> diss_condensed,
                                        py::array_t<int> cluster_labels,
                                        py::array_t<double> weights,
                                        int n, int nclusters) -> py::dict {
        auto diss_buf = diss_condensed.request();
        auto cluster_buf = cluster_labels.request();
        auto weights_buf = weights.request();
        
        int expected_size = n * (n - 1) / 2;
        if (diss_buf.size != expected_size) {
            throw std::runtime_error("Condensed distance array size mismatch");
        }
        
        if (cluster_buf.size != n || weights_buf.size != n) {
            throw std::runtime_error("Cluster labels and weights must have length n");
        }
        
        double* diss_ptr = static_cast<double*>(diss_buf.ptr);
        int* cluster_ptr = static_cast<int*>(cluster_buf.ptr);
        double* weights_ptr = static_cast<double*>(weights_buf.ptr);
        
        // Prepare output arrays
        auto asw_i = py::array_t<double>(n);
        auto asw_w = py::array_t<double>(n);
        
        auto asw_i_buf = asw_i.request();
        auto asw_w_buf = asw_w.request();
        
        double* asw_i_ptr = static_cast<double*>(asw_i_buf.ptr);
        double* asw_w_ptr = static_cast<double*>(asw_w_buf.ptr);
        
        // Call core function
        indiv_asw_dist(diss_ptr, cluster_ptr, weights_ptr, n, nclusters, asw_i_ptr, asw_w_ptr);
        
        // Return results as dictionary
        py::dict result;
        result["asw_individual"] = asw_i;
        result["asw_weighted"] = asw_w;
        
        return result;
    }, "Compute individual ASW scores for condensed distance array");
}