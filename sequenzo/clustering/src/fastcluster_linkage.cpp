/*
 * fastcluster_linkage.cpp
 *
 * Pure C++ bridge to fastcluster's core algorithms.
 * Provides compute_linkage_condensed() and compute_linkage_vector_euclidean()
 * without any Python/NumPy dependency.
 */

#include "fastcluster_linkage.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <algorithm>
#include <new>
#include <exception>
#include <stdexcept>
#include <string>

// fc_isnan must be defined before including fastcluster.cpp
#define fc_isnan(X) ((X)!=(X))

#include "fastcluster.cpp"

// ============================================================================
// linkage_output: writes cluster_result into (N-1)x4 flat array
// Adapted from fastcluster_python.cpp lines 82-104.
// ============================================================================

class linkage_output {
    t_float* Z;
public:
    explicit linkage_output(t_float* Z_) : Z(Z_) {}

    void append(t_index node1, t_index node2, t_float dist, t_float size) {
        if (node1 < node2) {
            *(Z++) = static_cast<t_float>(node1);
            *(Z++) = static_cast<t_float>(node2);
        } else {
            *(Z++) = static_cast<t_float>(node2);
            *(Z++) = static_cast<t_float>(node1);
        }
        *(Z++) = dist;
        *(Z++) = size;
    }
};

// ============================================================================
// generate_SciPy_dendrogram: cluster_result -> (N-1)x4 linkage matrix
// Adapted from fastcluster_python.cpp lines 117-144.
// ============================================================================

#undef Z_
#define Z_(r_, c_) (Z[(r_)*4+(c_)])
#define size_(r_) ( ((r_<N) ? 1 : Z_(r_-N,3)) )

template <bool sorted>
static void generate_dendrogram(t_float* Z, cluster_result& Z2, t_index N) {
    union_find nodes(sorted ? 0 : N);
    if (!sorted) {
        std::stable_sort(Z2[0], Z2[N - 1]);
    }

    linkage_output output(Z);
    t_index node1, node2;

    for (node const* NN = Z2[0]; NN != Z2[N - 1]; ++NN) {
        if (sorted) {
            node1 = NN->node1;
            node2 = NN->node2;
        } else {
            node1 = nodes.Find(NN->node1);
            node2 = nodes.Find(NN->node2);
            nodes.Union(node1, node2);
        }
        output.append(node1, node2, NN->dist, size_(node1) + size_(node2));
    }
}

#undef size_
#undef Z_

// ============================================================================
// euclidean_dissimilarity: pure C++ replacement for python_dissimilarity
// Only supports Ward/Centroid/Median with Euclidean metric.
// ============================================================================

class euclidean_dissimilarity {
    t_float* Xa;
    std::ptrdiff_t dim;
    t_index N;
    auto_array_ptr<t_float> Xnew;
    t_index* members;
    void (cluster_result::*postprocessfn)(const t_float) const;

    euclidean_dissimilarity();
    euclidean_dissimilarity(euclidean_dissimilarity const&);
    euclidean_dissimilarity& operator=(euclidean_dissimilarity const&);

public:
    euclidean_dissimilarity(t_float* X_data, t_index N_, std::ptrdiff_t dim_,
                            t_index* members_, int method_code,
                            bool temp_point_array)
        : Xa(X_data),
          dim(dim_),
          N(N_),
          Xnew(temp_point_array ? static_cast<std::ptrdiff_t>(N_ - 1) * dim_ : 0),
          members(members_),
          postprocessfn(nullptr)
    {
        switch (method_code) {
        case METHOD_METR_WARD:
            postprocessfn = &cluster_result::sqrtward;
            break;
        case METHOD_METR_WARD_D2:
            postprocessfn = &cluster_result::sqrtdouble;
            break;
        default:
            postprocessfn = &cluster_result::sqrt;
        }
    }

    ~euclidean_dissimilarity() {}

    inline t_float operator()(t_index i, t_index j) const {
        return sqeuclidean<false>(i, j);
    }

    inline t_float X(t_index i, t_index j) const {
        return Xa[i * dim + j];
    }

    void merge(t_index i, t_index j, t_index newnode) const {
        t_float const* Pi = i < N ? Xa + i * dim : Xnew + (i - N) * dim;
        t_float const* Pj = j < N ? Xa + j * dim : Xnew + (j - N) * dim;
        for (std::ptrdiff_t k = 0; k < dim; ++k) {
            Xnew[(newnode - N) * dim + k] =
                (Pi[k] * static_cast<t_float>(members[i]) +
                 Pj[k] * static_cast<t_float>(members[j])) /
                static_cast<t_float>(members[i] + members[j]);
        }
        members[newnode] = members[i] + members[j];
    }

    void merge_weighted(t_index i, t_index j, t_index newnode) const {
        t_float const* Pi = i < N ? Xa + i * dim : Xnew + (i - N) * dim;
        t_float const* Pj = j < N ? Xa + j * dim : Xnew + (j - N) * dim;
        for (std::ptrdiff_t k = 0; k < dim; ++k) {
            Xnew[(newnode - N) * dim + k] = (Pi[k] + Pj[k]) * 0.5;
        }
    }

    void merge_inplace(t_index i, t_index j) const {
        t_float const* Pi = Xa + i * dim;
        t_float* Pj = Xa + j * dim;
        for (std::ptrdiff_t k = 0; k < dim; ++k) {
            Pj[k] = (Pi[k] * static_cast<t_float>(members[i]) +
                      Pj[k] * static_cast<t_float>(members[j])) /
                     static_cast<t_float>(members[i] + members[j]);
        }
        members[j] += members[i];
    }

    void merge_inplace_weighted(t_index i, t_index j) const {
        t_float const* Pi = Xa + i * dim;
        t_float* Pj = Xa + j * dim;
        for (std::ptrdiff_t k = 0; k < dim; ++k) {
            Pj[k] = (Pi[k] + Pj[k]) * 0.5;
        }
    }

    void postprocess(cluster_result& Z2) const {
        if (postprocessfn != nullptr) {
            (Z2.*postprocessfn)(0);
        }
    }

    inline t_float ward(t_index i, t_index j) const {
        t_float mi = static_cast<t_float>(members[i]);
        t_float mj = static_cast<t_float>(members[j]);
        return sqeuclidean<true>(i, j) * mi * mj / (mi + mj);
    }

    inline t_float ward_initial(t_index i, t_index j) const {
        return sqeuclidean<true>(i, j);
    }

    inline static t_float ward_initial_conversion(t_float min) {
        return min * 0.5;
    }

    inline t_float ward_extended(t_index i, t_index j) const {
        t_float mi = static_cast<t_float>(members[i]);
        t_float mj = static_cast<t_float>(members[j]);
        return sqeuclidean_extended(i, j) * mi * mj / (mi + mj);
    }

    template <bool check_NaN>
    t_float sqeuclidean(t_index i, t_index j) const {
        t_float sum = 0;
        t_float const* Pi = Xa + i * dim;
        t_float const* Pj = Xa + j * dim;
        for (std::ptrdiff_t k = 0; k < dim; ++k) {
            t_float diff = Pi[k] - Pj[k];
            sum += diff * diff;
        }
        if (check_NaN) {
            if (fc_isnan(sum))
                throw(nan_error());
        }
        return sum;
    }

    t_float sqeuclidean_extended(t_index i, t_index j) const {
        t_float sum = 0;
        t_float const* Pi = i < N ? Xa + i * dim : Xnew + (i - N) * dim;
        t_float const* Pj = j < N ? Xa + j * dim : Xnew + (j - N) * dim;
        for (std::ptrdiff_t k = 0; k < dim; ++k) {
            t_float diff = Pi[k] - Pj[k];
            sum += diff * diff;
        }
        if (fc_isnan(sum))
            throw(nan_error());
        return sum;
    }
};

// ============================================================================
// Public API: compute_linkage_condensed
// ============================================================================

void compute_linkage_condensed(
    double* condensed, int N, int method_code,
    double* Z_out)
{
    if (N < 1) {
        throw std::runtime_error("At least one element is needed for clustering.");
    }
    if (N == 1) {
        return;
    }

    t_index n = static_cast<t_index>(N);
    t_float* D_ = condensed;
    cluster_result Z2(n - 1);

    auto_array_ptr<t_index> members;
    if (method_code == METHOD_METR_AVERAGE ||
        method_code == METHOD_METR_WARD ||
        method_code == METHOD_METR_WARD_D2 ||
        method_code == METHOD_METR_CENTROID) {
        members.init(n, 1);
    }

    // Operate on squared distances for these methods.
    if (method_code == METHOD_METR_WARD_D2 ||
        method_code == METHOD_METR_CENTROID ||
        method_code == METHOD_METR_MEDIAN) {
        for (t_float* DD = D_;
             DD != D_ + static_cast<std::ptrdiff_t>(n) * (n - 1) / 2;
             ++DD)
            *DD *= *DD;
    }

    switch (method_code) {
    case METHOD_METR_SINGLE:
        MST_linkage_core(n, D_, Z2);
        break;
    case METHOD_METR_COMPLETE:
        NN_chain_core<METHOD_METR_COMPLETE, t_index>(n, D_, NULL, Z2);
        break;
    case METHOD_METR_AVERAGE:
        NN_chain_core<METHOD_METR_AVERAGE, t_index>(n, D_, members, Z2);
        break;
    case METHOD_METR_WEIGHTED:
        NN_chain_core<METHOD_METR_WEIGHTED, t_index>(n, D_, NULL, Z2);
        break;
    case METHOD_METR_WARD:
    case METHOD_METR_WARD_D2:
        NN_chain_core<METHOD_METR_WARD_D2, t_index>(n, D_, members, Z2);
        break;
    case METHOD_METR_CENTROID:
        generic_linkage<METHOD_METR_CENTROID, t_index>(n, D_, members, Z2);
        break;
    case METHOD_METR_MEDIAN:
        generic_linkage<METHOD_METR_MEDIAN, t_index>(n, D_, NULL, Z2);
        break;
    default:
        throw std::runtime_error("Invalid method index.");
    }

    if (method_code == METHOD_METR_WARD_D2 ||
        method_code == METHOD_METR_CENTROID ||
        method_code == METHOD_METR_MEDIAN) {
        Z2.sqrt();
    }

    if (method_code == METHOD_METR_CENTROID ||
        method_code == METHOD_METR_MEDIAN) {
        generate_dendrogram<true>(Z_out, Z2, n);
    } else {
        generate_dendrogram<false>(Z_out, Z2, n);
    }
}

// ============================================================================
// Public API: compute_linkage_vector_euclidean
// ============================================================================

void compute_linkage_vector_euclidean(
    double* X, int N, int D, int method_code,
    double* Z_out)
{
    if (N < 1) {
        throw std::runtime_error("At least one element is needed for clustering.");
    }
    if (D < 1) {
        throw std::runtime_error("Invalid dimension of the data set.");
    }
    if (N == 1) {
        return;
    }

    t_index n = static_cast<t_index>(N);
    std::ptrdiff_t dim = static_cast<std::ptrdiff_t>(D);

    cluster_result Z2(n - 1);

    auto_array_ptr<t_index> members;
    if (method_code == METHOD_METR_WARD ||
        method_code == METHOD_METR_WARD_D2 ||
        method_code == METHOD_METR_CENTROID) {
        members.init(2 * n - 1, 1);
    }

    bool temp_point_array = (method_code == METHOD_METR_CENTROID ||
                             method_code == METHOD_METR_MEDIAN);

    euclidean_dissimilarity dist(X, n, dim, members, method_code,
                                 temp_point_array);

    // Both ward_d and ward_d2 use METHOD_VECTOR_WARD as the merge algorithm;
    // the only difference is in the postprocess function selected by
    // euclidean_dissimilarity's constructor based on method_code.
    switch (method_code) {
    case METHOD_METR_WARD:
    case METHOD_METR_WARD_D2:
        generic_linkage_vector<METHOD_VECTOR_WARD>(n, dist, Z2);
        break;
    case METHOD_METR_CENTROID:
        generic_linkage_vector_alternative<METHOD_VECTOR_CENTROID>(n, dist, Z2);
        break;
    case METHOD_METR_MEDIAN:
        generic_linkage_vector_alternative<METHOD_VECTOR_MEDIAN>(n, dist, Z2);
        break;
    default:
        throw std::runtime_error(
            "linkage_vector_euclidean only supports ward, ward_d2, centroid, median.");
    }

    if (method_code == METHOD_METR_WARD ||
        method_code == METHOD_METR_WARD_D2 ||
        method_code == METHOD_METR_CENTROID) {
        members.free();
    }

    dist.postprocess(Z2);

    if (method_code != METHOD_METR_SINGLE) {
        generate_dendrogram<true>(Z_out, Z2, n);
    } else {
        generate_dendrogram<false>(Z_out, Z2, n);
    }
}

// ============================================================================
// Public API: method_string_to_code
// ============================================================================

int method_string_to_code(const std::string& method) {
    if (method == "single")   return METHOD_METR_SINGLE;
    if (method == "complete") return METHOD_METR_COMPLETE;
    if (method == "average")  return METHOD_METR_AVERAGE;
    if (method == "weighted") return METHOD_METR_WEIGHTED;
    if (method == "ward" || method == "ward_d")
        return METHOD_METR_WARD;
    if (method == "ward_d2")  return METHOD_METR_WARD_D2;
    if (method == "centroid") return METHOD_METR_CENTROID;
    if (method == "median")   return METHOD_METR_MEDIAN;
    throw std::runtime_error(
        "Unsupported clustering method '" + method +
        "'. Supported: single, complete, average, weighted, "
        "ward, ward_d, ward_d2, centroid, median.");
}
