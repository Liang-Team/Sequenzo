#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace py = pybind11;

// Cluster Quality Index constants (matching R WeightedCluster package)
#define ClusterQualHPG 0   // Hubert's Gamma Prime (not implemented in this version)
#define ClusterQualHG 1    // Hubert's Gamma
#define ClusterQualHGSD 2  // Hubert's Gamma Standard Deviation
#define ClusterQualASWi 3  // Average Silhouette Width (individual)
#define ClusterQualASWw 4  // Average Silhouette Width (weighted)
#define ClusterQualF 5     // Calinski-Harabasz (F statistic)
#define ClusterQualR 6     // R-squared
#define ClusterQualF2 7    // Calinski-Harabasz squared
#define ClusterQualR2 8    // R-squared squared
#define ClusterQualHC 9    // Hierarchical Criterion
#define ClusterQualNumStat 10

/**
 * Class for caching pairwise distance comparisons used in Kendall's tau calculations
 * This corresponds to the CmpCluster class in R's implementation
 */
class CmpCluster {
public:
    double clustDist0;
    double clustDist1;
    
    CmpCluster() : clustDist0(0.0), clustDist1(0.0) {}
    ~CmpCluster() {}
};

typedef std::map<double, CmpCluster*> KendallTree;

/**
 * Core cluster quality computation functions
 * These match the R WeightedCluster package implementation
 */

/**
 * Compute all cluster quality indicators for a distance matrix
 * 
 * @param diss Distance matrix (full square form, n x n)
 * @param cluster Cluster labels (1-based, as in R)
 * @param weights Sample weights
 * @param n Number of samples
 * @param stats Output array for statistics [ClusterQualNumStat]
 * @param nclusters Number of clusters
 * @param asw Output array for cluster-level ASW [2 * nclusters]
 * @param kendall Reference to Kendall tree for caching
 */
void clusterquality(const double* diss, const int* cluster, const double* weights,
                   int n, double* stats, int nclusters, double* asw, 
                   KendallTree& kendall);

/**
 * Compute all cluster quality indicators for a condensed distance array
 * 
 * @param diss Condensed distance array (upper triangle, length n*(n-1)/2)
 * @param cluster Cluster labels (1-based, as in R)
 * @param weights Sample weights
 * @param n Number of samples
 * @param stats Output array for statistics [ClusterQualNumStat]
 * @param nclusters Number of clusters
 * @param asw Output array for cluster-level ASW [2 * nclusters]
 * @param kendall Reference to Kendall tree for caching
 */
void clusterquality_dist(const double* diss, const int* cluster, const double* weights,
                        int n, double* stats, int nclusters, double* asw,
                        KendallTree& kendall);

/**
 * Compute individual ASW scores for all samples
 * 
 * @param diss Distance matrix (full square form, n x n)
 * @param cluster Cluster labels (1-based, as in R)
 * @param weights Sample weights
 * @param n Number of samples
 * @param nclusters Number of clusters
 * @param asw_i Output array for individual ASW [n]
 * @param asw_w Output array for weighted individual ASW [n]
 */
void indiv_asw(const double* diss, const int* cluster, const double* weights,
               int n, int nclusters, double* asw_i, double* asw_w);

/**
 * Compute individual ASW scores for condensed distance array
 */
void indiv_asw_dist(const double* diss, const int* cluster, const double* weights,
                   int n, int nclusters, double* asw_i, double* asw_w);

/**
 * Simplified version that computes only basic statistics (without HG/HGSD)
 */
void clusterqualitySimple(const double* diss, const int* cluster, const double* weights,
                         int n, double* stats, int nclusters, double* asw);

void clusterqualitySimple_dist(const double* diss, const int* cluster, const double* weights,
                              int n, double* stats, int nclusters, double* asw);

/**
 * Helper functions for Kendall tree management
 */
void resetKendallTree(KendallTree& kendall);
void finalizeKendall(KendallTree& kendall);

/**
 * Utility functions
 */
inline int getCondensedIndex(int i, int j, int n) {
    // Convert (i,j) indices to condensed array index
    // Use SciPy/R standard upper triangle ordering: for i < j
    if (i > j) std::swap(i, j);  // Ensure i < j for upper triangle
    return i * n - i * (i + 1) / 2 + j - i - 1;
}

inline double getDistanceFromCondensed(const double* diss, int i, int j, int n) {
    if (i == j) return 0.0;
    // No need to swap here since getCondensedIndex handles it
    return diss[getCondensedIndex(i, j, n)];
}
