#include "cluster_quality.h"
#include <iostream>
#include <limits>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * Implementation matching R WeightedCluster exactly
 * Based on clusterqualitybody.cpp from R package
 */

void resetKendallTree(KendallTree& kendall) {
    for (auto& pair : kendall) {
        pair.second->clustDist0 = 0.0;
        pair.second->clustDist1 = 0.0;
    }
}

void finalizeKendall(KendallTree& kendall) {
    for (auto& pair : kendall) {
        delete pair.second;
    }
    kendall.clear();
}

/**
 * Core function exactly matching R WeightedCluster implementation
 */
template<bool UseCondensed>
void compute_cluster_quality_core(const double* diss, const int* cluster, const double* weights,
                                 int n, double* stats, int nclusters, double* asw,
                                 KendallTree& kendall) {
    
    // Initialize all statistics to NaN
    std::fill(stats, stats + ClusterQualNumStat, std::numeric_limits<double>::quiet_NaN());
    std::fill(asw, asw + 2 * nclusters, std::numeric_limits<double>::quiet_NaN());
    
    // Variables following R implementation exactly - use double like R
    double totweights = 0.0, wxy = 0.0, wx = 0.0, wy = 0.0, wx2 = 0.0;
    double ww, xx, covxy, covx, covy, pearson, xb, yb, xw, xxw;
    int ij = 0;
    
    // Allocate arrays like R version (0-based indexing)
    std::vector<double> errors(nclusters, 0.0);
    std::vector<double> errors2(nclusters, 0.0);
    std::vector<double> sizes(nclusters, 0.0);
    
    // Initialize ASW arrays (output)
    for (int i = 0; i < nclusters; i++) {
        asw[i] = 0.0;
        asw[i + nclusters] = 0.0;
    }
    
    // Initialize Kendall tree with zero distance node (like R)
    CmpCluster* ZeroDist;
    auto it_zero = kendall.find(0.0);
    if (it_zero != kendall.end()) {
        ZeroDist = it_zero->second;
    } else {
        ZeroDist = new CmpCluster();
        kendall[0.0] = ZeroDist;
    }
    
    // Main computation loop following R version exactly
    if constexpr (UseCondensed) {
        ij = -n;  // Condensed version initialization
    }
    
    for (int i = 0; i < n; i++) {
        int iclustIndex = cluster[i] - 1;  // Convert to 0-based for array access
        if (iclustIndex >= 0 && iclustIndex < nclusters) {
            sizes[iclustIndex] += weights[i];
        }
        
        if constexpr (!UseCondensed) {
            ij = i * n;  // Full matrix version
        } else {
            ij += n - i - 1;  // Condensed version offset
        }
        
        if (weights[i] > 0) {
            // Diagonal term (distance to self = 0)
            ww = weights[i] * weights[i];
            wy += ww;
            ZeroDist->clustDist0 += ww;
            totweights += ww;
            
            for (int j = i + 1; j < n; j++) {
                if (weights[j] > 0) {
                    ww = 2.0 * weights[i] * weights[j];  // Factor of 2 like R
                    
                    if constexpr (UseCondensed) {
                        // Use explicit condensed indexing to avoid stride/layout issues
                        xx = diss[getCondensedIndex(i, j, n)];
                    } else {
                        // Full square matrix (row-major) indexing
                        xx = diss[ij + j];
                    }
                    
                    // Find or create Kendall tree node
                    auto it = kendall.find(xx);
                    CmpCluster* cmpclust;
                    if (it != kendall.end()) {
                        cmpclust = it->second;
                    } else {
                        cmpclust = new CmpCluster();
                        kendall[xx] = cmpclust;
                    }
                    
                    xw = ww * xx;
                    xxw = xw * xx;
                    wx += xw;
                    wx2 += xxw;
                    
                    if (cluster[i] == cluster[j]) {
                        // Same cluster
                        if (iclustIndex >= 0 && iclustIndex < nclusters) {
                            errors[iclustIndex] += xw;
                            errors2[iclustIndex] += xxw;  // Add errors2 calculation like R
                        }
                        wxy += xw;
                        wy += ww;
                        cmpclust->clustDist0 += ww;
                    } else {
                        // Different clusters
                        cmpclust->clustDist1 += ww;
                    }
                    
                    totweights += ww;
                }
            }
        }
    }
    
    // Calculate Pearson correlation (HPG) exactly like R
    if (totweights > 0) {
        xb = wx / totweights;
        yb = wy / totweights;
        covx = wx2 / totweights - xb * xb;
        covy = wy / totweights - yb * yb;
        covxy = wxy / totweights - yb * xb;
        
        // Debug: Print intermediate values
        #ifdef DEBUG_PBC
        std::cout << "DEBUG PBC: totweights=" << totweights << ", wx=" << wx << ", wy=" << wy << ", wxy=" << wxy << ", wx2=" << wx2 << std::endl;
        std::cout << "DEBUG PBC: xb=" << xb << ", yb=" << yb << std::endl;
        std::cout << "DEBUG PBC: covx=" << covx << ", covy=" << covy << ", covxy=" << covxy << std::endl;
        #endif
        
        if (covx > 0 && covy > 0) {
            pearson = covxy / std::sqrt(covx * covy);
            double pbc_value = -1.0 * static_cast<double>(pearson);  // Apply negative to get positive PBC
            stats[ClusterQualHPG] = pbc_value;
            
            // Debug: Print final calculation
            #ifdef DEBUG_PBC
            std::cout << "DEBUG PBC: pearson=" << pearson << ", pbc_value=" << pbc_value << std::endl;
            #endif
        }
    }
    
    // Compute Kendall statistics (HG, HGSD, HC) exactly like R
    double nc = 0.0, nd = 0.0, currentclustdist0 = 0.0, currentclustdist1 = 0.0;
    double totdist0 = wy, totdist1 = totweights - wy, ntiesdist = 0.0;
    double Smin = 0.0, wSmin = wy, Smax = 0.0, wSmax = totdist1, currentww = 0.0;
    
    for (auto it = kendall.begin(); it != kendall.end(); ++it) {
        CmpCluster* cmpclust = it->second;
        ww = cmpclust->clustDist1 + cmpclust->clustDist0;
        
        if (ww > 0) {
            // Smin calculation
            if (currentww <= wSmin) {
                if (currentww + ww > wSmin) {
                    Smin += (wSmin - currentww) * it->first;
                } else {
                    Smin += ww * it->first;
                }
            }
            currentww += ww;
            
            // Smax calculation  
            if (currentww > wSmax) {
                if (currentww - ww < wSmax) {
                    Smax += (currentww - wSmax) * it->first;
                } else {
                    Smax += ww * it->first;
                }
            }
            
            // Count ties
            ntiesdist += cmpclust->clustDist1 * cmpclust->clustDist0;
            
            // Concordant and discordant pairs - exactly like R
            nc += cmpclust->clustDist1 * currentclustdist0;  // Bottom of table
            nd += cmpclust->clustDist0 * currentclustdist1;
            
            // Update running totals
            currentclustdist0 += cmpclust->clustDist0;
            currentclustdist1 += cmpclust->clustDist1;
            
            // Top of table
            nc += cmpclust->clustDist0 * (totdist1 - currentclustdist1);
            nd += cmpclust->clustDist1 * (totdist0 - currentclustdist0);
        }
    }
    
    // Compute final Kendall statistics (guard divisions to avoid NaN while matching R behavior)
    double denom_hg = (nc + nd);
    if (denom_hg > 0) {
        stats[ClusterQualHG] = static_cast<double>((nc - nd) / denom_hg);  // Gamma
    }

    // HGSD (Somers' D)
    double denom_hgsd = (nc + nd + ntiesdist);
    if (denom_hgsd > 0) {
        stats[ClusterQualHGSD] = (nc - nd) / denom_hgsd;
    } else {
        stats[ClusterQualHGSD] = 0.0; // avoid NaN in degenerate cases
    }

    // HC (Hierarchical Criterion)
    double denom_hc = (Smax - Smin);
    if (denom_hc > 0) {
        stats[ClusterQualHC] = (wxy - Smin) / denom_hc;
    } else {
        stats[ClusterQualHC] = 0.0; // avoid NaN when Smax == Smin
    }
    
    
    // Compute F and R statistics exactly like R
    double SSres = 0.0;
    double SS2res = 0.0;
    double total_cluster_weights = 0.0;
    
    for (int i = 0; i < nclusters; i++) {
        if (sizes[i] > 0) {
            SSres += errors[i] / sizes[i];
            SS2res += errors2[i] / sizes[i];
            total_cluster_weights += sizes[i];
        }
    }
    
    if (total_cluster_weights > 0) {
        double SSexpl = wx / total_cluster_weights - SSres;
        double SS2expl = wx2 / total_cluster_weights - SS2res;
        double dncluster = static_cast<double>(nclusters);
        
        if (total_cluster_weights > dncluster && SSres > 0) {
            stats[ClusterQualF] = (SSexpl / (dncluster - 1.0)) / (SSres / (total_cluster_weights - dncluster));
            stats[ClusterQualR] = SSexpl / (SSres + SSexpl);
            // F2 and R2 should be based on SS2, not squares of F and R
            stats[ClusterQualF2] = (SS2expl / (dncluster - 1.0)) / (SS2res / (total_cluster_weights - dncluster));
            stats[ClusterQualR2] = SS2expl / (SS2res + SS2expl);
        }
    }
    
    // Compute ASW exactly like R version
    double asw_i = 0.0;
    double asw_w = 0.0;
    
    // Reset ASW arrays
    for (int j = 0; j < nclusters; j++) {
        asw[j] = 0.0;
        asw[j + nclusters] = 0.0;
    }
    
    for (int i = 0; i < n; i++) {
        if (weights[i] > 0) {
            int iclustIndex = cluster[i] - 1;  // Convert to 0-based
            if (iclustIndex < 0 || iclustIndex >= nclusters) continue;
            
            double aik = 0.0;
            std::vector<double> othergroups(nclusters, 0.0);
            
            // Calculate distances to all other points
            if constexpr (!UseCondensed) {
                ij = i * n;
                for (int j = 0; j < n; j++) {
                    if (i == j) continue;
                    int jclustIndex = cluster[j] - 1;
                    if (jclustIndex < 0 || jclustIndex >= nclusters) continue;
                    
                    if (iclustIndex == jclustIndex) {
                        aik += weights[j] * diss[ij + j];
                    } else {
                        othergroups[jclustIndex] += weights[j] * diss[ij + j];
                    }
                }
            } else {
                // Condensed version
                for (int j = 0; j < n; j++) {
                    if (i == j) continue;
                    int jclustIndex = cluster[j] - 1;
                    if (jclustIndex < 0 || jclustIndex >= nclusters) continue;
                    
                    double dist_val = (i < j) ? diss[getCondensedIndex(i, j, n)] : diss[getCondensedIndex(j, i, n)];
                    
                    if (iclustIndex == jclustIndex) {
                        aik += weights[j] * dist_val;
                    } else {
                        othergroups[jclustIndex] += weights[j] * dist_val;
                    }
                }
            }
            
            // Find minimum average distance to other clusters
            double bik = std::numeric_limits<double>::max();
            for (int j = 0; j < nclusters; j++) {
                if (j != iclustIndex && sizes[j] > 0) {
                    double avg_dist = othergroups[j] / sizes[j];
                    if (bik >= avg_dist) {
                        bik = avg_dist;
                    }
                }
            }
            
            // Calculate ASW values like R
            double aik_w = aik / sizes[iclustIndex];  // Weighted version
            if (sizes[iclustIndex] <= 1.0) {
                aik = 0.0;  // Avoid division by zero for singletons
            } else {
                aik /= (sizes[iclustIndex] - 1.0);  // Unweighted version
            }
            
            if (bik != std::numeric_limits<double>::max()) {
                double sik_i = weights[i] * ((bik - aik) / std::max(aik, bik));
                double sik_w = weights[i] * ((bik - aik_w) / std::max(aik_w, bik));
                
                asw[iclustIndex] += sik_i;
                asw[iclustIndex + nclusters] += sik_w;
                asw_i += sik_i;
                asw_w += sik_w;
            }
        }
    }
    
    // Normalize cluster ASW by cluster sizes
    for (int j = 0; j < nclusters; j++) {
        if (sizes[j] > 0) {
            asw[j] /= sizes[j];
            asw[j + nclusters] /= sizes[j];
        }
    }
    
    if (total_cluster_weights > 0) {
        stats[ClusterQualASWi] = asw_i / total_cluster_weights;
        stats[ClusterQualASWw] = asw_w / total_cluster_weights;
    }
}

// Template instantiations
void clusterquality(const double* diss, const int* cluster, const double* weights,
                   int n, double* stats, int nclusters, double* asw, 
                   KendallTree& kendall) {
    compute_cluster_quality_core<false>(diss, cluster, weights, n, stats, nclusters, asw, kendall);
}

void clusterquality_dist(const double* diss, const int* cluster, const double* weights,
                        int n, double* stats, int nclusters, double* asw,
                        KendallTree& kendall) {
    compute_cluster_quality_core<true>(diss, cluster, weights, n, stats, nclusters, asw, kendall);
}

// Individual ASW functions (simplified, calling the main function)
void indiv_asw(const double* diss, const int* cluster, const double* weights,
               int n, int nclusters, double* asw_i, double* asw_w) {
    
    std::fill(asw_i, asw_i + n, std::numeric_limits<double>::quiet_NaN());
    std::fill(asw_w, asw_w + n, std::numeric_limits<double>::quiet_NaN());
    
    // For individual ASW, we can use simplified computation
    std::vector<double> sizes(nclusters, 0.0);
    for (int i = 0; i < n; i++) {
        int clustIndex = cluster[i] - 1;
        if (clustIndex >= 0 && clustIndex < nclusters) {
            sizes[clustIndex] += weights[i];
        }
    }
    
    for (int i = 0; i < n; i++) {
        int iclustIndex = cluster[i] - 1;
        if (iclustIndex < 0 || iclustIndex >= nclusters || sizes[iclustIndex] <= 1.0) {
            continue;
        }
        
        double aik = 0.0, aik_w = 0.0;
        std::vector<double> othergroups(nclusters, 0.0);
        
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            int jclustIndex = cluster[j] - 1;
            if (jclustIndex < 0 || jclustIndex >= nclusters) continue;
            
            double dist = diss[i * n + j];
            if (iclustIndex == jclustIndex) {
                aik += weights[j] * dist;
            } else {
                othergroups[jclustIndex] += weights[j] * dist;
            }
        }
        
        double bik = std::numeric_limits<double>::max();
        for (int j = 0; j < nclusters; j++) {
            if (j != iclustIndex && sizes[j] > 0) {
                double avg_dist = othergroups[j] / sizes[j];
                if (bik >= avg_dist) {
                    bik = avg_dist;
                }
            }
        }
        
        aik_w = aik / sizes[iclustIndex];
        aik /= (sizes[iclustIndex] - 1.0);
        
        if (bik != std::numeric_limits<double>::max()) {
            asw_i[i] = (bik - aik) / std::max(aik, bik);
            asw_w[i] = (bik - aik_w) / std::max(aik_w, bik);
        }
    }
}

void indiv_asw_dist(const double* diss, const int* cluster, const double* weights,
                   int n, int nclusters, double* asw_i, double* asw_w) {
    
    std::fill(asw_i, asw_i + n, std::numeric_limits<double>::quiet_NaN());
    std::fill(asw_w, asw_w + n, std::numeric_limits<double>::quiet_NaN());
    
    // For condensed version
    std::vector<double> sizes(nclusters, 0.0);
    for (int i = 0; i < n; i++) {
        int clustIndex = cluster[i] - 1;
        if (clustIndex >= 0 && clustIndex < nclusters) {
            sizes[clustIndex] += weights[i];
        }
    }
    
    for (int i = 0; i < n; i++) {
        int iclustIndex = cluster[i] - 1;
        if (iclustIndex < 0 || iclustIndex >= nclusters || sizes[iclustIndex] <= 1.0) {
            continue;
        }
        
        double aik = 0.0, aik_w = 0.0;
        std::vector<double> othergroups(nclusters, 0.0);
        
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            int jclustIndex = cluster[j] - 1;
            if (jclustIndex < 0 || jclustIndex >= nclusters) continue;
            
            double dist = getDistanceFromCondensed(diss, i, j, n);
            if (iclustIndex == jclustIndex) {
                aik += weights[j] * dist;
            } else {
                othergroups[jclustIndex] += weights[j] * dist;
            }
        }
        
        double bik = std::numeric_limits<double>::max();
        for (int j = 0; j < nclusters; j++) {
            if (j != iclustIndex && sizes[j] > 0) {
                double avg_dist = othergroups[j] / sizes[j];
                if (bik >= avg_dist) {
                    bik = avg_dist;
                }
            }
        }
        
        aik_w = aik / sizes[iclustIndex];
        aik /= (sizes[iclustIndex] - 1.0);
        
        if (bik != std::numeric_limits<double>::max()) {
            asw_i[i] = (bik - aik) / std::max(aik, bik);
            asw_w[i] = (bik - aik_w) / std::max(aik_w, bik);
        }
    }
}

// Simplified versions
void clusterqualitySimple(const double* diss, const int* cluster, const double* weights,
                         int n, double* stats, int nclusters, double* asw) {
    KendallTree kendall;
    clusterquality(diss, cluster, weights, n, stats, nclusters, asw, kendall);
    finalizeKendall(kendall);
}

void clusterqualitySimple_dist(const double* diss, const int* cluster, const double* weights,
                              int n, double* stats, int nclusters, double* asw) {
    KendallTree kendall;
    clusterquality_dist(diss, cluster, weights, n, stats, nclusters, asw, kendall);
    finalizeKendall(kendall);
}
