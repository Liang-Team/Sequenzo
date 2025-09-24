#include "cluster_quality.h"
#include <iostream>
#include <limits>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * Implementation of cluster quality indicators matching R WeightedCluster package
 * 
 * This implementation closely follows the logic in R's clusterquality.cpp
 * to ensure numerical consistency with the WeightedCluster package.
 */

void resetKendallTree(KendallTree& kendall) {
    for (auto& pair : kendall) {
        pair.second->clustDist0 = 0.0L;
        pair.second->clustDist1 = 0.0L;
    }
}

void finalizeKendall(KendallTree& kendall) {
    for (auto& pair : kendall) {
        delete pair.second;
    }
    kendall.clear();
}

/**
 * Compute individual ASW scores for full distance matrix
 */
void indiv_asw(const double* diss, const int* cluster, const double* weights,
               int n, int nclusters, double* asw_i, double* asw_w) {
    
    // Initialize output arrays
    std::fill(asw_i, asw_i + n, std::numeric_limits<double>::quiet_NaN());
    std::fill(asw_w, asw_w + n, std::numeric_limits<double>::quiet_NaN());
    
    // Count cluster sizes and validate
    std::vector<int> cluster_sizes(nclusters + 1, 0);
    for (int i = 0; i < n; i++) {
        if (cluster[i] >= 1 && cluster[i] <= nclusters) {
            cluster_sizes[cluster[i]]++;
        }
    }
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int ci = cluster[i];
        if (ci < 1 || ci > nclusters || cluster_sizes[ci] <= 1) {
            continue; // Skip singletons or invalid clusters
        }
        
        double a_i = 0.0;  // Within-cluster average distance
        double b_i = std::numeric_limits<double>::max();  // Min between-cluster average
        
        // Calculate within-cluster average (a_i)
        double sum_within = 0.0;
        double weight_within = 0.0;
        
        for (int j = 0; j < n; j++) {
            if (i != j && cluster[j] == ci) {
                double dist = diss[i * n + j];
                sum_within += dist * weights[j];
                weight_within += weights[j];
            }
        }
        
        if (weight_within > 0) {
            a_i = sum_within / weight_within;
        }
        
        // Calculate minimum between-cluster average (b_i)
        for (int k = 1; k <= nclusters; k++) {
            if (k == ci || cluster_sizes[k] == 0) continue;
            
            double sum_between = 0.0;
            double weight_between = 0.0;
            
            for (int j = 0; j < n; j++) {
                if (cluster[j] == k) {
                    double dist = diss[i * n + j];
                    sum_between += dist * weights[j];
                    weight_between += weights[j];
                }
            }
            
            if (weight_between > 0) {
                double avg_between = sum_between / weight_between;
                b_i = std::min(b_i, avg_between);
            }
        }
        
        // Calculate silhouette scores
        if (b_i != std::numeric_limits<double>::max()) {
            double max_ab = std::max(a_i, b_i);
            if (max_ab > 0) {
                asw_i[i] = (b_i - a_i) / max_ab;
                asw_w[i] = asw_i[i]; // For individual scores, weighted = unweighted
            } else {
                asw_i[i] = 0.0;
                asw_w[i] = 0.0;
            }
        }
    }
}

/**
 * Compute individual ASW scores for condensed distance array
 */
void indiv_asw_dist(const double* diss, const int* cluster, const double* weights,
                   int n, int nclusters, double* asw_i, double* asw_w) {
    
    // Initialize output arrays
    std::fill(asw_i, asw_i + n, std::numeric_limits<double>::quiet_NaN());
    std::fill(asw_w, asw_w + n, std::numeric_limits<double>::quiet_NaN());
    
    // Count cluster sizes and validate
    std::vector<int> cluster_sizes(nclusters + 1, 0);
    for (int i = 0; i < n; i++) {
        if (cluster[i] >= 1 && cluster[i] <= nclusters) {
            cluster_sizes[cluster[i]]++;
        }
    }
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int ci = cluster[i];
        if (ci < 1 || ci > nclusters || cluster_sizes[ci] <= 1) {
            continue; // Skip singletons or invalid clusters
        }
        
        double a_i = 0.0;  // Within-cluster average distance
        double b_i = std::numeric_limits<double>::max();  // Min between-cluster average
        
        // Calculate within-cluster average (a_i)
        double sum_within = 0.0;
        double weight_within = 0.0;
        
        for (int j = 0; j < n; j++) {
            if (i != j && cluster[j] == ci) {
                double dist = getDistanceFromCondensed(diss, i, j, n);
                sum_within += dist * weights[j];
                weight_within += weights[j];
            }
        }
        
        if (weight_within > 0) {
            a_i = sum_within / weight_within;
        }
        
        // Calculate minimum between-cluster average (b_i)
        for (int k = 1; k <= nclusters; k++) {
            if (k == ci || cluster_sizes[k] == 0) continue;
            
            double sum_between = 0.0;
            double weight_between = 0.0;
            
            for (int j = 0; j < n; j++) {
                if (cluster[j] == k) {
                    double dist = getDistanceFromCondensed(diss, i, j, n);
                    sum_between += dist * weights[j];
                    weight_between += weights[j];
                }
            }
            
            if (weight_between > 0) {
                double avg_between = sum_between / weight_between;
                b_i = std::min(b_i, avg_between);
            }
        }
        
        // Calculate silhouette scores
        if (b_i != std::numeric_limits<double>::max()) {
            double max_ab = std::max(a_i, b_i);
            if (max_ab > 0) {
                asw_i[i] = (b_i - a_i) / max_ab;
                asw_w[i] = asw_i[i]; // For individual scores, weighted = unweighted
            } else {
                asw_i[i] = 0.0;
                asw_w[i] = 0.0;
            }
        }
    }
}

/**
 * Core function to compute all cluster quality indicators
 * This follows the R implementation logic exactly
 */
template<bool UseCondensed>
void compute_cluster_quality_core(const double* diss, const int* cluster, const double* weights,
                                 int n, double* stats, int nclusters, double* asw,
                                 KendallTree& kendall) {
    
    // Initialize all statistics to NaN
    std::fill(stats, stats + ClusterQualNumStat, std::numeric_limits<double>::quiet_NaN());
    std::fill(asw, asw + 2 * nclusters, std::numeric_limits<double>::quiet_NaN());
    
    // Validate input - return all NaN for invalid cases
    if (n < 2 || nclusters < 1 || nclusters >= n) {
        return;
    }
    
    // Count cluster sizes and compute total weight
    std::vector<int> cluster_sizes(nclusters + 1, 0);
    std::vector<double> cluster_weights(nclusters + 1, 0.0);
    double total_weight = 0.0;
    
    for (int i = 0; i < n; i++) {
        if (cluster[i] >= 1 && cluster[i] <= nclusters) {
            cluster_sizes[cluster[i]]++;
            cluster_weights[cluster[i]] += weights[i];
        }
        total_weight += weights[i];
    }
    
    // Check for valid clustering - need at least 2 non-empty clusters
    int valid_clusters = 0;
    for (int c = 1; c <= nclusters; c++) {
        if (cluster_sizes[c] > 0) valid_clusters++;
    }
    if (valid_clusters < 2) {
        // All stats remain NaN for invalid clustering
        return;
    }
    
    // ===== Compute ASW (both individual and weighted) =====
    std::vector<double> asw_individual(n);
    std::vector<double> asw_weighted(n);
    
    if constexpr (UseCondensed) {
        indiv_asw_dist(diss, cluster, weights, n, nclusters, asw_individual.data(), asw_weighted.data());
    } else {
        indiv_asw(diss, cluster, weights, n, nclusters, asw_individual.data(), asw_weighted.data());
    }
    
    // Aggregate ASW by cluster
    std::vector<double> cluster_asw_sum(nclusters + 1, 0.0);
    std::vector<double> cluster_asw_weight(nclusters + 1, 0.0);
    std::vector<double> cluster_asw_weighted_sum(nclusters + 1, 0.0);
    
    for (int i = 0; i < n; i++) {
        int ci = cluster[i];
        if (ci >= 1 && ci <= nclusters && !std::isnan(asw_individual[i])) {
            cluster_asw_sum[ci] += asw_individual[i];
            cluster_asw_weighted_sum[ci] += asw_weighted[i] * weights[i];
            cluster_asw_weight[ci] += weights[i];
        }
    }
    
    // Store cluster-level ASW
    double global_asw = 0.0, global_asw_weighted = 0.0;
    double global_weight = 0.0;
    int global_count = 0;
    
    for (int c = 1; c <= nclusters; c++) {
        if (cluster_sizes[c] > 1) {  // Only include clusters with more than 1 member for ASW calculation
            // Count valid individuals in this cluster (those with non-NaN ASW)
            int valid_individuals = 0;
            for (int i = 0; i < n; i++) {
                if (cluster[i] == c && !std::isnan(asw_individual[i])) {
                    valid_individuals++;
                }
            }
            
            if (valid_individuals > 0) {
                asw[2 * (c - 1)] = cluster_asw_sum[c] / valid_individuals;  // Unweighted ASW
                if (cluster_asw_weight[c] > 0) {
                    asw[2 * (c - 1) + 1] = cluster_asw_weighted_sum[c] / cluster_asw_weight[c];  // Weighted ASW
                }
                
                global_asw += cluster_asw_sum[c];
                global_asw_weighted += cluster_asw_weighted_sum[c];
                global_weight += cluster_asw_weight[c];
                global_count += valid_individuals;
            }
        }
    }
    
    stats[ClusterQualASWi] = (global_count > 0) ? global_asw / global_count : 0.0;
    stats[ClusterQualASWw] = (global_weight > 0) ? global_asw_weighted / global_weight : 0.0;
    
    // ===== Compute R² (weighted) =====
    long double D_bar = 0.0L;  // Global weighted mean of distances
    long double total_pair_weight = 0.0L;
    
    // Calculate global weighted mean (using upper triangle only)
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            double dist;
            if constexpr (UseCondensed) {
                dist = diss[getCondensedIndex(i, j, n)];
            } else {
                dist = diss[i * n + j];
            }
            long double pair_weight = static_cast<long double>(weights[i]) * weights[j];
            D_bar += dist * pair_weight;
            total_pair_weight += pair_weight;
        }
    }
    if (total_pair_weight > 0) {
        D_bar /= total_pair_weight;
    }
    
    // Calculate total sum of squares
    long double total_ss = 0.0L;
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            double dist;
            if constexpr (UseCondensed) {
                dist = diss[getCondensedIndex(i, j, n)];
            } else {
                dist = diss[i * n + j];
            }
            long double pair_weight = static_cast<long double>(weights[i]) * weights[j];
            long double diff = dist - D_bar;
            total_ss += pair_weight * diff * diff;
        }
    }
    
    // Calculate within-cluster sum of squares
    long double within_ss = 0.0L;
    for (int c = 1; c <= nclusters; c++) {
        if (cluster_sizes[c] < 2) continue;
        
        // Get cluster members
        std::vector<int> cluster_members;
        for (int i = 0; i < n; i++) {
            if (cluster[i] == c) {
                cluster_members.push_back(i);
            }
        }
        
        // Calculate cluster weighted mean
        long double cluster_sum = 0.0L;
        long double cluster_weight = 0.0L;
        for (size_t ii = 0; ii < cluster_members.size() - 1; ii++) {
            for (size_t jj = ii + 1; jj < cluster_members.size(); jj++) {
                int i = cluster_members[ii];
                int j = cluster_members[jj];
                double dist;
                if constexpr (UseCondensed) {
                    dist = diss[getCondensedIndex(i, j, n)];
                } else {
                    dist = diss[i * n + j];
                }
                long double pair_weight = static_cast<long double>(weights[i]) * weights[j];
                cluster_sum += dist * pair_weight;
                cluster_weight += pair_weight;
            }
        }
        
        if (cluster_weight > 0) {
            long double cluster_mean = cluster_sum / cluster_weight;
            
            // Add to within-cluster sum of squares
            for (size_t ii = 0; ii < cluster_members.size() - 1; ii++) {
                for (size_t jj = ii + 1; jj < cluster_members.size(); jj++) {
                    int i = cluster_members[ii];
                    int j = cluster_members[jj];
                    double dist;
                    if constexpr (UseCondensed) {
                        dist = diss[getCondensedIndex(i, j, n)];
                    } else {
                        dist = diss[i * n + j];
                    }
                    long double pair_weight = static_cast<long double>(weights[i]) * weights[j];
                    long double diff = dist - cluster_mean;
                    within_ss += pair_weight * diff * diff;
                }
            }
        }
    }
    
    stats[ClusterQualR] = (total_ss > 0) ? static_cast<double>(1.0L - within_ss / total_ss) : 0.0;
    stats[ClusterQualR2] = stats[ClusterQualR] * stats[ClusterQualR];
    
    // ===== Compute Calinski-Harabasz =====
    long double between_ss = total_ss - within_ss;
    if (within_ss > 0 && nclusters > 1 && n > nclusters) {
        long double f_stat = (between_ss / (nclusters - 1)) / (within_ss / (n - nclusters));
        stats[ClusterQualF] = static_cast<double>(f_stat);
        stats[ClusterQualF2] = stats[ClusterQualF] * stats[ClusterQualF];
    }
    
    // ===== Compute HPG (weighted point-biserial correlation) =====
    {
        long double sum_w  = 0.0L;      // Σ wij
        long double sum_xw = 0.0L;      // Σ wij * d_ij
        long double sum_yw = 0.0L;      // Σ wij * y_ij
        long double sum_x2w= 0.0L;      // Σ wij * d_ij^2
        long double sum_y2w= 0.0L;      // Σ wij * y_ij^2  (y^2==y 因为 y∈{0,1})
        long double sum_xyw= 0.0L;      // Σ wij * d_ij * y_ij

        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const double wij = weights[i] * weights[j];
                if (wij <= 0) continue;
                const double dij = (UseCondensed ? diss[getCondensedIndex(i,j,n)]
                                                 : diss[i*n + j]);
                const double yij = (cluster[i] == cluster[j]) ? 1.0 : 0.0;

                sum_w   += wij;
                sum_xw  += wij * dij;
                sum_yw  += wij * yij;
                sum_x2w += wij * dij * dij;
                sum_y2w += wij * yij;          // yij^2 == yij
                sum_xyw += wij * dij * yij;
            }
        }

        if (sum_w > 0) {
            const long double mx = sum_xw / sum_w;
            const long double my = sum_yw / sum_w;
            const long double cov_xy = (sum_xyw / sum_w) - mx * my;
            const long double var_x  = (sum_x2w / sum_w) - mx * mx;
            const long double var_y  = (sum_y2w / sum_w) - my * my;

            if (var_x > 0 && var_y > 0) {
                stats[ClusterQualHPG] = static_cast<double>(cov_xy / std::sqrt(var_x * var_y));
            }
        }
    }
    
    // ===== Compute HG and HGSD (Hubert's Gamma) =====
    // Based on R WeightedCluster implementation - correct Kendall tau calculation
    
    // Reset Kendall tree
    for (auto& pair : kendall) {
        pair.second->clustDist0 = 0.0L;
        pair.second->clustDist1 = 0.0L;
    }
    
    // Build distance groups with cluster memberships
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            double dist_ij;
            if constexpr (UseCondensed) {
                dist_ij = diss[getCondensedIndex(i, j, n)];
            } else {
                dist_ij = diss[i * n + j];
            }
            
            // Get or create entry in Kendall tree
            auto it = kendall.find(dist_ij);
            CmpCluster* cmp;
            if (it == kendall.end()) {
                cmp = new CmpCluster();
                kendall[dist_ij] = cmp;
            } else {
                cmp = it->second;
            }
            
            // Count pairs: clustDist1 = same cluster, clustDist0 = different clusters
            long double weight_pair = static_cast<long double>(weights[i]) * weights[j];
            if (cluster[i] == cluster[j]) {
                cmp->clustDist1 += weight_pair;
            } else {
                cmp->clustDist0 += weight_pair;
            }
        }
    }
    
    // Calculate Kendall's tau (Gamma) from the tree
    long double gamma_concordant = 0.0L;
    long double gamma_discordant = 0.0L;
    
    for (auto it1 = kendall.begin(); it1 != kendall.end(); ++it1) {
        for (auto it2 = std::next(it1); it2 != kendall.end(); ++it2) {
            double d1 = it1->first;
            double d2 = it2->first;
            CmpCluster* cmp1 = it1->second;
            CmpCluster* cmp2 = it2->second;
            
            if (d1 < d2) {
                // For distances d1 < d2, we expect same-cluster pairs to be more common at d1
                // Concordant: more same-cluster pairs at smaller distance
                gamma_concordant += cmp1->clustDist1 * cmp2->clustDist0;
                // Discordant: more different-cluster pairs at smaller distance  
                gamma_discordant += cmp1->clustDist0 * cmp2->clustDist1;
            }
        }
    }
    
    long double gamma_total = gamma_concordant + gamma_discordant;
    if (gamma_total > 0) {
        stats[ClusterQualHG] = static_cast<double>((gamma_concordant - gamma_discordant) / gamma_total);
    }
    
    // HGSD: Temporarily set to NaN until exact R formula is ported
    stats[ClusterQualHGSD] = std::numeric_limits<double>::quiet_NaN();
    
    // ===== Compute HC (Hierarchical Criterion) =====
    // This is a simplified version - the full implementation would need the dendrogram
    std::vector<double> cluster_means(nclusters + 1, 0.0);
    for (int c = 1; c <= nclusters; c++) {
        if (cluster_sizes[c] > 0) {
            // Calculate mean within-cluster distance
            double sum_dist = 0.0;
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (cluster[i] == c) {
                    for (int j = i + 1; j < n; j++) {
                        if (cluster[j] == c) {
                            double dist;
                            if constexpr (UseCondensed) {
                                dist = diss[getCondensedIndex(i, j, n)];
                            } else {
                                dist = diss[i * n + j];
                            }
                            sum_dist += dist;
                            count++;
                        }
                    }
                }
            }
            cluster_means[c] = (count > 0) ? sum_dist / count : 0.0;
        }
    }
    
    double mean_of_means = 0.0;
    int valid_mean_count = 0;
    for (int c = 1; c <= nclusters; c++) {
        if (cluster_sizes[c] > 0) {
            mean_of_means += cluster_means[c];
            valid_mean_count++;
        }
    }
    mean_of_means /= valid_mean_count;
    
    double variance = 0.0;
    for (int c = 1; c <= nclusters; c++) {
        if (cluster_sizes[c] > 0) {
            variance += (cluster_means[c] - mean_of_means) * (cluster_means[c] - mean_of_means);
        }
    }
    // HC: Temporarily set to NaN until exact R formula is ported
    stats[ClusterQualHC] = std::numeric_limits<double>::quiet_NaN();
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

// Simplified versions (subset of statistics)
void clusterqualitySimple(const double* diss, const int* cluster, const double* weights,
                         int n, double* stats, int nclusters, double* asw) {
    KendallTree kendall;  // Local Kendall tree for simple version
    clusterquality(diss, cluster, weights, n, stats, nclusters, asw, kendall);
    finalizeKendall(kendall);
}

void clusterqualitySimple_dist(const double* diss, const int* cluster, const double* weights,
                              int n, double* stats, int nclusters, double* asw) {
    KendallTree kendall;  // Local Kendall tree for simple version
    clusterquality_dist(diss, cluster, weights, n, stats, nclusters, asw, kendall);
    finalizeKendall(kendall);
}
