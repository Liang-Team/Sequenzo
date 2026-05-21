"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 21/05/2026 18:25
@Desc    : Public API for sequence analysis uncertainty (timing uncertainty first).
"""
from .dist_mc_display import (
    format_dist_mc_brief,
    print_dist_mc,
    summary_dist_mc,
)
from .mc_clust import mc_clustcomp, mc_clustqual, MCClustQualResult
from .mc_corr import mc_disscorr, mc_mdscorr
from .mc_diss import (
    MCDissList,
    UDistResult,
    mc_disslist,
    mc_extract_dist,
    mc_nunique,
    mc_udist,
)
from .mc_pj import mc_pj
from .mc_seq_replicate import MCReplicateList, mc_seq_replicate
from .mc_seqdist_se import DistMCResult, mc_ratios, mc_seqdist_se
from .seqdist_mcse import seqdist_mcse
from .timing_change import ch_dur, ch_dur_indep, ch_dur_relat
from .plot_distance_uncertainty_heatmap import (
    plot_distance_uncertainty_heatmap,
)

# --- Primary user-facing names (timing uncertainty) ---
get_timing_perturbed_sequences = mc_seq_replicate
get_timing_error_distribution = mc_pj
get_distance_matrices_per_replicate = mc_disslist
get_distance_matrix_stability = mc_seqdist_se
get_distance_timing_uncertainty = seqdist_mcse
print_distance_uncertainty = print_dist_mc
summarize_distance_uncertainty = summary_dist_mc

# Secondary names (same timing-uncertainty scope)
get_distance_matrices_unique = mc_udist
extract_replicate_distance_matrix = mc_extract_dist
count_unique_replicate_sequences = mc_nunique
distance_uncertainty_ratios = mc_ratios
correlation_observed_vs_replicate_distances = mc_disscorr
correlation_mds_observed_vs_replicate = mc_mdscorr
compare_clusters_observed_vs_replicate = mc_clustcomp
cluster_quality_across_replicates = mc_clustqual

__all__ = [
    # Types
    "DistMCResult",
    "MCReplicateList",
    "MCDissList",
    "UDistResult",
    "MCClustQualResult",
    # Timing uncertainty — preferred
    "get_timing_perturbed_sequences",
    "get_timing_error_distribution",
    "get_distance_matrices_per_replicate",
    "get_distance_matrix_stability",
    "get_distance_timing_uncertainty",
    "print_distance_uncertainty",
    "summarize_distance_uncertainty",
    "plot_distance_uncertainty_heatmap",
    # Timing uncertainty — secondary
    "get_distance_matrices_unique",
    "extract_replicate_distance_matrix",
    "count_unique_replicate_sequences",
    "distance_uncertainty_ratios",
    "correlation_observed_vs_replicate_distances",
    "correlation_mds_observed_vs_replicate",
    "compare_clusters_observed_vs_replicate",
    "cluster_quality_across_replicates",
    "format_dist_mc_brief",
    # Low-level spell perturbation (advanced)
    "ch_dur",
    "ch_dur_indep",
    "ch_dur_relat",
]
