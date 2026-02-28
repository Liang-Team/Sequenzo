"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 11/02/2025 16:41
@Desc    : 
"""
# Eager load only lightweight datasets API for fast startup
from .datasets import load_dataset, list_datasets

# Lazy-load everything else on first access
import importlib
from typing import Any

# Mapping: attribute_name -> (module_path, attribute_name)
# Uses full module path for reliable import_module
_LAZY: dict[str, tuple[str, str]] = {
    # data_preprocessing
    "helpers": ("sequenzo.data_preprocessing", "helpers"),
    "clean_time_columns_auto": ("sequenzo.data_preprocessing.helpers", "clean_time_columns_auto"),
    "assign_unique_ids": ("sequenzo.data_preprocessing.helpers", "assign_unique_ids"),
    "wide_to_long_format_data": ("sequenzo.data_preprocessing.helpers", "wide_to_long_format_data"),
    "long_to_wide_format_data": ("sequenzo.data_preprocessing.helpers", "long_to_wide_format_data"),
    "summarize_missing_values": ("sequenzo.data_preprocessing.helpers", "summarize_missing_values"),
    "replace_cluster_id_by_labels": ("sequenzo.data_preprocessing.helpers", "replace_cluster_id_by_labels"),
    # define_sequence_data
    "SequenceData": ("sequenzo.define_sequence_data", "SequenceData"),
    # visualization
    "plot_sequence_index": ("sequenzo.visualization", "plot_sequence_index"),
    "plot_most_frequent_sequences": ("sequenzo.visualization", "plot_most_frequent_sequences"),
    "plot_single_medoid": ("sequenzo.visualization", "plot_single_medoid"),
    "plot_state_distribution": ("sequenzo.visualization", "plot_state_distribution"),
    "plot_modal_state": ("sequenzo.visualization", "plot_modal_state"),
    "plot_relative_frequency": ("sequenzo.visualization", "plot_relative_frequency"),
    "plot_mean_time": ("sequenzo.visualization", "plot_mean_time"),
    "plot_transition_matrix": ("sequenzo.visualization", "plot_transition_matrix"),
    # dissimilarity_measures
    "get_distance_matrix": ("sequenzo.dissimilarity_measures.get_distance_matrix", "get_distance_matrix"),
    "get_substitution_cost_matrix": ("sequenzo.dissimilarity_measures.get_substitution_cost_matrix", "get_substitution_cost_matrix"),
    "get_LCP_length_for_2_seq": ("sequenzo.dissimilarity_measures.utils.get_LCP_length_for_2_seq", "get_LCP_length_for_2_seq"),
    # clustering (triggers OpenMP setup on first use)
    "Cluster": ("sequenzo.clustering", "Cluster"),
    "ClusterResults": ("sequenzo.clustering", "ClusterResults"),
    "ClusterQuality": ("sequenzo.clustering", "ClusterQuality"),
    "KMedoids": ("sequenzo.clustering.KMedoids", "KMedoids"),
    "clara": ("sequenzo.big_data.clara.clara", "clara"),
    "plot_scores_from_dataframe": ("sequenzo.big_data.clara.visualization", "plot_scores_from_dataframe"),
    # multidomain
    "create_idcd_sequence_from_csvs": ("sequenzo.multidomain", "create_idcd_sequence_from_csvs"),
    "compute_cat_distance_matrix": ("sequenzo.multidomain", "compute_cat_distance_matrix"),
    "compute_dat_distance_matrix": ("sequenzo.multidomain", "compute_dat_distance_matrix"),
    "get_interactive_combined_typology": ("sequenzo.multidomain", "get_interactive_combined_typology"),
    "merge_sparse_combt_types": ("sequenzo.multidomain", "merge_sparse_combt_types"),
    "get_association_between_domains": ("sequenzo.multidomain", "get_association_between_domains"),
    "linked_polyadic_sequence_analysis": ("sequenzo.multidomain", "linked_polyadic_sequence_analysis"),
    # prefix_tree
    "build_prefix_tree": ("sequenzo.prefix_tree", "build_prefix_tree"),
    "compute_prefix_count": ("sequenzo.prefix_tree", "compute_prefix_count"),
    "IndividualDivergence": ("sequenzo.prefix_tree", "IndividualDivergence"),
    "extract_sequences": ("sequenzo.prefix_tree", "extract_sequences"),
    "get_state_space": ("sequenzo.prefix_tree", "get_state_space"),
    "compute_branching_factor": ("sequenzo.prefix_tree", "compute_branching_factor"),
    "compute_js_divergence": ("sequenzo.prefix_tree", "compute_js_divergence"),
    "convert_to_prefix_tree_data": ("sequenzo.prefix_tree", "convert_to_prefix_tree_data"),
    "SpellPrefixTree": ("sequenzo.prefix_tree", "SpellPrefixTree"),
    "build_spell_prefix_tree": ("sequenzo.prefix_tree", "build_spell_prefix_tree"),
    "compute_js_divergence_spell": ("sequenzo.prefix_tree", "compute_js_divergence_spell"),
    "convert_seqdata_to_spells": ("sequenzo.prefix_tree", "convert_seqdata_to_spells"),
    "SpellIndividualDivergence": ("sequenzo.prefix_tree", "SpellIndividualDivergence"),
    # suffix_tree (these override prefix_tree names; __all__ order puts suffix last)
    "plot_system_indicators": ("sequenzo.suffix_tree", "plot_system_indicators"),
    "plot_system_indicators_multiple_comparison": ("sequenzo.suffix_tree", "plot_system_indicators_multiple_comparison"),
    "plot_prefix_rarity_distribution": ("sequenzo.prefix_tree", "plot_prefix_rarity_distribution"),
    "plot_individual_indicators_correlation": ("sequenzo.prefix_tree", "plot_individual_indicators_correlation"),
    "build_suffix_tree": ("sequenzo.suffix_tree", "build_suffix_tree"),
    "get_depth_stats": ("sequenzo.suffix_tree", "get_depth_stats"),
    "compute_suffix_count": ("sequenzo.suffix_tree", "compute_suffix_count"),
    "compute_merging_factor": ("sequenzo.suffix_tree", "compute_merging_factor"),
    "compute_js_convergence": ("sequenzo.suffix_tree", "compute_js_convergence"),
    "IndividualConvergence": ("sequenzo.suffix_tree", "IndividualConvergence"),
    "convert_to_suffix_tree_data": ("sequenzo.suffix_tree", "convert_to_suffix_tree_data"),
    "plot_suffix_rarity_distribution": ("sequenzo.suffix_tree", "plot_suffix_rarity_distribution"),
    "SpellSuffixTree": ("sequenzo.suffix_tree", "SpellSuffixTree"),
    "build_spell_suffix_tree": ("sequenzo.suffix_tree", "build_spell_suffix_tree"),
    "compute_js_convergence_spell": ("sequenzo.suffix_tree", "compute_js_convergence_spell"),
    "SpellIndividualConvergence": ("sequenzo.suffix_tree", "SpellIndividualConvergence"),
    # sequence_characteristics
    "get_subsequences_in_single_sequence": ("sequenzo.sequence_characteristics", "get_subsequences_in_single_sequence"),
    "get_subsequences_all_sequences": ("sequenzo.sequence_characteristics", "get_subsequences_all_sequences"),
    "get_number_of_transitions": ("sequenzo.sequence_characteristics", "get_number_of_transitions"),
    "get_turbulence": ("sequenzo.sequence_characteristics", "get_turbulence"),
    "get_complexity_index": ("sequenzo.sequence_characteristics", "get_complexity_index"),
    "get_within_sequence_entropy": ("sequenzo.sequence_characteristics", "get_within_sequence_entropy"),
    "get_spell_duration_variance": ("sequenzo.sequence_characteristics", "get_spell_duration_variance"),
    "get_state_freq_and_entropy_per_seq": ("sequenzo.sequence_characteristics", "get_state_freq_and_entropy_per_seq"),
    "get_cross_sectional_entropy": ("sequenzo.sequence_characteristics", "get_cross_sectional_entropy"),
    "plot_cross_sectional_characteristics": ("sequenzo.sequence_characteristics", "plot_cross_sectional_characteristics"),
    "plot_longitudinal_characteristics": ("sequenzo.sequence_characteristics", "plot_longitudinal_characteristics"),
    # utils
    "weighted_mean": ("sequenzo.utils", "weighted_mean"),
    "weighted_variance": ("sequenzo.utils", "weighted_variance"),
    "weighted_five_number_summary": ("sequenzo.utils", "weighted_five_number_summary"),
    # compare_differences
    "compare_groups_across_positions": ("sequenzo.compare_differences", "compare_groups_across_positions"),
    "plot_group_differences_across_positions": ("sequenzo.compare_differences", "plot_group_differences_across_positions"),
    "print_group_differences_across_positions": ("sequenzo.compare_differences", "print_group_differences_across_positions"),
    "compare_groups_overall": ("sequenzo.compare_differences", "compare_groups_overall"),
    "compute_likelihood_ratio_test": ("sequenzo.compare_differences", "compute_likelihood_ratio_test"),
    "compute_bayesian_information_criterion_test": ("sequenzo.compare_differences", "compute_bayesian_information_criterion_test"),
    # with_event_history_analysis (SAMM)
    "SAMM": ("sequenzo.with_event_history_analysis", "SAMM"),
    "sequence_analysis_multi_state_model": ("sequenzo.with_event_history_analysis", "sequence_analysis_multi_state_model"),
    "plot_samm": ("sequenzo.with_event_history_analysis", "plot_samm"),
    "seqsammseq": ("sequenzo.with_event_history_analysis", "seqsammseq"),
    "set_typology": ("sequenzo.with_event_history_analysis", "set_typology"),
    "seqsammeha": ("sequenzo.with_event_history_analysis", "seqsammeha"),
    "seqsamm": ("sequenzo.with_event_history_analysis", "seqsamm"),
    # seqhmm
    "HMM": ("sequenzo.seqhmm", "HMM"),
    "build_hmm": ("sequenzo.seqhmm", "build_hmm"),
    "fit_model": ("sequenzo.seqhmm", "fit_model"),
    "predict": ("sequenzo.seqhmm", "predict"),
    "posterior_probs": ("sequenzo.seqhmm", "posterior_probs"),
    "plot_hmm": ("sequenzo.seqhmm", "plot_hmm"),
    "MHMM": ("sequenzo.seqhmm", "MHMM"),
    "build_mhmm": ("sequenzo.seqhmm", "build_mhmm"),
    "fit_mhmm": ("sequenzo.seqhmm", "fit_mhmm"),
    "predict_mhmm": ("sequenzo.seqhmm", "predict_mhmm"),
    "posterior_probs_mhmm": ("sequenzo.seqhmm", "posterior_probs_mhmm"),
    "plot_mhmm": ("sequenzo.seqhmm", "plot_mhmm"),
    "NHMM": ("sequenzo.seqhmm", "NHMM"),
    "build_nhmm": ("sequenzo.seqhmm", "build_nhmm"),
    "fit_nhmm": ("sequenzo.seqhmm", "fit_nhmm"),
    "aic": ("sequenzo.seqhmm", "aic"),
    "bic": ("sequenzo.seqhmm", "bic"),
    "compare_models": ("sequenzo.seqhmm", "compare_models"),
    "simulate_hmm": ("sequenzo.seqhmm", "simulate_hmm"),
    "simulate_mhmm": ("sequenzo.seqhmm", "simulate_mhmm"),
    "bootstrap_model": ("sequenzo.seqhmm", "bootstrap_model"),
    "fit_model_advanced": ("sequenzo.seqhmm", "fit_model_advanced"),
    "Formula": ("sequenzo.seqhmm", "Formula"),
    "create_model_matrix": ("sequenzo.seqhmm", "create_model_matrix"),
}

# Modules that need OpenMP setup before import (clustering, etc.)
_OPENMP_MODULES = frozenset({
    "sequenzo.clustering",
    "sequenzo.clustering.KMedoids",
    "sequenzo.big_data.clara.clara",
    "sequenzo.big_data.clara.visualization",
})

_loaded: dict[str, Any] = {}  # module_path -> module, for caching


def _setup_openmp_if_needed():
    """Setup OpenMP on Apple Silicon before using clustering."""
    import sys
    import os
    import platform
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return
    if os.environ.get("CONDA_DEFAULT_ENV"):
        return
    try:
        from sequenzo.openmp_setup import ensure_openmp_support
        ensure_openmp_support()
    except Exception:
        pass


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_path, attr = _LAZY[name]
    if mod_path in _OPENMP_MODULES:
        _setup_openmp_if_needed()
    if mod_path not in _loaded:
        _loaded[mod_path] = importlib.import_module(mod_path)
    return getattr(_loaded[mod_path], attr)


# Define `__all__` to specify the public API when using `from sequenzo import *`
__all__ = [
    "load_dataset",
    "list_datasets",
    "helpers",
    "clean_time_columns_auto",
    "assign_unique_ids",
    "wide_to_long_format_data",
    "long_to_wide_format_data",
    "summarize_missing_values",
    "replace_cluster_id_by_labels",
    "SequenceData",
    "plot_sequence_index",
    "plot_most_frequent_sequences",
    "plot_single_medoid",
    "plot_state_distribution",
    "plot_modal_state",
    "plot_relative_frequency",
    "plot_mean_time",
    "plot_transition_matrix",
    "get_distance_matrix",
    "get_substitution_cost_matrix",
    "get_LCP_length_for_2_seq",
    "Cluster",
    "ClusterResults",
    "ClusterQuality",
    "KMedoids",
    "clara",
    "plot_scores_from_dataframe",
    "create_idcd_sequence_from_csvs",
    "compute_cat_distance_matrix",
    "compute_dat_distance_matrix",
    "get_interactive_combined_typology",
    "merge_sparse_combt_types",
    "get_association_between_domains",
    "linked_polyadic_sequence_analysis",
    "build_prefix_tree",
    "compute_prefix_count",
    "IndividualDivergence",
    "extract_sequences",
    "get_state_space",
    "compute_branching_factor",
    "compute_js_divergence",
    "convert_to_prefix_tree_data",
    "plot_system_indicators",
    "plot_system_indicators_multiple_comparison",
    "plot_prefix_rarity_distribution",
    "plot_individual_indicators_correlation",
    "SpellPrefixTree",
    "build_spell_prefix_tree",
    "compute_js_divergence_spell",
    "convert_seqdata_to_spells",
    "SpellIndividualDivergence",
    "build_suffix_tree",
    "get_depth_stats",
    "compute_suffix_count",
    "compute_merging_factor",
    "compute_js_convergence",
    "IndividualConvergence",
    "convert_to_suffix_tree_data",
    "plot_suffix_rarity_distribution",
    "SpellSuffixTree",
    "build_spell_suffix_tree",
    "compute_js_convergence_spell",
    "SpellIndividualConvergence",
    "get_subsequences_in_single_sequence",
    "get_subsequences_all_sequences",
    "get_number_of_transitions",
    "get_turbulence",
    "get_complexity_index",
    "get_within_sequence_entropy",
    "get_spell_duration_variance",
    "get_state_freq_and_entropy_per_seq",
    "get_cross_sectional_entropy",
    "plot_longitudinal_characteristics",
    "plot_cross_sectional_characteristics",
    "weighted_mean",
    "weighted_variance",
    "weighted_five_number_summary",
    "compare_groups_across_positions",
    "plot_group_differences_across_positions",
    "print_group_differences_across_positions",
    "compare_groups_overall",
    "compute_likelihood_ratio_test",
    "compute_bayesian_information_criterion_test",
    "SAMM",
    "sequence_analysis_multi_state_model",
    "plot_samm",
    "seqsammseq",
    "set_typology",
    "seqsammeha",
    "seqsamm",
    "HMM",
    "build_hmm",
    "fit_model",
    "predict",
    "posterior_probs",
    "plot_hmm",
    "MHMM",
    "build_mhmm",
    "fit_mhmm",
    "predict_mhmm",
    "posterior_probs_mhmm",
    "plot_mhmm",
    "NHMM",
    "build_nhmm",
    "fit_nhmm",
    "aic",
    "bic",
    "compare_models",
    "simulate_hmm",
    "simulate_mhmm",
    "bootstrap_model",
    "fit_model_advanced",
    "Formula",
    "create_model_matrix",
]

# Version check (async, non-blocking)
def _check_version_update():
    try:
        from .version_check import check_version_update_async
        check_version_update_async()
    except Exception:
        pass

_check_version_update()
del _check_version_update
