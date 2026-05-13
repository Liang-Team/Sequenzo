"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 25/03/2026 19:52
@Desc    : 
"""
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_nt = _os.environ.get("SEQUENZO_NUM_THREADS")
if _nt is not None:
    _os.environ.setdefault("OMP_NUM_THREADS", str(_nt))
# Eager load only lightweight datasets API for fast startup
from .datasets import load_dataset, list_datasets

# Lazy-load everything else on first access
import importlib
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path as _Path
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from typing import Any

try:
    _pyproject = _Path(__file__).resolve().parents[1] / "pyproject.toml"
    __version__ = tomllib.loads(_pyproject.read_text(encoding="utf-8"))["project"]["version"]
except Exception:
    try:
        __version__ = _pkg_version("sequenzo")
    except PackageNotFoundError:
        __version__ = "0+unknown"

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
    "KMedoids": ("sequenzo.clustering.k_medoids", "KMedoids"),
    "clara": ("sequenzo.big_data.clara.clara", "clara"),
    "plot_scores_from_dataframe": ("sequenzo.big_data.clara.visualization", "plot_scores_from_dataframe"),
    # sequences to variables (Helske et al. 2024)
    "max_distance": ("sequenzo.clustering", "max_distance"),
    "cluster_labels_to_dummies": ("sequenzo.clustering", "cluster_labels_to_dummies"),
    "representativeness_matrix": ("sequenzo.clustering", "representativeness_matrix"),
    "medoid_indices_from_kmedoids_result": ("sequenzo.clustering", "medoid_indices_from_kmedoids_result"),
    "cluster_labels_from_kmedoids_result": ("sequenzo.clustering", "cluster_labels_from_kmedoids_result"),
    "hard_classification_variables": ("sequenzo.clustering", "hard_classification_variables"),
    "fanny_membership": ("sequenzo.clustering", "fanny_membership"),
    "representative_indices_from_membership": ("sequenzo.clustering", "representative_indices_from_membership"),
    "soft_classification_variables": ("sequenzo.clustering", "soft_classification_variables"),
    "pseudoclass_regression": ("sequenzo.clustering", "pseudoclass_regression"),
    # multidomain
    "create_idcd_sequence_from_csvs": ("sequenzo.multidomain", "create_idcd_sequence_from_csvs"),
    "create_idcd_sequence_from_dfs": ("sequenzo.multidomain", "create_idcd_sequence_from_dfs"),
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
    # sequence_characteristics_indicators
    "get_subsequences_in_single_sequence": ("sequenzo.sequence_characteristics_indicators", "get_subsequences_in_single_sequence"),
    "get_subsequences_all_sequences": ("sequenzo.sequence_characteristics_indicators", "get_subsequences_all_sequences"),
    "get_number_of_transitions": ("sequenzo.sequence_characteristics_indicators", "get_number_of_transitions"),
    "get_turbulence": ("sequenzo.sequence_characteristics_indicators", "get_turbulence"),
    "get_complexity_index": ("sequenzo.sequence_characteristics_indicators", "get_complexity_index"),
    "get_within_sequence_entropy": ("sequenzo.sequence_characteristics_indicators", "get_within_sequence_entropy"),
    "get_spell_duration_variance": ("sequenzo.sequence_characteristics_indicators", "get_spell_duration_variance"),
    "get_state_freq_and_entropy_per_seq": ("sequenzo.sequence_characteristics_indicators", "get_state_freq_and_entropy_per_seq"),
    "get_cross_sectional_entropy": ("sequenzo.sequence_characteristics_indicators", "get_cross_sectional_entropy"),
    "plot_cross_sectional_characteristics": ("sequenzo.visualization", "plot_cross_sectional_characteristics"),
    "plot_longitudinal_characteristics": ("sequenzo.visualization", "plot_longitudinal_characteristics"),
    # utils
    "weighted_mean": ("sequenzo.utils", "weighted_mean"),
    "weighted_variance": ("sequenzo.utils", "weighted_variance"),
    "weighted_five_number_summary": ("sequenzo.utils", "weighted_five_number_summary"),
    "get_computer_performance": ("sequenzo.utils", "get_computer_performance"),
    # statistics (user-facing)
    "get_weighted_mean": ("sequenzo.statistics", "get_weighted_mean"),
    "get_weighted_variance": ("sequenzo.statistics", "get_weighted_variance"),
    "get_weighted_five_number_summary": ("sequenzo.statistics", "get_weighted_five_number_summary"),
    "get_distinct_state_sequences": ("sequenzo.statistics", "get_distinct_state_sequences"),
    "get_state_spell_durations": ("sequenzo.statistics", "get_state_spell_durations"),
    "get_mean_time_by_state": ("sequenzo.statistics", "get_mean_time_by_state"),
    "get_individual_state_distribution": ("sequenzo.statistics", "get_individual_state_distribution"),
    "get_modal_state_sequence": ("sequenzo.statistics", "get_modal_state_sequence"),
    "get_sequence_length_summary": ("sequenzo.statistics", "get_sequence_length_summary"),
    "get_transition_count_summary": ("sequenzo.statistics", "get_transition_count_summary"),
    # sequence_operations
    "concatenate_sequences": ("sequenzo.sequence_operations", "concatenate_sequences"),
    "decompose_concatenated_sequences": ("sequenzo.sequence_operations", "decompose_concatenated_sequences"),
    "split_fixed_width_sequences": ("sequenzo.sequence_operations", "split_fixed_width_sequences"),
    "recode_sequence_states": ("sequenzo.sequence_operations", "recode_sequence_states"),
    "shift_sequence_with_missing_padding": ("sequenzo.sequence_operations", "shift_sequence_with_missing_padding"),
    "convert_sequences_to_numeric_matrix": ("sequenzo.sequence_operations", "convert_sequences_to_numeric_matrix"),
    "longest_common_prefix_length": ("sequenzo.sequence_operations", "longest_common_prefix_length"),
    "longest_common_subsequence_length": ("sequenzo.sequence_operations", "longest_common_subsequence_length"),
    "find_sequence_occurrences": ("sequenzo.sequence_operations", "find_sequence_occurrences"),
    "pairwise_sequence_alignment": ("sequenzo.sequence_operations", "pairwise_sequence_alignment"),
    # representative sequences/objects
    "get_distance_center": ("sequenzo.representative_sequences", "get_distance_center"),
    "get_relative_frequency_groups": ("sequenzo.representative_sequences", "get_relative_frequency_groups"),
    "get_representative_objects": ("sequenzo.representative_sequences", "get_representative_objects"),
    "get_relative_frequency_representatives": ("sequenzo.representative_sequences", "get_relative_frequency_representatives"),
    "get_representative_sequences": ("sequenzo.representative_sequences", "get_representative_sequences"),
    # discrepancy_analysis
    "overall_discrepancy": ("sequenzo.discrepancy_analysis", "overall_discrepancy"),
    "single_factor_association": ("sequenzo.discrepancy_analysis", "single_factor_association"),
    "permutation_test": ("sequenzo.discrepancy_analysis", "permutation_test"),
    "association_permutation_test": ("sequenzo.discrepancy_analysis", "association_permutation_test"),
    "multifactor_association": ("sequenzo.discrepancy_analysis", "multifactor_association"),
    "distance_multifactor_anova": ("sequenzo.discrepancy_analysis", "distance_multifactor_anova"),
    "individual_indicators": ("sequenzo.discrepancy_analysis", "individual_indicators"),
    "marginal_factor_association": ("sequenzo.discrepancy_analysis", "marginal_factor_association"),
    "merge_cluster_groups": ("sequenzo.discrepancy_analysis", "merge_cluster_groups"),
    "distance_tree": ("sequenzo.discrepancy_analysis", "distance_tree"),
    "sequence_tree": ("sequenzo.discrepancy_analysis", "sequence_tree"),
    "test_tree_split": ("sequenzo.discrepancy_analysis", "test_tree_split"),
    "compare_groups_across_positions": ("sequenzo.discrepancy_analysis", "compare_groups_across_positions"),
    "plot_group_differences_across_positions": ("sequenzo.discrepancy_analysis", "plot_group_differences_across_positions"),
    "print_group_differences_across_positions": ("sequenzo.discrepancy_analysis", "print_group_differences_across_positions"),
    # group_comparison
    "get_group_differences": ("sequenzo.group_comparison", "get_group_differences"),
    "get_lrt_test": ("sequenzo.group_comparison", "get_lrt_test"),
    "get_bic_test": ("sequenzo.group_comparison", "get_bic_test"),
    # decomposition
    "get_oaxaca_blinder_decomposition": ("sequenzo.decomposition", "get_oaxaca_blinder_decomposition"),
    "get_kob_decomposition": ("sequenzo.decomposition", "get_kob_decomposition"),
    "get_kob_decomposition_bootstrap": ("sequenzo.decomposition", "get_kob_decomposition_bootstrap"),
    "get_sa_kob_decomposition": ("sequenzo.decomposition", "get_sa_kob_decomposition"),
    "get_sa_kob_decomposition_bootstrap": ("sequenzo.decomposition", "get_sa_kob_decomposition_bootstrap"),
    "KOBDecompositionResult": ("sequenzo.decomposition", "KOBDecompositionResult"),
    "KOBBootstrapResult": ("sequenzo.decomposition", "KOBBootstrapResult"),
    "SAKOBDecompositionResult": ("sequenzo.decomposition", "SAKOBDecompositionResult"),
    "SAKOBBootstrapResult": ("sequenzo.decomposition", "SAKOBBootstrapResult"),
    # event_sequences (focused entrypoint)
    "find_frequent_subsequences": ("sequenzo.event_sequences", "find_frequent_subsequences"),
    "count_subsequence_occurrences": ("sequenzo.event_sequences", "count_subsequence_occurrences"),
    "compare_groups": ("sequenzo.event_sequences", "compare_groups"),
    "convert_event_sequences_to_tse": ("sequenzo.event_sequences", "convert_event_sequences_to_tse"),
    "compute_event_transition_matrix": ("sequenzo.event_sequences", "compute_event_transition_matrix"),
    "check_event_subsequence_containment": ("sequenzo.event_sequences", "check_event_subsequence_containment"),
    "plot_event_parallel_coordinates": ("sequenzo.event_sequences", "plot_event_parallel_coordinates"),
    "plot_subsequence_frequencies": ("sequenzo.event_sequences", "plot_subsequence_frequencies"),
    "plot_subsequence_group_contrasts": ("sequenzo.event_sequences", "plot_subsequence_group_contrasts"),
    "plot_event_dynamics": ("sequenzo.event_sequences", "plot_event_dynamics"),
    "EventSequence": ("sequenzo.event_sequences", "EventSequence"),
    "EventSequenceData": ("sequenzo.event_sequences", "EventSequenceData"),
    "EventSequenceList": ("sequenzo.event_sequences", "EventSequenceList"),
    "EventSequenceConstraint": ("sequenzo.event_sequences", "EventSequenceConstraint"),
    "SubsequenceList": ("sequenzo.event_sequences", "SubsequenceList"),
    "is_event_sequence": ("sequenzo.event_sequences", "is_event_sequence"),
    "is_event_sequence_collection": ("sequenzo.event_sequences", "is_event_sequence_collection"),
    "get_event_sequence_lengths": ("sequenzo.event_sequences", "get_event_sequence_lengths"),
    "get_event_sequence_weights": ("sequenzo.event_sequences", "get_event_sequence_weights"),
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
    # Feature Extraction & Selection (readable names preferred)
    "run_feature_extraction_and_selection_pipeline": (
        "sequenzo.feature_extraction_and_selection",
        "run_feature_extraction_and_selection_pipeline",
    ),
    "FeatureExtractionAndSelectionConfig": (
        "sequenzo.feature_extraction_and_selection",
        "FeatureExtractionAndSelectionConfig",
    ),
    "clustassoc_like_typology_validation": (
        "sequenzo.feature_extraction_and_selection",
        "clustassoc_like_typology_validation",
    ),
    "get_feature_extraction_and_selection_config_preset": (
        "sequenzo.feature_extraction_and_selection",
        "get_feature_extraction_and_selection_config_preset",
    ),
    # feature_extraction_and_selection (split APIs)
    "extract_sequence_features": ("sequenzo.feature_extraction_and_selection", "extract_sequence_features"),
    "select_relevant_features": ("sequenzo.feature_extraction_and_selection", "select_relevant_features"),
    "interpret_selected_features": ("sequenzo.feature_extraction_and_selection", "interpret_selected_features"),
    "cluster_correlated_features": ("sequenzo.feature_extraction_and_selection", "cluster_correlated_features"),
}

# Modules that need OpenMP setup before import (clustering, etc.)
_OPENMP_MODULES = frozenset({
    "sequenzo.clustering",
    "sequenzo.clustering.k_medoids",
    "sequenzo.big_data.clara.clara",
    "sequenzo.big_data.clara.visualization",
    "sequenzo.dissimilarity_measures",
})

_loaded: dict[str, Any] = {}  # module_path -> module, for caching


_openmp_setup_done = False
def _setup_openmp_if_needed():
    """Setup OpenMP on Apple Silicon before using clustering/dissimilarity."""
    global _openmp_setup_done
    if _openmp_setup_done:
        return
    _openmp_setup_done = True
    import sys
    import os
    import platform
    if sys.platform != "darwin" or platform.machine() != "arm64":
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
    "__version__",
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
    "max_distance",
    "cluster_labels_to_dummies",
    "representativeness_matrix",
    "medoid_indices_from_kmedoids_result",
    "cluster_labels_from_kmedoids_result",
    "hard_classification_variables",
    "fanny_membership",
    "representative_indices_from_membership",
    "soft_classification_variables",
    "pseudoclass_regression",
    "create_idcd_sequence_from_csvs",
    "create_idcd_sequence_from_dfs",
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
    "get_computer_performance",
    "get_weighted_mean",
    "get_weighted_variance",
    "get_weighted_five_number_summary",
    "get_distinct_state_sequences",
    "get_state_spell_durations",
    "get_mean_time_by_state",
    "get_individual_state_distribution",
    "get_modal_state_sequence",
    "get_sequence_length_summary",
    "get_transition_count_summary",
    "concatenate_sequences",
    "decompose_concatenated_sequences",
    "split_fixed_width_sequences",
    "recode_sequence_states",
    "shift_sequence_with_missing_padding",
    "convert_sequences_to_numeric_matrix",
    "longest_common_prefix_length",
    "longest_common_subsequence_length",
    "find_sequence_occurrences",
    "pairwise_sequence_alignment",
    "get_distance_center",
    "get_relative_frequency_groups",
    "get_representative_objects",
    "get_relative_frequency_representatives",
    "get_representative_sequences",
    "overall_discrepancy",
    "single_factor_association",
    "permutation_test",
    "association_permutation_test",
    "multifactor_association",
    "distance_multifactor_anova",
    "individual_indicators",
    "marginal_factor_association",
    "merge_cluster_groups",
    "distance_tree",
    "sequence_tree",
    "test_tree_split",
    "compare_groups_across_positions",
    "plot_group_differences_across_positions",
    "print_group_differences_across_positions",
    "get_group_differences",
    "get_lrt_test",
    "get_bic_test",
    "get_oaxaca_blinder_decomposition",
    "get_kob_decomposition",
    "get_kob_decomposition_bootstrap",
    "get_sa_kob_decomposition",
    "get_sa_kob_decomposition_bootstrap",
    "SAKOBDecompositionResult",
    "SAKOBBootstrapResult",
    "KOBDecompositionResult",
    "KOBBootstrapResult",
    "find_frequent_subsequences",
    "count_subsequence_occurrences",
    "compare_groups",
    "convert_event_sequences_to_tse",
    "compute_event_transition_matrix",
    "check_event_subsequence_containment",
    "plot_event_parallel_coordinates",
    "plot_subsequence_frequencies",
    "plot_subsequence_group_contrasts",
    "plot_event_dynamics",
    "EventSequence",
    "EventSequenceData",
    "EventSequenceList",
    "EventSequenceConstraint",
    "SubsequenceList",
    "is_event_sequence",
    "is_event_sequence_collection",
    "get_event_sequence_lengths",
    "get_event_sequence_weights",
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
    # Feature Extraction & Selection
    "run_feature_extraction_and_selection_pipeline",
    "FeatureExtractionAndSelectionConfig",
    "clustassoc_like_typology_validation",
    "get_feature_extraction_and_selection_config_preset",
    "extract_sequence_features",
    "select_relevant_features",
    "interpret_selected_features",
    "cluster_correlated_features",
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
