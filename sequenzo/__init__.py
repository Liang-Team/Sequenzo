"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 11/02/2025 16:41
@Desc    : 
"""
from .datasets import load_dataset, list_datasets

# Import the core functions that should be directly available from the sequenzo package

from .data_preprocessing import helpers
from .data_preprocessing.helpers import (assign_unique_ids,
                                         wide_to_long_format_data,
                                         long_to_wide_format_data,
                                         summarize_missing_values,
                                         replace_cluster_id_by_labels)

from sequenzo.define_sequence_data import *

from .visualization import (plot_sequence_index,
                            plot_most_frequent_sequences,
                            plot_single_medoid,
                            plot_state_distribution,
                            plot_modal_state,
                            plot_relative_frequency,
                            plot_mean_time,
                            plot_transition_matrix,
                            )

from .dissimilarity_measures.get_distance_matrix import get_distance_matrix
from .dissimilarity_measures.get_substitution_cost_matrix import get_substitution_cost_matrix
from .dissimilarity_measures.utils.get_LCP_length_for_2_seq import get_LCP_length_for_2_seq

from .clustering import Cluster, ClusterResults, ClusterQuality
from .clustering.KMedoids import KMedoids
from .big_data.clara.clara import clara
from .big_data.clara.visualization import plot_scores_from_dataframe

from .multidomain import (create_idcd_sequence_from_csvs,
                          compute_cat_distance_matrix,
                          compute_dat_distance_matrix,
                          get_interactive_combined_typology,
                          merge_sparse_combt_types,
                          get_association_between_domains,
                          linked_polyadic_sequence_analysis
                          )

from .prefix_tree import (
    build_prefix_tree,
    compute_prefix_count,
    IndividualDivergence,
    extract_sequences,
    get_state_space,
    compute_branching_factor,
    compute_js_divergence,
    convert_to_prefix_tree_data,
    plot_system_indicators,
    plot_system_indicators_multiple_comparison,
    plot_prefix_rarity_distribution,
    plot_individual_indicators_correlation
)

from .suffix_tree import (
    build_suffix_tree,
    compute_suffix_count,
    compute_merging_factor,
    compute_js_convergence,
    IndividualConvergence,
    convert_to_suffix_tree_data,
    plot_system_indicators,
    plot_system_indicators_multiple_comparison,
    plot_suffix_rarity_distribution,
)

from .sequence_characteristics import (
    get_subsequences_in_single_sequence,
    get_subsequences_all_sequences,
    get_number_of_transitions,

    get_turbulence,
    get_complexity_index,
    get_within_sequence_entropy,
    get_spell_duration_variance,
    get_state_freq_and_entropy_per_seq,

    get_cross_sectional_entropy,
    plot_cross_sectional_characteristics,
    plot_longitudinal_characteristics

)

# Event History Analysis (SAMM)
from .with_event_history_analysis import (
    SAMM,
    sequence_analysis_multi_state_model,
    plot_samm,
    seqsammseq,
    set_typology,
    seqsammeha,
    # Keep old names for backward compatibility
    seqsamm
)

# Hidden Markov Models (HMM)
from .seqhmm import (
    HMM,
    build_hmm,
    fit_model,
    predict,
    posterior_probs,
    plot_hmm,
    # Mixture HMM
    MHMM,
    build_mhmm,
    fit_mhmm,
    predict_mhmm,
    posterior_probs_mhmm,
    plot_mhmm,
    # Non-homogeneous HMM
    NHMM,
    build_nhmm,
    fit_nhmm,
    # Model comparison
    aic,
    bic,
    compare_models,
    # Simulation
    simulate_hmm,
    simulate_mhmm,
    # Bootstrap
    bootstrap_model,
    # Advanced optimization
    fit_model_advanced,
    # Formulas
    Formula,
    create_model_matrix
)

# Define `__all__` to specify the public API when using `from sequenzo import *`
__all__ = [
    # Datasets
    "load_dataset",
    "list_datasets",

    # Data preprocessing
    "helpers",
    "assign_unique_ids",
    "wide_to_long_format_data",
    "long_to_wide_format_data",
    "summarize_missing_values",
    "replace_cluster_id_by_labels",

    "SequenceData",

    # Visualization
    "plot_sequence_index",
    "plot_most_frequent_sequences",
    "plot_single_medoid",
    "plot_state_distribution",
    "plot_modal_state",
    "plot_relative_frequency",
    "plot_mean_time",
    "plot_transition_matrix",

    # Dissimilarity measures
    "get_distance_matrix",
    "get_substitution_cost_matrix",
    "get_LCP_length_for_2_seq",

    # Hierarchical clustering
    "Cluster",
    "ClusterResults",
    "ClusterQuality",
    "KMedoids",

    # Big data
    "clara",
    "plot_scores_from_dataframe",

    # Multi-domain sequence analysis
    "create_idcd_sequence_from_csvs",
    "compute_cat_distance_matrix",
    "compute_dat_distance_matrix",
    "get_interactive_combined_typology",
    "merge_sparse_combt_types",
    "get_association_between_domains",
    "linked_polyadic_sequence_analysis",

    # Prefix Tree
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

    # Suffix Tree
    "build_suffix_tree",
    "compute_suffix_count",
    "compute_merging_factor",
    "compute_js_convergence",
    "IndividualConvergence",
    "convert_to_suffix_tree_data",
    "plot_system_indicators",
    "plot_system_indicators_multiple_comparison",
    "plot_suffix_rarity_distribution",

    # Sequence characteristics
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

    # Event History Analysis (SAMM)
    "SAMM",
    "sequence_analysis_multi_state_model",
    "plot_samm",
    "seqsammseq",
    "set_typology",
    "seqsammeha",
    # Keep old names for backward compatibility
    "seqsamm",
    
    # Hidden Markov Models (HMM)
    "HMM",
    "build_hmm",
    "fit_model",
    "predict",
    "posterior_probs",
    "plot_hmm",
    # Mixture HMM
    "MHMM",
    "build_mhmm",
    "fit_mhmm",
    "predict_mhmm",
    "posterior_probs_mhmm",
    "plot_mhmm",
    # Non-homogeneous HMM
    "NHMM",
    "build_nhmm",
    "fit_nhmm",
    # Model comparison
    "aic",
    "bic",
    "compare_models",
    # Simulation
    "simulate_hmm",
    "simulate_mhmm",
    # Bootstrap
    "bootstrap_model",
    # Advanced optimization
    "fit_model_advanced",
    # Formulas
    "Formula",
    "create_model_matrix"
]

# Auto-setup OpenMP on Apple Silicon (runs on import)
def _setup_openmp_if_needed():
    """
    Automatically setup OpenMP dependencies on Apple Silicon Macs.
    This function runs on import to ensure OpenMP is available.
    """
    import sys
    import os
    import platform
    
    # Only run on macOS
    if sys.platform != 'darwin':
        return
    
    # Only run on Apple Silicon
    if platform.machine() != 'arm64':
        return
    
    # Check if we're in a conda environment (don't interfere)
    if os.environ.get('CONDA_DEFAULT_ENV'):
        return
    
    # Try to import and run the OpenMP setup
    try:
        from .openmp_setup import ensure_openmp_support
        ensure_openmp_support()
    except ImportError:
        # OpenMP setup module not available, skip
        pass
    except Exception:
        # Any other error, skip silently
        pass

# Run the setup function
_setup_openmp_if_needed()

# Clean up the setup function from the module namespace
del _setup_openmp_if_needed

# Version check: Check for updates asynchronously (non-blocking)
def _check_version_update():
    """
    Check if a newer version of Sequenzo is available.
    This runs asynchronously to avoid blocking the import process.
    """
    try:
        from .version_check import check_version_update_async
        check_version_update_async()
    except Exception:
        # Silently fail - version checks should never break imports
        pass

# Run version check in background
_check_version_update()

# Clean up the version check function from the module namespace
del _check_version_update