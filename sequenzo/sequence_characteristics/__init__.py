"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 22/09/2025 22:46
@Desc    :
"""
from .simple_characteristics import (get_subsequences_in_single_sequence, 
                                     get_subsequences_all_sequences, 
                                     get_number_of_transitions)

from .state_frequencies_and_entropy_per_sequence import get_state_freq_and_entropy_per_seq

from .within_sequence_entropy import get_within_sequence_entropy

from .overall_cross_sectional_entropy import get_cross_sectional_entropy

from .variance_of_spell_durations import get_spell_duration_variance

from .turbulence import get_turbulence

from .complexity_index import get_complexity_index

from .plot_characteristics import plot_longitudinal_characteristics, plot_cross_sectional_characteristics

# Basic indicators
from .basic_indicators import (get_sequence_length, get_spell_durations,
                              get_visited_states, get_recurrence,
                              get_mean_spell_duration, get_duration_standard_deviation)

# Diversity indicators
from .entropy_difference import get_entropy_difference

# Complexity indicators
from .volatility import get_volatility

# Binary indicators
from .binary_indicators import get_positive_negative_indicators
from .integration_index import get_integration_index

# Ranked indicators
from .ranked_indicators import (get_badness_index, get_degradation_index,
                               get_precarity_index, get_insecurity_index)

# Cross-sectional indicators
from .cross_sectional_indicators import (get_mean_time_in_states,
                                        get_modal_state_sequence)

__all__ = [
    # Original functions
    "get_subsequences_in_single_sequence",
    "get_subsequences_all_sequences",
    "get_number_of_transitions",
    "get_complexity_index",
    "get_state_freq_and_entropy_per_seq",
    "get_within_sequence_entropy",
    "get_cross_sectional_entropy",
    "get_spell_duration_variance",
    "get_turbulence",
    "plot_longitudinal_characteristics",
    "plot_cross_sectional_characteristics",
    
    # Basic indicators
    "get_sequence_length",
    "get_spell_durations",
    "get_visited_states",
    "get_recurrence",
    "get_mean_spell_duration",
    "get_duration_standard_deviation",
    
    # Diversity indicators
    "get_entropy_difference",
    
    # Complexity indicators
    "get_volatility",
    
    # Binary indicators
    "get_positive_negative_indicators",
    "get_integration_index",
    
    # Ranked indicators
    "get_badness_index",
    "get_degradation_index",
    "get_precarity_index",
    "get_insecurity_index",
    
    # Cross-sectional indicators
    "get_mean_time_in_states",
    "get_modal_state_sequence",
]