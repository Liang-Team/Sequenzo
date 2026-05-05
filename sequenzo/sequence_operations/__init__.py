"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 15/04/2026 12:07
@Desc    : 

Sequence operations compatible with TraMineR-style workflows.

This module implements utility operations often used to reshape or recode
sequence representations:
    - concatenate_sequences
    - decompose_concatenated_sequences
    - split_fixed_width_sequences
    - recode_sequence_states
    - shift_sequence_with_missing_padding
    - convert_sequences_to_numeric_matrix
    - find_sequence_occurrences
    - longest_common_prefix_length
    - longest_common_subsequence_length
    - pairwise_sequence_alignment
"""

from .operations import (
    concatenate_sequences,
    decompose_concatenated_sequences,
    split_fixed_width_sequences,
    recode_sequence_states,
    shift_sequence_with_missing_padding,
    convert_sequences_to_numeric_matrix,
    longest_common_prefix_length,
    longest_common_subsequence_length,
    find_sequence_occurrences,
    pairwise_sequence_alignment,
)

__all__ = [
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
]
