"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 30/09/2025 23:34
@Desc    : Event History Analysis module for sequence analysis
"""

from .sequence_analysis_multi_state_model import (
    SAMM,
    sequence_analysis_multi_state_model,
    plot_samm,
    seqsammseq,
    set_typology,
    seqsammeha,
    # Keep old names for backward compatibility
    seqsamm
)

from .sequence_history_analysis import (
    seqsha,
    person_level_to_person_period
)

__all__ = [
    'SAMM',
    'sequence_analysis_multi_state_model',
    'plot_samm',
    'seqsammseq',
    'set_typology',
    'seqsammeha',
    'seqsha',
    'person_level_to_person_period',
    # Keep old names for backward compatibility
    'seqsamm'
]
