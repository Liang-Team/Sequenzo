"""
User-facing sequence statistics aligned with TraMineR semantics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.utils.seqdss import seqdss
from sequenzo.dissimilarity_measures.utils.seqdur import seqdur
from sequenzo.sequence_characteristics_indicators.cross_sectional_indicators import get_mean_time_in_states
from sequenzo.sequence_characteristics_indicators.cross_sectional_indicators import (
    get_modal_state_sequence as _get_modal_state_sequence,
)
from sequenzo.sequence_characteristics_indicators.basic_indicators import get_sequence_length
from sequenzo.sequence_characteristics_indicators.simple_characteristics import get_number_of_transitions
from sequenzo.sequence_characteristics_indicators.state_frequencies_and_entropy_per_sequence import (
    get_state_freq_and_entropy_per_seq,
)


def get_distinct_state_sequences(seqdata: SequenceData, fill_value: int = -999) -> pd.DataFrame:
    """
    Equivalent to TraMineR::seqdss().
    """
    if not isinstance(seqdata, SequenceData):
        raise TypeError("`seqdata` must be a SequenceData object.")
    arr = seqdss(seqdata)
    cols = [f"Spell{i+1}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, index=seqdata.seqdata.index, columns=cols).fillna(fill_value)


def get_state_spell_durations(seqdata: SequenceData, fill_value: int = 0) -> pd.DataFrame:
    """
    Equivalent to TraMineR::seqdur().
    """
    if not isinstance(seqdata, SequenceData):
        raise TypeError("`seqdata` must be a SequenceData object.")
    arr = seqdur(seqdata)
    cols = [f"Duration{i+1}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, index=seqdata.seqdata.index, columns=cols).fillna(fill_value)


def get_mean_time_by_state(
    seqdata: SequenceData,
    weighted: bool = True,
    as_proportion: bool = False,
    show_standard_error: bool = False,
    with_missing: bool = False,
) -> pd.DataFrame:
    """
    Equivalent to TraMineR::seqmeant().
    """
    if not isinstance(seqdata, SequenceData):
        raise TypeError("`seqdata` must be a SequenceData object.")
    return get_mean_time_in_states(
        seqdata=seqdata,
        weighted=weighted,
        with_missing=with_missing,
        prop=as_proportion,
        serr=show_standard_error,
    )


def get_individual_state_distribution(seqdata: SequenceData, as_proportion: bool = False) -> pd.DataFrame:
    """
    Equivalent to TraMineR::seqistatd().
    """
    if not isinstance(seqdata, SequenceData):
        raise TypeError("`seqdata` must be a SequenceData object.")
    return get_state_freq_and_entropy_per_seq(seqdata, prop=as_proportion)


def get_modal_state_sequence(seqdata: SequenceData, weighted: bool = True, with_missing: bool = False) -> pd.DataFrame:
    """
    Equivalent to TraMineR::seqmodst().
    """
    if not isinstance(seqdata, SequenceData):
        raise TypeError("`seqdata` must be a SequenceData object.")
    return _get_modal_state_sequence(seqdata=seqdata, weighted=weighted, with_missing=with_missing)


def get_sequence_length_summary(seqdata: SequenceData, with_missing: bool = True) -> pd.DataFrame:
    """
    Return descriptive summary statistics for sequence lengths.
    """
    if not isinstance(seqdata, SequenceData):
        raise TypeError("`seqdata` must be a SequenceData object.")
    lengths = get_sequence_length(seqdata=seqdata, with_missing=with_missing)["Length"].to_numpy(dtype=float)
    summary = {
        "count": int(lengths.size),
        "mean": float(np.mean(lengths)) if lengths.size else np.nan,
        "median": float(np.median(lengths)) if lengths.size else np.nan,
        "min": float(np.min(lengths)) if lengths.size else np.nan,
        "q1": float(np.quantile(lengths, 0.25)) if lengths.size else np.nan,
        "q3": float(np.quantile(lengths, 0.75)) if lengths.size else np.nan,
        "max": float(np.max(lengths)) if lengths.size else np.nan,
    }
    return pd.DataFrame([summary])


def get_transition_count_summary(
    seqdata: SequenceData, normalize: bool = False, probability_weighted: bool = False
) -> pd.DataFrame:
    """
    Return descriptive summary statistics for transition counts.
    """
    if not isinstance(seqdata, SequenceData):
        raise TypeError("`seqdata` must be a SequenceData object.")
    transitions = get_number_of_transitions(
        seqdata=seqdata, norm=normalize, pwight=probability_weighted
    )["Transitions"].to_numpy(dtype=float)
    summary = {
        "count": int(transitions.size),
        "mean": float(np.mean(transitions)) if transitions.size else np.nan,
        "median": float(np.median(transitions)) if transitions.size else np.nan,
        "min": float(np.min(transitions)) if transitions.size else np.nan,
        "q1": float(np.quantile(transitions, 0.25)) if transitions.size else np.nan,
        "q3": float(np.quantile(transitions, 0.75)) if transitions.size else np.nan,
        "max": float(np.max(transitions)) if transitions.size else np.nan,
    }
    return pd.DataFrame([summary])
