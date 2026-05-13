"""
@Author  : Yuqi Liang 梁彧祺
@File    : feature_extraction.py
@Time    : 18/03/2026 19:31
@Desc    : Sequence feature extraction entrypoint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sequenzo.define_sequence_data import SequenceData

from .monthly_state_to_spells import EndTimeMode
from .time_binning_utils import TimeUnitHint


def extract_sequence_features(
    seqdata: SequenceData,
    *,
    state_groups: Optional[Dict[str, List[Any]]] = None,
    timing_bin_width: float = 12.0,
    time_unit_hint: TimeUnitHint = "same_as_labels",
    timing_include_start: bool = True,
    timing_include_end: bool = True,
    timing_count_method: str = "any",
    timing_bin_include_left: bool = True,
    end_time_mode: EndTimeMode = "last_observed",
    sequencing_max_k: int = 3,
    sequencing_min_support: float = 0.05,
    sequencing_top_mined_subsequences: Optional[int] = 1000,
    sequencing_count_method: str = "presence",
    sequencing_event_label_mode: str = "state",
    sequencing_weighted: bool = False,
    ids: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Build duration, timing, and sequencing feature matrices from sequence data.

    ``timing_bin_width`` uses the **same unit as** ``seqdata.time`` (months, years,
    or position indices). For yearly age labels use e.g. ``timing_bin_width=1.0``;
    for monthly position grids use e.g. ``12.0`` when ``time_unit_hint='month'``.

    Sequencing features are mined on the **spell-state sequence** (distinct
    successive states), not on the raw repeated state panel.
    """
    from .duration_timing_feature_builders import (
        build_duration_features,
        build_timing_features,
    )
    from .monthly_state_to_spells import extract_spells_with_times
    from .sequencing_feature_builders import build_sequencing_features
    from .time_binning_utils import coerce_numeric_time_labels, make_equal_width_bins

    if sequencing_weighted:
        raise NotImplementedError(
            "Weighted sequencing feature mining is not currently wired because "
            "EventSequenceData weights are not passed from this entrypoint."
        )

    spells_per_individual = extract_spells_with_times(
        seqdata,
        end_time_mode=end_time_mode,
    )
    if state_groups is None:
        state_groups = {str(s): [s] for s in seqdata.states}

    X_duration, duration_feature_names = build_duration_features(
        spells_per_individual,
        state_groups=state_groups,
    )

    numeric_time = coerce_numeric_time_labels(seqdata.time)
    time_bins = make_equal_width_bins(
        float(min(numeric_time)),
        float(max(numeric_time)),
        timing_bin_width,
    )
    X_timing, timing_feature_names = build_timing_features(
        spells_per_individual,
        state_groups=state_groups,
        start_bins=time_bins,
        include_start=timing_include_start,
        include_end=timing_include_end,
        count_method=timing_count_method,
        bin_include_left=timing_bin_include_left,
    )

    X_sequencing, sequencing_feature_names = build_sequencing_features(
        spells_per_individual,
        id_values=ids,
        max_k=sequencing_max_k,
        min_support=sequencing_min_support,
        count_method=sequencing_count_method,
        top_mined_subsequences=sequencing_top_mined_subsequences,
        event_label_mode=sequencing_event_label_mode,
        use_start_time=True,
        weighted=sequencing_weighted,
    )
    all_feature_names = duration_feature_names + timing_feature_names + sequencing_feature_names
    X_full = pd.DataFrame(
        data=np.hstack([X_duration, X_timing, X_sequencing]),
        columns=all_feature_names,
        index=ids,
    )

    return {
        "time_unit_hint": time_unit_hint,
        "timing_bin_width": timing_bin_width,
        "end_time_mode": end_time_mode,
        "X_duration": pd.DataFrame(X_duration, columns=duration_feature_names, index=ids),
        "X_timing": pd.DataFrame(X_timing, columns=timing_feature_names, index=ids),
        "X_sequencing": pd.DataFrame(X_sequencing, columns=sequencing_feature_names, index=ids),
        "X_full": X_full,
        "all_feature_names": all_feature_names,
    }


__all__ = ["extract_sequence_features"]
