"""
@Author  : Yuqi Liang 梁彧祺
@File    : duration_timing_feature_builders.py
@Time    : 19/03/2026 17:05
@Desc    :
    Build duration and timing feature matrices from spell trajectories.

    Timing features code spell **entry** (START) and **exit** (END) events:
    START is transition into a state; END is leaving a state (see
    ``end_time_mode`` in spell extraction).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .monthly_state_to_spells import SpellWithTimes
from .time_binning_utils import in_bin


def _build_state_groups_from_states(states: Iterable[Any]) -> Dict[str, List[Any]]:
    return {str(s): [s] for s in states}


def build_duration_features(
    spells_per_individual: List[List[SpellWithTimes]],
    *,
    state_groups: Optional[Dict[str, List[Any]]] = None,
    states: Optional[Iterable[Any]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Total time spent in each state group (summed over spells).

    Durations are in **sequence-position steps** on ``seqdata.time`` (same unit
    as the time grid), not necessarily calendar months.
    """
    if state_groups is None:
        if states is None:
            raise ValueError("Either state_groups or states must be provided.")
        state_groups = _build_state_groups_from_states(states)

    group_names = list(state_groups.keys())
    group_to_states = {g: set(state_groups[g]) for g in group_names}
    n = len(spells_per_individual)
    X = np.zeros((n, len(group_names)), dtype=float)

    for i, spells_i in enumerate(spells_per_individual):
        for k, g in enumerate(group_names):
            total = 0.0
            for sp in spells_i:
                if sp.state in group_to_states[g]:
                    total += float(sp.duration_steps)
            X[i, k] = total

    feature_names = [f"DUR_{g}" for g in group_names]
    return X, feature_names


def build_timing_features(
    spells_per_individual: List[List[SpellWithTimes]],
    *,
    state_groups: Optional[Dict[str, List[Any]]] = None,
    states: Optional[Iterable[Any]] = None,
    start_bins: List[tuple[float, float]],
    include_start: bool = True,
    include_end: bool = False,
    count_method: str = "any",
    bin_include_left: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Binary (or count) timing features for spell entry and exit events.

    - ``START_<group>_BIN*``: spell **entry** into a state (transition timing).
    - ``END_<group>_BIN*``: spell **exit** from a state (when ``include_end``).

    Bin boundaries use the same unit as ``seqdata.time`` (see ``timing_bin_width``).
    """
    if state_groups is None:
        if states is None:
            raise ValueError("Either state_groups or states must be provided.")
        state_groups = _build_state_groups_from_states(states)

    group_names = list(state_groups.keys())
    group_to_states = {g: set(state_groups[g]) for g in group_names}

    feature_specs: List[tuple[str, int, str]] = []
    for g in group_names:
        for b_idx in range(len(start_bins)):
            if include_start:
                feature_specs.append((g, b_idx, "START"))
            if include_end:
                feature_specs.append((g, b_idx, "END"))

    n = len(spells_per_individual)
    X = np.zeros((n, len(feature_specs)), dtype=float)

    for i, spells_i in enumerate(spells_per_individual):
        for j, (g, b_idx, which) in enumerate(feature_specs):
            start, end = start_bins[b_idx]
            matched = 0
            for sp in spells_i:
                if sp.state not in group_to_states[g]:
                    continue
                ts = sp.start_time if which == "START" else sp.end_time
                if in_bin(ts, start, end, include_left=bin_include_left):
                    matched += 1
            if count_method == "any":
                X[i, j] = 1.0 if matched > 0 else 0.0
            elif count_method == "count":
                X[i, j] = float(matched)
            else:
                raise ValueError("count_method must be either 'any' or 'count'.")

    feature_names = [f"{which}_{g}_BIN{b_idx+1}" for (g, b_idx, which) in feature_specs]
    return X, feature_names
