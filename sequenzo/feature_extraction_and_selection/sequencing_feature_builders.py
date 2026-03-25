"""
@Author  : Yuqi Liang 梁彧祺
@File    : sequencing_feature_builders.py
@Time    : 20/03/2026 08:48
@Desc    :
    Build sequencing features from frequent subsequences of spell-start events.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from sequenzo.with_event_history_analysis.event_sequence import (
    EventSequenceConstraint,
    count_subsequence_occurrences,
    create_event_sequences,
    find_frequent_subsequences,
)

from .monthly_state_to_spells import SpellWithTimes


def _sanitize_event_string(s: str, *, max_len: int = 60) -> str:
    s = s.replace(" ", "")
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def spells_to_event_tse(
    spells_per_individual: List[List[SpellWithTimes]],
    *,
    id_values: Optional[Sequence[Any]] = None,
    event_label_mode: str = "state",
    use_start_time: bool = True,
) -> pd.DataFrame:
    n = len(spells_per_individual)
    if id_values is None:
        id_values = list(range(n))
    if len(id_values) != n:
        raise ValueError("id_values length must match number of individuals.")

    rows = []
    for i, spells_i in enumerate(spells_per_individual):
        for sp in spells_i:
            ts = sp.start_time if use_start_time else sp.end_time
            if event_label_mode == "state":
                evt = str(sp.state)
            elif event_label_mode == "begin_end":
                evt = f"{'B' if use_start_time else 'E'}_{sp.state}"
            else:
                raise ValueError("event_label_mode must be 'state' or 'begin_end'.")
            rows.append({"id": id_values[i], "timestamp": float(ts), "event": evt})
    return pd.DataFrame(rows, columns=["id", "timestamp", "event"])


def build_sequencing_features(
    spells_per_individual: List[List[SpellWithTimes]],
    *,
    id_values: Optional[Sequence[Any]] = None,
    max_k: int = 3,
    min_support: Union[int, float] = 0.05,
    count_method: str = "presence",
    constraint: Optional[EventSequenceConstraint] = None,
    top_mined_subsequences: Optional[int] = None,
    event_label_mode: str = "state",
    use_start_time: bool = True,
    weighted: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    tse = spells_to_event_tse(
        spells_per_individual,
        id_values=id_values,
        event_label_mode=event_label_mode,
        use_start_time=use_start_time,
    )
    eseq = create_event_sequences(data=tse, tevent="transition", weighted=weighted)
    if constraint is None:
        constraint = EventSequenceConstraint()

    if isinstance(min_support, float) and 0 < min_support <= 1:
        fsub = find_frequent_subsequences(
            eseq,
            pmin_support=min_support,
            constraint=constraint,
            max_k=max_k,
            weighted=weighted,
        )
    else:
        fsub = find_frequent_subsequences(
            eseq,
            min_support=int(min_support),
            constraint=constraint,
            max_k=max_k,
            weighted=weighted,
        )

    if top_mined_subsequences is not None:
        fsub.subsequences = fsub.subsequences[:top_mined_subsequences]
        fsub.data = fsub.data.iloc[:top_mined_subsequences].reset_index(drop=True)

    counts = count_subsequence_occurrences(fsub, method=count_method, constraint=constraint)
    feature_names: List[str] = []
    for sub in fsub.subsequences:
        name = _sanitize_event_string(getattr(sub, "to_string", lambda: str(sub))())
        feature_names.append(f"SEQ_{name}")
    return counts, feature_names

