"""
@Author  : Yuqi Liang 梁彧祺
@File    : _spell_convert.py
@Time    : 08/05/2026 09:25
@Desc    : Convert DSS and spell durations back to SequenceData.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.utils.seqdss import seqdss
from sequenzo.dissimilarity_measures.utils.seqdur import seqdur


def extract_dss_dur(seqdata: SequenceData) -> tuple[np.ndarray, np.ndarray]:
    """Return (dss, sdur) numeric matrices for all sequences."""
    return seqdss(seqdata), seqdur(seqdata)


def _expand_spell_row(
    dss_row: np.ndarray,
    dur_row: np.ndarray,
    inverse_mapping: dict,
    *,
    n_cols: int,
    void_label: Optional[str],
) -> List:
    """Expand one row of DSS/durations into STS values (state labels)."""
    active = np.where(dur_row > 0)[0]
    tokens: List = []
    for idx in active:
        code = int(dss_row[idx])
        if code < 0:
            continue
        label = inverse_mapping.get(code)
        if label is None:
            continue
        tokens.extend([label] * int(dur_row[idx]))
    pad = void_label if void_label is not None else inverse_mapping.get(
        max(inverse_mapping.keys(), default=1)
    )
    if len(tokens) < n_cols:
        tokens.extend([pad] * (n_cols - len(tokens)))
    elif len(tokens) > n_cols:
        tokens = tokens[:n_cols]
    return tokens


def dss_dur_matrices_to_dataframe(
    template: SequenceData,
    dss: np.ndarray,
    sdur: np.ndarray,
    row_names: Sequence[str],
) -> pd.DataFrame:
    """Build a wide sequence DataFrame from DSS and duration matrices."""
    n_rows, _ = dss.shape
    n_cols = len(template.time)
    inv = template.inverse_state_mapping
    void_label = template.void if template.void is not None else None
    rows = []
    for i in range(n_rows):
        rows.append(
            _expand_spell_row(
                dss[i],
                sdur[i],
                inv,
                n_cols=n_cols,
                void_label=void_label,
            )
        )
    frame = pd.DataFrame(rows, columns=template.time, index=list(row_names))
    if template.id_col and template.id_col in template.data.columns:
        frame.insert(0, template.id_col, list(row_names))
    return frame


def dataframe_to_sequence_data(
    template: SequenceData,
    frame: pd.DataFrame,
    *,
    weights: Optional[np.ndarray] = None,
) -> SequenceData:
    """Create ``SequenceData`` sharing metadata with ``template``."""
    id_col = template.id_col
    data = frame.copy()
    if id_col and id_col not in data.columns:
        data.insert(0, id_col, frame.index.astype(str))
    seq = SequenceData(
        data,
        time=template.time,
        states=template.states,
        labels=template.labels,
        id_col=id_col,
        weights=weights,
        start=template.start,
        custom_colors=getattr(template, "custom_colors", None),
        additional_colors=getattr(template, "additional_colors", None),
        missing_values=getattr(template, "missing_values", None),
        void=template.void,
        alpha=getattr(template, "alpha", 1.0),
    )
    return seq


def replicate_dss_dur_to_sequence_data(
    template: SequenceData,
    dss: np.ndarray,
    sdur: np.ndarray,
    row_names: Sequence[str],
    weights: Optional[np.ndarray] = None,
) -> SequenceData:
    """Full path from altered DSS/dur to ``SequenceData``."""
    frame = dss_dur_matrices_to_dataframe(template, dss, sdur, row_names)
    return dataframe_to_sequence_data(template, frame, weights=weights)
