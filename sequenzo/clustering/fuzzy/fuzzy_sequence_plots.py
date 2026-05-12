"""
@Author  : Yuqi Liang 梁彧祺
@File    : fuzzy_sequence_plots.py
@Time    : 09/05/2025 10:11
@Desc    : 
Membership-weighted sequence plots (WeightedCluster ``fuzzyseqplot``).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from sequenzo.define_sequence_data import SequenceData
from sequenzo.visualization.plot_sequence_index import plot_sequence_index


def _membership_matrix(group: Union[np.ndarray, pd.DataFrame]) -> tuple[np.ndarray, list[str]]:
    if isinstance(group, pd.DataFrame):
        names = [str(col) for col in group.columns]
        return group.to_numpy(dtype=np.float64, copy=True), names
    membership = np.asarray(group, dtype=np.float64)
    if membership.ndim != 2:
        raise ValueError("group must be a membership matrix with one row per sequence.")
    names = [str(idx + 1) for idx in range(membership.shape[1])]
    return membership, names


def _expand_fuzzy_plot_inputs(
    seqdata: SequenceData,
    membership: np.ndarray,
    cluster_names: Sequence[str],
    membership_threshold: float,
    memb_exp: float,
    members_weighted: bool,
    level: Optional[str] = None,
) -> Dict[str, Any]:
    n_obs, n_clusters = membership.shape
    if n_obs != len(seqdata.values):
        raise ValueError("group must have one row per sequence.")

    membership = np.power(membership, memb_exp)
    flat_membership = membership.reshape(-1)
    expanded_index = np.repeat(np.arange(n_obs), n_clusters)
    group_labels = np.repeat(np.asarray(cluster_names, dtype=object), n_obs)

    if members_weighted and seqdata.weights is not None:
        expanded_weights = np.repeat(seqdata.weights, n_clusters) * flat_membership
    else:
        expanded_weights = flat_membership.copy()

    keep = flat_membership >= membership_threshold
    if level is not None:
        keep &= group_labels == level
    if not np.any(keep):
        raise ValueError("No sequence remains after applying the membership threshold.")

    expanded_index = expanded_index[keep]
    group_labels = group_labels[keep]
    expanded_weights = expanded_weights[keep]
    sort_values = flat_membership[keep]

    data = seqdata.data.iloc[expanded_index].reset_index(drop=True)
    expanded_seqdata = SequenceData(
        data=data,
        time=seqdata.time,
        states=seqdata.states,
        labels=seqdata.labels,
        id_col=seqdata.id_col,
        weights=expanded_weights,
        start=seqdata.start,
        custom_colors=seqdata.custom_colors,
    )
    return {
        "seqdata": expanded_seqdata,
        "group_labels": group_labels,
        "sort_values": sort_values,
    }


def fuzzy_sequence_plot(
    seqdata: SequenceData,
    group: Optional[Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
    membership_threshold: float = 0.0,
    plot_type: str = "index",
    members_weighted: bool = True,
    memb_exp: float = 1.0,
    sort_by: str = "unsorted",
    **kwargs,
):
    """
    Plot sequences with membership-weighted replication (``fuzzyseqplot``).

    When ``group`` is ``None`` or a one-dimensional grouping vector, this delegates
    to :func:`plot_sequence_index`. Otherwise ``group`` must be an n x k membership
    matrix. Rows are replicated once per cluster and weighted by membership.
    """
    if group is None:
        return plot_sequence_index(seqdata, **kwargs)

    if isinstance(group, pd.Series) or (isinstance(group, np.ndarray) and group.ndim == 1):
        group_series = pd.Series(group).reset_index(drop=True)
        group_df = pd.DataFrame({seqdata.id_col: seqdata.data[seqdata.id_col], "group": group_series})
        return plot_sequence_index(
            seqdata,
            group_dataframe=group_df,
            group_column_name="group",
            **kwargs,
        )

    membership, cluster_names = _membership_matrix(group)
    prepared = _expand_fuzzy_plot_inputs(
        seqdata=seqdata,
        membership=membership,
        cluster_names=cluster_names,
        membership_threshold=membership_threshold,
        memb_exp=memb_exp,
        members_weighted=members_weighted,
    )

    if sort_by == "membership":
        sort_by = "unsorted"
        kwargs["sort_by_ids"] = prepared["sort_values"].argsort()

    group_df = pd.DataFrame(
        {
            seqdata.id_col: prepared["seqdata"].data[seqdata.id_col],
            "fuzzy_group": prepared["group_labels"],
        }
    )
    return plot_sequence_index(
        prepared["seqdata"],
        group_dataframe=group_df,
        group_column_name="fuzzy_group",
        sort_by=sort_by,
        sort_by_weight=True,
        **kwargs,
    )


def fuzzy_sequence_plot_single(
    seqdata: SequenceData,
    level: Union[int, str],
    group: Union[np.ndarray, pd.DataFrame],
    membership_threshold: float = 0.0,
    plot_type: str = "index",
    members_weighted: bool = True,
    memb_exp: float = 1.0,
    **kwargs,
):
    """Plot one fuzzy cluster level (``fuzzyseqplotsingle``)."""
    membership, cluster_names = _membership_matrix(group)
    prepared = _expand_fuzzy_plot_inputs(
        seqdata=seqdata,
        membership=membership,
        cluster_names=cluster_names,
        membership_threshold=membership_threshold,
        memb_exp=memb_exp,
        members_weighted=members_weighted,
        level=str(level),
    )
    sort_by = kwargs.pop("sort_by", "unsorted")
    if sort_by == "membership":
        kwargs["sort_by_ids"] = prepared["sort_values"].argsort()
    return plot_sequence_index(
        prepared["seqdata"],
        sort_by=sort_by,
        sort_by_weight=True,
        include_legend=False,
        **kwargs,
    )
