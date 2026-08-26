"""Tests for plot_medoids subplot layouts, cluster subsetting, and grouping."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from sequenzo.define_sequence_data import SequenceData
from sequenzo.visualization import plot_medoids


def _toy_seqdata() -> SequenceData:
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "gender": ["Female", "Female", "Male", "Male"],
            "1": ["None", "Below", "None", "Comparative advantage"],
            "2": ["None", "Below", "Below", "Comparative advantage"],
            "3": ["Below", "None", "Below", "Comparative advantage"],
        }
    )
    return SequenceData(
        df,
        time=["1", "2", "3"],
        states=["None", "Below", "Comparative advantage"],
        id_col="id",
        void=None,
    )


def _hamming(seq: SequenceData) -> np.ndarray:
    values = seq.values
    n = len(values)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.sum(values[i] != values[j])
    return dist


def test_plot_medoids_column_and_cluster_subset(tmp_path: Path) -> None:
    seq = _toy_seqdata()
    out_all = tmp_path / "all_column.png"
    plot_medoids(
        seq,
        [0, 1, 3],
        cluster_labels=[1, 2, 3],
        layout="column",
        save_as=str(out_all),
        dpi=120,
        show=False,
    )
    assert out_all.exists() and out_all.stat().st_size > 0

    out_one = tmp_path / "cluster2.png"
    plot_medoids(
        seq,
        [0, 1, 3],
        cluster_labels=[1, 2, 3],
        clusters=2,
        layout="column",
        save_as=str(out_one),
        dpi=120,
        show=False,
    )
    assert out_one.exists() and out_one.stat().st_size > 0


def test_plot_medoids_grid_and_stacked(tmp_path: Path) -> None:
    seq = _toy_seqdata()
    out_grid = tmp_path / "grid.png"
    plot_medoids(
        seq,
        [0, 1, 2, 3],
        layout="grid",
        save_as=str(out_grid),
        dpi=120,
        show=False,
    )
    assert out_grid.exists()

    out_stack = tmp_path / "stacked.png"
    plot_medoids(
        seq,
        [0, 2, 3],
        layout="stacked",
        save_as=str(out_stack),
        dpi=120,
        show=False,
    )
    assert out_stack.exists()


def test_plot_medoids_group_by_column(tmp_path: Path) -> None:
    seq = _toy_seqdata()
    dist = _hamming(seq)
    out = tmp_path / "by_gender.png"
    plot_medoids(
        seq,
        distance_matrix=dist,
        group_by_column="gender",
        layout="column",
        save_as=str(out),
        dpi=120,
        show=False,
    )
    assert out.exists() and out.stat().st_size > 0

    out_one = tmp_path / "female_only.png"
    plot_medoids(
        seq,
        distance_matrix=dist,
        group_by_column="gender",
        groups="Female",
        save_as=str(out_one),
        dpi=120,
        show=False,
    )
    assert out_one.exists()


def test_plot_medoids_group_dataframe_clusters(tmp_path: Path) -> None:
    seq = _toy_seqdata()
    membership = pd.DataFrame(
        {"id": ["a", "b", "c", "d"], "Cluster": [1, 1, 2, 2]}
    )
    out = tmp_path / "by_cluster_table.png"
    plot_medoids(
        seq,
        distance_matrix=_hamming(seq),
        group_dataframe=membership,
        group_column_name="Cluster",
        groups=1,
        save_as=str(out),
        dpi=120,
        show=False,
    )
    assert out.exists() and out.stat().st_size > 0
