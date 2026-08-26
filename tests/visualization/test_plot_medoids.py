"""Tests for plot_medoids subplot layouts and cluster subsetting."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

from sequenzo.define_sequence_data import SequenceData
from sequenzo.visualization import plot_medoids


def _toy_seqdata() -> SequenceData:
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
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
