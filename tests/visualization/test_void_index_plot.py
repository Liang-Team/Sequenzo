"""Void padding is masked in index plots and omitted from legends."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from sequenzo.define_sequence_data import SequenceData
from sequenzo.visualization import plot_medoids, plot_sequence_index, plot_state_distribution


def _seq_with_void() -> SequenceData:
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "1": ["A", "A"],
            "2": ["B", "%"],
            "3": ["%", "%"],
        }
    )
    return SequenceData(
        df,
        time=["1", "2", "3"],
        states=["A", "B", "%"],
        id_col="id",
        void="%",
    )


def test_index_plot_masks_void_cells(tmp_path: Path) -> None:
    seq = _seq_with_void()
    plotted = seq.values_for_plot()
    assert np.isnan(plotted[0, 2])
    assert np.isnan(plotted[1, 1])
    assert np.isnan(plotted[1, 2])
    assert plotted[0, 0] == seq.state_mapping["A"]

    out = tmp_path / "index.png"
    plot_sequence_index(seq, save_as=str(out), dpi=80, show=False)
    assert out.exists() and out.stat().st_size > 0
    handles, labels = seq.get_legend()
    assert "%" not in labels
    assert "A" in labels and "B" in labels
    assert len(handles) == 2


def test_state_distribution_and_medoids_omit_void_legend(tmp_path: Path) -> None:
    seq = _seq_with_void()
    dist_path = tmp_path / "dist.png"
    plot_state_distribution(seq, save_as=str(dist_path), dpi=80, show=False)
    assert dist_path.exists() and dist_path.stat().st_size > 0

    dist = np.abs(seq.values[:, None, :] != seq.values[None, :, :]).sum(axis=2).astype(float)
    med_path = tmp_path / "medoids.png"
    plot_medoids(
        seq,
        medoid_indices=[0, 1],
        cluster_labels=["C1", "C2"],
        save_as=str(med_path),
        dpi=80,
        show=False,
    )
    assert med_path.exists() and med_path.stat().st_size > 0
