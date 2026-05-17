"""Tests for relational sequence visualizations in sequenzo.hierarchical."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from sequenzo.hierarchical import (
    compute_pair_residuals,
    compute_relational_distance_matrix,
    make_relational_sequences,
    plot_hierarchical_distance_heatmap,
    plot_level_1_sequence_panels,
    plot_level_2_sequence_panels,
    plot_level_portfolio_sequences,
    plot_pair_outlier_sequences,
    plot_relational_sequence_grid,
    run_hierarchical_sequence_analysis,
)


def _toy_long_data():
    rows = []
    for region in ("R1", "R2", "R3"):
        for cpc in ("C1", "C2"):
            for t, state in enumerate([0, 0, 1, 2]):
                rows.append(
                    {
                        "region_id": region,
                        "cpc_id": cpc,
                        "year": 2000 + t,
                        "state": state if region == "R1" else t % 3,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def relational_bundle():
    df = _toy_long_data()
    seq = make_relational_sequences(
        df, "region_id", "cpc_id", "year", "state", validate=False
    )
    dist = compute_relational_distance_matrix(seq, method="HAM")
    residuals = compute_pair_residuals(seq, dist)
    return seq, dist, residuals


def test_plot_relational_sequence_grid(relational_bundle, close_mpl_figures):
    seq, _, _ = relational_bundle
    ax = plot_relational_sequence_grid(seq, max_level_1=3, max_level_2=2)
    assert ax is not None
    assert len(ax.figure.axes) >= 6


def test_plot_hierarchical_distance_heatmap(relational_bundle, close_mpl_figures):
    _, dist, _ = relational_bundle
    ax = plot_hierarchical_distance_heatmap(dist, max_pairs=20)
    assert ax.images
    assert len(ax.lines) >= 0


def test_plot_level_portfolio_sequences(relational_bundle, close_mpl_figures):
    seq, _, residuals = relational_bundle
    plot_level_portfolio_sequences(seq, level=1, max_units=2, pair_residuals=residuals)
    plot_level_1_sequence_panels(seq, max_units=2)
    plot_level_2_sequence_panels(seq, max_units=2)
    assert len(plt.get_fignums()) >= 1


def test_plot_pair_outlier_sequences(relational_bundle, close_mpl_figures):
    seq, dist, residuals = relational_bundle
    ax = plot_pair_outlier_sequences(
        seq,
        residuals,
        distance_matrix=dist,
        top_n=3,
        show_medoids=True,
    )
    assert ax is not None
    assert len(ax.figure.axes) >= 3


def test_result_object_plot_helpers(close_mpl_figures):
    df = _toy_long_data()
    result = run_hierarchical_sequence_analysis(
        df,
        "region_id",
        "cpc_id",
        "year",
        "state",
        run_outliers=True,
        n_perm=0,
    )
    result.plot_relational_grid(max_level_1=2, max_level_2=2)
    result.plot_hierarchical_distance_heatmap(max_pairs=10)
    result.plot_outlier_sequences(top_n=2, show_medoids=False)
    plt.close("all")


def test_deprecated_plot_sequence_index_by_level(relational_bundle, close_mpl_figures):
    seq, _, _ = relational_bundle
    with pytest.warns(DeprecationWarning):
        from sequenzo.hierarchical import plot_sequence_index_by_level

        plot_sequence_index_by_level(seq, level=1, max_units=2)
