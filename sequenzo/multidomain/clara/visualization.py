"""
@Author  : Yuqi Liang 梁彧祺
@File    : visualization.py
@Time    : 19/05/2026 10:34
@Desc    : 
Visualization helpers for multidomain CLARA results.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sequenzo.big_data.clara.visualization import plot_scores_from_dataframe
from .results import MDClaraResult


def _stats_for_plot(result: MDClaraResult, criterion: Optional[str] = None) -> pd.DataFrame:
    stats = result.stats.copy()
    if criterion is not None and "criterion" in stats.columns:
        stats = stats[stats["criterion"] == criterion]
    if stats.empty:
        raise ValueError("No statistics available for plotting.")
    return stats.sort_values("k")


def plot_md_clara_quality(
    result: MDClaraResult,
    *,
    criterion: Optional[str] = None,
    metrics: Sequence[str] = ("avg_dist", "db", "xb", "pbm", "ams"),
    title: str = "Multidomain CLARA cluster quality",
    save_as: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Plot quality indices across the requested range of k values.

    Lower is better for distance, DB, XB; higher is better for PBM and AMS.
    Metrics are z-score normalized for comparability on one axis.
    """
    stats = _stats_for_plot(result, criterion=criterion)
    plot_df = stats[["k"] + [m for m in metrics if m in stats.columns]].copy()
    rename = {
        "avg_dist": "Avg dist",
        "db": "DB",
        "xb": "XB",
        "pbm": "PBM",
        "ams": "AMS",
    }
    plot_df = plot_df.rename(columns=rename)
    metric_cols = [rename.get(m, m) for m in metrics if m in stats.columns]

    plot_scores_from_dataframe(
        plot_df,
        k_col="k",
        metrics=metric_cols,
        title=title,
        save_as=save_as,
        **kwargs,
    )


def plot_md_clara_stability(
    result: MDClaraResult,
    *,
    figsize=(10, 5),
    title: str = "Multidomain CLARA stability",
    save_as: Optional[str] = None,
) -> None:
    """
    Plot stability summaries (ARI/JC counts and means) across k.
    """
    if not result.stability:
        raise ValueError("Result has no stability information. Re-run with stability=True.")

    rows = []
    for k, info in sorted(result.stability.items()):
        n_comparisons = info.get("n_comparisons")
        if n_comparisons is None:
            n_comparisons = max(int(result.settings.get("R", 0)) - 1, 0)
        rows.append(
            {
                "k": k,
                "ari08": info.get("ari08", np.nan),
                "jc08": info.get("jc08", np.nan),
                "mean_ari": info.get("mean_ari", np.nan),
                "mean_jc": info.get("mean_jc", np.nan),
                "n_comparisons": n_comparisons,
            }
        )
    df = pd.DataFrame(rows)
    df["ari08_rate"] = df["ari08"] / df["n_comparisons"].replace(0, np.nan)
    df["jc08_rate"] = df["jc08"] / df["n_comparisons"].replace(0, np.nan)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    axes[0, 0].plot(df["k"], df["mean_ari"], marker="o", label="Mean ARI")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_xlabel("Number of clusters (k)")
    axes[0, 0].set_ylabel("ARI (0–1)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(df["k"], df["mean_jc"], marker="o", label="Mean JC")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_xlabel("Number of clusters (k)")
    axes[0, 1].set_ylabel("JC (0–1)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(df["k"], df["ari08_rate"], marker="o", color="tab:green", label="Share with ARI ≥ 0.8")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xlabel("Number of clusters (k)")
    axes[1, 0].set_ylabel("Fraction of other repetitions")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(df["k"], df["jc08_rate"], marker="o", color="tab:purple", label="Share with JC ≥ 0.8")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xlabel("Number of clusters (k)")
    axes[1, 1].set_ylabel("Fraction of other repetitions")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_as:
        fig.savefig(save_as, dpi=200, bbox_inches="tight")
    plt.show()


def plot_md_clara_runtime(
    benchmark_df: pd.DataFrame,
    *,
    hue: str = "strategy",
    title: str = "MD-CLARA runtime",
    save_as: Optional[str] = None,
    figsize=(10, 5),
) -> None:
    """Plot runtime_seconds from a benchmark DataFrame."""
    if "runtime_seconds" not in benchmark_df.columns:
        raise ValueError("benchmark_df must contain 'runtime_seconds'.")
    plt.figure(figsize=figsize)
    if hue in benchmark_df.columns and benchmark_df[hue].nunique() > 1:
        sns.lineplot(data=benchmark_df, x="N", y="runtime_seconds", hue=hue, marker="o")
    else:
        plt.plot(benchmark_df["N"], benchmark_df["runtime_seconds"], marker="o")
    plt.xlabel("Number of sequences (N)")
    plt.ylabel("Runtime (seconds)")
    plt.title(title, fontweight="bold")
    plt.grid(True, alpha=0.3)
    if save_as:
        plt.savefig(save_as, dpi=200, bbox_inches="tight")
    plt.show()


def plot_md_clara_memory(
    benchmark_df: pd.DataFrame,
    *,
    hue: str = "strategy",
    title: str = "MD-CLARA peak memory",
    save_as: Optional[str] = None,
    figsize=(10, 5),
) -> None:
    """Plot peak_memory_mb from a benchmark DataFrame."""
    if "peak_memory_mb" not in benchmark_df.columns:
        raise ValueError("benchmark_df must contain 'peak_memory_mb'.")
    plt.figure(figsize=figsize)
    if hue in benchmark_df.columns and benchmark_df[hue].nunique() > 1:
        sns.lineplot(data=benchmark_df, x="N", y="peak_memory_mb", hue=hue, marker="o")
    else:
        plt.plot(benchmark_df["N"], benchmark_df["peak_memory_mb"], marker="o")
    plt.xlabel("Number of sequences (N)")
    plt.ylabel("Peak memory (MB)")
    plt.title(title, fontweight="bold")
    plt.grid(True, alpha=0.3)
    if save_as:
        plt.savefig(save_as, dpi=200, bbox_inches="tight")
    plt.show()


def plot_cross_strategy_agreement(
    agreement: pd.DataFrame,
    *,
    metric: str = "ari",
    title: str = "Cross-strategy partition agreement",
    figsize: tuple[float, float] = (5.0, 4.0),
    save_as: Optional[str] = None,
) -> None:
    """Heatmap of pairwise strategy agreement (ARI or Jaccard)."""
    if metric not in {"ari", "jaccard"}:
        raise ValueError("metric must be 'ari' or 'jaccard'.")
    if agreement.empty:
        raise ValueError("agreement table is empty.")

    strategies = sorted(
        set(agreement["strategy_left"]).union(agreement["strategy_right"])
    )
    matrix = pd.DataFrame(np.nan, index=strategies, columns=strategies)
    np.fill_diagonal(matrix.values, 1.0)

    for row in agreement.itertuples(index=False):
        value = getattr(row, metric)
        matrix.loc[row.strategy_left, row.strategy_right] = value
        matrix.loc[row.strategy_right, row.strategy_left] = value

    plt.figure(figsize=figsize)
    sns.heatmap(matrix, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="viridis")
    plt.title(title, fontweight="bold")
    if save_as:
        plt.savefig(save_as, dpi=200, bbox_inches="tight")
    plt.show()


def plot_dat_domain_contributions(
    contributions: pd.DataFrame,
    *,
    cluster: Union[int, str] = "all",
    title: Optional[str] = None,
    figsize: tuple[float, float] = (6.0, 4.0),
    save_as: Optional[str] = None,
) -> None:
    """Bar chart of DAT domain contribution shares."""
    subset = contributions[contributions["cluster"] == cluster]
    if subset.empty:
        raise ValueError(f"No domain contributions for cluster={cluster!r}.")

    plot_title = title or f"DAT domain contributions (cluster={cluster})"
    plt.figure(figsize=figsize)
    sns.barplot(
        data=subset,
        x="domain",
        y="contribution_share",
        color="steelblue",
    )
    plt.ylabel("Contribution share")
    plt.xlabel("Domain")
    plt.ylim(0, 1)
    plt.title(plot_title, fontweight="bold")
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=200, bbox_inches="tight")
    plt.show()


def plot_leave_one_domain_out_sensitivity(
    sensitivity: pd.DataFrame,
    *,
    metric: str = "ari_vs_all_domains",
    title: str = "Leave-one-domain-out sensitivity",
    figsize: tuple[float, float] = (6.0, 4.0),
    save_as: Optional[str] = None,
) -> None:
    """Bar chart of agreement with the full-domain model when one domain is omitted."""
    if metric not in sensitivity.columns:
        raise ValueError(
            f"Unknown metric {metric!r}. "
            f"Available columns: {list(sensitivity.columns)}"
        )
    if sensitivity.empty:
        raise ValueError("sensitivity table is empty.")

    plt.figure(figsize=figsize)
    sns.barplot(
        data=sensitivity,
        x="omitted_domain",
        y=metric,
        color="coral",
    )
    plt.ylabel(metric.replace("_", " "))
    plt.xlabel("Omitted domain")
    plt.ylim(0, 1)
    plt.title(title, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=200, bbox_inches="tight")
    plt.show()


__all__ = [
    "plot_md_clara_quality",
    "plot_md_clara_stability",
    "plot_md_clara_runtime",
    "plot_md_clara_memory",
    "plot_cross_strategy_agreement",
    "plot_dat_domain_contributions",
    "plot_leave_one_domain_out_sensitivity",
]
