"""
@Author  : Yuqi Liang 梁彧祺
@File    : visualization.py
@Time    : 19/05/2026 10:34
@Desc    : 
Visualization helpers for multidomain CLARA results.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sequenzo.big_data.clara.visualization import plot_scores_from_dataframe
from sequenzo.define_sequence_data import SequenceData

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
        rows.append(
            {
                "k": k,
                "ari08": info.get("ari08", np.nan),
                "jc08": info.get("jc08", np.nan),
                "mean_ari": info.get("mean_ari", np.nan),
                "mean_jc": info.get("mean_jc", np.nan),
                "trimmed_mean_ari": info.get("trimmed_mean_ari", np.nan),
                "trimmed_mean_jc": info.get("trimmed_mean_jc", np.nan),
            }
        )
    df = pd.DataFrame(rows)
    r_value = float(result.settings.get("R", np.nan))
    if np.isfinite(r_value) and r_value > 0:
        df["ari08_rate"] = df["ari08"] / r_value
        df["jc08_rate"] = df["jc08"] / r_value
    else:
        df["ari08_rate"] = np.nan
        df["jc08_rate"] = np.nan

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(df["k"], df["mean_ari"], marker="o", label="Mean ARI")
    axes[0, 0].plot(df["k"], df["trimmed_mean_ari"], marker="s", label="Trimmed mean ARI (top 20%)")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_xlabel("Number of clusters (k)")
    axes[0, 0].set_ylabel("ARI (0–1)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(df["k"], df["mean_jc"], marker="o", label="Mean JC")
    axes[0, 1].plot(df["k"], df["trimmed_mean_jc"], marker="s", label="Trimmed mean JC (top 20%)")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_xlabel("Number of clusters (k)")
    axes[0, 1].set_ylabel("JC (0–1)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(df["k"], df["ari08_rate"], marker="o", color="tab:green", label="Share with ARI ≥ 0.8")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xlabel("Number of clusters (k)")
    axes[1, 0].set_ylabel("Fraction of iterations")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(df["k"], df["jc08_rate"], marker="o", color="tab:purple", label="Share with JC ≥ 0.8")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xlabel("Number of clusters (k)")
    axes[1, 1].set_ylabel("Fraction of iterations")
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


def plot_md_cluster_by_domain(
    domains: List[SequenceData],
    labels: Union[np.ndarray, pd.Series],
    k: int,
    *,
    max_ids_per_cluster: int = 40,
    save_as: Optional[str] = None,
) -> None:
    """
    Plot state-sequence heatmaps with rows = domains and columns = clusters.

    The same sequence order is used across domains within each cluster column.
    """
    from sequenzo.multidomain.clara._utils import subset_sequence_data

    labels = np.asarray(labels).reshape(-1)
    unique_clusters = sorted(np.unique(labels[labels >= 0]))
    n_domains = len(domains)
    n_clusters = len(unique_clusters)

    fig, axes = plt.subplots(
        n_domains,
        n_clusters,
        figsize=(3.5 * n_clusters, 2.2 * n_domains),
        squeeze=False,
    )

    for col_idx, cluster_id in enumerate(unique_clusters):
        mask = labels == cluster_id
        order = np.where(mask)[0][:max_ids_per_cluster]
        for row_idx, domain in enumerate(domains):
            ax = axes[row_idx, col_idx]
            if order.size == 0:
                ax.axis("off")
                continue
            sub_seq = subset_sequence_data(domain, order)
            ax.imshow(sub_seq.values, aspect="auto", interpolation="nearest")
            ax.set_title(
                f"D{row_idx + 1} | C{int(cluster_id)}",
                fontsize=9,
            )
            ax.set_xlabel("Time")
            if col_idx == 0:
                ax.set_ylabel("Sequences")

    fig.suptitle(f"Multidomain typology (k={k})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_as:
        fig.savefig(save_as, dpi=200, bbox_inches="tight")
    plt.show()


__all__ = [
    "plot_md_clara_quality",
    "plot_md_clara_stability",
    "plot_md_clara_runtime",
    "plot_md_clara_memory",
    "plot_md_cluster_by_domain",
]
