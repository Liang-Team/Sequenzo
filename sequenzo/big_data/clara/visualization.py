"""
@Author  : Yuqi Liang 梁彧祺
@File    : visualization.py
@Time    : 04/04/2025 15:21
@Desc    :

"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_scores_from_dataframe(df,
                               k_col="k",
                               metrics=None,
                               norm="zscore",
                               title="CLARA Cluster Quality Metrics",
                               palette="Set2",
                               line_width=2,
                               style="whitegrid",
                               xlabel="Number of Clusters",
                               ylabel="Normalized Score",
                               grid=True,
                               save_as=None,
                               dpi=200,
                               figsize=(12, 8)):
    """
    Plot clustering metrics directly from a summary DataFrame (e.g., loaded from CSV).

    :param df: DataFrame with clustering metrics. Must include a 'k' column.
    :param k_col: Column name indicating the number of clusters.
    :param metrics: List of metric columns to plot. If None, auto-detect numeric columns.
    :param norm: Normalization method for plotting ('zscore', 'range', or 'none')
    :param title: Plot title
    :param palette: Color palette for the plot
    :param line_width: Width of plotted lines
    :param style: Seaborn style for the plot
    :param xlabel: X-axis label
    :param ylabel: Y-axis label
    :param grid: Whether to show grid lines
    :param save_as: File path to save the plot (optional)
    :param dpi: DPI for saved image
    :param figsize: Figure size in inches
    """
    df = df.copy()
    df = df.sort_values(by=k_col)

    if metrics is None:
        metrics = df.select_dtypes(include=[float, int]).columns.tolist()
        blacklist = ["Best iter", k_col] # Removed best iter as it is not part of the indicators for cluster quality evaluation
        metrics = [m for m in metrics if m not in blacklist]

    normed = {}
    for metric in metrics:
        values = df[metric].values.astype(float)
        if norm == "zscore":
            mean = np.nanmean(values)
            std = np.nanstd(values)
            normed[metric] = (values - mean) / std if std > 0 else values
        elif norm == "range":
            min_val = np.nanmin(values)
            max_val = np.nanmax(values)
            normed[metric] = (values - min_val) / (max_val - min_val) if max_val > min_val else values
        else:
            normed[metric] = values

    sns.set(style=style)
    palette_colors = sns.color_palette(palette, len(metrics))
    plt.figure(figsize=figsize)

    for idx, metric in enumerate(metrics):
        plt.plot(df[k_col], normed[metric],
                 label=metric,
                 linewidth=line_width,
                 color=palette_colors[idx])

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(df[k_col])
    plt.grid(grid, linestyle="--", alpha=0.6)
    plt.legend(title="Metric", fontsize=10)
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=dpi)
    plt.show()
