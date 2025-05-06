"""
@Author  : Yuqi Liang 梁彧祺
@File    : plot_relative_frequency.py
@Time    : 06/02/2025 10:17
@Desc    :
    Generate sequence relative frequency plots with medoids and dissimilarities.
    TODO: Update the xticks.
"""
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
# from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

from sequenzo.define_sequence_data import SequenceData
from sequenzo.visualization.utils import (
    save_and_show_results,
    set_up_time_labels_for_x_axis
)


# Delay imports to avoid circular dependency issues during installation
def _get_standard_scaler():
    try:
        from sklearn.preprocessing import StandardScaler
        return StandardScaler
    except ImportError:
        print(
            "Warning: Not able to install StandardScaler。Please ensure that you have installed scikit-learn successfully.")
        return None


def plot_relative_frequency(seqdata: SequenceData,
                            distance_matrix: np.ndarray,
                            num_groups: int = 12,
                            save_as=None,
                            dpi=200):
    """
    Generate a sequence relative frequency (seqrf) plot.

    :param seqdata: (SequenceData) The SequenceData object.
    :param distance_matrix: (np.ndarray) A 2D pairwise distance matrix.
    :param num_groups: (int) Number of frequency groups.
    :param save_as: (str, optional) File path to save the plot.
    :param dpi: (int) Resolution of the saved plot.
    """
    if isinstance(distance_matrix, pd.DataFrame):
        distance_matrix = distance_matrix.to_numpy()

    # Compute medoids and dissimilarities
    rep_sequences, dissimilarities, group_labels = _compute_seqrf(seqdata, distance_matrix, num_groups)

    # **Auto-adjust figure ratio**: dynamically scale aspect ratio
    num_seq = len(rep_sequences)
    fig_width = 14  # Fixed width
    fig_height = max(6, num_seq / 20)  # Adjust height based on the number of sequences

    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), gridspec_kw={'width_ratios': [2.5, 1]})
    sns.set_palette("muted")

    # Use color mapping stored in SequenceData
    state_palette = seqdata.color_map

    # **LEFT PLOT: Group Medoids (Sequence Index Plot)**
    ax = axes[0]
    for i, seq in enumerate(rep_sequences):
        for t, state_idx in enumerate(seq):
            color = state_palette.get(state_idx, "gray")  # 直接用整数查颜色
            ax.add_patch(Rectangle((t, i), 1, 1, color=color))

    ax.set_xlim(0, seqdata.values.shape[1])
    ax.set_ylim(0, len(rep_sequences))
    ax.set_title("Group Medoids", fontsize=14)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Frequency Group", fontsize=12)

    # X-axis labels
    # TODO 权宜之计，不然 index plot 里面没有，但是这里有但是在 quickstart 和 multidomain main_tutorial 里面
    # 因为time一个数字一个string导致不一样，太麻烦了
    # 仅显示一部分 xticks，避免过于密集
    xtick_positions = np.arange(len(seqdata.cleaned_time))
    skip = max(1, len(seqdata.cleaned_time) // 8)  # 每隔几个显示一个（可调）
    visible_positions = xtick_positions[::skip]
    visible_labels = [seqdata.cleaned_time[i] for i in visible_positions]

    ax.set_xticks(visible_positions)
    ax.set_xticklabels(visible_labels, fontsize=10, rotation=0, ha='right', color='gray')

    # Y-axis labels
    ax.set_yticks(range(0, num_groups, max(1, num_groups // 10)))
    ax.set_yticklabels(range(1, num_groups + 1, max(1, num_groups // 10)), fontsize=10, color='gray')

    # **Remove unwanted black outlines**
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # **RIGHT PLOT: Dissimilarity Box Plot**
    box_ax = axes[1]

    # Set box plot styling
    box_parts = box_ax.boxplot(
        dissimilarities,
        vert=False,  # Horizontal box plot
        patch_artist=True,  # Allow fill color
        boxprops=dict(facecolor='lightblue', edgecolor='gray', linewidth=1),  # Box style
        whiskerprops=dict(color='gray', linewidth=1),  # Whisker style
        capprops=dict(color='gray', linewidth=1),  # Cap line style
        medianprops=dict(color='red', linewidth=2),  # Median line style
        flierprops=dict(marker='o', markerfacecolor='gray', markersize=5, markeredgecolor='none')  # Outlier style
    )

    # Y-axis labels
    box_ax.set_yticks(range(0, num_groups, max(1, num_groups // 10)))
    box_ax.set_yticklabels(range(1, num_groups + 1, max(1, num_groups // 10)), fontsize=10, color='black')

    # Keep only the bottom x-axis visible
    box_ax.spines["top"].set_visible(False)
    box_ax.spines["right"].set_visible(False)
    box_ax.spines["left"].set_visible(True)
    box_ax.spines["bottom"].set_visible(True)

    # Set titles and labels
    box_ax.set_title("Dissimilarities to Medoid", fontsize=14)
    box_ax.set_xlabel("Dissimilarity", fontsize=12)
    box_ax.set_ylabel("Group", fontsize=12)

    # Adjust layout
    # TODO 出现问题的地方 - 状态多了就有问题(quickstart) ，状态比较少就没问题 Tutorial/multidomain/main_tutorial
    # plt.subplots_adjust(bottom=0.23, wspace=0.4)
    num_legend_items = len(state_palette)
    bottom_margin = min(0.3, 0.14 + num_legend_items * 0.015)
    plt.subplots_adjust(bottom=bottom_margin, wspace=0.4)

    # **Representation Quality Stats**
    r_squared, f_statistic, p_value = _compute_r2_f_statistic(distance_matrix, group_labels)

    # Compute significance level for p-value (show as *, **, ***)
    def get_p_value_stars(p_value):
        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        else:
            return ""

    # Format p-value for display
    p_value_stars = get_p_value_stars(p_value)
    p_value_text = f"p = {p_value:.2e} {p_value_stars}"

    # Explanation of p-value significance levels
    stars_explanation = "*: p < 0.05, **: p < 0.01, ***: p < 0.001"

    stats_text = (f"Representation quality: Pseudo/medoid-based R² = {r_squared:.2f}, F statistic = {f_statistic:.2f}, "
                  f"{p_value_text} ({stars_explanation})")

    # **LEGEND BELOW PLOTS**
    legend_patches = [
        Rectangle((0, 0), 1, 1, color=seqdata.color_map_by_label[label], label=label)
        for label in seqdata.labels
    ]

    # Automatically adjust legend layout (maximum of 7 items per row)
    # ncol = min(7, len(seqdata.states))  # Maximum of 7 legend items per row
    # legend = fig.legend(
    #     handles=legend_patches,
    #     loc='lower center',
    #     ncol=ncol,
    #     fontsize=12,
    #     frameon=False,
    #     bbox_to_anchor=(0.5, 0.05)  # Position legend at the bottom center
    # )

    # Estimate how many rows are needed for the legend
    max_items_per_row = 5
    n_states = len(seqdata.states)
    ncol = min(max_items_per_row, n_states)
    nrow = (n_states + max_items_per_row - 1) // max_items_per_row  # 向上取整

    legend = fig.legend(
        handles=legend_patches,
        loc='lower center',
        ncol=ncol,
        fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.5, 0.05 + 0.015 * (nrow - 1))  # 动态向上移动避免遮挡文本
    )

    # Display statistical information below the legend
    plt.figtext(
        0.5, 0.02,  # Adjust position, place below the legend
        stats_text,
        ha="center",
        fontsize=12,
        color="black"
    )

    # **Save or Show Plot**
    save_and_show_results(save_as, dpi)


def _compute_seqrf(seqdata: SequenceData, distance_matrix: np.ndarray, n_groups: int = 10,
                   weights: np.ndarray = None, grouping_method: str = "first"):
    """
    Compute the representative sequences (medoids) for each frequency group in a SequenceData object.

    :param seqdata: A SequenceData object.
    :param distance_matrix: A 2D pairwise distance matrix.
    :param n_groups: The number of frequency groups to divide sequences into.
    :param weights: Optional weight vector for sequences.
    :param grouping_method: Grouping method, either "first" (equal size) or "prop" (weighted).

    :return: (Tuple[np.ndarray, np.ndarray, np.ndarray])
        - rep_sequences: Representative sequences (medoids) for each group.
        - dissimilarities: Distances of sequences in each group to their respective medoid.
        - group_labels: Group assignments for each sequence.
    """
    n_sequences = seqdata.values.shape[0]
    if weights is None:
        weights = np.ones(n_sequences)  # Default to equal weights

    # **Step 1: Compute MDS using cmdscale()**
    mds_coords = _cmdscale(distance_matrix)  # Classic MDS
    mds_coords_1d = mds_coords[:, 0]  # Take only 1D result

    # **Step 2: Standardize MDS coordinates and sort**
    # 获取 StandardScaler
    scaler_class = _get_standard_scaler()
    if scaler_class is None:
        raise ImportError("需要 scikit-learn 来执行此功能。请安装: pip install scikit-learn")
    scaler = scaler_class()  # 实例化对象
    mds_coords_1d = scaler.fit_transform(mds_coords_1d.reshape(-1, 1)).flatten()

    # Eigenvector direction in np.linalg.eigh() may differ from R, causing cmdscale() to output reversed coordinates.
    mds_coords_1d = -mds_coords_1d  # Reverse direction
    sorted_indices = np.argsort(mds_coords_1d)  # Sort in ascending order
    sorted_coords = mds_coords_1d[sorted_indices]

    # **Step 3: Perform grouping based on different methods**
    if grouping_method == "first":
        # **Divide evenly, each group has an equal size**
        group_size = n_sequences // n_groups
        frequency_groups = [sorted_indices[i * group_size:(i + 1) * group_size] for i in range(n_groups)]
        if n_sequences % n_groups != 0:
            frequency_groups[-1] = np.append(frequency_groups[-1], sorted_indices[n_groups * group_size:])

    elif grouping_method == "prop":
        # **Divide based on weights**
        cumweights = np.cumsum(weights[sorted_indices])
        wsum = np.sum(weights)
        gsize = wsum / n_groups  # Target weight for each group

        frequency_groups = []
        start_idx = 0
        for i in range(n_groups):
            if i == n_groups - 1:
                group = sorted_indices[start_idx:]  # Last group includes remaining data
            else:
                end_idx = np.searchsorted(cumweights, (i + 1) * gsize)  # Find group boundary
                group = sorted_indices[start_idx:end_idx]
                start_idx = end_idx
            frequency_groups.append(group)

    else:
        raise ValueError("Invalid grouping_method! Use 'first' or 'prop'.")

    # **Step 4: Compute the medoid for each group**
    medoid_indices = np.array([
        _compute_group_medoid(distance_matrix, group, weights[group]) for group in frequency_groups
    ])
    rep_sequences = seqdata.values[medoid_indices]

    # **Step 5: Compute distances to medoid for each group**
    dissimilarities = [
        distance_matrix[np.ix_(group, [medoid_idx])].flatten() for group, medoid_idx in
        zip(frequency_groups, medoid_indices)
    ]

    # **Step 6: Assign group labels**
    group_labels = np.zeros(n_sequences)
    for i, group in enumerate(frequency_groups):
        group_labels[group] = i

    return rep_sequences, dissimilarities, group_labels


def _cmdscale(D):
    """
    Classic Multidimensional Scaling (MDS), equivalent to R's cmdscale()
    How Traminer uses cmdscale(): https://github.com/cran/TraMineR/blob/master/R/dissrf.R

    :param D: A NxN symmetric distance matrix
    :return: Y, a Nxd coordinate matrix, where d is the largest positive eigenvalues' count
    """
    n = len(D)

    # Step 1: Compute the centering matrix
    H = np.eye(n) - np.ones((n, n)) / n

    # Step 2: Compute the double centered distance matrix
    B = -0.5 * H @ (D ** 2) @ H

    # Step 3: Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(B)

    # Step 4: Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Step 5: Select only positive eigenvalues
    w, = np.where(eigvals > 0)
    L = np.diag(np.sqrt(eigvals[w]))
    L = np.diag(np.sqrt(eigvals[w]))
    V = eigvecs[:, w]

    return V @ L  # Return the MDS coordinates


def _compute_group_medoid(distance_matrix: np.ndarray, group_indices: np.ndarray, weights: np.ndarray = None) -> int:
    """Compute the weighted medoid of a given frequency group,
    matching R's disscenter() implementation.

    :param distance_matrix: (np.ndarray) A 2D symmetric pairwise distance matrix.
    :param group_indices: (np.ndarray) An array of indices representing the sequences in the group.
    :param weights: (np.ndarray, optional) A weight vector for sequences. Defaults to equal weights if not provided.

    :return: (int)
        The index of the medoid sequence, which has the minimum weighted sum of distances within the group.
    """
    group_distances = distance_matrix[np.ix_(group_indices, group_indices)]

    if weights is None:
        weights = np.ones(len(group_indices))  # Default to equal weights

    # **Fix: Compute the weighted sum of distances**
    total_distances = np.sum(group_distances * weights[:, np.newaxis], axis=1)

    # **Fix: Select the medoid with the minimum weighted distance**
    return group_indices[np.argmin(total_distances)]


def _compute_r2_f_statistic(distance_matrix: np.ndarray, group_labels: np.ndarray):
    """
    Compute the pseudo R² and F-statistic for sequence frequency grouping.
    :param distance_matrix: (np.ndarray) A 2D pairwise distance matrix.
    """
    unique_groups = np.unique(group_labels)
    total_var = np.var(distance_matrix)

    group_means = np.array([np.mean(distance_matrix[group_labels == g]) for g in unique_groups])
    within_group_vars = np.array([np.var(distance_matrix[group_labels == g]) for g in unique_groups])

    ss_between = sum(len(distance_matrix[group_labels == g]) * (mean - np.mean(distance_matrix)) ** 2
                     for g, mean in zip(unique_groups, group_means))
    ss_within = sum(within_group_vars)

    # Ensure valid ANOVA conditions
    valid_groups = [distance_matrix[group_labels == g].flatten() for g in unique_groups if
                    np.sum(group_labels == g) > 1]
    if len(valid_groups) > 1:
        f_statistic, p_value = f_oneway(*valid_groups)
    else:
        f_statistic, p_value = np.nan, np.nan

    r_squared = float(ss_between / total_var) if total_var > 0 else 0.0
    return r_squared, float(f_statistic), float(p_value)
