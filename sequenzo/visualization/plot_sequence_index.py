"""
@Author  : Yuqi Liang 梁彧祺
@File    : plot_sequence_index.py
@Time    : 29/12/2024 09:08
@Desc    : 
    Generate sequence index plots.
"""
import numpy as np
import matplotlib.pyplot as plt

from sequenzo import SequenceData
from sequenzo.visualization.utils import (
    set_up_time_labels_for_x_axis,
    save_figure_to_buffer,
    create_standalone_legend,
    combine_plot_with_legend,
    save_and_show_results,
    determine_layout
)


def sort_sequences_by_structure(seqdata, method="first_marriage", target_state=3, mask=None):
    """
    根据结构信息对 SequenceData 中的序列排序。

    :param seqdata: SequenceData object
    :param method: str, 排序方式
    :param target_state: int, first_marriage 时的目标状态
    :param mask: np.array(bool), 若提供则只对该子集排序
    :return: np.array 排序索引（相对于原始顺序）
    """
    values = seqdata.values.copy()
    time_points = np.arange(values.shape[1])

    if mask is not None:
        values = values[mask]

    if method == "first_marriage":
        sort_keys = np.array([
            np.where(row == target_state)[0][0] if target_state in row else 99
            for row in values
        ])
    elif method == "transition_count":
        sort_keys = np.sum(values[:, 1:] != values[:, :-1], axis=1)
    elif method == "final_state":
        sort_keys = values[:, -1]
    elif method == "happiness_slope":
        sort_keys = np.array([
            np.polyfit(time_points, row, 1)[0] for row in values
        ])
    else:
        raise ValueError(f"Unsupported method: {method}")

    return np.argsort(sort_keys)


def plot_sequence_index(seqdata: SequenceData,
                        id_group_df=None,
                        categories=None,
                        sort_by=None,
                        figsize=(10, 6),
                        title=None,
                        xlabel="Time",
                        ylabel="Sequences",
                        save_as=None,
                        dpi=200,
                        layout='column',
                        nrows: int = None,
                        ncols: int = None
                        ):
    """
    Creates sequence index plots, optionally grouped by categories.

    :param seqdata: SequenceData object containing sequence information
    :param id_group_df: DataFrame with entity IDs and group information (if None, creates a single plot)
    :param categories: Column name in id_group_df that contains grouping information
    :param figsize: Size of each subplot figure
    :param title: Title for the plot (if None, default titles will be used)
    :param xlabel: Label for the x-axis
    :param ylabel: Label for the y-axis
    :param save_as: File path to save the plot (if None, plot will be shown)
    :param dpi: DPI for saved image
    :param layout: Layout style - 'column' (default, 3xn), 'grid' (nxn)
    """
    # If no grouping information, create a single plot
    if id_group_df is None or categories is None:
        return _sequence_index_plot_single(seqdata, figsize, title, xlabel, ylabel, save_as, dpi)

    # Ensure ID columns match (convert if needed)
    id_col_name = "Entity ID" if "Entity ID" in id_group_df.columns else id_group_df.columns[0]

    # Get unique groups and sort them
    groups = sorted(id_group_df[categories].unique())
    num_groups = len(groups)

    # Calculate figure size and layout based on number of groups and specified layout
    nrows, ncols = determine_layout(num_groups, layout=layout, nrows=nrows, ncols=ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize[0] * ncols, figsize[1] * nrows),
        gridspec_kw={'wspace': 0.2, 'hspace': 0.3}
    )
    axes = axes.flatten()

    # Create a plot for each group
    for i, group in enumerate(groups):
        # Get IDs for this group
        group_ids = id_group_df[id_group_df[categories] == group][id_col_name].values

        # Match IDs with sequence data
        mask = np.isin(seqdata.ids, group_ids)
        if not np.any(mask):
            print(f"Warning: No matching sequences found for group '{group}'")
            continue

        # Extract sequences for this group
        group_sequences = seqdata.values[mask]

        # Sort sequences for better visualization
        if np.isnan(group_sequences).any():
            group_sequences = np.where(np.isnan(group_sequences), -1, group_sequences)

        if sort_by is not None:
            sorted_indices = sort_sequences_by_structure(seqdata=seqdata, method=sort_by, mask=mask)
        else:
            sorted_indices = np.lexsort(group_sequences.T[::-1])

        sorted_data = group_sequences[sorted_indices]

        # Plot on the corresponding axis
        ax = axes[i]
        im = ax.imshow(sorted_data, aspect='auto', cmap=seqdata.get_colormap(),
                       interpolation='nearest', vmin=1, vmax=len(seqdata.states))

        # Remove grid lines
        ax.grid(False)

        # Set up time labels
        set_up_time_labels_for_x_axis(seqdata, ax)

        # Enhance y-axis aesthetics
        num_sequences = sorted_data.shape[0]
        ytick_spacing = max(1, num_sequences // 10)

        ax.set_yticks(np.arange(0, num_sequences, step=ytick_spacing))
        ax.set_yticklabels(np.arange(1, num_sequences + 1, step=ytick_spacing), fontsize=10, color='black')

        # Customize axis style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        ax.spines['left'].set_linewidth(0.7)
        ax.spines['bottom'].set_linewidth(0.7)
        ax.tick_params(axis='x', colors='gray', length=4, width=0.7)
        ax.tick_params(axis='y', colors='gray', length=4, width=0.7)

        # Add group title
        group_title = f"{categories} {group} (n = {num_sequences})"
        ax.set_title(group_title, fontsize=12, loc='right')

        # Add axis labels
        if i % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=12, labelpad=10, color='black')

        if i >= num_groups - ncols:
            ax.set_xlabel(xlabel, fontsize=12, labelpad=10, color='black')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Add a common title if provided
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    # Adjust layout to remove tight_layout warning
    fig.subplots_adjust(wspace=0.2, hspace=0.3, bottom=0.1, top=0.9, right=0.9)

    # Save main figure to memory
    main_buffer = save_figure_to_buffer(fig, dpi=dpi)

    # Create standalone legend
    colors = seqdata.color_map_by_label
    legend_buffer = create_standalone_legend(
        colors=colors,
        labels=seqdata.labels,
        ncol=min(5, len(seqdata.states)),
        figsize=(figsize[0] * ncols, 1),
        fontsize=10,
        dpi=dpi
    )

    # Combine plot with legend
    if save_as and not save_as.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
        save_as = save_as + '.png'

    combined_img = combine_plot_with_legend(
        main_buffer,
        legend_buffer,
        output_path=save_as,
        dpi=dpi,
        padding=20
    )

    # Display combined image
    plt.figure(figsize=(figsize[0] * ncols, figsize[1] * nrows + 1))
    plt.imshow(combined_img)
    plt.axis('off')
    plt.show()
    plt.close()


def _sequence_index_plot_single(seqdata: SequenceData,
                                figsize=(10, 6),
                                title=None,
                                xlabel="Time",
                                ylabel="Sequences",
                                save_as=None,
                                dpi=200):
    """
    Efficiently creates a sequence index plot using `imshow` for faster rendering.

    :param data: (np.array or pd.DataFrame) 2D array where rows are sequences and columns are time points.
    :param num_colors: (int): Number of colors in the Spectral palette.
    :param reverse_colors: (bool): Whether to reverse the color scheme.
    :param figsize: (tuple): Size of the figure.
    :param xlabel: (str): Label for the x-axis.
    :param ylabel: (str): Label for the y-axis.
    :param title: (str): Title for the plot.

    :return None.
    """
    # Get sequence values as NumPy array
    sequence_values = seqdata.values.copy()

    # Ensure no NaN values interfere with sorting
    if np.isnan(sequence_values).any():
        sequence_values = np.where(np.isnan(sequence_values), -1, sequence_values)

    # Sort sequences lexicographically for better visualization
    sorted_indices = np.lexsort(sequence_values.T[::-1])
    sorted_data = sequence_values[sorted_indices]

    # Create the plot using imshow
    fig, ax = plt.subplots(figsize=figsize)
    # ax.imshow(sorted_data, aspect='auto', cmap=seqdata.get_colormap(), interpolation='nearest')
    ax.imshow(sorted_data, aspect='auto', cmap=seqdata.get_colormap(), interpolation='nearest', vmin=1,
              vmax=len(seqdata.states))

    # Disable background grid and all axis guide lines
    ax.grid(False)

    # Optional: remove tick marks and tick labels to avoid visual grid effects
    # ax.set_xticks([])
    # ax.set_yticks([])

    # x label
    set_up_time_labels_for_x_axis(seqdata, ax)

    # Enhance y-axis aesthetics
    num_sequences = sorted_data.shape[0]
    ytick_spacing = max(1, num_sequences // 10)

    ax.set_yticks(np.arange(0, num_sequences, step=ytick_spacing))
    ax.set_yticklabels(np.arange(1, num_sequences + 1, step=ytick_spacing), fontsize=10, color='black')

    # Enhance y-axis aesthetics
    ax.set_yticks(range(0, len(sorted_data), max(1, len(sorted_data) // 10)))
    ax.set_yticklabels(
        range(1, len(sorted_data) + 1, max(1, len(sorted_data) // 10)),
        fontsize=10,
        color='black'
    )

    # Customize axis line styles and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_linewidth(0.7)
    ax.tick_params(axis='x', colors='gray', length=4, width=0.7)
    ax.tick_params(axis='y', colors='gray', length=4, width=0.7)

    # Add labels and title
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10, color='black')

    ax.set_ylabel(ylabel, fontsize=12, labelpad=10, color='black')
    if title:
        ax.set_title(title, fontsize=14, color='black')

    # Use legend from SequenceData
    ax.legend(*seqdata.get_legend(), bbox_to_anchor=(1.05, 1), loc='upper left')

    save_and_show_results(save_as, dpi=200)