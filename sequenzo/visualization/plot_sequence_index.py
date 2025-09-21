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


def smart_sort_groups(groups):
    """
    Smart sorting: prioritize numeric prefix, fallback to string sorting
    
    :param groups: List of group names
    :return: Sorted list of group names
    """
    import re
    
    def sort_key(item):
        match = re.match(r'^(\d+)', str(item))
        return (int(match.group(1)), str(item)) if match else (float('inf'), str(item))
    
    return sorted(groups, key=sort_key)


def sort_sequences_by_structure(seqdata, method="first_marriage", target_state=3, mask=None):
    """
    Sort sequences in SequenceData based on structural information.

    :param seqdata: SequenceData object
    :param method: str, sorting method
    :param target_state: int, target state for first_marriage method
    :param mask: np.array(bool), if provided, sort only this subset
    :return: np.array sorting indices (relative to original order)
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
                        sort_by_weight=False,
                        weights="auto",
                        figsize=(10, 6),
                        title=None,
                        xlabel="Time",
                        ylabel="Sequences",
                        save_as=None,
                        dpi=200,
                        layout='column',
                        nrows: int = None,
                        ncols: int = None,
                        group_order=None,
                        sort_groups='auto',
                        fontsize=12
                        ):
    """
    Creates sequence index plots, optionally grouped by categories.

    :param seqdata: SequenceData object containing sequence information
    :param id_group_df: DataFrame with entity IDs and group information (if None, creates a single plot)
    :param categories: Column name in id_group_df that contains grouping information
    :param sort_by: Sorting method for sequences within groups
    :param sort_by_weight: If True, sort sequences by weight (descending), overrides other sorting
    :param weights: (np.ndarray or "auto") Weights for sequences. If "auto", uses seqdata.weights if available
    :param figsize: Size of each subplot figure
    :param title: Title for the plot (if None, default titles will be used)
    :param xlabel: Label for the x-axis
    :param ylabel: Label for the y-axis
    :param save_as: File path to save the plot (if None, plot will be shown)
    :param dpi: DPI for saved image
    :param layout: Layout style - 'column' (default, 3xn), 'grid' (nxn)
    :param group_order: List, manually specify group order (overrides sort_groups)
    :param sort_groups: String, sorting method: 'auto'(smart numeric), 'numeric'(numeric prefix), 'alpha'(alphabetical), 'none'(original order)
    :param fontsize: Base font size for text elements (titles use fontsize+2, ticks use fontsize-2)
    """
    # If no grouping information, create a single plot
    if id_group_df is None or categories is None:
        return _sequence_index_plot_single(seqdata, sort_by_weight, weights, figsize, title, xlabel, ylabel, save_as, dpi, fontsize)

    # Process weights
    if isinstance(weights, str) and weights == "auto":
        weights = getattr(seqdata, "weights", None)
    
    if weights is not None:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(weights) != len(seqdata.values):
            raise ValueError("Length of weights must equal number of sequences.")
    
    # Ensure ID columns match (convert if needed)
    id_col_name = "Entity ID" if "Entity ID" in id_group_df.columns else id_group_df.columns[0]

    # Get unique groups and sort them based on user preference
    if group_order:
        # Use manually specified order, filter out non-existing groups
        groups = [g for g in group_order if g in id_group_df[categories].unique()]
        missing_groups = [g for g in id_group_df[categories].unique() if g not in group_order]
        if missing_groups:
            print(f"[Warning] Groups not in group_order will be excluded: {missing_groups}")
    elif sort_groups == 'numeric' or sort_groups == 'auto':
        groups = smart_sort_groups(id_group_df[categories].unique())
    elif sort_groups == 'alpha':
        groups = sorted(id_group_df[categories].unique())
    elif sort_groups == 'none':
        groups = list(id_group_df[categories].unique())
    else:
        raise ValueError(f"Invalid sort_groups value: {sort_groups}. Use 'auto', 'numeric', 'alpha', or 'none'.")
    
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
        
        # Get weights for this group
        if weights is not None:
            group_weights = weights[mask]
        else:
            group_weights = None

        # Handle NaN values for better visualization
        if np.isnan(group_sequences).any():
            # Map NaN to a dedicated state code with proper masking
            group_sequences = group_sequences.astype(float)
            group_sequences[np.isnan(group_sequences)] = np.nan

        if sort_by_weight and group_weights is not None:
            # Sort by weight (descending), then by structure as secondary
            sorted_indices = np.argsort(-group_weights)
        elif sort_by is not None:
            sorted_indices = sort_sequences_by_structure(seqdata=seqdata, method=sort_by, mask=mask)
        else:
            sorted_indices = np.lexsort(group_sequences.T[::-1])

        sorted_data = group_sequences[sorted_indices]

        # Plot on the corresponding axis
        ax = axes[i]
        # Use masked array for better NaN handling
        data = sorted_data.astype(float)
        data[data < 1] = np.nan
        im = ax.imshow(np.ma.masked_invalid(data), aspect='auto', cmap=seqdata.get_colormap(),
                       interpolation='nearest', vmin=1, vmax=len(seqdata.states))

        # Remove grid lines
        ax.grid(False)

        # Set up time labels
        set_up_time_labels_for_x_axis(seqdata, ax)

        # Enhance y-axis aesthetics
        num_sequences = sorted_data.shape[0]
        ytick_spacing = max(1, num_sequences // 10)

        ax.set_yticks(np.arange(0, num_sequences, step=ytick_spacing))
        ax.set_yticklabels(np.arange(1, num_sequences + 1, step=ytick_spacing), fontsize=fontsize-2, color='black')

        # Customize axis style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        ax.spines['left'].set_linewidth(0.7)
        ax.spines['bottom'].set_linewidth(0.7)
        ax.tick_params(axis='x', colors='gray', length=4, width=0.7)
        ax.tick_params(axis='y', colors='gray', length=4, width=0.7)

        # Add group title with weight information
        # Check if we have effective weights (not all 1.0) and they were provided by user
        original_weights = getattr(seqdata, "weights", None)
        if original_weights is not None and not np.allclose(original_weights, 1.0) and group_weights is not None:
            sum_w = float(group_weights.sum())
            group_title = f"{group} (n = {num_sequences}, Σw = {sum_w:.1f})"
        else:
            group_title = f"{group} (n = {num_sequences})"
        ax.set_title(group_title, fontsize=fontsize, loc='right')

        # Add axis labels
        if i % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=10, color='black')

        # if i >= num_groups - ncols:
        ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=10, color='black')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Add a common title if provided
    if title:
        fig.suptitle(title, fontsize=fontsize+2, y=1.02)

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
        fontsize=fontsize-2,
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
                                sort_by_weight=False,
                                weights="auto",
                                figsize=(10, 6),
                                title=None,
                                xlabel="Time",
                                ylabel="Sequences",
                                save_as=None,
                                dpi=200,
                                fontsize=12):
    """
    Efficiently creates a sequence index plot using `imshow` for faster rendering.

    :param seqdata: SequenceData object containing sequence information
    :param sort_by_weight: If True, sort sequences by weight (descending)
    :param weights: (np.ndarray or "auto") Weights for sequences. If "auto", uses seqdata.weights if available
    :param figsize: (tuple): Size of the figure.
    :param title: (str): Title for the plot.
    :param xlabel: (str): Label for the x-axis.
    :param ylabel: (str): Label for the y-axis.
    :param save_as: File path to save the plot
    :param dpi: DPI for saved image

    :return None.
    """
    # Process weights
    if isinstance(weights, str) and weights == "auto":
        weights = getattr(seqdata, "weights", None)
    
    if weights is not None:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(weights) != len(seqdata.values):
            raise ValueError("Length of weights must equal number of sequences.")
    
    # Get sequence values as NumPy array
    sequence_values = seqdata.values.copy()

    # Handle NaN values for better visualization
    if np.isnan(sequence_values).any():
        # Keep NaN as float for proper masking
        sequence_values = sequence_values.astype(float)

    # Sort sequences for better visualization
    if sort_by_weight and weights is not None:
        # Sort by weight (descending)
        sorted_indices = np.argsort(-weights)
    else:
        # Sort sequences lexicographically for better visualization
        sorted_indices = np.lexsort(sequence_values.T[::-1])
    
    sorted_data = sequence_values[sorted_indices]

    # Create the plot using imshow with proper NaN handling
    fig, ax = plt.subplots(figsize=figsize)
    # Use masked array for better NaN handling
    data = sorted_data.astype(float)
    data[data < 1] = np.nan
    ax.imshow(np.ma.masked_invalid(data), aspect='auto', cmap=seqdata.get_colormap(), 
              interpolation='nearest', vmin=1, vmax=len(seqdata.states))

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
    ax.set_yticklabels(np.arange(1, num_sequences + 1, step=ytick_spacing), fontsize=fontsize-2, color='black')


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
    ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=10, color='black')

    ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=10, color='black')
    
    # Set title with weight information if available
    if title is not None:
        display_title = title
        
        # Check if we have effective weights (not all 1.0) and they were provided by user
        original_weights = getattr(seqdata, "weights", None)
        if original_weights is not None and not np.allclose(original_weights, 1.0) and weights is not None:
            sum_w = float(weights.sum())
            display_title += f" (n = {num_sequences}, Σw = {sum_w:.1f})"
        else:
            display_title += f" (n = {num_sequences})"
        
        ax.set_title(display_title, fontsize=fontsize+2, color='black')

    # Use legend from SequenceData
    ax.legend(*seqdata.get_legend(), bbox_to_anchor=(1.05, 1), loc='upper left')

    save_and_show_results(save_as, dpi=200)