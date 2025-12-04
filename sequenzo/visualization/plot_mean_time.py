"""
@Author  : Yuqi Liang 梁彧祺
@File    : plot_mean_time.py
@Time    : 14/02/2025 10:12
@Desc    :
    Implementation of Mean Time Plot for social sequence analysis,
    closely following ggseqplot's `ggseqmtplot` function,
    and TraMineR's `plot.stslist.meant.Rd` for mean time calculation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from sequenzo.define_sequence_data import SequenceData
from sequenzo.visualization.utils import (
    create_standalone_legend,
    combine_plot_with_legend,
    save_and_show_results,
    show_plot_title,
    save_figure_to_buffer
)


def _compute_mean_time(seqdata: SequenceData, weights="auto") -> pd.DataFrame:
    """
    Compute mean total time spent in each state across all sequences.
    Optimized version using pandas operations.

    :param seqdata: SequenceData object containing sequence information
    :param weights: (np.ndarray or "auto") Weights for sequences. If "auto", uses seqdata.weights if available
    :return: DataFrame with mean time spent and standard error for each state
    """
    # Process weights
    if isinstance(weights, str) and weights == "auto":
        weights = getattr(seqdata, "weights", None)
    
    if weights is not None:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(weights) != len(seqdata.values):
            raise ValueError("Length of weights must equal number of sequences.")
    
    # Get data and preprocess
    seq_df = seqdata.to_dataframe()
    inv = {v: k for k, v in seqdata.state_mapping.items()}
    states = list(range(1, len(seqdata.states) + 1))  # Use numerical state indices
    n = len(seq_df)

    # Get weights
    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, dtype=float)

    # Broadcast weights to each time point
    W = np.repeat(w[:, None], seq_df.shape[1], axis=1)

    # Convert to long format with weights
    df_long = seq_df.melt(value_name='state_idx')
    # Replicate weights for each time point
    W_long = pd.DataFrame(W, columns=seq_df.columns).melt(value_name='w')['w'].to_numpy()
    df_long['w'] = W_long
    # Keep state_idx as numerical for consistent grouping
    
    # Calculate mean time spent in each state per sequence
    # For each sequence, count time spent in each state, then take weighted average
    seq_state_times = {}
    for s in states:
        # Count occurrences of state s in each sequence
        state_counts = (seq_df == s).sum(axis=1)  # Sum across time for each sequence
        # Calculate weighted mean across sequences
        seq_state_times[s] = np.average(state_counts, weights=w) if len(state_counts) > 0 else 0.0
    
    mean_times = seq_state_times

    # Calculate standard errors for mean time
    se = {}
    n_sequences = len(seq_df)
    
    for s in states:
        if n_sequences > 1:
            # Count occurrences of state s in each sequence
            state_counts = (seq_df == s).sum(axis=1)
            # Calculate weighted standard error
            weighted_mean = seq_state_times[s]
            weighted_var = np.average((state_counts - weighted_mean) ** 2, weights=w)
            # Standard error of the weighted mean
            se[s] = np.sqrt(weighted_var / n_sequences) if weighted_var >= 0 else 0.0
        else:
            se[s] = 0.0

    # Create result DataFrame
    mean_time_df = pd.DataFrame({
        'State': [inv[s] for s in states],
        'MeanTime': [mean_times[s] for s in states],
        'StandardError': [se[s] for s in states]
    })

    mean_time_df.sort_values(by='MeanTime', ascending=True, inplace=True)

    return mean_time_df


def plot_mean_time(seqdata: SequenceData,
                   weights="auto",
                   show_error_bar: bool = True,
                   title=None,
                   x_label="Mean Time",
                   y_label="State",
                   fontsize: int = 12,
                   save_as: Optional[str] = None,
                   dpi: int = 200) -> None:
    """
    Plot Mean Time Plot for sequence data with clean white background.

    :param seqdata: SequenceData object containing sequence information
    :param weights: (np.ndarray or "auto") Weights for sequences. If "auto", uses seqdata.weights if available
    :param show_error_bar: Boolean flag to show or hide error bars
    :param title: Optional title for the plot
    :param x_label: Label for the x-axis
    :param y_label: Label for the y-axis
    :param save_as: Optional file path to save the plot
    :param dpi: Resolution of the saved plot
    """
    # Use default style as base
    plt.style.use('default')

    # Compute all required data at once
    mean_time_df = _compute_mean_time(seqdata, weights)

    # Create figure and preallocate memory
    fig = plt.figure(figsize=(12, 7))

    # Create main plot
    ax = plt.subplot(111)

    # Get color mapping - use original colors without enhancement
    cmap = seqdata.get_colormap()
    colors = [cmap.colors[i] for i in range(len(seqdata.states))]

    # Assign colors to states (without enhancing saturation)
    mean_time_df['Color'] = pd.Categorical(mean_time_df['State']).codes
    mean_time_df['Color'] = mean_time_df['Color'].map(lambda x: colors[x])

    # Create custom barplot
    for i, (_, row) in enumerate(mean_time_df.iterrows()):
        ax.barh(y=i, width=row['MeanTime'], height=0.7,
                color=row['Color'], edgecolor='white', linewidth=0.5)

    # Set y-axis ticks and labels
    ax.set_yticks(range(len(mean_time_df)))
    ax.set_yticklabels(mean_time_df['State'], fontsize=fontsize-2)

    # Add error bars if needed
    if show_error_bar:
        ax.errorbar(
            x=mean_time_df["MeanTime"],
            y=range(len(mean_time_df)),
            xerr=mean_time_df["StandardError"],
            fmt='none',
            ecolor='black',
            capsize=3,
            capthick=1,
            elinewidth=1.5
        )

    # Set plot properties
    if title:
        show_plot_title(ax, title, show=True, fontsize=fontsize+2, fontweight='bold', pad=20)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize, labelpad=15)

    # Clean white background with light grid
    ax.set_facecolor('white')
    ax.grid(axis='x', color='#E0E0E0', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)  # Place grid lines behind the bars

    # Customize borders
    for spine in ax.spines.values():
        spine.set_color('#CCCCCC')  # Light gray border
        spine.set_linewidth(0.5)

    # Adjust layout(1/2)
    plt.subplots_adjust(left=0.3)

    # Add a note about normalization
    relative_threshold = 0.01
    max_val = mean_time_df['MeanTime'].max()
    too_many_small = np.sum(mean_time_df['MeanTime'] < relative_threshold * max_val) >= 1
    if too_many_small:
        norm_note = f"Note: Some bars may appear as zero, but actually have small non-zero values."
        plt.figtext(0.5, -0.02, norm_note, ha='center', fontsize=fontsize-2, style='italic')

    # Adjust layout before saving
    plt.tight_layout()

    # Get colors for legend - use color_map_by_label like in plot_modal_state
    colors_for_legend = seqdata.color_map_by_label

    # Save main figure to memory
    main_buffer = save_figure_to_buffer(fig, dpi=dpi)

    # Create standalone legend
    legend_buffer = create_standalone_legend(
        colors=colors_for_legend,
        labels=seqdata.labels,
        ncol=min(5, len(seqdata.states)),
        figsize=(12, 1),
        fontsize=fontsize - 2,
        dpi=dpi
    )

    # Handle save_as extension
    if save_as and not save_as.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
        save_as = save_as + '.png'

    # Combine main plot with legend
    combined_img = combine_plot_with_legend(
        main_buffer,
        legend_buffer,
        output_path=save_as,
        dpi=dpi,
        padding=20  # Increased padding between plot and legend
    )

    # Display combined image
    plt.figure(figsize=(12, 8))
    plt.imshow(combined_img)
    plt.axis('off')
    plt.show()
    plt.close()

