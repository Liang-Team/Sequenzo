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
    save_and_show_results
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
    states = seqdata.states
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
    df_long['state'] = df_long['state_idx'].map(inv)

    # Calculate weighted time for each state
    wtime = df_long.groupby('state')['w'].sum()
    totw = float(w.sum())

    # Calculate proportions first  
    proportions = {s: float(wtime.get(s, 0.0)) / (totw if totw > 0 else 1.0)
                   for s in states}
    
    # Scale to time units (number of periods)
    T = seqdata.values.shape[1]  # Number of time periods
    mean_times = {s: T * proportions[s] for s in states}

    # Calculate effective sample size and standard errors in time units
    Neff = (w.sum() ** 2) / (np.sum(w ** 2)) if np.sum(w ** 2) > 0 else 1.0
    se = {s: (T * np.sqrt(proportions[s]*(1-proportions[s])/Neff) if Neff > 1 else 0.0) for s in states}

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
                   x_label="Mean Time (Periods)",
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
        ax.set_title(title, fontsize=fontsize+2, fontweight='bold', pad=20)
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

    # Adjust layout(2/2)
    plt.tight_layout()

    save_and_show_results(save_as, dpi=200)

