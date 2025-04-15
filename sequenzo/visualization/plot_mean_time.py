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


def _compute_mean_time(seqdata: SequenceData) -> pd.DataFrame:
    """
    Compute mean total time spent in each state across all sequences.
    Optimized version using pandas operations.

    :param seqdata: SequenceData object containing sequence information
    :return: DataFrame with mean time spent and standard error for each state
    """
    # Get data and preprocess
    seq_df = seqdata.to_dataframe()
    state_mapping = {v: k for k, v in seqdata.state_mapping.items()}
    states = seqdata.states
    n_sequences = len(seq_df)

    # Convert data to long format for easier aggregation
    df_long = seq_df.melt(value_name='state_idx')
    df_long['state'] = df_long['state_idx'].map(state_mapping)

    # Calculate total counts and mean time for each state
    state_counts = df_long['state'].value_counts()
    mean_times = state_counts / n_sequences

    # Use groupby to calculate state occurrences per sequence in one go
    state_counts_per_seq = df_long.groupby(['state'])['variable'].value_counts().unstack(fill_value=0)

    # Calculate standard errors
    std_errors = state_counts_per_seq.std(ddof=1) / np.sqrt(n_sequences)

    # Create result DataFrame
    mean_time_df = pd.DataFrame({
        'State': states,
        'MeanTime': [mean_times.get(state, 0) for state in states],
        'StandardError': [std_errors.get(state, 0) for state in states]
    })

    mean_time_df.sort_values(by='MeanTime', ascending=True, inplace=True)

    return mean_time_df


def plot_mean_time(seqdata: SequenceData,
                   show_error_bar: bool = True,
                   title=None,
                   x_label="Mean Time (Years)",
                   y_label="State",
                   save_as: Optional[str] = None,
                   dpi: int = 200) -> None:
    """
    Plot Mean Time Plot for sequence data with clean white background.

    :param seqdata: SequenceData object containing sequence information
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
    mean_time_df = _compute_mean_time(seqdata)

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
    ax.set_yticklabels(mean_time_df['State'], fontsize=10)

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
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    # Clean white background with light grid
    ax.set_facecolor('white')
    ax.grid(axis='x', color='#E0E0E0', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)  # Place grid lines behind the bars

    # Customize borders
    for spine in ax.spines.values():
        spine.set_color('#CCCCCC')  # Light gray border
        spine.set_linewidth(0.5)

    # Adjust layout
    plt.subplots_adjust(left=0.3)
    plt.tight_layout()

    save_and_show_results(save_as, dpi=200)

