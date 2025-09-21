"""
@Author  : Yuqi Liang 梁彧祺
@File    : plot_most_frequent_sequences.py
@Time    : 12/02/2025 10:40
@Desc    :
    Generate sequence frequency plots.

    This script plots the 10 most frequent sequences,
    similar to `seqfplot` in R's TraMineR package.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sequenzo.define_sequence_data import SequenceData
from sequenzo.visualization.utils import (
    set_up_time_labels_for_x_axis,
    save_and_show_results
)


def plot_most_frequent_sequences(seqdata: SequenceData, top_n: int = 10, weights="auto", save_as=None, dpi=200):
    """
    Generate a sequence frequency plot, similar to R's seqfplot.

    :param seqdata: (SequenceData) A SequenceData object containing sequences.
    :param top_n: (int) Number of most frequent sequences to display.
    :param weights: (np.ndarray or "auto") Weights for sequences. If "auto", uses seqdata.weights if available
    :param save_as: (str, optional) Path to save the plot.
    :param dpi: (int) Resolution of the saved plot.
    """
    sequences = seqdata.values.tolist()
    
    # Process weights
    if isinstance(weights, str) and weights == "auto":
        weights = getattr(seqdata, "weights", None)
    
    if weights is not None:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(weights) != len(seqdata.values):
            raise ValueError("Length of weights must equal number of sequences.")
    
    if weights is None:
        weights = np.ones(len(sequences))

    # Weighted counting of sequences
    agg = {}
    for seq, w in zip(sequences, weights):
        key = tuple(seq)
        agg[key] = agg.get(key, 0.0) + float(w)

    # Select Top-N by weighted frequency
    items = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    df = pd.DataFrame(items, columns=['sequence', 'wcount'])
    totw = float(np.sum(weights))
    df['freq'] = df['wcount'] / (totw if totw > 0 else 1.0) * 100.0

    # **Ensure colors match seqdef**
    inv_state_mapping = {v: k for k, v in seqdata.state_mapping.items()}  # Reverse mapping from numeric values to state names

    # **Plot settings**
    fig, ax = plt.subplots(figsize=(10, 6))

    # **Adjust y_positions calculation to ensure sequences fill the entire y-axis**
    y_positions = df['freq'].cumsum() - df['freq'] / 2  # Center the bars

    for i, (seq, freq) in enumerate(zip(df['sequence'], df['freq'])):
        left = 0  # Starting x position
        for t, state_idx in enumerate(seq):
            label = inv_state_mapping.get(state_idx, "Unknown")  # Get the actual state name
            color = seqdata.color_map_by_label.get(label, "gray")  # Get the corresponding color

            width = 1  # Width of each time slice
            ax.barh(y=y_positions[i], width=width * 1.01, left=left - 0.005,
                    height=freq, color=color, linewidth=0,
                    antialiased=False)
            left += width  # Move to the next time slice

    # **Formatting**
    ax.set_xlabel("Time", fontsize=12)
    if weights is not None and not np.allclose(weights, 1.0):
        # Show both count and weighted total if weights are used
        ax.set_ylabel("Cumulative Frequency (%)\nN={:,}, Σw={:.1f}".format(len(sequences), totw), fontsize=12)
    else:
        ax.set_ylabel("Cumulative Frequency (%)\nN={:,}".format(len(sequences)), fontsize=12)
    ax.set_title(f"Top {top_n} Most Frequent Sequences", fontsize=14, pad=20)  # Add some padding between title and plot

    # **Optimize X-axis ticks: align to the center of each bar**
    set_up_time_labels_for_x_axis(seqdata, ax)

    # **Set Y-axis ticks and labels**
    sum_freq_top_10 = df['freq'].sum()  # Cumulative frequency of top 10 sequences
    max_freq = df['freq'].max()  # Frequency of the top 1 sequence

    # Set Y-axis ticks: 0%, top1 frequency, top10 cumulative frequency
    y_ticks = [0, max_freq, sum_freq_top_10]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{ytick:.1f}%" for ytick in y_ticks], fontsize=10)

    # **Set Y-axis range to ensure the highest tick is the top10 cumulative frequency**
    # Force Y-axis range to be from 0 to sum_freq_top_10
    ax.set_ylim(0, sum_freq_top_10)

    # **Annotate the frequency percentage on the left side of the highest frequency sequence**
    ax.annotate(f"{max_freq:.1f}%", xy=(-0.5, y_positions.iloc[0]),
                xycoords="data", fontsize=12, color="black", ha="left", va="center")

    # **Annotate 0% at the bottom of the Y-axis**
    ax.annotate("0%", xy=(-0.5, 0), xycoords="data", fontsize=12, color="black", ha="left", va="center")

    # **Remove top, right, and left borders, keep only the x-axis and y-axis**
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)  # Do not keep the left border
    ax.spines["bottom"].set_visible(False)  # Do not keep the bottom border

    # Use legend from SequenceData
    ax.legend(*seqdata.get_legend(), bbox_to_anchor=(1.05, 1), loc='upper left')

    save_and_show_results(save_as, dpi=200)


