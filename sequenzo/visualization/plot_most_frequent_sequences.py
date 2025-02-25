"""
@Author  : Yuqi Liang 梁彧祺
@File    : plot_most_frequent_sequences.py
@Time    : 12/02/2025 10:40
@Desc    :
    Generate sequence frequency plots.

    This script plots the 10 most frequent sequences,
    similar to `seqfplot` in R's TraMineR package.
"""
from collections import Counter
from io import BytesIO  # Import BytesIO for in-memory operations

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sequenzo.define_sequence_data import SequenceData


def plot_most_frequent_sequences(seqdata: SequenceData, top_n: int = 10, save_as=None, dpi=200):
    """
    Generate a sequence frequency plot, similar to R's seqfplot.

    :param seqdata: (SequenceData) A SequenceData object containing sequences.
    :param top_n: (int) Number of most frequent sequences to display.
    :param save_as: (str, optional) Path to save the plot.
    :param dpi: (int) Resolution of the saved plot.
    """
    sequences = seqdata.values.tolist()

    # Count sequence occurrences
    sequence_counts = Counter(tuple(seq) for seq in sequences)
    most_common = sequence_counts.most_common(top_n)

    # Convert to DataFrame for visualization
    df = pd.DataFrame(most_common, columns=['sequence', 'count'])
    total_sequences = len(sequences)  # Total number of sequences in the dataset
    df['freq'] = df['count'] / total_sequences * 100  # Convert to percentage based on the entire dataset

    # Infer x-axis labels dynamically based on sequence length
    sequence_length = len(df['sequence'].iloc[0])  # Get sequence length dynamically
    x_ticks = np.arange(sequence_length) + 0.5  # Align X-axis ticks to the center of bars

    # Use provided time labels if available, otherwise use generic "C1, C2, ..."
    if seqdata.var:
        x_labels = seqdata.cleaned_time
    else:
        x_labels = [f"{i + 1}" for i in range(sequence_length)]

    # **Ensure colors match seqdef**
    state_colors = seqdata.color_map  # Directly get the color mapping from seqdef
    inv_state_mapping = {v: k for k, v in seqdata.state_mapping.items()}  # Reverse mapping from numeric values to state names

    # **Plot settings**
    fig, ax = plt.subplots(figsize=(10, 6))

    # **Adjust y_positions calculation to ensure sequences fill the entire y-axis**
    y_positions = df['freq'].cumsum() - df['freq'] / 2  # Center the bars

    for i, (seq, freq) in enumerate(zip(df['sequence'], df['freq'])):
        left = 0  # Starting x position
        for t, state_idx in enumerate(seq):
            state_label = inv_state_mapping.get(state_idx, "Unknown")  # Get the actual state name
            color = state_colors.get(state_label, "gray")  # Get the corresponding color

            width = 1  # Width of each time slice
            ax.barh(y=y_positions[i], width=width, left=left, height=freq, color=color, edgecolor="none")
            left += width  # Move to the next time slice

    # **Formatting**
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Cumulative Frequency (%)\nN={:,}".format(total_sequences), fontsize=12)
    ax.set_title(f"Top {top_n} Most Frequent Sequences", fontsize=14, pad=20)  # Add some padding between title and plot

    # **Optimize X-axis ticks: align to the center of each bar**
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=10)

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

    ax.grid(axis='x', linestyle='--', alpha=0.5)

    # **Remove top, right, and left borders, keep only the x-axis and y-axis**
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)  # Do not keep the left border
    ax.spines["bottom"].set_visible(False)  # Do not keep the bottom border

    # **Save or show**
    if save_as:
        # Ensure the file format is correct
        if not save_as.lower().endswith(('.png', '.jpeg', '.jpg', '.pdf')):
            save_as += '.png'  # Default to saving as PNG format

        # Save the main plot to memory
        main_buffer = BytesIO()
        plt.savefig(main_buffer, format='png', dpi=dpi, bbox_inches='tight')
        plt.close(fig)  # Close the plot to free memory
        main_buffer.seek(0)  # Reset the pointer to the beginning of the file

        # Generate the legend and save it to memory
        legend_buffer = _plot_legend(seqdata, dpi=dpi)

        # Combine the main plot and legend
        _combine_images(main_buffer, legend_buffer, save_as, dpi=dpi)
    else:
        plt.show()


def _plot_legend(seqdata: SequenceData, dpi=200):
    """
    Generates a slim vertical legend for sequence state colors
    and returns it as an in-memory image.

    :param seqdata: (SequenceData) A SequenceData object containing sequences.
    :param dpi: (int) Resolution of the legend.
    :return: (BytesIO) In-memory image of the legend.
    """
    # Create the figure which is slim (narrow width) and vertical (tall height)
    fig, ax = plt.subplots(figsize=(2, 6))
    ax.legend(handles=seqdata.legend_handles, loc='center', title="States", fontsize=10, ncol=1)
    ax.axis('off')

    # Save the legend to memory
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
    # Close the plot to free memory
    plt.close(fig)
    # Reset the pointer to the beginning of the file
    buffer.seek(0)
    return buffer


def _combine_images(main_buffer, legend_buffer, output_path, dpi=200):
    """
    Combines the main plot and legend into a single image.

    :param main_buffer: (BytesIO) In-memory image of the main plot.
    :param legend_buffer: (BytesIO) In-memory image of the legend.
    :param output_path: (str) Path to save the combined image.
    :param dpi: (int) Resolution of the output image.
    """
    main_image = Image.open(main_buffer)
    legend_image = Image.open(legend_buffer)

    # Calculate the combined width and height
    combined_width = main_image.width + legend_image.width
    combined_height = max(main_image.height, legend_image.height)

    # Create a new blank image
    combined_image = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))

    # Paste the main plot and legend
    combined_image.paste(main_image, (0, 0))
    combined_image.paste(legend_image, (main_image.width, 0))

    # Save the combined image
    if not output_path.lower().endswith(('.png', '.jpeg', '.jpg', '.pdf')):
        output_path += '.png'  # Default to saving as PNG format
    combined_image.save(output_path, dpi=(dpi, dpi))


