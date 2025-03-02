"""
@Author  : Yuqi Liang 梁彧祺
@File    : plot_state_distribution.py
@Time    : 15/02/2025 22:03
@Desc    : 
"""
import pandas as pd
import matplotlib.pyplot as plt

from sequenzo import SequenceData
from sequenzo.visualization.utils import (
    set_up_time_labels_for_x_axis,
    save_and_show_results
)


def plot_state_distribution(seqdata: SequenceData,
                            figsize=(12, 7),
                            title=None,
                            xlabel="Time",
                            ylabel="State Distribution (%)",
                            stacked=True,
                            save_as=None,
                            dpi=200) -> None:
    """
    Creates a state distribution plot showing how the prevalence of states changes over time,
    with enhanced color vibrancy.

    :param seqdata: (SequenceData) A SequenceData object containing sequences
    :param figsize: (tuple) Size of the figure
    :param title: (str) Optional title for the plot
    :param xlabel: (str) Label for the x-axis
    :param ylabel: (str) Label for the y-axis
    :param stacked: (bool) Whether to create a stacked area plot (True) or line plot (False)
    :param save_as: (str) Optional file path to save the plot
    :param dpi: (int) Resolution of the saved plot

    :return: None
    """
    # Get sequence data as a DataFrame
    seq_df = seqdata.to_dataframe()

    # Create a state mapping from numerical values back to state names
    inv_state_mapping = {v: k for k, v in seqdata.state_mapping.items()}

    # Calculate state distributions at each time point
    distributions = []
    for col in seq_df.columns:
        # Count occurrences of each state at this time point
        state_counts = seq_df[col].value_counts().sort_index()

        # Convert to percentages
        total = len(seq_df)
        state_percentages = (state_counts / total) * 100

        # Create a dictionary with states as keys and percentages as values
        # Ensure all states are included (with 0% if not present)
        dist = {inv_state_mapping.get(i, 'Missing'): state_percentages.get(i, 0)
                for i in range(1, len(seqdata.states) + 1)}

        # Add time point and distribution to the list
        distributions.append(dict(time=col, **dist))

    # Ensure percentages sum to exactly 100% to avoid gaps
    for i in range(len(distributions)):
        # Get sum of all state percentages for this time point
        total_percentage = sum(distributions[i][state] for state in seqdata.states)

        # If there's a gap, add the difference to the top-most state
        if total_percentage < 100:
            # Get the last (top-most) state in your stack
            top_state = seqdata.states[-1]
            # Add the difference to make total exactly 100%
            distributions[i][top_state] += (100 - total_percentage)

    # Convert to DataFrame for plotting
    dist_df = pd.DataFrame(distributions)

    # Create the plot
    plt.style.use('default')  # Start with default style for clean slate
    fig, ax = plt.subplots(figsize=figsize)

    # Get colors for each state and enhance vibrancy
    base_colors = [seqdata.color_map[state] for state in seqdata.states]

    # Plot the data
    if stacked:
        # Create a stacked area plot with enhanced colors
        ax.stackplot(range(len(dist_df)),
                     [dist_df[state] for state in seqdata.states],
                     labels=seqdata.states,
                     colors=base_colors,
                     alpha=1.0)  # Full opacity for maximum vibrancy

        # Add grid lines behind the stack plot
        ax.grid(axis='y', linestyle='-', alpha=0.2)
        ax.set_axisbelow(True)
    else:
        # Create a line plot with enhanced colors
        for i, state in enumerate(seqdata.states):
            ax.plot(range(len(dist_df)), dist_df[state],
                    label=state, color=base_colors[i],
                    linewidth=2.5, marker='o', markersize=5)

        # Add grid lines
        ax.grid(True, linestyle='-', alpha=0.2)

    # Set axis labels and title
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Set x-axis labels based on time points
    set_up_time_labels_for_x_axis(seqdata, ax)

    # Enhance aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_linewidth(0.7)

    # Set y-axis limits from 0 to 100%
    ax.set_ylim(0, 100)

    # Add legend
    legend = ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
                       frameon=False, fontsize=10)

    # Adjust layout to make room for the legend
    plt.tight_layout()

    save_and_show_results(save_as, dpi=200)


