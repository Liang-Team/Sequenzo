"""
@Author  : Yuqi Liang 梁彧祺
@File    : plot_transition_matrix.py
@Time    : 13/02/2025 12:39
@Desc    :
    This implementation closely follows ggseqplot's `ggseqtrplot` function.

    TODO: Current implementation only handles STS (State-Transition-State) format.
          DSS (Distinct-State-Sequence) format will be addressed in future updates.
          (https://maraab23.github.io/ggseqplot/articles/ggseqplot.html)
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from sequenzo.define_sequence_data import SequenceData
from sequenzo.visualization.utils import (
    save_and_show_results
)


def compute_transition_matrix(seqdata: SequenceData, with_missing: bool = False) -> np.ndarray:
    """
    Compute transition rate matrix using vectorized operations for optimal performance.

    :param seqdata: SequenceData object containing sequence information and states
    :param with_missing: Flag to include missing values in computation
    :return: numpy.ndarray containing the transition rate matrix
    """
    num_states = len(seqdata.states)
    seqdata_df = seqdata.to_dataframe()

    # Convert to numpy array for faster operations
    seq_array = seqdata_df.to_numpy()

    # Create arrays for current and next states
    current_states = seq_array[:, :-1].flatten()
    next_states = seq_array[:, 1:].flatten()

    # Create mask for valid transitions
    valid_mask = np.logical_and(
        np.logical_and(current_states >= 0, current_states < num_states),
        np.logical_and(next_states >= 0, next_states < num_states)
    )

    # Filter valid transitions
    current_states = current_states[valid_mask]
    next_states = next_states[valid_mask]

    # Initialize transition matrix
    trans = np.zeros((num_states, num_states))

    # Compute transitions using histogram2d
    if len(current_states) > 0:
        trans, _, _ = np.histogram2d(
            current_states,
            next_states,
            bins=(num_states, num_states),
            range=[[0, num_states], [0, num_states]]
        )

    # Compute transition rates
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
    transition_rates = trans / row_sums

    return transition_rates


def print_transition_matrix(seqdata: SequenceData, transition_rates: np.ndarray) -> None:
    """
    Print the transition rate matrix with improved readability.
    Uses consistent decimal format for all probabilities.

    :param seqdata: SequenceData object containing state information
    :param transition_rates: numpy array containing transition rates
    """
    state_labels = seqdata.states

    # Calculate max width needed for state labels
    max_label_width = max(len(s) for s in state_labels) + 3  # +3 for arrow

    # Print header
    print("\nTransition Rate Matrix:")
    print("-" * 100)  # Separator line

    # Print column headers
    print(" " * max_label_width, end=" ")
    for label in state_labels:
        # Use shorter labels for columns by taking first word
        short_label = label.split('&')[0].strip()
        print(f"{short_label:>10}", end=" ")
    print("\n" + "-" * 100)  # Separator line

    # Print each row
    for i, from_state in enumerate(state_labels):
        # Print row label
        print(f"{from_state:>{max_label_width}} →", end=" ")

        # Print transition rates
        for prob in transition_rates[i]:
            # Consistent format for all values
            print(f"{prob:>10.4f}", end=" ")
        print()  # New line after each row

    print("-" * 100)  # Separator line
    print("Note: Values represent transition probabilities (range 0-1)")
    print("      All values shown with 4 decimal places for consistency")


def plot_transition_matrix(seqdata: SequenceData,
                           title: Optional[str] = None,
                           save_as: Optional[str] = None,
                           dpi: int = 200) -> None:
    """
    Plot state transition rate matrix as a heatmap.

    :param seqdata: SequenceData object containing sequence information
    :param title: optional title for the plot
    :param save_as: optional file path to save the plot
    :param dpi: resolution of the saved plot
    """
    # Compute transition matrix
    transition_matrix = compute_transition_matrix(seqdata)
    transition_matrix = np.array(transition_matrix)

    # Create upper triangle mask (show diagonal)
    mask = np.triu(np.ones(transition_matrix.shape, dtype=bool), k=1)

    # Set figure size
    plt.figure(figsize=(12, 10))

    # Use fresh color scheme
    cmap = sns.color_palette("light:#5A9", as_cmap=True)

    # Generate heatmap
    ax = sns.heatmap(
        transition_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        xticklabels=seqdata.labels,
        yticklabels=seqdata.labels,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"shrink": 0.8},
        square=True
    )

    # Show all the borderlines
    for _, spine in ax.spines.items():
        spine.set_visible(True)

    # Adjust format
    if title:
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
    # plt.title("State Transition Rate Matrix", fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("State at t + 1", fontsize=12, labelpad=10)
    plt.ylabel("State at t", fontsize=12, labelpad=10)

    # Adjust label rotation and alignment
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Adjust layout
    plt.tight_layout()

    save_and_show_results(save_as, dpi=200)

