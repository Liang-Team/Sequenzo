"""
@Author  : Yuqi Liang 梁彧祺, Sebastian Daza
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
    save_and_show_results,
    show_plot_title
)


def compute_transition_matrix(seqdata: SequenceData, with_missing: bool = False, weights="auto") -> np.ndarray:
    """
    Compute transition rate matrix using vectorized operations for optimal performance.

    :param seqdata: SequenceData object containing sequence information and states
    :param with_missing: Flag to include missing values in computation
    :param weights: (np.ndarray or "auto") Weights for sequences. If "auto", uses seqdata.weights if available
    :return: numpy.ndarray containing the transition rate matrix
    """
    # Process weights
    if isinstance(weights, str) and weights == "auto":
        weights = getattr(seqdata, "weights", None)

    if weights is not None:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(weights) != len(seqdata.values):
            raise ValueError("Length of weights must equal number of sequences.")

    num_states = len(seqdata.states)
    A = seqdata.to_dataframe().to_numpy()
    n, T = A.shape

    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, dtype=float)

    # Flatten arrays while synchronizing weights for each transition
    current = A[:, :-1].flatten()
    nxt = A[:, 1:].flatten()
    w_pair = np.repeat(w, T-1)  # Each sequence weight replicated (T-1) times

    # Filter valid transitions (states are encoded as 1, 2, 3, ..., num_states)
    valid = (current >= 1) & (current <= num_states) & (nxt >= 1) & (nxt <= num_states)
    current, nxt, w_pair = current[valid], nxt[valid], w_pair[valid]

    # Compute weighted transition counts
    # Create mapping from state codes to matrix indices
    state_codes = sorted(set(current) | set(nxt))
    code_to_idx = {code: idx for idx, code in enumerate(state_codes)}

    # Use only the actual number of unique states for matrix size
    actual_num_states = len(state_codes)
    trans = np.zeros((actual_num_states, actual_num_states), dtype=float)

    for c, n2, ww in zip(current, nxt, w_pair):
        trans[code_to_idx[int(c)], code_to_idx[int(n2)]] += ww

    # Compute transition rates
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    transition_rates = trans / row_sums

    # Create a properly sized matrix with correct mapping to original states
    final_matrix = np.zeros((num_states, num_states), dtype=float)

    # Map back to the original state positions
    for i, from_code in enumerate(state_codes):
        for j, to_code in enumerate(state_codes):
            final_matrix[from_code-1, to_code-1] = transition_rates[i, j]

    return final_matrix


def print_transition_matrix(seqdata: SequenceData, transition_rates: np.ndarray) -> None:
    """
    Print the transition rate matrix with improved readability.
    Uses consistent decimal format for all probabilities.

    :param seqdata: SequenceData object containing state information
    :param transition_rates: numpy array containing transition rates
    """
    state_labels = seqdata.labels

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
        print(f"{from_state:>{max_label_width}} ->", end=" ")

        # Print transition rates
        for prob in transition_rates[i]:
            # Consistent format for all values
            print(f"{prob:>10.4f}", end=" ")
        print()  # New line after each row

    print("-" * 100)  # Separator line
    print("Note: Values represent transition probabilities (range 0-1)")
    print("      All values shown with 4 decimal places for consistency")


def plot_transition_matrix(seqdata: SequenceData,
                           weights="auto",
                           title: str = "State Transition Rate Matrix",
                           fontsize: int = 12,
                           save_as: Optional[str] = None,
                           dpi: int = 200,
                           format: str = ".2f") -> None:
    """
    Plot state transition rate matrix as a heatmap.

    :param seqdata: SequenceData object containing sequence information
    :param weights: (np.ndarray or "auto") Weights for sequences. If "auto", uses seqdata.weights if available
    :param title: optional title for the plot
    :param fontsize: base font size for labels
    :param save_as: optional file path to save the plot
    :param dpi: resolution of the saved plot
    :param format: format string for annotations (default "%.2f")
    """

    # Compute transition matrix with weights
    transition_matrix = compute_transition_matrix(seqdata, weights=weights)
    transition_matrix = np.array(transition_matrix)

    # Set figure size
    plt.figure(figsize=(12, 10))

    # Use fresh color scheme
    cmap = sns.color_palette("light:#5A9", as_cmap=True)

    # Generate heatmap using pre-formatted annotation strings
    ax = sns.heatmap(
        transition_matrix,
        annot=True,
        fmt=format,
        cmap=cmap,
        xticklabels=seqdata.labels,
        yticklabels=seqdata.labels,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"shrink": 0.8},
        square=True,
        annot_kws={"fontsize": fontsize - 2}
    )

    # Show all the borderlines
    for _, spine in ax.spines.items():
        spine.set_visible(True)

    # Adjust format
    if title:
        show_plot_title(plt.gca(), title, show=True, fontsize=fontsize+2, fontweight='bold', pad=20)

    plt.xlabel("State at t + 1", fontsize=fontsize, labelpad=10)
    plt.ylabel("State at t", fontsize=fontsize, labelpad=10)

    # Adjust label rotation and alignment
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Adjust layout
    plt.tight_layout()

    save_and_show_results(save_as, dpi=dpi)