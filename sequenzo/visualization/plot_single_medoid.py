"""
@Author  : Yuqi Liang 梁彧祺
@File    : plot_single_medoid.py
@Time    : 15/02/2025 14:58
@Desc    :
    Identify and visualize a single medoid sequence from a dataset.
    TODO: Optimize efficiency by computing only one medoid and coverage instead of multiple.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

from sequenzo.define_sequence_data import SequenceData
from sequenzo.visualization.utils import (
    save_and_show_results,
    show_plot_title
)


def plot_single_medoid(seqdata: SequenceData,
                       distance_matrix: np.ndarray,
                       weights="auto",
                       show_legend: bool = True,
                       title: Optional[str] = None,
                       fontsize: int = 12,
                       save_as: Optional[str] = None) -> None:
    """
    Plots a single medoid sequence with colors corresponding to sequence states.

    :param seqdata: SequenceData object containing the sequence dataset.
    :param distance_matrix: Pairwise distance matrix for computing medoids.
    :param weights: (np.ndarray or "auto") Weights for sequences. If "auto", uses seqdata.weights if available
    :param show_legend: Boolean flag to display legend (default: True).
    :param title: Optional title for the plot.
    :param save_as: Optional filename to save the plot.
    :return: None
    """
    # Process weights
    if isinstance(weights, str) and weights == "auto":
        weights = getattr(seqdata, "weights", None)
    
    if weights is not None:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(weights) != len(seqdata.values):
            raise ValueError("Length of weights must equal number of sequences.")
    
    _, medoid_indices = compute_medoids_from_distance_matrix(distance_matrix, seqdata, weights=weights, top_k=1)
    medoid_coverages = _compute_individual_medoid_coverage(distance_matrix, medoid_indices)

    medoid_index = medoid_indices[0]
    coverage = medoid_coverages[0]

    fig, ax = plt.subplots(figsize=(10, 2))
    n_timepoints = seqdata.values.shape[1]
    inv_state_mapping = {v: k for k, v in seqdata.state_mapping.items()}

    # Retrieve the medoid sequence
    sequence = seqdata.values[medoid_index]
    states = [inv_state_mapping[state] for state in sequence]
    colors = [seqdata.color_map[state] for state in states]

    # Plot the sequence
    left = 0
    for t in range(n_timepoints):
        ax.barh(y=0, width=1, height=0.8, left=left, color=colors[t], edgecolor="none", label=states[t] if t == 0 else "")
        left += 1

    # Configure X-axis
    ax.set_xticks(np.arange(n_timepoints) + 0.5)
    ax.set_xticklabels(seqdata.cleaned_time, fontsize=fontsize-2, ha='center')
    ax.tick_params(axis='x', which='major', pad=0, length=0, width=0)

    # Configure Y-axis
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])

    # Display legend if required
    if show_legend:
        handles, labels = seqdata.get_legend()
        ax.legend(handles, labels, title="States", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Set plot title
    show_plot_title(ax, title if title else f"Medoid Sequence (ID: {medoid_index}, Coverage: {coverage * 100:.2f}%)", show=True)

    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    save_and_show_results(save_as, dpi=200)


def compute_medoids_from_distance_matrix(distance_matrix: np.ndarray, seqdata: SequenceData, weights="auto", top_k: Optional[int] = None) -> tuple:
    """
    Computes the medoid(s) based on total distance minimization.

    :param distance_matrix: Pairwise distance matrix.
    :param seqdata: SequenceData object containing sequences.
    :param weights: (np.ndarray or "auto") Weights for sequences. If "auto", uses seqdata.weights if available
    :param top_k: Number of top representative sequences to return.
    :return: Tuple containing the medoid sequences and their indices.
    """
    if not isinstance(seqdata, SequenceData):
        raise TypeError("[X] seqdata must be a SequenceData object.")

    # Process weights
    if isinstance(weights, str) and weights == "auto":
        weights = getattr(seqdata, "weights", None)
    
    if weights is not None:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(weights) != len(seqdata.values):
            raise ValueError("Length of weights must equal number of sequences.")
    
    if weights is None:
        # For each candidate medoid m: sum_i D(i, m)
        total_distances = distance_matrix.sum(axis=0)
    else:
        # For each m: sum_i w_i * D(i, m)
        total_distances = distance_matrix.T @ weights
    
    min_distance = np.min(total_distances)
    medoid_indices = np.where(total_distances == min_distance)[0]

    if top_k is not None:
        sorted_indices = np.argsort(total_distances)
        medoid_indices = sorted_indices[:top_k]

    medoid_sequences = [seqdata.values[idx] for idx in medoid_indices]
    medoid_indices = medoid_indices.tolist()

    if not all(isinstance(idx, int) for idx in medoid_indices):
        raise ValueError("[X] medoid_indices must be a list of integers.")

    return medoid_sequences, medoid_indices


def _compute_individual_medoid_coverage(distance_matrix: np.ndarray, medoid_indices: List[int], threshold_ratio: float = 0.10) -> List[float]:
    """
    Computes the coverage contribution of each medoid.

    :param distance_matrix: Pairwise distance matrix.
    :param medoid_indices: List of medoid indices.
    :param threshold_ratio: Distance threshold ratio for coverage.
    :return: List of coverage proportions for each medoid.
    """
    max_distance = np.max(distance_matrix)
    threshold = max_distance * threshold_ratio
    medoid_coverages = [np.sum(distance_matrix[:, medoid] <= threshold) / len(distance_matrix) for medoid in medoid_indices]
    return medoid_coverages


