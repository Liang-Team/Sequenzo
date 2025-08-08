"""
@Author  : Yuqi Liang 梁彧祺
@File    : system_level_indicators.py
@Time    : 08/08/2025 15:42
@Desc    : 
    This module provides tools for building suffix trees and computing convergence indicators including suffix counts, 
    merging factors, and Jensen-Shannon convergence measures. These indicators quantify how trajectories consolidate 
    toward shared futures and identify convergence patterns in sequence systems over time. Visualization functions 
    support comprehensive analysis of system-level convergence dynamics.
"""
from collections import defaultdict, Counter
import numpy as np
from scipy.stats import zscore
from numpy import array
from scipy.spatial.distance import jensenshannon

from sequenzo.visualization.utils import save_and_show_results
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict


class SuffixTree:
    def __init__(self):
        self.root = {}
        self.counts = defaultdict(int)  # suffix -> count
        self.total_sequences = 0

    def insert(self, sequence):
        """Insert all suffixes of a sequence into the tree."""
        suffix = []
        node = self.root
        # Process sequence in reverse to build suffixes
        for state in reversed(sequence):
            suffix.insert(0, state)  # Build suffix from end to start
            key = tuple(suffix)
            self.counts[key] += 1
            if state not in node:
                node[state] = {}
            node = node[state]

    def get_suffixes_at_depth(self, depth):
        """Get all suffixes of a specific length."""
        return [k for k in self.counts if len(k) == depth]

    def get_children(self, suffix):
        """
        Given a suffix (as a list or tuple), return its immediate children in the tree.
        
        Returns:
            dict: mapping from child state -> subtree dict
        """
        node = self.root
        for state in suffix:
            node = node.get(state, {})
        return node

    def get_children_count(self, suffix):
        """Count immediate children of a given suffix."""
        node = self.root
        for state in suffix:
            node = node.get(state, {})
        return len(node)

    def describe(self):
        """Print a structural overview of the suffix tree."""
        depths = [len(k) for k in self.counts.keys()]
        max_depth = max(depths) if depths else 0
        total_suffixes = len(self.counts)
        print("\n[SuffixTree Overview]")
        print(f"[>] Total sequences inserted: {self.total_sequences}")
        print(f"[>] Max depth (time points): {max_depth}")
        print(f"[>] Total distinct suffixes: {total_suffixes}")

        for t in range(1, max_depth + 1):
            level_suffixes = self.get_suffixes_at_depth(t)
            print(f"    Level {t}: {len(level_suffixes)} unique suffixes")

    def __repr__(self):
        """
        Returns a brief textual summary of the suffix tree object.
        
        Note:
            This method provides a lightweight, one-line overview
            (e.g., max depth and total suffix count). For a full structural report
            including per-level statistics, use the `.describe()` method instead.
        """
        depths = [len(k) for k in self.counts.keys()]
        return f"SuffixTree(max_depth={max(depths) if depths else 0}, total_suffixes={len(self.counts)})"


def build_suffix_tree(sequences):
    """
    Build a suffix tree from a list of sequences.
    
    :param sequences: List of sequences (each sequence is a list/tuple of states).
    :return: SuffixTree object with all sequences inserted.
    """
    tree = SuffixTree()
    tree.total_sequences = len(sequences)
    for seq in sequences:
        for t in range(len(seq)):
            tree.insert(seq[t:])  # Insert suffix starting at position t
    return tree


def compute_suffix_count(tree, max_depth):
    """Compute suffix count at each depth - measures path consolidation toward shared futures."""
    return [len(tree.get_suffixes_at_depth(t)) for t in range(1, max_depth + 1)]


def compute_merging_factor(sequences, max_depth):
    """Compute merging factor - measures how different pasts converge into shared futures."""
    T = max_depth
    result = [0]  # No merging at t=1
    
    for t in range(2, T + 1):
        # Get all suffixes starting at time t
        suffix_to_prefixes = defaultdict(set)
        
        for seq in sequences:
            if len(seq) >= t:
                suffix = tuple(seq[t-1:])  # Suffix from time t onward
                prefix = tuple(seq[:t-1])   # Prefix up to time t-1
                suffix_to_prefixes[suffix].add(prefix)
        
        if not suffix_to_prefixes:
            result.append(0)
            continue
            
        # Calculate average number of distinct prefixes per suffix
        prefix_counts = [len(prefixes) for prefixes in suffix_to_prefixes.values()]
        result.append(np.mean(prefix_counts))
    
    return result


def compute_js_convergence(sequences, state_set):
    """Compute Jensen-Shannon convergence - measures stabilization in state distributions over time."""
    T = len(sequences[0])
    distros = []
    for t in range(T):
        counter = Counter(seq[t] for seq in sequences)
        dist = np.array([counter[s] for s in state_set], dtype=float)
        dist = dist / dist.sum()
        distros.append(dist)

    # JS convergence: declining JS divergence indicates convergence
    js_scores = [0.0]
    for t in range(1, T):
        js = jensenshannon(distros[t], distros[t - 1])
        js_scores.append(1.0 - js)  # Invert: higher values = more convergence
    return js_scores


def plot_convergence_indicators(suffix_counts: List[float],
                               merging_factors: List[float],
                               js_convergence: Optional[List[float]] = None,
                               save_as: Optional[str] = None,
                               dpi: int = 200,
                               custom_colors: Optional[Dict[str, str]] = None,
                               show: bool = True,
                               plot_distributions: bool = False) -> None:
    """
    Plot system-level convergence indicators over time using:
    - Left axis: raw suffix counts
    - Right axis: z-score of other indicators
    - Optionally: individual raw distribution plots of all indicators
    """
    T = len(suffix_counts)
    x = list(range(1, T + 1))

    # Normalize others
    mf_z = zscore(array(merging_factors))
    js_z = zscore(array(js_convergence)) if js_convergence else None

    color_defaults = {
        "Suffix Count": "#1f77b4",
        "Merging Factor": "#ff7f0e",
        "JS Convergence": "#2ca02c",
    }
    colors = {**color_defaults, **(custom_colors or {})}

    # --- Main line plot with dual axes ---
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel("Time (t)")
    ax1.set_ylabel("Suffix Count", color=colors["Suffix Count"])
    ax1.plot(x, suffix_counts, marker='o', color=colors["Suffix Count"], label="Suffix Count")
    ax1.tick_params(axis='y', labelcolor=colors["Suffix Count"])
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Z-score (Convergence Indicators)")
    ax2.plot(x, mf_z, marker='s', label='Merging Factor (z)', color=colors["Merging Factor"])
    if js_z is not None:
        ax2.plot(x, js_z, marker='^', label='JS Convergence (z)', color=colors["JS Convergence"])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.set_title("System-Level Convergence Indicators: Raw vs. Normalized")
    fig.tight_layout()

    save_and_show_results(save_as=save_as, dpi=dpi, show=show)

    # --- Distribution plots if requested ---
    if plot_distributions:
        raw_data = {
            "Suffix Count": suffix_counts,
            "Merging Factor": merging_factors,
        }
        if js_convergence:
            raw_data["JS Convergence"] = js_convergence

        n = len(raw_data)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, (label, values) in zip(axes, raw_data.items()):
            sns.histplot(values, kde=True, ax=ax, color=colors.get(label, None))
            ax.set_title(f"{label} Distribution")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")

        fig.tight_layout()
        suffix = "_distributions" if save_as else None
        dist_path = save_as.replace(".png", f"{suffix}.png") if save_as else None
        save_and_show_results(save_as=dist_path, dpi=dpi, show=show)
