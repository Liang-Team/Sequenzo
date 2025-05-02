"""
@Author  : Yuqi Liang 梁彧祺
@File    : system_level_indicators.py
@Time    : 02/05/2025 11:06
@Desc    : 
"""
from collections import defaultdict, Counter
import numpy as np
from scipy.spatial.distance import jensenshannon

from sequenzo.visualization.utils import save_and_show_results, save_figure_to_buffer
import matplotlib.pyplot as plt
from typing import List, Optional, Dict


class PrefixTree:
    def __init__(self):
        self.root = {}
        self.counts = defaultdict(int)  # prefix -> count
        self.total_sequences = 0

    def insert(self, sequence):
        prefix = []
        node = self.root
        for state in sequence:
            prefix.append(state)
            key = tuple(prefix)
            self.counts[key] += 1
            if state not in node:
                node[state] = {}
            node = node[state]

    def get_prefixes_at_depth(self, depth):
        return [k for k in self.counts if len(k) == depth]

    def get_children_count(self, prefix):
        node = self.root
        for state in prefix:
            node = node.get(state, {})
        return len(node)

    def describe(self):
        depths = [len(k) for k in self.counts.keys()]
        max_depth = max(depths) if depths else 0
        total_prefixes = len(self.counts)
        print("\n[PrefixTree Overview]")
        print(f"[>] Total sequences inserted: {self.total_sequences}")
        print(f"[>] Max depth (time points): {max_depth}")
        print(f"[>] Total distinct prefixes: {total_prefixes}")

        for t in range(1, max_depth + 1):
            level_prefixes = self.get_prefixes_at_depth(t)
            print(f"    Level {t}: {len(level_prefixes)} unique prefixes")

    def __repr__(self):
        """
        Returns a brief textual summary of the prefix tree object.

        Note:
            This method is intended to provide a lightweight, one-line overview
            (e.g., max depth and total prefix count). For a full structural report
            including per-level statistics, use the `.describe()` method instead.
        """
        depths = [len(k) for k in self.counts.keys()]
        return f"PrefixTree(max_depth={max(depths) if depths else 0}, total_prefixes={len(self.counts)})"


def compute_prefix_count(tree, max_depth):
    return [len(tree.get_prefixes_at_depth(t)) for t in range(1, max_depth + 1)]


def compute_branching_factor(tree, max_depth):
    result = []
    for t in range(2, max_depth + 1):
        prefixes = tree.get_prefixes_at_depth(t - 1)
        if not prefixes:
            result.append(0)
            continue
        child_counts = [tree.get_children_count(p) for p in prefixes]
        result.append(np.mean(child_counts))
    return [0] + result  # pad to align with prefix count


def compute_js_divergence(sequences, state_set):
    T = len(sequences[0])
    distros = []
    for t in range(T):
        counter = Counter(seq[t] for seq in sequences)
        dist = np.array([counter[s] for s in state_set], dtype=float)
        dist = dist / dist.sum()
        distros.append(dist)

    js_scores = [0.0]
    for t in range(1, T):
        js = jensenshannon(distros[t], distros[t - 1])
        js_scores.append(js)
    return js_scores


def compute_composite_score(prefix_counts, branching_factors, js_divergence=None):
    pc_z = (np.array(prefix_counts) - np.mean(prefix_counts)) / np.std(prefix_counts)
    bf_z = (np.array(branching_factors) - np.mean(branching_factors)) / np.std(branching_factors)
    score = pc_z + bf_z

    if js_divergence is not None:
        js_z = (np.array(js_divergence) - np.mean(js_divergence)) / np.std(js_divergence)
        score += js_z

    return score.tolist()


def build_prefix_tree(sequences):
    tree = PrefixTree()
    tree.total_sequences = len(sequences)
    for seq in sequences:
        for t in range(1, len(seq) + 1):
            tree.insert(seq[:t])
    return tree


import matplotlib.pyplot as plt
from typing import List, Optional, Dict
from scipy.stats import zscore
from numpy import array
from sequenzo.visualization.utils import save_and_show_results

def plot_system_indicators(prefix_counts: List[float],
                            branching_factors: List[float],
                            js_divergence: Optional[List[float]] = None,
                            composite_score: Optional[List[float]] = None,
                            save_as: Optional[str] = None,
                            dpi: int = 200,
                            custom_colors: Optional[Dict[str, str]] = None,
                            show: bool = True,
                            norm: bool = False) -> None:
    """
    Plot system-level indicators over time.

    Parameters:
        prefix_counts (List[float]): Time series of prefix count values.
        branching_factors (List[float]): Time series of branching factors.
        js_divergence (List[float], optional): JS divergence values over time.
        composite_score (List[float], optional): Composite divergence score.
        save_as (str, optional): File path to save the figure (e.g., 'divergence.png').
        dpi (int): Image resolution.
        custom_colors (dict, optional): Custom color mapping (e.g., {'Prefix Count': 'red'}).
        show (bool): Whether to display the plot.
        norm (bool): Whether to z-score normalize all indicators.
    """
    T = len(prefix_counts)
    x = list(range(1, T + 1))

    if norm:
        prefix_counts = zscore(array(prefix_counts))
        branching_factors = zscore(array(branching_factors))
        if js_divergence:
            js_divergence = zscore(array(js_divergence))
        if composite_score:
            composite_score = zscore(array(composite_score))

    color_defaults = {
        "Prefix Count": "#1f77b4",
        "Branching Factor": "#ff7f0e",
        "JS Divergence": "#2ca02c",
        "Composite Score": "#000000",
    }
    colors = {**color_defaults, **(custom_colors or {})}

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, prefix_counts, marker='o', label='Prefix Count', color=colors["Prefix Count"])
    ax.plot(x, branching_factors, marker='s', label='Branching Factor', color=colors["Branching Factor"])

    if js_divergence is not None:
        ax.plot(x, js_divergence, marker='^', label='JS Divergence', color=colors["JS Divergence"])

    if composite_score is not None:
        ax.plot(x, composite_score, linestyle='--', label='Composite Score', color=colors["Composite Score"])

    ax.set_title("System-Level Trajectory Indicators Over Time", fontsize=14)
    ax.set_xlabel("Time (t)", fontsize=12)
    ax.set_ylabel("Value" + (" (z-score)" if norm else ""), fontsize=12)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    save_and_show_results(save_as=save_as, dpi=dpi, show=show)
