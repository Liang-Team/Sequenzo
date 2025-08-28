"""
@Author  : Yuqi Liang 梁彧祺
@File    : system_level_indicators.py
@Time    : 02/05/2025 11:06
@Desc    : 
    This module includes tools for building suffix trees, computing suffix counts, merging factors, and Jensen-Shannon convergence,
    as well as generating composite scores to summarize system-level sequence convergence and consolidation over time.
    Visualization functions are also provided to plot these indicators and their distributions, 
    supporting comprehensive analysis of sequence system dynamics.
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
        suffix = []
        node = self.root
        for state in sequence:
            suffix.append(state)
            key = tuple(suffix)
            self.counts[key] += 1
            if state not in node:
                node[state] = {}
            node = node[state]

    def get_suffixes_at_depth(self, depth):
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
        node = self.root
        for state in suffix:
            node = node.get(state, {})
        return len(node)

    def describe(self):
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
            This method is intended to provide a lightweight, one-line overview
            (e.g., max depth and total suffix count). For a full structural report
            including per-level statistics, use the `.describe()` method instead.
        """
        depths = [len(k) for k in self.counts.keys()]
        return f"SuffixTree(max_depth={max(depths) if depths else 0}, total_suffixes={len(self.counts)})"


def compute_suffix_count(tree, max_depth):
    return [len(tree.get_suffixes_at_depth(t)) for t in range(1, max_depth + 1)]


def compute_merging_factor(tree, max_depth):
    result = []
    for t in range(2, max_depth + 1):
        suffixes = tree.get_suffixes_at_depth(t - 1)
        if not suffixes:
            result.append(0)
            continue
        child_counts = [tree.get_children_count(s) for s in suffixes]
        result.append(np.mean(child_counts))
    return [0] + result  # pad to align with suffix count


def compute_js_convergence(sequences, state_set):
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


def build_suffix_tree(sequences):
    tree = SuffixTree()
    tree.total_sequences = len(sequences)
    for seq in sequences:
        for t in range(len(seq)):
            tree.insert(seq[t:])
    return tree


def plot_system_indicators(
    suffix_counts: List[float],
    merging_factors: List[float],
    js_convergence: Optional[List[float]] = None,
    x_values: Optional[List] = None,
    x_label: str = "Time (t)",
    legend_loc: str = 'lower right',
    save_as: Optional[str] = None,
    figsize: Optional[tuple] = None,
    dpi: int = 300,
    custom_colors: Optional[Dict[str, str]] = None,
    show: bool = True,
    plot_distributions: bool = False,
    style: Optional[str] = None
) -> None:
    """
    Plot a single group's system-level indicators using the same visual style as
    `plot_system_indicators_multiple_comparison`, but for one subplot.

    Design:
    - Left y-axis: raw Suffix Count
    - Right y-axis: z-score of Merging Factor and (optionally) JS Convergence
    - Consistent colors/markers and legend handling with the multi-comparison API

    Parameters:
    - suffix_counts: List[float]
        Raw suffix counts per time step
    - merging_factors: List[float]
        Merging factor per time step
    - js_convergence: Optional[List[float]]
        JS convergence per time step; if None, only merging factor is shown on right axis
    - x_values: Optional[List]
        Custom x-axis ticks (e.g., years). If None, uses 1..T. Length must equal data length
    - x_label: str
        Label for x-axis. Default: "Time (t)"
    - legend_loc: str
        Legend location, e.g., 'upper left', 'upper right', 'lower right', 'best', etc. Default: 'lower right'
    - save_as: Optional[str]
        If provided, save the figure to this path (png). DPI controlled by `dpi`
    - figsize: Optional[tuple]
        Figure size (width, height). Default: (12, 6)
    - dpi: int
        Figure DPI when saving. Default: 300
    - custom_colors: Optional[Dict[str, str]]
        Optional color overrides. Keys: "Suffix Count", "Merging Factor", "JS Convergence"
    - show: bool
        Whether to display the figure
    - plot_distributions: bool
        If True, additionally show raw distributions (histograms) of indicators

    Example:
    >>> plot_system_indicators(
    ...     suffix_counts=india_suffix_counts,
    ...     merging_factors=india_merging_factors,
    ...     js_convergence=india_js_scores,
    ...     x_values=[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
    ...     x_label="Year",
    ...     legend_loc="lower right",
    ...     figsize=(12, 6),
    ...     dpi=300,
    ... )
    """
    T = len(suffix_counts)
    # Set x values to align with multi-group API
    if x_values is None:
        x_values = list(range(1, T + 1))
    if len(x_values) != T:
        raise ValueError("Length of x_values must match data length")

    # Normalize others
    mf_z = zscore(array(merging_factors))
    js_z = zscore(array(js_convergence)) if js_convergence else None

    color_defaults = {
        "Suffix Count": "#6BB6FF",    # Soft sky blue (like Monet's water lilies)
        "Merging Factor": "#FFB347",  # Warm peach/coral (like sunset reflections)
        "JS Convergence": "#98D8C8",  # Soft mint green (convergence = coming together)
    }
    colors = {**color_defaults, **(custom_colors or {})}

    # --- Main line plot with dual axes ---
    if figsize is None:
        figsize = (12, 6)
    
    # Apply style if specified
    if style is not None:
        # Check if it's a seaborn style
        seaborn_styles = ['whitegrid', 'darkgrid', 'white', 'dark', 'ticks']
        if style in seaborn_styles:
            sns.set_style(style)
        else:
            plt.style.use(style)
    
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Suffix Count", color=colors["Suffix Count"])
    ax1.plot(x_values, suffix_counts, marker='o', color=colors["Suffix Count"], label="Suffix Count")
    ax1.tick_params(axis='y', labelcolor=colors["Suffix Count"])

    ax2 = ax1.twinx()
    ax2.set_ylabel("Z-score (Other Indicators)")
    ax2.plot(x_values, mf_z, marker='s', label='Merging Factor (z)', color=colors["Merging Factor"])
    if js_z is not None:
        ax2.plot(x_values, js_z, marker='^', label='JS Convergence (z)', color=colors["JS Convergence"])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=legend_loc)

    ax1.set_title("System-Level Trajectory Indicators: Raw vs. Normalized")
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


def plot_system_indicators_multiple_comparison(
    groups_data: Dict[str, Dict[str, List[float]]],
    group_names: Optional[List[str]] = None,
    subplot_titles: Optional[List[str]] = None,
    x_values: Optional[List] = None,
    x_label: str = "Time (t)",
    legend_loc: str = 'lower right',
    save_as: Optional[str] = None,
    figsize: Optional[tuple] = None,
    dpi: int = 300,
    custom_colors: Optional[Dict[str, str]] = None,
    show: bool = True,
    style: Optional[str] = None
) -> None:
    """
    Plot system-level indicators comparison across multiple groups using dual y-axis design.
    
    Parameters:
    -----------
    groups_data : Dict[str, Dict[str, List[float]]]
        Dictionary with group names as keys and data dictionaries as values.
        Each data dict should contain 'suffix_counts', 'merging_factors', and 'js_convergence'.
        Example: {
            "Group1": {
                "suffix_counts": [10, 15, 20, ...],
                "merging_factors": [1.2, 1.5, 1.8, ...], 
                "js_convergence": [0.1, 0.2, 0.15, ...]
            },
            "Group2": {...}
        }
    group_names : Optional[List[str]]
        Custom names for groups. If None, uses keys from groups_data.
        Used for default subplot titles if subplot_titles is not provided.
    subplot_titles : Optional[List[str]]
        Custom titles for each subplot. If None, uses default format:
        "{group_name} - System-Level Trajectory Indicators: Raw vs. Normalized"
    x_values : Optional[List]
        Custom x-axis values. If None, uses 1, 2, 3, ...
    x_label : str
        Label for x-axis. Default: "Time (t)"
    legend_loc : str
        Legend location. Options: 'upper left', 'upper right', 'lower left', 
        'lower right', 'center', 'best', etc. Default: 'lower right'
    save_as : Optional[str]
        File path to save the plot (without extension)
    figsize : Optional[tuple]
        Figure size (width, height). If None, auto-calculated based on number of groups
    dpi : int
        DPI for saving. Default: 300
    custom_colors : Optional[Dict[str, str]]
        Custom colors for indicators. Default uses standard colors.
    show : bool
        Whether to show the plot. Default: True
    style : Optional[str]
        Style to apply. Seaborn styles ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks') 
        or matplotlib styles. If None, uses default style. Default: None
        
    Example:
    --------
    >>> data = {
    ...     "India": {
    ...         "suffix_counts": india_suffix_counts,
    ...         "merging_factors": india_merging_factors,
    ...         "js_convergence": india_js_scores
    ...     },
    ...     "US": {
    ...         "suffix_counts": us_suffix_counts,
    ...         "merging_factors": us_merging_factors,
    ...         "js_convergence": us_js_scores
    ...     }
    ... }
    >>> plot_system_indicators_multiple_comparison(
    ...     groups_data=data,
    ...     x_label="Years",
    ...     legend_loc='upper right',
    ...     save_as="multi_country_comparison"
    ... )
    
    >>> # With custom subplot titles
    >>> plot_system_indicators_multiple_comparison(
    ...     groups_data=data,
    ...     subplot_titles=["印度发展轨迹", "美国发展轨迹"],
    ...     x_label="年份",
    ...     save_as="custom_titles_comparison"
    ... )
    """
    
    # Validate input
    if not groups_data:
        raise ValueError("groups_data cannot be empty")
    
    # Get group names
    if group_names is None:
        group_names = list(groups_data.keys())
    
    if len(group_names) != len(groups_data):
        raise ValueError("Length of group_names must match number of groups in groups_data")
    
    # Validate subplot_titles
    if subplot_titles is not None and len(subplot_titles) != len(groups_data):
        raise ValueError("Length of subplot_titles must match number of groups in groups_data")
    
    # Get first group to determine data length
    first_group_data = list(groups_data.values())[0]
    T = len(first_group_data['suffix_counts'])
    
    # Set x values
    if x_values is None:
        x_values = list(range(1, T + 1))
    
    if len(x_values) != T:
        raise ValueError("Length of x_values must match data length")
    
    # Color settings
    color_defaults = {
        "Suffix Count": "#6BB6FF",    # Soft sky blue (like Monet's water lilies)
        "Merging Factor": "#FFB347",  # Warm peach/coral (like sunset reflections)
        "JS Convergence": "#98D8C8",  # Soft mint green (convergence = coming together)
    }
    colors = {**color_defaults, **(custom_colors or {})}
    
    # Calculate figure size
    n_groups = len(groups_data)
    if figsize is None:
        figsize = (12, 4 * n_groups + 2)  # Dynamic height based on number of groups
    
    # Apply style if specified
    if style is not None:
        # Check if it's a seaborn style
        seaborn_styles = ['whitegrid', 'darkgrid', 'white', 'dark', 'ticks']
        if style in seaborn_styles:
            sns.set_style(style)
        else:
            plt.style.use(style)
    
    # Create subplots
    fig, axes = plt.subplots(n_groups, 1, figsize=figsize)
    
    # Handle single group case
    if n_groups == 1:
        axes = [axes]
    
    # Plot each group
    for i, (group_key, group_name) in enumerate(zip(groups_data.keys(), group_names)):
        data = groups_data[group_key]
        ax = axes[i]
        
        # Validate data completeness
        required_keys = ['suffix_counts', 'merging_factors', 'js_convergence']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing '{key}' in data for group '{group_key}'")
        
        # Normalize data (z-score)
        mf_z = zscore(array(data['merging_factors']))
        js_z = zscore(array(data['js_convergence']))
        
        # Left y-axis: raw suffix counts
        ax.set_ylabel("Suffix Count", color=colors["Suffix Count"])
        ax.plot(x_values, data['suffix_counts'], marker='o', 
                color=colors["Suffix Count"], label="Suffix Count")
        ax.tick_params(axis='y', labelcolor=colors["Suffix Count"])
        
        # Right y-axis: normalized indicators
        ax_twin = ax.twinx()
        ax_twin.set_ylabel("Z-score (Other Indicators)")
        ax_twin.plot(x_values, mf_z, marker='s', 
                     label='Merging Factor (z)', color=colors["Merging Factor"])
        ax_twin.plot(x_values, js_z, marker='^', 
                     label='JS Convergence (z)', color=colors["JS Convergence"])
        
        # Legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax_twin.legend(lines1 + lines2, labels1 + labels2, loc=legend_loc)
        
        # Title and labels
        if subplot_titles is not None:
            title = subplot_titles[i]
        else:
            title = f"{group_name} - System-Level Trajectory Indicators: Raw vs. Normalized"
        ax.set_title(title)
        
        # Only set x-label for the bottom subplot
        if i == n_groups - 1:
            ax.set_xlabel(x_label)
    
    plt.tight_layout()
    save_and_show_results(save_as=save_as, dpi=dpi, show=show)
