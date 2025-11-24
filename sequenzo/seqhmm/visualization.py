"""
@Author  : Yuqi Liang 梁彧祺
@File    : visualization.py
@Time    : 2025-11-18 07:25
@Desc    : Visualization functions for HMM models

This module provides visualization functions for HMM models, similar to
seqHMM's plot.hmm() and plot.mhmm() functions in R.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, Wedge
from matplotlib.collections import PatchCollection
from typing import Optional, List, Union
from .hmm import HMM
from .mhmm import MHMM

# Try to import networkx for network layout, but make it optional
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


def plot_hmm(
    model: HMM,
    which: str = 'transition',
    figsize: Optional[tuple] = None,
    ax: Optional[plt.Axes] = None,
    # Network plot parameters (similar to R's plot.hmm)
    vertex_size: float = 50,
    vertex_label_dist: float = 1.5,
    edge_curved: Union[bool, float] = 0.5,
    edge_label_cex: float = 0.8,
    vertex_label: str = 'initial.probs',
    loops: bool = False,
    trim: float = 1e-15,
    combine_slices: float = 0.05,
    with_legend: Union[bool, str] = 'bottom',
    layout: str = 'horizontal',
    **kwargs
) -> plt.Figure:
    """
    Plot HMM model parameters.
    
    This function visualizes HMM model parameters, including:
    - Transition probability matrix
    - Emission probability matrix
    - Initial state probabilities
    - Network graph (similar to R's plot.hmm())
    
    It is similar to seqHMM's plot.hmm() function in R.
    
    Args:
        model: Fitted HMM model object
        which: What to plot. Options:
            - 'transition': Transition probability matrix (default)
            - 'emission': Emission probability matrix
            - 'initial': Initial state probabilities
            - 'network': Network graph with pie chart nodes (like R's plot.hmm)
            - 'all': All three plots (transition, emission, initial)
        figsize: Figure size tuple (width, height). If None, uses default.
        ax: Optional matplotlib axes to plot on. If None, creates new figure.
        
        # Network plot parameters (only used when which='network'):
        vertex_size: Size of vertices (nodes). Default 50.
        vertex_label_dist: Distance of vertex labels from center. Default 1.5.
        edge_curved: Whether to plot curved edges. Can be bool or float (curvature). Default 0.5.
        edge_label_cex: Character expansion factor for edge labels. Default 0.8.
        vertex_label: Labels for vertices. Options: 'initial.probs', 'names', or custom list. Default 'initial.probs'.
        loops: Whether to plot self-loops (transitions back to same state). Default False.
        trim: Minimum transition probability to plot. Default 1e-15.
        combine_slices: Emission probabilities below this are combined into 'others'. Default 0.05.
        with_legend: Whether and where to plot legend. Options: True, False, 'bottom', 'top', 'left', 'right'. Default 'bottom'.
        layout: Layout of vertices. Options: 'horizontal', 'vertical'. Default 'horizontal'.
        **kwargs: Additional arguments passed to network plot.
        
    Returns:
        matplotlib Figure: The figure object
        
    Examples:
        >>> from sequenzo import SequenceData, load_dataset
        >>> from sequenzo.seqhmm import build_hmm, fit_model, plot_hmm
        >>> 
        >>> # Load and prepare data
        >>> df = load_dataset('mvad')
        >>> seq = SequenceData(df, time=range(15, 86), states=['EM', 'FE', 'HE', 'JL', 'SC', 'TR'])
        >>> 
        >>> # Build and fit model
        >>> hmm = build_hmm(seq, n_states=4, random_state=42)
        >>> hmm = fit_model(hmm)
        >>> 
        >>> # Plot network graph (like R's plot.hmm)
        >>> plot_hmm(hmm, which='network', vertex_size=50, edge_curved=0.5)
        >>> plt.show()
        >>> 
        >>> # Plot transition matrix
        >>> plot_hmm(hmm, which='transition')
        >>> plt.show()
    """
    if model.log_likelihood is None:
        raise ValueError("Model must be fitted before plotting. Use fit_model() first.")
    
    if which == 'network':
        return _plot_hmm_network(
            model, figsize=figsize, ax=ax,
            vertex_size=vertex_size,
            vertex_label_dist=vertex_label_dist,
            edge_curved=edge_curved,
            edge_label_cex=edge_label_cex,
            vertex_label=vertex_label,
            loops=loops,
            trim=trim,
            combine_slices=combine_slices,
            with_legend=with_legend,
            layout=layout,
            **kwargs
        )
    elif which == 'all':
        # Create subplots for all three
        if figsize is None:
            figsize = (15, 5)
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Remove outer borders for a cleaner look
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#cccccc')
            ax.spines['left'].set_color('#cccccc')
        
        # Plot each component
        _plot_transition_matrix(model, ax=axes[0])
        _plot_emission_matrix(model, ax=axes[1])
        _plot_initial_probs(model, ax=axes[2])
        
        plt.tight_layout()
        return fig
    
    elif which == 'transition':
        return _plot_transition_matrix(model, figsize=figsize, ax=ax)
    elif which == 'emission':
        return _plot_emission_matrix(model, figsize=figsize, ax=ax)
    elif which == 'initial':
        return _plot_initial_probs(model, figsize=figsize, ax=ax)
    else:
        raise ValueError(f"Unknown 'which' option: {which}. Must be 'transition', 'emission', 'initial', 'network', or 'all'.")


def _plot_transition_matrix(
    model: HMM,
    figsize: Optional[tuple] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """Plot transition probability matrix as a heatmap."""
    if ax is None:
        if figsize is None:
            figsize = (8, 6)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Create heatmap with a more elegant colormap
    im = ax.imshow(model.transition_probs, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar with cleaner style
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Transition Probability', rotation=270, labelpad=20, fontsize=10)
    cbar.outline.set_visible(False)
    
    # Set ticks and labels
    ax.set_xticks(range(model.n_states))
    ax.set_yticks(range(model.n_states))
    ax.set_xticklabels(model.state_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(model.state_names, fontsize=9)
    
    # Add text annotations
    for i in range(model.n_states):
        for j in range(model.n_states):
            text = ax.text(j, i, f'{model.transition_probs[i, j]:.2f}',
                          ha="center", va="center", 
                          color="black" if model.transition_probs[i, j] < 0.5 else "white",
                          fontsize=9, weight='medium')
    
    ax.set_xlabel('To State', fontsize=10)
    ax.set_ylabel('From State', fontsize=10)
    ax.set_title('Transition Probability Matrix', fontsize=11, pad=10, weight='medium')
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['left'].set_color('#cccccc')
    
    return fig


def _plot_emission_matrix(
    model: HMM,
    figsize: Optional[tuple] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """Plot emission probability matrix as a heatmap."""
    if ax is None:
        if figsize is None:
            figsize = (10, 6)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Create heatmap with a more elegant colormap
    im = ax.imshow(model.emission_probs, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar with cleaner style
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Emission Probability', rotation=270, labelpad=20, fontsize=10)
    cbar.outline.set_visible(False)
    
    # Set ticks and labels
    ax.set_xticks(range(model.n_symbols))
    ax.set_yticks(range(model.n_states))
    ax.set_xticklabels(model.alphabet, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(model.state_names, fontsize=9)
    
    # Add text annotations (only if matrix is not too large)
    if model.n_states <= 10 and model.n_symbols <= 15:
        for i in range(model.n_states):
            for j in range(model.n_symbols):
                text = ax.text(j, i, f'{model.emission_probs[i, j]:.2f}',
                              ha="center", va="center",
                              color="black" if model.emission_probs[i, j] < 0.5 else "white",
                              fontsize=8, weight='medium')
    
    ax.set_xlabel('Observed Symbol', fontsize=10)
    ax.set_ylabel('Hidden State', fontsize=10)
    ax.set_title('Emission Probability Matrix', fontsize=11, pad=10, weight='medium')
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['left'].set_color('#cccccc')
    
    return fig


def _plot_initial_probs(
    model: HMM,
    figsize: Optional[tuple] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """Plot initial state probabilities as a bar chart."""
    if ax is None:
        if figsize is None:
            figsize = (8, 5)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Create bar chart with a more elegant color
    bars = ax.bar(range(model.n_states), model.initial_probs, 
                  color='#4A90E2', alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, model.initial_probs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.3f}',
                ha='center', va='bottom', fontsize=9, weight='medium')
    
    ax.set_xticks(range(model.n_states))
    ax.set_xticklabels(model.state_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_title('Initial State Probabilities', fontsize=11, pad=10, weight='medium')
    ax.set_ylim(0, max(model.initial_probs) * 1.2)
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['left'].set_color('#cccccc')
    
    return fig


def plot_mhmm(
    model: MHMM,
    which: str = 'clusters',
    figsize: Optional[tuple] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot Mixture HMM model parameters.
    
    This function visualizes Mixture HMM model parameters, including:
    - Cluster probabilities
    - Transition matrices for each cluster
    - Emission matrices for each cluster
    
    It is similar to seqHMM's plot.mhmm() function in R.
    
    Args:
        model: Fitted MHMM model object
        which: What to plot. Options:
            - 'clusters': Cluster probabilities (default)
            - 'transition': Transition matrices for all clusters
            - 'emission': Emission matrices for all clusters
            - 'all': All plots
        figsize: Figure size tuple (width, height). If None, uses default.
        ax: Optional matplotlib axes to plot on. If None, creates new figure.
        
    Returns:
        matplotlib Figure: The figure object
    """
    if model.log_likelihood is None:
        raise ValueError("Model must be fitted before plotting. Use fit_mhmm() first.")
    
    if which == 'all':
        # Create subplots for all components
        if figsize is None:
            figsize = (18, 6)
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot each component
        _plot_cluster_probs(model, ax=axes[0])
        _plot_mhmm_transitions(model, ax=axes[1])
        _plot_mhmm_emissions(model, ax=axes[2])
        
        plt.tight_layout()
        return fig
    
    elif which == 'clusters':
        return _plot_cluster_probs(model, figsize=figsize, ax=ax)
    elif which == 'transition':
        return _plot_mhmm_transitions(model, figsize=figsize, ax=ax)
    elif which == 'emission':
        return _plot_mhmm_emissions(model, figsize=figsize, ax=ax)
    else:
        raise ValueError(
            f"Unknown 'which' option: {which}. Must be 'clusters', 'transition', 'emission', or 'all'."
        )


def _plot_cluster_probs(
    model: MHMM,
    figsize: Optional[tuple] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """Plot cluster probabilities as a bar chart."""
    if ax is None:
        if figsize is None:
            figsize = (8, 5)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Create bar chart
    bars = ax.bar(range(model.n_clusters), model.cluster_probs, 
                  color='steelblue', alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, model.cluster_probs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.3f}',
                ha='center', va='bottom')
    
    ax.set_xticks(range(model.n_clusters))
    ax.set_xticklabels(model.cluster_names, rotation=45, ha='right')
    ax.set_ylabel('Probability')
    ax.set_title('Cluster Probabilities')
    ax.set_ylim(0, max(model.cluster_probs) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    return fig


def _plot_mhmm_transitions(
    model: MHMM,
    figsize: Optional[tuple] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """Plot transition matrices for all clusters."""
    if ax is None:
        if figsize is None:
            figsize = (6 * model.n_clusters, 6)
        fig, axes = plt.subplots(1, model.n_clusters, figsize=figsize)
        if model.n_clusters == 1:
            axes = [axes]
    else:
        fig = ax.figure
        axes = [ax] * model.n_clusters
    
    for k in range(model.n_clusters):
        cluster = model.clusters[k]
        trans_probs = cluster.transition_probs
        
        # Create heatmap
        im = axes[k].imshow(trans_probs, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        axes[k].set_xticks(range(cluster.n_states))
        axes[k].set_yticks(range(cluster.n_states))
        axes[k].set_xticklabels(cluster.state_names, rotation=45, ha='right', fontsize=8)
        axes[k].set_yticklabels(cluster.state_names, fontsize=8)
        
        # Add text annotations
        for i in range(cluster.n_states):
            for j in range(cluster.n_states):
                text = axes[k].text(j, i, f'{trans_probs[i, j]:.2f}',
                                  ha="center", va="center",
                                  color="black" if trans_probs[i, j] < 0.5 else "white",
                                  fontsize=7)
        
        axes[k].set_xlabel('To State')
        axes[k].set_ylabel('From State')
        axes[k].set_title(f'{model.cluster_names[k]}\nTransition Matrix')
    
    if model.n_clusters > 1:
        plt.colorbar(im, ax=axes, orientation='horizontal', pad=0.1)
    
    plt.tight_layout()
    return fig


def _plot_mhmm_emissions(
    model: MHMM,
    figsize: Optional[tuple] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """Plot emission matrices for all clusters."""
    if ax is None:
        if figsize is None:
            figsize = (6 * model.n_clusters, 6)
        fig, axes = plt.subplots(1, model.n_clusters, figsize=figsize)
        if model.n_clusters == 1:
            axes = [axes]
    else:
        fig = ax.figure
        axes = [ax] * model.n_clusters
    
    for k in range(model.n_clusters):
        cluster = model.clusters[k]
        emission_probs = cluster.emission_probs
        
        # Create heatmap
        im = axes[k].imshow(emission_probs, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        axes[k].set_xticks(range(cluster.n_symbols))
        axes[k].set_yticks(range(cluster.n_states))
        axes[k].set_xticklabels(cluster.alphabet, rotation=45, ha='right', fontsize=8)
        axes[k].set_yticklabels(cluster.state_names, fontsize=8)
        
        # Add text annotations (only if matrix is not too large)
        if cluster.n_states <= 10 and cluster.n_symbols <= 15:
            for i in range(cluster.n_states):
                for j in range(cluster.n_symbols):
                    text = axes[k].text(j, i, f'{emission_probs[i, j]:.2f}',
                                      ha="center", va="center",
                                      color="black" if emission_probs[i, j] < 0.5 else "white",
                                      fontsize=7)
        
        axes[k].set_xlabel('Observed Symbol')
        axes[k].set_ylabel('Hidden State')
        axes[k].set_title(f'{model.cluster_names[k]}\nEmission Matrix')
    
    if model.n_clusters > 1:
        plt.colorbar(im, ax=axes, orientation='horizontal', pad=0.1)
    
    plt.tight_layout()
    return fig


def _plot_hmm_network(
    model: HMM,
    figsize: Optional[tuple] = None,
    ax: Optional[plt.Axes] = None,
    vertex_size: float = 50,
    vertex_label_dist: float = 1.5,
    edge_curved: Union[bool, float] = 0.5,
    edge_label_cex: float = 0.8,
    vertex_label: str = 'initial.probs',
    loops: bool = False,
    trim: float = 1e-15,
    combine_slices: float = 0.05,
    with_legend: Union[bool, str] = 'bottom',
    layout: str = 'horizontal',
    legend_prop: float = 0.5,
    **kwargs
) -> plt.Figure:
    """
    Plot HMM as a network graph with pie chart nodes (similar to R's plot.hmm).
    
    This function creates a directed graph where:
    - Nodes are pie charts showing emission probabilities for each hidden state
    - Edges are arrows showing transition probabilities between states
    - Node labels show initial probabilities or state names
    
    Args:
        model: Fitted HMM model object
        figsize: Figure size tuple (width, height)
        ax: Optional matplotlib axes
        vertex_size: Size of vertices (nodes)
        vertex_label_dist: Distance of vertex labels from center
        edge_curved: Whether to plot curved edges (bool or float for curvature)
        edge_label_cex: Character expansion factor for edge labels
        vertex_label: Labels for vertices ('initial.probs', 'names', or custom list)
        loops: Whether to plot self-loops
        trim: Minimum transition probability to plot
        combine_slices: Emission probabilities below this are combined into 'others'
        with_legend: Whether and where to plot legend
        layout: Layout of vertices ('horizontal' or 'vertical')
        legend_prop: Proportion of figure used for legend (0-1). Default 0.5.
        **kwargs: Additional arguments
    
    Returns:
        matplotlib Figure: The figure object
    """
    # Determine if we need separate subplots for legend (like R's layout)
    use_separate_legend = (with_legend and with_legend != False and 
                          with_legend in ['bottom', 'top', 'left', 'right'])
    
    if ax is None:
        if figsize is None:
            # Adjust figsize based on legend position
            if use_separate_legend and with_legend in ['bottom', 'top']:
                figsize = (12, 8)
            elif use_separate_legend and with_legend in ['left', 'right']:
                figsize = (14, 6)
            else:
                figsize = (12, 6)
        
        # Create figure with subplots if legend is needed
        if use_separate_legend:
            if with_legend == 'bottom':
                fig = plt.figure(figsize=figsize)
                gs = fig.add_gridspec(2, 1, height_ratios=[1 - legend_prop, legend_prop], hspace=0.3)
                ax = fig.add_subplot(gs[0])
                ax_legend = fig.add_subplot(gs[1])
            elif with_legend == 'top':
                fig = plt.figure(figsize=figsize)
                gs = fig.add_gridspec(2, 1, height_ratios=[legend_prop, 1 - legend_prop], hspace=0.3)
                ax_legend = fig.add_subplot(gs[0])
                ax = fig.add_subplot(gs[1])
            elif with_legend == 'right':
                fig = plt.figure(figsize=figsize)
                gs = fig.add_gridspec(1, 2, width_ratios=[1 - legend_prop, legend_prop], wspace=0.3)
                ax = fig.add_subplot(gs[0])
                ax_legend = fig.add_subplot(gs[1])
            elif with_legend == 'left':
                fig = plt.figure(figsize=figsize)
                gs = fig.add_gridspec(1, 2, width_ratios=[legend_prop, 1 - legend_prop], wspace=0.3)
                ax_legend = fig.add_subplot(gs[0])
                ax = fig.add_subplot(gs[1])
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax_legend = None
    else:
        fig = ax.figure
        ax_legend = None
        use_separate_legend = False
    
    # Get model parameters
    n_states = model.n_states
    transition_probs = model.transition_probs.copy()
    emission_probs = model.emission_probs.copy()
    initial_probs = model.initial_probs.copy()
    
    # Get colors for observed states
    # Try to get colors from observations if available
    if hasattr(model.observations, 'color_map') and model.observations.color_map:
        colors = [model.observations.color_map.get(sym, '#808080') for sym in model.alphabet]
    else:
        # Default color palette (similar to TraMineR)
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
                         '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        colors = default_colors[:len(model.alphabet)]
        # Extend if needed
        while len(colors) < len(model.alphabet):
            colors.append('#808080')
    
    # Trim transitions (remove very small probabilities)
    transition_probs[transition_probs < trim] = 0
    
    # Remove self-loops if not requested
    if not loops:
        np.fill_diagonal(transition_probs, 0)
    
    # Calculate node positions (similar to R's layout)
    # First, determine coordinate limits (similar to R's xlim/ylim calculation)
    if layout == 'horizontal':
        x_min, x_max = -0.1, n_states - 1 + 0.1
        y_min, y_max = -0.5, 0.5
        # Horizontal layout: nodes in a line
        positions = {i: (i, 0) for i in range(n_states)}
    elif layout == 'vertical':
        x_min, x_max = -0.5, 0.5
        y_min, y_max = -0.1, n_states - 1 + 0.1
        positions = {i: (0, -i) for i in range(n_states)}
    else:
        x_min, x_max = -0.1, n_states - 1 + 0.1
        y_min, y_max = -0.5, 0.5
        positions = {i: (i, 0) for i in range(n_states)}
    
    # Scale positions to fit in plot area
    if positions:
        x_coords = [pos[0] for pos in positions.values()]
        y_coords = [pos[1] for pos in positions.values()]
        
        # Normalize and scale positions to fit within xlim/ylim
        if max(x_coords) != min(x_coords):
            x_range = max(x_coords) - min(x_coords)
            positions = {i: ((pos[0] - min(x_coords)) / x_range * (x_max - x_min) + x_min, 
                           pos[1]) for i, pos in positions.items()}
        else:
            positions = {i: (x_min + (x_max - x_min) / 2, pos[1]) 
                        for i, pos in positions.items()}
    
    # Prepare emission probabilities for pie charts
    pie_values = []
    pie_colors_list = []
    combined_slice_probs = []
    legend_labels_list = []
    legend_colors_list = []
    
    for i in range(n_states):
        emis = emission_probs[i, :].copy()
        combined_prob = 0
        
        # Combine small slices
        if combine_slices > 0:
            small_mask = emis < combine_slices
            if np.any(small_mask):
                combined_prob = np.sum(emis[small_mask])
                emis[small_mask] = 0
                if combined_prob > 0:
                    emis = np.append(emis, combined_prob)
                    pie_colors_list.append(colors + ['white'])
                    # Track which colors are used for legend
                    used_colors = [colors[j] for j in range(len(emis) - 1) if emis[j] > 0]
                    used_labels = [model.alphabet[j] for j in range(len(emis) - 1) if emis[j] > 0]
                    legend_labels_list.append(used_labels + ['others'])
                    legend_colors_list.append(used_colors + ['white'])
                else:
                    pie_colors_list.append(colors)
                    legend_labels_list.append([model.alphabet[j] for j in range(len(emis)) if emis[j] > 0])
                    legend_colors_list.append([colors[j] for j in range(len(emis)) if emis[j] > 0])
            else:
                pie_colors_list.append(colors)
                legend_labels_list.append([model.alphabet[j] for j in range(len(emis)) if emis[j] > 0])
                legend_colors_list.append([colors[j] for j in range(len(emis)) if emis[j] > 0])
        else:
            pie_colors_list.append(colors)
            legend_labels_list.append([model.alphabet[j] for j in range(len(emis)) if emis[j] > 0])
            legend_colors_list.append([colors[j] for j in range(len(emis)) if emis[j] > 0])
        
        # Remove zero probabilities
        non_zero_mask = emis > 0
        pie_values.append(emis[non_zero_mask])
        combined_slice_probs.append(combined_prob)
    
    # Collect unique legend items (by appearance order, like R)
    if use_separate_legend:
        unique_labels = []
        unique_colors = []
        seen = set()
        for labels, cols in zip(legend_labels_list, legend_colors_list):
            for label, col in zip(labels, cols):
                if (label, col) not in seen:
                    unique_labels.append(label)
                    unique_colors.append(col)
                    seen.add((label, col))
        # Add 'others' if needed
        if combine_slices > 0 and any(combined_slice_probs):
            if 'others' not in unique_labels:
                unique_labels.append('others')
                unique_colors.append('white')
    
    # Calculate node radius in data coordinates
    # Convert vertex_size (in points) to data coordinates
    # R uses vertex.size directly in the plot coordinate system
    # We'll use a reasonable scaling factor based on the coordinate range
    if layout == 'horizontal':
        # Estimate data coordinate range
        data_range = max(x_max - x_min, 1.0)
        # Convert vertex_size to data coordinates
        # Scale factor: vertex_size of 50 should be about 0.2-0.25 of the spacing between nodes
        # For horizontal layout, spacing is approximately (x_max - x_min) / max(n_states - 1, 1)
        spacing = (x_max - x_min) / max(n_states - 1, 1) if n_states > 1 else (x_max - x_min)
        node_radius = (vertex_size / 50.0) * spacing * 0.25
    else:
        data_range = max(y_max - y_min, 1.0)
        spacing = (y_max - y_min) / max(n_states - 1, 1) if n_states > 1 else (y_max - y_min)
        node_radius = (vertex_size / 50.0) * spacing * 0.25
    
    # Draw edges (transitions) first (so they appear behind nodes)
    edge_widths = []
    edge_labels = {}
    edges_to_draw = []
    
    # Get all non-zero transitions
    transitions = []
    for i in range(n_states):
        for j in range(n_states):
            prob = transition_probs[i, j]
            if prob > 0:
                edges_to_draw.append((i, j))
                transitions.append(prob)
    
    # Calculate edge widths (similar to R: transitions * (7 / max(transitions)))
    if transitions:
        max_trans = max(transitions)
        edge_widths = [t * (7.0 / max_trans) if max_trans > 0 else 1.0 for t in transitions]
        # Format edge labels
        for (i, j), prob in zip(edges_to_draw, transitions):
            if prob >= 0.001 or prob == 0:
                edge_labels[(i, j)] = f'{prob:.3f}'
            else:
                edge_labels[(i, j)] = f'{prob:.2e}'
    else:
        edge_widths = [1.0] * len(edges_to_draw)
    
    # Draw edges
    for (i, j), width in zip(edges_to_draw, edge_widths):
        x1, y1 = positions[i]
        x2, y2 = positions[j]
        
        # Calculate arrow properties
        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist > 0:
            # Normalize direction
            dx_norm = dx / dist
            dy_norm = dy / dist
            
            # Adjust start/end to account for node radius
            start_x = x1 + dx_norm * node_radius
            start_y = y1 + dy_norm * node_radius
            end_x = x2 - dx_norm * node_radius
            end_y = y2 - dy_norm * node_radius
            
            # Curved edge
            if edge_curved and (isinstance(edge_curved, bool) or edge_curved != 0):
                curvature = edge_curved if isinstance(edge_curved, (int, float)) else 0.5
                # Create curved path
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2
                # Perpendicular direction for curve
                perp_x = -dy_norm * curvature * dist * 0.3
                perp_y = dx_norm * curvature * dist * 0.3
                control_x = mid_x + perp_x
                control_y = mid_y + perp_y
                
                # Use quadratic bezier curve
                from matplotlib.path import Path
                path_data = [
                    (Path.MOVETO, (start_x, start_y)),
                    (Path.CURVE3, (control_x, control_y)),
                    (Path.CURVE3, (end_x, end_y)),
                ]
                codes, verts = zip(*path_data)
                path = Path(verts, codes)
                
                arrow = FancyArrowPatch(
                    path=path,
                    arrowstyle='->',
                    lw=max(width, 0.8),
                    color='#666666',
                    alpha=0.8,
                    zorder=1,
                    mutation_scale=15
                )
            else:
                # Straight edge
                arrow = FancyArrowPatch(
                    (start_x, start_y),
                    (end_x, end_y),
                    arrowstyle='->',
                    lw=max(width, 0.8),
                    color='#666666',
                    alpha=0.8,
                    zorder=1,
                    mutation_scale=15
                )
            
            ax.add_patch(arrow)
            
            # Add edge label
            if (i, j) in edge_labels:
                label_x = (start_x + end_x) / 2
                label_y = (start_y + end_y) / 2
                if edge_curved and (isinstance(edge_curved, bool) or edge_curved != 0):
                    curvature = edge_curved if isinstance(edge_curved, (int, float)) else 0.5
                    perp_x = -dy_norm * curvature * dist * 0.3
                    perp_y = dx_norm * curvature * dist * 0.3
                    label_x += perp_x * 0.5
                    label_y += perp_y * 0.5
                
                ax.text(label_x, label_y, edge_labels[(i, j)],
                       fontsize=9 * edge_label_cex,
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                alpha=0.9, edgecolor='gray', linewidth=0.5),
                       zorder=3)
    
    # Draw nodes (pie charts)
    for i in range(n_states):
        x, y = positions[i]
        emis = pie_values[i]
        node_colors = pie_colors_list[i][:len(emis)]
        
        # Draw pie chart
        if len(emis) > 0 and np.sum(emis) > 0:
            # Normalize to sum to 1
            emis_norm = emis / np.sum(emis)
            angles = np.cumsum(emis_norm * 2 * np.pi)
            angles = np.insert(angles, 0, 0)
            
            # Draw wedges
            for j in range(len(emis_norm)):
                if emis_norm[j] > 0:
                    theta1 = angles[j] * 180 / np.pi
                    theta2 = angles[j + 1] * 180 / np.pi
                    wedge = Wedge((x, y), node_radius, theta1, theta2,
                                 facecolor=node_colors[j], edgecolor='black', 
                                 linewidth=2, zorder=2)
                    ax.add_patch(wedge)
        
        # Add node label
        if vertex_label == 'initial.probs':
            label_text = f'{initial_probs[i]:.2f}'
        elif vertex_label == 'names':
            label_text = model.state_names[i] if i < len(model.state_names) else f'State {i+1}'
        else:
            if isinstance(vertex_label, list) and i < len(vertex_label):
                label_text = str(vertex_label[i])
            else:
                label_text = f'{initial_probs[i]:.2f}'
        
        # Position label (similar to R's vertex.label.dist)
        if isinstance(vertex_label_dist, (int, float)) and vertex_label_dist != 'auto':
            label_dist = vertex_label_dist
        else:
            # Auto: place outside vertex
            label_dist = node_radius * 1.4
        
        if layout == 'horizontal':
            label_x = x
            label_y = y - label_dist
        else:
            label_x = x - label_dist
            label_y = y
        
        ax.text(label_x, label_y, label_text,
               fontsize=11, ha='center', va='top' if layout == 'horizontal' else 'center',
               weight='bold', zorder=4)
    
    # Set axis properties
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Set limits (matching R's xlim/ylim)
    if layout == 'horizontal':
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    
    # Add legend in separate subplot if requested
    if use_separate_legend and ax_legend is not None:
        ax_legend.axis('off')
        # Create legend elements
        legend_elements = [mpatches.Patch(facecolor=col, edgecolor='black', linewidth=1, label=label)
                          for col, label in zip(unique_colors, unique_labels)]
        
        # Calculate number of columns
        ncol = min(len(unique_labels), 6) if with_legend in ['bottom', 'top'] else 1
        
        ax_legend.legend(handles=legend_elements, loc='center', 
                        ncol=ncol, frameon=True, fontsize=10, 
                        handlelength=1.5, handletextpad=0.5)
    elif with_legend and with_legend != False and not use_separate_legend:
        # Fallback: use regular legend (may overlap)
        legend_labels = list(model.alphabet)
        if combine_slices > 0 and any(combined_slice_probs):
            legend_labels.append('others')
            legend_colors = colors + ['white']
        else:
            legend_colors = colors
        
        legend_elements = [mpatches.Patch(facecolor=col, edgecolor='black', label=label)
                          for col, label in zip(legend_colors[:len(legend_labels)], legend_labels)]
        
        if with_legend == 'bottom' or with_legend == True:
            ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1),
                     ncol=min(len(legend_labels), 6), frameon=True, fontsize=9)
        elif with_legend == 'top':
            ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1),
                     ncol=min(len(legend_labels), 6), frameon=True, fontsize=9)
    
    plt.tight_layout()
    return fig
