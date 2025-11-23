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
from typing import Optional, List
from .hmm import HMM
from .mhmm import MHMM


def plot_hmm(
    model: HMM,
    which: str = 'transition',
    figsize: Optional[tuple] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot HMM model parameters.
    
    This function visualizes HMM model parameters, including:
    - Transition probability matrix
    - Emission probability matrix
    - Initial state probabilities
    
    It is similar to seqHMM's plot.hmm() function in R.
    
    Args:
        model: Fitted HMM model object
        which: What to plot. Options:
            - 'transition': Transition probability matrix (default)
            - 'emission': Emission probability matrix
            - 'initial': Initial state probabilities
            - 'all': All three plots
        figsize: Figure size tuple (width, height). If None, uses default.
        ax: Optional matplotlib axes to plot on. If None, creates new figure.
        
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
        >>> # Plot transition matrix
        >>> plot_hmm(hmm, which='transition')
        >>> plt.show()
        >>> 
        >>> # Plot emission matrix
        >>> plot_hmm(hmm, which='emission')
        >>> plt.show()
    """
    if model.log_likelihood is None:
        raise ValueError("Model must be fitted before plotting. Use fit_model() first.")
    
    if which == 'all':
        # Create subplots for all three
        if figsize is None:
            figsize = (15, 5)
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
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
        raise ValueError(f"Unknown 'which' option: {which}. Must be 'transition', 'emission', 'initial', or 'all'.")


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
    
    # Create heatmap
    im = ax.imshow(model.transition_probs, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Transition Probability', rotation=270, labelpad=20)
    
    # Set ticks and labels
    ax.set_xticks(range(model.n_states))
    ax.set_yticks(range(model.n_states))
    ax.set_xticklabels(model.state_names, rotation=45, ha='right')
    ax.set_yticklabels(model.state_names)
    
    # Add text annotations
    for i in range(model.n_states):
        for j in range(model.n_states):
            text = ax.text(j, i, f'{model.transition_probs[i, j]:.2f}',
                          ha="center", va="center", color="black" if model.transition_probs[i, j] < 0.5 else "white")
    
    ax.set_xlabel('To State')
    ax.set_ylabel('From State')
    ax.set_title('Transition Probability Matrix')
    
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
    
    # Create heatmap
    im = ax.imshow(model.emission_probs, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Emission Probability', rotation=270, labelpad=20)
    
    # Set ticks and labels
    ax.set_xticks(range(model.n_symbols))
    ax.set_yticks(range(model.n_states))
    ax.set_xticklabels(model.alphabet, rotation=45, ha='right')
    ax.set_yticklabels(model.state_names)
    
    # Add text annotations (only if matrix is not too large)
    if model.n_states <= 10 and model.n_symbols <= 15:
        for i in range(model.n_states):
            for j in range(model.n_symbols):
                text = ax.text(j, i, f'{model.emission_probs[i, j]:.2f}',
                              ha="center", va="center",
                              color="black" if model.emission_probs[i, j] < 0.5 else "white",
                              fontsize=8)
    
    ax.set_xlabel('Observed Symbol')
    ax.set_ylabel('Hidden State')
    ax.set_title('Emission Probability Matrix')
    
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
    
    # Create bar chart
    bars = ax.bar(range(model.n_states), model.initial_probs, color='steelblue', alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, model.initial_probs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.3f}',
                ha='center', va='bottom')
    
    ax.set_xticks(range(model.n_states))
    ax.set_xticklabels(model.state_names, rotation=45, ha='right')
    ax.set_ylabel('Probability')
    ax.set_title('Initial State Probabilities')
    ax.set_ylim(0, max(model.initial_probs) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
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
