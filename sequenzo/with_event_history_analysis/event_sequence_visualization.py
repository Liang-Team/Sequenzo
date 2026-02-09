"""
@Author  : Yuqi Liang 梁彧祺
@File    : event_sequence_visualization.py
@Time    : 09/02/2026 16:39
@Desc    : Event Sequence Analysis  Visualization

Event Sequence Visualization Module for Sequenzo

TraMineR equivalents:
- plot_event_sequences()  -> plot.seqelist()
- plot_subsequence_frequencies() -> plot.subseqelist()

This module is separate from event_sequence.py to keep visualization logic
in one place and core event sequence types in another.
"""

from __future__ import annotations

from typing import Union, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import types from event_sequence to avoid circular imports when __init__ pulls from both modules
from .event_sequence import EventSequenceList, SubsequenceList


def plot_event_sequences(
    eseq: EventSequenceList,
    type: str = "index",
    group: Optional[Union[list, "np.ndarray"]] = None,
    top_n: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
    save_as: Optional[str] = None,
    dpi: int = 200,
    **kwargs,
) -> "plt.Figure":
    """
    Plot event sequences.

    TraMineR equivalent: plot.seqelist()

    Args:
        eseq: EventSequenceList object
        type: Plot type ("index" for index plot, "parallel" for parallel coordinates)
        group: Optional group membership for coloring sequences
        top_n: Optional number of sequences to plot (default: all)
        title: Plot title
        figsize: Figure size (width, height)
        save_as: Optional filename to save the plot
        dpi: Resolution for saved figure
        **kwargs: Additional plotting arguments

    Returns:
        matplotlib Figure object
    """
    if type == "index":
        return _plot_event_index(eseq, group, top_n, title, figsize, save_as, dpi, **kwargs)
    elif type == "parallel":
        return _plot_event_parallel_coordinates(
            eseq, group, top_n, title, figsize, save_as, dpi, **kwargs
        )
    else:
        raise ValueError(f"Unknown plot type: {type}. Use 'index' or 'parallel'")


def plot_subsequence_frequencies(
    subseq: SubsequenceList,
    top_n: Optional[int] = None,
    use_count: bool = False,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    save_as: Optional[str] = None,
    dpi: int = 200,
    **kwargs,
) -> "plt.Figure":
    """
    Plot frequencies of subsequences.

    TraMineR equivalent: plot.subseqelist()

    Args:
        subseq: SubsequenceList object
        top_n: Number of top subsequences to plot (default: all or 20)
        use_count: If True, use Count instead of Support
        title: Plot title
        figsize: Figure size (width, height)
        save_as: Optional filename to save the plot
        dpi: Resolution for saved figure
        **kwargs: Additional plotting arguments (e.g., cex for text size)

    Returns:
        matplotlib Figure object
    """
    if len(subseq) == 0:
        raise ValueError("SubsequenceList is empty")

    if top_n is None:
        top_n = min(20, len(subseq))
    else:
        top_n = min(top_n, len(subseq))

    if use_count:
        y_values = subseq.data["Count"].values[:top_n]
        y_label = "Count"
    else:
        y_values = subseq.data["Support"].values[:top_n]
        y_label = "Support"

    subseq_labels = []
    for i in range(top_n):
        subseq_str = subseq.subsequences[i].to_string()
        subseq_labels.append(subseq_str if subseq_str else f"Subseq_{i+1}")

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(top_n), y_values, **kwargs)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(subseq_labels, fontsize=kwargs.get("cex", 1) * 10)
    ax.set_xlabel(y_label, fontsize=kwargs.get("cex", 1) * 12)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    if title is None:
        title = f"Subsequence Frequencies (Top {top_n})"
    ax.set_title(title, fontsize=kwargs.get("cex", 1) * 14, fontweight="bold")
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=dpi, bbox_inches="tight")
    return fig


def _plot_event_index(
    eseq: EventSequenceList,
    group: Optional[Union[list, np.ndarray]] = None,
    top_n: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
    save_as: Optional[str] = None,
    dpi: int = 200,
    **kwargs,
) -> "plt.Figure":
    """Plot event sequences as index plot (timeline view). When group is set, one subplot per group."""
    n_seqs = len(eseq)
    if top_n is not None:
        n_seqs = min(top_n, n_seqs)
    group_arr = np.asarray(group) if group is not None else None

    if group_arr is not None and len(group_arr) >= n_seqs:
        # One subplot per group (TraMineR-style: plot(eseq, group=...))
        group_sub = group_arr[:n_seqs]
        uniq = np.unique(group_sub)
        n_axes = len(uniq)
        if n_axes == 0:
            n_axes = 1
            fig, _ax = plt.subplots(1, 1, figsize=figsize)
            axes_list = [_ax]
            uniq = None
        else:
            ncols = min(n_axes, 3)
            nrows = (n_axes + ncols - 1) // ncols
            fig, axes_flat = plt.subplots(nrows, ncols, figsize=(figsize[0] * ncols / max(ncols, 1), figsize[1] * nrows))
            axes_list = np.atleast_1d(axes_flat).ravel()
    else:
        n_axes = 1
        fig, ax_one = plt.subplots(1, 1, figsize=figsize)
        axes_list = [ax_one]
        uniq = None

    n_events = len(eseq.dictionary)
    colors = plt.cm.tab20(np.linspace(0, 1, n_events))

    for ax_idx, ax in enumerate(axes_list[: n_axes if uniq is not None else 1]):
        if uniq is not None:
            g = uniq[ax_idx]
            inds = np.where(group_sub == g)[0]
            sub_title = f"Group {g} (n={len(inds)})"
        else:
            inds = np.arange(n_seqs)
            sub_title = title if title else f"Event Sequence Index Plot (n={n_seqs})"

        n_here = len(inds)
        for row, i in enumerate(inds):
            seq = eseq.sequences[i]
            y_pos = n_here - row - 1
            if len(seq.timestamps) == 0:
                continue
            for j, (t, e) in enumerate(zip(seq.timestamps, seq.events)):
                color = colors[(e - 1) % len(colors)]
                ax.scatter(t, y_pos, c=[color], s=50, zorder=3)
                if j < len(seq.timestamps) - 1:
                    next_t = seq.timestamps[j + 1]
                    ax.plot(
                        [t, next_t],
                        [y_pos, y_pos],
                        color=color,
                        linewidth=2,
                        alpha=0.5,
                        zorder=1,
                    )
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Sequence Index", fontsize=12)
        ax.set_yticks(range(n_here))
        ax.set_yticklabels([f"Seq {inds[r]+1}" for r in range(n_here)])
        ax.set_title(sub_title, fontsize=14, fontweight="bold")
        if ax_idx == 0:
            legend_elements = [
                mpatches.Patch(facecolor=colors[i], label=eseq.dictionary[i])
                for i in range(n_events)
            ]
            ax.legend(handles=legend_elements, loc="best", fontsize=10)

    # Hide unused subplots when we have multiple groups
    for ax in axes_list[n_axes:]:
        ax.set_visible(False)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=dpi, bbox_inches="tight")
    return fig


def _plot_event_parallel_coordinates(
    eseq: EventSequenceList,
    group: Optional[Union[list, "np.ndarray"]] = None,
    top_n: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
    save_as: Optional[str] = None,
    dpi: int = 200,
    **kwargs,
) -> "plt.Figure":
    """
    Plot event sequences as parallel coordinates (TraMineR seqpcplot style).

    X-axis = Position (1, 2, 3, ...) within each sequence.
    Y-axis = Event type (categorical, labels from dictionary).
    Each sequence is one line connecting (position, event) points.
    """
    n_seqs = len(eseq)
    if top_n is not None:
        n_seqs = min(top_n, n_seqs)
    sequences = eseq.sequences[:n_seqs]
    dictionary = eseq.dictionary
    n_events = len(dictionary)

    # Build (position, event_code) per sequence. Position = 1, 2, 3, ...
    trajectories = []
    for seq in sequences:
        if len(seq.timestamps) == 0:
            continue
        positions = np.arange(1, len(seq.timestamps) + 1, dtype=np.float64)
        events = np.asarray(seq.events, dtype=np.int32)  # 1-based codes
        trajectories.append((positions, events))
    if not trajectories:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel("Position", fontsize=12)
        ax.set_ylabel("Event", fontsize=12)
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save_as:
            plt.savefig(save_as, dpi=dpi, bbox_inches="tight")
        return fig

    max_pos = max(len(t[0]) for t in trajectories)
    x_min, x_max = 0.5, max_pos + 0.5
    y_min, y_max = 0.5, n_events + 0.5

    fig, ax = plt.subplots(figsize=figsize)
    # TraMineR-style: small gray square per (position, event), not full cell (grid.scale ~ 1/5)
    grid_scale = 0.2
    half = np.sqrt(grid_scale) / 2  # half-side of each small gray block

    # One small gray block per (position, event) — background stays white
    for xi in range(1, max_pos + 1):
        for yi in range(1, n_events + 1):
            rect = plt.Rectangle(
                (xi - half, yi - half),
                2 * half,
                2 * half,
                facecolor=(0.95, 0.95, 0.95),
                edgecolor=(0.6, 0.6, 0.6),
                linewidth=0.5,
                zorder=0,
            )
            ax.add_patch(rect)

    # Color palette for lines (one color per trajectory type or per sequence)
    try:
        from matplotlib.colors import to_rgba
        _dark2 = plt.cm.get_cmap("Set2")
        if _dark2 is None:
            _dark2 = plt.cm.get_cmap("tab10")
    except Exception:
        _dark2 = plt.cm.tab10
    n_traj = len(trajectories)
    colors = [_dark2(i % 8) for i in range(n_traj)]

    # Draw lines and points for each sequence (wider line by default for visibility)
    lw = kwargs.get("linewidth", kwargs.get("lw", 2.0))
    for idx, (positions, events) in enumerate(trajectories):
        color = colors[idx % len(colors)]
        # Line connecting (position, event) — event is 1-based, plot as y = event
        ax.plot(
            positions,
            events,
            color=color,
            linewidth=lw,
            alpha=0.7,
            zorder=2,
        )
        # Points at each (position, event)
        ax.scatter(
            positions,
            events,
            c=[color],
            s=25,
            edgecolors=(0.4, 0.4, 0.4),
            linewidths=0.5,
            zorder=3,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel("", fontsize=12)
    ax.set_yticks(np.arange(1, n_events + 1))
    ax.set_yticklabels(dictionary, fontsize=kwargs.get("cex", 1) * 9)
    ax.set_xticks(np.arange(1, max_pos + 1))
    if title is None:
        title = f"Event Sequence Parallel Coordinates (n={n_seqs})"
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=dpi, bbox_inches="tight")
    return fig
