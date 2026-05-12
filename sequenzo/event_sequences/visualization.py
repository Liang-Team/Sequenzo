"""
@Author  : Yuqi Liang 梁彧祺
@File    : visualization.py
@Time    : 05/01/2026 07:27
@Desc    : 

Event-sequence visualization utilities.

This module provides intuitive plotting APIs for event-sequence analysis
while preserving TraMineR-like behavior.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import BoundaryNorm
from matplotlib import colors as mcolors
from matplotlib.transforms import blended_transform_factory

from .core import EventSequenceList, SubsequenceList


def _as_array_or_none(x):
    return None if x is None else np.asarray(x)


def _finalize_figure(fig, save_as: Optional[str] = None, dpi: int = 200, show: bool = False):
    """
    Shared post-processing for event-sequence plots.

    - save_as: optional output path; appends ".png" when no known extension is provided.
    - show: if True, call plt.show() before returning.
    """
    if save_as:
        if not save_as.lower().endswith((".png", ".jpg", ".jpeg", ".pdf", ".svg")):
            save_as = save_as + ".png"
        fig.savefig(save_as, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _get_event_colors(dictionary: Sequence[str], cpal=None, seqelist: Optional[EventSequenceList] = None):
    """
    Build a stable event->color mapping.

    Priority:
    1) User-provided `cpal`
    2) SequenceData default palette logic (for visual consistency across package)
    3) Matplotlib fallback
    """
    n = len(dictionary)

    if isinstance(cpal, dict):
        # Explicit event->color map from caller
        missing = [ev for ev in dictionary if ev not in cpal]
        if missing:
            raise ValueError(f"cpal dict is missing colors for events: {missing}")
        cols = [cpal[ev] for ev in dictionary]
    elif cpal is not None:
        cols = list(cpal)
        if len(cols) < n:
            raise ValueError("cpal has fewer colors than number of events.")
    else:
        # If EventSequenceData constructors attached state-consistent event colors, use them first.
        if seqelist is not None and hasattr(seqelist, "event_color_map"):
            cmap = getattr(seqelist, "event_color_map", {}) or {}
            if all(ev in cmap for ev in dictionary):
                cols = [cmap[ev] for ev in dictionary]
                return {ev: cols[i] for i, ev in enumerate(dictionary)}

        # Reuse SequenceData default palette generator so state/event visuals stay aligned.
        try:
            from sequenzo.define_sequence_data import SequenceData

            cols = SequenceData.get_default_color_palette(
                n_states=n,
                reverse_colors=True,
                palette_name="default",
                return_format="rgb",
            )
        except Exception:
            # Safe fallback if import/environment issues occur.
            cmap = plt.get_cmap("tab20")
            cols = [cmap(i % 20) for i in range(n)]

    return {ev: cols[i] for i, ev in enumerate(dictionary)}


def _event_code_to_name(code: int, dictionary: Sequence[str]) -> str:
    if 1 <= int(code) <= len(dictionary):
        return dictionary[int(code) - 1]
    return f"Event{int(code)}"


def _apply_axis_style(
    ax,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    fontsize: int = 11,
    hide_y_axis: bool = False,
):
    """
    Apply Sequenzo-style axis aesthetics (aligned with plot_sequence_index).
    """
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("gray")
    ax.spines["bottom"].set_linewidth(0.7)
    ax.spines["bottom"].set_position(("outward", 5))

    if hide_y_axis:
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.spines["left"].set_visible(False)
    else:
        ax.spines["left"].set_color("gray")
        ax.spines["left"].set_linewidth(0.7)
        ax.spines["left"].set_position(("outward", 5))

    ax.tick_params(axis="x", colors="gray", length=4, width=0.7, which="major", direction="out")
    if not hide_y_axis:
        ax.tick_params(axis="y", colors="gray", length=4, width=0.7, which="major", direction="out")

    ax.xaxis.set_ticks_position("bottom")
    if not hide_y_axis:
        ax.yaxis.set_ticks_position("left")

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=10, color="black")
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=10, color="black")


def _seqpcplot_y_margin_frac(n_events: int, fontsize: int, max_label_chars: int) -> float:
    """Extra figure width on the left so swatches + long tick labels are not clipped."""
    fs = float(fontsize)
    nc = max(int(max_label_chars), 6)
    ne = max(int(n_events), 4)
    return float(
        np.clip(
            0.052 + nc * 0.0041 + (fs - 9.0) * 0.0038 + ne * 0.00065,
            0.055,
            0.22,
        )
    )


def _add_seqpcplot_y_axis_swatches(
    ax,
    dictionary: Sequence[str],
    event_colors: dict,
    fontsize: int,
    *,
    max_label_chars: int,
) -> None:
    """Draw color squares in the left margin; spacing scales with label length and font size."""
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ny = len(dictionary)
    nc = max(int(max_label_chars), 6)
    fs = float(fontsize)

    sw_h = float(np.clip(0.82 - 0.024 * ny, 0.34, 0.62))
    sw_w = float(np.clip(0.013 + 0.00035 * fs, 0.013, 0.021))

    # Gap between plot spine and swatch right edge (axes coords): grows with long names.
    gap_plot_to_swatch = float(
        np.clip(0.010 + 0.0033 * nc + 0.0022 * (fs - 9.0), 0.010, 0.085)
    )
    # Gap between swatch left edge and tick labels (via tick pad below).
    x_right = -gap_plot_to_swatch
    x0 = x_right - sw_w

    for yi, name in enumerate(dictionary, start=1):
        fc = event_colors.get(name, "#888888")
        if isinstance(fc, str):
            fc = _normalize_color_name(fc)
        if isinstance(fc, str) and not mcolors.is_color_like(fc):
            fc = "#888888"
        ax.add_patch(
            plt.Rectangle(
                (x0, yi - sw_h / 2),
                sw_w,
                sw_h,
                transform=trans,
                facecolor=fc,
                edgecolor="0.38",
                linewidth=0.35,
                clip_on=False,
                zorder=25,
            )
        )

    pad_pts = int(np.clip(10 + 0.62 * fs + 1.55 * nc + 0.35 * ny, 14, 56))
    ax.tick_params(axis="y", pad=pad_pts)


def _normalize_color_name(color):
    """
    Normalize color names with lightweight R-style compatibility.

    Example:
    - "grey80" / "gray80" -> grayscale string "0.8" (valid in matplotlib)
    """
    if not isinstance(color, str):
        return color
    c = color.strip().lower()
    if c.startswith("grey") or c.startswith("gray"):
        num = c[4:]
        if num.isdigit():
            v = int(num)
            if 0 <= v <= 100:
                return str(v / 100.0)
    return color


def plot_event_parallel_coordinates(
    event_sequences: EventSequenceList,
    group_labels: Optional[Union[Sequence, np.ndarray]] = None,
    color_palette=None,
    event_labels_order: Optional[Sequence[str]] = None,
    order_align: str = "first",
    title: Union[str, None] = "auto",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    rows: int = 1,
    cols: Optional[int] = None,
    figsize: Tuple[float, float] = (12, 8),
    linewidth: float = 3.0,
    alpha: float = 0.45,
    fontsize: int = 11,
    grid_scale: float = 1 / 5,
    hidden_color: str = "grey80",
    seed: Optional[int] = 1,
    line_filter=None,
    line_order: Optional[str] = None,
    line_course: str = "upwards",
    save_as: Optional[str] = None,
    dpi: int = 200,
    show: bool = False,
):
    """
    Parallel-coordinate style plot for event sequences.

    Event colors use small squares in the left margin; spacing between swatches,
    tick labels, and the plot grows with the longest event name and font size
    (no separate figure legend).

    TraMineR parameter mapping: ``event_sequences`` -> ``seqelist``,
    ``group_labels`` -> ``group``, ``color_palette`` -> ``cpal``,
    ``event_labels_order`` -> ``alphabet``, ``title`` -> ``main``,
    ``x_label`` -> ``xlab``, ``y_label`` -> ``ylab``.
    """
    if not isinstance(event_sequences, EventSequenceList):
        raise TypeError("event_sequences must be an EventSequenceList.")
    if line_course not in {"upwards", "downwards"}:
        raise ValueError("line_course must be 'upwards' or 'downwards'.")
    if line_order is None:
        line_order = "foreground" if line_filter is not None else "background"
    if line_order not in {"background", "foreground"}:
        raise ValueError("line_order must be 'background' or 'foreground'.")

    group_arr = _as_array_or_none(group_labels)
    if group_arr is not None and len(group_arr) != len(event_sequences):
        raise ValueError("group_labels length must match number of sequences.")

    dictionary = list(
        event_sequences.dictionary if event_labels_order is None else event_labels_order
    )
    event_colors = _get_event_colors(dictionary, cpal=color_palette, seqelist=event_sequences)
    _max_label_chars = max((len(str(e)) for e in dictionary), default=8)

    if group_arr is None:
        group_levels = [None]
    else:
        group_levels = list(np.unique(group_arr))

    n_panels = len(group_levels)
    if cols is None:
        cols = min(3, n_panels)
    rows = max(rows, int(np.ceil(n_panels / cols)))

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    rng = np.random.default_rng(seed)

    for panel_idx, g in enumerate(group_levels):
        ax = axes_flat[panel_idx]
        indices = np.arange(len(event_sequences)) if g is None else np.where(group_arr == g)[0]

        # Build trajectories and aggregate by identical pattern (TraMineR-like "unique" ltype).
        trajs = []
        for i in indices:
            seq = event_sequences[i]
            if len(seq.events) == 0:
                continue

            if order_align == "time":
                x = np.asarray(seq.timestamps, dtype=float)
            elif order_align == "last":
                x = np.arange(-len(seq.events) + 1, 1, dtype=float)
            else:
                x = np.arange(1, len(seq.events) + 1, dtype=float)

            y = np.asarray(seq.events, dtype=int)
            key = tuple(zip(np.round(x, 6), y.tolist()))
            w = float(event_sequences.weights[i]) if hasattr(event_sequences, "weights") else 1.0
            trajs.append((key, x, y, w))

        if not trajs:
            ax.set_ylim(0.5, len(dictionary) + 0.5)
            ax.set_yticks(np.arange(1, len(dictionary) + 1))
            ax.set_yticklabels(dictionary, fontsize=fontsize - 2, color="black")
            _apply_axis_style(
                ax,
                xlabel=("position" if x_label is None else x_label),
                ylabel=("event" if y_label is None else y_label),
                fontsize=fontsize,
            )
            _add_seqpcplot_y_axis_swatches(
                ax, dictionary, event_colors, fontsize, max_label_chars=_max_label_chars
            )
            continue

        agg = {}
        for key, x, y, w in trajs:
            if key not in agg:
                agg[key] = {"x": x, "y": y, "w": 0.0}
            agg[key]["w"] += w

        max_w = max(v["w"] for v in agg.values()) if agg else 1.0
        max_x = int(max(np.max(v["x"]) for v in agg.values()))
        ny = len(dictionary)
        total_w = sum(v["w"] for v in agg.values()) if agg else 1.0

        # Layer 1: background square grid (the "different boxes" visual language).
        side = np.sqrt(grid_scale)
        half = side / 2
        for xi in range(1, max_x + 1):
            for yi in range(1, ny + 1):
                rect = plt.Rectangle(
                    (xi - half, yi - half),
                    side,
                    side,
                    facecolor="gainsboro",
                    edgecolor="gray",
                    linewidth=0.8,
                    alpha=0.45,
                    zorder=0,
                )
                ax.add_patch(rect)

        # Layer 2 + 3: frequency-weighted jittered lines and squares.
        # Draw order follows TraMineR's lorder logic.
        traj_items = sorted(agg.values(), key=lambda d: d["w"], reverse=(line_order == "background"))
        traj_palette = plt.get_cmap("Dark2")(np.linspace(0, 1, max(3, len(traj_items))))

        # Filter behavior close to TraMineR:
        # - numeric: minfreq(level)
        # - list/dict with {"type":"function","value":"minfreq|cumfreq|linear","level":...}
        # - "linear" produces continuous desaturation
        prop = np.array([item["w"] / total_w for item in traj_items], dtype=float)
        color_strength = np.ones(len(traj_items), dtype=float)
        if line_filter is not None and len(traj_items) > 0:
            ftype = "function"
            fvalue = "minfreq"
            flevel = None

            if isinstance(line_filter, (int, float)):
                fvalue = "minfreq"
                flevel = float(line_filter)
            elif isinstance(line_filter, dict):
                ftype = line_filter.get("type", "function")
                fvalue = line_filter.get("value", "minfreq")
                flevel = line_filter.get("level", None)
            elif isinstance(line_filter, str):
                fvalue = line_filter

            if ftype == "function":
                if fvalue == "linear":
                    # Scale to [0, 1], keep high-frequency colorful
                    pmin, pmax = float(np.min(prop)), float(np.max(prop))
                    if pmax > pmin:
                        color_strength = (prop - pmin) / (pmax - pmin)
                    else:
                        color_strength = np.ones_like(prop)
                elif fvalue == "cumfreq":
                    level = 0.5 if flevel is None else float(flevel)
                    level = min(max(level, 0.0), 1.0)
                    order_desc = np.argsort(prop)[::-1]
                    keep = np.zeros(len(prop), dtype=bool)
                    csum = 0.0
                    for idx_desc in order_desc:
                        keep[idx_desc] = True
                        csum += prop[idx_desc]
                        if csum >= level:
                            break
                    color_strength = keep.astype(float)
                else:
                    # Default minfreq
                    level = 0.05 if flevel is None else float(flevel)
                    color_strength = (prop >= level).astype(float)

        for t_idx, item in enumerate(traj_items):
            x = item["x"]
            y = item["y"].astype(float)
            rel = 0.15 + 0.85 * np.sqrt(item["w"] / max_w)

            # Deterministic tiny jitter inside each square
            jx = rng.uniform(-0.18, 0.18, size=len(x))
            jy = rng.uniform(-0.18, 0.18, size=len(x))
            xj = x + jx
            yj = y + jy

            # lcourse control for simultaneous x positions.
            if len(xj) > 1:
                order_idx = np.arange(len(xj))
                # stable sort by x, then y direction
                if line_course == "downwards":
                    order_idx = np.lexsort((yj, xj))
                else:
                    order_idx = np.lexsort((-yj, xj))
                xj = xj[order_idx]
                yj = yj[order_idx]
                y_base = item["y"][order_idx]
            else:
                y_base = item["y"]

            traj_col = traj_palette[t_idx % len(traj_palette)]
            strength = float(color_strength[t_idx])
            if strength <= 0:
                line_col = _normalize_color_name(hidden_color)
                line_alpha = min(alpha, 0.20)
            elif line_filter is not None and strength < 1:
                line_col = traj_col
                line_alpha = max(0.15, alpha * (0.15 + 0.85 * strength))
            else:
                line_col = _normalize_color_name(hidden_color) if (line_filter is None and len(traj_items) > 12) else traj_col
                line_alpha = alpha
            # Safety fallback if user passed an unsupported color string.
            if isinstance(line_col, str) and not mcolors.is_color_like(line_col):
                line_col = "0.8"
            ax.plot(
                xj,
                yj,
                color=line_col,
                linewidth=max(1.0, linewidth * rel),
                alpha=line_alpha,
                zorder=1,
                solid_capstyle="round",
            )

            # Colored square points at event nodes (state-semantic colors)
            for k in range(len(xj)):
                ev_name = _event_code_to_name(int(y_base[k]), dictionary)
                ev_col = event_colors.get(ev_name, traj_col)
                s = side * rel * 0.92
                node_alpha = min(0.95, 0.35 + 0.55 * rel)
                if strength <= 0:
                    ev_col = _normalize_color_name(hidden_color)
                    node_alpha = min(node_alpha, 0.25)
                elif line_filter is not None and strength < 1:
                    node_alpha = max(0.2, node_alpha * (0.25 + 0.75 * strength))
                if isinstance(ev_col, str) and not mcolors.is_color_like(ev_col):
                    ev_col = "0.8"
                ax.add_patch(
                    plt.Rectangle(
                        (xj[k] - s / 2, yj[k] - s / 2),
                        s,
                        s,
                        facecolor=ev_col,
                        edgecolor="white",
                        linewidth=0.2,
                        alpha=node_alpha,
                        zorder=2,
                    )
                )

        ax.set_xlim(0.5, max_x + 0.5)
        ax.set_ylim(0.5, len(dictionary) + 0.5)
        ax.set_yticks(np.arange(1, len(dictionary) + 1))
        ax.set_yticklabels(dictionary, fontsize=fontsize - 2, color="black")
        if order_align == "time":
            xlabel = "time" if x_label is None else x_label
        elif order_align == "last":
            xlabel = "position (from end)" if x_label is None else x_label
        else:
            xlabel = "position" if x_label is None else x_label
        ylabel = "event" if y_label is None else y_label
        _apply_axis_style(ax, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize)
        _add_seqpcplot_y_axis_swatches(
            ax, dictionary, event_colors, fontsize, max_label_chars=_max_label_chars
        )

        if title == "auto":
            panel_title = f"group: {g}" if g is not None else "event sequences"
        else:
            panel_title = title
        if panel_title is not None:
            ax.set_title(panel_title, fontsize=fontsize + 1, color="black")

    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Event colors are y-axis swatches; widen left margin when labels are long.
    _left_m = _seqpcplot_y_margin_frac(len(dictionary), fontsize, _max_label_chars)
    fig.tight_layout(rect=(_left_m, 0.03, 0.98, 0.97))
    return _finalize_figure(fig, save_as=save_as, dpi=dpi, show=show)


def plot_event_sequences(
    event_sequences: EventSequenceList,
    type: str = "index",
    top_n: Optional[int] = None,
    group=None,
    color_palette=None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
    fontsize: int = 11,
    save_as: Optional[str] = None,
    dpi: int = 200,
    show: bool = False,
    **kwargs,
):
    """
    Compatibility wrapper for event-sequence index and parallel plots.
    """
    if not isinstance(event_sequences, EventSequenceList):
        raise TypeError("event_sequences must be an EventSequenceList.")

    plot_type = str(type).lower()
    if plot_type == "parallel":
        return plot_event_parallel_coordinates(
            event_sequences,
            group_labels=group,
            color_palette=color_palette,
            title=title if title is not None else "auto",
            figsize=figsize,
            fontsize=fontsize,
            save_as=save_as,
            dpi=dpi,
            show=show,
            **kwargs,
        )
    if plot_type != "index":
        raise ValueError("type must be either 'index' or 'parallel'.")

    n_sequences = len(event_sequences)
    n_plot = n_sequences if top_n is None else min(max(int(top_n), 0), n_sequences)
    if n_plot == 0:
        raise ValueError("No event sequences to plot.")

    dictionary = list(event_sequences.dictionary)
    event_colors = _get_event_colors(dictionary, cpal=color_palette, seqelist=event_sequences)

    fig, ax = plt.subplots(figsize=figsize)
    for row, seq_idx in enumerate(range(n_plot)):
        seq = event_sequences[seq_idx]
        events = np.asarray(seq.events, dtype=int)
        if len(events) == 0:
            continue
        timestamps = np.asarray(seq.timestamps, dtype=float)
        if len(timestamps) != len(events):
            timestamps = np.arange(1, len(events) + 1, dtype=float)
        y = np.full(len(events), row, dtype=float)
        colors = [
            event_colors.get(_event_code_to_name(int(code), dictionary), "#888888")
            for code in events
        ]
        if len(timestamps) > 1:
            ax.plot(timestamps, y, color="0.82", linewidth=0.8, zorder=0)
        ax.scatter(timestamps, y, c=colors, s=22, edgecolors="white", linewidths=0.25, zorder=2)

    ax.set_ylim(-0.5, n_plot - 0.5)
    ax.invert_yaxis()
    ax.set_yticks(np.arange(n_plot))
    ax.set_yticklabels([str(event_sequences[i].id) for i in range(n_plot)], fontsize=fontsize - 2)
    _apply_axis_style(ax, xlabel="time", ylabel="sequence", fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize + 1, color="black")
    fig.tight_layout()
    return _finalize_figure(fig, save_as=save_as, dpi=dpi, show=show)


def plot_subsequence_frequencies(
    subsequence_results: SubsequenceList,
    frequency_values: Optional[Sequence[float]] = None,
    top_n: Optional[int] = None,
    text_scale: float = 1.0,
    color: str = "steelblue",
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    fontsize: int = 11,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    save_as: Optional[str] = None,
    dpi: int = 200,
    show: bool = False,
):
    """
    Plot frequent subsequence frequencies.

    TraMineR parameter mapping: ``subsequence_results`` -> ``x``,
    ``frequency_values`` -> ``freq``, ``text_scale`` -> ``cex``,
    ``title`` -> ``main``, ``x_label`` -> ``xlab``, ``y_label`` -> ``ylab``.
    """
    if not isinstance(subsequence_results, SubsequenceList):
        raise TypeError("subsequence_results must be a SubsequenceList.")
    if len(subsequence_results) == 0:
        raise ValueError("subsequence_results is empty.")

    if frequency_values is None:
        if "Support" in subsequence_results.data.columns:
            values = np.asarray(subsequence_results.data["Support"], dtype=float)
            default_xlab = "support"
        else:
            raise ValueError("No 'Support' column found and no frequency_values provided.")
    else:
        values = np.asarray(frequency_values, dtype=float)
        default_xlab = "frequency"

    labels = [sub.to_string() for sub in subsequence_results.subsequences]
    if top_n is not None:
        n_keep = min(max(int(top_n), 0), len(labels))
        if n_keep == 0:
            raise ValueError("top_n leaves no subsequences to plot.")
        labels = labels[:n_keep]
        values = values[:n_keep]
    order = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(order, values, color=color, alpha=0.9)
    ax.set_yticks(order)
    ax.set_yticklabels(labels, fontsize=(fontsize - 2) * text_scale, color="black")
    ax.invert_yaxis()
    final_xlab = default_xlab if x_label is None else x_label
    _apply_axis_style(ax, xlabel=final_xlab, ylabel=y_label, fontsize=int(fontsize * text_scale))
    ax.grid(axis="x", linestyle="--", alpha=0.12)
    if title is not None:
        ax.set_title(title, fontsize=(fontsize + 1) * text_scale, color="black")
    return _finalize_figure(fig, save_as=save_as, dpi=dpi, show=show)


def plot_subsequence_group_contrasts(
    group_contrast_results: SubsequenceList,
    y_limit_mode: str = "uniform",
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    resid_levels: Sequence[float] = (0.05, 0.01),
    color_palette=None,
    plot_type: str = "freq",
    legend_title: Optional[str] = None,
    show_legend: bool = True,
    legend_text_scale: float = 1.0,
    figsize: Tuple[float, float] = (13, 7),
    fontsize: int = 11,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    save_as: Optional[str] = None,
    dpi: int = 200,
    show: bool = False,
):
    """
    Plot group contrasts for discriminating subsequences.

    TraMineR parameter mapping: ``group_contrast_results`` -> ``x``,
    ``y_limit_mode`` -> ``ylim``, ``plot_type`` -> ``ptype``,
    ``show_legend`` -> ``with.legend``, ``x_label`` -> ``xlab``,
    ``y_label`` -> ``ylab``.
    """
    if not isinstance(group_contrast_results, SubsequenceList):
        raise TypeError("group_contrast_results must be a SubsequenceList.")
    if len(group_contrast_results) == 0:
        raise ValueError("group_contrast_results is empty.")

    group_labels = getattr(group_contrast_results, "labels", None)
    if group_labels is None:
        # Infer from result columns.
        group_labels = [
            col[len("Freq."):] for col in group_contrast_results.data.columns if col.startswith("Freq.")
        ]
    if not group_labels:
        raise ValueError("No group frequency columns found (Freq.<group>).")

    n_groups = len(group_labels)
    if cols is None:
        cols = min(3, n_groups)
    if rows is None:
        rows = int(np.ceil(n_groups / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    # Palette for residual bins (blue-neutral-red style).
    if color_palette is None:
        color_palette = plt.get_cmap("RdBu")(np.linspace(0.08, 0.92, 1 + 2 * len(resid_levels)))
    cpal = np.asarray(color_palette)

    # z critical values for two-sided p-levels.
    z_cut = []
    for p in resid_levels:
        if np.isclose(p, 0.05):
            z_cut.append(1.96)
        elif np.isclose(p, 0.01):
            z_cut.append(2.5758)
        else:
            z_cut.append(1.96)
    z_cut = sorted(z_cut)
    bounds = [-np.inf, -z_cut[-1], -z_cut[0], z_cut[0], z_cut[-1], np.inf]
    if len(cpal) != len(bounds) - 1:
        # Resample if user provided a different-length palette.
        cpal = plt.get_cmap("RdBu")(np.linspace(0.08, 0.92, len(bounds) - 1))
    norm = BoundaryNorm(bounds, ncolors=len(cpal), clip=False)

    # Determine y-limits for uniform mode.
    ylims = []
    for g in group_labels:
        col = f"Freq.{g}" if plot_type != "resid" else f"Resid.{g}"
        ylims.append(np.nanmax(np.abs(group_contrast_results.data[col].values)) if plot_type == "resid" else np.nanmax(group_contrast_results.data[col].values))
    uniform_lim = max(ylims) if ylims else 1.0

    labels = [s.to_string() for s in group_contrast_results.subsequences]
    y_idx = np.arange(len(labels))

    for i, g in enumerate(group_labels):
        ax = axes_flat[i]
        if plot_type == "resid":
            vals = np.asarray(group_contrast_results.data[f"Resid.{g}"], dtype=float)
            colors = cpal[norm(vals)]
            ax.barh(y_idx, vals, color=colors)
            ax.axvline(0.0, color="black", linewidth=1.0)
            lim = uniform_lim if y_limit_mode == "uniform" else np.nanmax(np.abs(vals))
            ax.set_xlim(-lim * 1.05, lim * 1.05)
            panel_xlab = "Pearson residual"
        else:
            vals = np.asarray(group_contrast_results.data[f"Freq.{g}"], dtype=float)
            # Color by residual even in frequency mode, as in TraMineR.
            resid = np.asarray(group_contrast_results.data[f"Resid.{g}"], dtype=float)
            colors = cpal[norm(resid)]
            ax.barh(y_idx, vals, color=colors)
            lim = uniform_lim if y_limit_mode == "uniform" else np.nanmax(vals)
            ax.set_xlim(0.0, lim * 1.05 if lim > 0 else 1.0)
            panel_xlab = "frequency"

        ax.set_yticks(y_idx)
        ax.set_yticklabels(labels, fontsize=fontsize - 2, color="black")
        ax.invert_yaxis()
        final_xlab = panel_xlab if x_label is None else x_label
        _apply_axis_style(ax, xlabel=final_xlab, ylabel=y_label, fontsize=fontsize)
        ax.grid(axis="x", alpha=0.12, linestyle="--")
        ax.set_title(str(g), fontsize=fontsize + 1, color="black")

    for j in range(n_groups, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if show_legend:
        if legend_title is None:
            legend_title = "Residual bins"
        # Build discrete legend labels.
        legend_labels = [
            f"< -{z_cut[-1]:.2f}",
            f"[-{z_cut[-1]:.2f}, -{z_cut[0]:.2f})",
            f"[-{z_cut[0]:.2f}, {z_cut[0]:.2f}]",
            f"({z_cut[0]:.2f}, {z_cut[-1]:.2f}]",
            f"> {z_cut[-1]:.2f}",
        ]
        handles = [Patch(facecolor=cpal[k], edgecolor="none", label=legend_labels[k]) for k in range(len(legend_labels))]
        fig.legend(handles=handles, loc="right", frameon=False, title=legend_title, fontsize=9 * legend_text_scale)
        fig.tight_layout(rect=(0, 0, 0.86, 1))
    else:
        fig.tight_layout()

    return _finalize_figure(fig, save_as=save_as, dpi=dpi, show=show)


def plot_event_dynamics(
    event_sequences: EventSequenceList,
    group_labels: Optional[Union[Sequence, np.ndarray]] = None,
    num_bins: int = 20,
    time_range: Optional[Tuple[float, float]] = None,
    title: Union[str, None] = "auto",
    curve_type: str = "survival",
    excluded_events: Optional[Iterable[str]] = None,
    show_legend: Union[str, bool] = "auto",
    color_palette=None,
    figsize: Tuple[float, float] = (12, 8),
    linewidth: float = 2.0,
    fontsize: int = 11,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    save_as: Optional[str] = None,
    dpi: int = 200,
    show: bool = False,
):
    """
    Plot event dynamics as survival or hazard-style curves.

    TraMineRextras equivalent: ``seqedplot()``.
    TraMineR parameter mapping: ``event_sequences`` -> ``seqe``,
    ``group_labels`` -> ``group``, ``num_bins`` -> ``breaks``,
    ``time_range`` -> ``ages``, ``curve_type`` -> ``type``,
    ``excluded_events`` -> ``ignore``, ``show_legend`` -> ``with.legend``.

    The hazard-style view summarizes event occurrence by time bins and is
    descriptive rather than a full event-history hazard model.
    """
    if not isinstance(event_sequences, EventSequenceList):
        raise TypeError("event_sequences must be an EventSequenceList.")
    if curve_type not in {"survival", "hazard"}:
        raise ValueError("curve_type must be 'survival' or 'hazard'.")

    group_arr = _as_array_or_none(group_labels)
    if group_arr is not None and len(group_arr) != len(event_sequences):
        raise ValueError("group_labels length must match number of sequences.")

    excluded_events = set(excluded_events or [])
    events = [e for e in event_sequences.dictionary if e not in excluded_events]
    if not events:
        raise ValueError("No events left to plot after filtering ignore.")

    if time_range is None:
        all_ts = np.concatenate([s.timestamps for s in event_sequences.sequences if len(s.timestamps) > 0]) if len(event_sequences) else np.array([0.0])
        age_min = float(np.nanmin(all_ts)) if len(all_ts) else 0.0
        age_max = float(np.nanmax(all_ts)) if len(all_ts) else 1.0
    else:
        age_min, age_max = float(time_range[0]), float(time_range[1])

    if group_arr is None:
        group_levels = [None]
    else:
        group_levels = list(np.unique(group_arr))

    n_panels = len(group_levels)
    cols = min(3, n_panels)
    rows = int(np.ceil(n_panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    ev_color = _get_event_colors(events, cpal=color_palette, seqelist=event_sequences)

    for i, g in enumerate(group_levels):
        ax = axes_flat[i]
        indices = np.arange(len(event_sequences)) if g is None else np.where(group_arr == g)[0]

        if curve_type == "survival":
            x_grid = np.linspace(age_min, age_max, 250)
            for ev in events:
                y = _compute_survival_curve(event_sequences, indices, ev, x_grid, age_max)
                ax.plot(x_grid, y, linewidth=linewidth, color=ev_color[ev], label=ev)
            ylabel = "survival probability"
        else:
            # Hazard representation: mean event count per time bin.
            bin_edges = np.linspace(age_min, age_max, int(num_bins) + 1)
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            for ev in events:
                y = _compute_hazard_curve(event_sequences, indices, ev, bin_edges)
                ax.plot(centers, y, linewidth=linewidth, color=ev_color[ev], label=ev)
            ylabel = "mean number of events"

        final_xlab = "time" if x_label is None else x_label
        final_ylab = ylabel if y_label is None else y_label
        _apply_axis_style(ax, xlabel=final_xlab, ylabel=final_ylab, fontsize=fontsize)
        ax.grid(axis="both", alpha=0.1, linestyle="--")

        if title == "auto":
            panel_title = f"group: {g}" if g is not None else curve_type
        else:
            panel_title = title
        if panel_title is not None:
            ax.set_title(panel_title, fontsize=fontsize + 1, color="black")

    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    should_show_legend = (show_legend is True) or (show_legend == "auto")
    if should_show_legend:
        handles = [Line2D([0], [0], color=ev_color[e], lw=2, label=e) for e in events]
        fig.legend(handles=handles, loc="right", frameon=False, title="Events")
        fig.tight_layout(rect=(0, 0, 0.88, 1))
    else:
        fig.tight_layout()

    return _finalize_figure(fig, save_as=save_as, dpi=dpi, show=show)


def _compute_survival_curve(
    seqe: EventSequenceList,
    indices: np.ndarray,
    event_name: str,
    x_grid: np.ndarray,
    censor_time: float,
) -> np.ndarray:
    """Weighted first-occurrence survival estimate: S(t) = P(T > t)."""
    code = seqe.dictionary.index(event_name) + 1
    first_times = np.full(len(indices), np.inf, dtype=float)
    weights = seqe.weights[indices]

    for k, idx in enumerate(indices):
        seq = seqe[idx]
        hits = seq.timestamps[seq.events == code]
        if len(hits) > 0:
            first_times[k] = np.min(hits)

    # Censoring: no occurrence by censor_time.
    first_times[np.isinf(first_times)] = censor_time + 1e-9
    denom = np.sum(weights)
    if denom <= 0:
        return np.ones_like(x_grid)
    out = np.empty_like(x_grid, dtype=float)
    for i, t in enumerate(x_grid):
        out[i] = np.sum(weights[first_times > t]) / denom
    return out


def _compute_hazard_curve(
    seqe: EventSequenceList,
    indices: np.ndarray,
    event_name: str,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """Weighted mean count per bin across sequences."""
    code = seqe.dictionary.index(event_name) + 1
    weights = seqe.weights[indices]
    denom = np.sum(weights)
    if denom <= 0:
        return np.zeros(len(bin_edges) - 1, dtype=float)

    total = np.zeros(len(bin_edges) - 1, dtype=float)
    for w, idx in zip(weights, indices):
        seq = seqe[idx]
        ts = seq.timestamps[seq.events == code]
        if len(ts) == 0:
            continue
        cnt, _ = np.histogram(ts, bins=bin_edges)
        total += w * cnt
    return total / denom


__all__ = [
    "plot_event_sequences",
    "plot_event_parallel_coordinates",
    "plot_subsequence_frequencies",
    "plot_subsequence_group_contrasts",
    "plot_event_dynamics",
]
