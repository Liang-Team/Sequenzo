"""
@Author  : 梁彧祺 Yuqi Liang, Jan Meyerhoff-Liang
@File    : visualization.py
@Time    : 23/04/2026 16:33
@Desc    :
    Visualization helpers for hierarchical sequence analysis.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns

from .data import RelationalSequenceData, RelationalSequenceRecord
from .decomposition.marginal import HierarchicalDecompositionResult
from .distances import RelationalDistanceMatrix
from .representation import encode_states


def plot_marginal_pseudo_r2(
    decomposition: HierarchicalDecompositionResult,
    *,
    ax: Optional[plt.Axes] = None,
    title: str = "Marginal pseudo-R² (non-exclusive)",
    level_1_label: str = "Level 1",
    level_2_label: str = "Level 2",
    colors: Optional[List[str]] = None,
) -> plt.Axes:
    """
    Bar chart of marginal level-1 and level-2 pseudo-R².

    These are not additive components and should not be read as shares summing to 1.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    labels = [level_1_label, level_2_label]
    values = [decomposition.level_1.pseudo_r2, decomposition.level_2.pseudo_r2]
    if decomposition.additive is not None:
        labels.append("Joint (additive)")
        values.append(decomposition.additive.joint_share)

    bar_colors = colors or ["#4C72B0", "#55A868", "#8172B2"][: len(labels)]
    ax.bar(labels, values, color=bar_colors)
    ax.set_ylabel("Pseudo-R²")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    plt.tight_layout()
    return ax


def plot_additive_marginal_shares(
    decomposition: HierarchicalDecompositionResult,
    *,
    ax: Optional[plt.Axes] = None,
    title: str = "Additive Type-III partial shares (non-exclusive)",
    level_1_label: str = "Level 1",
    level_2_label: str = "Level 2",
    colors: Optional[List[str]] = None,
) -> plt.Axes:
    """
    Type-III partial contributions from the additive model.

    ``level_1_share`` and ``level_2_share`` are marginal partial effects; they
    need **not** sum to 1 with the residual. For shares that sum to 1, use
    :func:`plot_joint_residual_shares`.
    """
    if decomposition.additive is None:
        raise ValueError(
            "Additive decomposition not available. Run hierarchical_sequence_discrepancy "
            "with include_additive=True."
        )
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    a = decomposition.additive
    labels = [
        f"{level_1_label}\n(partial)",
        f"{level_2_label}\n(partial)",
        "Joint explained",
    ]
    values = [a.level_1_share, a.level_2_share, a.joint_share]
    bar_colors = colors or ["#4C72B0", "#55A868", "#8172B2"]
    ax.bar(labels, values, color=bar_colors)
    ax.set_ylabel("Pseudo-R² share")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    plt.tight_layout()
    return ax


def plot_joint_residual_shares(
    decomposition: HierarchicalDecompositionResult,
    *,
    ax: Optional[plt.Axes] = None,
    title: str = "Joint explained vs pair-specific residual",
    colors: Optional[List[str]] = None,
) -> plt.Axes:
    """
    Additive decomposition that sums to 1: joint explained + residual.

    This is the recommended “component” view for the additive model.
    """
    if decomposition.additive is None:
        raise ValueError(
            "Additive decomposition not available. Run hierarchical_sequence_discrepancy "
            "with include_additive=True."
        )
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    a = decomposition.additive
    labels = ["Joint explained\n(level_1 + level_2)", "Pair-specific\nresidual"]
    values = [a.joint_share, a.residual_share]
    bar_colors = colors or ["#4C72B0", "#C44E52"]
    ax.bar(labels, values, color=bar_colors)
    ax.set_ylabel("Share of total discrepancy")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    plt.tight_layout()
    return ax


def plot_additive_component_shares(
    decomposition: HierarchicalDecompositionResult,
    **kwargs: Any,
) -> plt.Axes:
    """
    Deprecated: use :func:`plot_joint_residual_shares` instead.

    Previously plotted partial level shares together with residual as if they
    summed to 1, which is misleading under Type-III SS.
    """
    warnings.warn(
        "plot_additive_component_shares is deprecated; use plot_joint_residual_shares() "
        "for shares that sum to 1, or plot_additive_marginal_shares() for partial "
        "level-1 / level-2 contributions.",
        DeprecationWarning,
        stacklevel=2,
    )
    return plot_joint_residual_shares(decomposition, **kwargs)


def plot_crossed_component_shares(
    decomposition: HierarchicalDecompositionResult,
    *,
    ax: Optional[plt.Axes] = None,
    title: str = "Crossed interaction model (experimental)",
    level_1_label: str = "Level 1",
    level_2_label: str = "Level 2",
    colors: Optional[List[str]] = None,
) -> plt.Axes:
    """Bar chart for experimental crossed decomposition including interaction."""
    if decomposition.crossed is None:
        raise ValueError("Crossed decomposition not computed (include_crossed=True).")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    c = decomposition.crossed
    labels = [level_1_label, level_2_label, "Interaction", "Residual"]
    values = [c.level_1_share, c.level_2_share, c.interaction_share, c.residual_share]
    bar_colors = colors or ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]
    ax.bar(labels, values, color=bar_colors)
    ax.set_ylabel("Share of discrepancy")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    plt.tight_layout()
    return ax


def plot_decomposition_bar(
    decomposition: Union[HierarchicalDecompositionResult, dict],
    *,
    ax: Optional[plt.Axes] = None,
    chart: str = "auto",
    title: Optional[str] = None,
    level_1_label: str = "Level 1",
    level_2_label: str = "Level 2",
    colors: Optional[List[str]] = None,
) -> plt.Axes:
    """
    Plot decomposition results.

    Parameters
    ----------
    chart : str
        - ``"auto"``: joint + residual if additive model available, else marginal
        - ``"marginal"``: single-factor marginal pseudo-R² (non-exclusive)
        - ``"additive_marginal"``: Type-III partial level-1 / level-2 + joint
        - ``"joint_residual"`` or ``"additive"``: joint explained + residual (sums to 1)
        - ``"crossed"``: experimental interaction model
    """
    if isinstance(decomposition, HierarchicalDecompositionResult):
        if chart == "auto":
            chart = "joint_residual" if decomposition.additive is not None else "marginal"
        if chart in ("additive", "joint_residual"):
            return plot_joint_residual_shares(
                decomposition,
                ax=ax,
                title=title or "Joint explained vs pair-specific residual",
                colors=colors,
            )
        if chart == "additive_marginal":
            return plot_additive_marginal_shares(
                decomposition,
                ax=ax,
                title=title or "Additive Type-III partial shares",
                level_1_label=level_1_label,
                level_2_label=level_2_label,
                colors=colors,
            )
        if chart == "crossed":
            return plot_crossed_component_shares(
                decomposition,
                ax=ax,
                title=title or "Crossed interaction model (experimental)",
                level_1_label=level_1_label,
                level_2_label=level_2_label,
                colors=colors,
            )
        return plot_marginal_pseudo_r2(
            decomposition,
            ax=ax,
            title=title or "Marginal pseudo-R²",
            level_1_label=level_1_label,
            level_2_label=level_2_label,
            colors=colors,
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    labels = [level_1_label, level_2_label, "Residual"]
    values = [
        decomposition.get("level_1_share", 0),
        decomposition.get("level_2_share", 0),
        decomposition.get("residual_share", 0),
    ]
    ax.bar(labels, values, color=colors or ["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylabel("Share")
    ax.set_ylim(0, 1)
    ax.set_title(title or "Decomposition")
    plt.tight_layout()
    return ax


def plot_distance_heatmap(
    distance_matrix: RelationalDistanceMatrix,
    *,
    ax: Optional[plt.Axes] = None,
    max_labels: int = 40,
    title: str = "Pairwise sequence distances",
    cmap: str = "viridis",
) -> plt.Axes:
    """Heatmap of the pair-level distance matrix (subsampled labels if large)."""
    if ax is None:
        size = min(10, max(4, distance_matrix.n_pairs * 0.15))
        _, ax = plt.subplots(figsize=(size, size))

    df = distance_matrix.to_dataframe()
    if distance_matrix.n_pairs > max_labels:
        df = df.iloc[:max_labels, :max_labels]

    sns.heatmap(df, ax=ax, cmap=cmap, square=True, cbar_kws={"label": "Distance"})
    ax.set_title(title)
    plt.tight_layout()
    return ax


def plot_level_similarity_matrix(
    distance_matrix: RelationalDistanceMatrix,
    level: int = 1,
    *,
    ax: Optional[plt.Axes] = None,
    aggregation: str = "mean",
) -> plt.Axes:
    """Aggregate pair distances to a level-1 or level-2 mean distance matrix."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ids = distance_matrix.level_1_ids if level == 1 else distance_matrix.level_2_ids
    unique = pd.unique(ids)
    n = len(unique)
    agg = np.zeros((n, n))

    for i, u1 in enumerate(unique):
        idx1 = np.where(ids == u1)[0]
        for j, u2 in enumerate(unique):
            idx2 = np.where(ids == u2)[0]
            block = distance_matrix.matrix[np.ix_(idx1, idx2)]
            if i == j:
                iu = np.triu_indices(block.shape[0], k=1)
                vals = block[iu] if len(iu[0]) else np.array([0.0])
            else:
                vals = block.ravel()
            agg[i, j] = float(np.mean(vals)) if aggregation == "mean" else float(np.median(vals))

    sns.heatmap(
        pd.DataFrame(agg, index=unique, columns=unique),
        ax=ax,
        cmap="mako_r",
        square=True,
    )
    level_name = "level-1" if level == 1 else "level-2"
    ax.set_title(f"{level_name} mean distance matrix")
    plt.tight_layout()
    return ax


def plot_pair_outliers(
    outliers: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    title: str = "Pair-specific residual outliers",
) -> plt.Axes:
    """Bar chart of top pair outlier scores (standardized residual)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    score_col = "outlier_score" if "outlier_score" in outliers.columns else "standardized_residual"
    plot_df = outliers.sort_values(score_col, ascending=True)
    ax.barh(plot_df["pair_id"].astype(str), plot_df[score_col])
    ax.set_xlabel("Standardized residual (distance)")
    ax.set_title(title)
    plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Relational sequence visualization (level-1 × level-2 × time)
# ---------------------------------------------------------------------------

_DEFAULT_STATE_COLORS = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B2",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
    "#CCB974",
    "#64B5CD",
]


def _is_missing_state(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (ValueError, TypeError):
        return False


def _pair_lookup(
    sequence_data: RelationalSequenceData,
) -> Dict[Tuple[Any, Any], RelationalSequenceRecord]:
    return {(rec.level_1_id, rec.level_2_id): rec for rec in sequence_data.records}


def _build_state_cmap(
    alphabet: List[Any],
    state_palette: Optional[Union[Mapping[Any, str], Sequence[str]]] = None,
) -> Tuple[ListedColormap, Dict[Any, int]]:
    """Map categorical states to a ListedColormap and index dictionary."""
    if not alphabet:
        cmap = ListedColormap(["#DDDDDD"])
        return cmap, {}

    if isinstance(state_palette, Mapping):
        colors = [state_palette.get(s, "#CCCCCC") for s in alphabet]
    elif isinstance(state_palette, Sequence) and len(state_palette) >= len(alphabet):
        colors = list(state_palette[: len(alphabet)])
    else:
        colors = [
            _DEFAULT_STATE_COLORS[i % len(_DEFAULT_STATE_COLORS)]
            for i in range(len(alphabet))
        ]

    state_to_idx = {s: i for i, s in enumerate(alphabet)}
    return ListedColormap(colors), state_to_idx


def _sequences_to_code_matrix(
    sequences: Sequence[Sequence[Any]],
    state_to_idx: Dict[Any, int],
) -> np.ndarray:
    """Encode sequences with a global state index map (consistent colors across panels)."""
    rows = []
    for seq in sequences:
        rows.append([state_to_idx.get(s, 0) for s in seq])
    return np.asarray(rows, dtype=float)


def _draw_sequence_strip(
    ax: plt.Axes,
    sequence: Sequence[Any],
    *,
    cmap: ListedColormap,
    state_to_idx: Dict[Any, int],
    time_points: Optional[Sequence[Any]] = None,
    missing_color: str = "#EEEEEE",
) -> None:
    """Draw one horizontal state-sequence strip (time left-to-right)."""
    ax.set_xlim(0, max(len(sequence), 1))
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if len(sequence) == 0:
        ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, facecolor=missing_color, linewidth=0))
        return

    for t, state in enumerate(sequence):
        if _is_missing_state(state):
            color = missing_color
        else:
            color = cmap(state_to_idx[state])
        ax.add_patch(
            mpatches.Rectangle(
                (t, 0),
                1,
                1,
                facecolor=color,
                edgecolor="white",
                linewidth=0.3,
            )
        )

    if time_points is not None and len(time_points) == len(sequence):
        ax.set_xticks([i + 0.5 for i in range(len(sequence))])
        ax.set_xticklabels([str(tp) for tp in time_points], fontsize=5, rotation=45, ha="right")


def _mean_profile_code(sequences: Sequence[Sequence[Any]]) -> np.ndarray:
    """Lexicographic mean profile: per position, mean code over sequences."""
    if not sequences:
        return np.array([])
    codes, _ = encode_states(sequences)
    if codes.size == 0:
        return np.array([])
    profile = np.zeros(codes.shape[1], dtype=float)
    for j in range(codes.shape[1]):
        col = codes[:, j]
        valid = col[col >= 0]
        profile[j] = float(np.mean(valid)) if len(valid) else -1.0
    return profile


def _order_level_units(
    units: Sequence[Any],
    df: pd.DataFrame,
    level_col: str,
    *,
    sort_by: str,
    pair_residuals: Optional[pd.DataFrame] = None,
) -> List[Any]:
    """Order level-1 or level-2 units for grids and portfolio panels."""
    units = list(pd.unique(units))
    if sort_by in ("level_id", "id", "alphabetical"):
        return sorted(units, key=lambda x: str(x))

    if sort_by == "frequency":
        counts = df.groupby(level_col, observed=True).size()
        return sorted(units, key=lambda u: (-counts.get(u, 0), str(u)))

    if sort_by == "residual" and pair_residuals is not None:
        abs_col = (
            "abs_standardized_residual"
            if "abs_standardized_residual" in pair_residuals.columns
            else None
        )
        if abs_col is None and "standardized_residual" in pair_residuals.columns:
            tmp = pair_residuals.copy()
            tmp["_abs"] = tmp["standardized_residual"].abs()
            abs_col = "_abs"
            pair_residuals = tmp
        if abs_col:
            other_col = "level_2_id" if level_col == "level_1_id" else "level_1_id"
            merged = pair_residuals.groupby(level_col, observed=True)[abs_col].mean()
            return sorted(units, key=lambda u: (-merged.get(u, 0.0), str(u)))

    if sort_by == "profile":
        profiles: Dict[Any, np.ndarray] = {}
        for unit in units:
            seqs = df.loc[df[level_col] == unit, "sequence"].tolist()
            _, alphabet = encode_states(seqs)
            profiles[unit] = _mean_profile_code(seqs)
        return sorted(units, key=lambda u: profiles[u].tolist())

    return sorted(units, key=lambda x: str(x))


def _hierarchical_pair_order(
    level_1_ids: np.ndarray,
    level_2_ids: np.ndarray,
    order_by: Tuple[str, ...] = ("level_1", "level_2"),
) -> np.ndarray:
    """Return row/column permutation sorting pairs by hierarchical identifiers."""
    col_map = {"level_1": "level_1_id", "level_2": "level_2_id", "l1": "level_1_id", "l2": "level_2_id"}
    sort_cols = []
    for key in order_by:
        name = col_map.get(key, key)
        if name == "level_1_id":
            sort_cols.append(level_1_ids)
        elif name == "level_2_id":
            sort_cols.append(level_2_ids)
        else:
            raise ValueError(f"Unknown order_by key {key!r}. Use 'level_1' and/or 'level_2'.")

    order = np.lexsort(sort_cols[::-1]) if sort_cols else np.arange(len(level_1_ids))
    return order


def _level_boundary_positions(values: np.ndarray) -> List[int]:
    """Indices after which a block boundary should be drawn (exclusive end of block)."""
    if len(values) <= 1:
        return []
    boundaries = []
    for i in range(len(values) - 1):
        if values[i] != values[i + 1]:
            boundaries.append(i + 1)
    return boundaries


def _medoid_record_index(
    candidate_indices: np.ndarray,
    matrix: np.ndarray,
) -> int:
    """Index of the medoid sequence among candidates (minimum mean distance to others)."""
    if len(candidate_indices) == 0:
        raise ValueError("No candidate indices for medoid.")
    if len(candidate_indices) == 1:
        return int(candidate_indices[0])

    sub = matrix[np.ix_(candidate_indices, candidate_indices)]
    mean_d = sub.sum(axis=1) / max(len(candidate_indices) - 1, 1)
    return int(candidate_indices[int(np.argmin(mean_d))])


def plot_relational_sequence_grid(
    sequence_data: RelationalSequenceData,
    *,
    level_1_order: Optional[Sequence[Any]] = None,
    level_2_order: Optional[Sequence[Any]] = None,
    max_level_1: int = 12,
    max_level_2: int = 12,
    sort_by: str = "profile",
    state_palette: Optional[Union[Mapping[Any, str], Sequence[str]]] = None,
    show_labels: bool = True,
    pair_residuals: Optional[pd.DataFrame] = None,
    axes: Optional[np.ndarray] = None,
    embedded: bool = False,
    title: str = "Relational sequence grid (level-1 × level-2)",
    missing_color: str = "#F0F0F0",
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Axes:
    """
    Core relational visualization: one mini sequence per (level-1, level-2) cell.

    Rows are level-1 units (e.g. regions); columns are level-2 units (e.g. technologies).
    Each cell shows the pair-level trajectory for that crossed relation — not a pooled
    index plot of all sequences.
    """
    df = sequence_data.to_dataframe()
    lookup = _pair_lookup(sequence_data)

    l1_units = _order_level_units(
        df["level_1_id"].unique(),
        df,
        "level_1_id",
        sort_by=sort_by,
        pair_residuals=pair_residuals,
    )
    l2_units = _order_level_units(
        df["level_2_id"].unique(),
        df,
        "level_2_id",
        sort_by=sort_by,
        pair_residuals=pair_residuals,
    )

    if level_1_order is not None:
        l1_units = [u for u in level_1_order if u in set(l1_units)]
    if level_2_order is not None:
        l2_units = [u for u in level_2_order if u in set(l2_units)]

    l1_units = l1_units[:max_level_1]
    l2_units = l2_units[:max_level_2]

    all_sequences = [rec.sequence for rec in sequence_data.records]
    _, alphabet = encode_states(all_sequences)
    cmap, state_to_idx = _build_state_cmap(alphabet, state_palette)

    n_rows, n_cols = len(l1_units), len(l2_units)
    if n_rows == 0 or n_cols == 0:
        raise ValueError("No level-1 or level-2 units to plot after filtering.")

    if axes is not None:
        embedded = True
        axes = np.asarray(axes)
        if axes.shape != (n_rows, n_cols):
            raise ValueError(
                f"axes must have shape ({n_rows}, {n_cols}); got {axes.shape}."
            )
        fig = axes.flat[0].figure
    else:
        cell_w, cell_h = 1.1, 0.55
        w = max(4.0, n_cols * cell_w + (1.5 if show_labels else 0.5))
        h = max(3.0, n_rows * cell_h + (1.0 if show_labels else 0.3))
        if figsize is not None:
            w, h = figsize
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(w, h),
            squeeze=False,
            gridspec_kw={"wspace": 0.15, "hspace": 0.25},
        )

    for i, l1 in enumerate(l1_units):
        for j, l2 in enumerate(l2_units):
            cell_ax = axes[i, j]
            rec = lookup.get((l1, l2))
            if rec is not None:
                _draw_sequence_strip(
                    cell_ax,
                    rec.sequence,
                    cmap=cmap,
                    state_to_idx=state_to_idx,
                    time_points=rec.time_points if i == n_rows - 1 else None,
                )
            else:
                cell_ax.set_facecolor(missing_color)
                cell_ax.set_xlim(0, 1)
                cell_ax.set_ylim(0, 1)
                cell_ax.set_xticks([])
                cell_ax.set_yticks([])
                for spine in cell_ax.spines.values():
                    spine.set_visible(False)

            if show_labels:
                if j == 0:
                    cell_ax.set_ylabel(str(l1), fontsize=8, rotation=0, ha="right", va="center")
                if i == 0:
                    cell_ax.set_title(str(l2), fontsize=8, pad=2)

    if not embedded:
        fig.suptitle(title, fontsize=11, y=1.02)
        if alphabet:
            handles = [
                mpatches.Patch(color=cmap(state_to_idx[s]), label=str(s)) for s in alphabet
            ]
            fig.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=min(6, len(handles)),
                fontsize=7,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.tight_layout()
    elif title:
        axes[0, 0].set_title(title, fontsize=9, loc="left")
    return axes[0, 0]


def plot_hierarchical_distance_heatmap(
    distance_matrix: RelationalDistanceMatrix,
    *,
    order_by: Tuple[str, ...] = ("level_1", "level_2"),
    show_level_boundaries: bool = True,
    max_pairs: int = 300,
    ax: Optional[plt.Axes] = None,
    title: str = "Hierarchical block distance heatmap",
    cmap: str = "viridis",
    level_1_label: str = "Level 1",
    level_2_label: str = "Level 2",
) -> plt.Axes:
    """
    Pairwise distance heatmap with rows/columns sorted by crossed identifiers.

    Block boundaries mark changes in level-1 and level-2 blocks so the matrix can be
    read as a structured relational object, not an anonymous pooled distance matrix.
    """
    n = distance_matrix.n_pairs
    if n == 0:
        raise ValueError("Distance matrix is empty.")

    order = _hierarchical_pair_order(
        distance_matrix.level_1_ids,
        distance_matrix.level_2_ids,
        order_by=order_by,
    )
    if n > max_pairs:
        order = order[:max_pairs]

    sub_matrix = distance_matrix.matrix[np.ix_(order, order)]
    l1_ord = distance_matrix.level_1_ids[order]
    l2_ord = distance_matrix.level_2_ids[order]
    pair_ord = distance_matrix.pair_ids[order]

    if ax is None:
        size = min(12, max(5, len(order) * 0.12))
        _, ax = plt.subplots(figsize=(size, size))

    im = ax.imshow(sub_matrix, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Distance")

    if show_level_boundaries:
        for b in _level_boundary_positions(l1_ord):
            ax.axhline(b - 0.5, color="white", linewidth=1.2, alpha=0.9)
            ax.axvline(b - 0.5, color="white", linewidth=1.2, alpha=0.9)
        for b in _level_boundary_positions(l2_ord):
            ax.axhline(b - 0.5, color="black", linewidth=0.6, alpha=0.35, linestyle="--")
            ax.axvline(b - 0.5, color="black", linewidth=0.6, alpha=0.35, linestyle="--")

    tick_step = max(1, len(order) // 25)
    tick_pos = list(range(0, len(order), tick_step))
    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    ax.set_xticklabels([str(pair_ord[i]) for i in tick_pos], rotation=90, fontsize=5)
    ax.set_yticklabels([str(pair_ord[i]) for i in tick_pos], fontsize=5)
    ax.set_title(title)
    ax.set_xlabel(f"Pairs (sorted by {level_1_label}, then {level_2_label})")
    ax.set_ylabel(f"Pairs (sorted by {level_1_label}, then {level_2_label})")
    plt.tight_layout()
    return ax


def plot_pair_outlier_sequences(
    sequence_data: RelationalSequenceData,
    pair_residuals: pd.DataFrame,
    *,
    distance_matrix: Optional[RelationalDistanceMatrix] = None,
    top_n: int = 12,
    order_by: str = "abs_standardized_residual",
    show_medoids: bool = True,
    state_palette: Optional[Union[Mapping[Any, str], Sequence[str]]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Pair-specific residual outliers (sequences)",
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Axes:
    """
    Sequence-oriented view of top pair residuals.

    Each row shows pair metadata and z-score, the observed pair sequence, and optionally
    the level-1 and level-2 medoid sequences for contextual comparison.
    """
    res = pair_residuals.copy()
    if order_by not in res.columns:
        if order_by == "abs_standardized_residual" and "standardized_residual" in res.columns:
            res["abs_standardized_residual"] = res["standardized_residual"].abs()
        else:
            raise ValueError(f"Column {order_by!r} not found in pair_residuals.")

    res = res.sort_values(order_by, ascending=False).head(top_n)
    if res.empty:
        raise ValueError("No pair residuals to plot.")

    lookup = {r.pair_id: r for r in sequence_data.records}
    all_sequences = [rec.sequence for rec in sequence_data.records]
    _, alphabet = encode_states(all_sequences)
    cmap, state_to_idx = _build_state_cmap(alphabet, state_palette)

    n_cols = 4 if show_medoids and distance_matrix is not None else 2
    col_titles = ["Pair / z", "Observed pair"]
    if n_cols == 4:
        col_titles.extend(["Level-1 medoid", "Level-2 medoid"])

    if ax is not None:
        raise ValueError("plot_pair_outlier_sequences requires its own figure; pass ax=None.")

    n_rows = len(res)
    w, h = figsize or (2.2 * n_cols + 1.5, max(3.0, 0.55 * n_rows + 1.0))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(w, h),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.2] + [2.0] * (n_cols - 1)},
    )

    matrix = distance_matrix.matrix if distance_matrix is not None else None
    l1_all = sequence_data.level_1_ids
    l2_all = sequence_data.level_2_ids
    id_to_idx = {pid: i for i, pid in enumerate(sequence_data.pair_ids)}

    for row_i, (_, prow) in enumerate(res.iterrows()):
        pid = prow["pair_id"]
        rec = lookup.get(pid)
        if rec is None:
            continue

        meta_ax = axes[row_i, 0]
        meta_ax.axis("off")
        z = prow.get("standardized_residual", np.nan)
        meta_lines = [
            str(pid),
            f"{prow.get('level_1_id', rec.level_1_id)} × {prow.get('level_2_id', rec.level_2_id)}",
            f"z = {z:+.2f}" if np.isfinite(z) else "z = —",
        ]
        meta_ax.text(0.0, 0.5, "\n".join(meta_lines), va="center", fontsize=8, family="monospace")

        _draw_sequence_strip(
            axes[row_i, 1],
            rec.sequence,
            cmap=cmap,
            state_to_idx=state_to_idx,
            time_points=rec.time_points,
        )
        if row_i == 0:
            axes[row_i, 1].set_title(col_titles[1], fontsize=9)

        if n_cols == 4 and matrix is not None:
            idx = id_to_idx[pid]
            l1_idx = np.where(l1_all == rec.level_1_id)[0]
            l2_idx = np.where(l2_all == rec.level_2_id)[0]
            med_l1 = _medoid_record_index(l1_idx, matrix)
            med_l2 = _medoid_record_index(l2_idx, matrix)
            med_rec_l1 = sequence_data.records[med_l1]
            med_rec_l2 = sequence_data.records[med_l2]
            _draw_sequence_strip(
                axes[row_i, 2],
                med_rec_l1.sequence,
                cmap=cmap,
                state_to_idx=state_to_idx,
            )
            _draw_sequence_strip(
                axes[row_i, 3],
                med_rec_l2.sequence,
                cmap=cmap,
                state_to_idx=state_to_idx,
            )
            if row_i == 0:
                axes[row_i, 2].set_title(col_titles[2], fontsize=9)
                axes[row_i, 3].set_title(col_titles[3], fontsize=9)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    return axes[0, 0]


def plot_level_portfolio_sequences(
    sequence_data: RelationalSequenceData,
    level: int = 1,
    *,
    max_units: int = 6,
    max_sequences_per_unit: int = 40,
    sort_by: str = "profile",
    sort_sequences_by: str = "profile",
    state_palette: Optional[Union[Mapping[Any, str], Sequence[str]]] = None,
    pair_residuals: Optional[pd.DataFrame] = None,
    title: Optional[str] = None,
    level_1_label: str = "Level 1",
    level_2_label: str = "Level 2",
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Axes:
    """
    Portfolio panels: all relational trajectories tied to one higher-level unit.

    ``level=1``: for each level-1 unit, show its level-2-specific trajectories.
    ``level=2``: for each level-2 unit, show its level-1-specific trajectories.

    This is the relational analogue of a multidomain portfolio — same higher-level
    unit, many crossed counterparts — not two domains of the same actor.
    """
    if level not in (1, 2):
        raise ValueError("level must be 1 or 2")

    df = sequence_data.to_dataframe()
    unit_col = "level_1_id" if level == 1 else "level_2_id"
    counterpart_col = "level_2_id" if level == 1 else "level_1_id"
    unit_label = level_1_label if level == 1 else level_2_label
    counterpart_label = level_2_label if level == 1 else level_1_label

    units = _order_level_units(
        df[unit_col].unique(),
        df,
        unit_col,
        sort_by=sort_by,
        pair_residuals=pair_residuals,
    )[:max_units]

    all_sequences = [rec.sequence for rec in sequence_data.records]
    _, alphabet = encode_states(all_sequences)
    cmap, state_to_idx = _build_state_cmap(alphabet, state_palette)

    n_panels = len(units)
    if n_panels == 0:
        raise ValueError("No units to plot.")

    w, h = figsize or (10, max(2.5, 2.2 * n_panels))
    fig, panel_axes = plt.subplots(n_panels, 1, figsize=(w, h), squeeze=False)

    for p, unit in enumerate(units):
        pax = panel_axes[p, 0]
        sub = df[df[unit_col] == unit].copy()

        if sort_sequences_by in ("counterpart", "level_id", "alphabetical"):
            sub = sub.sort_values(counterpart_col, key=lambda s: s.astype(str))
        elif sort_sequences_by == "residual" and pair_residuals is not None:
            abs_col = (
                "abs_standardized_residual"
                if "abs_standardized_residual" in pair_residuals.columns
                else "standardized_residual"
            )
            sub = sub.merge(
                pair_residuals[["pair_id", abs_col]],
                on="pair_id",
                how="left",
            )
            sub = sub.sort_values(abs_col, ascending=False, na_position="last")
        else:
            profiles = {
                row["pair_id"]: _mean_profile_code([row["sequence"]])
                for _, row in sub.iterrows()
            }
            sub["_sort"] = sub["pair_id"].map(lambda pid: profiles[pid].tolist())
            sub = sub.sort_values("_sort")

        sub = sub.head(max_sequences_per_unit)
        codes = _sequences_to_code_matrix(sub["sequence"].tolist(), state_to_idx)
        pax.imshow(
            codes,
            aspect="auto",
            cmap=cmap,
            vmin=-0.5,
            vmax=max(len(alphabet) - 0.5, 0.5),
            interpolation="nearest",
        )
        pax.set_yticks(range(len(sub)))
        pax.set_yticklabels(
            [f"{row[counterpart_col]}" for _, row in sub.iterrows()],
            fontsize=7,
        )
        if sub.iloc[0]["time_points"]:
            tp = sub.iloc[0]["time_points"]
            pax.set_xticks(range(len(tp)))
            pax.set_xticklabels([str(t) for t in tp], rotation=45, ha="right", fontsize=7)
        pax.set_ylabel(counterpart_label, fontsize=8)
        pax.set_title(f"{unit_label}: {unit}  (n={len(sub)})", fontsize=9, loc="left")
        pax.set_xlabel("Time")

    default_title = (
        f"Level-1 portfolios ({level_1_label})"
        if level == 1
        else f"Level-2 portfolios ({level_2_label})"
    )
    fig.suptitle(title or default_title, fontsize=11)
    if alphabet:
        handles = [mpatches.Patch(color=cmap(state_to_idx[s]), label=str(s)) for s in alphabet]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=min(6, len(handles)), fontsize=7)
    plt.tight_layout()
    return panel_axes[0, 0]


def plot_level_1_sequence_panels(
    sequence_data: RelationalSequenceData,
    **kwargs: Any,
) -> plt.Axes:
    """Same-region (level-1) relational portfolios across level-2 counterparts."""
    return plot_level_portfolio_sequences(sequence_data, level=1, **kwargs)


def plot_level_2_sequence_panels(
    sequence_data: RelationalSequenceData,
    **kwargs: Any,
) -> plt.Axes:
    """Same-technology (level-2) relational portfolios across level-1 counterparts."""
    return plot_level_portfolio_sequences(sequence_data, level=2, **kwargs)


def plot_sequence_index_by_level(
    sequence_data: RelationalSequenceData,
    level: int = 1,
    **kwargs: Any,
) -> plt.Axes:
    """
    Deprecated: use :func:`plot_level_portfolio_sequences` or level-specific panel helpers.

    Kept for backward compatibility; delegates to portfolio panels.
    """
    warnings.warn(
        "plot_sequence_index_by_level is deprecated; use plot_level_portfolio_sequences(), "
        "plot_level_1_sequence_panels(), or plot_level_2_sequence_panels().",
        DeprecationWarning,
        stacklevel=2,
    )
    kwargs.setdefault("max_units", kwargs.pop("max_units", 8))
    kwargs.setdefault("max_sequences_per_unit", kwargs.pop("max_sequences_per_unit", 40))
    kwargs.pop("cmap", None)
    return plot_level_portfolio_sequences(sequence_data, level=level, **kwargs)
