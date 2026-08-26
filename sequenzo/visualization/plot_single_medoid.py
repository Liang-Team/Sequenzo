"""
@Author  : Yuqi Liang 梁彧祺
@File    : plot_single_medoid.py
@Time    : 26/08/2026 20:10
@Desc    :
    Group / cluster medoid plots in the same visual language as ``plot_sequence_index``.

    * ``plot_medoids`` — one panel per group or cluster (column / grid), or all
      medoids stacked as rows. Grouping uses the same two APIs as the index plot:
      ``group_by_column`` (e.g. gender already in the data) or
      ``group_dataframe`` + ``group_column_name`` (e.g. a cluster membership table).
    * ``plot_single_medoid`` — backward-compatible helper that finds one global
      medoid from a distance matrix.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

from sequenzo.define_sequence_data import SequenceData
from sequenzo.visualization.plot_sequence_index import smart_sort_groups
from sequenzo.visualization.utils import (
    combine_plot_with_legend,
    create_standalone_legend,
    determine_layout,
    legend_ncol,
    save_and_show_results,
    save_figure_to_buffer,
    set_up_time_labels_for_x_axis,
    show_plot_title,
    _to_square_matrix,
)


def _as_domain_list(
    seqdata: Union[SequenceData, Sequence[SequenceData]],
) -> List[SequenceData]:
    if isinstance(seqdata, SequenceData):
        return [seqdata]
    domains = list(seqdata)
    if not domains:
        raise ValueError("seqdata must contain at least one SequenceData object.")
    if not all(isinstance(item, SequenceData) for item in domains):
        raise TypeError("seqdata must be a SequenceData or a sequence of SequenceData.")
    return domains


def _validate_aligned_domains(domains: List[SequenceData]) -> SequenceData:
    reference = domains[0]
    n_seq = len(reference.values)
    n_time = reference.values.shape[1]
    states = list(reference.states)
    for i, domain in enumerate(domains[1:], start=2):
        if len(domain.values) != n_seq:
            raise ValueError(
                f"Domain {i} has {len(domain.values)} sequences; expected {n_seq}."
            )
        if domain.values.shape[1] != n_time:
            raise ValueError(
                f"Domain {i} has {domain.values.shape[1]} time points; expected {n_time}."
            )
        if list(domain.ids) != list(reference.ids):
            raise ValueError("All domains must use the same entity IDs in the same order.")
        if list(domain.states) != states:
            raise ValueError(
                "plot_medoids with multiple domains requires the same state alphabet "
                "in the same order in every SequenceData object."
            )
    return reference


def _coerce_group_subset(groups: Optional[Union[int, str, Sequence]]) -> Optional[list]:
    if groups is None:
        return None
    if isinstance(groups, (list, tuple, np.ndarray)):
        return list(groups)
    return [groups]


def _match_group_subset(keys: list, wanted: list) -> list:
    wanted_ints = []
    for item in wanted:
        try:
            wanted_ints.append(int(item))
        except (TypeError, ValueError):
            continue
    keep = []
    for key in keys:
        if key in wanted:
            keep.append(key)
            continue
        try:
            if int(key) in wanted_ints:
                keep.append(key)
        except (TypeError, ValueError):
            continue
    if not keep:
        raise ValueError(
            f"No groups matched groups={wanted!r}. Available groups: {keys}."
        )
    return keep


def _grouping_id_column(group_dataframe: pd.DataFrame, reference: SequenceData) -> str:
    if reference.id_col and reference.id_col in group_dataframe.columns:
        return reference.id_col
    if "Entity ID" in group_dataframe.columns:
        return "Entity ID"
    return str(group_dataframe.columns[0])


def _drop_missing_group_values(values) -> list:
    unique = list(pd.unique(values))
    return [g for g in unique if pd.notna(g)]


def _ordered_groups(
    values,
    *,
    group_order,
    group_labels,
    sort_groups: str,
) -> list:
    unique = _drop_missing_group_values(values)
    if group_order:
        groups = [g for g in group_order if g in unique]
        missing = [g for g in unique if g not in group_order]
        if missing:
            print(f"[Warning] Groups not in group_order will be excluded: {missing}")
        return groups
    if group_labels is not None:
        mapped = []
        available = set(unique)
        for _original_key, label_value in group_labels.items():
            if label_value in available:
                mapped.append(label_value)
        leftover = available - set(mapped)
        if leftover:
            print(
                f"[Warning] Some groups in data are not in group_labels "
                f"and will be excluded: {leftover}"
            )
        return mapped
    if sort_groups in {"numeric", "auto"}:
        return smart_sort_groups(unique)
    if sort_groups == "alpha":
        return sorted(unique, key=lambda x: str(x))
    if sort_groups == "none":
        return unique
    raise ValueError(
        f"Invalid sort_groups value: {sort_groups}. "
        "Use 'auto', 'numeric', 'alpha', or 'none'."
    )


def _build_group_table_from_column(
    reference: SequenceData,
    group_by_column: str,
    group_labels: Optional[dict],
) -> pd.DataFrame:
    if group_by_column not in reference.data.columns:
        available_cols = [
            col
            for col in reference.data.columns
            if col not in reference.time and col != reference.id_col
        ]
        raise ValueError(
            f"Column '{group_by_column}' not found in the data. "
            f"Available columns for grouping: {available_cols}"
        )
    table = reference.data[[reference.id_col, group_by_column]].copy()
    table.columns = ["Entity ID", "Category"]
    unique_values = _drop_missing_group_values(reference.data[group_by_column])
    if group_labels is not None:
        missing_keys = set(unique_values) - set(group_labels.keys())
        if missing_keys:
            raise ValueError(
                f"group_labels missing mappings for values: {missing_keys}. "
                f"Please provide labels for all unique values in '{group_by_column}': "
                f"{sorted(unique_values)}"
            )
        table["Category"] = table["Category"].map(group_labels)
    n_categories = len(_drop_missing_group_values(table["Category"]))
    print(
        f"[>] Creating grouped medoid plots by '{group_by_column}' "
        f"with {n_categories} categories"
    )
    return table


def _apply_group_labels_to_table(
    group_dataframe: pd.DataFrame,
    group_column_name: str,
    group_labels: Optional[dict],
) -> pd.DataFrame:
    if group_labels is None:
        return group_dataframe
    unique_values = _drop_missing_group_values(group_dataframe[group_column_name])
    missing_keys = set(unique_values) - set(group_labels.keys())
    if missing_keys:
        raise ValueError(
            f"group_labels missing mappings for values: {missing_keys}. "
            f"Please provide labels for all unique values in '{group_column_name}': "
            f"{sorted(unique_values)}"
        )
    out = group_dataframe.copy()
    out[group_column_name] = out[group_column_name].map(group_labels)
    return out


def _member_positions(reference: SequenceData, group_ids) -> np.ndarray:
    seq_ids = np.asarray(reference.ids)
    mask = np.isin(seq_ids, np.asarray(group_ids))
    if not np.any(mask):
        mask = np.isin(seq_ids.astype(str), np.asarray(group_ids).astype(str))
    return np.where(mask)[0]


def _resolve_grouping(
    reference: SequenceData,
    *,
    group_by_column,
    group_dataframe,
    group_column_name,
    group_labels,
    group_order,
    sort_groups: str,
    groups_subset,
) -> Optional[tuple[list, list[np.ndarray], str]]:
    """Return ``(group_keys, member_index_lists, grouping_name)`` or ``None``."""
    grouping_name = group_by_column or group_column_name or ""
    if group_by_column is not None:
        group_dataframe = _build_group_table_from_column(
            reference, group_by_column, group_labels
        )
        group_column_name = "Category"
    elif group_column_name is not None and group_dataframe is None:
        print(
            "[>] Reminder: You passed `group_column_name` but not `group_dataframe`.\n"
            "    • `group_column_name` is used together with `group_dataframe` "
            "(e.g. a separate table with cluster membership).\n"
            "    • To group by a column that is already in your sequence data, "
            "use `group_by_column` instead (e.g. group_by_column='gender').\n"
            "    Proceeding without grouping."
        )
        group_column_name = None
    elif group_dataframe is not None and group_column_name is None:
        print(
            "[>] Reminder: You passed `group_dataframe` but not `group_column_name`.\n"
            "    • When using `group_dataframe` you must also specify "
            "`group_column_name` (the column that contains group IDs).\n"
            "    • Alternatively, to group by a column already in your sequence data, "
            "use `group_by_column`.\n"
            "    Proceeding without grouping."
        )
        group_dataframe = None

    if group_dataframe is None or group_column_name is None:
        return None

    if group_column_name not in group_dataframe.columns:
        raise ValueError(
            f"group_column_name {group_column_name!r} is not in group_dataframe "
            f"columns: {list(group_dataframe.columns)}"
        )

    if group_by_column is None:
        group_dataframe = _apply_group_labels_to_table(
            group_dataframe, group_column_name, group_labels
        )

    id_col_name = _grouping_id_column(group_dataframe, reference)
    group_keys = _ordered_groups(
        group_dataframe[group_column_name],
        group_order=group_order,
        group_labels=group_labels,
        sort_groups=sort_groups,
    )
    subset = _coerce_group_subset(groups_subset)
    if subset is not None:
        group_keys = _match_group_subset(group_keys, subset)

    keys_out: list = []
    members_out: list[np.ndarray] = []
    for key in group_keys:
        group_ids = group_dataframe.loc[
            group_dataframe[group_column_name] == key, id_col_name
        ].to_numpy()
        positions = _member_positions(reference, group_ids)
        if len(positions) == 0:
            print(f"[>] Skipping group '{key}' (no matching sequences).")
            continue
        keys_out.append(key)
        members_out.append(positions)
    if not keys_out:
        raise ValueError(
            "No groups have matching sequences in the data. "
            "Cannot create grouped medoid plot."
        )
    return keys_out, members_out, str(grouping_name)


def _medoid_index_in_group(
    distance_matrix: np.ndarray,
    member_indices: np.ndarray,
    weights: Optional[np.ndarray],
) -> int:
    if len(member_indices) == 0:
        raise ValueError("Cannot compute a medoid for an empty group.")
    if len(member_indices) == 1:
        return int(member_indices[0])
    sub = distance_matrix[np.ix_(member_indices, member_indices)]
    if weights is None:
        totals = sub.sum(axis=0)
    else:
        totals = sub.T @ np.asarray(weights)[member_indices]
    return int(member_indices[int(np.argmin(totals))])


def _panel_label(key, *, grouped: bool, grouping_name: str) -> str:
    key_s = str(key)
    if key_s.lower().startswith("cluster"):
        return key_s
    if grouped and "cluster" in grouping_name.lower():
        return f"Cluster {key}"
    if grouped:
        return key_s
    return f"Cluster {key}"


def _filter_medoids(
    medoid_indices: np.ndarray,
    cluster_keys: list,
    titles: list,
    entity_ids: list,
    clusters: Optional[list],
) -> tuple:
    if clusters is None:
        return medoid_indices, cluster_keys, titles, entity_ids

    wanted = list(clusters)
    wanted_ints = []
    for item in wanted:
        try:
            wanted_ints.append(int(item))
        except (TypeError, ValueError):
            continue
    keep = []
    for i, key in enumerate(cluster_keys):
        if key in wanted:
            keep.append(i)
            continue
        try:
            if int(key) in wanted_ints:
                keep.append(i)
        except (TypeError, ValueError):
            continue
    if not keep:
        raise ValueError(
            f"No medoids matched groups={wanted!r}. "
            f"Available group keys: {cluster_keys}."
        )
    keep_arr = np.asarray(keep, dtype=int)
    return (
        medoid_indices[keep_arr],
        [cluster_keys[i] for i in keep_arr],
        [titles[i] for i in keep_arr],
        [entity_ids[i] for i in keep_arr],
    )


def _select_domains(
    domains: List[SequenceData],
    domain_names: List[str],
    medoid_indices: np.ndarray,
    *,
    idle_state: Optional[str],
    drop_idle_domains: bool,
    sort_domains: Optional[str],
) -> tuple[List[SequenceData], List[str]]:
    rows = np.asarray(medoid_indices, dtype=int)
    scores = []
    for domain in domains:
        if idle_state is not None and idle_state in domain.state_mapping:
            idle = domain.state_mapping[idle_state]
            scores.append(float(np.mean(domain.values[rows] != idle)))
        else:
            scores.append(1.0)

    order = list(range(len(domains)))
    if drop_idle_domains:
        kept = [i for i in order if scores[i] > 0]
        order = kept if kept else order
    if sort_domains == "activity":
        order = sorted(order, key=lambda i: scores[i], reverse=True)
    elif sort_domains not in {None, "none"}:
        raise ValueError("sort_domains must be None, 'none', or 'activity'.")
    return [domains[i] for i in order], [domain_names[i] for i in order]


def _style_index_axes(ax, *, hide_y: bool) -> None:
    """Match sequence-index axis chrome: gray outward spines, no top/right."""
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if hide_y:
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.spines["left"].set_visible(False)
    else:
        ax.spines["left"].set_color("gray")
        ax.spines["left"].set_linewidth(0.7)
        ax.spines["left"].set_position(("outward", 5))
        ax.tick_params(axis="y", colors="gray", length=4, width=0.7, which="major")
        ax.yaxis.set_ticks_position("left")
    ax.spines["bottom"].set_color("gray")
    ax.spines["bottom"].set_linewidth(0.7)
    ax.spines["bottom"].set_position(("outward", 5))
    ax.tick_params(axis="x", colors="gray", length=4, width=0.7, which="major")
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(axis="both", which="major", direction="out")


def _prepare_imshow_values(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    data[data < 1] = np.nan
    return data


def _draw_medoid_panel(
    ax,
    *,
    values: np.ndarray,
    seqdata: SequenceData,
    title: Optional[str],
    ytick_labels: Optional[Sequence[str]],
    fontsize: int,
    xlabel: Optional[str],
    ylabel: Optional[str],
    show_cluster_titles: bool,
) -> None:
    data = _prepare_imshow_values(values)
    ax.imshow(
        np.ma.masked_invalid(data),
        aspect="auto",
        cmap=seqdata.get_colormap(),
        interpolation="nearest",
        vmin=1,
        vmax=len(seqdata.states),
    )
    set_up_time_labels_for_x_axis(seqdata, ax)
    n_rows = data.shape[0]
    if ytick_labels is None or len(ytick_labels) == 0:
        _style_index_axes(ax, hide_y=True)
    else:
        if n_rows <= 12:
            positions = np.arange(n_rows)
        else:
            n_ticks = min(11, n_rows)
            positions = np.unique(np.linspace(0, n_rows - 1, num=n_ticks, dtype=int))
        ax.set_yticks(positions)
        ax.set_yticklabels(
            [str(ytick_labels[int(i)]) for i in positions],
            fontsize=max(6, fontsize - 3),
            color="black",
        )
        _style_index_axes(ax, hide_y=False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=10, color="black")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=10, color="black")
    if show_cluster_titles and title:
        show_plot_title(ax, title, show=True, fontsize=fontsize, loc="right")


def plot_medoids(
    seqdata: Union[SequenceData, Sequence[SequenceData]],
    medoid_indices: Optional[Sequence[int]] = None,
    *,
    distance_matrix: Optional[np.ndarray] = None,
    cluster_labels: Optional[Sequence] = None,
    clusters: Optional[Union[int, str, Sequence]] = None,
    groups: Optional[Union[int, str, Sequence]] = None,
    group_by_column=None,
    group_dataframe=None,
    group_column_name=None,
    group_labels=None,
    group_order=None,
    sort_groups: str = "auto",
    ids: Optional[Sequence] = None,
    domain_names: Optional[Sequence[str]] = None,
    layout: str = "column",
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    xlabel: str = "Time",
    ylabel: Optional[str] = None,
    idle_state: Optional[str] = None,
    drop_idle_domains: bool = False,
    sort_domains: Optional[str] = None,
    save_as: Optional[str] = None,
    dpi: int = 300,
    fontsize: int = 12,
    include_legend: bool = True,
    show_ids: bool = True,
    show_group_titles: bool = True,
    show_cluster_titles: Optional[bool] = None,
    show_title: bool = True,
    show: bool = True,
    weights="auto",
):
    """
    Plot group or cluster medoids with sequence-index styling and subplot layouts.

    Grouping matches ``plot_sequence_index``:

    1. **Simplified API** — grouping column already in the sequence table::

           plot_medoids(seqdata, distance_matrix=D, group_by_column="gender")

    2. **Membership table** — clusters or any grouping in a separate dataframe::

           plot_medoids(
               seqdata,
               distance_matrix=D,
               group_dataframe=membership_df,
               group_column_name="Cluster",
           )

    ``groups=`` (or the alias ``clusters=``) keeps a subset, e.g. ``groups="Female"``
    or ``clusters=3``. If you already know the medoid row for each group, pass
    ``medoid_indices`` instead of computing them from ``distance_matrix``.
    """
    domains = _as_domain_list(seqdata)
    reference = _validate_aligned_domains(domains)

    if show_cluster_titles is not None:
        show_group_titles = show_cluster_titles

    if groups is not None and clusters is not None and groups != clusters:
        raise ValueError("Pass only one of groups= or clusters=; they are aliases.")
    groups_subset = groups if groups is not None else clusters

    if isinstance(weights, str) and weights == "auto":
        weight_vec = getattr(reference, "weights", None)
    elif weights is not None:
        weight_vec = np.asarray(weights, dtype=float).reshape(-1)
        if len(weight_vec) != len(reference.values):
            raise ValueError("Length of weights must equal number of sequences.")
    else:
        weight_vec = None

    grouping = _resolve_grouping(
        reference,
        group_by_column=group_by_column,
        group_dataframe=group_dataframe,
        group_column_name=group_column_name,
        group_labels=group_labels,
        group_order=group_order,
        sort_groups=sort_groups,
        groups_subset=groups_subset if (
            group_by_column is not None or group_dataframe is not None
        ) else None,
    )

    grouped = grouping is not None
    grouping_name = ""
    if grouping is not None:
        group_keys, members_by_group, grouping_name = grouping
        provided = (
            None
            if medoid_indices is None
            else np.asarray(medoid_indices, dtype=int).reshape(-1)
        )
        if provided is not None and len(provided) == len(group_keys):
            indices = provided
        elif distance_matrix is not None:
            distance_square = _to_square_matrix(distance_matrix)
            indices = np.asarray(
                [
                    _medoid_index_in_group(distance_square, members, weight_vec)
                    for members in members_by_group
                ],
                dtype=int,
            )
        elif provided is not None:
            raise ValueError(
                "medoid_indices length must match the number of groups "
                f"({len(group_keys)}). Got {len(provided)}. "
                "Pass one index per group, or a distance_matrix to compute them."
            )
        else:
            raise ValueError(
                "Grouped medoid plots need a distance_matrix (to find one medoid "
                "per group) or medoid_indices aligned with the groups."
            )
        cluster_keys = list(group_keys)
    else:
        if medoid_indices is None:
            if distance_matrix is None:
                raise ValueError("Provide medoid_indices or a distance_matrix.")
            _, computed = compute_medoids_from_distance_matrix(
                distance_matrix, reference, weights=weight_vec, top_k=1
            )
            medoid_indices = computed

        indices = np.asarray(medoid_indices, dtype=int).reshape(-1)
        if cluster_labels is None:
            cluster_keys = list(range(1, len(indices) + 1))
        else:
            cluster_keys = list(cluster_labels)
            if len(cluster_keys) != len(indices):
                raise ValueError("cluster_labels length must match medoid_indices.")

    if np.any((indices < 0) | (indices >= len(reference.values))):
        raise ValueError("medoid_indices must be 0-based row positions in seqdata.")

    if ids is None:
        entity_ids = [reference.ids[i] for i in indices]
    else:
        entity_ids = list(ids)
        if len(entity_ids) != len(indices):
            raise ValueError("ids length must match medoid_indices.")

    titles = []
    for key, entity_id in zip(cluster_keys, entity_ids):
        label = _panel_label(key, grouped=grouped, grouping_name=grouping_name)
        titles.append(f"{label} · {entity_id}" if show_ids else label)

    if grouping is None:
        indices, cluster_keys, titles, entity_ids = _filter_medoids(
            indices,
            cluster_keys,
            titles,
            entity_ids,
            _coerce_group_subset(groups_subset),
        )
    n_medoids = len(indices)

    if domain_names is None:
        names = [f"Domain {i + 1}" for i in range(len(domains))]
    else:
        names = list(domain_names)
        if len(names) != len(domains):
            raise ValueError("domain_names length must match the number of domains.")

    domains, names = _select_domains(
        domains,
        names,
        indices,
        idle_state=idle_state,
        drop_idle_domains=drop_idle_domains,
        sort_domains=sort_domains,
    )
    n_domains = len(domains)
    reference = domains[0]

    layout = layout.lower()
    if layout not in {"column", "grid", "stacked"}:
        raise ValueError("layout must be 'column', 'grid', or 'stacked'.")
    if layout == "stacked" and n_domains > 1:
        raise ValueError(
            "layout='stacked' is for a single SequenceData. "
            "Pass one domain, or use layout='column' / 'grid' for multiple domains."
        )

    if ylabel is None:
        ylabel = "CPC" if n_domains > 1 else ("Medoid" if layout == "stacked" else "")

    def _medoid_matrix(medoid_index: int) -> np.ndarray:
        if n_domains == 1:
            return domains[0].values[medoid_index]
        return np.vstack([domain.values[medoid_index] for domain in domains])

    if layout == "stacked":
        stacked = np.vstack([domains[0].values[i] for i in indices])
        fig_w, fig_h = figsize if figsize is not None else (10.0, max(2.2, 0.55 * n_medoids + 1.2))
        figsize = (fig_w, fig_h)
        fig, ax = plt.subplots(figsize=figsize)
        y_labels = titles if show_group_titles else [str(x) for x in entity_ids]
        _draw_medoid_panel(
            ax,
            values=stacked,
            seqdata=reference,
            title=None,
            ytick_labels=y_labels,
            fontsize=fontsize,
            xlabel=xlabel,
            ylabel=ylabel or "Medoid",
            show_cluster_titles=False,
        )
        axes_for_layout = [ax]
        nrows_out, ncols_out = 1, 1
    else:
        if layout == "column" and nrows is None and ncols is None:
            nrows_out, ncols_out = n_medoids, 1
        else:
            nrows_out, ncols_out = determine_layout(
                n_medoids,
                layout="grid" if layout == "grid" else "column",
                nrows=nrows,
                ncols=ncols,
            )
            if layout == "column" and nrows is None and ncols is None:
                nrows_out, ncols_out = n_medoids, 1

        if figsize is None:
            if n_domains == 1:
                panel_h = 1.45
            else:
                panel_h = min(11.0, max(2.6, 0.09 * n_domains + 1.5))
            figsize = (max(8.5, 5.2 * ncols_out), panel_h * nrows_out + 0.8)

        hspace = 0.42 if layout == "column" else 0.30
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(
            nrows=nrows_out,
            ncols=ncols_out,
            figure=fig,
            hspace=hspace,
            wspace=0.18,
        )
        axes = np.empty((nrows_out, ncols_out), dtype=object)
        first_ax = None
        for r in range(nrows_out):
            for c in range(ncols_out):
                if first_ax is None:
                    axes[r, c] = fig.add_subplot(gs[r, c])
                    first_ax = axes[r, c]
                else:
                    axes[r, c] = fig.add_subplot(gs[r, c], sharex=first_ax, sharey=first_ax)

        axes_flat = axes.flatten()
        for i in range(n_medoids):
            ax = axes_flat[i]
            row_i, col_i = divmod(i, ncols_out)
            y_labels = None
            if n_domains > 1:
                y_labels = names if n_domains <= 60 else None
            elif show_ids:
                y_labels = [str(entity_ids[i])]
            xlabel_i = xlabel if (row_i == nrows_out - 1 or i >= n_medoids - ncols_out) else None
            ylabel_i = ylabel if col_i == 0 else None
            _draw_medoid_panel(
                ax,
                values=_medoid_matrix(int(indices[i])),
                seqdata=reference,
                title=titles[i],
                ytick_labels=y_labels,
                fontsize=fontsize,
                xlabel=xlabel_i,
                ylabel=ylabel_i,
                show_cluster_titles=show_group_titles,
            )
        for j in range(n_medoids, len(axes_flat)):
            axes_flat[j].set_visible(False)
            axes_flat[j].axis("off")
        axes_for_layout = axes_flat

    if title and show_title:
        fig.suptitle(title, fontsize=fontsize + 2, y=1.02)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.90, wspace=0.18, hspace=0.42)

    if include_legend:
        main_buffer = save_figure_to_buffer(fig, dpi=dpi)
        legend_buffer = create_standalone_legend(
            colors=reference.color_map_by_label,
            labels=reference.labels,
            ncol=legend_ncol(len(reference.states)),
            figsize=(figsize[0] if figsize is not None else 10.0, 1.0),
            fontsize=fontsize - 2,
            dpi=dpi,
        )
        if save_as and not str(save_as).lower().endswith((".png", ".jpg", ".jpeg", ".pdf")):
            save_as = str(save_as) + ".png"
        combined_img = combine_plot_with_legend(
            main_buffer,
            legend_buffer,
            output_path=save_as,
            dpi=dpi,
            padding=20,
        )
        display_h = max(4.0, (figsize[1] if figsize is not None else 6.0) + 1.0)
        plt.figure(figsize=(figsize[0] if figsize is not None else 10.0, display_h))
        plt.imshow(combined_img)
        plt.axis("off")
        if show:
            plt.show()
        plt.close()
        return None

    if save_as and not str(save_as).lower().endswith((".png", ".jpg", ".jpeg", ".pdf")):
        save_as = str(save_as) + ".png"
    save_and_show_results(save_as, dpi=dpi, show=show)
    return fig, axes_for_layout


def plot_single_medoid(
    seqdata: SequenceData,
    distance_matrix: np.ndarray,
    weights="auto",
    show_legend: bool = True,
    title: Optional[str] = None,
    fontsize: int = 12,
    save_as: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
) -> None:
    """
    Identify one global medoid from a pairwise distance matrix and plot it.

    For cluster medoids (including subplot layouts), use :func:`plot_medoids`.
    """
    if weights != "auto" and weights is not None:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(weights) != len(seqdata.values):
            raise ValueError("Length of weights must equal number of sequences.")

    distance_matrix = _to_square_matrix(distance_matrix)
    _, medoid_indices = compute_medoids_from_distance_matrix(
        distance_matrix, seqdata, weights=weights, top_k=1
    )
    coverages = _compute_individual_medoid_coverage(distance_matrix, medoid_indices)
    medoid_index = medoid_indices[0]
    entity_id = seqdata.ids[medoid_index]
    default_title = (
        title
        if title
        else f"Medoid sequence (ID: {entity_id}, coverage: {coverages[0] * 100:.1f}%)"
    )
    plot_medoids(
        seqdata,
        [medoid_index],
        layout="column",
        title=default_title,
        fontsize=fontsize,
        include_legend=show_legend,
        save_as=save_as,
        dpi=dpi,
        show=show,
        show_cluster_titles=True,
        cluster_labels=["Medoid"],
        ids=[entity_id],
    )


def compute_medoids_from_distance_matrix(
    distance_matrix: np.ndarray,
    seqdata: SequenceData,
    weights="auto",
    top_k: Optional[int] = None,
) -> tuple:
    """Compute medoid(s) by minimizing total (weighted) distance."""
    if not isinstance(seqdata, SequenceData):
        raise TypeError("[X] seqdata must be a SequenceData object.")

    distance_matrix = _to_square_matrix(distance_matrix)

    if isinstance(weights, str) and weights == "auto":
        weights = getattr(seqdata, "weights", None)

    if weights is not None:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(weights) != len(seqdata.values):
            raise ValueError("Length of weights must equal number of sequences.")

    if weights is None:
        total_distances = distance_matrix.sum(axis=0)
    else:
        total_distances = distance_matrix.T @ weights

    min_distance = np.min(total_distances)
    medoid_indices = np.where(total_distances == min_distance)[0]

    if top_k is not None:
        sorted_indices = np.argsort(total_distances)
        medoid_indices = sorted_indices[:top_k]

    medoid_sequences = [seqdata.values[idx] for idx in medoid_indices]
    medoid_indices = [int(idx) for idx in np.asarray(medoid_indices).tolist()]
    return medoid_sequences, medoid_indices


def _compute_individual_medoid_coverage(
    distance_matrix: np.ndarray,
    medoid_indices: List[int],
    threshold_ratio: float = 0.10,
) -> List[float]:
    """Coverage share of cases within ``threshold_ratio`` of max distance of each medoid."""
    max_distance = np.max(distance_matrix)
    threshold = max_distance * threshold_ratio
    return [
        float(np.sum(distance_matrix[:, medoid] <= threshold) / len(distance_matrix))
        for medoid in medoid_indices
    ]
