"""
@Author  : Yuqi Liang 梁彧祺
@File    : plot_single_medoid.py
@Time    : 26/08/2026 20:10
@Desc    :
    Cluster-medoid plots in the same visual language as ``plot_sequence_index``.

    * ``plot_medoids`` — one panel per cluster (column / grid) or all medoids
      stacked as rows; optional subset via ``clusters=``.
    * ``plot_single_medoid`` — backward-compatible helper that finds one global
      medoid from a distance matrix.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from sequenzo.define_sequence_data import SequenceData
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


def _coerce_cluster_subset(clusters: Optional[Union[int, str, Sequence]]) -> Optional[list]:
    if clusters is None:
        return None
    if isinstance(clusters, (list, tuple, np.ndarray)):
        return list(clusters)
    return [clusters]


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
            f"No medoids matched clusters={wanted!r}. "
            f"Available cluster keys: {cluster_keys}."
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
    show_cluster_titles: bool = True,
    show_title: bool = True,
    show: bool = True,
    weights="auto",
):
    """
    Plot cluster medoids with sequence-index styling and subplot layouts.

    Parameters
    ----------
    seqdata
        One ``SequenceData``, or a list of aligned domain objects (same IDs,
        same states, same length). A list draws each cluster as a domain × time
        panel, like a one-row index plot per cluster.
    medoid_indices
        0-based row indices of the medoids (one per cluster). If omitted,
        ``distance_matrix`` is used to compute a single global medoid.
    cluster_labels
        Cluster ids aligned with ``medoid_indices``. Defaults to ``1..k``.
    clusters
        Subset of cluster ids to draw. Pass an int (``clusters=3``) to inspect
        one cluster, or a list (``clusters=[1, 4]``) for a subset. ``None``
        draws every medoid.
    layout
        ``'column'`` — one subplot per cluster, stacked (index-plot groups).
        ``'grid'`` — wrapped subplot grid (same helper as ``plot_sequence_index``).
        ``'stacked'`` — all medoids as rows of a single index-style image
        (single-domain only).
    idle_state, drop_idle_domains, sort_domains
        Optional multi-domain cleanup: drop domains that are uniformly
        ``idle_state`` on the medoids, and/or sort remaining domains by activity.
    dpi
        Saved-figure resolution (default 300).
    """
    domains = _as_domain_list(seqdata)
    reference = _validate_aligned_domains(domains)

    if medoid_indices is None:
        if distance_matrix is None:
            raise ValueError("Provide medoid_indices or a distance_matrix.")
        _, computed = compute_medoids_from_distance_matrix(
            distance_matrix, reference, weights=weights, top_k=1
        )
        medoid_indices = computed

    indices = np.asarray(medoid_indices, dtype=int).reshape(-1)
    if np.any((indices < 0) | (indices >= len(reference.values))):
        raise ValueError("medoid_indices must be 0-based row positions in seqdata.")

    if cluster_labels is None:
        cluster_keys = list(range(1, len(indices) + 1))
    else:
        cluster_keys = list(cluster_labels)
        if len(cluster_keys) != len(indices):
            raise ValueError("cluster_labels length must match medoid_indices.")

    if ids is None:
        entity_ids = [reference.ids[i] for i in indices]
    else:
        entity_ids = list(ids)
        if len(entity_ids) != len(indices):
            raise ValueError("ids length must match medoid_indices.")

    titles = []
    for key, entity_id in zip(cluster_keys, entity_ids):
        label = f"Cluster {key}" if not str(key).lower().startswith("cluster") else str(key)
        titles.append(f"{label} · {entity_id}" if show_ids else label)

    indices, cluster_keys, titles, entity_ids = _filter_medoids(
        indices,
        cluster_keys,
        titles,
        entity_ids,
        _coerce_cluster_subset(clusters),
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
        y_labels = titles if show_cluster_titles else [str(x) for x in entity_ids]
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
                show_cluster_titles=show_cluster_titles,
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
