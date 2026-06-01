"""
Multidomain sequence index plots: stitch per-domain cluster column plots side by side.

For each domain, :func:`sequenzo.visualization.plot_sequence_index` builds a column
layout (one subplot per cluster, all sequences in each cluster). Domain panels are
then combined horizontally so rows align by cluster.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageChops
from sequenzo.define_sequence_data import SequenceData
from sequenzo.visualization.plot_sequence_index import plot_sequence_index, smart_sort_groups


def _resolve_groups(
    group_dataframe: pd.DataFrame,
    group_column_name: str,
    group_labels: Optional[Mapping],
    group_order: Optional[Sequence],
    sort_groups: str,
    seqdata: SequenceData,
) -> tuple[pd.DataFrame, List]:
    """Return (prepared group dataframe, ordered group labels)."""
    id_col_name = (
        "Entity ID" if "Entity ID" in group_dataframe.columns else group_dataframe.columns[0]
    )

    if group_labels is not None and group_column_name in group_dataframe.columns:
        unique_values = group_dataframe[group_column_name].unique()
        missing_keys = set(unique_values) - set(group_labels.keys())
        remapping_performed = False

        if missing_keys:
            expected_cluster_count = len(group_labels)
            missing_values_list = list(missing_keys)
            all_numeric = all(
                isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v)
                for v in missing_values_list
            )
            all_large = all(
                isinstance(v, (int, float, np.integer, np.floating))
                and not pd.isna(v)
                and (v > expected_cluster_count or v < 1)
                for v in missing_values_list
            )

            if all_numeric and all_large and len(missing_values_list) == expected_cluster_count:
                sorted_missing = sorted(missing_values_list)
                medoid_to_cluster = {
                    val: idx + 1 for idx, val in enumerate(sorted_missing)
                }
                group_dataframe = group_dataframe.copy()
                remapping_performed = True
                group_dataframe[group_column_name] = group_dataframe[group_column_name].map(
                    medoid_to_cluster
                )
            else:
                raise ValueError(
                    f"group_labels missing mappings for values: {missing_keys}. "
                    f"Provide labels for all values in '{group_column_name}'."
                )

        if not remapping_performed:
            group_dataframe = group_dataframe.copy()
        group_dataframe[group_column_name] = group_dataframe[group_column_name].map(group_labels)

    if group_order:
        groups = [g for g in group_order if g in group_dataframe[group_column_name].unique()]
    elif group_labels is not None:
        available_labels = set(group_dataframe[group_column_name].unique())
        groups = [label for label in group_labels.values() if label in available_labels]
    elif sort_groups in ("numeric", "auto"):
        groups = smart_sort_groups(group_dataframe[group_column_name].unique())
    elif sort_groups == "alpha":
        groups = sorted(group_dataframe[group_column_name].unique())
    elif sort_groups == "none":
        groups = list(group_dataframe[group_column_name].unique())
    else:
        raise ValueError(
            f"Invalid sort_groups value: {sort_groups}. "
            "Use 'auto', 'numeric', 'alpha', or 'none'."
        )

    groups_with_data = []
    for g in groups:
        group_ids = group_dataframe[group_dataframe[group_column_name] == g][id_col_name].values
        if np.any(np.isin(seqdata.ids, group_ids)):
            groups_with_data.append(g)
    if not groups_with_data:
        raise ValueError("No groups have matching sequences in the data.")
    return group_dataframe, groups_with_data


def _render_domain_cluster_column_plot(
    seqdata: SequenceData,
    *,
    group_dataframe: pd.DataFrame,
    group_column_name: str,
    group_labels: Optional[Mapping],
    group_order: Optional[Sequence],
    sort_groups: str,
    domain_title: Optional[str],
    sort_by: str,
    sort_by_weight: bool,
    weights: Union[str, np.ndarray],
    plot_style: str,
    figsize: tuple[float, float],
    xlabel: str,
    ylabel: str,
    dpi: int,
    fontsize: int,
    include_legend: bool,
    proportional_scaling: bool,
    hide_y_axis: bool,
    show_cluster_titles: bool,
    show_domain_title: bool,
    sequence_gap: int,
    sequence_rows: int,
    sort_by_ids: Optional[np.ndarray],
    return_sorted_ids: bool,
) -> tuple[Image.Image, Optional[Dict]]:
    """One domain: column layout with rows = clusters (via plot_sequence_index)."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        out_path = Path(tmp.name)

    try:
        result = plot_sequence_index(
            seqdata,
            group_dataframe=group_dataframe,
            group_column_name=group_column_name,
            group_labels=group_labels,
            group_order=group_order,
            sort_groups=sort_groups,
            sort_by=sort_by,
            sort_by_weight=sort_by_weight,
            weights=weights,
            plot_style=plot_style,
            figsize=figsize,
            title=domain_title,
            xlabel=xlabel,
            ylabel=ylabel,
            save_as=str(out_path),
            dpi=dpi,
            layout="column",
            fontsize=fontsize,
            show_group_titles=show_cluster_titles,
            include_legend=include_legend,
            sequence_selection="all",
            show_title=show_domain_title,
            proportional_scaling=proportional_scaling,
            hide_y_axis=hide_y_axis,
            sequence_gap=sequence_gap,
            sequence_rows=sequence_rows,
            sort_by_ids=sort_by_ids,
            return_sorted_ids=return_sorted_ids,
            show=False,
        )
        image = Image.open(out_path).convert("RGB")
        return image, result
    finally:
        out_path.unlink(missing_ok=True)


def _trim_whitespace(image: Image.Image) -> Image.Image:
    """Crop uniform white margins from a rendered domain panel."""
    background = Image.new(image.mode, image.size, (255, 255, 255))
    bbox = ImageChops.difference(image, background).getbbox()
    if bbox:
        return image.crop(bbox)
    return image


def _stitch_images_horizontally(images: Sequence[Image.Image], gap: int = 8) -> Image.Image:
    trimmed = [_trim_whitespace(im) for im in images]
    max_h = max(im.height for im in trimmed)
    total_w = sum(im.width for im in trimmed) + gap * max(0, len(trimmed) - 1)
    canvas = Image.new("RGB", (total_w, max_h), (255, 255, 255))
    x = 0
    for im in trimmed:
        canvas.paste(im, (x, 0))
        x += im.width + gap
    return canvas


def _display_combined_image(
    image: Image.Image,
    *,
    title: Optional[str],
    show_title: bool,
    fontsize: int,
    save_as: Optional[str],
    dpi: int,
    show: bool,
) -> None:
    fig_w = max(6.0, image.width / dpi)
    fig_h = max(4.0, image.height / dpi)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(np.asarray(image))
    ax.axis("off")
    if title and show_title:
        fig.suptitle(title, fontsize=fontsize + 2, y=0.98)
    if save_as:
        if not save_as.lower().endswith((".png", ".jpg", ".jpeg", ".pdf")):
            save_as = save_as + ".png"
        fig.savefig(save_as, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_multidomain_sequence_index_by_cluster(
    seqdata_list: Sequence[SequenceData],
    *,
    domain_names: Optional[Sequence[str]] = None,
    group_dataframe: Optional[pd.DataFrame] = None,
    group_column_name: Optional[str] = None,
    group_labels: Optional[Mapping] = None,
    group_order: Optional[Sequence] = None,
    sort_groups: str = "auto",
    align_sort_across_domains: bool = True,
    sort_by: str = "lexicographic",
    sort_by_weight: bool = False,
    weights: Union[str, np.ndarray] = "auto",
    plot_style: str = "standard",
    figsize: tuple[float, float] = (5.0, 3.5),
    title: Optional[str] = None,
    xlabel: str = "Time",
    ylabel: str = "Sequences",
    save_as: Optional[str] = None,
    dpi: int = 200,
    fontsize: int = 12,
    include_legend: bool = False,
    proportional_scaling: bool = True,
    show_cluster_titles: bool = True,
    show_domain_titles: bool = False,
    show_title: bool = False,
    domain_gap: int = 8,
    sequence_gap: int = 0,
    sequence_rows: int = 1,
    show: bool = True,
) -> Optional[Dict]:
    """
    Multidomain sequence index plots: one column block per domain, stitched horizontally.

    Step 1 — for each domain, call :func:`~sequenzo.visualization.plot_sequence_index`
    with ``layout='column'`` (one row per cluster, all sequences per cluster).

    Step 2 — paste domain panels side by side so cluster rows line up.

    When ``align_sort_across_domains`` is True (default), row order within each cluster
    follows the first domain and is reused in the other domains via ``sort_by_ids``.

    Parameters
    ----------
    seqdata_list
        One :class:`SequenceData` per domain, left to right.
    domain_names
        Per-domain titles (e.g. ``["Occupation", "Seniority"]``).
    group_dataframe, group_column_name, group_labels, group_order, sort_groups
        Cluster membership (same API as ``plot_sequence_index``).
    align_sort_across_domains
        Reuse sort order from the first domain in the others.
    sort_by, sort_by_weight, weights, plot_style, figsize, xlabel, ylabel
        Passed to each per-domain ``plot_sequence_index`` call.
    save_as, dpi, fontsize, include_legend, proportional_scaling
        Output and layout options for each domain panel.
    include_legend
        If True, append a state legend under each domain panel (default False).
    show_cluster_titles, show_domain_titles, show_title
        Title visibility for clusters, domain headers, and the full figure.
    domain_gap
        Pixels of white space between domain panels after margin trimming.
    sequence_gap, sequence_rows
        Spacing options (same as ``plot_sequence_index``).
    show
        Whether to display the combined figure (set False to only save).

    Returns
    -------
    dict or None
        If ``align_sort_across_domains`` is True, mapping cluster label → sorted IDs
        from the first domain; otherwise None.
    """
    if len(seqdata_list) < 2:
        raise ValueError("seqdata_list must contain at least two SequenceData objects.")

    if group_dataframe is None or group_column_name is None:
        raise ValueError(
            "group_dataframe and group_column_name are required "
            "(e.g. clustering membership with worker IDs)."
        )

    n_domains = len(seqdata_list)
    if domain_names is None:
        domain_names = [f"Domain {i + 1}" for i in range(n_domains)]
    elif len(domain_names) != n_domains:
        raise ValueError(
            f"domain_names length ({len(domain_names)}) must match "
            f"seqdata_list length ({n_domains})."
        )

    ref = seqdata_list[0]
    for i, sd in enumerate(seqdata_list[1:], start=1):
        if len(sd.values) != len(ref.values):
            raise ValueError(
                f"Domain {i + 1} has {len(sd.values)} sequences; "
                f"domain 1 has {len(ref.values)}."
            )
        if not np.array_equal(sd.ids, ref.ids):
            raise ValueError(
                "All domains must use the same entity IDs in the same order. "
                "Re-align your SequenceData objects before plotting."
            )

    group_dataframe, groups = _resolve_groups(
        group_dataframe,
        group_column_name,
        group_labels,
        group_order,
        sort_groups,
        ref,
    )

    sequence_gap = max(0, int(sequence_gap))
    sequence_rows = max(1, int(sequence_rows))

    domain_images: List[Image.Image] = []
    sorted_ids_by_cluster: Optional[Dict] = None
    sort_by_ids: Optional[np.ndarray] = None

    for col_idx, seqdata in enumerate(seqdata_list):
        domain_title = str(domain_names[col_idx]) if show_domain_titles else None
        return_sorted = align_sort_across_domains and col_idx == 0

        image, sort_result = _render_domain_cluster_column_plot(
            seqdata,
            group_dataframe=group_dataframe,
            group_column_name=group_column_name,
            group_labels=group_labels,
            group_order=group_order,
            sort_groups=sort_groups,
            domain_title=domain_title,
            sort_by=sort_by,
            sort_by_weight=sort_by_weight,
            weights=weights,
            plot_style=plot_style,
            figsize=figsize,
            xlabel=xlabel,
            ylabel="",
            dpi=dpi,
            fontsize=fontsize,
            include_legend=include_legend,
            proportional_scaling=proportional_scaling,
            hide_y_axis=True,
            show_cluster_titles=show_cluster_titles,
            show_domain_title=show_domain_titles,
            sequence_gap=sequence_gap,
            sequence_rows=sequence_rows,
            sort_by_ids=sort_by_ids,
            return_sorted_ids=return_sorted,
        )
        domain_images.append(image)

        if return_sorted and sort_result is not None:
            sorted_ids_by_cluster = sort_result
            parts = [
                np.asarray(sort_result[g])
                for g in groups
                if g in sort_result and len(sort_result[g]) > 0
            ]
            if parts:
                sort_by_ids = np.concatenate(parts)

    combined = _stitch_images_horizontally(domain_images, gap=max(0, int(domain_gap)))
    _display_combined_image(
        combined,
        title=title,
        show_title=show_title,
        fontsize=fontsize,
        save_as=save_as,
        dpi=dpi,
        show=show,
    )

    if align_sort_across_domains:
        return sorted_ids_by_cluster
    return None


__all__ = ["plot_multidomain_sequence_index_by_cluster"]
