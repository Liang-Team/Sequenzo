"""
@Author  : Yuqi Liang 梁彧祺
@File    : sa_kob.py
@Time    : 2026-04-20 16:30
@Desc    :
Sequence Analysis – Kitagawa-Oaxaca-Blinder (SA-KOB) convenience API.

Reference: 
    Rowold, C., Struffolino, E., & Fasang, A. E. (2025). Life-course-sensitive analysis of group inequalities: Combining sequence analysis with the Kitagawa–Oaxaca–Blinder decomposition. Sociological Methods & Research, 54(2), 646-705.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .kob import (
    _bootstrap_sample_indices,
    _percentile_bounds,
    _percentile_ci,
    _validate_confidence_level,
    get_kob_decomposition,
)
from .results import SAKOBBootstrapResult, SAKOBDecompositionResult

_CLUSTER_TERM_ID = 0
ClusterCoefficientReference = Literal["majority", "group0", "group1"]


@dataclass(frozen=True)
class ClusterCovariates:
    """Regression-ready cluster dummies and metadata for SA-KOB."""

    X: np.ndarray
    category_ids: np.ndarray
    column_labels: np.ndarray
    column_names: list[str]
    reference_category_id: int
    reference_label: Any
    category_id_to_label: dict[int, Any]
    k: int
    term_ids: np.ndarray
    categories: np.ndarray


def _resolve_reference_index(
    k_val: int,
    uniq: np.ndarray,
    *,
    reference_category_index: Optional[int],
    reference_cluster_label: Optional[Any],
    reference_cluster: Optional[int],
) -> int:
    if reference_cluster_label is not None:
        label_to_id = {label: idx for idx, label in enumerate(uniq)}
        if reference_cluster_label not in label_to_id:
            raise ValueError(
                "[build_cluster_covariates] reference_cluster_label must appear in categories."
            )
        return int(label_to_id[reference_cluster_label])

    ref_idx = 0 if reference_category_index is None else int(reference_category_index)
    if reference_cluster is not None:
        warnings.warn(
            "[build_cluster_covariates] reference_cluster is deprecated; "
            "use reference_category_index or reference_cluster_label.",
            DeprecationWarning,
            stacklevel=3,
        )
        ref_idx = int(reference_cluster)

    if ref_idx < 0 or ref_idx >= k_val:
        raise ValueError(
            f"[build_cluster_covariates] reference index must be between 0 and {k_val - 1}."
        )
    return ref_idx


def build_cluster_covariates(
    cluster_labels: np.ndarray,
    *,
    k: Optional[int] = None,
    categories: Optional[Sequence[Any]] = None,
    reference_category_index: Optional[int] = None,
    reference_cluster_label: Optional[Any] = None,
    reference_cluster: Optional[int] = None,
    name_prefix: str = "cluster_",
) -> ClusterCovariates:
    """
    Build cluster-membership dummies (one baseline omitted) for KOB regression.

    Internal ``category_ids`` are always contiguous integers ``0 .. k-1``.
    Original cluster labels are preserved in ``column_labels`` and
    ``category_id_to_label`` for reporting.
    """
    labels = np.asarray(cluster_labels)
    if labels.ndim != 1:
        raise ValueError("[build_cluster_covariates] cluster_labels must be a 1D array.")

    if categories is not None:
        uniq = np.asarray(list(categories), dtype=object)
        if len(pd.unique(uniq)) != len(uniq):
            raise ValueError("[build_cluster_covariates] categories must be unique.")
    else:
        uniq = np.unique(labels)

    k_val = int(k) if k is not None else int(len(uniq))
    if k_val < 2:
        raise ValueError(
            "[build_cluster_covariates] At least two clusters are required."
        )
    if len(uniq) != k_val:
        raise ValueError(
            f"[build_cluster_covariates] Number of categories ({len(uniq)}) "
            f"does not match k={k_val}."
        )

    label_to_id = {label: idx for idx, label in enumerate(uniq)}
    if not set(labels.tolist()).issubset(set(uniq.tolist())):
        raise ValueError("[build_cluster_covariates] cluster_labels contain values outside categories.")

    internal_labels = np.array([label_to_id[label] for label in labels], dtype=int)
    ref_idx = _resolve_reference_index(
        k_val,
        uniq,
        reference_category_index=reference_category_index,
        reference_cluster_label=reference_cluster_label,
        reference_cluster=reference_cluster,
    )

    column_category_ids = np.array([i for i in range(k_val) if i != ref_idx], dtype=int)
    dummies = np.column_stack(
        [(internal_labels == cid).astype(float) for cid in column_category_ids]
    )
    column_labels = np.array([uniq[i] for i in column_category_ids], dtype=object)
    column_names = [f"{name_prefix}{label}" for label in column_labels]
    category_id_to_label = {int(i): uniq[i] for i in range(k_val)}
    term_ids = np.zeros(dummies.shape[1], dtype=int)

    return ClusterCovariates(
        X=dummies,
        category_ids=column_category_ids,
        column_labels=column_labels,
        column_names=column_names,
        reference_category_id=int(ref_idx),
        reference_label=uniq[ref_idx],
        category_id_to_label=category_id_to_label,
        k=k_val,
        term_ids=term_ids,
        categories=uniq,
    )


def _resolve_binary_group_masks(
    group: np.ndarray,
    *,
    group0_value: Any,
    group1_value: Any,
) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    unique_groups = np.unique(group)
    if unique_groups.size != 2:
        raise ValueError("[sa_kob] group must have exactly two distinct values.")

    if group0_value is None and group1_value is None:
        group0_label, group1_label = unique_groups[0], unique_groups[1]
    elif group0_value is not None and group1_value is not None:
        group0_label, group1_label = group0_value, group1_value
        if group0_label not in unique_groups or group1_label not in unique_groups:
            raise ValueError("[sa_kob] group0_value and group1_value must appear in group.")
        if group0_label == group1_label:
            raise ValueError("[sa_kob] group0_value and group1_value must be different.")
    else:
        raise ValueError("[sa_kob] Provide both group0_value and group1_value, or neither.")

    mask0 = group == group0_label
    mask1 = group == group1_label
    return group0_label, group1_label, mask0, mask1


def _row_share_in_cluster(
    labels: np.ndarray,
    group_mask: np.ndarray,
    cluster_label: Any,
) -> float:
    labels = np.asarray(labels)
    in_group = int(group_mask.sum())
    if in_group == 0:
        return 0.0
    return float(np.sum(labels[group_mask] == cluster_label) / in_group)


def _relative_row_gap_percent(share0: float, share1: float) -> float:
    if share0 > share1:
        return 100.0 * (share0 - share1) / share0 if share0 > 0 else 0.0
    if share1 > share0:
        return 100.0 * (share1 - share0) / share1 if share1 > 0 else 0.0
    return 0.0


def detect_cluster_coefficient_owners(
    group: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    k: int,
    category_id_to_label: Mapping[int, Any],
    group0_value: Any = None,
    group1_value: Any = None,
    majority_gap_threshold: float = 50.0,
    neutral_cluster_owner: int = 0,
    owner_overrides: Optional[Mapping[Any, int]] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Detect cluster-specific reference owners (Rowold, Struffolino, and Fasang,
    2024, option III) for all ``k`` clusters, including the omitted baseline.

    ``owner_overrides`` may be keyed by original cluster labels or internal
    category ids; original labels take precedence when both could apply.
    """
    if neutral_cluster_owner not in (0, 1):
        raise ValueError("[detect_cluster_coefficient_owners] neutral_cluster_owner must be 0 or 1.")

    _, _, mask0, mask1 = _resolve_binary_group_masks(
        group,
        group0_value=group0_value,
        group1_value=group1_value,
    )
    labels = np.asarray(cluster_labels)

    rows = []
    owners_by_category = np.zeros(k, dtype=int)
    for category_id in range(k):
        cluster_label = category_id_to_label[category_id]
        share0 = _row_share_in_cluster(labels, mask0, cluster_label)
        share1 = _row_share_in_cluster(labels, mask1, cluster_label)
        gap_percent = _relative_row_gap_percent(share0, share1)

        override_key = None
        if owner_overrides is not None:
            if cluster_label in owner_overrides:
                override_key = cluster_label
            elif category_id in owner_overrides:
                override_key = category_id

        if override_key is not None:
            owner = int(owner_overrides[override_key])
            classification = "override"
        elif gap_percent > majority_gap_threshold:
            owner = 0 if share0 > share1 else 1
            classification = "group_specific"
        else:
            owner = int(neutral_cluster_owner)
            classification = "neutral"

        if owner not in (0, 1):
            raise ValueError(
                "[detect_cluster_coefficient_owners] owner_overrides must map to 0 or 1."
            )

        owners_by_category[category_id] = owner
        rows.append(
            {
                "category_id": category_id,
                "cluster": cluster_label,
                "share_group0": share0,
                "share_group1": share1,
                "relative_row_gap_percent": gap_percent,
                "classification": classification,
                "coefficient_owner": owner,
            }
        )

    owner_table = pd.DataFrame(rows)
    return owner_table, owners_by_category


def cluster_group_composition_table(
    group: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    group0_value: Any = None,
    group1_value: Any = None,
    cluster_order: Optional[Sequence[Any]] = None,
) -> pd.DataFrame:
    """Cluster-by-group composition table (Rowold et al., Table 2 style)."""
    group0_label, group1_label, mask0, mask1 = _resolve_binary_group_masks(
        group,
        group0_value=group0_value,
        group1_value=group1_value,
    )
    labels = np.asarray(cluster_labels)
    clusters = list(cluster_order) if cluster_order is not None else list(pd.unique(labels))

    rows = []
    for cluster in clusters:
        n0 = int(np.sum(labels[mask0] == cluster))
        n1 = int(np.sum(labels[mask1] == cluster))
        n_total = n0 + n1
        row_share0 = n0 / int(mask0.sum()) if int(mask0.sum()) > 0 else np.nan
        row_share1 = n1 / int(mask1.sum()) if int(mask1.sum()) > 0 else np.nan
        col_share0 = n0 / n_total if n_total > 0 else np.nan
        col_share1 = n1 / n_total if n_total > 0 else np.nan
        rows.append(
            {
                "cluster": cluster,
                "n_group0": n0,
                "n_group1": n1,
                "n_total": n_total,
                "row_share_group0": row_share0,
                "row_share_group1": row_share1,
                "col_share_group0": col_share0,
                "col_share_group1": col_share1,
            }
        )

    df = pd.DataFrame(rows)
    df.attrs["group0_label"] = group0_label
    df.attrs["group1_label"] = group1_label
    return df


def _common_support_table(
    group: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    group0_value: Any,
    group1_value: Any,
    cluster_order: Sequence[Any],
    min_group_count: int = 1,
) -> pd.DataFrame:
    table = cluster_group_composition_table(
        group,
        cluster_labels,
        group0_value=group0_value,
        group1_value=group1_value,
        cluster_order=cluster_order,
    )
    table = table.copy()
    table["limited_common_support"] = (table["n_group0"] < min_group_count) | (
        table["n_group1"] < min_group_count
    )
    return table


def _apply_optional_filters(
    y: np.ndarray,
    group: np.ndarray,
    cluster_labels: np.ndarray,
    X_controls: Optional[np.ndarray],
    *,
    silhouette: Optional[np.ndarray],
    silhouette_threshold: Optional[float],
    drop_missing: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    y = np.asarray(y, dtype=float)
    group = np.asarray(group)
    labels = np.asarray(cluster_labels)
    finite = np.isfinite(y)

    if X_controls is not None:
        X_controls = np.asarray(X_controls, dtype=float)
        if X_controls.ndim != 2:
            raise ValueError("[sa_kob] X_controls must be a 2D array.")
        if X_controls.shape[0] != y.shape[0]:
            raise ValueError("[sa_kob] X_controls must have the same number of rows as y.")
        finite &= np.all(np.isfinite(X_controls), axis=1)

    if silhouette is not None:
        silhouette = np.asarray(silhouette, dtype=float).ravel()
        if silhouette.shape[0] != y.shape[0]:
            raise ValueError("[sa_kob] silhouette must have the same length as y.")
        finite &= np.isfinite(silhouette)

    if not np.all(finite):
        if not drop_missing:
            raise ValueError(
                "[get_sa_kob_decomposition] y, controls, or silhouette contain missing values. "
                "Set drop_missing=True to exclude them."
            )
        keep = finite.copy()
    else:
        keep = np.ones_like(finite, dtype=bool)

    if silhouette is not None and silhouette_threshold is not None:
        keep &= silhouette >= float(silhouette_threshold)

    y = y[keep]
    group = group[keep]
    labels = labels[keep]
    if X_controls is not None:
        X_controls = X_controls[keep]

    return y, group, labels, X_controls, keep


def _assemble_design_matrix(
    cluster_covariates: ClusterCovariates,
    X_controls: Optional[np.ndarray],
    control_variable_names: Optional[Sequence[str]],
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    parts = [cluster_covariates.X]
    names = list(cluster_covariates.column_names)
    term_ids = [cluster_covariates.term_ids]
    category_ids = [cluster_covariates.category_ids]

    if X_controls is not None:
        X_controls = np.asarray(X_controls, dtype=float)
        n_controls = X_controls.shape[1]
        if control_variable_names is None:
            control_names = [f"control_{i + 1}" for i in range(n_controls)]
        else:
            control_names = list(control_variable_names)
            if len(control_names) != n_controls:
                raise ValueError(
                    "[sa_kob] control_variable_names length must match number of control columns."
                )
        parts.append(X_controls)
        names.extend(control_names)
        term_ids.append(np.arange(1, n_controls + 1, dtype=int))
        category_ids.append(np.arange(n_controls, dtype=int))

    X = np.hstack(parts)
    term_ids_arr = np.concatenate(term_ids).astype(int)
    category_ids_arr = np.concatenate([np.asarray(c, dtype=int) for c in category_ids])
    return X, names, term_ids_arr, category_ids_arr


def _owners_for_columns(owners_by_category: np.ndarray, column_category_ids: np.ndarray) -> np.ndarray:
    return np.asarray([owners_by_category[int(cid)] for cid in column_category_ids], dtype=int)


def _build_by_cluster(
    kob_result,
    cluster_covariates: ClusterCovariates,
) -> pd.DataFrame:
    by_category = kob_result.by_category
    if by_category.empty:
        raise RuntimeError("[get_sa_kob_decomposition] Expected non-empty by_category for SA-KOB.")

    cluster_rows = by_category[by_category["term_id"] == _CLUSTER_TERM_ID].copy()
    id_to_column = {
        int(cid): name
        for cid, name in zip(cluster_covariates.category_ids, cluster_covariates.column_names)
    }
    cluster_rows["cluster"] = cluster_rows["category_id"].map(cluster_covariates.category_id_to_label)
    cluster_rows["cluster_column"] = cluster_rows["category_id"].map(
        lambda cid: id_to_column.get(int(cid))
    )
    cluster_rows["is_reference_category"] = (
        cluster_rows["category_id"] == cluster_covariates.reference_category_id
    )
    return cluster_rows.sort_values("category_id").reset_index(drop=True)


def _run_sa_kob_core(
    y: np.ndarray,
    group: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    cluster_covariates: ClusterCovariates,
    owners_by_category: np.ndarray,
    owner_table: pd.DataFrame,
    X_controls: Optional[np.ndarray],
    control_variable_names: Optional[Sequence[str]],
    group0_value: Any,
    group1_value: Any,
    fallback_reference: Literal["group0", "group1", "pooled"],
    normalize_categorical: bool,
    drop_missing: bool,
    warn_common_support: bool,
    min_group_count_per_cluster: int,
) -> SAKOBDecompositionResult:
    X, variable_names, term_ids, category_ids = _assemble_design_matrix(
        cluster_covariates,
        X_controls,
        control_variable_names,
    )
    coefficient_owner_by_column = _owners_for_columns(
        owners_by_category,
        cluster_covariates.category_ids,
    )
    if X_controls is not None:
        control_owners = np.full(X_controls.shape[1], -1, dtype=int)
        coefficient_owner_by_column = np.concatenate([coefficient_owner_by_column, control_owners])

    owner_table = owner_table.copy()
    owner_table["is_reference_category"] = (
        owner_table["category_id"] == cluster_covariates.reference_category_id
    )

    kob_result = get_kob_decomposition(
        y=y,
        group=group,
        X=X,
        variable_names=variable_names,
        term_ids=term_ids,
        reference=fallback_reference,
        coefficient_owner_by_column=coefficient_owner_by_column,
        group0_value=group0_value,
        group1_value=group1_value,
        normalize_categorical=normalize_categorical,
        categorical_terms=[_CLUSTER_TERM_ID] if normalize_categorical else None,
        category_ids=category_ids,
        n_categories_by_term={_CLUSTER_TERM_ID: cluster_covariates.k},
        owner_by_category_by_term={_CLUSTER_TERM_ID: owners_by_category.tolist()},
        drop_missing=drop_missing,
    )

    composition = cluster_group_composition_table(
        group,
        cluster_labels,
        group0_value=group0_value,
        group1_value=group1_value,
        cluster_order=list(cluster_covariates.categories),
    )
    common_support_table = _common_support_table(
        group,
        cluster_labels,
        group0_value=group0_value,
        group1_value=group1_value,
        cluster_order=list(cluster_covariates.categories),
        min_group_count=min_group_count_per_cluster,
    )
    common_support_messages = [
        f"Cluster {row['cluster']!r} has limited common support "
        f"(n_group0={row['n_group0']}, n_group1={row['n_group1']})."
        for _, row in common_support_table.iterrows()
        if row["n_group0"] < min_group_count_per_cluster or row["n_group1"] < min_group_count_per_cluster
    ]
    if warn_common_support and common_support_messages:
        warnings.warn(
            "[get_sa_kob_decomposition] Common support concerns:\n" + "\n".join(common_support_messages),
            RuntimeWarning,
            stacklevel=3,
        )

    by_cluster = _build_by_cluster(kob_result, cluster_covariates)
    diagnostics = dict(kob_result.diagnostics)
    diagnostics.update(
        {
            "sa_kob": {
                "reference_cluster_label": cluster_covariates.reference_label,
                "reference_category_id": cluster_covariates.reference_category_id,
                "cluster_coefficient_reference_note": (
                    "cluster_coefficient_reference assigns cluster-level coefficient owners. "
                    "fallback_reference applies to neutral clusters and non-cluster covariates."
                ),
                "common_support_table": common_support_table,
                "common_support_messages": common_support_messages,
                "explained_detailed": float(by_cluster["explained"].sum()),
                "returns_detailed": float(by_cluster["returns"].sum()),
            }
        }
    )
    kob_result.diagnostics = diagnostics

    return SAKOBDecompositionResult(
        kob=kob_result,
        cluster_composition=composition,
        cluster_owners=owner_table,
        by_cluster=by_cluster,
        cluster_covariates=cluster_covariates,
        common_support_table=common_support_table,
    )


def get_sa_kob_decomposition(
    y: np.ndarray,
    group: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    X_controls: Optional[np.ndarray] = None,
    control_variable_names: Optional[Sequence[str]] = None,
    k: Optional[int] = None,
    categories: Optional[Sequence[Any]] = None,
    reference_category_index: Optional[int] = None,
    reference_cluster_label: Optional[Any] = None,
    reference_cluster: Optional[int] = None,
    cluster_coefficient_reference: ClusterCoefficientReference = "majority",
    majority_gap_threshold: float = 50.0,
    neutral_cluster_owner: int = 0,
    cluster_owner_overrides: Optional[Mapping[Any, int]] = None,
    group0_value: Any = None,
    group1_value: Any = None,
    fallback_reference: Literal["group0", "group1", "pooled"] = "group0",
    normalize_categorical: bool = True,
    drop_missing: bool = False,
    silhouette: Optional[np.ndarray] = None,
    silhouette_threshold: Optional[float] = None,
    warn_common_support: bool = True,
    min_group_count_per_cluster: int = 1,
    cluster_name_prefix: str = "cluster_",
    cluster_reference_mode: Optional[ClusterCoefficientReference] = None,
    reference: Optional[Literal["group0", "group1", "pooled"]] = None,
) -> SAKOBDecompositionResult:
    """
    SA-KOB decomposition with life-course cluster typology (Rowold, Struffolino,
    and Fasang, 2024).

    ``cluster_coefficient_reference`` controls cluster-level coefficient owners
    (option III). ``fallback_reference`` is the generic KOB fallback for neutral
    clusters and non-cluster covariates.

    ``cluster_owner_overrides`` may be keyed by original cluster labels or
    internal category ids; original labels take precedence when both could apply.

    SA-KOB always uses categorical normalization so ``by_cluster`` includes all
    ``k`` clusters (including the omitted baseline).
    """
    if cluster_reference_mode is not None:
        warnings.warn(
            "[get_sa_kob_decomposition] cluster_reference_mode is deprecated; "
            "use cluster_coefficient_reference.",
            DeprecationWarning,
            stacklevel=2,
        )
        cluster_coefficient_reference = cluster_reference_mode
    if reference is not None:
        warnings.warn(
            "[get_sa_kob_decomposition] reference is deprecated; use fallback_reference.",
            DeprecationWarning,
            stacklevel=2,
        )
        fallback_reference = reference

    if cluster_coefficient_reference not in ("majority", "group0", "group1"):
        raise ValueError(
            "[get_sa_kob_decomposition] cluster_coefficient_reference must be "
            "'majority', 'group0', or 'group1'."
        )
    if fallback_reference not in ("group0", "group1", "pooled"):
        raise ValueError(
            "[get_sa_kob_decomposition] fallback_reference must be 'group0', 'group1', or 'pooled'."
        )
    if not normalize_categorical:
        raise ValueError(
            "[get_sa_kob_decomposition] normalize_categorical must be True for SA-KOB, "
            "because by_cluster requires full category-level contributions including "
            "the baseline cluster."
        )

    y, group, cluster_labels, X_controls, _ = _apply_optional_filters(
        y,
        group,
        cluster_labels,
        X_controls,
        silhouette=silhouette,
        silhouette_threshold=silhouette_threshold,
        drop_missing=drop_missing,
    )

    cluster_covariates = build_cluster_covariates(
        cluster_labels,
        k=k,
        categories=categories,
        reference_category_index=reference_category_index,
        reference_cluster_label=reference_cluster_label,
        reference_cluster=reference_cluster,
        name_prefix=cluster_name_prefix,
    )

    if cluster_coefficient_reference == "majority":
        owner_table, owners_by_category = detect_cluster_coefficient_owners(
            group,
            cluster_labels,
            k=cluster_covariates.k,
            category_id_to_label=cluster_covariates.category_id_to_label,
            group0_value=group0_value,
            group1_value=group1_value,
            majority_gap_threshold=majority_gap_threshold,
            neutral_cluster_owner=neutral_cluster_owner,
            owner_overrides=cluster_owner_overrides,
        )
    elif cluster_coefficient_reference == "group0":
        owners_by_category = np.zeros(cluster_covariates.k, dtype=int)
        owner_table = pd.DataFrame(
            {
                "category_id": np.arange(cluster_covariates.k, dtype=int),
                "cluster": [cluster_covariates.category_id_to_label[i] for i in range(cluster_covariates.k)],
                "coefficient_owner": owners_by_category,
                "classification": "fixed_group0",
                "is_reference_category": [
                    i == cluster_covariates.reference_category_id for i in range(cluster_covariates.k)
                ],
            }
        )
    elif cluster_coefficient_reference == "group1":
        owners_by_category = np.ones(cluster_covariates.k, dtype=int)
        owner_table = pd.DataFrame(
            {
                "category_id": np.arange(cluster_covariates.k, dtype=int),
                "cluster": [cluster_covariates.category_id_to_label[i] for i in range(cluster_covariates.k)],
                "coefficient_owner": owners_by_category,
                "classification": "fixed_group1",
                "is_reference_category": [
                    i == cluster_covariates.reference_category_id for i in range(cluster_covariates.k)
                ],
            }
        )
    else:
        raise RuntimeError(
            "[get_sa_kob_decomposition] Unhandled cluster_coefficient_reference value."
        )

    return _run_sa_kob_core(
        y,
        group,
        cluster_labels,
        cluster_covariates=cluster_covariates,
        owners_by_category=owners_by_category,
        owner_table=owner_table,
        X_controls=X_controls,
        control_variable_names=control_variable_names,
        group0_value=group0_value,
        group1_value=group1_value,
        fallback_reference=fallback_reference,
        normalize_categorical=normalize_categorical,
        drop_missing=drop_missing,
        warn_common_support=warn_common_support,
        min_group_count_per_cluster=min_group_count_per_cluster,
    )


def get_sa_kob_decomposition_bootstrap(
    y: np.ndarray,
    group: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    X_controls: Optional[np.ndarray] = None,
    control_variable_names: Optional[Sequence[str]] = None,
    k: Optional[int] = None,
    categories: Optional[Sequence[Any]] = None,
    reference_category_index: Optional[int] = None,
    reference_cluster_label: Optional[Any] = None,
    reference_cluster: Optional[int] = None,
    cluster_coefficient_reference: ClusterCoefficientReference = "majority",
    majority_gap_threshold: float = 50.0,
    neutral_cluster_owner: int = 0,
    cluster_owner_overrides: Optional[Mapping[Any, int]] = None,
    group0_value: Any = None,
    group1_value: Any = None,
    fallback_reference: Literal["group0", "group1", "pooled"] = "group0",
    normalize_categorical: bool = True,
    drop_missing: bool = False,
    silhouette: Optional[np.ndarray] = None,
    silhouette_threshold: Optional[float] = None,
    warn_common_support: bool = False,
    min_group_count_per_cluster: int = 1,
    cluster_name_prefix: str = "cluster_",
    n_boot: int = 500,
    random_state: Optional[int] = None,
    confidence_level: float = 0.95,
    recompute_owners_each_draw: bool = True,
    stratified: bool = True,
    cluster_reference_mode: Optional[ClusterCoefficientReference] = None,
    reference: Optional[Literal["group0", "group1", "pooled"]] = None,
) -> SAKOBBootstrapResult:
    """Bootstrap uncertainty for SA-KOB with optional stratified resampling."""
    if cluster_reference_mode is not None:
        warnings.warn(
            "[get_sa_kob_decomposition_bootstrap] cluster_reference_mode is deprecated; "
            "use cluster_coefficient_reference.",
            DeprecationWarning,
            stacklevel=2,
        )
        cluster_coefficient_reference = cluster_reference_mode
    if reference is not None:
        warnings.warn(
            "[get_sa_kob_decomposition_bootstrap] reference is deprecated; use fallback_reference.",
            DeprecationWarning,
            stacklevel=2,
        )
        fallback_reference = reference

    if n_boot < 2:
        raise ValueError("[get_sa_kob_decomposition_bootstrap] n_boot must be at least 2.")
    _validate_confidence_level(confidence_level)
    if not normalize_categorical:
        raise ValueError(
            "[get_sa_kob_decomposition_bootstrap] normalize_categorical must be True for SA-KOB."
        )

    point = get_sa_kob_decomposition(
        y=y,
        group=group,
        cluster_labels=cluster_labels,
        X_controls=X_controls,
        control_variable_names=control_variable_names,
        k=k,
        categories=categories,
        reference_category_index=reference_category_index,
        reference_cluster_label=reference_cluster_label,
        reference_cluster=reference_cluster,
        cluster_coefficient_reference=cluster_coefficient_reference,
        majority_gap_threshold=majority_gap_threshold,
        neutral_cluster_owner=neutral_cluster_owner,
        cluster_owner_overrides=cluster_owner_overrides,
        group0_value=group0_value,
        group1_value=group1_value,
        fallback_reference=fallback_reference,
        normalize_categorical=normalize_categorical,
        drop_missing=drop_missing,
        silhouette=silhouette,
        silhouette_threshold=silhouette_threshold,
        warn_common_support=warn_common_support,
        min_group_count_per_cluster=min_group_count_per_cluster,
        cluster_name_prefix=cluster_name_prefix,
        cluster_reference_mode=cluster_reference_mode,
        reference=reference,
    )

    y_arr = np.asarray(y, dtype=float)
    g_arr = np.asarray(group)
    labels_arr = np.asarray(cluster_labels)
    Xc_arr = None if X_controls is None else np.asarray(X_controls, dtype=float)
    sil_arr = None if silhouette is None else np.asarray(silhouette, dtype=float).ravel()

    y_arr, g_arr, labels_arr, Xc_arr, _ = _apply_optional_filters(
        y_arr,
        g_arr,
        labels_arr,
        Xc_arr,
        silhouette=sil_arr,
        silhouette_threshold=silhouette_threshold,
        drop_missing=drop_missing,
    )

    frozen_owners = None
    if not recompute_owners_each_draw:
        frozen_owners = point.cluster_owners.set_index("category_id")["coefficient_owner"].to_dict()

    rng = np.random.default_rng(random_state)
    boot_total_gap = np.empty(n_boot, dtype=float)
    boot_explained = np.empty(n_boot, dtype=float)
    boot_returns = np.empty(n_boot, dtype=float)
    boot_intercept = np.empty(n_boot, dtype=float)
    boot_by_cluster = np.empty((n_boot, point.by_cluster.shape[0], 2), dtype=float)

    for b in range(n_boot):
        idx = _bootstrap_sample_indices(
            g_arr,
            group0_value=group0_value,
            group1_value=group1_value,
            rng=rng,
            stratified=stratified,
        )
        draw_labels = labels_arr[idx]
        draw_cov = build_cluster_covariates(
            draw_labels,
            k=point.cluster_covariates.k,
            categories=list(point.cluster_covariates.categories),
            reference_category_index=point.cluster_covariates.reference_category_id,
            name_prefix=cluster_name_prefix,
        )
        if recompute_owners_each_draw:
            owner_table, owners_by_category = detect_cluster_coefficient_owners(
                g_arr[idx],
                draw_labels,
                k=draw_cov.k,
                category_id_to_label=draw_cov.category_id_to_label,
                group0_value=group0_value,
                group1_value=group1_value,
                majority_gap_threshold=majority_gap_threshold,
                neutral_cluster_owner=neutral_cluster_owner,
                owner_overrides=cluster_owner_overrides,
            )
        else:
            owners_by_category = np.array(
                [frozen_owners[cid] for cid in range(draw_cov.k)],
                dtype=int,
            )
            owner_table = point.cluster_owners.copy()

        res = _run_sa_kob_core(
            y_arr[idx],
            g_arr[idx],
            draw_labels,
            cluster_covariates=draw_cov,
            owners_by_category=owners_by_category,
            owner_table=owner_table,
            X_controls=None if Xc_arr is None else Xc_arr[idx],
            control_variable_names=control_variable_names,
            group0_value=group0_value,
            group1_value=group1_value,
            fallback_reference=fallback_reference,
            normalize_categorical=normalize_categorical,
            drop_missing=False,
            warn_common_support=False,
            min_group_count_per_cluster=min_group_count_per_cluster,
        )

        boot_total_gap[b] = res.total_gap
        boot_explained[b] = res.explained
        boot_returns[b] = res.unexplained_returns
        boot_intercept[b] = res.unexplained_intercept
        boot_by_cluster[b, :, 0] = res.by_cluster["explained"].to_numpy()
        boot_by_cluster[b, :, 1] = res.by_cluster["returns"].to_numpy()

    by_cluster_se = point.by_cluster.copy()
    by_cluster_se["explained_se"] = np.std(boot_by_cluster[:, :, 0], axis=0, ddof=1)
    by_cluster_se["returns_se"] = np.std(boot_by_cluster[:, :, 1], axis=0, ddof=1)
    by_cluster_ci = point.by_cluster.copy()
    ci_lower, ci_upper = _percentile_bounds(confidence_level)
    by_cluster_ci["explained_ci_lower"] = np.percentile(boot_by_cluster[:, :, 0], ci_lower, axis=0)
    by_cluster_ci["explained_ci_upper"] = np.percentile(boot_by_cluster[:, :, 0], ci_upper, axis=0)
    by_cluster_ci["returns_ci_lower"] = np.percentile(boot_by_cluster[:, :, 1], ci_lower, axis=0)
    by_cluster_ci["returns_ci_upper"] = np.percentile(boot_by_cluster[:, :, 1], ci_upper, axis=0)

    return SAKOBBootstrapResult(
        point_estimate=point,
        standard_errors={
            "total_gap": float(np.std(boot_total_gap, ddof=1)),
            "explained": float(np.std(boot_explained, ddof=1)),
            "unexplained_returns": float(np.std(boot_returns, ddof=1)),
            "unexplained_intercept": float(np.std(boot_intercept, ddof=1)),
        },
        confidence_intervals={
            "total_gap": _percentile_ci(boot_total_gap, confidence_level),
            "explained": _percentile_ci(boot_explained, confidence_level),
            "unexplained_returns": _percentile_ci(boot_returns, confidence_level),
            "unexplained_intercept": _percentile_ci(boot_intercept, confidence_level),
        },
        by_cluster_standard_errors=by_cluster_se,
        by_cluster_confidence_intervals=by_cluster_ci,
        n_boot=n_boot,
        confidence_level=confidence_level,
        recompute_owners_each_draw=recompute_owners_each_draw,
    )


__all__ = [
    "ClusterCovariates",
    "SAKOBDecompositionResult",
    "SAKOBBootstrapResult",
    "build_cluster_covariates",
    "detect_cluster_coefficient_owners",
    "cluster_group_composition_table",
    "get_sa_kob_decomposition",
    "get_sa_kob_decomposition_bootstrap",
]
