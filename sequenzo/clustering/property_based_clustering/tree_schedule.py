"""
@Author  : Yuqi Liang 梁彧祺
@File    : tree_schedule.py
@Time    : 13/05/2026 21:48
@Desc    :
Tree scheduling and pruning utilities for property-based clustering.

Mirrors WeightedCluster ``clusterSplitSchedule``, ``dtprune``, ``dtcut``,
and ``dtlabels``.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from sequenzo.discrepancy_analysis.stats.single_factor_association import single_factor_association
from sequenzo.discrepancy_analysis.trees.tree_leaf_helpers import get_leaf_membership
from sequenzo.discrepancy_analysis.trees.tree_node import DissTreeNode


def _reset_split_schedule(node: DissTreeNode) -> None:
    node.info["splitschedule"] = 0
    if not node.is_terminal:
        _reset_split_schedule(node.kids[0])
        _reset_split_schedule(node.kids[1])


def _set_split_schedule(node: DissTreeNode, schedule: int, node_ids: List[int]) -> None:
    if node.id in node_ids:
        node.info["splitschedule"] = schedule
    elif not node.is_terminal:
        _set_split_schedule(node.kids[0], schedule, node_ids)
        _set_split_schedule(node.kids[1], schedule, node_ids)


def _count_nodes(node: DissTreeNode) -> tuple[int, int]:
    total = 1
    splits = 0 if node.is_terminal else 1
    if not node.is_terminal:
        left_total, left_splits = _count_nodes(node.kids[0])
        right_total, right_splits = _count_nodes(node.kids[1])
        total += left_total + right_total
        splits += left_splits + right_splits
    return total, splits


def _find_best_unscheduled_split(
    node: DissTreeNode,
    sc_explained: Dict[str, float],
    child_ids: Dict[str, List[int]],
) -> None:
    if node.is_terminal:
        return
    if node.kids[0].info.get("splitschedule", 0) == 0:
        total_sc = node.info["vardis"] * node.info["n"]
        r2 = node.split.info.get("R2", 0.0) if node.split and node.split.info else 0.0
        sc_explained[str(node.id)] = r2 * total_sc
        child_ids[str(node.id)] = [node.kids[0].id, node.kids[1].id]
    else:
        _find_best_unscheduled_split(node.kids[0], sc_explained, child_ids)
        _find_best_unscheduled_split(node.kids[1], sc_explained, child_ids)


def cluster_split_schedule(tree: Dict[str, Any]) -> Dict[str, Any]:
    """
    Order internal splits by global relevance (explained discrepancy).

    Mirrors WeightedCluster ``clusterSplitSchedule``.
    """
    tree = deepcopy(tree)
    root = tree["root"]
    _, num_splits = _count_nodes(root)
    _reset_split_schedule(root)
    root.info["splitschedule"] = 1

    for schedule in range(2, num_splits + 2):
        sc_explained: Dict[str, float] = {}
        child_ids: Dict[str, List[int]] = {}
        _find_best_unscheduled_split(root, sc_explained, child_ids)
        if not sc_explained:
            break
        best_id = max(sc_explained, key=sc_explained.get)
        _set_split_schedule(root, schedule, child_ids[best_id])

    tree["root"] = root
    tree["info"]["split_schedule_applied"] = True
    return tree


def _format_split_condition(tree: Dict[str, Any], split) -> List[str]:
    var_name = tree["data"].columns[split.varindex]
    if split.breaks is not None:
        return [f"{var_name} <= {split.breaks:g}", f"{var_name} > {split.breaks:g}"]
    left = [split.labels[i] for i in range(len(split.index)) if split.index[i] == 1]
    right = [split.labels[i] for i in range(len(split.index)) if split.index[i] == 2]
    conditions = [
        f"{var_name} in [{', '.join(left)}]",
        f"{var_name} in [{', '.join(right)}]",
    ]
    if split.naGroup is not None:
        conditions[split.naGroup - 1] += " with NA"
    return conditions


def tree_labels(tree: Dict[str, Any]) -> Dict[str, str]:
    """
    Build human-readable leaf labels from split paths.

    Mirrors WeightedCluster ``dtlabels``.
    """
    labels: Dict[str, List[str]] = {}

    def _add_label(parent: DissTreeNode, child: DissTreeNode, text: str) -> None:
        parent_id = str(parent.id)
        child_id = str(child.id)
        labels.setdefault(child_id, [])
        if parent_id in labels:
            labels[child_id] = labels[parent_id] + [text]
        else:
            labels[child_id] = [text]

    def _walk(node: DissTreeNode) -> None:
        if node.is_terminal:
            return
        left_text, right_text = _format_split_condition(tree, node.split)
        _add_label(node, node.kids[0], left_text)
        _add_label(node, node.kids[1], right_text)
        _walk(node.kids[0])
        _walk(node.kids[1])

    _walk(tree["root"])
    return {node_id: " & ".join(parts) for node_id, parts in labels.items()}


def cut_tree(
    tree: Dict[str, Any],
    n_clusters: int,
    labels: bool = True,
) -> Union[np.ndarray, pd.Series]:
    """
    Obtain a flat partition with ``n_clusters`` groups from a scheduled tree.

    Mirrors WeightedCluster ``dtcut``.
    """
    root = tree["root"]
    fitted = tree["fitted"]["(fitted)"].to_numpy()
    max_k = len(np.unique(fitted))
    if n_clusters > max_k:
        raise ValueError(f"The maximum number of groups is {max_k}.")

    assignment = np.full(len(fitted), root.id, dtype=int)

    def _assign(node: DissTreeNode) -> None:
        schedule = node.info.get("splitschedule", 0)
        if schedule <= n_clusters:
            assignment[node.info["ind"]] = node.id
        if not node.is_terminal:
            _assign(node.kids[0])
            _assign(node.kids[1])

    _assign(root)
    if not labels:
        return assignment

    label_map = tree_labels(tree)
    return pd.Series([label_map.get(str(node_id), f"Node_{node_id}") for node_id in assignment])


def prune_property_tree(
    tree: Dict[str, Any],
    n_clusters: int,
    diss: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Prune a scheduled tree to ``n_clusters`` terminal groups.

    Mirrors WeightedCluster ``dtprune``.
    """
    tree = deepcopy(tree)
    root = tree["root"]

    def _prune(node: DissTreeNode) -> None:
        if node.is_terminal:
            return
        child_schedule = node.kids[0].info.get("splitschedule", 0)
        if child_schedule > n_clusters:
            node.split = None
            node.kids = None
        else:
            _prune(node.kids[0])
            _prune(node.kids[1])

    _prune(root)
    tree["root"] = root
    tree["fitted"] = pd.DataFrame({"(fitted)": get_leaf_membership(tree)})
    tree["info"]["prune"] = n_clusters

    if diss is not None:
        weights = tree.get("weights")
        weight_permutation = tree["info"].get("weight_permutation", "none")
        tree["info"]["adjustment"] = single_factor_association(
            distance_matrix=diss,
            group=tree["fitted"]["(fitted)"].to_numpy(),
            weights=weights,
            R=tree["info"]["parameters"].get("R", 1),
            weight_permutation=weight_permutation,
            squared=False,
        )
    else:
        tree["info"]["adjustment"] = None
    return tree
