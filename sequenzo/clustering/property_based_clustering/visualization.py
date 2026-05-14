"""
@Author  : Yuqi Liang 梁彧祺
@File    : visualization.py
@Time    : 14/05/2026 08:33
@Desc    :
Text and graphical display of property-based clustering trees.

Mirrors WeightedCluster / TraMineR ``print.seqtreeclust`` and ``seqtreedisplay``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from sequenzo.discrepancy_analysis.trees.tree_visualization import plot_tree, print_tree

from .tree_schedule import prune_property_tree, tree_labels


def _node_discrepancy(node) -> float:
    return float(node.info.get("vardis", 0.0))


def _split_strength(node) -> Optional[float]:
    if node.is_terminal or node.split is None or node.split.info is None:
        return None
    return float(node.split.info.get("R2", 0.0))


def print_property_tree(
    tree: Dict[str, Any],
    digits: int = 5,
    show_schedule: bool = True,
) -> None:
    """
    Print a property-based clustering tree in WeightedCluster style.

    The output follows the layout of ``print(seqpropclust(...))`` in the R tutorial.
    """
    params = tree["info"].get("parameters", {})
    print("Dissimilarity tree:")
    print(
        f" Parameters: min.size={params.get('min_size', 'NA')}, "
        f"max.depth={params.get('max_depth', 'NA')}, "
        f"R={params.get('R', 'NA')}, pval={params.get('pval', 'NA')} "
    )
    formula = tree["info"].get("formula")
    if formula:
        print(f" Formula: {formula}")
    adjustment = tree["info"].get("adjustment")
    if isinstance(adjustment, dict) and "R2" in adjustment:
        print(f" Global R2: {adjustment['R2']:.{digits}g}")
    print("\n Fitted tree:\n")
    print_tree(tree, digits=digits, medoid=True)


def plot_property_tree(
    tree: Dict[str, Any],
    diss: Optional[Any] = None,
    n_clusters: Optional[int] = None,
    filename: Optional[str] = None,
    show_tree: bool = True,
    **kwargs,
) -> None:
    """
    Plot a property-based clustering tree.

    Basic text and node-link display. Embedded sequence-distribution panels like
    WeightedCluster ``seqtreedisplay(type=\"d\")`` are not implemented yet.

    If ``n_clusters`` is set, the tree is pruned before plotting. When ``diss``
    is supplied, discrepancy-based quality annotations can be added by the
    underlying tree plotter.
    """
    plotted = tree if n_clusters is None else prune_property_tree(tree, n_clusters, diss=diss)
    print_tree(plotted, **{k: v for k, v in kwargs.items() if k in {"digits", "medoid"}})
    plot_tree(plotted, filename=filename, show_tree=show_tree, **kwargs)
