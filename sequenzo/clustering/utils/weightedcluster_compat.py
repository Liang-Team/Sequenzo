"""
Helpers for hierarchical clustering labels.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import cut_tree

from .special_linkage import beta_flexible_linkage, diana_linkage


def divisive_hclust_linkage(diss: np.ndarray, method: str) -> np.ndarray:
    """
    Build a linkage matrix with ``diana`` or ``beta.flexible``.
    """
    method = method.lower()
    if method == "diana":
        return diana_linkage(diss)
    if method == "beta.flexible":
        return beta_flexible_linkage(diss)
    raise ValueError("method must be 'diana' or 'beta.flexible'.")


def cutree_labels(linkage_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    """Return 1-based cluster labels from a linkage matrix."""
    return cut_tree(linkage_matrix, n_clusters=n_clusters).ravel() + 1
