"""
@Author  : Yuqi Liang 梁彧祺
@File    : property_clustering.py
@Time    : 13/05/2026 13:51
@Desc    :
Property-based divisive clustering for sequence analysis (Studer 2018).

Mirrors WeightedCluster ``seqpropclust``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from sequenzo.define_sequence_data import SequenceData
from sequenzo.discrepancy_analysis.trees.sequence_tree import sequence_tree

from .property_extraction import extract_sequence_properties
from .tree_schedule import cluster_split_schedule, prune_property_tree


def property_based_clustering(
    seqdata: SequenceData,
    diss: np.ndarray,
    properties: Sequence[str] = ("state", "duration"),
    other_properties: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    properties_only: bool = False,
    max_clusters: Optional[int] = None,
    with_missing: bool = True,
    pmin_support: float = 0.05,
    max_k: int = -1,
    R: int = 1,
    weight_permutation: str = "diss",
    min_size: Union[float, int] = 0.01,
    max_depth: int = 5,
    pval: float = 1.0,
    verbose: bool = True,
    **tree_kwargs: Any,
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Property-based divisive clustering on sequence data.

    This function extracts sequence properties, builds a discrepancy tree with
    :func:`~sequenzo.discrepancy_analysis.trees.sequence_tree.sequence_tree`,
    orders splits by global relevance, and optionally prunes the tree to
    ``max_clusters`` groups.

    Parameters mirror WeightedCluster ``seqpropclust``.

    Parameters
    ----------
    seqdata : SequenceData
        State sequence object.
    diss : np.ndarray
        Precomputed distance matrix.
    properties : sequence of str
        Property names to extract (see :func:`extract_sequence_properties`).
    other_properties : DataFrame, optional
        Additional user-defined properties.
    properties_only : bool, default False
        If True, return only the extracted property matrix.
    max_clusters : int, optional
        If provided, prune the tree to this number of groups (``maxcluster`` in R).
    with_missing, pmin_support, max_k
        Passed to property extraction.
    R, weight_permutation, min_size, max_depth, pval
        Passed to the underlying discrepancy tree.
    verbose : bool, default True
        Print extraction progress.
    **tree_kwargs
        Extra arguments forwarded to :func:`sequence_tree`.

    Returns
    -------
    dict or DataFrame
        Property matrix if ``properties_only=True``; otherwise a scheduled /
        pruned tree object with classes ``seqtreeclust`` metadata in ``info``.
    """
    predictors = extract_sequence_properties(
        seqdata=seqdata,
        properties=properties,
        other_properties=other_properties,
        with_missing=with_missing,
        pmin_support=pmin_support,
        max_k=max_k,
        verbose=verbose,
    )
    if properties_only:
        return predictors

    diss = np.asarray(diss, dtype=np.float64)
    if diss.shape[0] != seqdata.seqdata.shape[0]:
        raise ValueError("diss must have one row/column per sequence.")

    tree = sequence_tree(
        seqdata=seqdata,
        predictors=predictors,
        distance_matrix=diss,
        weighted=True,
        min_size=min_size,
        max_depth=max_depth,
        R=R,
        pval=pval,
        weight_permutation=weight_permutation,
        **tree_kwargs,
    )
    tree["info"]["method"] = "seqpropclust"
    tree["info"]["properties"] = list(predictors.columns)
    tree["info"]["formula"] = "seqdata ~ " + " + ".join(f"`{col}`" for col in predictors.columns)
    tree["seqdata"] = seqdata
    tree["object"] = seqdata

    tree = cluster_split_schedule(tree)
    if max_clusters is not None:
        tree = prune_property_tree(tree, n_clusters=max_clusters, diss=diss)
    return tree


# R / WeightedCluster alias
seqpropclust = property_based_clustering
