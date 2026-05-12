"""
Special hierarchical linkage methods used by ``compare_cluster_methods``.

Implements ``diana`` and ``beta.flexible`` (R ``cluster`` / WeightedCluster
``wcCmpCluster``) without an R runtime.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def _validate_square_diss(diss: np.ndarray) -> np.ndarray:
    diss = np.asarray(diss, dtype=np.float64, order="C")
    if diss.ndim != 2 or diss.shape[0] != diss.shape[1]:
        raise ValueError("diss must be a square distance matrix.")
    np.fill_diagonal(diss, 0.0)
    return diss


def _pair_key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def beta_flexible_linkage(diss: np.ndarray, *, par_method: float = 0.625) -> np.ndarray:
    """
    Agglomerative beta-flexible linkage (R ``agnes(..., method='flexible')``).
    """
    diss = _validate_square_diss(diss)
    n = diss.shape[0]
    if n < 2:
        raise ValueError("diss must contain at least two observations.")

    alpha = float(par_method)
    beta = 1.0 - 2.0 * alpha
    if not (0.0 < alpha <= 1.0):
        raise ValueError("par_method must be in (0, 1].")

    active = set(range(n))
    dist = {_pair_key(i, j): diss[i, j] for i, j in combinations(range(n), 2)}
    linkage = np.zeros((n - 1, 4), dtype=np.float64)
    counts = {i: 1 for i in range(n)}
    node_id = {i: i for i in range(n)}
    next_cluster = n

    for step in range(n - 1):
        i, j = min(dist, key=dist.get)
        distance = dist.pop((i, j))
        linkage[step, 0] = node_id[i]
        linkage[step, 1] = node_id[j]
        linkage[step, 2] = distance
        linkage[step, 3] = counts[i] + counts[j]

        active.remove(i)
        active.remove(j)
        new = next_cluster
        next_cluster += 1
        counts[new] = int(linkage[step, 3])
        node_id[new] = n + step
        active.add(new)

        for other in list(active):
            if other == new:
                continue
            dist_new = alpha * dist.pop(_pair_key(i, other)) + alpha * dist.pop(_pair_key(j, other)) + beta * distance
            dist[_pair_key(new, other)] = dist_new

    return linkage


def _diana_split(cluster: np.ndarray, diss: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    members = np.asarray(cluster, dtype=np.int32)
    if members.size < 2:
        return members, np.array([], dtype=np.int32)

    sub = diss[np.ix_(members, members)]
    avg = sub.mean(axis=1)
    splinter_seed = int(members[int(np.argmax(avg))])
    splinter = {splinter_seed}
    remaining = {int(idx) for idx in members} - splinter

    changed = True
    while changed and remaining:
        changed = False
        for idx in list(remaining):
            dist_splinter = min(diss[idx, member] for member in splinter)
            rest = [other for other in remaining if other != idx]
            dist_rest = np.mean([diss[idx, other] for other in rest]) if rest else np.inf
            if dist_splinter < dist_rest:
                splinter.add(idx)
                remaining.remove(idx)
                changed = True

    if not remaining:
        remaining = {int(members[0])}
        splinter.discard(int(members[0]))

    return np.fromiter(remaining, dtype=np.int32), np.fromiter(splinter, dtype=np.int32)


def diana_linkage(diss: np.ndarray) -> np.ndarray:
    """
    Divisive hierarchical clustering (R ``diana``).
    """
    diss = _validate_square_diss(diss)
    n = diss.shape[0]
    if n < 2:
        raise ValueError("diss must contain at least two observations.")

    clusters = [np.arange(n, dtype=np.int32)]
    splits: list[tuple[np.ndarray, np.ndarray, float]] = []
    while any(cluster.size > 1 for cluster in clusters):
        diameters = []
        for members in clusters:
            if members.size < 2:
                diameters.append(-1.0)
                continue
            sub = diss[np.ix_(members, members)]
            diameters.append(float(np.max(sub)))

        split_idx = int(np.argmax(diameters))
        members = clusters.pop(split_idx)
        left, right = _diana_split(members, diss)
        height = float(np.max(diss[np.ix_(members, members)]))
        splits.append((left, right, height))
        clusters.append(left)
        clusters.append(right)

    nodes = {frozenset({idx}): idx for idx in range(n)}
    counts = {idx: 1 for idx in range(n)}
    linkage = np.zeros((n - 1, 4), dtype=np.float64)
    for step, (left, right, height) in enumerate(reversed(splits)):
        key_left = frozenset(int(value) for value in left)
        key_right = frozenset(int(value) for value in right)
        left_id = nodes[key_left]
        right_id = nodes[key_right]
        linkage[step, 0] = left_id
        linkage[step, 1] = right_id
        linkage[step, 2] = height
        linkage[step, 3] = counts[left_id] + counts[right_id]
        new_id = n + step
        nodes[key_left | key_right] = new_id
        counts[new_id] = int(linkage[step, 3])

    return linkage
