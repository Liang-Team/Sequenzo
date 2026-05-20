"""
@Author  : Yuqi Liang 梁彧祺
@File    : wfcmdd_fuzzy_clustering.py
@Time    : 09/05/2025 18:01
@Desc    :
Distance-based fuzzy C-medoids (WeightedCluster ``wfcmdd``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

METHODS = ("NCdd", "HNCdd", "FCMdd", "PCMdd")


@dataclass
class WfcmddResult:
    """Result object returned by :func:`wfcmdd`."""

    dnoise: Optional[float]
    memb: np.ndarray
    mobile_centers: np.ndarray
    functional: float
    method: str
    m: float
    dist_to_clusters: np.ndarray

    def to_dict(self):
        return {
            "dnoise": self.dnoise,
            "memb": self.memb,
            "mobileCenters": self.mobile_centers,
            "functional": self.functional,
            "method": self.method,
            "m": self.m,
            "dist2clusters": self.dist_to_clusters,
        }


def _validate_diss(diss: np.ndarray) -> np.ndarray:
    diss = np.asarray(diss, dtype=np.float64)
    if diss.ndim != 2 or diss.shape[0] != diss.shape[1]:
        raise ValueError("diss must be a square distance matrix.")
    if np.any(~np.isfinite(diss)):
        raise ValueError("diss contains NaN or infinite values.")
    if np.any(diss < 0):
        raise ValueError("diss must be non-negative.")
    if not np.allclose(diss, diss.T):
        raise ValueError("diss must be symmetric.")
    if not np.allclose(np.diag(diss), 0):
        raise ValueError("diss diagonal must be zero.")
    return diss


def _validate_weights(weights: np.ndarray, n: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.shape[0] != n:
        raise ValueError("weights must have one value per observation.")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and non-negative.")
    if np.sum(weights) <= 0:
        raise ValueError("at least one weight must be positive.")
    return weights


def _initialize_membership(
    memb: Union[np.ndarray, Sequence[int]],
    n: int,
    normalize_rows: bool,
) -> np.ndarray:
    if isinstance(memb, np.ndarray) and memb.ndim == 2:
        if memb.shape[0] != n:
            raise ValueError("membership rows must match the distance matrix dimension.")
        if memb.shape[1] <= 1:
            raise ValueError("membership matrix must have more than one column.")
        u = np.asarray(memb, dtype=np.float64).copy()
        if np.any(~np.isfinite(u)):
            raise ValueError("membership contains NaN or infinite values.")
        if np.any(u < 0):
            raise ValueError("membership values must be non-negative.")
        row_sums = u.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError("membership rows must have positive sums.")
        if normalize_rows:
            u /= row_sums
        return u

    memb = np.asarray(memb, dtype=int).reshape(-1)
    if memb.ndim != 1:
        raise ValueError("Provide a membership matrix or a vector of medoid seeds.")
    u = np.zeros((n, memb.size), dtype=np.float64)
    for cluster_idx, seed in enumerate(memb):
        if seed < 0 or seed >= n:
            raise ValueError("medoid seed indices must lie in [0, n-1].")
        u[seed, cluster_idx] = 1.0
    return u


def _update_membership(
    method: str,
    dist2med: np.ndarray,
    m: float,
    mexp: float,
    eta: Optional[np.ndarray],
) -> np.ndarray:
    if method == "HNCdd":
        u = np.zeros_like(dist2med)
        min_c = np.argmin(dist2med, axis=1)
        u[np.arange(dist2med.shape[0]), min_c] = 1.0
        return u

    if method in ("FCMdd", "NCdd"):
        zero_dist = dist2med == 0.0
        all_med = np.any(zero_dist, axis=1)
        u = np.zeros_like(dist2med)
        nonzero = dist2med > 0.0
        u[nonzero] = np.power(dist2med[nonzero], mexp)
        u[all_med, :] = 0.0
        u[zero_dist] = 1.0
        row_sums = u.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return u / row_sums

    if method == "PCMdd":
        if eta is None:
            raise ValueError("eta is required for PCMdd.")
        u = np.empty_like(dist2med)
        for cluster_idx in range(dist2med.shape[1]):
            ratio = dist2med[:, cluster_idx] / eta[cluster_idx]
            u[:, cluster_idx] = 1.0 / (1.0 + np.power(ratio, -mexp))
        u[dist2med == 0.0] = 1.0
        return u

    raise ValueError(f"Unsupported method: {method}")


def _compute_functional(
    method: str,
    dist2med: np.ndarray,
    u: np.ndarray,
    weights: np.ndarray,
    m: float,
    eta: Optional[np.ndarray],
    k_mov: int,
) -> float:
    if method in ("NCdd", "FCMdd"):
        return float(np.sum(dist2med * np.power(u, m) * weights[:, None]))
    if method == "HNCdd":
        return float(np.sum(dist2med * (np.power(u, m) * weights[:, None])))
    if method == "PCMdd":
        if eta is None:
            raise ValueError("eta is required for PCMdd.")
        total = 0.0
        for cluster_idx in range(k_mov):
            total += np.sum(dist2med[:, cluster_idx] * np.power(u[:, cluster_idx], m) * weights)
            total += np.sum(eta[cluster_idx] * np.power(1.0 - u[:, cluster_idx], m) * weights)
        return float(total)
    raise ValueError(f"Unsupported method: {method}")


def wfcmdd(
    diss: np.ndarray,
    memb: Union[np.ndarray, Sequence[int]],
    weights: Optional[np.ndarray] = None,
    method: str = "FCMdd",
    m: float = 2.0,
    dnoise: Optional[float] = None,
    eta: Optional[np.ndarray] = None,
    alpha: float = 0.001,
    iter_max: int = 100,
    verbose: bool = False,
    dlambda: Optional[float] = None,
) -> WfcmddResult:
    """
    Iterative distance-based fuzzy C-medoids (WeightedCluster ``wfcmdd``).

  This is a distance-based fuzzy C-medoids implementation inspired by
  WeightedCluster. It is not the Fanny algorithm used in Studer (2018).
    """
    method = next((name for name in METHODS if name.lower() == method.lower()), None)
    if method is None:
        raise ValueError(f"method must be one of {METHODS}.")

    diss = _validate_diss(diss)
    n = diss.shape[0]
    weights = _validate_weights(np.ones(n, dtype=np.float64) if weights is None else weights, n)

    if method == "NCdd" and dnoise is None and dlambda is None:
        raise ValueError("NCdd requires dnoise or dlambda.")
    if method == "HNCdd" and dnoise is None:
        raise ValueError("HNCdd requires dnoise.")
    if method == "PCMdd" and eta is None:
        raise ValueError("PCMdd requires eta.")
    if method in ("NCdd", "HNCdd") and dnoise is not None and dnoise <= 0:
        raise ValueError("dnoise must be positive.")

    if method == "HNCdd":
        m = 1.0
        mexp = None
    else:
        if m <= 1:
            raise ValueError("m must be greater than 1 for fuzzy clustering methods.")
        mexp = -(1.0 / (m - 1.0))

    if method == "NCdd" and dlambda is not None and dnoise is None:
        dnoise = 1.0

    normalize_init = method in ("FCMdd", "NCdd", "HNCdd")
    u = _initialize_membership(memb, n, normalize_rows=normalize_init)
    k_mov = u.shape[1]
    if method == "PCMdd":
        eta = np.asarray(eta, dtype=np.float64).reshape(-1)
        if len(eta) != k_mov:
            raise ValueError("eta must have one value per mobile cluster.")

    if method in ("NCdd", "HNCdd"):
        u = np.hstack([u, np.zeros((n, 1), dtype=np.float64)])
    k_mov_nc = u.shape[1]

    med = np.full(k_mov, -1, dtype=int)
    dist2med = np.zeros((n, k_mov_nc), dtype=np.float64)
    kmotion_divisor = k_mov * float(np.sum(weights)) if dlambda is not None else None

    if method in ("NCdd", "HNCdd"):
        dist2med[:, k_mov_nc - 1] = dnoise

    u_prev = u.copy()
    iteration = 0
    for iteration in range(1, iter_max + 1):
        for cluster_idx in range(k_mov):
            used_medoids = med[:cluster_idx]
            candidates_mask = np.ones(n, dtype=bool)
            for used in used_medoids:
                if used >= 0:
                    candidates_mask[used] = False
            candidates = np.flatnonzero(candidates_mask)
            if candidates.size == 0:
                raise RuntimeError("No medoid candidates remain for wfcmdd.")
            weighted = np.power(u[:, cluster_idx], m) * weights
            dist_sum = weighted @ diss[:, candidates]
            med[cluster_idx] = int(candidates[int(np.argmin(dist_sum))])
            dist2med[:, cluster_idx] = diss[:, med[cluster_idx]]

        if dlambda is not None and method == "NCdd":
            dnoise = dlambda * float(np.sum(dist2med[:, :k_mov_nc - 1] * weights[:, None]) / kmotion_divisor)
            dist2med[:, k_mov_nc - 1] = dnoise

        u = _update_membership(method, dist2med, m, mexp, eta)

        diff = np.max(np.abs(u - u_prev))
        if verbose:
            print(".", end="", flush=True)
        if diff <= alpha:
            break
        u_prev = u.copy()

    functional = _compute_functional(method, dist2med, u, weights, m, eta, k_mov)
    if verbose:
        print(f"\nIterations: {iteration} Functional: {functional}")

    return WfcmddResult(
        dnoise=dnoise,
        memb=u,
        mobile_centers=med[:k_mov].copy(),
        functional=functional,
        method=method,
        m=m,
        dist_to_clusters=dist2med.copy(),
    )


def crispness(membership: np.ndarray, norm: bool = True) -> np.ndarray:
    """
    Fuzzy partition crispness (WeightedCluster ``crispness``).

    Returns row-wise sum of squared memberships, optionally rescaled to [0, 1].
    """
    membership = np.asarray(membership, dtype=np.float64)
    values = np.sum(np.square(membership), axis=1)
    if not norm:
        return values
    k = membership.shape[1]
    if k <= 1:
        return values
    return (values - 1.0 / k) / (1.0 - 1.0 / k)
