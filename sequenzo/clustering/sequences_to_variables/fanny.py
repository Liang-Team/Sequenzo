"""
@Author  : Yuqi Liang 梁彧祺
@File    : fanny.py
@Time    : 02/03/2026 09:52
@Desc    :
Fuzzy Analysis Clustering (FANNY), port of R package ``cluster::fanny``.

Reference: Kaufman & Rousseeuw (1990), chapter 4; R ``cluster`` src/fanny.c.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .helpers import validate_diss_matrix


def _import_fanny_cpp():
    try:
        from sequenzo.clustering import clustering_c_code as cpp
        if hasattr(cpp, "fanny_from_diss"):
            return cpp
    except ImportError:
        pass
    return None


_cpp = None


def _get_fanny_cpp():
    global _cpp
    if _cpp is None:
        _cpp = _import_fanny_cpp()
    return _cpp


def _square_to_condensed(diss: np.ndarray) -> np.ndarray:
    """Upper-triangle condensed distances (same layout as R ``cluster`` / SciPy)."""
    n = diss.shape[0]
    dss = np.empty(n * (n - 1) // 2, dtype=float)
    pos = 0
    for lo in range(n - 1):
        for hi in range(lo + 1, n):
            dss[pos] = diss[lo, hi]
            pos += 1
    return dss


def _validate_diss_matrix(diss: np.ndarray) -> None:
    validate_diss_matrix(diss)


def _pair_dist(dss: np.ndarray, m: int, i: int, n: int) -> float:
    if m == i:
        return 0.0
    lo, hi = (m, i) if m < i else (i, m)
    return dss[lo * n - (lo + 1) * (lo + 2) // 2 + hi]


def _caddy(
    p: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Reorder membership columns to match crisp clusters (R ``caddy``).

    Returns
    -------
    p_reordered : (n, k)
    clustering : (n,) 0-based cluster ids
    k_crisp : int
    """
    n, k = p.shape
    nfuzz = np.zeros(k, dtype=int)
    ncluv = np.zeros(n, dtype=int)

    pbest = p[0, 0]
    nbest = 0
    for i in range(1, k):
        if pbest < p[0, i]:
            pbest = p[0, i]
            nbest = i
    nfuzz[0] = nbest
    ncluv[0] = 0
    ktrue = 1

    for m in range(1, n):
        pbest = p[m, 0]
        nbest = 0
        for i in range(1, k):
            if pbest < p[m, i]:
                pbest = p[m, i]
                nbest = i
        stay = False
        for ktry in range(ktrue):
            if nfuzz[ktry] == nbest:
                stay = True
                ncluv[m] = ktry
                break
        if not stay:
            nfuzz[ktrue] = nbest
            ktrue += 1
            ncluv[m] = ktrue - 1

    if ktrue < k:
        for kwalk in range(ktrue, k):
            for kleft in range(k):
                if kleft not in nfuzz[:kwalk]:
                    nfuzz[kwalk] = kleft
                    break

    p_out = np.empty_like(p)
    for m in range(n):
        for j in range(k):
            p_out[m, j] = p[m, nfuzz[j]]
    return p_out, ncluv, ktrue


@dataclass
class FannyResult:
    """Result of :func:`fanny`.

    ``iterations`` is the iteration count when converged, or ``-1`` if not converged
    (R ``cluster::fanny`` convention).
    """

    membership: np.ndarray
    clustering: np.ndarray
    memb_exp: float
    objective: float
    converged: bool
    iterations: int
    k_crisp: int
    partition_coefficient: float
    normalized_coefficient: float


def fanny(
    diss: np.ndarray,
    k: int,
    memb_exp: float = 2.0,
    max_iter: int = 500,
    tol: float = 1e-15,
    ini_mem_p: Optional[np.ndarray] = None,
    reorder_columns: bool = True,
) -> FannyResult:
    """
    Fuzzy Analysis Clustering on a dissimilarity matrix (R ``cluster::fanny``).

    Parameters
    ----------
    diss : np.ndarray
        Square ``(n, n)`` distance matrix.
    k : int
        Number of clusters. Must satisfy ``1 <= k <= n // 2 - 1`` (R ``cluster::fanny``
        convention, not a general fuzzy-clustering bound). For small ``n`` this is tight;
        e.g. ``n = 4`` allows at most ``k = 1``.
    memb_exp : float, default 2.0
        Fuzziness exponent ``m`` (must be > 1). Helske et al. (2024) use 1.4.
    max_iter : int, default 500
        Maximum iterations.
    tol : float, default 1e-15
        Relative convergence tolerance on the objective.
    ini_mem_p : np.ndarray, optional
        Initial ``(n, k)`` membership matrix with nonnegative entries and rows
        summing to 1. If None, R's default initialization is used.
    reorder_columns : bool, default True
        If True, reorder columns via the R ``caddy`` step so columns align with
        crisp clusters.

    Returns
    -------
    FannyResult
    """
    diss = np.asarray(diss, dtype=float)
    if diss.ndim != 2 or diss.shape[0] != diss.shape[1]:
        raise ValueError("diss must be a square matrix")
    n = diss.shape[0]
    if k < 1:
        raise ValueError("k must be at least 1")
    if memb_exp <= 1.0 or not np.isfinite(memb_exp):
        raise ValueError("memb_exp must be a finite number > 1")
    if max_iter < 0:
        raise ValueError("max_iter must be non-negative")
    _validate_diss_matrix(diss)

    if k > n // 2 - 1:
        raise ValueError(f"k must be at most n//2 - 1; got k={k}, n={n}")

    cpp = _get_fanny_cpp()
    if cpp is not None:
        ini_arg = None if ini_mem_p is None else np.asarray(ini_mem_p, dtype=float, order="C")
        raw = cpp.fanny_from_diss(
            np.asarray(diss, dtype=np.float64, order="C"),
            int(k),
            float(memb_exp),
            int(max_iter),
            float(tol),
            ini_arg,
            bool(reorder_columns),
        )
        iterations = int(raw["iterations"])
        converged = bool(raw["converged"])
        if not converged:
            warnings.warn(
                f"FANNY algorithm has not converged in max_iter={max_iter} iterations",
                stacklevel=2,
            )
        r = float(memb_exp)
        pc = float(raw["partition_coefficient"])
        if r == 2.0:
            npc = (k * pc - 1.0) / (k - 1.0)
        else:
            npc = float(raw["normalized_coefficient"])
        return FannyResult(
            membership=np.asarray(raw["membership"], dtype=float),
            clustering=np.asarray(raw["clustering"], dtype=int),
            memb_exp=r,
            objective=float(raw["objective"]),
            converged=converged,
            iterations=iterations,
            k_crisp=int(raw["k_crisp"]),
            partition_coefficient=pc,
            normalized_coefficient=npc,
        )

    dss = _square_to_condensed(diss)
    r = float(memb_exp)
    reen = 1.0 / (r - 1.0)

    compute_p = ini_mem_p is None
    p = np.zeros((n, k), dtype=float)
    if compute_p:
        p0 = 0.1 / (k - 1)
        p.fill(p0)
        ndk = n // k
        nd = ndk
        j = 0
        for m in range(n):
            p[m, j] = 0.9
            if m + 1 >= nd:
                j += 1
                if j + 1 == k:
                    nd = n
                else:
                    nd += ndk
    else:
        ini_mem_p = np.asarray(ini_mem_p, dtype=float)
        if ini_mem_p.shape != (n, k):
            raise ValueError("ini_mem_p must have shape (n, k)")
        if np.any(ini_mem_p < 0):
            raise ValueError("ini_mem_p must be nonnegative")
        row_sums = ini_mem_p.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError("ini_mem_p rows must sum to 1")
        p[:] = ini_mem_p

    p **= r

    dp = np.zeros((n, k), dtype=float)
    esp = np.zeros(k, dtype=float)
    ef = np.zeros(k, dtype=float)

    cryt = 0.0
    for j in range(k):
        for m in range(n):
            esp[j] += p[m, j]
            for i in range(n):
                if i != m:
                    d_mi = _pair_dist(dss, m, i, n)
                    dp[m, j] += p[i, j] * d_mi
                    ef[j] += p[i, j] * p[m, j] * d_mi
        cryt += ef[j] / (esp[j] * 2.0)
    crt = cryt

    pt = np.zeros(k, dtype=float)
    converged = False
    it = 0
    while it < max_iter:
        it += 1
        for m in range(n):
            dt = 0.0
            for j in range(k):
                denom = dp[m, j] - ef[j] / (2.0 * esp[j])
                pt[j] = (esp[j] / denom) ** reen
                dt += pt[j]
            xx = 0.0
            for j in range(k):
                pt[j] /= dt
                if pt[j] < 0.0:
                    xx += pt[j]
            for j in range(k):
                if pt[j] > 0.0:
                    pt[j] = (pt[j] / (1.0 - xx)) ** r
                else:
                    pt[j] = 0.0
                d_mj = pt[j] - p[m, j]
                esp[j] += d_mj
                for i in range(n):
                    if i != m:
                        d_mi = _pair_dist(dss, m, i, n)
                        ddd = d_mj * d_mi
                        dp[i, j] += ddd
                        ef[j] += p[i, j] * 2.0 * ddd
                p[m, j] = pt[j]

        cryt = 0.0
        for j in range(k):
            cryt += ef[j] / (esp[j] * 2.0)

        if abs(cryt - crt) <= tol * abs(cryt):
            converged = True
            break
        crt = cryt

    if not converged:
        warnings.warn(
            f"FANNY algorithm has not converged in max_iter={max_iter} iterations",
            stacklevel=2,
        )

    inv_r = 1.0 / r
    p **= inv_r

    esp_sum = esp.sum() / n
    if r == 2.0:
        pc = esp_sum
        npc = (k * pc - 1.0) / (k - 1.0)
    else:
        pc = float(np.sum(p ** 2) / n)
        xx = k ** (r - 1.0)
        npc = (xx * pc - 1.0) / (xx - 1.0)

    if reorder_columns:
        p, clustering, k_crisp = _caddy(p)
    else:
        clustering = np.argmax(p, axis=1)
        k_crisp = len(np.unique(clustering))

    return FannyResult(
        membership=p,
        clustering=clustering,
        memb_exp=r,
        objective=cryt,
        converged=converged,
        iterations=it if converged else -1,
        k_crisp=k_crisp,
        partition_coefficient=pc,
        normalized_coefficient=npc,
    )


def highest_membership_indices_from_membership(membership: np.ndarray) -> np.ndarray:
    """
    Row index with highest membership in each cluster column.

    These are **not** PAM medoids. For Helske representativeness variables,
    use medoid indices from :func:`KMedoids`, not this helper.
    """
    membership = np.asarray(membership, dtype=float)
    k = membership.shape[1]
    return np.array([int(np.argmax(membership[:, j])) for j in range(k)], dtype=int)


def fanny_membership(
    diss: np.ndarray,
    k: int,
    m: float = 1.4,
    max_iter: int = 500,
    tol: float = 1e-15,
    ini_mem_p: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FANNY membership matrix (Helske et al. 2024 soft / pseudoclass step).

    Wrapper around :func:`fanny` using membership exponent ``m`` (``memb.exp`` in R).

    Parameters
    ----------
    diss : np.ndarray
        ``(n, n)`` distance matrix.
    k : int
        Number of clusters.
    m : float, default 1.4
        Fuzziness exponent (> 1), as in Helske et al. (2024).
    max_iter, tol
        Passed to :func:`fanny`.
    ini_mem_p : np.ndarray, optional
        Initial membership matrix ``(n, k)``. FANNY initialization is deterministic
        when this is ``None`` (same as R ``cluster::fanny``).

    Returns
    -------
    U : np.ndarray, shape (n, k)
        Row-stochastic membership matrix.
    highest_membership_indices : np.ndarray, shape (k,)
        Row index of the observation with highest membership in each cluster
        column. Not PAM medoids — do not pass to :func:`representativeness_matrix`.
    """
    result = fanny(
        diss,
        k=k,
        memb_exp=m,
        max_iter=max_iter,
        tol=tol,
        ini_mem_p=ini_mem_p,
        reorder_columns=True,
    )
    highest_membership_indices = highest_membership_indices_from_membership(result.membership)
    return result.membership, highest_membership_indices


def medoid_membership_approximation(
    diss: np.ndarray,
    k: int,
    m: float = 1.4,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Distance-to-PAM-medoid membership approximation (not exact FANNY).

    Computes PAM medoids, then sets ``u_ik ∝ (1/d_ik)^(1/(m-1))`` normalized
    per row. Useful as a fast heuristic when exact FANNY is unnecessary.

    For Helske et al. (2024) soft classification, use :func:`fanny_membership`.
    """
    if m <= 1.0 or not np.isfinite(m):
        raise ValueError("m must be a finite number > 1")
    from ..k_medoids import KMedoids
    from .helpers import medoid_indices_from_kmedoids_result

    n = diss.shape[0]
    if weights is None:
        weights = np.ones(n, dtype=float)
    labels_pam = KMedoids(diss, k=k, weights=weights, method="PAMonce", verbose=False)
    medoid_indices = medoid_indices_from_kmedoids_result(labels_pam)
    d = diss[:, medoid_indices]
    eps = np.finfo(float).eps * (1 + np.max(d))
    d = np.maximum(d, eps)
    inv_exp = 1.0 / (m - 1.0)
    u = np.power(1.0 / d, inv_exp)
    u /= np.maximum(u.sum(axis=1, keepdims=True), 1e-15)
    for j, med in enumerate(medoid_indices):
        u[med, :] = 0.0
        u[med, j] = 1.0
    return u, medoid_indices
