"""
EMLT analysis (TraMineRextras ``seqemlt``).

Builds time-stamped situation profiles, distances, Benzécri covariances,
PCA coordinates, and situation correlations for sequence data.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from sequenzo.define_sequence_data import SequenceData

from .results import EMLTResult


def _alphabet(seq_matrix: np.ndarray, states: list) -> np.ndarray:
    """State alphabet in column order (TraMineR ``alphabet(seqdata)``)."""
    return np.asarray(states)


def _situation_layout(
    seq_matrix: np.ndarray,
    alphabet: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Build situation index vectors shared across EMLT steps.

    Situations are ordered as: state_1@t1, state_2@t1, ..., state_K@t1,
    state_1@t2, ... (same ordering as TraMineR ``seqemlt``).
    """
    n_periods = seq_matrix.shape[1]
    n_states = len(alphabet)
    n_situations = n_states * n_periods

    sit_time = np.repeat(np.arange(1, n_periods + 1), n_states)
    sit_states = np.tile(alphabet, n_periods)
    labels_dot = [f"{s}.{t}" for s, t in zip(sit_states, sit_time)]
    return sit_time, sit_states, np.arange(n_situations), labels_dot


def situation_frequencies(
    seq_matrix: np.ndarray,
    alphabet: np.ndarray,
) -> pd.Series:
    """
    Count sequences in each time-stamped situation (TraMineR inner ``freq``).
    """
    sit_time, sit_states, _, labels_dot = _situation_layout(seq_matrix, alphabet)
    counts = np.zeros(len(labels_dot), dtype=float)
    for i, (t_idx, state) in enumerate(zip(sit_time, sit_states)):
        counts[i] = np.sum(seq_matrix[:, t_idx - 1] == state)
    return pd.Series(counts, index=labels_dot, name="sit.freq")


def disjunctive_matrix(
    seq_matrix: np.ndarray,
    alphabet: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """
    Complete disjunctive coding of sequences (TraMineR inner ``disjonctif``).

    Row ``i`` is 1 in column ``state.t`` when sequence ``i`` occupies that situation.
    """
    sit_time, sit_states, _, labels_dot = _situation_layout(seq_matrix, alphabet)
    n_seq, n_periods = seq_matrix.shape
    n_cols = len(labels_dot)
    disj = np.zeros((n_seq, n_cols), dtype=float)

    for i in range(n_seq):
        for j in range(n_periods):
            state = seq_matrix[i, j]
            col_idx = np.where((sit_time == j + 1) & (sit_states == state))[0]
            if len(col_idx):
                disj[i, col_idx[0]] = 1.0

    return disj, labels_dot


def transition_rates(
    seq_matrix: np.ndarray,
    disj: np.ndarray,
    alphabet: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Weighted transition-rate matrix between situations (TraMineR ``transrate``).

    For situations ``i`` and ``j`` with ``j >= i``, the rate is the weighted
    co-occurrence of sequences in both situations divided by the weight in ``i``.
    """
    sit_time, _, _, labels_dot = _situation_layout(seq_matrix, alphabet)
    n_sit = len(labels_dot)
    disj_weighted = disj.copy()
    if weights is not None:
        disj_weighted = weights[:, None] * disj

    rates = np.zeros((n_sit, n_sit), dtype=float)
    for i in range(n_sit):
        nb = disj_weighted[:, i].sum()
        for j in range(i, n_sit):
            u = disj_weighted[:, i] @ disj_weighted[:, j]
            rates[i, j] = u / nb if nb > 0 else np.nan

    return pd.DataFrame(rates, index=labels_dot, columns=labels_dot)


def situation_profiles(
    seq_matrix: np.ndarray,
    transrate: pd.DataFrame,
    alphabet: np.ndarray,
    param_a: float = 1.0,
    param_b: float = 1.0,
) -> pd.DataFrame:
    """
    Time-discounted, row-normalized transition profiles (TraMineR ``profil``).

    Each row is divided by ``param_a * (t_j - t_i) + param_b`` and normalized to sum 1.
    """
    sit_time, _, _, labels_dot = _situation_layout(seq_matrix, alphabet)
    n_sit = len(labels_dot)
    tr = transrate.to_numpy()
    profiles = np.zeros((n_sit, n_sit), dtype=float)

    for i in range(n_sit):
        for j in range(i, n_sit):
            t_lag = sit_time[j] - sit_time[i]
            beta = param_a * t_lag + param_b
            if not np.isnan(tr[i, j]):
                profiles[i, j] = tr[i, j] / beta
            else:
                profiles[i, j] = np.nan
        # Match R ``sum()`` (NA if any entry is NA) and ``x[i, ] / alpha``.
        alpha = profiles[i, :].sum()
        profiles[i, :] = profiles[i, :] / alpha

    return pd.DataFrame(profiles, index=labels_dot, columns=labels_dot)


def profile_distance_matrix(
    seq_matrix: np.ndarray,
    profiles: pd.DataFrame,
    alphabet: np.ndarray,
    sit_freq: pd.Series,
) -> pd.DataFrame:
    """
    Squared Euclidean distances between situation profiles (TraMineR ``distsquare``).

    Distances are defined only for situations observed at least once; others are NA.
    """
    _, _, _, labels_dot = _situation_layout(seq_matrix, alphabet)
    active = sit_freq.to_numpy() != 0
    prof = profiles.loc[active, active].to_numpy()
    col_sums = prof.sum(axis=0, keepdims=True)
    n_active = prof.shape[0]
    d1 = np.zeros((n_active, n_active), dtype=float)

    for i in range(n_active):
        for j in range(i + 1):
            dp = prof[i, :] - prof[j, :]
            scaled = dp / col_sums.ravel()
            u = float(scaled @ dp)
            d1[i, j] = u
            d1[j, i] = u

    full = np.full((len(labels_dot), len(labels_dot)), np.nan, dtype=float)
    idx = np.where(active)[0]
    ix = np.ix_(idx, idx)
    full[ix] = d1
    return pd.DataFrame(full, index=labels_dot, columns=labels_dot)


def benzecri_covariance(distance_submatrix: np.ndarray) -> np.ndarray:
    """
    Benzécri double-centered covariance from a distance submatrix (TraMineR ``benz``).
    """
    d = np.asarray(distance_submatrix, dtype=float)
    row_mean = d.mean(axis=1, keepdims=True)
    col_mean = d.mean(axis=0, keepdims=True)
    grand_mean = row_mean.mean()
    return -0.5 * (d - col_mean - row_mean + grand_mean)


def situation_correlations(
    seq_matrix: np.ndarray,
    benz: np.ndarray,
    alphabet: np.ndarray,
    sit_freq: pd.Series,
) -> pd.DataFrame:
    """
    Pearson correlations between observed situations (TraMineR ``corel``).

    Column labels use no separator (e.g. ``'15'`` for state 1 at time 5), matching R.
    """
    sit_time, sit_states, _, labels_dot = _situation_layout(seq_matrix, alphabet)
    labels_plain = [f"{s}{t}" for s, t in zip(sit_states, sit_time)]
    active = sit_freq.to_numpy() != 0
    cor_sub = np.corrcoef(benz, rowvar=False)
    full = np.full((len(labels_dot), len(labels_dot)), np.nan, dtype=float)
    idx = np.where(active)[0]
    full[np.ix_(idx, idx)] = cor_sub
    return pd.DataFrame(full, index=labels_plain, columns=labels_plain)


def princomp_cor(x: np.ndarray) -> dict[str, np.ndarray]:
    """
    Replicate ``stats::princomp(x, cor = TRUE)`` scores (R default method).

    Uses ``cov.wt`` scaling ``cov * (1 - 1/n)`` before correlation and
    sign-fixing of eigenvector columns by the sign of their first element.
    """
    z = np.asarray(x, dtype=float)
    n_obs, _ = z.shape
    center = z.mean(axis=0)
    deviations = z - center
    cov = (deviations.T @ deviations) / (n_obs - 1)
    cov = cov * (1.0 - 1.0 / n_obs)

    sds = np.sqrt(np.diag(cov))
    if np.any(sds == 0):
        raise ValueError("Cannot use cor=TRUE with a constant variable.")
    cv = cov / np.outer(sds, sds)

    eigvals, eigvecs = np.linalg.eigh(cv)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    for col in range(eigvecs.shape[1]):
        if eigvecs[0, col] < 0:
            eigvecs[:, col] *= -1

    scaled = (z - center) / sds
    scores = scaled @ eigvecs
    sdev = np.sqrt(np.maximum(eigvals, 0.0))
    return {"scores": scores, "sdev": sdev, "loadings": eigvecs, "center": center}


def sequence_coordinates(
    disj_active: np.ndarray,
    pca_scores: np.ndarray,
) -> np.ndarray:
    """Project sequences onto PCA axes (TraMineR ``recode``)."""
    return disj_active @ pca_scores


def compute_emlt(
    seqdata: SequenceData,
    a: float = 1.0,
    b: float = 1.0,
    weighted: bool = True,
) -> EMLTResult:
    """
    Event-sequence-like situation analysis (TraMineRextras ``seqemlt``).

    Parameters
    ----------
    seqdata
        Input sequence dataset.
    a, b
        Time-discount parameters in ``1 / (a * lag + b)`` for profile construction.
    weighted
        If True, use ``seqdata.weights`` when computing transition rates; if False,
        use unweighted counts (TraMineR sets ``attr(seqdata, 'weights')`` to NULL).

    Returns
    -------
    EMLTResult
        All intermediate matrices and sequence coordinates on PCA axes.
    """
    seq_matrix = seqdata.values.astype(float)
    alphabet = _alphabet(seq_matrix, seqdata.states)
    labels_dot = [f"{s}.{t}" for s, t in zip(
        np.tile(alphabet, seq_matrix.shape[1]),
        np.repeat(np.arange(1, seq_matrix.shape[1] + 1), len(alphabet)),
    )]

    sit_freq = situation_frequencies(seq_matrix, alphabet)
    disj, _ = disjunctive_matrix(seq_matrix, alphabet)

    weights: Optional[np.ndarray]
    if weighted:
        weights = np.asarray(seqdata.weights, dtype=float)
    else:
        weights = None

    sit_transrate = transition_rates(seq_matrix, disj, alphabet, weights=weights)
    sit_profil = situation_profiles(seq_matrix, sit_transrate, alphabet, param_a=a, param_b=b)
    dist_c = profile_distance_matrix(seq_matrix, sit_profil, alphabet, sit_freq)

    active = sit_freq.to_numpy() != 0
    benz = benzecri_covariance(dist_c.loc[active, active].to_numpy())
    pca = princomp_cor(benz)
    disj_active = disj[:, active]
    coord = sequence_coordinates(disj_active, pca["scores"])
    sit_cor = situation_correlations(seq_matrix, benz, alphabet, sit_freq)

    sit_time = np.repeat(np.arange(1, seq_matrix.shape[1] + 1), len(alphabet))
    situations = np.array([f"{s}{t}" for s, t in zip(
        np.tile(alphabet, seq_matrix.shape[1]), sit_time,
    )])

    return EMLTResult(
        states=alphabet,
        period=seq_matrix.shape[1],
        sit_time=sit_time,
        situations=situations,
        sit_states=np.tile(alphabet, seq_matrix.shape[1]),
        sit_freq=sit_freq,
        disjunctive=disj,
        sit_transrate=sit_transrate,
        sit_profil=sit_profil,
        distance_matrix=dist_c,
        benz_covariance=benz,
        pca=pca,
        coord=coord,
        sit_cor=sit_cor,
        seqdata=seqdata,
        a=a,
        b=b,
        weighted=weighted,
    )


# Public alias aligned with TraMineRextras naming.
seqemlt = compute_emlt
