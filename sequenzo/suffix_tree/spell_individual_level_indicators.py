"""
@Author  : Yuqi Liang 梁彧祺
@File    : spell_individual_level_indicators.py
@Time    : 2026/1/30 15:57
@Desc    : Individual-level indicators for spell-based suffix tree analysis.

Spell-based Suffix Tree: Individual-level convergence indicators.

Provides per-sequence (per-individual) rarity and convergence measures when the
unit of analysis is SPELL from the end. Each "level" is one spell from the end
(last spell, last two spells, ...). Lower rarity = more typical ending pattern.
Variable-length sequences are supported: individuals with fewer spells have NaN
at spell levels beyond their length.

Design mirrors: sequenzo/suffix_tree/individual_level_indicators.py (position-based).
- Position version: level = time index from end, suffix = states from year t to end.
- Spell version:    level = spell index from end, suffix = last k spells.

Usage:
    from sequenzo.suffix_tree import build_spell_suffix_tree
    from sequenzo.suffix_tree.spell_individual_level_indicators import SpellIndividualConvergence

    tree = build_spell_suffix_tree(seqdata, expcost=0)
    ind = SpellIndividualConvergence(tree)
    rarity_per_spell = ind.compute_suffix_rarity_per_spell()
    converged = ind.compute_converged(method="zscore", z_threshold=1.5)
"""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .spell_level_indicators import SpellSuffixTree


_EPS = 1e-10


class SpellIndividualConvergence:
    """
    Individual-level convergence and rarity for spell-based suffix trees.

    Requires a SpellSuffixTree built with build_spell_suffix_tree(seqdata, ...),
    so that tree._spell_states and tree._spell_durations exist and tree.counts /
    tree.total_sequences are populated. Suffix at level k = last k spells (from end).
    Lower rarity = more typical ending; converged = low rarity (z < -z_threshold).
    """

    def __init__(self, tree: SpellSuffixTree):
        if not isinstance(tree, SpellSuffixTree):
            raise TypeError(
                "[!] SpellIndividualConvergence requires a SpellSuffixTree. "
                "Use: build_spell_suffix_tree(seqdata) then SpellIndividualConvergence(tree)"
            )
        if not hasattr(tree, "_spell_states") or not hasattr(tree, "_spell_durations"):
            raise ValueError(
                "[!] SpellSuffixTree must be built with build_spell_suffix_tree(seqdata) "
                "so that _spell_states and _spell_durations are attached."
            )
        self.tree = tree
        self.spell_states = tree._spell_states
        self.spell_durations = tree._spell_durations
        self.N = tree.total_sequences
        self.max_spells = max(len(s) for s in self.spell_states) if self.spell_states else 0

    def _build_rarity_matrix(self) -> np.ndarray:
        """
        Build (N, max_spells) matrix of suffix rarity at each spell level (from end).
        Level k = last k spells. rarity_{i,k} = -log( freq(suffix_{i,k}) / N ).
        Cells where individual i has no spell at that level from end are np.nan.
        """
        N, max_spells = self.N, self.max_spells
        counts = self.tree.counts
        rarity = np.full((N, max_spells), np.nan, dtype=float)
        for i, states_i in enumerate(self.spell_states):
            rev = list(reversed(states_i))
            for k in range(len(rev)):
                key = tuple(rev[: k + 1])
                freq = counts.get(key, 0) / max(N, 1)
                rarity[i, k] = -np.log(freq + _EPS)
        return rarity

    def compute_suffix_rarity_per_spell(
        self,
        as_dataframe: bool = True,
        column_prefix: str = "k",
        zscore: bool = False,
    ):
        """
        Compute per-spell-level suffix rarity for each individual (from end).

        Level k = last k spells. Higher rarity = rarer ending pattern.
        Levels beyond an individual's spell length are NaN.
        """
        rarity = self._build_rarity_matrix()
        if zscore:
            col_means = np.nanmean(rarity, axis=0)
            col_stds = np.nanstd(rarity, axis=0, ddof=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                rarity = (rarity - col_means) / col_stds
            rarity = np.where(np.isfinite(rarity), rarity, np.nan)
        if not as_dataframe:
            return rarity
        columns = [f"{column_prefix}{k + 1}" for k in range(self.max_spells)]
        return pd.DataFrame(rarity, columns=columns)

    def compute_suffix_rarity_score(self) -> List[float]:
        """
        One aggregated rarity score per individual: sum of -log(freq/N) over spell levels (from end).
        Higher = rarer ending path.
        """
        rarity = self._build_rarity_matrix()
        scores = []
        for i in range(self.N):
            row = rarity[i, :]
            valid = np.isfinite(row)
            scores.append(float(np.sum(row[valid])) if np.any(valid) else np.nan)
        return scores

    def compute_standardized_rarity_score(
        self,
        min_k: int = 1,
        window: int = 1,
    ) -> List[float]:
        """
        Standardized rarity score per individual for convergence classification.

        For convergence we take the minimum (most typical): standardized_score_i =
        min over starting spell level of (max over window of z_{i,k}). Lower = more typical.
        """
        rarity = self._build_rarity_matrix()
        col_means = np.nanmean(rarity, axis=0)
        col_stds = np.nanstd(rarity, axis=0, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rarity_z = (rarity - col_means) / col_stds
        rarity_z = np.where(np.isfinite(rarity_z), rarity_z, np.nan)

        start_min = min_k - 1
        start_max = max(0, self.max_spells - window)
        standardized_scores = []
        for i in range(self.N):
            z_row = rarity_z[i, :]
            candidate_values = []
            for t0 in range(start_min, start_max + 1):
                vals = [z_row[t0 + j] for j in range(window)]
                if not np.all(np.isfinite(vals)):
                    continue
                candidate_values.append(float(np.max(vals)))
            standardized_scores.append(float(np.nanmin(candidate_values)) if candidate_values else np.nan)
        return standardized_scores

    def compute_converged(
        self,
        z_threshold: float = 1.5,
        min_k: int = 1,
        window: int = 1,
        inclusive: bool = False,
        group_labels: Optional[Any] = None,
        *,
        method: str = "zscore",
        proportion: Optional[float] = None,
        quantile_p: Optional[float] = None,
        min_count: int = 1,
    ) -> List[int]:
        """
        Compute binary convergence flags (0/1) per individual. Converged = low rarity (typical).

        - "zscore": converged if there exists a window where all z-scores < -z_threshold (or <= if inclusive).
        - "top_proportion": select the proportion with smallest standardized scores (most typical).
        - "quantile": converged if standardized score <= quantile_p (e.g. 0.10 = bottom 10%).
        """
        N = self.N
        start_min = min_k - 1
        start_max = max(0, self.max_spells - window)
        method_norm = (method or "zscore").lower()

        if method_norm in {"top_proportion", "topk", "proportion", "rank"}:
            p = proportion if proportion is not None else 0.10
            scores = np.asarray(
                self.compute_standardized_rarity_score(min_k=min_k, window=window), dtype=float
            )
            if group_labels is None:
                vals = scores
                finite_mask = np.isfinite(vals)
                n_valid = int(np.sum(finite_mask))
                if n_valid == 0:
                    return [0] * N
                k = int(np.floor(p * n_valid))
                if k < int(min_count):
                    k = int(min_count)
                if k > n_valid:
                    k = n_valid
                order = np.argsort(np.where(np.isfinite(vals), vals, np.inf), kind="mergesort")
                flags = np.zeros(N, dtype=int)
                if k >= 1:
                    flags[order[:k]] = 1
                return flags.tolist()
            else:
                labels = np.asarray(group_labels)
                flags = np.zeros(N, dtype=int)
                for g in pd.unique(labels):
                    idx = np.where(labels == g)[0]
                    vals = scores[idx]
                    finite_mask = np.isfinite(vals)
                    n_valid = int(np.sum(finite_mask))
                    if n_valid == 0:
                        continue
                    k = int(np.floor(p * n_valid))
                    if k < int(min_count):
                        k = int(min_count)
                    if k > n_valid:
                        k = n_valid
                    order_local = np.argsort(np.where(np.isfinite(vals), vals, np.inf), kind="mergesort")
                    if k >= 1:
                        selected_global = idx[order_local[:k]]
                        flags[selected_global] = 1
                return flags.tolist()

        if method_norm == "quantile":
            q = quantile_p if quantile_p is not None else 0.10
            scores = np.asarray(
                self.compute_standardized_rarity_score(min_k=min_k, window=window), dtype=float
            )
            flags = np.zeros(N, dtype=int)
            if group_labels is None:
                valid = scores[np.isfinite(scores)]
                if valid.size == 0:
                    return flags.tolist()
                try:
                    xq = float(np.nanquantile(scores, q))
                except Exception:
                    xq = float(np.quantile(valid, q))
                flags[scores <= xq] = 1
                return flags.tolist()
            else:
                labels = np.asarray(group_labels)
                for g in pd.unique(labels):
                    idx = np.where(labels == g)[0]
                    vals = scores[idx]
                    valid = vals[np.isfinite(vals)]
                    if valid.size == 0:
                        continue
                    try:
                        xq = float(np.nanquantile(vals, q))
                    except Exception:
                        xq = float(np.quantile(valid, q))
                    flags[idx[vals <= xq]] = 1
                return flags.tolist()

        rarity = self._build_rarity_matrix()
        col_means = np.nanmean(rarity, axis=0)
        col_stds = np.nanstd(rarity, axis=0, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rarity_z = (rarity - col_means) / col_stds
        rarity_z = np.where(np.isfinite(rarity_z), rarity_z, np.nan)

        flags = []
        for i in range(N):
            z_row = rarity_z[i, :]
            converged = 0
            for t0 in range(start_min, start_max + 1):
                vals = [z_row[t0 + j] for j in range(window)]
                if not np.all(np.isfinite(vals)):
                    continue
                if inclusive:
                    condition = all(v <= -z_threshold for v in vals)
                else:
                    condition = all(v < -z_threshold for v in vals)
                if condition:
                    converged = 1
                    break
            flags.append(converged)
        return flags

    def _compute_window_max_list(self, min_k: int, window: int) -> np.ndarray:
        """Per-individual, per starting spell level: max z in that window (for first_convergence_spell)."""
        rarity = self._build_rarity_matrix()
        col_means = np.nanmean(rarity, axis=0)
        col_stds = np.nanstd(rarity, axis=0, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rarity_z = (rarity - col_means) / col_stds
        rarity_z = np.where(np.isfinite(rarity_z), rarity_z, np.nan)

        start_min = min_k - 1
        start_max = max(0, self.max_spells - window)
        n_starts = max(0, start_max - start_min + 1)
        window_maxes = np.full((self.N, n_starts), np.nan, dtype=float)
        for i in range(self.N):
            z_row = rarity_z[i, :]
            for idx, t0 in enumerate(range(start_min, start_max + 1)):
                vals = [z_row[t0 + j] for j in range(window)]
                if np.all(np.isfinite(vals)):
                    window_maxes[i, idx] = float(np.max(vals))
        return window_maxes

    def compute_first_convergence_spell(
        self,
        z_threshold: float = 1.5,
        min_k: int = 1,
        window: int = 1,
        inclusive: bool = False,
        group_labels: Optional[Any] = None,
        *,
        method: str = "zscore",
        proportion: Optional[float] = None,
        quantile_p: Optional[float] = None,
        min_count: int = 1,
    ) -> List[Optional[int]]:
        """
        First spell level (1-indexed from end) at which the individual is converged, or None.
        Level 1 = last spell, level 2 = last two spells, etc.
        """
        N = self.N
        start_min = min_k - 1
        start_max = max(0, self.max_spells - window)
        method_norm = (method or "zscore").lower()

        if method_norm in {"top_proportion", "topk", "proportion", "rank", "quantile"}:
            agg_scores = np.asarray(
                self.compute_standardized_rarity_score(min_k=min_k, window=window), dtype=float
            )
            per_start_window_max = self._compute_window_max_list(min_k, window)
            n_starts = per_start_window_max.shape[1]

            if method_norm in {"top_proportion", "topk", "proportion", "rank"}:
                p = proportion if proportion is not None else 0.10
                if group_labels is None:
                    vals = agg_scores
                    finite_mask = np.isfinite(vals)
                    n_valid = int(np.sum(finite_mask))
                    if n_valid == 0:
                        return [None] * N
                    k = int(np.floor(p * n_valid))
                    if k < int(min_count):
                        k = int(min_count)
                    if k > n_valid:
                        k = n_valid
                    order = np.argsort(np.where(np.isfinite(vals), vals, np.inf), kind="mergesort")
                    selected_idx = set(order[:k].tolist()) if k >= 1 else set()
                    thresh_val = vals[order[k - 1]] if k >= 1 else np.nan
                    spells = []
                    for i in range(N):
                        if i not in selected_idx or not np.isfinite(thresh_val):
                            spells.append(None)
                            continue
                        wm = per_start_window_max[i, :]
                        first_spell = None
                        for t_idx in range(n_starts):
                            if np.isfinite(wm[t_idx]) and wm[t_idx] <= float(thresh_val):
                                first_spell = t_idx + min_k
                                break
                        spells.append(first_spell)
                    return spells
                else:
                    labels = np.asarray(group_labels)
                    spells = [None] * N
                    for g in pd.unique(labels):
                        idx = np.where(labels == g)[0]
                        vals = agg_scores[idx]
                        finite_mask = np.isfinite(vals)
                        n_valid = int(np.sum(finite_mask))
                        if n_valid == 0:
                            continue
                        k = int(np.floor(p * n_valid))
                        if k < int(min_count):
                            k = int(min_count)
                        if k > n_valid:
                            k = n_valid
                        order_local = np.argsort(np.where(np.isfinite(vals), vals, np.inf), kind="mergesort")
                        selected_local = set(order_local[:k].tolist()) if k >= 1 else set()
                        thresh_val = vals[order_local[k - 1]] if k >= 1 else np.nan
                        for j_local, i_global in enumerate(idx):
                            if j_local not in selected_local or not np.isfinite(thresh_val):
                                continue
                            wm = per_start_window_max[i_global, :]
                            for t_idx in range(n_starts):
                                if np.isfinite(wm[t_idx]) and wm[t_idx] <= float(thresh_val):
                                    spells[i_global] = t_idx + min_k
                                    break
                    return spells

            q = quantile_p if quantile_p is not None else 0.10
            spells = [None] * N
            n_starts = per_start_window_max.shape[1]
            if group_labels is None:
                valid = agg_scores[np.isfinite(agg_scores)]
                if valid.size == 0:
                    return spells
                try:
                    xq = float(np.nanquantile(agg_scores, q))
                except Exception:
                    xq = float(np.quantile(valid, q))
                for i in range(N):
                    if not np.isfinite(agg_scores[i]) or agg_scores[i] > xq:
                        continue
                    wm = per_start_window_max[i, :]
                    for t_idx in range(n_starts):
                        if np.isfinite(wm[t_idx]) and wm[t_idx] <= xq:
                            spells[i] = t_idx + min_k
                            break
                return spells
            else:
                labels = np.asarray(group_labels)
                for g in pd.unique(labels):
                    idx = np.where(labels == g)[0]
                    vals = agg_scores[idx]
                    valid = vals[np.isfinite(vals)]
                    if valid.size == 0:
                        continue
                    try:
                        xq = float(np.nanquantile(vals, q))
                    except Exception:
                        xq = float(np.quantile(valid, q))
                    for j_local, i_global in enumerate(idx):
                        if not np.isfinite(vals[j_local]) or vals[j_local] > xq:
                            continue
                        wm = per_start_window_max[i_global, :]
                        for t_idx in range(n_starts):
                            if np.isfinite(wm[t_idx]) and wm[t_idx] <= xq:
                                spells[i_global] = t_idx + min_k
                                break
                return spells

        rarity = self._build_rarity_matrix()
        col_means = np.nanmean(rarity, axis=0)
        col_stds = np.nanstd(rarity, axis=0, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rarity_z = (rarity - col_means) / col_stds
        rarity_z = np.where(np.isfinite(rarity_z), rarity_z, np.nan)

        spells = []
        for i in range(N):
            z_row = rarity_z[i, :]
            first_spell = None
            for t0 in range(start_min, start_max + 1):
                vals = [z_row[t0 + j] for j in range(window)]
                if not np.all(np.isfinite(vals)):
                    continue
                if inclusive:
                    condition = all(v <= -z_threshold for v in vals)
                else:
                    condition = all(v < -z_threshold for v in vals)
                if condition:
                    first_spell = t0 + 1
                    break
            spells.append(first_spell)
        return spells

    def compute_path_uniqueness(self) -> List[int]:
        """
        Per individual: count of spell levels (from end) at which the suffix is unique (freq == 1).
        """
        counts = self.tree.counts
        uniqueness = []
        for i, states_i in enumerate(self.spell_states):
            rev = list(reversed(states_i))
            count_unique = 0
            for k in range(len(rev)):
                key = tuple(rev[: k + 1])
                if counts.get(key, 0) == 1:
                    count_unique += 1
            uniqueness.append(count_unique)
        return uniqueness

    def diagnose_convergence_calculation(
        self,
        z_threshold: float = 1.5,
        min_k: int = 1,
        window: int = 1,
    ) -> Dict[str, Any]:
        """
        Diagnostic for spell-level convergence: variance per spell level, number converged, distribution.
        """
        rarity = self._build_rarity_matrix()
        rarity_df = pd.DataFrame(rarity)
        rarity_std = rarity_df.std(axis=0, ddof=1)
        levels_zero_var = [
            k + 1 for k in range(self.max_spells)
            if pd.isna(rarity_std.iloc[k]) or rarity_std.iloc[k] < 1e-10
        ]
        convergence_spells = self.compute_first_convergence_spell(
            z_threshold=z_threshold, min_k=min_k, window=window, method="zscore"
        )
        n_converged = sum(1 for s in convergence_spells if s is not None)
        spell_dist = pd.Series(convergence_spells).value_counts(dropna=False).sort_index().to_dict()
        return {
            "rarity_std_by_spell": rarity_std.tolist(),
            "spell_levels_with_zero_variance": levels_zero_var,
            "n_individuals_with_convergence": n_converged,
            "convergence_spell_distribution": spell_dist,
            "total_individuals": self.N,
            "parameters_used": {"z_threshold": z_threshold, "min_k": min_k, "window": window},
        }
