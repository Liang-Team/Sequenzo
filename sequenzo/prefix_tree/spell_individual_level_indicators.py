"""
Spell-based Prefix Tree: Individual-level divergence indicators.

Provides per-sequence (per-individual) rarity and divergence measures when the
unit of analysis is SPELL rather than time index. Each "level" is one spell;
rarity and divergence are defined along spell levels (1st spell, 2nd spell, ...).
Variable-length sequences are supported: individuals with fewer spells have NaN
at spell levels beyond their length.

Design mirrors: sequenzo/prefix_tree/individual_level_indicators.py (position-based).
- Position version: level = time index t, prefix = states up to year t.
- Spell version:    level = spell index k, prefix = states of first k spells.

Usage:
    from sequenzo.prefix_tree import build_spell_prefix_tree
    from sequenzo.prefix_tree.spell_individual_level_indicators import SpellIndividualDivergence

    tree = build_spell_prefix_tree(seqdata, expcost=0)
    ind = SpellIndividualDivergence(tree)
    rarity_per_spell = ind.compute_prefix_rarity_per_spell()
    diverged = ind.compute_diverged(method="zscore", z_threshold=1.5)

@Author  : Yuqi Liang 梁彧祺
@File    : spell_individual_level_indicators.py
@Time    : 2026/1/30
@Desc    : Individual-level indicators for spell-based prefix tree analysis.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .spell_level_indicators import SpellPrefixTree


# Small constant to avoid log(0) in rarity computation
_EPS = 1e-10


class SpellIndividualDivergence:
    """
    Individual-level divergence and rarity for spell-based prefix trees.

    Requires a SpellPrefixTree that was built with build_spell_prefix_tree(seqdata, ...),
    so that tree._spell_states and tree._spell_durations exist and tree.counts /
    tree.total_sequences are populated.
    """

    def __init__(self, tree: SpellPrefixTree):
        if not isinstance(tree, SpellPrefixTree):
            raise TypeError(
                "[!] SpellIndividualDivergence requires a SpellPrefixTree. "
                "Use: build_spell_prefix_tree(seqdata) then SpellIndividualDivergence(tree)"
            )
        if not hasattr(tree, "_spell_states") or not hasattr(tree, "_spell_durations"):
            raise ValueError(
                "[!] SpellPrefixTree must be built with build_spell_prefix_tree(seqdata) "
                "so that _spell_states and _spell_durations are attached."
            )
        self.tree = tree
        self.spell_states = tree._spell_states
        self.spell_durations = tree._spell_durations
        self.N = tree.total_sequences
        self.max_spells = max(len(s) for s in self.spell_states) if self.spell_states else 0

    def _build_rarity_matrix(self) -> np.ndarray:
        """
        Build (N, max_spells) matrix of prefix rarity at each spell level.
        rarity_{i,k} = -log( freq(prefix_{i,k}) / N ).
        Cells where individual i has no spell at level k are set to np.nan.
        """
        N, max_spells = self.N, self.max_spells
        counts = self.tree.counts
        rarity = np.full((N, max_spells), np.nan, dtype=float)
        for i, states_i in enumerate(self.spell_states):
            prefix = []
            for k, state in enumerate(states_i):
                prefix.append(state)
                key = tuple(prefix)
                freq = counts.get(key, 0) / max(N, 1)
                rarity[i, k] = -np.log(freq + _EPS)
        return rarity

    def compute_prefix_rarity_per_spell(
        self,
        as_dataframe: bool = True,
        column_prefix: str = "k",
        zscore: bool = False,
    ):
        """
        Compute per-spell-level prefix rarity for each individual.

        For each individual i and spell level k (1..max_spells),
        rarity_{i,k} = -log( freq(prefix_{i,k}) / N )
        where prefix_{i,k} is the state sequence of the first k spells for individual i,
        freq(prefix) is how many individuals share that exact spell prefix, and N is total count.
        Levels beyond an individual's spell length are NaN.

        Parameters
        ----------
        as_dataframe : bool, default True
            If True, returns a pandas DataFrame with columns "k1", "k2", ... .
            If False, returns a NumPy array of shape (N, max_spells).
        column_prefix : str, default "k"
            Column name prefix when returning a DataFrame (e.g. "k1", "k2").
        zscore : bool, default False
            If True, z-standardize rarity column-wise (by spell level) using sample std (ddof=1).
            NaN entries are ignored in mean/std per column.

        Returns
        -------
        pandas.DataFrame or np.ndarray
            Per-spell-level rarity (optionally z-scored). NaN where no spell at that level.
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

    def compute_prefix_rarity_score(self) -> List[float]:
        """
        Compute one aggregated rarity score per individual: sum of -log(freq/N) over spell levels.

        Same idea as position-based compute_prefix_rarity_score: higher = more atypical path.
        Only spell levels that exist for that individual are summed (variable length).
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
        min_k: int = 2,
        window: int = 1,
    ) -> List[float]:
        """
        Standardized rarity score per individual for divergence classification.

        Formula (aligned with position version): for each individual i,
        standardized_score_i = max over starting spell level of (min over window of z_{i,k}).
        Here z_{i,k} is the column-wise (by spell level k) z-score of rarity, with NaN
        for levels beyond the individual's length. Only windows entirely within [min_k, ...]
        and with no NaN are considered.

        Parameters
        ----------
        min_k : int, default 2
            Minimum spell level (1-indexed) to consider for divergence (same role as min_t).
        window : int, default 1
            Number of consecutive spell levels in the window.

        Returns
        -------
        List[float]
            One standardized score per individual. Higher = more atypical. NaN if no valid window.
        """
        rarity = self._build_rarity_matrix()
        col_means = np.nanmean(rarity, axis=0)
        col_stds = np.nanstd(rarity, axis=0, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rarity_z = (rarity - col_means) / col_stds
        rarity_z = np.where(np.isfinite(rarity_z), rarity_z, np.nan)

        # max_spells is 0-indexed length; valid starting indices for window: 0..max_spells-window
        # min_k is 1-indexed, so min_k-1 is the first index to consider
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
                candidate_values.append(float(np.min(vals)))
            standardized_scores.append(float(np.nanmax(candidate_values)) if candidate_values else np.nan)
        return standardized_scores

    def compute_diverged(
        self,
        z_threshold: float = 1.5,
        min_k: int = 2,
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
        Compute binary divergence flags (0/1) per individual using spell-level rarity.

        Methods (same as position version):
        - "zscore": diverged if there exists a window of length `window` starting at spell level
          in [min_k, ...] such that all z-scores of rarity at those levels are above z_threshold.
        - "top_proportion": select the top `proportion` by standardized rarity (within group or global).
        - "quantile": diverged if standardized rarity >= quantile_p within group or global.

        Parameters
        ----------
        z_threshold : float, default 1.5
            Used for method "zscore". Diverged when z > z_threshold (or >= if inclusive).
        min_k : int, default 2
            Minimum spell level (1-indexed) to consider.
        window : int, default 1
            Window length for zscore method and for standardized score.
        inclusive : bool, default False
            If True, use >= instead of > for z_threshold.
        group_labels : array-like or None
            If provided, top_proportion and quantile are applied within each group.
        method : str, default "zscore"
            One of "zscore", "top_proportion" (aliases: "topk","proportion","rank"), "quantile".
        proportion : float or None
            For top_proportion. Fraction (0,1). Default 0.10.
        quantile_p : float or None
            For quantile. Default 0.90.
        min_count : int, default 1
            Minimum number selected per group for top_proportion.

        Returns
        -------
        List[int]
            0/1 per individual (1 = diverged).
        """
        N = self.N
        start_min = min_k - 1
        start_max = max(0, self.max_spells - window)

        # --- top_proportion / quantile: use standardized scores ---
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
                order = np.argsort(np.where(np.isfinite(vals), vals, -np.inf), kind="mergesort")
                flags = np.zeros(N, dtype=int)
                if k >= 1:
                    flags[order[-k:]] = 1
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
                    order_local = np.argsort(np.where(np.isfinite(vals), vals, -np.inf), kind="mergesort")
                    if k >= 1:
                        selected_global = idx[order_local[-k:]]
                        flags[selected_global] = 1
                return flags.tolist()

        if method_norm == "quantile":
            q = quantile_p if quantile_p is not None else 0.90
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
                flags[scores >= xq] = 1
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
                    flags[idx[vals >= xq]] = 1
                return flags.tolist()

        # --- zscore: build rarity matrix, z-score by column, then window logic ---
        rarity = self._build_rarity_matrix()
        col_means = np.nanmean(rarity, axis=0)
        col_stds = np.nanstd(rarity, axis=0, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rarity_z = (rarity - col_means) / col_stds
        rarity_z = np.where(np.isfinite(rarity_z), rarity_z, np.nan)

        flags = []
        for i in range(N):
            z_row = rarity_z[i, :]
            diverged = 0
            for t0 in range(start_min, start_max + 1):
                vals = [z_row[t0 + j] for j in range(window)]
                if not np.all(np.isfinite(vals)):
                    continue
                if inclusive:
                    condition = all(v >= z_threshold for v in vals)
                else:
                    condition = all(v > z_threshold for v in vals)
                if condition:
                    diverged = 1
                    break
            flags.append(diverged)
        return flags

    def _compute_window_max_list(self, min_k: int, window: int) -> np.ndarray:
        """
        Per-individual, per starting spell level: max z in that window.
        Used by compute_first_divergence_spell for rank/quantile methods.
        """
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

    def compute_first_divergence_spell(
        self,
        z_threshold: float = 1.5,
        min_k: int = 2,
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
        First spell level (1-indexed) at which the individual is diverged, or None.

        Same methods as compute_diverged. For zscore: first starting spell level (in [min_k, ...])
        where the window of z-scores is above threshold. For top_proportion/quantile: first level
        where the window-max z-score reaches the selection threshold (for selected individuals only).
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
                    order = np.argsort(np.where(np.isfinite(vals), vals, -np.inf), kind="mergesort")
                    selected_idx = set(order[-k:].tolist()) if k >= 1 else set()
                    thresh_val = vals[order[-k]] if k >= 1 else np.nan
                    spells = []
                    for i in range(N):
                        if i not in selected_idx or not np.isfinite(thresh_val):
                            spells.append(None)
                            continue
                        wm = per_start_window_max[i, :]
                        first_spell = None
                        for t_idx in range(n_starts):
                            if np.isfinite(wm[t_idx]) and wm[t_idx] >= float(thresh_val):
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
                        order_local = np.argsort(np.where(np.isfinite(vals), vals, -np.inf), kind="mergesort")
                        selected_local = set(order_local[-k:].tolist()) if k >= 1 else set()
                        thresh_val = vals[order_local[-k]] if k >= 1 else np.nan
                        for j_local, i_global in enumerate(idx):
                            if j_local not in selected_local or not np.isfinite(thresh_val):
                                continue
                            wm = per_start_window_max[i_global, :]
                            for t_idx in range(n_starts):
                                if np.isfinite(wm[t_idx]) and wm[t_idx] >= float(thresh_val):
                                    spells[i_global] = t_idx + min_k
                                    break
                    return spells

            # quantile
            q = quantile_p if quantile_p is not None else 0.90
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
                    if not np.isfinite(agg_scores[i]) or agg_scores[i] < xq:
                        continue
                    wm = per_start_window_max[i, :]
                    for t_idx in range(n_starts):
                        if np.isfinite(wm[t_idx]) and wm[t_idx] >= xq:
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
                        if not np.isfinite(vals[j_local]) or vals[j_local] < xq:
                            continue
                        wm = per_start_window_max[i_global, :]
                        for t_idx in range(n_starts):
                            if np.isfinite(wm[t_idx]) and wm[t_idx] >= xq:
                                spells[i_global] = t_idx + min_k
                                break
                return spells

        # --- zscore: first window where all z above threshold ---
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
                    condition = all(v >= z_threshold for v in vals)
                else:
                    condition = all(v > z_threshold for v in vals)
                if condition:
                    first_spell = t0 + 1
                    break
            spells.append(first_spell)
        return spells

    def compute_path_uniqueness(self) -> List[int]:
        """
        Per individual: count of spell levels at which the spell prefix is unique (freq == 1).

        Same idea as position-based path_uniqueness: how many steps along the path
        does this individual have a prefix shared by no one else.
        """
        counts = self.tree.counts
        N = self.N
        uniqueness = []
        for i, states_i in enumerate(self.spell_states):
            prefix = []
            count_unique = 0
            for state in states_i:
                prefix.append(state)
                if counts.get(tuple(prefix), 0) == 1:
                    count_unique += 1
            uniqueness.append(count_unique)
        return uniqueness

    def diagnose_divergence_calculation(
        self,
        z_threshold: float = 1.5,
        min_k: int = 2,
        window: int = 1,
    ) -> Dict[str, Any]:
        """
        Diagnostic for spell-level divergence: variance per spell level, number diverged, distribution.

        Returns
        -------
        dict
            rarity_std_by_spell, spell_levels_with_zero_variance, n_individuals_with_divergence,
            divergence_spell_distribution, total_individuals, parameters_used.
        """
        rarity = self._build_rarity_matrix()
        rarity_df = pd.DataFrame(rarity)
        rarity_std = rarity_df.std(axis=0, ddof=1)
        levels_zero_var = [k + 1 for k in range(self.max_spells) if pd.isna(rarity_std.iloc[k]) or rarity_std.iloc[k] < 1e-10]

        divergence_spells = self.compute_first_divergence_spell(
            z_threshold=z_threshold, min_k=min_k, window=window, method="zscore"
        )
        n_diverged = sum(1 for s in divergence_spells if s is not None)
        spell_dist = pd.Series(divergence_spells).value_counts(dropna=False).sort_index().to_dict()

        return {
            "rarity_std_by_spell": rarity_std.tolist(),
            "spell_levels_with_zero_variance": levels_zero_var,
            "n_individuals_with_divergence": n_diverged,
            "divergence_spell_distribution": spell_dist,
            "total_individuals": self.N,
            "parameters_used": {"z_threshold": z_threshold, "min_k": min_k, "window": window},
        }
