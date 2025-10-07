"""
@Author  : Yuqi Liang 梁彧祺
@File    : individual_level_indicators.py
@Time    : 02/05/2025 11:07
@Desc    : 
    This module provides methods for calculating individual-level indicators in sequence data analysis.
    It includes tools to assess sequence divergence, identify divergence timing, measure prefix rarity,
    and evaluate path uniqueness for individuals or groups. 
    
    These indicators help quantify how typical or unique an individual's sequence is within a population, 
    and can be used for both overall and subgroup analyses.
"""
from collections import defaultdict, Counter
from typing import Optional
import numpy as np
import pandas as pd


class IndividualDivergence:
    def __init__(self, sequences):
        # Handle case where sequences might already be an IndividualDivergence object
        if isinstance(sequences, IndividualDivergence):
            # Extract sequences from existing object
            self.sequences = sequences.sequences
        elif hasattr(sequences, 'sequences'):
            # Handle case where input might be another object with sequences attribute
            self.sequences = sequences.sequences
        else:
            # Normal case: sequences is a list of sequences
            self.sequences = sequences
        
        # Validate input
        if not self.sequences or len(self.sequences) == 0:
            raise ValueError("sequences cannot be empty")
        if not hasattr(self.sequences[0], '__len__') and not hasattr(self.sequences[0], '__iter__'):
            raise ValueError("sequences must be a list of sequences (e.g., [[1,2,3], [2,3,1], ...])")
        
        self.T = len(self.sequences[0])
        self.prefix_freq_by_year = self._build_prefix_frequencies()

    def _build_prefix_frequencies(self):
        freq_by_year = [defaultdict(int) for _ in range(self.T)]
        for seq in self.sequences:
            prefix = []
            for t in range(self.T):
                prefix.append(seq[t])
                freq_by_year[t][tuple(prefix)] += 1
        return freq_by_year

    def compute_diverged(
        self,
        z_threshold=1.5,
        min_t=2,
        window=1,
        inclusive=False,
        group_labels=None,
        *,
        method: str = "zscore",
        proportion: Optional[float] = None,
        quantile_p: Optional[float] = None,
        min_count: int = 1,
    ):
        """
        Compute binary divergence flags with multiple selection methods.

        Definition (common intuition): higher prefix-rarity implies more atypical behavior.
        We compute per-year rarity via prefix frequencies and then detect divergence using
        one of the following methods:

        Methods
        -------
        - "zscore" (window-based, default):
          Uses per-year z-scores of rarity. A person is diverged if there exists a window
          of length `window` starting between years `[min_t, max_t]` where all z-scores are
          above `z_threshold` (use `inclusive=True` for `>=`). Zero-variance years remain
          NaN and any window containing NaN is skipped.

        - "top_proportion" (aka "topk"/"proportion"/"rank"):
          Uses the aggregated standardized score from `compute_standardized_rarity_score`
          (higher = more atypical). Select the most atypical `proportion` within each group if
          `group_labels` is provided, otherwise globally. `min_count` ensures at least the
          specified number per group.

        - "quantile":
          Uses a quantile threshold (`quantile_p`) on the aggregated standardized score,
          within each group (or globally if no `group_labels`). Individuals at or above the
          threshold are marked diverged.

        Parameters
        ----------
        z_threshold : float, default 1.5
            zscore method only. Diverged when z > z_threshold (or >= if inclusive=True).
        min_t, max_t : int
            Search interval for the starting year (1-indexed). If max_t is None, uses T - window + 1.
        window : int, default 1
            Number of consecutive years required in zscore method and used in standardized aggregation.
        inclusive : bool, default False
            zscore method only. If True, use >= comparisons.
        group_labels : array-like or None
            If provided, proportion/quantile selections are computed within each group.
        method : str, default "zscore"
            One of {"zscore", "top_proportion" (aliases: "topk","proportion","rank"), "quantile"}.
        proportion : float or None
            For top_proportion. Fraction (0,1) to select as diverged. Defaults to 0.10 if None.
        quantile_p : float or None
            For quantile. Quantile in (0,1) used as threshold. Defaults to 0.90 if None.
        min_count : int, default 1
            For top_proportion. Lower bound for number selected per group.

        Returns
        -------
        List[int]
            0/1 indicator for each individual.
        """
        N = len(self.sequences)
        method_norm = (method or "zscore").lower()
        max_t = self.T - window + 1

        # Branch: rank/quantile style selections using aggregated standardized scores
        if method_norm in {"top_proportion", "topk", "proportion", "rank"}:
            p = proportion if proportion is not None else 0.10
            scores = np.asarray(
                self.compute_standardized_rarity_score(min_t=min_t, window=window), dtype=float
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
                # For divergence: higher scores = more atypical, so take the largest k values
                order = np.argsort(np.where(np.isfinite(vals), vals, -np.inf), kind="mergesort")
                flags = np.zeros(N, dtype=int)
                if k >= 1:
                    selected = order[-k:]  # Take the k largest (most divergent)
                    flags[selected] = 1
                return flags.tolist()
            else:
                # Group-wise selection - need to implement similar to suffix_tree
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
                        selected_local = order_local[-k:]  # Take k largest within group
                        selected_global = idx[selected_local]
                        flags[selected_global] = 1
                return flags.tolist()

        if method_norm == "quantile":
            q = quantile_p if quantile_p is not None else 0.90  # High quantile for divergence
            scores = np.asarray(
                self.compute_standardized_rarity_score(min_t=min_t, window=window), dtype=float
            )
            flags = np.zeros(N, dtype=int)
            if group_labels is None:
                # Global quantile
                valid = scores[np.isfinite(scores)]
                if valid.size == 0:
                    return flags.tolist()
                try:
                    xq = float(np.nanquantile(scores, q))
                except Exception:
                    xq = float(np.quantile(valid, q))
                flags[np.where(scores >= xq)[0]] = 1
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
                    local = np.where(vals >= xq)[0]
                    flags[idx[local]] = 1
                return flags.tolist()

        # Default branch: z-score window logic
        rarity_matrix = []
        for seq in self.sequences:
            prefix = []
            score = []
            for t in range(self.T):
                prefix.append(seq[t])
                freq = self.prefix_freq_by_year[t][tuple(prefix)] / N
                score.append(-np.log(freq + 1e-10))
            rarity_matrix.append(score)

        rarity_df = pd.DataFrame(rarity_matrix)
        rarity_z = rarity_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        # Keep NaNs for zero-variance years and skip NaN windows
        rarity_z = rarity_z.replace([np.inf, -np.inf], np.nan)

        flags = []
        for i in range(N):
            z = rarity_z.iloc[i]
            diverged = 0
            for t in range(min_t - 1, max_t):
                # Skip windows containing NaN (zero-variance years)
                vals = [z.iloc[t + k] for k in range(window)]
                if not np.all(np.isfinite(vals)):
                    continue
                # 发散 = 高稀有（更不典型）
                if inclusive:
                    condition = all(v >= z_threshold for v in vals)
                else:
                    condition = all(v > z_threshold for v in vals)
                if condition:
                    diverged = 1
                    break
            flags.append(diverged)
        return flags

    def compute_first_divergence_year(
        self,
        z_threshold=1.5,
        min_t=2,
        window=1,
        inclusive=False,
        group_labels=None,
        *,
        method: str = "zscore",
        proportion: Optional[float] = None,
        quantile_p: Optional[float] = None,
        min_count: int = 1,
    ):
        """
        Compute the first divergence year per individual with multiple selection methods.

        Methods
        -------
        - "zscore" (default):
          Find the earliest starting year t in [min_t, max_t] such that all z-scores in the
          length-`window` block are above `z_threshold` (or >= if inclusive=True). Zero-variance
          years are NaN; windows containing NaN are skipped.

        - "top_proportion" (aka "topk"/"proportion"/"rank"):
          Use aggregated standardized scores to pick the most atypical `proportion` within each group
          (or globally). For the selected individuals, return the earliest t where the per-window
          max z-score is >= the selection threshold; others return None. `min_count` is respected.

        - "quantile":
          Use per-group (or global) quantile threshold `quantile_p` on aggregated standardized scores;
          individuals at or above the threshold return the earliest qualifying year; others return None.

        Parameters
        ----------
        z_threshold, min_t, window, inclusive, group_labels
            Same definitions as in `compute_diverged` for the zscore method.
        method : str, default "zscore"
            One of {"zscore", "top_proportion" (aliases: "topk","proportion","rank"), "quantile"}.
        proportion : float or None
            For top_proportion. Fraction (0,1) to select as diverged. Defaults to 0.10 if None.
        quantile_p : float or None
            For quantile. Quantile in (0,1) used as threshold. Defaults to 0.90 if None.
        min_count : int, default 1
            For top_proportion. Lower bound for number selected per group.

        Returns
        -------
        List[Optional[int]]
            First divergence years (1-indexed). None indicates no divergence.
        """
        N = len(self.sequences)
        method_norm = (method or "zscore").lower()
        max_t = self.T - window + 1

        # Helper: standardized z matrix and per-t window maxima per individual
        def _compute_window_max_list():
            # Build rarity matrix and columnwise z (global standardization)
            rarity_matrix = []
            for seq in self.sequences:
                prefix = []
                score = []
                for t in range(self.T):
                    prefix.append(seq[t])
                    freq = self.prefix_freq_by_year[t][tuple(prefix)] / N
                    score.append(-np.log(freq + 1e-10))
                rarity_matrix.append(score)
            rarity_arr = np.asarray(rarity_matrix, dtype=float)
            col_means = np.nanmean(rarity_arr, axis=0)
            col_stds = np.nanstd(rarity_arr, axis=0, ddof=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                rarity_z = (rarity_arr - col_means) / col_stds
            rarity_z = np.where(np.isfinite(rarity_z), rarity_z, np.nan)
            # Compute per-individual window maxima sequence over t
            window_maxes = []  # list of list per i
            for i in range(N):
                z_scores = rarity_z[i, :]
                vals_per_t = []
                for t0 in range(min_t - 1, max_t):
                    vals = [z_scores[t0 + k] for k in range(window)]
                    if not np.all(np.isfinite(vals)):
                        vals_per_t.append(np.nan)
                    else:
                        vals_per_t.append(float(np.max(vals)))
                window_maxes.append(vals_per_t)
            return np.asarray(window_maxes, dtype=float)

        # Branches for rank/quantile-style thresholds
        if method_norm in {"top_proportion", "topk", "proportion", "rank", "quantile"}:
            # Compute aggregated scores for thresholding
            agg_scores = np.asarray(
                self.compute_standardized_rarity_score(min_t=min_t, window=window), dtype=float
            )
            per_t_window_max = _compute_window_max_list()

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
                    # For divergence: take highest scores (most atypical)
                    order = np.argsort(np.where(np.isfinite(vals), vals, -np.inf), kind="mergesort")
                    selected_idx = set(order[-k:].tolist()) if k >= 1 else set()
                    years = []
                    for i in range(N):
                        if i not in selected_idx:
                            years.append(None)
                            continue
                        wm = per_t_window_max[i]
                        # threshold value is kth largest value
                        thresh_val = vals[order[-k]] if k >= 1 else np.nan
                        if not np.isfinite(thresh_val):
                            years.append(None)
                            continue
                        # earliest t where window_max >= threshold
                        yr = None
                        for t_idx, wv in enumerate(wm):
                            if np.isfinite(wv) and wv >= float(thresh_val):
                                yr = t_idx + min_t  # Convert back to 1-indexed
                                break
                        years.append(yr)
                    return years
                else:
                    labels = np.asarray(group_labels)
                    years = [None] * N
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
                            wm = per_t_window_max[i_global]
                            for t_idx, wv in enumerate(wm):
                                if np.isfinite(wv) and wv >= float(thresh_val):
                                    years[i_global] = int(t_idx + min_t)
                                    break
                    return years

            # quantile branch
            q = quantile_p if quantile_p is not None else 0.90
            years = [None] * N
            if group_labels is None:
                valid = agg_scores[np.isfinite(agg_scores)]
                if valid.size == 0:
                    return years
                try:
                    xq = float(np.nanquantile(agg_scores, q))
                except Exception:
                    xq = float(np.quantile(valid, q))
                for i in range(N):
                    if not np.isfinite(agg_scores[i]) or agg_scores[i] < xq:
                        continue
                    wm = per_t_window_max[i]
                    for t_idx, wv in enumerate(wm):
                        if np.isfinite(wv) and wv >= xq:
                            years[i] = int(t_idx + min_t)
                            break
                return years
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
                        wm = per_t_window_max[i_global]
                        for t_idx, wv in enumerate(wm):
                            if np.isfinite(wv) and wv >= xq:
                                years[i_global] = t_idx + min_t
                                break
                return years

        # Default branch: z-score window logic
        rarity_matrix = []
        for seq in self.sequences:
            prefix = []
            score = []
            for t in range(self.T):
                prefix.append(seq[t])
                freq = self.prefix_freq_by_year[t][tuple(prefix)] / N
                score.append(-np.log(freq + 1e-10))
            rarity_matrix.append(score)

        rarity_df = pd.DataFrame(rarity_matrix)
        rarity_z = rarity_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        # Keep NaNs for zero-variance years and skip NaN windows
        rarity_z = rarity_z.replace([np.inf, -np.inf], np.nan)

        years = []
        for i in range(N):
            z = rarity_z.iloc[i]
            year = None
            for t in range(min_t - 1, max_t):
                # Skip windows containing NaN (zero-variance years)
                vals = [z.iloc[t + k] for k in range(window)]
                if not np.all(np.isfinite(vals)):
                    continue
                # 发散 = 高稀有（更不典型）
                if inclusive:
                    condition = all(v >= z_threshold for v in vals)
                else:
                    condition = all(v > z_threshold for v in vals)
                if condition:
                    year = int(t + 1)  # Convert to 1-indexed integer
                    break
            years.append(year)
        return years

    def compute_prefix_rarity_per_year(self, as_dataframe: bool = True, column_prefix: str = "t", zscore: bool = False):
        """
        Compute per-year prefix rarity scores for each individual.

        For each individual i and year t (1..T), rarity score is defined as:
            rarity_{i,t} = -log( freq(prefix_{i,t}) / N )
        where prefix_{i,t} is the sequence of observed states up to year t for individual i,
        freq(prefix) counts how many individuals share that exact prefix up to year t,
        and N is the total number of individuals.

        Parameters
        ----------
        as_dataframe : bool, default True
            If True, returns a pandas DataFrame with columns f"{column_prefix}1"..f"{column_prefix}T".
            If False, returns a NumPy array of shape (N, T).
        column_prefix : str, default "t"
            Column name prefix when returning a DataFrame.
        zscore : bool, default False
            If True, z-standardize the rarity scores column-wise (by year) using 
            sample standard deviation (ddof=1).

        Returns
        -------
        pandas.DataFrame or np.ndarray
            Per-year rarity scores (optionally z-scored).
        """
        N = len(self.sequences)
        rarity_matrix = []

        for seq in self.sequences:
            prefix = []
            score_list = []
            for t in range(self.T):
                prefix.append(seq[t])
                freq = self.prefix_freq_by_year[t][tuple(prefix)] / N
                score_list.append(-np.log(freq + 1e-10))
            rarity_matrix.append(score_list)

        rarity_arr = np.array(rarity_matrix, dtype=float)

        if zscore:
            # Column-wise z-score; handle zero-std columns gracefully (leave as NaN)
            col_means = np.nanmean(rarity_arr, axis=0)
            col_stds = np.nanstd(rarity_arr, axis=0, ddof=1)  # Use sample std for consistency with pandas
            with np.errstate(invalid='ignore', divide='ignore'):
                rarity_arr = (rarity_arr - col_means) / col_stds

        if not as_dataframe:
            return rarity_arr

        columns = [f"{column_prefix}{t+1}" for t in range(self.T)]
        return pd.DataFrame(rarity_arr, columns=columns)

    def compute_prefix_rarity_score(self):
        rarity_scores = []
        N = len(self.sequences)

        for seq in self.sequences:
            prefix = []
            score = 0.0
            for t in range(self.T):
                prefix.append(seq[t])
                freq = self.prefix_freq_by_year[t][tuple(prefix)] / N
                score += -np.log(freq + 1e-10)  # small constant to avoid log(0)
            rarity_scores.append(score)
        return rarity_scores

    def diagnose_divergence_calculation(self, z_threshold=1.5, min_t=2, window=1):
        """
        Diagnostic function to analyze divergence year calculation and identify 
        years with insufficient variance (std ≈ 0) that cannot trigger divergence.
        
        This is methodologically appropriate: when all individuals follow similar 
        trajectories in a given year, no divergence should be detected.
        
        Returns:
        --------
        dict: Diagnostic information including:
            - years_with_zero_variance: List of years where std ≈ 0
            - rarity_std_by_year: Standard deviation of rarity scores per year
            - n_individuals_with_divergence: Count of individuals with any divergence
            - divergence_year_distribution: Value counts of divergence years
        """
        N = len(self.sequences)
        rarity_matrix = []

        # Calculate rarity scores (same as in compute_first_divergence_year)
        for seq in self.sequences:
            prefix = []
            score = []
            for t in range(self.T):
                prefix.append(seq[t])
                freq = self.prefix_freq_by_year[t][tuple(prefix)] / N
                score.append(-np.log(freq + 1e-10))
            rarity_matrix.append(score)

        rarity_df = pd.DataFrame(rarity_matrix)
        
        # Calculate standard deviations by year
        rarity_std_by_year = rarity_df.std(axis=0)
        years_with_zero_variance = []
        
        # Identify years with near-zero variance (threshold can be adjusted)
        for t, std_val in enumerate(rarity_std_by_year):
            if pd.isna(std_val) or std_val < 1e-10:
                years_with_zero_variance.append(t + 1)  # 1-indexed
        
        # Calculate z-scores
        rarity_z = rarity_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        
        # Count individuals with divergence
        divergence_years = self.compute_first_divergence_year(z_threshold, min_t, window)
        n_individuals_with_divergence = sum(1 for year in divergence_years if year is not None)
        
        # Distribution of divergence years
        divergence_year_counts = pd.Series(divergence_years).value_counts(dropna=False).sort_index()
        
        return {
            'years_with_zero_variance': years_with_zero_variance,
            'rarity_std_by_year': rarity_std_by_year.tolist(),
            'n_individuals_with_divergence': n_individuals_with_divergence,
            'divergence_year_distribution': divergence_year_counts.to_dict(),
            'total_individuals': N,
            'parameters_used': {
                'z_threshold': z_threshold,
                'min_t': min_t, 
                'window': window
            }
        }

    def compute_standardized_rarity_score(self, min_t=2, window=1):
        """
        Compute standardized rarity scores for divergence classification and visualization
        using true statistical z-scores.
        
        This method computes standardized rarity scores used for individual-level 
        divergence classification:
        standardized_score_i = max_t min_{k=0..window-1} z_{i,t+k}
        
        Where z_{i,t} are the year-wise standardized prefix rarity scores using column-wise 
        standardization with sample standard deviation (ddof=1, as computed by pandas).
        
        The standardized scores can be used with a threshold (e.g., z >= 1.5) to classify 
        individuals as diverged/not diverged, and are particularly useful for visualization.
        
        Parameters:
        -----------
        min_t : int, default=2
            Minimum year (1-indexed) after which divergence is considered valid
        window : int, default=1
            Number of consecutive high-z years required
            
        Returns:
        --------
        List[float]
            Standardized rarity scores for each individual. Values >= z_threshold indicate divergence.
            
        Notes:
        ------
        The standardization uses sample standard deviation (ddof=1) for each year column,
        which is consistent with pandas' default behavior for DataFrame.std().
        This is essentially the z-score normalized version of prefix rarity scores.
        """
        N = len(self.sequences)
        # Step 1: Compute rarity matrix (same as in compute_diverged)
        rarity_matrix = []
        
        for seq in self.sequences:
            prefix = []
            score = []
            for t in range(self.T):
                prefix.append(seq[t])
                freq = self.prefix_freq_by_year[t][tuple(prefix)] / N
                score.append(-np.log(freq + 1e-10))
            rarity_matrix.append(score)
        
        # Step 2: Column-wise true z-score standardization (by year, ddof=1)
        rarity_arr = np.asarray(rarity_matrix, dtype=float)
        col_means = np.nanmean(rarity_arr, axis=0)
        col_stds = np.nanstd(rarity_arr, axis=0, ddof=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            rarity_z = (rarity_arr - col_means) / col_stds
        # Keep NaN for zero-variance years to allow window skipping downstream
        rarity_z = np.where(np.isfinite(rarity_z), rarity_z, np.nan)
        
        # Step 3: Compute standardized rarity score for each individual
        standardized_scores = []
        for i in range(N):
            z_scores = rarity_z[i, :]
            candidate_values = []
            
            # For each possible starting time t
            for t in range(min_t - 1, self.T - window + 1):
                vals = [z_scores[t + k] for k in range(window)]
                # Skip windows containing NaN (e.g., zero-variance years)
                if not np.all(np.isfinite(vals)):
                    continue
                # For divergence, take minimum within window (ensure all finite)
                window_min = float(np.min(vals))
                candidate_values.append(window_min)
            
            # Take the maximum across all starting times
            if candidate_values:
                standardized_score = np.nanmax(candidate_values)
            else:
                standardized_score = np.nan
                
            standardized_scores.append(standardized_score)
        
        return standardized_scores

    def compute_path_uniqueness(self):
        uniqueness_scores = []
        for seq in self.sequences:
            prefix = []
            count = 0
            for t in range(self.T):
                prefix.append(seq[t])
                if self.prefix_freq_by_year[t][tuple(prefix)] == 1:
                    count += 1
            uniqueness_scores.append(count)
        return uniqueness_scores


def plot_prefix_rarity_distribution(
    data,
    # === Core Parameters ===
    group_names=None,
    colors=None,
    # === Threshold Settings ===
    show_threshold=True,
    threshold_method="zscore",  # Changed default to top_proportion
    proportion_p=0.07,  # Simplified parameter name, default 7%
    # === Plotting Options ===
    figsize=(10, 6),
    kde_bw=None,
    # === Export Options ===
    save_as=None,
    dpi=300,
    show=True,
    # === Parameters for Different Methods ===
    z_threshold=1.5,
    is_standardized_score=False,
    quantile_p=0.90
):
    """
    Plot prefix rarity score distribution(s) with clean threshold lines.
    
    Parameters
    ----------
    data : dict or array-like
        Data to plot. If dict: {"group1": scores1, "group2": scores2}
        If array-like: single group data
    group_names : list, optional
        Custom group names. Auto-detected from dict keys if not provided
    colors : dict or list, optional
        Colors for groups. If None, uses default palette
    
    show_threshold : bool, default True
        Whether to show threshold vertical lines
    threshold_method : str, default "top_proportion"
        Threshold method:
        - "top_proportion": Select top proportion_p% most extreme values
        - "quantile": Use quantile_p percentile 
        - "zscore": Use z-score threshold (for standardized data)
    proportion_p : float, default 0.05
        Proportion for top_proportion method (e.g., 0.05 = top 5%)
    
    figsize : tuple, default (10, 6)
        Figure size (width, height)
    kde_bw : float, optional
        KDE bandwidth adjustment. If None, uses seaborn default
    
    save_as : str, optional
        Save path (without extension)
    dpi : int, default 300
        Resolution for saved figure
    show : bool, default True
        Whether to display plot
        
    Returns
    -------
    dict
        Statistics including threshold values per group
    
    Examples
    --------
    # Basic usage - top 5% threshold (default)
    >>> plot_prefix_rarity_distribution({"India": india_scores, "US": us_scores})
    
    # Custom threshold proportion  
    >>> plot_prefix_rarity_distribution(
    ...     data={"India": india_scores, "US": us_scores},
    ...     proportion_p=0.03,  # top 3%
    ...     save_as="rarity_comparison"
    ... )
    
    # Quantile-based threshold
    >>> plot_prefix_rarity_distribution(
    ...     data={"India": india_scores, "US": us_scores},
    ...     threshold_method="quantile",
    ...     quantile_p=0.90,  # 90th percentile
    ... )
    
    # Clean plot without thresholds
    >>> plot_prefix_rarity_distribution(
    ...     data, 
    ...     show_threshold=False,
    ...     colors={"India": "#E8B88A", "US": "#A3BFD9"}
    ... )
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Process input data
    if isinstance(data, dict):
        # Multi-group case
        groups = data
        if group_names is None:
            group_names = list(groups.keys())
    else:
        # Single group case
        if group_names is None:
            group_names = ["Group"]
        groups = {group_names[0]: data}
    
    # Set up colors (simplified)
    if colors is None:
        default_colors = ["#A3BFD9", "#E8B88A", "#C6A5CF", "#A6C1A9", "#F4A460", "#87CEEB"]
        color_map = dict(zip(group_names, default_colors[:len(group_names)]))
    elif isinstance(colors, dict):
        color_map = colors
    else:
        color_map = dict(zip(group_names, colors))
    
    # Normalize method and prepare stats
    threshold_method = (threshold_method or "top_proportion").lower()
    
    # Handle legacy parameter mapping
    if threshold_method in {"top_proportion", "topk", "proportion", "rank"}:
        # Use the simplified proportion_p parameter
        top_proportion_p = proportion_p
        topk_min_count = 1
    elif threshold_method == "quantile":
        # Use quantile_p for quantile method
        pass
    elif threshold_method in {"zscore", "z"} and is_standardized_score:
        # Auto-handle standardized scores
        pass
    
    stats = {"per_group": {}, "threshold_method": threshold_method}

    # Validate quantiles if needed
    def _check_q(q: float):
        if not (0 < float(q) < 1):
            raise ValueError(f"quantile must be in (0,1), got {q}")
    quantiles_to_draw = None
    if threshold_method == "quantile":
        _check_q(quantile_p)
        quantiles_to_draw = [quantile_p]  # Simplified - no additional_quantiles
        # Per-group quantile(s)
        for g in group_names:
            if g in groups:
                arr = np.asarray(groups[g], dtype=float)
                # Compute requested quantiles with NaN handling
                valid = arr[~np.isnan(arr)]
                thresholds_g = {}
                if valid.size > 0:
                    for q in quantiles_to_draw:
                        try:
                            xq = float(np.nanquantile(arr, q))
                        except Exception:
                            xq = float(np.quantile(valid, q))
                        thresholds_g[f"p{int(round(q*100)):02d}"] = xq
                else:
                    for q in quantiles_to_draw:
                        thresholds_g[f"p{int(round(q*100)):02d}"] = np.nan
                # Primary threshold (for backward compatibility)
                primary_label = f"p{int(round(quantile_p*100)):02d}"
                primary_value = thresholds_g.get(primary_label, np.nan)
                # Proportion below primary
                vals = valid
                prop_below = float(np.nanmean(vals <= primary_value)) if vals.size > 0 and not np.isnan(primary_value) else np.nan
                stats["per_group"][g] = {
                    "threshold_values": thresholds_g,
                    "is_group_relative": True,
                    "threshold_value": primary_value,
                    "primary_quantile": primary_label,
                    "prop_below": prop_below
                }
    elif threshold_method in {"zscore", "z"}:
        # z-score method (backward compatibility)
        for g in group_names:
            if g in groups:
                arr = np.asarray(groups[g], dtype=float)
                mean_g = np.nanmean(arr)
                std_g = np.nanstd(arr, ddof=1)  # sample std to match pandas
                if is_standardized_score:
                    x_thresh_g = float(z_threshold)
                else:
                    # For prefix (divergence): high scores indicate divergence, so mean + z*std
                    x_thresh_g = float(mean_g + z_threshold * std_g)
                vals = arr[~np.isnan(arr)]
                prop_above = float(np.nanmean(vals >= x_thresh_g)) if vals.size > 0 and not np.isnan(x_thresh_g) else np.nan
                stats["per_group"][g] = {
                    "mean": float(mean_g),
                    "std": float(std_g),
                    "threshold_value": float(x_thresh_g),
                    "z_threshold": float(z_threshold),
                    "is_group_relative": True,
                    "prop_above": prop_above,
                    "num_above": int(np.sum(vals >= x_thresh_g)) if vals.size > 0 and not np.isnan(x_thresh_g) else 0,
                    "n": int(vals.size)
                }
    elif threshold_method in {"topk", "top_proportion", "proportion", "rank"}:
        # Rank-based proportion selection within each group: pick top p% (highest values for prefix divergence)
        if not (0 < float(proportion_p) < 1):
            raise ValueError(f"proportion_p must be in (0,1), got {proportion_p}")
        top_proportion_p = proportion_p  # Map to internal variable
        for g in group_names:
            if g in groups:
                arr = np.asarray(groups[g], dtype=float)
                finite_mask = np.isfinite(arr)
                vals = arr[finite_mask]
                n_valid = int(vals.size)
                if n_valid == 0:
                    stats["per_group"][g] = {
                        "threshold_value": np.nan,
                        "k": 0,
                        "n": 0,
                        "prop_selected": np.nan,
                        "num_geq_threshold": 0
                    }
                    continue
                k = int(np.floor(top_proportion_p * n_valid))
                if k < int(topk_min_count):
                    k = int(topk_min_count)
                if k > n_valid:
                    k = n_valid
                # Sort descending (most divergent first for prefix)
                order = np.argsort(vals, kind="mergesort")[::-1]
                thresh_val = vals[order[k - 1]] if k >= 1 else np.nan
                num_geq = int(np.sum(vals >= thresh_val)) if k >= 1 and np.isfinite(thresh_val) else 0
                stats["per_group"][g] = {
                    "threshold_value": float(thresh_val) if np.isfinite(thresh_val) else np.nan,
                    "k": int(k),
                    "n": int(n_valid),
                    "prop_selected": (k / n_valid) if n_valid > 0 else np.nan,
                    "num_geq_threshold": num_geq
                }
        stats["threshold_method"] = "topk"
    else:
        raise ValueError(f"Unknown threshold_method: {threshold_method}")
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot distributions
    for idx, group_name in enumerate(group_names):
        if group_name in groups:
            scores = groups[group_name]
            color = color_map.get(group_name, "#1f77b4")
            arr = np.asarray(scores, dtype=float)
            vmin = np.nanmin(arr) if np.isfinite(arr).any() else None
            vmax = np.nanmax(arr) if np.isfinite(arr).any() else None
            kde_kwargs = {"label": group_name, "fill": True, "color": color, "linewidth": 2}
            if kde_bw is not None:
                kde_kwargs["bw_adjust"] = kde_bw
            if vmin is not None and vmax is not None and vmin < vmax:
                kde_kwargs["clip"] = (vmin, vmax)
            sns.kdeplot(arr, **kde_kwargs)
    
    # Add per-group threshold lines if requested (color-matched)
    if show_threshold:
        for i, g in enumerate(group_names):
            if g in stats["per_group"]:
                color = color_map.get(g, "#1f77b4")
                ax = plt.gca()
                y_max = ax.get_ylim()[1]
                x_min, x_max = ax.get_xlim()
                text_y = y_max * 0.9
                x_offset = (x_max - x_min) * 0.005 * (i + 1)
                if threshold_method == "quantile":
                    thresholds_g = stats["per_group"][g]["threshold_values"]
                    # Draw multiple lines if multiple quantiles
                    for k_idx, (q_lbl, xg) in enumerate(sorted(thresholds_g.items())):
                        if np.isnan(xg):
                            continue
                        # Clean threshold line without text label
                        plt.axvline(xg, color=color, linestyle="--", linewidth=1.6)
                elif threshold_method in {"zscore", "z"}:
                    xg = stats["per_group"][g]["threshold_value"]
                    # Clean threshold line without text label
                    plt.axvline(xg, color=color, linestyle="--", linewidth=1.6)
                else:  # top_proportion
                    xg = stats["per_group"][g]["threshold_value"]
                    if np.isfinite(xg):
                        # Clean threshold line without text label
                        plt.axvline(xg, color=color, linestyle="--", linewidth=1.6)
    
    # Formatting
    if is_standardized_score:
        plt.xlabel("Standardized Prefix Rarity Score", fontsize=13)
    else:
        plt.xlabel("Prefix Rarity Score", fontsize=13)
    plt.ylabel("Density", fontsize=13)
    if len(group_names) > 1:
        plt.legend(title="Country")
    sns.despine()
    plt.tight_layout()
    
    # Save and show
    if save_as:
        plt.savefig(f"{save_as}.png", dpi=dpi, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return stats


def plot_individual_indicators_correlation(
    df,
    indicator_columns=None,
    correlation_method='pearson',
    group_column=None,
    figsize=(10, 8),
    cmap='RdBu_r',
    center=0,
    annot=True,
    fmt='.2f',
    save_as=None,
    dpi=300,
    show=True
):
    """
    Plot correlation heatmap of individual-level indicators with beautiful styling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing individual-level indicators
    indicator_columns : list, optional
        List of column names to include in correlation analysis.
        If None, automatically detects indicator columns (diverged, divergence_year, 
        prefix_rarity_score, path_uniqueness, etc.)
    correlation_method : str, default='pearson'
        Correlation method: 'pearson', 'spearman', 'kendall'
    group_column : str, optional
        Column name for grouping (e.g., 'country'). If provided, shows separate 
        heatmaps for each group
    figsize : tuple, default=(10, 8)
        Figure size (width, height)
    cmap : str, default='RdBu_r'
        Colormap for heatmap. Options: 'RdBu_r', 'coolwarm', 'viridis', 'plasma'
    center : float, default=0
        Value to center the colormap at
    annot : bool, default=True
        Whether to annotate cells with correlation values
    fmt : str, default='.2f'
        Format for annotations
    save_as : str, optional
        Path to save the figure (without extension)
    dpi : int, default=300
        DPI for saving
    show : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    dict: Correlation matrix/matrices and statistics
    
    Example:
    --------
    # Basic usage
    >>> plot_individual_indicators_correlation(df)
    
    # Custom indicators with grouping
    >>> plot_individual_indicators_correlation(
    ...     df, 
    ...     indicator_columns=['diverged', 'prefix_rarity_score', 'path_uniqueness'],
    ...     group_column='country',
    ...     correlation_method='spearman'
    ... )
    
    # Custom styling
    >>> plot_individual_indicators_correlation(
    ...     df,
    ...     cmap='plasma',
    ...     figsize=(12, 10),
    ...     save_as="indicators_correlation_heatmap"
    ... )
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    # Auto-detect indicator columns if not provided
    if indicator_columns is None:
        # Common individual-level indicator patterns
        potential_indicators = [
            'diverged', 'first_divergence_year', 'divergence_year',
            'prefix_rarity_score', 'path_uniqueness',
            'rarity_score', 'uniqueness_score'
        ]
        indicator_columns = [col for col in df.columns if col in potential_indicators]
        
        # Also include numeric columns that might be indicators
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in indicator_columns and any(
                keyword in col.lower() for keyword in 
                ['score', 'index', 'count', 'factor', 'rate', 'ratio']
            ):
                indicator_columns.append(col)
    
    # Filter and clean data
    df_indicators = df[indicator_columns].copy()
    
    # Handle missing values and convert data types
    for col in df_indicators.columns:
        if df_indicators[col].dtype == 'object':
            # Try to convert to numeric
            df_indicators[col] = pd.to_numeric(df_indicators[col], errors='coerce')
    
    # Remove columns with too many missing values (>50%)
    valid_cols = []
    for col in df_indicators.columns:
        if df_indicators[col].notna().sum() / len(df_indicators) > 0.5:
            valid_cols.append(col)
    
    df_indicators = df_indicators[valid_cols]
    
    # Drop rows with any missing values for correlation calculation
    df_clean = df_indicators.dropna()
    
    if len(df_clean) == 0:
        raise ValueError("No valid data remaining after cleaning. Check for missing values.")
    
    # Calculate correlations
    results = {}
    
    if group_column is None or group_column not in df.columns:
        # Single correlation matrix
        corr_matrix = df_clean.corr(method=correlation_method)
        results['overall'] = corr_matrix
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Create mask for upper triangle (optional - makes it cleaner)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            center=center,
            square=True,
            cbar_kws={"shrink": .8, "label": f"{correlation_method.title()} Correlation"},
            linewidths=0.5
        )
        
        plt.title(f"Individual-Level Indicators Correlation Heatmap\n({correlation_method.title()} Correlation)", 
                 fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
    else:
        # Group-based correlation matrices
        groups = df[group_column].unique()
        n_groups = len(groups)
        
        # Calculate subplot layout
        if n_groups <= 2:
            nrows, ncols = 1, n_groups
            figsize = (figsize[0] * n_groups, figsize[1])
        else:
            ncols = min(3, n_groups)
            nrows = (n_groups + ncols - 1) // ncols
            figsize = (figsize[0] * ncols, figsize[1] * nrows)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_groups == 1:
            axes = [axes]
        elif nrows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, group in enumerate(groups):
            group_data = df[df[group_column] == group][indicator_columns].dropna()
            
            if len(group_data) < 2:
                print(f"Warning: Group '{group}' has insufficient data for correlation")
                continue
                
            corr_matrix = group_data.corr(method=correlation_method)
            results[group] = corr_matrix
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Plot heatmap
            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=annot,
                fmt=fmt,
                cmap=cmap,
                center=center,
                square=True,
                cbar=i == 0,  # Only show colorbar for first subplot
                cbar_kws={"shrink": .8, "label": f"{correlation_method.title()} Correlation"} if i == 0 else {},
                linewidths=0.5,
                ax=axes[i]
            )
            
            axes[i].set_title(f"{group}\n({len(group_data)} individuals)", fontsize=12)
            axes[i].set_xticks(axes[i].get_xticks())
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
            axes[i].set_yticks(axes[i].get_yticks())
            axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=0)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle(f"Individual-Level Indicators Correlation by {group_column.title()}\n({correlation_method.title()} Correlation)", 
                    fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save and show
    if save_as:
        plt.savefig(f"{save_as}.png", dpi=dpi, bbox_inches='tight')
    
    if show:
        plt.show()
    
    # Add summary statistics
    if group_column is None:
        sample_size = len(df_clean)
    else:
        sizes = {}
        for g in df[group_column].unique():
            g_clean = df[df[group_column]==g][indicator_columns].apply(pd.to_numeric, errors='coerce').dropna()
            sizes[g] = len(g_clean)
        sample_size = sizes

    results['summary'] = {
        'method': correlation_method,
        'n_indicators': len(valid_cols),
        'indicators_included': valid_cols,
        'sample_size': sample_size
    }
    
    return results


def compute_path_uniqueness_by_group_prefix(sequences, group_labels):
        """
        Compute path uniqueness within each subgroup defined by group_labels using prefix-based approach.
        This is consistent with the divergence module's prefix-based logic.
        :param sequences: List of sequences.
        :param group_labels: List of group keys (same length as sequences), e.g., country, gender.
        :return: List of path uniqueness scores (same order as input).
        """
        from collections import defaultdict

        T = len(sequences[0])
        df = pd.DataFrame({
            "sequence": sequences,
            "group": group_labels
        })

        # Step 1: Precompute prefix frequency tables per group
        group_prefix_freq = {}
        for group, group_df in df.groupby("group"):
            prefix_freq = [defaultdict(int) for _ in range(T)]
            for seq in group_df["sequence"]:
                prefix = []
                for t in range(T):
                    prefix.append(seq[t])
                    prefix_freq[t][tuple(prefix)] += 1
            group_prefix_freq[group] = prefix_freq

        # Step 2: Compute path uniqueness per individual
        uniqueness_scores = []
        for seq, group in zip(sequences, group_labels):
            prefix_freq = group_prefix_freq[group]
            prefix = []
            count = 0
            for t in range(T):
                prefix.append(seq[t])
                if prefix_freq[t][tuple(prefix)] == 1:
                    count += 1
            uniqueness_scores.append(count)

        return uniqueness_scores
