"""
@Author  : Yuqi Liang 梁彧祺
@File    : individual_level_indicators.py
@Time    : 02/05/2025 11:07
@Desc    : 
"""
from collections import defaultdict, Counter
import numpy as np
import math


class IndividualDivergence:
    def __init__(self, sequences):
        self.sequences = sequences
        self.T = len(sequences[0])
        self.prefix_freq_by_year = self._build_prefix_frequencies()

    def _build_prefix_frequencies(self):
        freq_by_year = [defaultdict(int) for _ in range(self.T)]
        for seq in self.sequences:
            prefix = []
            for t in range(self.T):
                prefix.append(seq[t])
                freq_by_year[t][tuple(prefix)] += 1
        return freq_by_year

    def compute_diverged(self, delta=0.15):
        diverged_flags = []
        N = len(self.sequences)

        for seq in self.sequences:
            prefix = []
            diverged = 0
            for t in range(self.T):
                prefix.append(seq[t])
                count = self.prefix_freq_by_year[t][tuple(prefix)]
                sorted_freqs = sorted(self.prefix_freq_by_year[t].values())
                threshold = np.percentile(sorted_freqs, delta * 100)
                if count <= threshold:
                    diverged = 1
                    break
            diverged_flags.append(diverged)
        return diverged_flags

    def compute_divergence_year(self, threshold=0.01):
        divergence_years = []
        N = len(self.sequences)

        for seq in self.sequences:
            prefix = []
            year = None
            for t in range(self.T):
                prefix.append(seq[t])
                freq = self.prefix_freq_by_year[t][tuple(prefix)] / N
                if freq < threshold:
                    year = t + 1
                    break
            divergence_years.append(year)
        return divergence_years

    def compute_prefix_rarity_score(self):
        rarity_scores = []
        N = len(self.sequences)

        for seq in self.sequences:
            prefix = []
            score = 0.0
            for t in range(self.T):
                prefix.append(seq[t])
                freq = self.prefix_freq_by_year[t][tuple(prefix)] / N
                score += -math.log(freq + 1e-10)  # small constant to avoid log(0)
            rarity_scores.append(score)
        return rarity_scores

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
