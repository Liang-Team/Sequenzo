"""
Spell-based Suffix Tree: System-level convergence indicators.

Unlike position-based suffix trees (which use time-index alignment),
spell-based trees use SPELL as the unit: the k-th level = last k spells.
Same state sequence merges into the same path regardless of spell duration
(analogous to RLCPspell with expcost=0 for structure).
Duration can optionally influence derived indicators via expcost.

Design aligned with: sequenzo/dissimilarity_measures/src/LCPspellDistance.cpp
- Position RLCP : compare from last time index backward
- Spell RLCPspell: compare from last spell backward (k-th spell from end)

Usage:
    from sequenzo import SequenceData, build_suffix_tree

    seqdata = SequenceData(df, time=time_cols, id_col="id", states=states)

    # Spell-based suffix tree (convergence from last spell backward)
    tree = build_suffix_tree(seqdata, mode="spell", expcost=0)

    # expcost=0: structure ignores duration (state-only merge)
    # expcost>0: duration influences JS convergence (spell-length weighting)

@Author  : Yuqi Liang 梁彧祺
@File    : spell_level_indicators.py
@Time    : 2026/1/30 07:38
@Desc    : Spell-based suffix tree for sequence convergence analysis.
"""
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import jensenshannon

from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.utils.seqdss import seqdss
from sequenzo.dissimilarity_measures.utils.seqdur import seqdur
from sequenzo.prefix_tree.spell_level_indicators import convert_seqdata_to_spells


class SpellSuffixTree:
    """
    Suffix tree where each level = spells from the end (last 1 spell, last 2 spells, ...).

    Same as position-based SuffixTree in structure, but depth = number of spells
    from the end. Two sequences merge on the same path if they share the same
    suffix state sequence (from last spell backward), regardless of duration.
    """

    def __init__(self):
        self.root = {}
        self.counts = defaultdict(int)  # suffix (tuple of states) -> count
        self.total_sequences = 0
        self.prefix_durations = defaultdict(list)  # for optional expcost use

    def insert(self, spell_sequence: List[Any], seq_idx: int = 0, durations: Optional[List[float]] = None):
        """Insert one spell sequence in reverse order (suffix = from last spell backward)."""
        suffix = []
        node = self.root
        for k, state in enumerate(spell_sequence):
            suffix.append(state)
            key = tuple(suffix)
            self.counts[key] += 1
            if durations is not None and k < len(durations):
                self.prefix_durations[key].append((seq_idx, durations[-(k + 1) :]))  # durations from end
            if state not in node:
                node[state] = {}
            node = node[state]

    def get_suffixes_at_depth(self, depth: int) -> List[Tuple]:
        """Return all suffix tuples at spell-depth `depth` (1 = last spell, 2 = last two spells, ...)."""
        return [k for k in self.counts if len(k) == depth]

    def get_children(self, suffix) -> Dict:
        """Return immediate children of `suffix` (mapping: child_state -> subtree)."""
        node = self.root
        for state in suffix:
            node = node.get(state, {})
        return node

    def get_children_count(self, suffix) -> int:
        """Number of distinct child states (merging factor) at this suffix."""
        return len(self.get_children(suffix))

    def describe(self):
        depths = [len(k) for k in self.counts.keys()]
        max_d = max(depths) if depths else 0
        print("\n[SpellSuffixTree Overview]")
        print(f"[>] Total sequences: {self.total_sequences}")
        print(f"[>] Max spell depth: {max_d}")
        print(f"[>] Total distinct spell-suffixes: {len(self.counts)}")
        for t in range(1, max_d + 1):
            n = len(self.get_suffixes_at_depth(t))
            print(f"    Spell level {t}: {n} unique suffixes")

    def __repr__(self):
        depths = [len(k) for k in self.counts.keys()]
        return f"SpellSuffixTree(max_spell_depth={max(depths) if depths else 0}, total_suffixes={len(self.counts)})"


def build_spell_suffix_tree(
    seqdata: SequenceData,
    expcost: float = 0.0,
) -> SpellSuffixTree:
    """
    Build a spell-based suffix tree from SequenceData.

    Parameters
    ----------
    seqdata : SequenceData
        Sequence data object. Must have been created with SequenceData(...).
    expcost : float, default 0.0
        Duration weight for derived indicators (e.g. JS convergence).
        - expcost=0: pure state-based; duration not used in structure or indicators.
        - expcost>0: when computing JS convergence, weight each spell's state
          by its duration (longer spells contribute more to the distribution).

    Returns
    -------
    SpellSuffixTree
        Tree where level k = last k spells (from end backward).
    """
    spell_states, spell_durations, _ = convert_seqdata_to_spells(seqdata)

    tree = SpellSuffixTree()
    tree.total_sequences = len(spell_states)
    tree._expcost = expcost
    tree._spell_states = spell_states
    tree._spell_durations = spell_durations

    for i, (states_i, durs_i) in enumerate(zip(spell_states, spell_durations)):
        if not states_i:
            continue
        # Suffix = from last spell backward (reverse order)
        rev_states = list(reversed(states_i))
        rev_durs = list(reversed(durs_i))
        tree.insert(rev_states, seq_idx=i, durations=rev_durs if expcost != 0 else None)

    return tree


def get_depth_stats_spell(tree: SpellSuffixTree) -> Dict[str, Any]:
    """Depth-level stats for spell suffix tree (same interface as position version)."""
    depth_counts = defaultdict(int)
    depth_to_suffixes = defaultdict(list)
    for k in tree.counts:
        d = len(k)
        depth_counts[d] += 1
        depth_to_suffixes[d].append(k)
    return {"depth_counts": dict(depth_counts), "depth_to_suffixes": dict(depth_to_suffixes)}


def compute_suffix_count_spell(
    tree: SpellSuffixTree,
    max_depth: int,
    depth_stats: Optional[Dict[str, Any]] = None,
) -> List[int]:
    """Suffix counts per spell level 1..max_depth."""
    if depth_stats is None:
        depth_counts = defaultdict(int)
        for k in tree.counts:
            depth_counts[len(k)] += 1
        depth_counts = dict(depth_counts)
    else:
        depth_counts = depth_stats["depth_counts"]
    return [depth_counts.get(t, 0) for t in range(1, max_depth + 1)]


def compute_merging_factor_spell(
    tree: SpellSuffixTree,
    max_depth: int,
    depth_suffixes: Optional[Dict[int, List[Tuple]]] = None,
) -> List[float]:
    """Merging factor per spell level (same logic as position version)."""
    if depth_suffixes is None:
        depth_to_suffixes = defaultdict(list)
        for k in tree.counts:
            depth_to_suffixes[len(k)].append(k)
        depth_to_suffixes = dict(depth_to_suffixes)
    else:
        depth_to_suffixes = depth_suffixes
    result = []
    for t in range(2, max_depth + 1):
        suffixes = depth_to_suffixes.get(t - 1, [])
        if not suffixes:
            result.append(0.0)
            continue
        child_counts = [tree.get_children_count(p) for p in suffixes]
        result.append(float(np.mean(child_counts)))
    return [0.0] + result


def compute_js_convergence_spell(
    spell_states: List[List[Any]],
    spell_durations: List[List[float]],
    state_set: List[Any],
    expcost: float = 0.0,
) -> List[float]:
    """
    Jensen-Shannon convergence between consecutive spell-level distributions
    (from last spell backward).

    At each spell level k (from end), we build the state distribution.
    - expcost=0: each sequence contributes 1/N.
    - expcost>0: weight by duration of that spell (from end).
    """
    state_list = list(state_set)
    n_states = len(state_list)
    state_to_idx = {s: i for i, s in enumerate(state_list)}
    N = len(spell_states)

    max_spells = max(len(s) for s in spell_states)
    if max_spells < 2:
        return [0.0]

    distros = []
    for k in range(max_spells):
        counts = np.zeros(n_states, dtype=float)
        total = 0.0
        for i, (states_i, durs_i) in enumerate(zip(spell_states, spell_durations)):
            # k-th level from end = index len(states_i) - 1 - k
            idx_from_end = len(states_i) - 1 - k
            if idx_from_end < 0:
                continue
            s = states_i[idx_from_end]
            if s not in state_to_idx:
                continue
            j = state_to_idx[s]
            w = 1.0
            if expcost > 0 and idx_from_end < len(durs_i):
                w = float(durs_i[idx_from_end]) ** expcost
            counts[j] += w
            total += w
        if total > 0:
            distros.append(counts / total)
        else:
            distros.append(counts)

    js_scores = [0.0]
    for i in range(1, len(distros)):
        p, q = distros[i], distros[i - 1]
        if np.sum(p) <= 0 or np.sum(q) <= 0:
            js_scores.append(0.0)
        else:
            js = jensenshannon(p, q)
            js_scores.append(float(js))
    return js_scores
