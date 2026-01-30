"""
Spell-based Prefix Tree: System-level divergence indicators.

Unlike position-based prefix trees (which use time-index alignment),
spell-based trees use SPELL as the unit: the k-th level = after the k-th spell.
Same state sequence merges into the same path regardless of spell duration
(analogous to LCPspell with expcost=0 for structure).
Duration can optionally influence derived indicators via expcost.

Design aligned with: sequenzo/dissimilarity_measures/src/LCPspellDistance.cpp
- Position LCP  : compare state at same time index (t)
- Spell LCPspell: compare k-th spell of A with k-th spell of B

Usage:
    from sequenzo import SequenceData, build_prefix_tree

    seqdata = SequenceData(df, time=time_cols, id_col="id", states=states)

    # Spell-based prefix tree (divergence from first spell onward)
    tree = build_prefix_tree(seqdata, mode="spell", expcost=0)

    # expcost=0: structure ignores duration (state-only merge)
    # expcost>0: duration influences JS divergence (spell-length weighting)

@Author  : Yuqi Liang 梁彧祺
@File    : spell_level_indicators.py
@Time    : 2026/1/30 08:47
@Desc    : Spell-based prefix tree for sequence divergence analysis.
"""
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import jensenshannon

from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.utils.seqdss import seqdss
from sequenzo.dissimilarity_measures.utils.seqdur import seqdur


def convert_seqdata_to_spells(
    seqdata: SequenceData,
) -> Tuple[List[List[Any]], List[List[float]], List[Any]]:
    """
    Convert SequenceData to spell representation (DSS + duration).

    A "spell" is a maximal run of consecutive same-state positions.
    Aligned with seqdss/seqdur used by LCPspell and OMspell distances.

    Returns
    -------
    spell_states : list of list
        Each inner list = [s1, s2, ...] state labels for spells 1, 2, ...
    spell_durations : list of list
        Each inner list = [d1, d2, ...] duration (time points) per spell
    state_list : list
        Ordered state labels (for consistent mapping)
    """
    if not isinstance(seqdata, SequenceData):
        raise TypeError(
            "[!] Spell mode requires SequenceData. "
            "Use: SequenceData(df, time=..., id_col=..., states=...) "
            "Then: build_prefix_tree(seqdata, mode='spell')"
        )

    dss = seqdss(seqdata)  # (n, max_spells), int32, -999 padding
    dur = seqdur(seqdata)  # (n, max_spells), float duration per spell

    state_list = list(seqdata.states)
    n_seq = dss.shape[0]
    max_spells = dss.shape[1]

    spell_states = []
    spell_durations = []

    for i in range(n_seq):
        states_i = []
        durs_i = []
        for j in range(max_spells):
            val = int(dss[i, j])
            if val < 0:  # padding (-999)
                break
            if val < len(state_list):
                states_i.append(state_list[val])
            else:
                states_i.append(val)  # fallback: use raw value
            durs_i.append(float(dur[i, j]))
        spell_states.append(states_i)
        spell_durations.append(durs_i)

    return spell_states, spell_durations, state_list


class SpellPrefixTree:
    """
    Prefix tree where each level = one spell (1st spell, 2nd spell, ...).

    Same as position-based PrefixTree in structure, but depth = number of spells
    instead of number of time points. Two sequences merge on the same path if
    they share the same state sequence (spell order), regardless of duration.
    """

    def __init__(self):
        self.root = {}
        self.counts = defaultdict(int)  # prefix (tuple of states) -> count
        self.total_sequences = 0
        # Optional: store duration info per prefix for expcost-weighted indicators
        self.prefix_durations = defaultdict(list)  # prefix -> list of (seq_idx, durations)

    def insert(self, spell_sequence: List[Any], seq_idx: int = 0, durations: Optional[List[float]] = None):
        """Insert one spell sequence. States only determine structure; durations stored for optional use."""
        prefix = []
        node = self.root
        for k, state in enumerate(spell_sequence):
            prefix.append(state)
            key = tuple(prefix)
            self.counts[key] += 1
            if durations is not None and k < len(durations):
                self.prefix_durations[key].append((seq_idx, durations[: k + 1]))
            if state not in node:
                node[state] = {}
            node = node[state]

    def get_prefixes_at_depth(self, depth: int) -> List[Tuple]:
        """Return all prefix tuples at spell-depth `depth` (1 = first spell, 2 = first two spells, ...)."""
        return [k for k in self.counts if len(k) == depth]

    def get_children(self, prefix) -> Dict:
        """Return immediate children of `prefix` (mapping: child_state -> subtree)."""
        node = self.root
        for state in prefix:
            node = node.get(state, {})
        return node

    def get_children_count(self, prefix) -> int:
        """Number of distinct child states (branching) at this prefix."""
        return len(self.get_children(prefix))

    def describe(self):
        depths = [len(k) for k in self.counts.keys()]
        max_d = max(depths) if depths else 0
        print("\n[SpellPrefixTree Overview]")
        print(f"[>] Total sequences: {self.total_sequences}")
        print(f"[>] Max spell depth: {max_d}")
        print(f"[>] Total distinct spell-prefixes: {len(self.counts)}")
        for t in range(1, max_d + 1):
            n = len(self.get_prefixes_at_depth(t))
            print(f"    Spell level {t}: {n} unique prefixes")

    def __repr__(self):
        depths = [len(k) for k in self.counts.keys()]
        return f"SpellPrefixTree(max_spell_depth={max(depths) if depths else 0}, total_prefixes={len(self.counts)})"


def build_spell_prefix_tree(
    seqdata: SequenceData,
    expcost: float = 0.0,
) -> SpellPrefixTree:
    """
    Build a spell-based prefix tree from SequenceData.

    Parameters
    ----------
    seqdata : SequenceData
        Sequence data object. Must have been created with SequenceData(...).
    expcost : float, default 0.0
        Duration weight for derived indicators (e.g. JS divergence).
        - expcost=0: pure state-based; duration not used in structure or indicators.
        - expcost>0: when computing JS divergence, weight each spell's state
          by its duration (longer spells contribute more to the distribution).

    Returns
    -------
    SpellPrefixTree
        Tree where level k = after k-th spell.
    """
    spell_states, spell_durations, _ = convert_seqdata_to_spells(seqdata)

    tree = SpellPrefixTree()
    tree.total_sequences = len(spell_states)
    tree._expcost = expcost
    tree._spell_states = spell_states
    tree._spell_durations = spell_durations

    for i, (states_i, durs_i) in enumerate(zip(spell_states, spell_durations)):
        if not states_i:
            continue
        tree.insert(states_i, seq_idx=i, durations=durs_i if expcost != 0 else None)

    return tree


def get_depth_stats_spell(tree: SpellPrefixTree) -> Dict[str, Any]:
    """Depth-level stats for spell tree (same interface as position version)."""
    depth_counts = defaultdict(int)
    depth_to_prefixes = defaultdict(list)
    for k in tree.counts:
        d = len(k)
        depth_counts[d] += 1
        depth_to_prefixes[d].append(k)
    return {"depth_counts": dict(depth_counts), "depth_to_prefixes": dict(depth_to_prefixes)}


def compute_prefix_count_spell(
    tree: SpellPrefixTree,
    max_depth: int,
    depth_stats: Optional[Dict[str, Any]] = None,
) -> List[int]:
    """Prefix counts per spell level 1..max_depth."""
    if depth_stats is None:
        depth_counts = defaultdict(int)
        for k in tree.counts:
            depth_counts[len(k)] += 1
        depth_counts = dict(depth_counts)
    else:
        depth_counts = depth_stats["depth_counts"]
    return [depth_counts.get(t, 0) for t in range(1, max_depth + 1)]


def compute_branching_factor_spell(
    tree: SpellPrefixTree,
    max_depth: int,
    depth_prefixes: Optional[Dict[int, List[Tuple]]] = None,
) -> List[float]:
    """Branching factor per spell level (same logic as position version)."""
    if depth_prefixes is None:
        depth_to_prefixes = defaultdict(list)
        for k in tree.counts:
            depth_to_prefixes[len(k)].append(k)
        depth_to_prefixes = dict(depth_to_prefixes)
    else:
        depth_to_prefixes = depth_prefixes
    result = []
    for t in range(2, max_depth + 1):
        prefixes = depth_to_prefixes.get(t - 1, [])
        if not prefixes:
            result.append(0.0)
            continue
        child_counts = [tree.get_children_count(p) for p in prefixes]
        result.append(float(np.mean(child_counts)))
    return [0.0] + result


def compute_js_divergence_spell(
    spell_states: List[List[Any]],
    spell_durations: List[List[float]],
    state_set: List[Any],
    expcost: float = 0.0,
) -> List[float]:
    """
    Jensen-Shannon divergence between consecutive spell-level distributions.

    At each spell level k, we build the state distribution across sequences.
    - expcost=0: each sequence contributes 1/N to the state it has at spell k.
    - expcost>0: weight by duration; a spell of length d contributes d times
      as much as a spell of length 1 (duration-aware distribution).
    """
    state_list = list(state_set)
    n_states = len(state_list)
    state_to_idx = {s: i for i, s in enumerate(state_list)}
    N = len(spell_states)

    max_spells = max(len(s) for s in spell_states)
    if max_spells < 2:
        return [0.0]

    # Per-spell-level distributions
    distros = []
    for k in range(max_spells):
        counts = np.zeros(n_states, dtype=float)
        total = 0.0
        for i, (states_i, durs_i) in enumerate(zip(spell_states, spell_durations)):
            if k >= len(states_i):
                continue
            s = states_i[k]
            if s not in state_to_idx:
                continue
            idx = state_to_idx[s]
            w = 1.0
            if expcost > 0 and k < len(durs_i):
                w = float(durs_i[k]) ** expcost  # duration weighting
            counts[idx] += w
            total += w
        if total > 0:
            distros.append(counts / total)
        else:
            distros.append(counts)

    js_scores = [0.0]
    for k in range(1, len(distros)):
        p, q = distros[k], distros[k - 1]
        if np.sum(p) <= 0 or np.sum(q) <= 0:
            js_scores.append(0.0)
        else:
            js = jensenshannon(p, q)
            js_scores.append(float(js))
    return js_scores
