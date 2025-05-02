"""
@Author  : Yuqi Liang 梁彧祺
@File    : system_level_indicators.py
@Time    : 02/05/2025 11:06
@Desc    : 
"""
from collections import defaultdict, Counter
import numpy as np
from scipy.spatial.distance import jensenshannon


class PrefixTree:
    def __init__(self):
        self.root = {}
        self.counts = defaultdict(int)  # prefix -> count

    def insert(self, sequence):
        prefix = []
        node = self.root
        for state in sequence:
            prefix.append(state)
            key = tuple(prefix)
            self.counts[key] += 1
            if state not in node:
                node[state] = {}
            node = node[state]

    def get_prefixes_at_depth(self, depth):
        return [k for k in self.counts if len(k) == depth]

    def get_children_count(self, prefix):
        node = self.root
        for state in prefix:
            node = node.get(state, {})
        return len(node)


def compute_prefix_count(tree, max_depth):
    return [len(tree.get_prefixes_at_depth(t)) for t in range(1, max_depth + 1)]


def compute_branching_factor(tree, max_depth):
    result = []
    for t in range(2, max_depth + 1):
        prefixes = tree.get_prefixes_at_depth(t - 1)
        if not prefixes:
            result.append(0)
            continue
        child_counts = [tree.get_children_count(p) for p in prefixes]
        result.append(np.mean(child_counts))
    return [0] + result  # pad to align with prefix count


def compute_js_divergence(sequences, state_set):
    T = len(sequences[0])
    distros = []
    for t in range(T):
        counter = Counter(seq[t] for seq in sequences)
        dist = np.array([counter[s] for s in state_set], dtype=float)
        dist = dist / dist.sum()
        distros.append(dist)

    js_scores = [0.0]
    for t in range(1, T):
        js = jensenshannon(distros[t], distros[t - 1])
        js_scores.append(js)
    return js_scores


def compute_composite_score(prefix_counts, branching_factors, js_divergence=None):
    pc_z = (np.array(prefix_counts) - np.mean(prefix_counts)) / np.std(prefix_counts)
    bf_z = (np.array(branching_factors) - np.mean(branching_factors)) / np.std(branching_factors)
    score = pc_z + bf_z

    if js_divergence is not None:
        js_z = (np.array(js_divergence) - np.mean(js_divergence)) / np.std(js_divergence)
        score += js_z

    return score.tolist()


def build_prefix_tree(sequences):
    tree = PrefixTree()
    for seq in sequences:
        for t in range(1, len(seq) + 1):
            tree.insert(seq[:t])
    return tree
