"""Partition JC (WeightedCluster) vs pairwise binom Jaccard vs clusterwise set Jaccard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import comb

from sequenzo.big_data.clara.clara import jaccardCoef
from sequenzo.multidomain.clara._utils import (
    clusterwise_set_jaccard,
    pairwise_jaccard_from_contingency,
    weightedcluster_partition_jaccard,
)


def test_weightedcluster_partition_jaccard_matches_seqclararange():
    tab = pd.DataFrame(np.array([[2, 1], [0, 1]], dtype=float))
    assert weightedcluster_partition_jaccard(tab) == pytest.approx(float(jaccardCoef(tab)))
    assert weightedcluster_partition_jaccard(tab) == pytest.approx(0.5)


def test_pairwise_jaccard_matches_binom_formula():
    # A: [0,0,0,1], B: [0,0,1,1] → one co-clustered pair among the first three.
    tab = np.array([[2, 1], [0, 1]], dtype=float)
    n11 = float(np.sum(comb(tab, 2)))
    n01 = float(np.sum(comb(tab.sum(axis=0), 2)))
    n10 = float(np.sum(comb(tab.sum(axis=1), 2)))
    expected = n11 / (n01 + n10 - n11)
    assert pairwise_jaccard_from_contingency(tab) == pytest.approx(expected)
    assert expected == pytest.approx(0.25)


def test_pairwise_jaccard_differs_from_weightedcluster_self_pairs():
    tab = pd.DataFrame(np.array([[2, 1], [0, 1]], dtype=float))
    pairwise = pairwise_jaccard_from_contingency(tab)
    wc = weightedcluster_partition_jaccard(tab)
    assert pairwise == pytest.approx(0.25)
    assert wc == pytest.approx(0.5)


def test_identical_partitions_are_one():
    tab = np.array([[3, 0], [0, 2]], dtype=float)
    assert pairwise_jaccard_from_contingency(tab) == 1.0
    assert weightedcluster_partition_jaccard(tab) == 1.0
    assert pairwise_jaccard_from_contingency(np.array([[4.0]])) == 1.0
    assert weightedcluster_partition_jaccard(np.array([[4.0]])) == 1.0


def test_clusterwise_set_jaccard_best_overlap():
    left = np.array([1, 1, 1, 2, 2])
    right = np.array([1, 1, 3, 2, 2])
    assert clusterwise_set_jaccard(left, right, 1) == pytest.approx(2.0 / 3.0)
    assert clusterwise_set_jaccard(left, right, 2) == pytest.approx(1.0)
