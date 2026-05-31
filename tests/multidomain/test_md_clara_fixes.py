"""
Regression tests for MD-CLARA profile aggregation, sampling weights, and objectives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData
from sequenzo.big_data.clara.utils.aggregatecases import DataFrameAggregator
from sequenzo.multidomain.clara._utils import build_multidomain_profile_frame
from sequenzo.multidomain.clara.distance_providers import (
    CATDistanceProvider,
    DATDistanceProvider,
    IDCDDistanceProvider,
)
from sequenzo.multidomain.clara.md_clara import md_clara

try:
    from simulation.metrics import (
        medoid_average_distance_from_dist_to_medoids,
        medoid_total_objective_from_dist_to_medoids,
    )
except ImportError:  # pragma: no cover
    medoid_total_objective_from_dist_to_medoids = None
    medoid_average_distance_from_dist_to_medoids = None

pytestmark = pytest.mark.skipif(
    medoid_total_objective_from_dist_to_medoids is None,
    reason="multidomain_clara/simulation not on PYTHONPATH",
)

_TIME = ["t1", "t2", "t3", "t4"]
_STATES = [0, 1]


def _make_domain(rows: list[list[int]]) -> SequenceData:
    df = pd.DataFrame(rows, columns=_TIME)
    return SequenceData(data=df, time=_TIME, states=_STATES)


def _aggregate_profiles(domains: list[SequenceData]) -> dict:
    profiles = build_multidomain_profile_frame(domains)
    return DataFrameAggregator().aggregate(profiles)


def test_multidomain_aggregation_keeps_cases_with_different_second_domain():
    """Identical domain 1 but different domain 2 must not merge."""
    domain1 = _make_domain([[0, 0, 1, 1], [0, 0, 1, 1]])
    domain2 = _make_domain([[0, 1, 1, 1], [1, 1, 0, 0]])
    ac = _aggregate_profiles([domain1, domain2])
    assert len(ac["aggWeights"]) == 2


def test_multidomain_aggregation_merges_true_duplicates():
    """Identical trajectories in every domain merge with summed frequency."""
    domain1 = _make_domain([[0, 0, 1, 1], [0, 0, 1, 1]])
    domain2 = _make_domain([[0, 1, 1, 1], [0, 1, 1, 1]])
    ac = _aggregate_profiles([domain1, domain2])
    assert len(ac["aggWeights"]) == 1
    assert ac["aggWeights"][0] == 2


def test_weighted_objective_raw_matches_aggregated():
    """Total objective on raw cases equals weighted objective on unique profiles."""
    domain1 = _make_domain([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0]])
    domain2 = _make_domain([[0, 1, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]])
    ac = _aggregate_profiles([domain1, domain2])

    raw_dist = np.array(
        [
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 5.0],
            [5.0, 5.0, 0.0],
        ],
        dtype=float,
    )
    agg_dist = raw_dist[np.ix_([0, 2], [0, 2])]
    medoids = np.array([0], dtype=int)

    raw_total = medoid_total_objective_from_dist_to_medoids(
        raw_dist[:, medoids],
        weights=np.ones(3),
    )
    agg_total = medoid_total_objective_from_dist_to_medoids(
        agg_dist[:, medoids],
        weights=np.asarray(ac["aggWeights"], dtype=float),
    )
    assert raw_total == pytest.approx(agg_total)


def test_total_and_average_objectives_rank_repetitions_equally():
    """Minimizing total and average nearest-medoid distance picks the same winner."""
    diss = np.array(
        [
            [0.0, 4.0, 5.0, 3.0],
            [4.0, 0.0, 3.0, 2.0],
            [5.0, 3.0, 0.0, 1.5],
            [3.0, 2.0, 1.5, 0.0],
        ],
        dtype=float,
    )
    weights = np.array([2.0, 1.0, 3.0, 1.0], dtype=float)

    totals = []
    averages = []
    for medoid in range(diss.shape[0]):
        dist_to_medoid = diss[:, [medoid]]
        totals.append(
            medoid_total_objective_from_dist_to_medoids(dist_to_medoid, weights)
        )
        averages.append(
            medoid_average_distance_from_dist_to_medoids(dist_to_medoid, weights)
        )
    assert int(np.argmin(totals)) == int(np.argmin(averages))


@pytest.fixture(scope="module")
def tiny_domains():
    rows = [
        [0, 0, 1, 1],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 1],
    ]
    d1 = _make_domain(rows)
    d2 = _make_domain([row[::-1] for row in rows])
    return [d1, d2]


@pytest.mark.parametrize(
    "provider_cls,params",
    [
        (
            IDCDDistanceProvider,
            {"method": "OM", "sm": "CONSTANT", "indel": 1, "norm": "none"},
        ),
        (
            CATDistanceProvider,
            {
                "method": "OM",
                "sm": ["CONSTANT", "CONSTANT"],
                "indel": "auto",
                "norm": "none",
            },
        ),
        (
            DATDistanceProvider,
            {
                "method_params": [{"method": "HAM"}, {"method": "HAM"}],
                "domain_weights": [1.0, 1.0],
                "link": "sum",
            },
        ),
    ],
)
def test_provider_distances_match_full_matrix(tiny_domains, provider_cls, params):
    """Provider sample/medoid queries match the full pairwise matrix."""
    provider = provider_cls(tiny_domains, **params)
    n = provider.n_sequences()
    full = provider.sample_distance_matrix(np.arange(n, dtype=int))

    sample_idx = np.array([0, 2, 4], dtype=int)
    sample = provider.sample_distance_matrix(sample_idx)
    assert np.allclose(sample, full[np.ix_(sample_idx, sample_idx)])

    medoids = np.array([1, 3], dtype=int)
    to_medoids = provider.distance_to_medoids(medoids)
    assert np.allclose(to_medoids, full[:, medoids])


def test_aggregation_indices_are_one_based():
    df = pd.DataFrame(
        {
            "t1": ["A", "A", "B"],
            "t2": ["A", "A", "B"],
        }
    )
    ac = DataFrameAggregator().aggregate(df)
    assert np.min(ac["aggIndex"]) >= 1
    assert np.min(ac["disaggIndex"]) >= 1
    assert len(ac["aggWeights"]) == 2
    assert np.sum(ac["aggWeights"]) == 3


def test_stability_excludes_selected_repetition(tiny_domains):
    result = md_clara(
        tiny_domains,
        strategy="dat",
        distance_params={"method_params": [{"method": "HAM"}, {"method": "HAM"}]},
        R=5,
        sample_size=4,
        kvals=[2],
        stability=True,
        n_jobs=1,
        verbose=False,
        random_state=1,
    )
    info = result.stability[2]
    assert info["n_comparisons"] == 4
    assert len(info["ari"]) == 4
    assert len(info["jc"]) == 4


def test_md_clara_rejects_non_distance_criteria(tiny_domains):
    with pytest.raises(ValueError, match="Unknown criterion 'ams'"):
        md_clara(
            tiny_domains,
            strategy="dat",
            distance_params={"method_params": [{"method": "HAM"}, {"method": "HAM"}]},
            criteria=("ams",),
            R=2,
            sample_size=4,
            kvals=[2],
            n_jobs=1,
            verbose=False,
        )


def test_md_clara_caps_sample_size_to_unique_profiles():
    """Requested b above N* is capped with a warning and recorded in settings."""
    domain1 = _make_domain([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0]])
    domain2 = _make_domain([[0, 1, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]])
    domains = [domain1, domain2]

    with pytest.warns(UserWarning, match="exceeds the number of unique"):
        result = md_clara(
            domains,
            strategy="dat",
            distance_params={
                "method_params": [{"method": "HAM"}, {"method": "HAM"}],
            },
            R=2,
            sample_size=10,
            kvals=[2],
            n_jobs=1,
            verbose=False,
            random_state=0,
        )

    assert result.settings["requested_sample_size"] == 10
    assert result.settings["effective_sample_size"] == 2
    assert result.settings["n_unique_profiles"] == 2


def test_domain_weights_combine_with_duplicate_profiles():
    """Duplicate profiles merge with summed original case weights."""
    time_cols = ["t1", "t2"]
    domain1 = SequenceData(
        data=pd.DataFrame([[0, 0], [0, 0]], columns=time_cols),
        time=time_cols,
        states=_STATES,
    )
    domain2 = SequenceData(
        data=pd.DataFrame([[0, 1], [0, 1]], columns=time_cols),
        time=time_cols,
        states=_STATES,
    )
    weights = np.array([3.0, 1.0], dtype=float)
    seq1 = SequenceData(
        data=domain1.data,
        time=time_cols,
        states=_STATES,
        weights=weights,
    )
    seq2 = SequenceData(
        data=domain2.data,
        time=time_cols,
        states=_STATES,
        weights=weights,
    )
    profiles = build_multidomain_profile_frame([seq1, seq2])
    ac = DataFrameAggregator().aggregate(profiles, weights=weights)
    assert len(ac["aggWeights"]) == 1
    assert ac["aggWeights"][0] == 4


def test_domain_weights_must_match_across_domains():
    domain1 = SequenceData(
        data=pd.DataFrame([[0, 0], [1, 1]], columns=["t1", "t2"]),
        time=["t1", "t2"],
        states=_STATES,
        weights=np.array([1.0, 2.0]),
    )
    domain2 = SequenceData(
        data=pd.DataFrame([[0, 1], [1, 0]], columns=["t1", "t2"]),
        time=["t1", "t2"],
        states=_STATES,
        weights=np.array([1.0, 3.0]),
    )
    with pytest.raises(ValueError, match="do not match domain 0"):
        md_clara(
            [domain1, domain2],
            strategy="dat",
            distance_params={"method_params": [{"method": "HAM"}, {"method": "HAM"}]},
            R=1,
            sample_size=2,
            kvals=[2],
            n_jobs=1,
            verbose=False,
        )


@pytest.mark.parametrize("strategy,distance_params", [
    ("idcd", {"method": "OM", "sm": "CONSTANT", "indel": 1, "norm": "none"}),
    (
        "cat",
        {
            "method": "OM",
            "sm": ["CONSTANT", "CONSTANT"],
            "indel": "auto",
            "norm": "none",
        },
    ),
    (
        "dat",
        {
            "method_params": [{"method": "HAM"}, {"method": "HAM"}],
            "domain_weights": [1.0, 1.0],
            "link": "sum",
        },
    ),
])
def test_tiny_end_to_end_smoke(strategy, distance_params):
    """Tiny end-to-end run checks finite objectives, CQIs, and RSS stats."""
    from simulation.data_generators import simulate_multidomain_sequences
    from simulation.memory_runtime import measure_runtime_memory

    sim = simulate_multidomain_sequences(
        N=100,
        D=2,
        T=8,
        K_true=3,
        states_per_domain=3,
        noise=0.05,
        random_state=0,
    )

    def _run():
        return md_clara(
            sim.domains,
            strategy=strategy,
            distance_params=distance_params,
            R=3,
            sample_size=30,
            kvals=[3],
            stability=False,
            n_jobs=1,
            verbose=False,
            random_state=0,
        )

    result, stats = measure_runtime_memory(_run)
    row = result.stats.iloc[0]
    assert np.isfinite(row["total_diss"])
    assert np.isfinite(row["avg_dist"])
    assert np.isfinite(row["ams"])
    assert np.isfinite(stats["peak_memory_mb"])
    assert result.settings["effective_sample_size"] <= result.settings["n_unique_profiles"]


def test_md_clara_stats_include_total_and_average_distance(tiny_domains):
    result = md_clara(
        tiny_domains,
        strategy="dat",
        distance_params={
            "method_params": [{"method": "HAM"}, {"method": "HAM"}],
        },
        R=3,
        sample_size=4,
        kvals=[2],
        n_jobs=1,
        verbose=False,
        random_state=0,
    )
    assert "total_diss" in result.stats.columns
    assert "avg_dist" in result.stats.columns
    assert result.stats.loc[0, "total_diss"] >= result.stats.loc[0, "avg_dist"]
