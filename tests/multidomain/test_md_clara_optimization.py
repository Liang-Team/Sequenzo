"""
Tests for MD-CLARA Phase 3 optimizations and Phase 4 diagnostics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import squareform

from sequenzo import SequenceData, load_dataset
from sequenzo.clustering.k_medoids import KMedoids
from sequenzo.clustering.sequences_to_variables.helske_regression_variables import (
    medoid_indices_from_kmedoids_result,
)
from sequenzo.multidomain.clara._utils import (
    assert_condensed_distance_shape,
    freeze_seqdist_costs,
    subset_sequence_data,
    validate_profile_weights,
)
from sequenzo.multidomain.clara.clara_engine import _assemble_distance_to_medoids
from sequenzo.multidomain.clara.diagnostics import (
    compare_md_clara_strategies,
    dat_domain_contributions,
    summarize_combined_state_space,
)
from sequenzo.multidomain.clara.distance_providers import (
    CATDistanceProvider,
    DATDistanceProvider,
    DistanceProvider,
    IDCDDistanceProvider,
)
from sequenzo.multidomain.clara.md_clara import md_clara


@pytest.fixture(scope="module")
def biofam_domains():
    left_df = load_dataset("biofam_left_domain").head(60)
    children_df = load_dataset("biofam_child_domain").head(60)
    married_df = load_dataset("biofam_married_domain").head(60)
    time_cols = [col for col in children_df.columns if col != "id"]

    return [
        SequenceData(
            data=left_df,
            time=time_cols,
            id_col="id",
            states=[0, 1],
            labels=["At home", "Left home"],
        ),
        SequenceData(
            data=children_df,
            time=time_cols,
            id_col="id",
            states=[0, 1],
            labels=["No child", "Child"],
        ),
        SequenceData(
            data=married_df,
            time=time_cols,
            id_col="id",
            states=[0, 1],
            labels=["Not married", "Married"],
        ),
    ]


class _FakeMedoidProvider(DistanceProvider):
    """Minimal provider that records distance_to_medoids calls."""

    def __init__(self, n_sequences: int = 10) -> None:
        self._n = n_sequences
        self.calls: list[tuple[int, ...]] = []

    def n_sequences(self) -> int:
        return self._n

    def sample_distances(self, sample_indices, *, condensed: bool = False):
        raise NotImplementedError

    def distance_to_medoids(self, medoid_indices):
        medoids = tuple(int(m) for m in medoid_indices)
        self.calls.append(medoids)
        n = self._n
        matrix = np.zeros((n, len(medoids)), dtype=float)
        for col, medoid in enumerate(medoids):
            matrix[:, col] = float(medoid)
        return matrix


@pytest.mark.parametrize("provider_cls,provider_kwargs", [
    (
        IDCDDistanceProvider,
        {"method": "HAM", "sm": "CONSTANT", "indel": 1, "norm": "none"},
    ),
    (
        CATDistanceProvider,
        {
            "method": "HAM",
            "sm": ["CONSTANT", "CONSTANT", "CONSTANT"],
            "indel": 1,
            "norm": "none",
        },
    ),
    (
        DATDistanceProvider,
        {"method_params": [{"method": "HAM"}] * 3, "link": "sum"},
    ),
])
def test_condensed_matches_square(biofam_domains, provider_cls, provider_kwargs):
    provider = provider_cls(biofam_domains, **provider_kwargs)
    indices = [0, 1, 4, 7, 12]

    square = provider.sample_distances(indices, condensed=False)
    condensed = provider.sample_distances(indices, condensed=True)
    assert_condensed_distance_shape(condensed, indices)
    assert np.allclose(squareform(condensed), square)


@pytest.mark.parametrize("provider_cls,provider_kwargs", [
    (
        IDCDDistanceProvider,
        {"method": "HAM", "sm": "CONSTANT", "indel": 1, "norm": "none"},
    ),
    (
        CATDistanceProvider,
        {
            "method": "HAM",
            "sm": ["CONSTANT", "CONSTANT", "CONSTANT"],
            "indel": 1,
            "norm": "none",
        },
    ),
    (
        DATDistanceProvider,
        {"method_params": [{"method": "HAM"}] * 3, "link": "sum"},
    ),
])
def test_distance_to_medoids_matches_full_columns(
    biofam_domains, provider_cls, provider_kwargs
):
    provider = provider_cls(biofam_domains, **provider_kwargs)
    all_indices = np.arange(provider.n_sequences())
    medoids = [1, 4, 8]
    full_square = provider.sample_distances(all_indices, condensed=False)
    actual = provider.distance_to_medoids(medoids)
    assert np.allclose(actual, full_square[:, medoids])


def test_dat_condensed_sum_matches_manual(biofam_domains):
    provider = DATDistanceProvider(
        biofam_domains,
        method_params=[{"method": "HAM"}] * 3,
        link="sum",
    )
    indices = [0, 2, 5, 9]
    condensed = provider.sample_distances(indices, condensed=True)
    square = provider.sample_distances(indices, condensed=False)
    assert np.allclose(squareform(condensed), square)


def test_dat_weighted_components_sum_to_distance_matrix(biofam_domains):
    provider = DATDistanceProvider(
        biofam_domains,
        method_params=[{"method": "HAM"}] * 3,
        domain_weights=[0.5, 2.0, 1.0],
        link="mean",
    )
    medoids = [0, 5, 12]
    combined = provider.distance_to_medoids(medoids)
    components = provider.weighted_per_domain_distance_to_medoids(medoids)
    reconstructed = sum(components)
    assert np.allclose(reconstructed, combined)


@pytest.mark.parametrize(
    "domain_weights,link",
    [
        ([1.0, 1.0, 1.0], "sum"),
        ([0.5, 2.0, 1.0], "sum"),
        ([1.0, 1.0, 1.0], "mean"),
        ([0.5, 2.0, 1.0], "mean"),
    ],
)
def test_dat_domain_contributions_sum_to_one(
    biofam_domains,
    domain_weights,
    link,
):
    provider = DATDistanceProvider(
        biofam_domains,
        method_params=[{"method": "HAM"}] * 3,
        domain_weights=domain_weights,
        link=link,
    )
    medoids = [0, 8]
    n = provider.n_sequences()
    clustering = np.zeros(n, dtype=int)
    weights = np.ones(n, dtype=float)

    frame = dat_domain_contributions(
        provider,
        medoids=medoids,
        clustering=clustering,
        profile_weights=weights,
    )
    all_rows = frame[frame["cluster"] == "all"]
    assert np.isclose(all_rows["contribution_share"].sum(), 1.0, atol=1e-6)


def test_kmedoids_accepts_condensed_distance_vector(biofam_domains):
    provider = IDCDDistanceProvider(
        biofam_domains,
        method="HAM",
        sm="CONSTANT",
        indel=1,
        norm="none",
    )
    indices = [0, 1, 4, 7, 12, 20]
    condensed = provider.sample_distances(indices, condensed=True)
    weights = np.ones(len(indices))

    result = KMedoids(
        diss=condensed,
        k=2,
        initialclust=None,
        weights=weights,
        verbose=False,
    )
    assert result is not None
    assert len(result) == len(indices)


def test_kmedoids_condensed_matches_square_medoids(biofam_domains):
    provider = IDCDDistanceProvider(
        biofam_domains,
        method="HAM",
        sm="CONSTANT",
        indel=1,
        norm="none",
    )
    indices = [0, 1, 4, 7, 12, 20]
    weights = np.ones(len(indices))
    condensed = provider.sample_distances(indices, condensed=True)
    square = provider.sample_distances(indices, condensed=False)

    medoids_condensed = medoid_indices_from_kmedoids_result(
        KMedoids(diss=condensed, k=2, initialclust=None, weights=weights, verbose=False)
    )
    medoids_square = medoid_indices_from_kmedoids_result(
        KMedoids(diss=square, k=2, initialclust=None, weights=weights, verbose=False)
    )
    assert np.array_equal(medoids_condensed, medoids_square)


def test_condensed_subsample_matches_square_path(biofam_domains):
    common = dict(
        domains=biofam_domains,
        strategy="dat",
        distance_params={
            "method_params": [{"method": "HAM"}] * 3,
            "domain_weights": [0.5, 2.0, 1.0],
            "link": "mean",
        },
        R=4,
        sample_size=30,
        kvals=[2, 3, 4],
        n_jobs=1,
        verbose=False,
        random_state=11,
        combined_state_space=False,
        use_medoid_cache=False,
    )
    square = md_clara(**common, condensed_subsample=False)
    condensed = md_clara(**common, condensed_subsample=True)
    assert np.allclose(square.stats["total_diss"], condensed.stats["total_diss"])
    assert np.array_equal(
        square.clustering.to_numpy(),
        condensed.clustering.to_numpy(),
    )


def test_cached_and_uncached_results_match(biofam_domains):
    common = dict(
        domains=biofam_domains,
        strategy="idcd",
        distance_params={
            "method": "HAM",
            "sm": "CONSTANT",
            "indel": 1,
            "norm": "none",
        },
        R=4,
        sample_size=30,
        kvals=[2, 3, 4],
        n_jobs=1,
        verbose=False,
        random_state=7,
        combined_state_space=False,
    )
    cached = md_clara(**common, use_medoid_cache=True)
    uncached = md_clara(**common, use_medoid_cache=False)

    assert np.allclose(cached.stats["total_diss"], uncached.stats["total_diss"])
    assert np.array_equal(
        cached.clustering.to_numpy(),
        uncached.clustering.to_numpy(),
    )
    cache_stats = cached.route_diagnostics.get("medoid_cache", {})
    assert cache_stats.get("hits", 0) > 0
    assert cache_stats.get("peak_cached_columns", 0) > 0


def test_medoid_column_cache_hits_and_misses():
    provider = _FakeMedoidProvider()
    cache: dict[int, np.ndarray] = {}

    _, hits1, misses1 = _assemble_distance_to_medoids(
        provider, np.array([1, 3]), cache
    )
    assert hits1 == 0
    assert misses1 == 2
    assert provider.calls == [(1, 3)]

    _, hits2, misses2 = _assemble_distance_to_medoids(
        provider, np.array([1, 3, 5]), cache
    )
    assert hits2 == 2
    assert misses2 == 1
    assert provider.calls == [(1, 3), (5,)]


def test_summarize_combined_state_space(biofam_domains):
    summary = summarize_combined_state_space(biofam_domains)
    assert summary["theoretical_combined_states"] == 8
    assert summary["observed_combined_states"] <= 8
    assert 0 < summary["coverage"] <= 1
    assert summary["rare_state_share_basis"] == "case_frequency"
    assert "min_position_frequency" in summary
    assert summary["max_position_frequency"] > 0
    assert summary["max_case_frequency"] > 0


def test_freeze_seqdist_costs_replaces_string_sm(biofam_domains):
    from sequenzo.multidomain.idcd import create_idcd_sequence_from_domains

    md = create_idcd_sequence_from_domains(biofam_domains, quiet=True)
    frozen = freeze_seqdist_costs(
        md,
        {"method": "OM", "sm": "INDELSLOG", "indel": "auto", "norm": "none"},
    )
    assert not isinstance(frozen["sm"], str)
    assert not (isinstance(frozen["indel"], str) and frozen["indel"] == "auto")
    assert len(np.asarray(frozen["indel"]).reshape(-1)) == len(md.states)


def test_idcd_provider_freezes_data_dependent_costs(biofam_domains):
    from sequenzo.multidomain.idcd import create_idcd_sequence_from_domains

    md = create_idcd_sequence_from_domains(biofam_domains, quiet=True)
    provider = IDCDDistanceProvider(
        biofam_domains,
        method="OM",
        sm="INDELSLOG",
        indel="auto",
        norm="none",
    )
    assert not isinstance(provider._dist_args["sm"], str)
    indel = provider._dist_args["indel"]
    assert len(np.asarray(indel).reshape(-1)) == len(md.states)


def test_idcd_indelslog_uses_fixed_metric(biofam_domains):
    provider = IDCDDistanceProvider(
        biofam_domains,
        method="OM",
        sm="INDELSLOG",
        indel="auto",
        norm="none",
    )
    sample = np.array([0, 2, 4, 6])
    d_sample = provider.sample_distances(sample, condensed=False)
    all_indices = np.arange(provider.n_sequences())
    d_full = provider.sample_distances(all_indices, condensed=False)
    np.testing.assert_allclose(
        d_sample,
        d_full[np.ix_(sample, sample)],
        rtol=1e-10,
        atol=1e-10,
    )


def test_dat_indelslog_uses_fixed_metric(biofam_domains):
    provider = DATDistanceProvider(
        biofam_domains,
        method_params=[
            {"method": "OM", "sm": "INDELSLOG", "indel": "auto", "norm": "none"}
        ]
        * 3,
        link="sum",
    )
    sample = np.array([0, 2, 4, 6])
    d_sample = provider.sample_distances(sample, condensed=False)
    all_indices = np.arange(provider.n_sequences())
    d_full = provider.sample_distances(all_indices, condensed=False)
    np.testing.assert_allclose(
        d_sample,
        d_full[np.ix_(sample, sample)],
        rtol=1e-10,
        atol=1e-10,
    )


def test_guard_rejects_chi2(biofam_domains):
    from sequenzo.multidomain.idcd import create_idcd_sequence_from_domains

    md = create_idcd_sequence_from_domains(biofam_domains, quiet=True)
    with pytest.raises(NotImplementedError, match="CHI2"):
        freeze_seqdist_costs(md, {"method": "CHI2", "norm": "none"})


def test_guard_rejects_elzingastuder(biofam_domains):
    from sequenzo.multidomain.idcd import create_idcd_sequence_from_domains

    md = create_idcd_sequence_from_domains(biofam_domains, quiet=True)
    with pytest.raises(NotImplementedError, match="ElzingaStuder"):
        freeze_seqdist_costs(
            md,
            {"method": "OM", "sm": "CONSTANT", "indel": 1, "norm": "ElzingaStuder"},
        )


def test_md_clara_default_distance_params(biofam_domains):
    result = md_clara(
        biofam_domains,
        strategy="idcd",
        R=2,
        sample_size=25,
        kvals=[2],
        n_jobs=1,
        verbose=False,
        random_state=0,
    )
    assert result.settings["distance_params"]["method"] == "OM"
    assert result.settings["distance_params"]["sm"] == "CONSTANT"
    labels = result.clustering["Cluster 2"]
    assert pd.api.types.is_integer_dtype(labels.dtype)
    assert set(labels.unique()).issubset({-1, 1, 2})

    best = result.best_clustering(2)
    assert best.dtype.kind in {"i", "u"}
    assert best.min() >= 1
    assert best.max() <= 2


def test_compare_md_clara_strategies(biofam_domains):
    common = dict(
        domains=biofam_domains,
        R=3,
        sample_size=28,
        kvals=[2],
        n_jobs=1,
        verbose=False,
        random_state=0,
        combined_state_space=False,
    )
    idcd = md_clara(
        strategy="idcd",
        distance_params={"method": "HAM", "sm": "CONSTANT", "indel": 1, "norm": "none"},
        **common,
    )
    dat = md_clara(
        strategy="dat",
        distance_params={"method_params": [{"method": "HAM"}] * 3},
        **common,
    )
    table = compare_md_clara_strategies({"idcd": idcd, "dat": dat}, k=2)
    assert len(table) == 1
    assert "ari" in table.columns


def test_compare_md_clara_strategies_rejects_misaligned_index(biofam_domains):
    common = dict(
        domains=biofam_domains,
        R=2,
        sample_size=25,
        kvals=[2],
        n_jobs=1,
        verbose=False,
        random_state=0,
        combined_state_space=False,
    )
    r1 = md_clara(
        strategy="idcd",
        distance_params={"method": "HAM", "sm": "CONSTANT", "indel": 1, "norm": "none"},
        **common,
    )
    r2 = md_clara(
        strategy="dat",
        distance_params={"method_params": [{"method": "HAM"}] * 3},
        **common,
    )
    misaligned_clustering = r2.clustering.copy()
    misaligned_clustering.index = list(misaligned_clustering.index)[::-1]
    r2.clustering = misaligned_clustering

    with pytest.raises(ValueError, match="different case order"):
        compare_md_clara_strategies({"idcd": r1, "dat": r2}, k=2)


def test_cat_sm_length_mismatch(biofam_domains):
    with pytest.raises(ValueError, match="sm length"):
        CATDistanceProvider(
            biofam_domains,
            method="HAM",
            sm=["CONSTANT", "CONSTANT"],
            indel=1,
            norm="none",
        )


def test_validate_profile_weights_rejects_invalid():
    with pytest.raises(ValueError, match="strictly positive"):
        validate_profile_weights([0.0, 2.0], n_profiles=2)
    with pytest.raises(ValueError, match="strictly positive"):
        validate_profile_weights([-1.0, 2.0], n_profiles=2)


def test_md_clara_rejects_n_jobs_zero(biofam_domains):
    with pytest.raises(ValueError, match="n_jobs must not be 0"):
        md_clara(
            biofam_domains,
            strategy="idcd",
            distance_params={"method": "HAM", "sm": "CONSTANT", "indel": 1, "norm": "none"},
            R=2,
            sample_size=25,
            kvals=[2],
            n_jobs=0,
            verbose=False,
        )


def test_dat_rejects_n_jobs_domains_zero(biofam_domains):
    with pytest.raises(ValueError, match="n_jobs_domains"):
        DATDistanceProvider(
            biofam_domains,
            method_params=[{"method": "HAM"}] * 3,
            n_jobs_domains=0,
        )


def test_dat_domain_weights_must_be_one_dimensional(biofam_domains):
    with pytest.raises(ValueError, match="one-dimensional"):
        DATDistanceProvider(
            biofam_domains,
            method_params=[{"method": "HAM"}] * 3,
            domain_weights=[[1.0], [2.0], [1.0]],
        )


def test_engine_rejects_duplicate_kvals(biofam_domains):
    from sequenzo.big_data.clara.utils.aggregatecases import DataFrameAggregator
    from sequenzo.multidomain.clara._utils import aggregate_domains, build_multidomain_profile_frame
    from sequenzo.multidomain.clara.clara_engine import clara_from_distance_provider

    profiles = build_multidomain_profile_frame(biofam_domains)
    ac = DataFrameAggregator().aggregate(profiles)
    agg_domains = aggregate_domains(biofam_domains, ac)
    provider = IDCDDistanceProvider(
        agg_domains,
        method="HAM",
        sm="CONSTANT",
        indel=1,
        norm="none",
    )
    with pytest.raises(ValueError, match="duplicate"):
        clara_from_distance_provider(
            provider,
            reference_seqdata=biofam_domains[0],
            aggregation=ac,
            R=2,
            sample_size=25,
            kvals=[2, 3, 3],
            n_jobs=1,
            verbose=False,
        )


def test_subset_sequence_data_rejects_out_of_range(biofam_domains):
    domain = biofam_domains[0]
    with pytest.raises(IndexError, match="outside the valid row range"):
        subset_sequence_data(domain, [0, len(domain.data)])


def test_nested_parallelism_warning():
    from sequenzo.multidomain.clara._utils import warn_nested_parallelism

    with pytest.warns(UserWarning, match="oversubscribe"):
        warn_nested_parallelism(n_jobs=2, n_jobs_domains=2)
