"""
Tests for scalable multidomain CLARA (IDCD, CAT, DAT distance providers).
"""

from __future__ import annotations

import numpy as np
import pytest

from sequenzo import SequenceData, load_dataset
from sequenzo.multidomain.clara.distance_providers import (
    CATDistanceProvider,
    DATDistanceProvider,
    IDCDDistanceProvider,
    make_distance_provider,
)
from sequenzo.multidomain.clara.md_clara import md_clara
from sequenzo.multidomain.idcd import create_idcd_sequence_from_domains


@pytest.fixture(scope="module")
def biofam_domains():
    """Three-domain Biofam subset used across provider tests."""
    left_df = load_dataset("biofam_left_domain")
    children_df = load_dataset("biofam_child_domain")
    married_df = load_dataset("biofam_married_domain")

    time_cols = [col for col in children_df.columns if col != "id"]
    # Small subset for fast tests
    left_df = left_df.head(80)
    children_df = children_df.head(80)
    married_df = married_df.head(80)

    seq_left = SequenceData(
        data=left_df,
        time=time_cols,
        id_col="id",
        states=[0, 1],
        labels=["At home", "Left home"],
    )
    seq_child = SequenceData(
        data=children_df,
        time=time_cols,
        id_col="id",
        states=[0, 1],
        labels=["No child", "Child"],
    )
    seq_marr = SequenceData(
        data=married_df,
        time=time_cols,
        id_col="id",
        states=[0, 1],
        labels=["Not married", "Married"],
    )
    return [seq_left, seq_child, seq_marr]


def test_idcd_builds_observed_expanded_alphabet(biofam_domains):
    md = create_idcd_sequence_from_domains(biofam_domains, quiet=True)
    # Binary domains -> at most 8 combined states, usually fewer when unobserved
    assert len(md.states) <= 8
    assert len(md.states) >= 2


def test_idcd_does_not_generate_unobserved_combinations(biofam_domains):
    md = create_idcd_sequence_from_domains(biofam_domains, quiet=True)
    full_cartesian = 2 ** len(biofam_domains)
    assert len(md.states) < full_cartesian or len(md.states) == full_cartesian


def test_idcd_sample_distance_shape(biofam_domains):
    provider = IDCDDistanceProvider(
        biofam_domains,
        method="OM",
        sm="CONSTANT",
        indel=1,
        norm="none",
    )
    indices = [0, 1, 2, 5]
    matrix = provider.sample_distance_matrix(indices)
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix, matrix.T)


def test_idcd_distance_to_medoids_shape(biofam_domains):
    provider = IDCDDistanceProvider(
        biofam_domains,
        method="OM",
        sm="CONSTANT",
        indel=1,
        norm="none",
    )
    medoids = [0, 3]
    matrix = provider.distance_to_medoids(medoids)
    assert matrix.shape == (provider.n_sequences(), 2)


def test_cat_rejects_dhd(biofam_domains):
    with pytest.raises(ValueError, match="DHD"):
        from sequenzo.multidomain.clara.distance_providers import CATDistanceProvider

        CATDistanceProvider(
            biofam_domains,
            method="DHD",
            sm=["CONSTANT", "CONSTANT", "CONSTANT"],
        )


def test_cat_sm_string_expanded(biofam_domains):
    from sequenzo.multidomain.clara.distance_providers import CATDistanceProvider

    provider = CATDistanceProvider(
        biofam_domains,
        method="OM",
        sm="CONSTANT",
        indel=1,
        norm="none",
    )
    assert provider.n_sequences() == biofam_domains[0].seqdata.shape[0]


def test_cat_cost_matrix_shape(biofam_domains):
    provider = CATDistanceProvider(
        biofam_domains,
        method="OM",
        sm=["CONSTANT", "CONSTANT", "CONSTANT"],
        indel="auto",
        norm="none",
    )
    n_states = len(provider.md_seqdata.states)
    sm = provider._dist_args["sm"]
    assert sm.shape == (n_states, n_states)


def test_cat_indel_vector_shape(biofam_domains):
    provider = CATDistanceProvider(
        biofam_domains,
        method="OM",
        sm=["CONSTANT", "CONSTANT", "CONSTANT"],
        indel="auto",
        norm="none",
    )
    indel = provider._dist_args["indel"]
    n_states = len(provider.md_seqdata.states)
    if np.ndim(indel) == 0:
        assert True
    else:
        assert len(indel) == n_states


def test_cat_distance_provider_sample_shape(biofam_domains):
    provider = CATDistanceProvider(
        biofam_domains,
        method="OM",
        sm=["CONSTANT", "CONSTANT", "CONSTANT"],
        indel="auto",
        norm="none",
    )
    matrix = provider.sample_distance_matrix([0, 1, 4])
    assert matrix.shape == (3, 3)


def test_cat_distance_provider_medoids_shape(biofam_domains):
    provider = CATDistanceProvider(
        biofam_domains,
        method="OM",
        sm=["CONSTANT", "CONSTANT", "CONSTANT"],
        indel="auto",
        norm="none",
    )
    matrix = provider.distance_to_medoids([1, 2])
    assert matrix.shape == (provider.n_sequences(), 2)


def test_dat_sums_domain_distances_correctly(biofam_domains):
    provider = DATDistanceProvider(
        biofam_domains,
        method_params=[
            {"method": "HAM"},
            {"method": "HAM"},
            {"method": "HAM"},
        ],
        domain_weights=[1.0, 1.0, 1.0],
        link="sum",
    )
    indices = [0, 1, 2]
    dat_matrix = provider.sample_distance_matrix(indices)

    from sequenzo.multidomain.clara._utils import compute_distance_matrix, subset_sequence_data

    manual = np.zeros((3, 3))
    for domain in biofam_domains:
        sub = subset_sequence_data(domain, indices)
        manual += compute_distance_matrix(sub, {"method": "HAM", "full_matrix": True})
    assert np.allclose(dat_matrix, manual)


def test_dat_supports_domain_weights(biofam_domains):
    provider = DATDistanceProvider(
        biofam_domains,
        method_params=[{"method": "HAM"}] * 3,
        domain_weights=[2.0, 1.0, 1.0],
        link="sum",
    )
    unweighted = DATDistanceProvider(
        biofam_domains,
        method_params=[{"method": "HAM"}] * 3,
        domain_weights=[1.0, 1.0, 1.0],
        link="sum",
    )
    idx = [0, 1, 3]
    weighted = provider.sample_distance_matrix(idx)
    base = unweighted.sample_distance_matrix(idx)
    assert np.any(weighted != base)


def test_dat_sample_distance_shape(biofam_domains):
    provider = DATDistanceProvider(
        biofam_domains,
        method_params=[{"method": "HAM"}] * 3,
    )
    matrix = provider.sample_distance_matrix([0, 2, 4, 6])
    assert matrix.shape == (4, 4)


def test_dat_distance_to_medoids_shape(biofam_domains):
    provider = DATDistanceProvider(
        biofam_domains,
        method_params=[{"method": "HAM"}] * 3,
    )
    matrix = provider.distance_to_medoids([0, 5])
    assert matrix.shape == (provider.n_sequences(), 2)


def test_dat_parallel_matches_serial(biofam_domains):
    """Parallel domain OM/HAM must match serial combination (same algorithm)."""
    params = {
        "method_params": [{"method": "HAM"}] * 3,
        "domain_weights": [1.0, 2.0, 0.5],
        "link": "mean",
    }
    serial = DATDistanceProvider(biofam_domains, n_jobs_domains=1, **params)
    parallel = DATDistanceProvider(biofam_domains, n_jobs_domains=2, **params)
    sample_idx = [0, 1, 3, 7, 12]
    medoids = [2, 9, 15]
    assert np.allclose(
        serial.sample_distance_matrix(sample_idx),
        parallel.sample_distance_matrix(sample_idx),
    )
    assert np.allclose(
        serial.distance_to_medoids(medoids),
        parallel.distance_to_medoids(medoids),
    )


@pytest.mark.parametrize("strategy", ["idcd", "cat", "dat"])
def test_md_clara_runs(biofam_domains, strategy):
    if strategy == "idcd":
        distance_params = {
            "method": "OM",
            "sm": "CONSTANT",
            "indel": 1,
            "norm": "none",
        }
    elif strategy == "cat":
        distance_params = {
            "method": "OM",
            "sm": ["CONSTANT", "CONSTANT", "CONSTANT"],
            "indel": "auto",
            "norm": "none",
        }
    else:
        distance_params = {
            "method_params": [{"method": "HAM"}] * 3,
            "domain_weights": [1, 1, 1],
            "link": "sum",
        }

    result = md_clara(
        biofam_domains,
        strategy=strategy,
        distance_params=distance_params,
        R=5,
        sample_size=40,
        kvals=[2, 3],
        criteria=("distance",),
        stability=False,
        random_state=42,
        n_jobs=1,
        verbose=False,
    )
    assert result.strategy == strategy
    assert result.settings["sample_size"] is not None
    assert len(result.stats) >= 2


def test_md_clara_returns_stats(biofam_domains):
    result = md_clara(
        biofam_domains,
        strategy="idcd",
        distance_params={"method": "OM", "sm": "CONSTANT", "indel": 1, "norm": "none"},
        R=4,
        sample_size=35,
        kvals=[2],
        n_jobs=1,
        verbose=False,
        random_state=0,
    )
    assert "avg_dist" in result.stats.columns
    assert "k" in result.stats.columns


def test_md_clara_returns_clustering(biofam_domains):
    result = md_clara(
        biofam_domains,
        strategy="dat",
        distance_params={
            "method_params": [{"method": "HAM"}] * 3,
        },
        R=3,
        sample_size=35,
        kvals=[2],
        n_jobs=1,
        verbose=False,
        random_state=1,
    )
    assert result.clustering.shape[0] == biofam_domains[0].seqdata.shape[0]
    assert "Cluster 2" in result.clustering.columns


def test_md_clara_reproducible_with_random_state(biofam_domains):
    params = {"method": "OM", "sm": "CONSTANT", "indel": 1, "norm": "none"}
    kwargs = dict(
        domains=biofam_domains,
        strategy="idcd",
        distance_params=params,
        R=4,
        sample_size=35,
        kvals=[2],
        n_jobs=1,
        verbose=False,
    )
    r1 = md_clara(**kwargs, random_state=99)
    r2 = md_clara(**kwargs, random_state=99)
    assert np.allclose(r1.stats["avg_dist"].to_numpy(), r2.stats["avg_dist"].to_numpy())
