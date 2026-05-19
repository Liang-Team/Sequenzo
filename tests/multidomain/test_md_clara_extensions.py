"""Tests for MD-CLARA extensions (simulation, single-criterion guard)."""

from __future__ import annotations

import numpy as np
import pytest

from sequenzo.multidomain.clara.md_clara import md_clara

try:
    from simulation import generate_multidomain_sequences, run_md_clara_benchmark
except ImportError:  # pragma: no cover
    generate_multidomain_sequences = None
    run_md_clara_benchmark = None

pytestmark = pytest.mark.skipif(
    generate_multidomain_sequences is None,
    reason="multidomain_clara/simulation not on PYTHONPATH",
)


@pytest.fixture
def small_sim():
    return generate_multidomain_sequences(
        n_sequences=120,
        n_domains=3,
        n_timepoints=10,
        alphabet_size=3,
        n_clusters=3,
        domain_association="high",
        noise=0.05,
        random_state=1,
    )


def test_generate_multidomain_sequences_shape(small_sim):
    assert len(small_sim.domains) == 3
    assert small_sim.true_labels.shape == (120,)
    assert small_sim.domains[0].seqdata.shape == (120, 10)


def test_md_clara_rejects_multiple_criteria(small_sim):
    with pytest.raises(ValueError, match="exactly one clustering criterion"):
        md_clara(
            small_sim.domains,
            strategy="idcd",
            distance_params={"method": "OM", "sm": "CONSTANT", "indel": 1, "norm": "none"},
            criteria=("distance", "db"),
            R=2,
            sample_size=40,
            kvals=[2],
            n_jobs=1,
            verbose=False,
        )


def test_md_clara_rejects_fuzzy_method(small_sim):
    with pytest.raises(ValueError, match="method='crisp' only"):
        md_clara(
            small_sim.domains,
            strategy="dat",
            distance_params={"method_params": [{"method": "HAM"}] * 3},
            method="fuzzy",
            R=2,
            sample_size=40,
            kvals=[2],
            n_jobs=1,
            verbose=False,
        )


def test_run_md_clara_benchmark_small_grid():
    df = run_md_clara_benchmark(
        n_sequences_grid=(80,),
        n_domains_grid=(2,),
        strategies=("idcd", "dat"),
        R=2,
        sample_size=40,
        kvals=[2],
        n_jobs=1,
        random_state=0,
    )
    assert not df.empty
    assert df["ok"].all()
    assert "runtime_seconds" in df.columns


def test_stability_populated(small_sim):
    result = md_clara(
        small_sim.domains,
        strategy="dat",
        distance_params={"method_params": [{"method": "HAM"}] * 3},
        R=5,
        sample_size=50,
        kvals=[2],
        stability=True,
        n_jobs=1,
        verbose=False,
        random_state=2,
    )
    assert result.stability is not None
    assert 2 in result.stability
    assert "mean_ari" in result.stability[2]
    assert result.settings["sample_size"] is not None
