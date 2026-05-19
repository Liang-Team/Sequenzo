"""Guards and small end-to-end runs for crisp MD-CLARA."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sequenzo.define_sequence_data import SequenceData
from sequenzo.multidomain.clara import md_clara, make_distance_provider
from sequenzo.multidomain.clara._utils import check_sample_size_for_k
from sequenzo.multidomain.idcd import create_idcd_sequence_from_domains

try:
    from simulation import generate_multidomain_sequences
except ImportError:
    generate_multidomain_sequences = None


@pytest.fixture
def tiny_domains():
    if generate_multidomain_sequences is not None:
        sim = generate_multidomain_sequences(
            n_sequences=40,
            n_domains=3,
            n_timepoints=8,
            alphabet_size=3,
            n_clusters=3,
            domain_association="high",
            noise=0.05,
            random_state=99,
        )
        return sim.domains

    time_cols = [f"t{i}" for i in range(8)]
    domains = []
    for d in range(3):
        rng = np.random.default_rng(99 + d)
        arr = rng.integers(0, 3, size=(40, 8))
        df = pd.DataFrame(arr, columns=time_cols)
        df.insert(0, "id", np.arange(1, 41))
        domains.append(
            SequenceData(
                df,
                time=time_cols,
                id_col="id",
                states=[0, 1, 2],
                labels=["0", "1", "2"],
            )
        )
    return domains


def test_check_sample_size_rejects_k_below_2():
    with pytest.raises(ValueError, match=">= 2"):
        check_sample_size_for_k(20, [1, 2, 3])


def test_provider_distance_shapes(tiny_domains):
    provider = make_distance_provider(
        tiny_domains,
        strategy="idcd",
        distance_params={
            "method": "OM",
            "sm": "CONSTANT",
            "indel": 1,
            "norm": "none",
        },
    )
    n = provider.n_sequences()
    assert provider.sample_distance_matrix([0, 1, 2]).shape == (3, 3)
    assert provider.distance_to_medoids([0, 2]).shape == (n, 2)


def test_idcd_without_id_col():
    time_cols = [f"t{i}" for i in range(6)]
    domains = []
    for d in range(2):
        arr = np.zeros((10, 6), dtype=int) + d
        df = pd.DataFrame(arr, columns=time_cols)
        domains.append(
            SequenceData(df, time=time_cols, states=[0, 1], labels=["0", "1"])
        )
    md = create_idcd_sequence_from_domains(domains, quiet=True)
    assert md.id_col == "__mdclara_id__"
    assert len(md.states) >= 1


@pytest.mark.parametrize("strategy", ["idcd", "cat", "dat"])
def test_md_clara_small_end_to_end(tiny_domains, strategy):
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
            "sm": ["CONSTANT"] * 3,
            "indel": 1,
            "norm": "none",
        }
    else:
        distance_params = {
            "method_params": [{"method": "HAM"}] * 3,
            "domain_weights": [1.0] * 3,
            "link": "sum",
        }

    result = md_clara(
        tiny_domains,
        strategy=strategy,
        distance_params=distance_params,
        R=5,
        sample_size=20,
        kvals=[2, 3],
        criteria=("distance",),
        stability=True,
        n_jobs=1,
        random_state=123,
        verbose=False,
    )

    assert not result.stats.empty
    assert "avg_dist" in result.stats.columns
    assert result.clustering.shape[0] == tiny_domains[0].seqdata.shape[0]
    assert 2 in result.medoids
    labels = result.best_clustering(2)
    assert labels.shape[0] == tiny_domains[0].seqdata.shape[0]
    assert result.settings["sample_size"] == 20
    assert result.stability is not None
