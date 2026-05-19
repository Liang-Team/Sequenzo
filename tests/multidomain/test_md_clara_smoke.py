"""
Smoke tests for crisp MD-CLARA (IDCD, CAT, DAT) on a small empirical subset.

These mirror the manual checks recommended before calling v1 complete:
stats, clustering, medoids, best_clustering(k), and stability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData, load_dataset
from sequenzo.multidomain.clara import md_clara

_SMOKE_KWARGS = dict(
    R=5,
    sample_size=20,
    kvals=[2, 3],
    criteria=("distance",),
    stability=True,
    n_jobs=1,
    random_state=123,
    verbose=False,
)


@pytest.fixture(scope="module")
def biofam_smoke_domains():
    """Biofam three-domain subset (small N for fast smoke runs)."""
    left_df = load_dataset("biofam_left_domain").head(80)
    children_df = load_dataset("biofam_child_domain").head(80)
    married_df = load_dataset("biofam_married_domain").head(80)
    time_cols = [c for c in children_df.columns if c != "id"]

    return [
        SequenceData(
            left_df,
            time=time_cols,
            id_col="id",
            states=[0, 1],
            labels=["At home", "Left home"],
        ),
        SequenceData(
            children_df,
            time=time_cols,
            id_col="id",
            states=[0, 1],
            labels=["No child", "Child"],
        ),
        SequenceData(
            married_df,
            time=time_cols,
            id_col="id",
            states=[0, 1],
            labels=["Not married", "Married"],
        ),
    ]


def _assert_smoke_result(result, *, n_sequences: int, strategy: str) -> None:
    """Shared assertions after md_clara() smoke run."""
    assert result.strategy == strategy
    assert result.method == "crisp"
    assert result.settings["sample_size"] == 20
    assert result.settings["criteria"] == ["distance"]

    stats = result.stats
    assert isinstance(stats, pd.DataFrame)
    assert not stats.empty
    for col in ("k", "avg_dist", "criterion", "ari08", "jc08"):
        assert col in stats.columns
    assert set(stats["k"].astype(int)) >= {2, 3}

    clustering = result.clustering
    assert isinstance(clustering, pd.DataFrame)
    assert clustering.shape[0] == n_sequences
    assert "Cluster 2" in clustering.columns
    assert "Cluster 3" in clustering.columns
    assert clustering.head().shape[0] <= n_sequences

    assert 2 in result.medoids
    assert 3 in result.medoids
    assert len(result.medoids[2]) >= 1
    assert len(result.medoids[3]) >= 1

    labels_k2 = result.best_clustering(2)
    assert labels_k2.shape == (n_sequences,)
    assert np.all(labels_k2 >= 1)

    assert result.stability is not None
    assert 2 in result.stability
    assert 3 in result.stability
    for k in (2, 3):
        stab = result.stability[k]
        assert "mean_ari" in stab
        assert "ari08" in stab
        assert "jc08" in stab


def test_smoke_md_clara_idcd(biofam_smoke_domains):
    domains = biofam_smoke_domains
    result = md_clara(
        domains,
        strategy="idcd",
        distance_params={
            "method": "OM",
            "sm": "CONSTANT",
            "indel": 1,
            "norm": "none",
        },
        **_SMOKE_KWARGS,
    )
    _assert_smoke_result(result, n_sequences=len(domains[0].seqdata), strategy="idcd")


def test_smoke_md_clara_cat(biofam_smoke_domains):
    domains = biofam_smoke_domains
    n_dom = len(domains)
    result = md_clara(
        domains,
        strategy="cat",
        distance_params={
            "method": "OM",
            "sm": ["CONSTANT"] * n_dom,
            "indel": 1,
            "norm": "none",
        },
        **_SMOKE_KWARGS,
    )
    _assert_smoke_result(result, n_sequences=len(domains[0].seqdata), strategy="cat")


def test_smoke_md_clara_dat(biofam_smoke_domains):
    domains = biofam_smoke_domains
    n_dom = len(domains)
    result = md_clara(
        domains,
        strategy="dat",
        distance_params={
            "method_params": [
                {"method": "OM", "sm": "CONSTANT", "indel": 1},
            ]
            * n_dom,
            "domain_weights": [1.0] * n_dom,
            "link": "sum",
        },
        **_SMOKE_KWARGS,
    )
    _assert_smoke_result(result, n_sequences=len(domains[0].seqdata), strategy="dat")
