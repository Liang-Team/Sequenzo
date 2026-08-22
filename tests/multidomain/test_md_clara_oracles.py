"""Independent-oracle checks: traditional IDCD/CAT/DAT vs MD-CLARA providers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData
from sequenzo.big_data.clara.utils.aggregatecases import DataFrameAggregator
from sequenzo.multidomain.cat import compute_cat_distance_matrix
from sequenzo.multidomain.clara._utils import (
    build_multidomain_profile_frame,
    compute_distance_matrix,
)
from sequenzo.multidomain.clara.distance_providers import make_aggregated_distance_provider
from sequenzo.multidomain.dat import compute_dat_distance_matrix
from sequenzo.multidomain.idcd import create_idcd_sequence_from_domains


_STATES = [0, 1, 2]


def _make_domain(rows, weights=None) -> SequenceData:
    n_time = len(rows[0])
    time_cols = [f"t{i + 1}" for i in range(n_time)]
    df = pd.DataFrame(rows, columns=time_cols)
    kwargs = {"data": df, "time": time_cols, "states": _STATES}
    if weights is not None:
        kwargs["weights"] = np.asarray(weights, dtype=float)
    return SequenceData(**kwargs)


def _two_domains(n=18, t=5, seed=4):
    rng = np.random.default_rng(seed)
    rows1 = rng.integers(0, 3, size=(n, t)).tolist()
    rows2 = rng.integers(0, 3, size=(n, t)).tolist()
    return [_make_domain(rows1), _make_domain(rows2)]


def _as_square(matrix) -> np.ndarray:
    if isinstance(matrix, pd.DataFrame):
        return matrix.to_numpy(dtype=float)
    return np.asarray(matrix, dtype=float)


def _agg_view(domains):
    ac = DataFrameAggregator().aggregate(build_multidomain_profile_frame(domains))
    agg_idx = np.asarray(ac["aggIndex"], dtype=int) - 1
    return ac, agg_idx


@pytest.mark.parametrize(
    "params",
    [
        {"method": "OM", "sm": "CONSTANT", "indel": 1, "norm": "none"},
        {"method": "OM", "sm": "INDELSLOG", "indel": "auto", "norm": "none"},
    ],
)
def test_idcd_provider_matches_combined_sequence_oracle(params):
    domains = _two_domains()
    ac, agg_idx = _agg_view(domains)
    md = create_idcd_sequence_from_domains(domains, quiet=True)
    traditional = compute_distance_matrix(md, params, full_matrix=True)
    expected = traditional[np.ix_(agg_idx, agg_idx)]
    provider = make_aggregated_distance_provider(domains, "idcd", params, ac)
    got = provider.sample_distances(np.arange(provider.n_sequences()), condensed=False)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("cweight", [None, [1.0, 0.5]])
def test_cat_provider_matches_traditional_diss_oracle(cweight):
    domains = _two_domains()
    ac, agg_idx = _agg_view(domains)
    params = {
        "method": "OM",
        "sm": ["CONSTANT", "CONSTANT"],
        "indel": 1,
        "norm": "none",
    }
    if cweight is not None:
        params["cweight"] = cweight
    traditional = _as_square(
        compute_cat_distance_matrix(domains, what="diss", **params)
    )
    expected = traditional[np.ix_(agg_idx, agg_idx)]
    provider = make_aggregated_distance_provider(domains, "cat", params, ac)
    got = provider.sample_distances(np.arange(provider.n_sequences()), condensed=False)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("link", ["sum", "mean"])
@pytest.mark.parametrize("weights", [None, [1.0, 2.0]])
def test_dat_provider_matches_domain_sum_oracle(link, weights):
    domains = _two_domains()
    ac, agg_idx = _agg_view(domains)
    method_params = [
        {"method": "OM", "sm": "CONSTANT", "indel": 1, "norm": "none"}
        for _ in domains
    ]
    raw = [
        _as_square(compute_distance_matrix(domain, block, full_matrix=True))
        for domain, block in zip(domains, method_params)
    ]
    w = np.array(weights if weights is not None else [1.0, 1.0], dtype=float)
    stacked = sum(wi * mat for wi, mat in zip(w, raw))
    if link == "mean":
        stacked = stacked / float(np.sum(w))
    expected = stacked[np.ix_(agg_idx, agg_idx)]
    params = {"method_params": method_params, "domain_weights": weights, "link": link}
    provider = make_aggregated_distance_provider(domains, "dat", params, ac)
    got = provider.sample_distances(np.arange(provider.n_sequences()), condensed=False)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


def test_dat_unweighted_sum_matches_legacy_full_dat():
    domains = _two_domains()
    ac, agg_idx = _agg_view(domains)
    method_params = [
        {"method": "OM", "sm": "CONSTANT", "indel": 1, "norm": "none"}
        for _ in domains
    ]
    legacy = _as_square(compute_dat_distance_matrix(domains, method_params=method_params))
    expected = legacy[np.ix_(agg_idx, agg_idx)]
    provider = make_aggregated_distance_provider(
        domains, "dat", {"method_params": method_params, "link": "sum"}, ac
    )
    got = provider.sample_distances(np.arange(provider.n_sequences()), condensed=False)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)
