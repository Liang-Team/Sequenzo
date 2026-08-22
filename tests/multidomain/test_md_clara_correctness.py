"""Correctness freeze: query equality, aggregation weights, PBM, sampling, LODO."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import squareform

from sequenzo import SequenceData
from sequenzo.big_data.clara.utils.aggregatecases import DataFrameAggregator
from sequenzo.multidomain.clara._utils import (
    aggregate_domains,
    build_multidomain_profile_frame,
    compute_distance_matrix,
    freeze_seqdist_costs,
)
from sequenzo.multidomain.clara.clara_engine import _pbm_index
from sequenzo.multidomain.clara.diagnostics import leave_one_domain_out_sensitivity
from sequenzo.multidomain.clara.distance_providers import (
    make_aggregated_distance_provider,
)
from sequenzo.multidomain.clara.md_clara import md_clara
from sequenzo.multidomain.idcd import create_idcd_sequence_from_domains


_STATES = [0, 1, 2]


def _make_domain(rows: list[list[int]], weights=None) -> SequenceData:
    n_time = len(rows[0])
    time_cols = [f"t{i + 1}" for i in range(n_time)]
    df = pd.DataFrame(rows, columns=time_cols)
    kwargs = {"data": df, "time": time_cols, "states": _STATES}
    if weights is not None:
        kwargs["weights"] = np.asarray(weights, dtype=float)
    return SequenceData(**kwargs)


def _duplicate_heavy_domains():
    """Three unique profiles: frequencies 5, 3, 1."""
    a = [0, 0, 1, 1]
    b = [1, 1, 0, 0]
    c = [2, 0, 2, 0]
    rows1 = [a] * 5 + [b] * 3 + [c]
    rows2 = [list(reversed(r)) for r in rows1]
    return [_make_domain(rows1), _make_domain(rows2)]


def test_aggregate_domains_writes_frequency_weights():
    domains = _duplicate_heavy_domains()
    ac = DataFrameAggregator().aggregate(build_multidomain_profile_frame(domains))
    agg = aggregate_domains(domains, ac)
    expected = np.asarray(ac["aggWeights"], dtype=float)
    for domain in agg:
        assert np.allclose(np.asarray(domain.weights, dtype=float), expected)
    assert sorted(expected.tolist()) == [1.0, 3.0, 5.0]


def test_pbm_matches_studer_e1_equals_one():
    total_diss = 12.0
    d_max = 4.0
    k = 3
    expected = ((1.0 / k) * (1.0 / total_diss) * d_max) ** 2
    assert _pbm_index(k, total_diss, d_max) == pytest.approx(expected)
    assert _pbm_index(k, 0.0, d_max) == np.inf


@pytest.mark.parametrize("strategy", ["idcd", "cat", "dat"])
@pytest.mark.parametrize(
    "cost_spec",
    [
        "constant",
        "trate",
    ],
)
def test_provider_queries_match_traditional_full_matrix(strategy, cost_spec):
    rng = np.random.default_rng(7)
    n, d, t = 24, 2, 6
    domains = []
    for _ in range(d):
        rows = rng.integers(0, 3, size=(n, t)).tolist()
        domains.append(_make_domain(rows))

    if strategy == "idcd":
        if cost_spec == "constant":
            params = {"method": "OM", "sm": "CONSTANT", "indel": 1, "norm": "none"}
        else:
            params = {"method": "OM", "sm": "TRATE", "indel": 1, "norm": "none"}
    elif strategy == "cat":
        sm = "CONSTANT" if cost_spec == "constant" else "TRATE"
        params = {"method": "OM", "sm": [sm, sm], "indel": 1, "norm": "none"}
    else:
        sm = "CONSTANT" if cost_spec == "constant" else "TRATE"
        params = {
            "method_params": [
                {"method": "OM", "sm": sm, "indel": 1, "norm": "none"}
                for _ in range(d)
            ],
            "link": "sum",
        }

    ac = DataFrameAggregator().aggregate(build_multidomain_profile_frame(domains))
    provider = make_aggregated_distance_provider(domains, strategy, params, ac)
    n_star = provider.n_sequences()
    all_idx = np.arange(n_star, dtype=int)
    full = np.asarray(provider.sample_distances(all_idx, condensed=False), dtype=float)

    rng2 = np.random.default_rng(11)
    b = rng2.choice(n_star, size=min(8, n_star), replace=False)
    got_b = provider.sample_distances(b, condensed=False)
    assert np.allclose(got_b, full[np.ix_(b, b)], rtol=1e-10, atol=1e-10)

    medoids = rng2.choice(n_star, size=min(3, n_star), replace=False)
    got_m = provider.distance_to_medoids(medoids)
    assert np.allclose(got_m, full[:, medoids], rtol=1e-10, atol=1e-10)

    condensed = provider.sample_distances(b, condensed=True)
    assert np.allclose(squareform(condensed), got_b, rtol=1e-10, atol=1e-10)


def test_idcd_constant_matches_traditional_create_idcd_route():
    domains = _duplicate_heavy_domains()
    params = {"method": "OM", "sm": "CONSTANT", "indel": 1, "norm": "none"}
    ac = DataFrameAggregator().aggregate(build_multidomain_profile_frame(domains))
    provider = make_aggregated_distance_provider(domains, "idcd", params, ac)

    agg_idx = np.asarray(ac["aggIndex"], dtype=int) - 1
    md = create_idcd_sequence_from_domains(domains, quiet=True)
    traditional = compute_distance_matrix(md, params, full_matrix=True)
    traditional_agg = traditional[np.ix_(agg_idx, agg_idx)]
    via_provider = provider.sample_distances(np.arange(provider.n_sequences()), condensed=False)
    assert np.allclose(via_provider, traditional_agg, rtol=1e-10, atol=1e-10)


def test_trate_costs_use_original_case_frequencies_not_unique_profiles():
    """TRATE on frequency-weighted originals must not match unweighted unique profiles."""
    domains = _duplicate_heavy_domains()
    params = {"method": "OM", "sm": "TRATE", "indel": 1, "norm": "none"}
    ac = DataFrameAggregator().aggregate(build_multidomain_profile_frame(domains))
    agg_idx = np.asarray(ac["aggIndex"], dtype=int) - 1

    md_full = create_idcd_sequence_from_domains(domains, quiet=True)
    frozen_full = freeze_seqdist_costs(md_full, dict(params))

    unweighted_unique = aggregate_domains(domains, ac)
    # Force unit weights so this is the incorrect (pre-fix) estimator.
    for domain in unweighted_unique:
        domain.weights = np.ones(len(domain.weights), dtype=float)
    md_wrong = create_idcd_sequence_from_domains(unweighted_unique, quiet=True)
    frozen_wrong = freeze_seqdist_costs(md_wrong, dict(params))

    assert frozen_full["sm"].shape == frozen_wrong["sm"].shape
    assert not np.allclose(frozen_full["sm"], frozen_wrong["sm"])

    provider = make_aggregated_distance_provider(domains, "idcd", params, ac)
    traditional = compute_distance_matrix(md_full, frozen_full, full_matrix=True)
    traditional_agg = traditional[np.ix_(agg_idx, agg_idx)]
    via_provider = provider.sample_distances(np.arange(provider.n_sequences()), condensed=False)
    assert np.allclose(via_provider, traditional_agg, rtol=1e-10, atol=1e-10)


def test_dat_weighted_components_sum_to_combined():
    domains = _duplicate_heavy_domains()
    params = {
        "method_params": [
            {"method": "OM", "sm": "CONSTANT", "indel": 1, "norm": "none"},
            {"method": "OM", "sm": "CONSTANT", "indel": 1, "norm": "none"},
        ],
        "domain_weights": [0.5, 2.0],
        "link": "sum",
    }
    ac = DataFrameAggregator().aggregate(build_multidomain_profile_frame(domains))
    provider = make_aggregated_distance_provider(domains, "dat", params, ac)
    medoids = [0, 1]
    combined = provider.distance_to_medoids(medoids)
    parts = provider.weighted_per_domain_distance_to_medoids(medoids)
    assert np.allclose(sum(parts), combined, rtol=1e-10, atol=1e-10)


def test_lodo_two_domain_branch_is_deterministic():
    domains = _duplicate_heavy_domains()
    kwargs = dict(
        domains=domains,
        strategy="idcd",
        k=2,
        R=4,
        sample_size=3,
        distance_params={"method": "OM", "sm": "CONSTANT", "indel": 1, "norm": "none"},
        random_state=21,
        n_jobs=1,
        verbose=False,
        sampling_policy="uniform_unique_profiles",
    )
    a = leave_one_domain_out_sensitivity(**kwargs)
    b = leave_one_domain_out_sensitivity(**kwargs)
    pd.testing.assert_frame_equal(
        a.drop(columns=["full_medoids", "reduced_medoids"]),
        b.drop(columns=["full_medoids", "reduced_medoids"]),
        check_dtype=False,
    )
    assert (a["reduced_model"] == "md_clara_single_domain").all()
    assert "random_state" in a.columns
    assert "reduced_objective" in a.columns


def test_sampling_policies_change_draws_but_run():
    domains = _duplicate_heavy_domains()
    params = {"method": "OM", "sm": "CONSTANT", "indel": 1, "norm": "none"}
    common = dict(
        domains=domains,
        strategy="idcd",
        distance_params=params,
        R=3,
        sample_size=4,
        kvals=[2],
        n_jobs=1,
        verbose=False,
        random_state=3,
        subsample_diagnostics=True,
    )
    uniform = md_clara(**common, sampling_policy="uniform_unique_profiles")
    case = md_clara(**common, sampling_policy="case_sample_then_aggregate")
    pps = md_clara(**common, sampling_policy="frequency_proportional_profiles")
    studer = md_clara(**common, sampling_policy="studer_bootstrap")
    alias = md_clara(
        **common, sampling_policy="frequency_proportional_with_replacement"
    )
    assert uniform.settings["sampling_policy"] == "uniform_unique_profiles"
    assert case.settings["sampling_policy"] == "case_sample_then_aggregate"
    assert pps.settings["sampling_policy"] == "frequency_proportional_profiles"
    assert studer.settings["sampling_policy"] == "studer_bootstrap"
    assert alias.settings["sampling_policy"] == "studer_bootstrap"
    assert uniform.best_clustering(2).shape[0] == 9
    assert case.subsample_diagnostics["sampled_profiles"].max() <= 3
    # With replacement, unique profiles in a draw of b=4 from 3 profiles
    # cannot exceed N*, and PAM weights are multiplicities summing to b.
    studer_diag = studer.subsample_diagnostics
    assert studer_diag["sampled_profiles"].max() <= 3
    assert np.allclose(studer_diag["subsample_pam_weight_sum"], 4.0)


def test_studer_bootstrap_uses_draw_counts_not_full_frequencies():
    """seqclararange: p_u ∝ a_u, replace=True, subsample PAM weights = counts."""
    from sequenzo.multidomain.clara.clara_engine import (
        _draw_profile_sample,
        _unique_with_counts_first_occurrence,
    )

    draws = np.array([2, 0, 2, 1, 0])
    uniq, counts = _unique_with_counts_first_occurrence(draws)
    np.testing.assert_array_equal(uniq, [2, 0, 1])
    np.testing.assert_array_equal(counts, [2.0, 2.0, 1.0])

    weights = np.array([100.0, 10.0, 1.0])
    rng = np.random.default_rng(0)
    indices, pam_w = _draw_profile_sample(
        rng,
        sampling_policy="studer_bootstrap",
        n_profiles=3,
        sample_size=40,
        profile_weights=weights,
        disagg=np.arange(111),
        max_k=2,
    )
    assert indices.size <= 3
    assert indices.size >= 2
    assert pam_w.shape == indices.shape
    assert np.isclose(pam_w.sum(), 40.0)
    # Draw multiplicities are not the full-data frequencies.
    assert not np.allclose(pam_w, weights[indices])
    # The common profile should appear in almost every bootstrap of size 40.
    assert 0 in set(indices.tolist())

    uniform_idx, uniform_w = _draw_profile_sample(
        np.random.default_rng(0),
        sampling_policy="uniform_unique_profiles",
        n_profiles=3,
        sample_size=2,
        profile_weights=weights,
        disagg=np.arange(111),
        max_k=2,
    )
    np.testing.assert_array_equal(uniform_w, weights[uniform_idx])


def test_studer_bootstrap_is_a_single_draw():
    from sequenzo.multidomain.clara.clara_engine import _draw_profile_sample

    class _Once:
        def __init__(self):
            self.calls = 0

        def choice(self, n, size, replace, p):
            self.calls += 1
            return np.zeros(size, dtype=int)

    rng = _Once()
    with pytest.raises(ValueError, match="seqclararange does not redraw"):
        _draw_profile_sample(
            rng,
            sampling_policy="studer_bootstrap",
            n_profiles=3,
            sample_size=5,
            profile_weights=np.array([1.0, 1.0, 1.0]),
            disagg=np.arange(3),
            max_k=3,
        )
    assert rng.calls == 1


def test_studer_bootstrap_matches_manual_choice_and_counts():
    from sequenzo.multidomain.clara.clara_engine import (
        _draw_profile_sample,
        _unique_with_counts_first_occurrence,
    )

    weights = np.array([8.0, 2.0, 1.0])
    seed = 17
    b = 20
    expected_rng = np.random.default_rng(seed)
    p = weights / weights.sum()
    draws = expected_rng.choice(3, size=b, replace=True, p=p)
    exp_idx, exp_w = _unique_with_counts_first_occurrence(draws)
    got_idx, got_w = _draw_profile_sample(
        np.random.default_rng(seed),
        sampling_policy="studer_bootstrap",
        n_profiles=3,
        sample_size=b,
        profile_weights=weights,
        disagg=np.arange(11),
        max_k=2,
    )
    np.testing.assert_array_equal(got_idx, exp_idx)
    np.testing.assert_array_equal(got_w, exp_w)


def test_max_draw_multiplicity_is_one_for_uniform_not_frequency():
    from sequenzo.multidomain.clara.clara_engine import _subsample_coverage_row

    weights = np.array([200.0, 3.0, 1.0])
    uniform = _subsample_coverage_row(
        repetition=0,
        local_indices=np.array([0, 2]),
        profile_weights=weights,
        rare_profile_threshold=5,
        sampling_policy="uniform_unique_profiles",
        requested_sample_size=2,
        subsample_pam_weights=weights[[0, 2]],
    )
    assert uniform["max_draw_multiplicity"] == 1.0
    assert uniform["max_subsample_pam_weight"] == 200.0

    studer = _subsample_coverage_row(
        repetition=0,
        local_indices=np.array([0, 2]),
        profile_weights=weights,
        rare_profile_threshold=5,
        sampling_policy="studer_bootstrap",
        requested_sample_size=5,
        subsample_pam_weights=np.array([4.0, 1.0]),
    )
    assert studer["max_draw_multiplicity"] == 4.0
    assert studer["max_subsample_pam_weight"] == 4.0
    assert studer["subsample_pam_weight_sum"] == 5.0


def test_studer_schedule_requires_multiplicities():
    domains = _duplicate_heavy_domains()
    params = {"method": "OM", "sm": "CONSTANT", "indel": 1, "norm": "none"}
    with pytest.raises(ValueError, match="multiplicities"):
        md_clara(
            domains,
            strategy="idcd",
            distance_params=params,
            R=1,
            sample_size=4,
            kvals=[2],
            n_jobs=1,
            verbose=False,
            sampling_policy="studer_bootstrap",
            subsample_schedule=[[0, 1]],
        )


def test_studer_schedule_dict_preserves_counts():
    domains = _duplicate_heavy_domains()
    params = {"method": "OM", "sm": "CONSTANT", "indel": 1, "norm": "none"}
    schedule = [{"indices": np.array([0, 1]), "pam_weights": np.array([3.0, 1.0])}]
    result = md_clara(
        domains,
        strategy="idcd",
        distance_params=params,
        R=1,
        sample_size=4,
        kvals=[2],
        n_jobs=1,
        verbose=False,
        subsample_diagnostics=True,
        sampling_policy="studer_bootstrap",
        subsample_schedule=schedule,
    )
    diag = result.subsample_diagnostics
    assert int(diag["sampled_profiles"].iloc[0]) == 2
    assert float(diag["subsample_pam_weight_sum"].iloc[0]) == 4.0


def test_constant_auto_indel_is_not_frequency_dependent():
    from sequenzo.multidomain.clara._utils import distance_params_are_data_dependent

    assert not distance_params_are_data_dependent(
        "idcd", {"method": "OM", "sm": "CONSTANT", "indel": "auto"}
    )
    assert distance_params_are_data_dependent(
        "idcd", {"method": "OM", "sm": "TRATE", "indel": "auto"}
    )
    sm = np.array([[0.0, 2.0, 2.0], [2.0, 0.0, 2.0], [2.0, 2.0, 0.0]])
    frozen = freeze_seqdist_costs(
        _make_domain([[0, 1], [1, 0]]),
        {"method": "OM", "sm": sm, "indel": "auto", "norm": "none"},
    )
    assert frozen["indel"] == pytest.approx(1.0)
    const = freeze_seqdist_costs(
        _make_domain([[0, 1], [1, 0]]),
        {"method": "OM", "sm": "CONSTANT", "indel": "auto", "norm": "none"},
    )
    assert not (isinstance(const["indel"], str) and const["indel"] == "auto")


def test_explicit_subsample_schedule_is_honored():
    domains = _duplicate_heavy_domains()
    params = {"method": "OM", "sm": "CONSTANT", "indel": 1, "norm": "none"}
    schedule = [[0, 1], [0, 2], [1, 2]]
    result = md_clara(
        domains,
        strategy="idcd",
        distance_params=params,
        R=3,
        sample_size=2,
        kvals=[2],
        n_jobs=1,
        verbose=False,
        subsample_diagnostics=True,
        subsample_schedule=schedule,
        sampling_policy="uniform_unique_profiles",
        random_state=0,
    )
    sampled = result.subsample_diagnostics["sampled_profiles"].to_numpy()
    assert np.all(sampled == 2)


def test_n_jobs_one_matches_n_jobs_two_on_tiny_data():
    domains = _duplicate_heavy_domains()
    params = {"method": "OM", "sm": "CONSTANT", "indel": 1, "norm": "none"}
    kwargs = dict(
        domains=domains,
        strategy="dat",
        distance_params={
            "method_params": [dict(params), dict(params)],
            "link": "sum",
        },
        R=4,
        sample_size=3,
        kvals=[2],
        verbose=False,
        random_state=8,
        sampling_policy="uniform_unique_profiles",
    )
    serial = md_clara(**kwargs, n_jobs=1)
    parallel = md_clara(**kwargs, n_jobs=2)
    assert np.array_equal(serial.best_clustering(2), parallel.best_clustering(2))
    assert np.array_equal(serial.medoids[2], parallel.medoids[2])
