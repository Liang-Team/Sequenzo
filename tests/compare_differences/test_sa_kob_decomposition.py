"""
Tests for Sequence Analysis – Kitagawa-Oaxaca-Blinder (SA-KOB) API.
"""

import warnings

import numpy as np
import pytest

from sequenzo.decomposition import (
    build_cluster_covariates,
    cluster_group_composition_table,
    detect_cluster_coefficient_owners,
    get_sa_kob_decomposition,
    get_sa_kob_decomposition_bootstrap,
)


def _make_sa_kob_data(seed=42):
    rng = np.random.default_rng(seed)
    n0, n1 = 100, 100
    labels0 = rng.integers(1, 5, size=n0)
    labels1 = rng.integers(1, 5, size=n1)
    cluster_labels = np.concatenate([labels0, labels1])
    group = np.array(["men"] * n0 + ["women"] * n1)

    cluster_labels[group == "women"] = np.where(
        np.arange(n1) % 3 == 0,
        4,
        cluster_labels[group == "women"],
    )

    cluster_effect = {1: 0.0, 2: 0.5, 3: 1.0, 4: 1.5}
    y0 = np.array([cluster_effect[c] for c in labels0]) + 10.0 + rng.normal(scale=0.1, size=n0)
    y1 = np.array([cluster_effect[c] for c in labels1]) + 8.0 + rng.normal(scale=0.1, size=n1)
    y = np.concatenate([y0, y1])
    return y, group, cluster_labels


def test_build_cluster_covariates_internal_ids_and_string_labels():
    labels = np.array(["A", "B", "A", "C"])
    cov = build_cluster_covariates(
        labels,
        categories=["A", "B", "C"],
        reference_cluster_label="A",
    )
    assert cov.X.shape == (4, 2)
    assert list(cov.category_ids) == [1, 2]
    assert list(cov.column_labels) == ["B", "C"]
    assert cov.reference_category_id == 0
    assert cov.reference_label == "A"
    assert cov.k == 3


def test_build_cluster_covariates_rejects_single_cluster():
    labels = np.array([1, 1, 1, 1])
    with pytest.raises(ValueError, match="At least two clusters"):
        build_cluster_covariates(labels, k=1, categories=[1])


def test_build_cluster_covariates_rejects_invalid_reference_index():
    labels = np.array([1, 2, 3, 1])
    with pytest.raises(ValueError, match="reference index"):
        build_cluster_covariates(labels, k=3, reference_category_index=5)


def test_cluster_group_composition_row_shares_sum_to_one():
    y, group, cluster_labels = _make_sa_kob_data()
    table = cluster_group_composition_table(
        group,
        cluster_labels,
        group0_value="men",
        group1_value="women",
    )
    assert np.isclose(table["row_share_group0"].sum(), 1.0, atol=1e-10)
    assert np.isclose(table["row_share_group1"].sum(), 1.0, atol=1e-10)


def test_detect_cluster_coefficient_owners_majority_rule_all_clusters():
    n0, n1 = 80, 80
    group = np.array(["men"] * n0 + ["women"] * n1)
    cluster_labels = np.concatenate([np.full(n0, 2), np.full(n1, 4)])
    category_id_to_label = {0: 1, 1: 2, 2: 3, 3: 4}

    owner_table, owners = detect_cluster_coefficient_owners(
        group,
        cluster_labels,
        k=4,
        category_id_to_label=category_id_to_label,
        group0_value="men",
        group1_value="women",
        majority_gap_threshold=50.0,
        neutral_cluster_owner=0,
    )
    cluster4_row = owner_table.loc[owner_table["cluster"] == 4].iloc[0]
    assert cluster4_row["coefficient_owner"] == 1
    assert owners.shape[0] == 4


def test_sa_kob_decomposition_identity_and_full_by_cluster():
    y, group, cluster_labels = _make_sa_kob_data()
    result = get_sa_kob_decomposition(
        y=y,
        group=group,
        cluster_labels=cluster_labels,
        k=4,
        categories=[1, 2, 3, 4],
        reference_category_index=0,
        group0_value="men",
        group1_value="women",
        cluster_coefficient_reference="majority",
        majority_gap_threshold=50.0,
        neutral_cluster_owner=0,
    )

    recon = result.explained + result.unexplained_returns + result.unexplained_intercept
    assert np.isclose(result.total_gap, recon, atol=1e-6)
    assert result.by_cluster.shape[0] == 4
    assert result.cluster_owners.shape[0] == 4
    assert result.by_cluster["is_reference_category"].any()
    assert result.cluster_owners["is_reference_category"].notna().all()
    assert int(result.cluster_owners["is_reference_category"].sum()) == 1
    assert np.isclose(
        result.by_cluster["explained"].sum(),
        result.by_term.loc[result.by_term["term_id"] == 0, "explained"].iloc[0],
        atol=1e-6,
    )


def test_sa_kob_with_controls():
    rng = np.random.default_rng(1)
    y, group, cluster_labels = _make_sa_kob_data()
    controls = rng.normal(size=(len(y), 1))
    y = y + 0.5 * controls[:, 0]

    result = get_sa_kob_decomposition(
        y=y,
        group=group,
        cluster_labels=cluster_labels,
        X_controls=controls,
        control_variable_names=["cohort"],
        k=4,
        categories=[1, 2, 3, 4],
        group0_value="men",
        group1_value="women",
    )
    assert result.by_cluster.shape[0] == 4
    assert result.by_column.shape[0] == 3 + 1


def test_sa_kob_drop_missing_raises_without_flag():
    y = np.array([1.0, np.nan, 3.0, 4.0])
    group = np.array(["men", "men", "women", "women"])
    clusters = np.array([1, 1, 2, 2])
    with pytest.raises(ValueError, match="missing values"):
        get_sa_kob_decomposition(
            y=y,
            group=group,
            cluster_labels=clusters,
            k=2,
            categories=[1, 2],
        )


def test_sa_kob_silhouette_filter_excludes_low_fit():
    y, group, cluster_labels = _make_sa_kob_data()
    rng = np.random.default_rng(0)
    silhouette = rng.uniform(0.3, 0.7, size=len(y))

    full = get_sa_kob_decomposition(
        y=y,
        group=group,
        cluster_labels=cluster_labels,
        k=4,
        categories=[1, 2, 3, 4],
        group0_value="men",
        group1_value="women",
    )
    filtered = get_sa_kob_decomposition(
        y=y,
        group=group,
        cluster_labels=cluster_labels,
        k=4,
        categories=[1, 2, 3, 4],
        group0_value="men",
        group1_value="women",
        silhouette=silhouette,
        silhouette_threshold=0.5,
    )
    assert filtered.diagnostics["group0"]["n"] < full.diagnostics["group0"]["n"]
    assert filtered.diagnostics["group1"]["n"] < full.diagnostics["group1"]["n"]


def test_sa_kob_silhouette_threshold_without_drop_missing():
    y, group, cluster_labels = _make_sa_kob_data()
    rng = np.random.default_rng(0)
    silhouette = rng.uniform(0.3, 0.7, size=len(y))

    full = get_sa_kob_decomposition(
        y=y,
        group=group,
        cluster_labels=cluster_labels,
        k=4,
        categories=[1, 2, 3, 4],
        group0_value="men",
        group1_value="women",
    )
    filtered = get_sa_kob_decomposition(
        y=y,
        group=group,
        cluster_labels=cluster_labels,
        k=4,
        categories=[1, 2, 3, 4],
        group0_value="men",
        group1_value="women",
        silhouette=silhouette,
        silhouette_threshold=0.5,
        drop_missing=False,
    )
    assert filtered.diagnostics["group0"]["n"] < full.diagnostics["group0"]["n"]
    assert filtered.diagnostics["group1"]["n"] < full.diagnostics["group1"]["n"]


def test_build_cluster_covariates_rejects_duplicate_categories():
    labels = np.array([1, 2, 1, 2])
    with pytest.raises(ValueError, match="categories must be unique"):
        build_cluster_covariates(labels, categories=[1, 1, 2])


def test_bootstrap_resolves_deprecated_reference_params():
    y, group, cluster_labels = _make_sa_kob_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        boot = get_sa_kob_decomposition_bootstrap(
            y=y,
            group=group,
            cluster_labels=cluster_labels,
            k=4,
            categories=[1, 2, 3, 4],
            group0_value="men",
            group1_value="women",
            cluster_reference_mode="group1",
            reference="pooled",
            n_boot=15,
            random_state=0,
            recompute_owners_each_draw=True,
        )
    assert (boot.point_estimate.cluster_owners["coefficient_owner"] == 1).all()


def test_sa_kob_common_support_warning():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    group = np.array(["men", "men", "women", "women"])
    cluster_labels = np.array([1, 2, 3, 3])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = get_sa_kob_decomposition(
            y=y,
            group=group,
            cluster_labels=cluster_labels,
            k=3,
            categories=[1, 2, 3],
            group0_value="men",
            group1_value="women",
            warn_common_support=True,
        )
        assert any("Common support" in str(w.message) for w in caught)
    assert "common_support_table" in result.diagnostics["sa_kob"]


def test_sa_kob_bootstrap_runs_with_fixed_categories():
    y, group, cluster_labels = _make_sa_kob_data()
    boot = get_sa_kob_decomposition_bootstrap(
        y=y,
        group=group,
        cluster_labels=cluster_labels,
        k=4,
        categories=[1, 2, 3, 4],
        group0_value="men",
        group1_value="women",
        n_boot=30,
        random_state=0,
    )
    assert boot.n_boot == 30
    assert boot.standard_errors["total_gap"] >= 0.0
    assert boot.by_cluster_standard_errors.shape[0] == boot.point_estimate.by_cluster.shape[0]


def test_sa_kob_bootstrap_rejects_n_boot_lt_2():
    y, group, cluster_labels = _make_sa_kob_data()
    with pytest.raises(ValueError, match="n_boot must be at least 2"):
        get_sa_kob_decomposition_bootstrap(
            y=y,
            group=group,
            cluster_labels=cluster_labels,
            k=4,
            categories=[1, 2, 3, 4],
            n_boot=1,
        )


def test_invalid_cluster_coefficient_reference_raises():
    y, group, cluster_labels = _make_sa_kob_data()
    with pytest.raises(ValueError, match="cluster_coefficient_reference"):
        get_sa_kob_decomposition(
            y=y,
            group=group,
            cluster_labels=cluster_labels,
            k=4,
            categories=[1, 2, 3, 4],
            cluster_coefficient_reference="invalid",
        )


def test_invalid_fallback_reference_raises():
    y, group, cluster_labels = _make_sa_kob_data()
    with pytest.raises(ValueError, match="fallback_reference"):
        get_sa_kob_decomposition(
            y=y,
            group=group,
            cluster_labels=cluster_labels,
            k=4,
            categories=[1, 2, 3, 4],
            fallback_reference="male",
        )


def test_invalid_cluster_owner_override_raises():
    y, group, cluster_labels = _make_sa_kob_data()
    with pytest.raises(ValueError, match="owner_overrides must map"):
        detect_cluster_coefficient_owners(
            group,
            cluster_labels,
            k=3,
            category_id_to_label={0: 1, 1: 2, 2: 3},
            group0_value="men",
            group1_value="women",
            owner_overrides={1: 2},
        )


def _minimal_sa_kob_for_categories(categories, seed=0):
    rng = np.random.default_rng(seed)
    k = len(categories)
    n0, n1 = 40, 40
    group = np.array(["men"] * n0 + ["women"] * n1)
    labels0 = rng.choice(categories, size=n0)
    labels1 = rng.choice(categories, size=n1)
    cluster_labels = np.concatenate([labels0, labels1])
    y = rng.normal(size=n0 + n1)
    return y, group, cluster_labels, k


@pytest.mark.parametrize(
    "categories",
    [
        [0, 1, 2],
        [1, 2, 3],
        ["A", "B", "C"],
    ],
)
def test_by_cluster_returns_all_k_for_label_types(categories):
    y, group, cluster_labels, k = _minimal_sa_kob_for_categories(categories)
    result = get_sa_kob_decomposition(
        y=y,
        group=group,
        cluster_labels=cluster_labels,
        k=k,
        categories=categories,
        reference_category_index=0,
        group0_value="men",
        group1_value="women",
    )
    assert result.by_cluster.shape[0] == k
    assert result.by_cluster["is_reference_category"].sum() == 1
    ref_row = result.by_cluster.loc[result.by_cluster["is_reference_category"]].iloc[0]
    assert ref_row["category_id"] == 0


def test_total_gap_is_group0_minus_group1():
    y, group, cluster_labels = _make_sa_kob_data()
    result = get_sa_kob_decomposition(
        y=y,
        group=group,
        cluster_labels=cluster_labels,
        k=4,
        categories=[1, 2, 3, 4],
        group0_value="men",
        group1_value="women",
    )
    gap_direct = y[group == "men"].mean() - y[group == "women"].mean()
    assert np.isclose(result.total_gap, gap_direct, atol=1e-10)


def test_sa_kob_rejects_normalize_categorical_false():
    y, group, cluster_labels = _make_sa_kob_data()
    with pytest.raises(ValueError, match="normalize_categorical must be True"):
        get_sa_kob_decomposition(
            y=y,
            group=group,
            cluster_labels=cluster_labels,
            k=4,
            categories=[1, 2, 3, 4],
            normalize_categorical=False,
        )


def test_sa_kob_bootstrap_survives_missing_rare_cluster_in_draw():
    rng = np.random.default_rng(99)
    n0, n1 = 80, 80
    group = np.array(["men"] * n0 + ["women"] * n1)
    cluster_labels = np.concatenate(
        [
            rng.choice([1, 2, 3], size=n0),
            rng.choice([1, 2, 3], size=n1),
        ]
    )
    # cluster 3 is rare in the sample but fixed in category universe
    cluster_labels[0] = 3
    y = rng.normal(size=n0 + n1)

    boot = get_sa_kob_decomposition_bootstrap(
        y=y,
        group=group,
        cluster_labels=cluster_labels,
        k=3,
        categories=[1, 2, 3],
        group0_value="men",
        group1_value="women",
        n_boot=25,
        random_state=7,
        recompute_owners_each_draw=False,
    )
    assert boot.by_cluster_standard_errors.shape[0] == 3


def test_bootstrap_by_cluster_ci_respects_confidence_level():
    from sequenzo.decomposition.kob import _percentile_bounds

    assert _percentile_bounds(0.90) == pytest.approx((5.0, 95.0))

    y, group, cluster_labels = _make_sa_kob_data()
    boot90 = get_sa_kob_decomposition_bootstrap(
        y=y,
        group=group,
        cluster_labels=cluster_labels,
        k=4,
        categories=[1, 2, 3, 4],
        group0_value="men",
        group1_value="women",
        n_boot=80,
        random_state=0,
        confidence_level=0.90,
    )
    boot95 = get_sa_kob_decomposition_bootstrap(
        y=y,
        group=group,
        cluster_labels=cluster_labels,
        k=4,
        categories=[1, 2, 3, 4],
        group0_value="men",
        group1_value="women",
        n_boot=80,
        random_state=0,
        confidence_level=0.95,
    )
    assert boot90.confidence_level == 0.90
    width90 = (
        boot90.confidence_intervals["total_gap"][1]
        - boot90.confidence_intervals["total_gap"][0]
    )
    width95 = (
        boot95.confidence_intervals["total_gap"][1]
        - boot95.confidence_intervals["total_gap"][0]
    )
    assert width90 <= width95

    cluster_width90 = (
        boot90.by_cluster_confidence_intervals["explained_ci_upper"]
        - boot90.by_cluster_confidence_intervals["explained_ci_lower"]
    )
    cluster_width95 = (
        boot95.by_cluster_confidence_intervals["explained_ci_upper"]
        - boot95.by_cluster_confidence_intervals["explained_ci_lower"]
    )
    assert (cluster_width90 <= cluster_width95 + 1e-12).all()


def test_invalid_confidence_level_raises():
    y, group, cluster_labels = _make_sa_kob_data()
    with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
        get_sa_kob_decomposition_bootstrap(
            y=y,
            group=group,
            cluster_labels=cluster_labels,
            k=4,
            categories=[1, 2, 3, 4],
            confidence_level=95,
            n_boot=5,
        )
