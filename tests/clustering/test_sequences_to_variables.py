"""
Tests for sequences-to-variables (Helske et al. 2024) implementation.
"""
import os
import subprocess
import tempfile
import importlib

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform

from sequenzo.clustering.sequences_to_variables.helpers import (
    cluster_labels_to_dummies,
    cluster_labels_from_kmedoids_result,
    dummy_column_names,
    max_distance,
    medoid_indices_from_kmedoids_result,
    validate_diss_matrix,
    validate_membership_matrix,
)
from sequenzo.clustering.sequences_to_variables import (
    fanny,
    fanny_membership,
    hard_classification_variables,
    medoid_membership_approximation,
    pseudoclass_regression,
    representativeness_matrix,
    soft_classification_variables,
)


def _ruspini_diss():
    x = np.array([
        [4, 53], [5, 63], [6, 70], [8, 76], [10, 83], [12, 88], [14, 93],
        [16, 97], [18, 100], [20, 102], [22, 104], [24, 105], [26, 106],
        [30, 107], [34, 108], [38, 108], [42, 107], [46, 106], [50, 104], [54, 101],
    ], dtype=float)
    return squareform(pdist(x))


def _toy_block_diss():
    return np.array([
        [0, 1, 1, 1, 9, 9, 9, 9],
        [1, 0, 1, 1, 9, 9, 9, 9],
        [1, 1, 0, 1, 9, 9, 9, 9],
        [1, 1, 1, 0, 9, 9, 9, 9],
        [9, 9, 9, 9, 0, 1, 1, 1],
        [9, 9, 9, 9, 1, 0, 1, 1],
        [9, 9, 9, 9, 1, 1, 0, 1],
        [9, 9, 9, 9, 1, 1, 1, 0],
    ], dtype=float)


def _r_fanny_reference(diss, k, memb_exp):
    with tempfile.TemporaryDirectory() as tmp:
        diss_path = os.path.join(tmp, "diss.csv")
        memb_path = os.path.join(tmp, "memb.csv")
        clu_path = os.path.join(tmp, "clu.csv")
        obj_path = os.path.join(tmp, "obj.txt")
        pc_path = os.path.join(tmp, "pc.txt")
        npc_path = os.path.join(tmp, "npc.txt")
        pd.DataFrame(diss).to_csv(diss_path)
        script = os.path.join(tmp, "run.R")
        with open(script, "w") as f:
            f.write(
                f"""
suppressPackageStartupMessages(library(cluster))
d <- as.matrix(read.csv("{diss_path}", row.names=1))
fit <- fanny(d, k={k}, diss=TRUE, memb.exp={memb_exp}, maxit=500, tol=1e-15)
write.table(fit$membership, "{memb_path}", row.names=FALSE, col.names=FALSE)
write.table(fit$clustering, "{clu_path}", row.names=FALSE, col.names=FALSE)
writeLines(as.character(fit$objective[["objective"]]), "{obj_path}")
writeLines(as.character(fit$coeff[["dunn_coeff"]]), "{pc_path}")
writeLines(as.character(fit$coeff[["normalized"]]), "{npc_path}")
writeLines(as.character(fit$k.crisp), "{os.path.join(tmp, "kcrisp.txt")}")
"""
            )
        proc = subprocess.run(
            ["Rscript", script], capture_output=True, text=True, timeout=180,
        )
        if proc.returncode != 0:
            pytest.skip(f"R comparison failed: {proc.stderr}")
        memb = np.loadtxt(memb_path)
        clustering = np.loadtxt(clu_path, dtype=int)
        objective = float(open(obj_path).read().strip())
        pc = float(open(pc_path).read().strip())
        npc = float(open(npc_path).read().strip())
        k_crisp = int(open(os.path.join(tmp, "kcrisp.txt")).read().strip())
        return memb, objective, clustering - 1, pc, npc, k_crisp


RSCRIPT_AVAILABLE = subprocess.run(["which", "Rscript"], capture_output=True).returncode == 0


def test_max_distance_square_and_condensed():
    diss = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
    assert max_distance(diss) == 3.0
    assert max_distance(squareform(diss)) == 3.0
    assert max_distance(np.array([[0.0]])) == 0.0


def test_cluster_labels_to_dummies_0_based():
    labels = np.array([0, 1, 2, 0, 1])
    d = cluster_labels_to_dummies(labels, k=3, reference=0)
    assert d.shape == (5, 2)
    np.testing.assert_array_equal(d[1], [1, 0])
    np.testing.assert_array_equal(d[2], [0, 1])


def test_cluster_labels_to_dummies_1_based():
    labels = np.array([1, 2, 3, 1, 2])
    d = cluster_labels_to_dummies(labels, k=3, reference=0)
    assert d.shape == (5, 2)


def test_dummy_column_names():
    labels = np.array([1, 2, 3, 1, 2])
    assert dummy_column_names(labels, k=3, reference=0) == ["C_2", "C_3"]
    with pytest.raises(ValueError, match="does not match k"):
        dummy_column_names([1, 1, 2], k=3, reference=0)


@pytest.mark.parametrize("bad_reference", [0.5, True])
def test_hard_reference_must_be_integer_index(bad_reference):
    labels = np.array([1, 2, 3, 1, 2])
    with pytest.raises(ValueError, match="integer category index"):
        cluster_labels_to_dummies(labels, k=3, reference=bad_reference)
    with pytest.raises(ValueError, match="integer category index"):
        dummy_column_names(labels, k=3, reference=bad_reference)
    with pytest.raises(ValueError, match="integer category index"):
        hard_classification_variables(labels, k=3, reference=bad_reference)


@pytest.mark.parametrize("bad_labels", [[1.0, 2.2, 3.0], [1, True, 2]])
def test_hard_cluster_labels_must_be_integer_values(bad_labels):
    with pytest.raises(ValueError, match="labels must contain integer"):
        cluster_labels_to_dummies(bad_labels, k=3, reference=0)
    with pytest.raises(ValueError, match="labels must contain integer"):
        dummy_column_names(bad_labels, k=3, reference=0)


@pytest.mark.parametrize("bad_indices", [[1.0, 2.2], [1, True]])
def test_kmedoids_helpers_reject_noninteger_assigned_medoids(bad_indices):
    with pytest.raises(ValueError, match="assigned_medoid_indices must contain integer"):
        medoid_indices_from_kmedoids_result(bad_indices)
    with pytest.raises(ValueError, match="assigned_medoid_indices must contain integer"):
        cluster_labels_from_kmedoids_result(bad_indices)


def test_representativeness_matrix_basic():
    diss = np.array([
        [0, 1, 2, 4],
        [1, 0, 3, 3],
        [2, 3, 0, 2],
        [4, 3, 2, 0],
    ], dtype=float)
    R = representativeness_matrix(diss, [0, 3], d_max=4.0)
    assert R.shape == (4, 2)
    assert R[0, 0] == 1.0
    assert R[3, 1] == 1.0


def test_representativeness_matrix_is_not_row_normalized():
    diss = np.array([
        [0, 2, 8],
        [2, 0, 5],
        [8, 5, 0],
    ], dtype=float)

    R = representativeness_matrix(diss, [0, 1], d_max=8.0)

    np.testing.assert_allclose(R, [
        [1.0, 0.75],
        [0.75, 1.0],
        [0.0, 0.375],
    ])
    assert not np.allclose(R.sum(axis=1), 1.0)


@pytest.mark.parametrize("bad_medoids", [[0.0, 1.2], [0, True]])
def test_representativeness_medoid_indices_must_be_integer_values(bad_medoids):
    diss = np.array([[0, 1], [1, 0]], dtype=float)
    with pytest.raises(ValueError, match="medoid_indices must contain integer"):
        representativeness_matrix(diss, bad_medoids, d_max=1.0)


def test_representativeness_matrix_dataframe_index():
    diss = pd.DataFrame(
        [[0, 1], [1, 0]],
        index=["s1", "s2"],
        columns=["s1", "s2"],
    )
    df = representativeness_matrix(diss, [0], d_max=1.0, as_dataframe=True)
    assert list(df.index) == ["s1", "s2"]


def test_representativeness_dataframe_ids_override_distance_index_order():
    diss = pd.DataFrame(
        [[0.0, 2.0, 4.0], [2.0, 0.0, 3.0], [4.0, 3.0, 0.0]],
        index=["distance_row_a", "distance_row_b", "distance_row_c"],
        columns=["distance_row_a", "distance_row_b", "distance_row_c"],
    )
    ids = ["person_03", "person_01", "person_02"]

    df = representativeness_matrix(
        diss, [0, 2], d_max=4.0, ids=ids, as_dataframe=True
    )

    assert list(df.index) == ids
    np.testing.assert_allclose(df.to_numpy()[0], [1.0, 0.0])


def test_representativeness_rejects_string_representative_names():
    diss = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="representative_names"):
        representativeness_matrix(
            diss,
            [0, 1],
            representative_names="AB",
            as_dataframe=True,
        )


def test_representativeness_dmax_zero():
    diss = np.zeros((3, 3), dtype=float)
    R = representativeness_matrix(diss, [0, 1], d_max=0.0)
    np.testing.assert_allclose(R, np.ones((3, 2)))


def test_representativeness_rejects_invalid_diss():
    diss = np.array([[0, -1], [-1, 0]], dtype=float)
    with pytest.raises(ValueError, match="nonnegative"):
        representativeness_matrix(diss, [0])


def test_representativeness_rejects_nonfinite_diss_and_bad_dmax():
    with pytest.raises(ValueError, match="finite"):
        representativeness_matrix(np.array([[0.0, np.inf], [np.inf, 0.0]]), [0])

    diss = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="finite"):
        representativeness_matrix(diss, [0], d_max=np.nan)
    with pytest.raises(ValueError, match="nonnegative"):
        representativeness_matrix(diss, [0], d_max=-1.0)
    with pytest.raises(ValueError, match="positive"):
        representativeness_matrix(diss, [0], d_max=0.0)

    larger_diss = np.array([[0.0, 2.0], [2.0, 0.0]])
    with pytest.raises(ValueError, match="maximum observed distance"):
        representativeness_matrix(larger_diss, [0], d_max=1.0)


def test_soft_classification_variables_reference_omitted():
    U = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    X = soft_classification_variables(U, reference=0)
    assert X.shape == (3, 2)
    np.testing.assert_allclose(X, U[:, 1:])


def test_soft_classification_dataframe_preserves_supplied_ids_and_names():
    U = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    ids = ["case_c", "case_a", "case_b"]

    df = soft_classification_variables(
        U,
        reference=1,
        ids=ids,
        as_dataframe=True,
        cluster_names=["stable", "mobile", "late"],
    )

    assert list(df.index) == ids
    assert list(df.columns) == ["P_stable", "P_late"]
    np.testing.assert_allclose(df.to_numpy(), U[:, [0, 2]])


@pytest.mark.parametrize("bad_reference", [0.5, True])
def test_soft_reference_must_be_integer_index(bad_reference):
    U = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    with pytest.raises(ValueError, match="integer category index"):
        soft_classification_variables(U, reference=bad_reference)


def test_soft_classification_rejects_string_cluster_names():
    U = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]])
    with pytest.raises(ValueError, match="cluster_names"):
        soft_classification_variables(
            U,
            cluster_names="ABC",
            as_dataframe=True,
        )


def test_validate_membership_matrix_rejects_bad_rows():
    with pytest.raises(ValueError, match="Rows of U must sum to 1"):
        validate_membership_matrix(np.array([[0.5, 0.3, 0.1]] * 3))


def test_validate_membership_matrix_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="finite membership"):
        validate_membership_matrix(np.array([[0.5, np.inf, 0.5]]))


def test_pseudoclass_regression_ols_runs():
    rng = np.random.default_rng(0)
    n = 40
    U = np.array([[0.6, 0.3, 0.1]] * n, dtype=float)
    y = rng.normal(size=n)
    result = pseudoclass_regression(y, U, M=10, random_state=0, model_type="ols")
    assert result["m_eff"] > 0
    assert result["failed"] == 10 - result["m_eff"]
    assert result["param_names"] == ["const", "C_2", "C_3"]
    assert len(result["beta_combined"]) == len(result["param_names"])
    assert result["within_cov"].shape == result["cov_combined"].shape
    assert result["between_cov"].shape == result["cov_combined"].shape
    assert result["success_rate"] == result["m_eff"] / 10
    assert result["M"] == 10
    assert result["reference"] == 0
    assert result["model_type"] == "ols"
    assert result["add_intercept"] is True


def test_pseudoclass_regression_logit_runs():
    rng = np.random.default_rng(1)
    n = 60
    U = np.array([[0.5, 0.3, 0.2]] * n, dtype=float)
    y = (rng.normal(size=n) > 0).astype(float)
    result = pseudoclass_regression(y, U, M=10, random_state=1, model_type="logit")
    assert result["m_eff"] > 0


def test_pseudoclass_regression_validates_m_and_y():
    U = np.array([[0.6, 0.4]] * 5)
    y = np.zeros(3)
    with pytest.raises(ValueError, match="model_type"):
        pseudoclass_regression(y, U, M=5, model_type=1)
    with pytest.raises(ValueError, match="M must be an integer"):
        pseudoclass_regression(y, U, M=1.5)
    with pytest.raises(ValueError, match="M must be at least 1"):
        pseudoclass_regression(y, U, M=0)
    with pytest.raises(ValueError, match="same number of rows"):
        pseudoclass_regression(y, U, M=5)
    with pytest.raises(ValueError, match="at least two clusters"):
        pseudoclass_regression(y, np.ones((5, 1)), M=5)


def test_pseudoclass_regression_rejects_string_names_and_nonbool_intercept():
    y = np.arange(6.0)
    U = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] * 2)
    X_fixed = np.arange(18.0).reshape(6, 3)

    with pytest.raises(ValueError, match="x_fixed_names"):
        pseudoclass_regression(y, U, X_fixed=X_fixed, x_fixed_names="age", M=2)
    with pytest.raises(ValueError, match="cluster_names"):
        pseudoclass_regression(y, U, cluster_names="ABC", M=2)
    with pytest.raises(ValueError, match="add_intercept"):
        pseudoclass_regression(y, U, add_intercept="False", M=2)


def test_pseudoclass_regression_rejects_multicolumn_y_and_noninteger_reference():
    U = np.array([[0.6, 0.4]] * 6)
    with pytest.raises(ValueError, match="1D array or a single-column"):
        pseudoclass_regression(np.zeros((2, 3)), U, M=5)
    with pytest.raises(ValueError, match="integer cluster index"):
        pseudoclass_regression(np.zeros(6), U, M=5, reference=0.5)


def test_pseudoclass_regression_validates_y_values_before_draws():
    U = np.array([[0.6, 0.4]] * 5)
    with pytest.raises(ValueError, match="y must contain only finite values"):
        pseudoclass_regression(np.array([0.0, 1.0, np.nan, 0.0, 1.0]), U, M=5)
    with pytest.raises(ValueError, match="binary 0/1"):
        pseudoclass_regression(
            np.array([0.0, 1.0, 2.0, 0.0, 1.0]), U, M=5, model_type="logit"
        )
    with pytest.raises(ValueError, match="both 0 and 1"):
        pseudoclass_regression(np.zeros(5), U, M=5, model_type="logit")


def test_max_distance_rejects_nan_condensed():
    with pytest.raises(ValueError, match="NA values"):
        max_distance(np.array([0.0, 1.0, np.nan]))


def test_pseudoclass_regression_1d_x_fixed():
    rng = np.random.default_rng(2)
    n = 30
    U = np.array([[0.5, 0.3, 0.2]] * n, dtype=float)
    y = rng.normal(size=n)
    x = rng.normal(size=n)
    result = pseudoclass_regression(y, U, X_fixed=x, M=10, random_state=2, model_type="ols")
    assert result["m_eff"] > 0


def test_pseudoclass_regression_rejects_misaligned_x_fixed_rows():
    y = np.linspace(0.0, 1.0, 6)
    U = np.array([[0.6, 0.4]] * 6, dtype=float)
    X_fixed = np.ones((5, 1), dtype=float)

    with pytest.raises(ValueError, match="X_fixed must have same number of rows as U"):
        pseudoclass_regression(y, U, X_fixed=X_fixed, M=4, random_state=0)


def test_pseudoclass_regression_existing_intercept_is_not_duplicated():
    rng = np.random.default_rng(22)
    n = 80
    U = np.array([[0.4, 0.35, 0.25]] * n, dtype=float)
    y = rng.normal(size=n)
    x_fixed = np.column_stack([np.ones(n), rng.normal(size=n)])

    result = pseudoclass_regression(
        y, U, X_fixed=x_fixed, M=10, random_state=22, model_type="ols"
    )

    assert result["m_eff"] > 0
    assert result["param_names"] == ["const", "X_fixed_2", "C_2", "C_3"]
    assert len(result["beta_combined"]) == 4


def test_pseudoclass_regression_near_one_column_is_not_named_const():
    rng = np.random.default_rng(23)
    n = 80
    U = np.array([[0.5, 0.5]] * n, dtype=float)
    y = rng.normal(size=n)
    x_fixed = np.full((n, 1), 1.0 + 1e-7)

    result = pseudoclass_regression(
        y, U, X_fixed=x_fixed, M=8, random_state=23, model_type="ols", add_intercept=False
    )

    assert result["m_eff"] > 0
    assert result["param_names"] == ["X_fixed_1", "C_2"]


def test_pseudoclass_regression_all_failed_draws_report_reasons():
    y = np.linspace(-1.0, 1.0, 12)
    U = np.array([[1.0, 0.0]] * y.size, dtype=float)

    with pytest.raises(RuntimeError, match="rank_deficient"):
        pseudoclass_regression(y, U, M=4, random_state=5, model_type="ols")


def test_pseudoclass_regression_logit_all_failed_draws_report_reasons():
    y = np.array([0.0] * 10 + [1.0] * 10)
    U = np.array([[1.0, 0.0]] * 10 + [[0.0, 1.0]] * 10, dtype=float)

    with pytest.raises(RuntimeError, match="linear_algebra_error"):
        pseudoclass_regression(y, U, M=4, random_state=6, model_type="logit")


def test_pseudoclass_regression_missing_cluster_draw():
    """Draws may omit a cluster without raising from dummy encoding."""
    rng = np.random.default_rng(3)
    n = 40
    U = np.array([[0.7, 0.2, 0.1]] * n, dtype=float)
    y = rng.normal(size=n)
    with pytest.warns(RuntimeWarning, match="pseudoclass replications failed"):
        result = pseudoclass_regression(y, U, M=50, random_state=3, model_type="ols")
    assert result["m_eff"] > 0


def test_pseudoclass_regression_rubin_pooling_values_are_recomputable():
    rng = np.random.default_rng(30)
    n = 120
    U = np.array([[0.34, 0.33, 0.33]] * n, dtype=float)
    y = rng.normal(size=n)

    result = pseudoclass_regression(y, U, M=12, random_state=30, model_type="ols")

    beta_stack = np.array(result["beta_list"])
    cov_stack = np.array(result["cov_list"])
    W = np.mean(cov_stack, axis=0)
    B = np.atleast_2d(np.cov(beta_stack, rowvar=False, ddof=1))
    T = W + (1.0 + 1.0 / result["m_eff"]) * B

    np.testing.assert_allclose(result["within_cov"], W)
    np.testing.assert_allclose(result["between_cov"], B)
    np.testing.assert_allclose(result["cov_combined"], T)
    assert np.all(np.isfinite(result["se_combined"]))
    assert np.all(np.diag(result["cov_combined"]) >= 0)


def test_pseudoclass_regression_single_parameter_between_cov_shape():
    y = np.linspace(-1.0, 1.0, 20)
    U = np.array([[0.0, 1.0]] * y.size, dtype=float)
    result = pseudoclass_regression(
        y, U, M=4, reference=0, random_state=4, model_type="ols", add_intercept=False
    )

    assert result["param_names"] == ["C_2"]
    assert result["between_cov"].shape == (1, 1)
    np.testing.assert_allclose(result["between_cov"], np.zeros((1, 1)))
    np.testing.assert_allclose(result["cov_combined"], result["within_cov"])


def test_pseudoclass_regression_named_nonzero_reference_and_late_intercept():
    rng = np.random.default_rng(31)
    n = 90
    U = np.array([[0.3, 0.4, 0.3]] * n, dtype=float)
    y = rng.normal(size=n)
    X_fixed = np.column_stack([rng.normal(size=n), np.ones(n)])

    result = pseudoclass_regression(
        y,
        U,
        X_fixed=X_fixed,
        M=10,
        reference=1,
        random_state=31,
        x_fixed_names=["age", "manual_const"],
        cluster_names=["A", "B", "C"],
    )

    assert result["param_names"] == ["age", "const", "C_A", "C_C"]
    assert len(result["beta_combined"]) == len(result["param_names"])


def test_pseudoclass_regression_same_seed_is_deterministic():
    rng = np.random.default_rng(32)
    n = 70
    U = np.array([[0.45, 0.35, 0.20]] * n, dtype=float)
    y = rng.normal(size=n)

    result_a = pseudoclass_regression(y, U, M=8, random_state=32, model_type="ols")
    result_b = pseudoclass_regression(y, U, M=8, random_state=32, model_type="ols")

    np.testing.assert_allclose(result_a["beta_combined"], result_b["beta_combined"])
    np.testing.assert_allclose(result_a["cov_combined"], result_b["cov_combined"])
    assert result_a["failed_reasons"] == result_b["failed_reasons"]


def test_pseudoclass_regression_one_hot_matches_hard_ols():
    import statsmodels.api as sm

    y = np.array([1.0, 2.0, 1.5, 4.0, 5.0, 4.5])
    U = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
                  [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    hard_X = sm.add_constant(cluster_labels_to_dummies([0, 0, 0, 1, 1, 1], k=2))
    hard_fit = sm.OLS(y, hard_X).fit()

    result = pseudoclass_regression(y, U, M=6, random_state=123, model_type="ols")

    np.testing.assert_allclose(result["beta_combined"], hard_fit.params)
    np.testing.assert_allclose(result["within_cov"], hard_fit.cov_params())
    np.testing.assert_allclose(result["between_cov"], np.zeros((2, 2)))
    assert result["m_eff"] == 6
    assert result["failed"] == 0


def test_pseudoclass_regression_reports_failed_draw_reasons():
    rng = np.random.default_rng(0)
    n = 30
    U = np.array([[0.95, 0.04, 0.01]] * n, dtype=float)
    y = rng.normal(size=n)

    with pytest.warns(RuntimeWarning, match="pseudoclass replications failed"):
        result = pseudoclass_regression(y, U, M=20, random_state=0, model_type="ols")

    assert 0 < result["failed"] < 20
    assert result["failed_reasons"]["rank_deficient"] == result["failed"]
    assert result["success_rate"] == result["m_eff"] / 20


def test_fanny_membership_accepts_weights_for_clara_seed_path():
    diss = _ruspini_diss()
    weights = np.linspace(1.0, 2.0, diss.shape[0])
    U, highest = fanny_membership(diss, k=4, m=1.5, weights=weights, max_iter=500)
    assert U.shape == (diss.shape[0], 4)
    assert highest.shape == (4,)
    np.testing.assert_allclose(U.sum(axis=1), np.ones(diss.shape[0]), atol=1e-10)


@pytest.mark.parametrize(
    ("method", "expected_width"),
    [("fuzzy", 2), ("noise", 3)],
)
def test_clara_fuzzy_paths_return_matrix_valued_public_output(
    monkeypatch,
    method,
    expected_width,
):
    from sequenzo.define_sequence_data import SequenceData

    clara_module = importlib.import_module("sequenzo.big_data.clara.clara")
    captured = {"weights": []}

    class FakeParallel:
        def __init__(self, n_jobs):
            self.n_jobs = n_jobs

        def __call__(self, tasks):
            return [task() for task in tasks]

    class FakeClustering:
        def __init__(self, functional):
            self.functional = functional
            self.mobile_centers = np.array([0, 1], dtype=int)
            self.dnoise = 2.0

    def fake_delayed(func):
        def _bind(*args, **kwargs):
            return lambda: func(*args, **kwargs)
        return _bind

    def fake_linkage(diss, method):
        return {"n": diss.shape[0]}

    def fake_cut_tree(hc, n_clusters):
        labels = np.arange(hc["n"]) % n_clusters
        return labels.reshape(-1, 1)

    def fake_wfcmdd(diss, memb, weights, method, m, dnoise):
        return FakeClustering(functional=1.0)

    def fake_fanny_membership(diss, k, m, weights, **kwargs):
        captured["weights"].append(np.asarray(weights, dtype=float).copy())
        membership = np.zeros((diss.shape[0], k), dtype=float)
        membership[np.arange(diss.shape[0]), np.arange(diss.shape[0]) % k] = 1.0
        return membership, np.arange(k)

    def fake_get_distance_matrix(opts):
        n = len(opts["seqdata"].seqdata)
        if "refseq" in opts:
            k = len(opts["refseq"][1])
            arr = 1.0 + np.abs(np.arange(n)[:, None] - np.arange(k)[None, :])
            return pd.DataFrame(arr)
        arr = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(float)
        return pd.DataFrame(arr)

    monkeypatch.setattr(clara_module, "Parallel", FakeParallel)
    monkeypatch.setattr(clara_module, "delayed", fake_delayed)
    monkeypatch.setattr(clara_module, "linkage", fake_linkage)
    monkeypatch.setattr(clara_module, "cut_tree", fake_cut_tree)
    monkeypatch.setattr(clara_module, "wfcmdd", fake_wfcmdd)
    monkeypatch.setattr(clara_module, "fanny_membership", fake_fanny_membership)
    monkeypatch.setattr(clara_module, "get_distance_matrix", fake_get_distance_matrix)
    monkeypatch.setattr(
        clara_module.np.random,
        "choice",
        lambda a, size, p, replace: np.arange(size) % a,
    )

    df = pd.DataFrame({
        "id": np.arange(1, 9),
        "T1": [0, 0, 1, 1, 0, 0, 1, 1],
        "T2": [0, 1, 1, 0, 0, 1, 1, 0],
    })
    seqdata = SequenceData(df, time=["T1", "T2"], id_col="id", states=[0, 1])

    result = clara_module.clara(
        seqdata,
        R=1,
        kvals=[2],
        sample_size=6,
        method=method,
        criteria=["distance"],
        stability=True,
        dist_args={"method": "OM", "sm": "CONSTANT", "indel": 1},
    )

    assert captured["weights"]
    assert captured["weights"][0].ndim == 1
    assert result["clara"][0]["method"] == method
    assert result["clara"][0]["arimatrix"].shape == (1, 2)
    assert result["clara"][0]["clustering"].shape == (len(df), expected_width)
    first_public_cell = result["clustering"].iloc[0, 0]
    assert isinstance(first_public_cell, np.ndarray)
    assert first_public_cell.shape == (expected_width,)
    assert np.isclose(first_public_cell.sum(), 1.0)


def test_clara_fuzzy_path_falls_back_when_optional_fanny_seed_unavailable(monkeypatch):
    from sequenzo.define_sequence_data import SequenceData

    clara_module = importlib.import_module("sequenzo.big_data.clara.clara")
    wfcmdd_calls = []

    class FakeParallel:
        def __init__(self, n_jobs):
            self.n_jobs = n_jobs

        def __call__(self, tasks):
            return [task() for task in tasks]

    class FakeClustering:
        def __init__(self, functional):
            self.functional = functional
            self.mobile_centers = np.array([0, 1], dtype=int)
            self.dnoise = 2.0

    def fake_delayed(func):
        def _bind(*args, **kwargs):
            return lambda: func(*args, **kwargs)
        return _bind

    def fake_linkage(diss, method):
        return {"n": diss.shape[0]}

    def fake_cut_tree(hc, n_clusters):
        labels = np.arange(hc["n"]) % n_clusters
        return labels.reshape(-1, 1)

    def fake_wfcmdd(diss, memb, weights, method, m, dnoise):
        wfcmdd_calls.append(np.asarray(memb, dtype=float).copy())
        return FakeClustering(functional=1.0)

    def fake_fanny_membership(diss, k, m, weights, **kwargs):
        raise ValueError(
            f"k must be at most n//2 - 1; got k={k}, n={diss.shape[0]}"
        )

    def fake_get_distance_matrix(opts):
        n = len(opts["seqdata"].seqdata)
        if "refseq" in opts:
            k = len(opts["refseq"][1])
            arr = 1.0 + np.abs(np.arange(n)[:, None] - np.arange(k)[None, :])
            return pd.DataFrame(arr)
        arr = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(float)
        return pd.DataFrame(arr)

    monkeypatch.setattr(clara_module, "Parallel", FakeParallel)
    monkeypatch.setattr(clara_module, "delayed", fake_delayed)
    monkeypatch.setattr(clara_module, "linkage", fake_linkage)
    monkeypatch.setattr(clara_module, "cut_tree", fake_cut_tree)
    monkeypatch.setattr(clara_module, "wfcmdd", fake_wfcmdd)
    monkeypatch.setattr(clara_module, "fanny_membership", fake_fanny_membership)
    monkeypatch.setattr(clara_module, "get_distance_matrix", fake_get_distance_matrix)
    monkeypatch.setattr(
        clara_module.np.random,
        "choice",
        lambda a, size, p, replace: np.arange(size) % a,
    )

    df = pd.DataFrame({
        "id": np.arange(1, 9),
        "T1": [0, 0, 1, 1, 0, 0, 1, 1],
        "T2": [0, 1, 1, 0, 0, 1, 1, 0],
    })
    seqdata = SequenceData(df, time=["T1", "T2"], id_col="id", states=[0, 1])

    result = clara_module.clara(
        seqdata,
        R=1,
        kvals=[2],
        sample_size=6,
        method="fuzzy",
        criteria=["distance"],
        stability=True,
        dist_args={"method": "OM", "sm": "CONSTANT", "indel": 1},
    )

    assert len(wfcmdd_calls) == 1
    assert result["clara"][0]["method"] == "fuzzy"
    assert result["clara"][0]["clustering"].shape == (len(df), 2)


def test_clara_fuzzy_path_does_not_swallow_downstream_value_errors(monkeypatch):
    from sequenzo.define_sequence_data import SequenceData

    clara_module = importlib.import_module("sequenzo.big_data.clara.clara")
    wfcmdd_calls = 0

    class FakeParallel:
        def __init__(self, n_jobs):
            self.n_jobs = n_jobs

        def __call__(self, tasks):
            return [task() for task in tasks]

    class FakeClustering:
        functional = 1.0
        mobile_centers = np.array([0, 1], dtype=int)
        dnoise = 2.0

    def fake_delayed(func):
        def _bind(*args, **kwargs):
            return lambda: func(*args, **kwargs)
        return _bind

    def fake_linkage(diss, method):
        return {"n": diss.shape[0]}

    def fake_cut_tree(hc, n_clusters):
        labels = np.arange(hc["n"]) % n_clusters
        return labels.reshape(-1, 1)

    def fake_wfcmdd(diss, memb, weights, method, m, dnoise):
        nonlocal wfcmdd_calls
        wfcmdd_calls += 1
        if wfcmdd_calls == 2:
            raise ValueError("k must be at most n//2 - 1 downstream failure")
        return FakeClustering()

    def fake_fanny_membership(diss, k, m, weights, **kwargs):
        return np.full((diss.shape[0], k), 1.0 / k), None

    def fake_get_distance_matrix(opts):
        n = len(opts["seqdata"].seqdata)
        if "refseq" in opts:
            k = len(opts["refseq"][1])
            arr = 1.0 + np.abs(np.arange(n)[:, None] - np.arange(k)[None, :])
            return pd.DataFrame(arr)
        arr = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(float)
        return pd.DataFrame(arr)

    monkeypatch.setattr(clara_module, "Parallel", FakeParallel)
    monkeypatch.setattr(clara_module, "delayed", fake_delayed)
    monkeypatch.setattr(clara_module, "linkage", fake_linkage)
    monkeypatch.setattr(clara_module, "cut_tree", fake_cut_tree)
    monkeypatch.setattr(clara_module, "wfcmdd", fake_wfcmdd)
    monkeypatch.setattr(clara_module, "fanny_membership", fake_fanny_membership)
    monkeypatch.setattr(clara_module, "get_distance_matrix", fake_get_distance_matrix)
    monkeypatch.setattr(
        clara_module.np.random,
        "choice",
        lambda a, size, p, replace: np.arange(size) % a,
    )

    df = pd.DataFrame({
        "id": np.arange(1, 9),
        "T1": [0, 0, 1, 1, 0, 0, 1, 1],
        "T2": [0, 1, 1, 0, 0, 1, 1, 0],
    })
    seqdata = SequenceData(df, time=["T1", "T2"], id_col="id", states=[0, 1])

    with pytest.raises(ValueError, match="downstream failure"):
        clara_module.clara(
            seqdata,
            R=1,
            kvals=[2],
            sample_size=6,
            method="fuzzy",
            criteria=["distance"],
            stability=True,
            dist_args={"method": "OM", "sm": "CONSTANT", "indel": 1},
        )


def test_clara_preserves_actual_k_metadata_for_non_default_kvals(monkeypatch):
    from sequenzo.define_sequence_data import SequenceData

    clara_module = importlib.import_module("sequenzo.big_data.clara.clara")

    class FakeParallel:
        def __init__(self, n_jobs):
            self.n_jobs = n_jobs

        def __call__(self, tasks):
            return [task() for task in tasks]

    def fake_delayed(func):
        def _bind(*args, **kwargs):
            return lambda: func(*args, **kwargs)
        return _bind

    def fake_linkage(diss, method):
        return {"n": diss.shape[0]}

    def fake_kmedoids(diss, k, **kwargs):
        return (np.arange(diss.shape[0]) % k) + 1

    def fake_get_distance_matrix(opts):
        n = len(opts["seqdata"].seqdata)
        if "refseq" in opts:
            k = len(opts["refseq"][1])
            arr = 1.0 + np.abs(np.arange(n)[:, None] - np.arange(k)[None, :])
            return pd.DataFrame(arr)
        arr = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(float)
        return pd.DataFrame(arr)

    monkeypatch.setattr(clara_module, "Parallel", FakeParallel)
    monkeypatch.setattr(clara_module, "delayed", fake_delayed)
    monkeypatch.setattr(clara_module, "linkage", fake_linkage)
    monkeypatch.setattr(clara_module, "KMedoids", fake_kmedoids)
    monkeypatch.setattr(clara_module, "get_distance_matrix", fake_get_distance_matrix)
    monkeypatch.setattr(
        clara_module.np.random,
        "choice",
        lambda a, size, p, replace: np.arange(size) % a,
    )

    df = pd.DataFrame({
        "id": np.arange(1, 10),
        "T1": [0, 0, 1, 1, 2, 2, 0, 1, 2],
        "T2": [0, 1, 1, 2, 2, 0, 1, 2, 0],
    })
    seqdata = SequenceData(df, time=["T1", "T2"], id_col="id", states=[0, 1, 2])

    result = clara_module.clara(
        seqdata,
        R=1,
        kvals=[3],
        sample_size=7,
        method="crisp",
        criteria=["distance"],
        stability=False,
        dist_args={"method": "OM", "sm": "CONSTANT", "indel": 1},
    )

    assert result["kvals"] == [3]
    assert list(result["clustering"].columns) == ["Cluster 3"]
    assert result["stats"]["k_num"].tolist() == [3]
    assert result["clara"][0]["k"] == 3


@pytest.mark.parametrize("bad_max_dist", [0.0, -1.0, np.inf, np.nan, True])
def test_clara_representativeness_rejects_invalid_max_dist(bad_max_dist):
    from sequenzo.define_sequence_data import SequenceData

    clara_module = importlib.import_module("sequenzo.big_data.clara.clara")
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "T1": [0, 1, 0],
        "T2": [1, 1, 0],
    })
    seqdata = SequenceData(df, time=["T1", "T2"], id_col="id", states=[0, 1])

    with pytest.raises(ValueError, match="max.dist"):
        clara_module.clara(
            seqdata,
            R=1,
            kvals=[2],
            sample_size=3,
            method="representativeness",
            criteria=["distance"],
            max_dist=bad_max_dist,
            dist_args={"method": "OM", "sm": "CONSTANT", "indel": 1},
        )


def test_clara_representativeness_returns_matrix_valued_public_output(monkeypatch):
    from sequenzo.define_sequence_data import SequenceData

    clara_module = importlib.import_module("sequenzo.big_data.clara.clara")

    class FakeParallel:
        def __init__(self, n_jobs):
            self.n_jobs = n_jobs

        def __call__(self, tasks):
            return [task() for task in tasks]

    def fake_delayed(func):
        def _bind(*args, **kwargs):
            return lambda: func(*args, **kwargs)
        return _bind

    def fake_linkage(diss, method):
        return {"n": diss.shape[0]}

    def fake_kmedoids(diss, k, **kwargs):
        return (np.arange(diss.shape[0]) % k) + 1

    def fake_get_distance_matrix(opts):
        n = len(opts["seqdata"].seqdata)
        if "refseq" in opts:
            k = len(opts["refseq"][1])
            arr = 1.0 + np.abs(np.arange(n)[:, None] - np.arange(k)[None, :])
            return pd.DataFrame(arr)
        arr = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(float)
        return pd.DataFrame(arr)

    monkeypatch.setattr(clara_module, "Parallel", FakeParallel)
    monkeypatch.setattr(clara_module, "delayed", fake_delayed)
    monkeypatch.setattr(clara_module, "linkage", fake_linkage)
    monkeypatch.setattr(clara_module, "KMedoids", fake_kmedoids)
    monkeypatch.setattr(clara_module, "get_distance_matrix", fake_get_distance_matrix)
    monkeypatch.setattr(
        clara_module.np.random,
        "choice",
        lambda a, size, p, replace: np.arange(size) % a,
    )

    df = pd.DataFrame({
        "id": np.arange(1, 9),
        "T1": [0, 0, 1, 1, 0, 0, 1, 1],
        "T2": [0, 1, 1, 0, 0, 1, 1, 0],
    })
    seqdata = SequenceData(df, time=["T1", "T2"], id_col="id", states=[0, 1])

    result = clara_module.clara(
        seqdata,
        R=1,
        kvals=[2],
        sample_size=6,
        method="representativeness",
        criteria=["distance"],
        max_dist=10.0,
        dist_args={"method": "OM", "sm": "CONSTANT", "indel": 1},
    )

    assert result["clara"][0]["method"] == "representativeness"
    assert result["clara"][0]["clustering"].shape == (len(df), 2)
    first_public_cell = result["clustering"].iloc[0, 0]
    assert isinstance(first_public_cell, np.ndarray)
    assert first_public_cell.shape == (2,)
    assert np.all((0.0 <= first_public_cell) & (first_public_cell <= 1.0))


def test_clara_representativeness_uses_aggregated_medoids_for_refseq(monkeypatch):
    from sequenzo.define_sequence_data import SequenceData

    clara_module = importlib.import_module("sequenzo.big_data.clara.clara")

    class FakeParallel:
        def __init__(self, n_jobs):
            self.n_jobs = n_jobs

        def __call__(self, tasks):
            return [task() for task in tasks]

    def fake_delayed(func):
        def _bind(*args, **kwargs):
            return lambda: func(*args, **kwargs)
        return _bind

    def fake_linkage(diss, method):
        return {"n": diss.shape[0]}

    def fake_kmedoids(diss, k, **kwargs):
        assert k == 2
        assert diss.shape[0] == 4
        return np.array([3, 4, 3, 4])

    seen_ref_medoids = []

    def fake_get_distance_matrix(opts):
        n = len(opts["seqdata"].seqdata)
        if "refseq" in opts:
            medoids = np.asarray(opts["refseq"][1], dtype=int)
            assert np.all(medoids < n), (
                f"refseq medoids out of range for aggregated seqdata: "
                f"medoids={medoids.tolist()}, n={n}"
            )
            seen_ref_medoids.append(medoids.copy())
            arr = 1.0 + np.abs(np.arange(n)[:, None] - medoids[None, :])
            return pd.DataFrame(arr)
        arr = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(float)
        return pd.DataFrame(arr)

    monkeypatch.setattr(clara_module, "Parallel", FakeParallel)
    monkeypatch.setattr(clara_module, "delayed", fake_delayed)
    monkeypatch.setattr(clara_module, "linkage", fake_linkage)
    monkeypatch.setattr(clara_module, "KMedoids", fake_kmedoids)
    monkeypatch.setattr(clara_module, "get_distance_matrix", fake_get_distance_matrix)
    monkeypatch.setattr(
        clara_module.np.random,
        "choice",
        lambda a, size, p, replace: np.arange(size) % a,
    )

    df = pd.DataFrame({
        "id": np.arange(1, 6),
        "T1": [0, 0, 1, 2, 3],
        "T2": [1, 1, 2, 3, 4],
    })
    seqdata = SequenceData(df, time=["T1", "T2"], id_col="id", states=[0, 1, 2, 3, 4])

    result = clara_module.clara(
        seqdata,
        R=1,
        kvals=[2],
        sample_size=4,
        method="representativeness",
        criteria=["distance"],
        max_dist=10.0,
        dist_args={"method": "OM", "sm": "CONSTANT", "indel": 1},
    )

    assert seen_ref_medoids
    assert all(np.array_equal(medoids, np.array([2, 3])) for medoids in seen_ref_medoids)
    np.testing.assert_array_equal(result["clara"][0]["medoids_agg"], np.array([2, 3]))
    np.testing.assert_array_equal(result["clara"][0]["medoids"], np.array([3, 4]))


def test_validate_diss_matrix_rejects_nonsquare():
    with pytest.raises(ValueError, match="square matrix"):
        validate_diss_matrix(np.array([[0, 1, 2], [1, 0, 3]]))


def test_medoid_indices_input_base():
    assigned = np.array([4, 4, 8, 8, 11, 11])
    np.testing.assert_array_equal(
        medoid_indices_from_kmedoids_result(assigned, input_base=1),
        np.array([3, 7, 10]),
    )
    np.testing.assert_array_equal(
        medoid_indices_from_kmedoids_result(assigned, input_base=0),
        np.array([4, 8, 11]),
    )


def test_cluster_labels_input_base():
    assigned = np.array([4, 4, 8, 8, 11, 11])
    np.testing.assert_array_equal(
        cluster_labels_from_kmedoids_result(assigned, input_base=0),
        np.array([0, 0, 1, 1, 2, 2]),
    )


def test_fanny_rows_sum_to_one():
    diss = _ruspini_diss()
    U, _ = fanny_membership(diss, k=4, m=1.5, max_iter=500)
    np.testing.assert_allclose(U.sum(axis=1), np.ones(20), atol=1e-10)


def test_pair_dist_matches_square_matrix():
    diss = np.array(
        [
            [0, 1, 2, 3],
            [1, 0, 4, 5],
            [2, 4, 0, 6],
            [3, 5, 6, 0],
        ],
        dtype=float,
    )
    from sequenzo.clustering.sequences_to_variables.fanny import _pair_dist, _square_to_condensed

    dss = _square_to_condensed(diss)
    n = diss.shape[0]
    for i in range(n):
        for j in range(n):
            assert _pair_dist(dss, i, j, n) == diss[i, j]


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_fanny_k_one(n):
    diss = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(float)
    result = fanny(diss, k=1, memb_exp=1.4)
    np.testing.assert_allclose(result.membership, np.ones((n, 1)))
    assert result.converged
    assert result.iterations > 0
    assert result.objective == 0.0
    assert result.k_crisp == 1
    assert result.r_parity is False


def test_fanny_k_one_validates_initial_membership():
    diss = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="shape"):
        fanny(diss, k=1, ini_mem_p=np.ones((2, 2)))
    with pytest.raises(ValueError, match="nonnegative"):
        fanny(diss, k=1, ini_mem_p=np.array([[1.0], [-1.0]]))
    with pytest.raises(ValueError, match="rows must sum"):
        fanny(diss, k=1, ini_mem_p=np.array([[0.5], [0.5]]))


def test_fanny_membership_rejects_invalid_weights():
    diss = _ruspini_diss()
    with pytest.raises(ValueError, match="shape"):
        fanny_membership(diss, k=4, weights=np.ones(diss.shape[0] + 1))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        fanny_membership(diss, k=4, weights=np.full(diss.shape[0], np.nan))
    weights = np.ones(diss.shape[0])
    weights[0] = -1.0
    with pytest.raises(ValueError, match="finite and nonnegative"):
        fanny_membership(diss, k=4, weights=weights)


def test_fanny_rejects_invalid_scalar_parameters():
    diss = _toy_block_diss()
    with pytest.raises(ValueError, match="k must be an integer"):
        fanny(diss, k=2.5, memb_exp=1.4)
    with pytest.raises(ValueError, match="k must be an integer"):
        fanny(diss, k=True, memb_exp=1.4)
    with pytest.raises(ValueError, match="memb_exp must be a finite number"):
        fanny(diss, k=2, memb_exp="bad")
    with pytest.raises(ValueError, match="memb_exp must be a finite number > 1.0"):
        fanny(diss, k=2, memb_exp=1.0)
    with pytest.raises(ValueError, match="max_iter must be an integer"):
        fanny(diss, k=2, memb_exp=1.4, max_iter=2.5)
    with pytest.raises(ValueError, match="tol must be a finite number"):
        fanny(diss, k=2, memb_exp=1.4, tol=np.nan)


def test_fanny_rejects_nonfinite_distances():
    diss = _toy_block_diss()
    diss[0, 1] = np.inf
    diss[1, 0] = np.inf

    with pytest.raises(ValueError, match="finite dissimilarities"):
        fanny(diss, k=2, memb_exp=1.4)


def test_fanny_rejects_degenerate_zero_distances_for_multiple_clusters():
    diss = np.zeros((6, 6), dtype=float)

    with pytest.raises(ValueError, match="at least one positive distance"):
        fanny(diss, k=2, memb_exp=1.4)
    with pytest.raises(ValueError, match="at least one positive distance"):
        fanny_membership(diss, k=2, m=1.4)


def test_medoid_membership_approximation_rejects_degenerate_zero_distances():
    diss = np.zeros((6, 6), dtype=float)

    with pytest.raises(ValueError, match="at least one positive distance"):
        medoid_membership_approximation(diss, k=2, m=1.4)


def test_fanny_non_convergence_iterations_match_r():
    diss = _toy_block_diss()
    with pytest.warns(UserWarning, match="has not converged"):
        result = fanny(diss, k=2, memb_exp=1.4, max_iter=1, tol=1e-15)
    assert not result.converged
    assert result.iterations == -1


@pytest.mark.skipif(not RSCRIPT_AVAILABLE, reason="Rscript not available")
def test_fanny_against_R_cluster_fanny_small_matrix():
    diss = _toy_block_diss()
    rmemb, r_obj, r_clu, r_pc, r_npc, r_k_crisp = _r_fanny_reference(diss, k=2, memb_exp=1.4)
    py = fanny(diss, k=2, memb_exp=1.4, max_iter=500, tol=1e-15)
    np.testing.assert_allclose(py.membership, rmemb, atol=1e-6)
    assert abs(py.objective - r_obj) < 1e-5
    np.testing.assert_array_equal(py.clustering, r_clu)
    assert py.k_crisp == r_k_crisp
    assert abs(py.partition_coefficient - r_pc) < 1e-3
    assert abs(py.normalized_coefficient - r_npc) < 1e-3


@pytest.mark.parametrize("memb_exp", [1.4, 1.5, 2.0])
@pytest.mark.skipif(not RSCRIPT_AVAILABLE, reason="Rscript not available")
def test_fanny_against_R_ruspini(memb_exp):
    diss = _ruspini_diss()
    rmemb, r_obj, _, r_pc, r_npc, _ = _r_fanny_reference(diss, k=4, memb_exp=memb_exp)
    py = fanny(diss, k=4, memb_exp=memb_exp, max_iter=500, tol=1e-15)
    np.testing.assert_allclose(py.membership, rmemb, atol=1e-6)
    assert abs(py.objective - r_obj) < 1e-5
    assert abs(py.partition_coefficient - r_pc) < 1e-3
    assert abs(py.normalized_coefficient - r_npc) < 1e-3


def test_hard_classification_variables_dataframe():
    labels = np.array([1, 2, 3, 1, 2])
    ids = ["i", "j", "k", "l", "m"]
    df = hard_classification_variables(
        labels, k=3, reference=0, ids=ids, as_dataframe=True,
    )
    assert list(df.columns) == ["C_2", "C_3"]
    assert list(df.index) == ids
    np.testing.assert_array_equal(df.to_numpy()[2], [0, 1])
