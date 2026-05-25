import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData
from sequenzo.seqhmm import build_mnhmm, estimate_mnhmm
from sequenzo.seqhmm.mnhmm import _reduce_eta_B_full


REF_ROOT = "tests/seqhmm_parity/_refs"
PREREG = "tests/seqhmm_parity/test_mnhmm_matrix.py"
LOGLIK_ATOL = 1e-6
SQRT2_INV = 1.0 / np.sqrt(2.0)


MNHMM_TIME_COLS = ["t1", "t2", "t3", "t4", "t5"]
MNHMM_ROWS = [
    ["A", "A", "A", "A", "A"],
    ["A", "A", "B", "A", "A"],
    ["B", "B", "B", "B", "B"],
    ["B", "B", "A", "B", "B"],
]
MNHMM_FITTED_COMPONENT_ROWS = [
    ["A", "A", "A", "B", "B"],
    ["A", "A", "B", "B", "B"],
    ["A", "A", "A", "A", "B"],
    ["B", "B", "B", "B", "B"],
    ["B", "B", "A", "B", "A"],
    ["A", "B", "B", "A", "B"],
]
MNHMM_MC_TIME_COLS = ["t1", "t2", "t3", "t4"]
MNHMM_MC_CH1_ROWS = [
    ["A", "A", "A", "A"],
    ["A", "B", "A", "A"],
    ["B", "B", "B", "B"],
    ["B", "A", "B", "B"],
]
MNHMM_MC_CH2_ROWS = [
    ["X", "X", "Y", "X"],
    ["X", "Y", "Y", "X"],
    ["Y", "Y", "Y", "Y"],
    ["Y", "X", "X", "Y"],
]
ALL_FAMILIES_TIME_COLS = [1, 2, 3, 4]
ALL_FAMILIES_STATES = ["A", "B", "C"]


def _mnhmm_seqdata(rows, ids):
    df = pd.DataFrame(rows, columns=MNHMM_TIME_COLS)
    df.insert(0, "id", ids)
    return SequenceData(df, time=MNHMM_TIME_COLS, states=["A", "B"], id_col="id")


def _mnhmm_fitted_component_seqdata():
    return _mnhmm_seqdata(
        MNHMM_FITTED_COMPONENT_ROWS,
        [f"s{i}" for i in range(6)],
    )


def _mnhmm_multichannel_seqdata():
    ids = [f"s{i}" for i in range(4)]
    ch1 = pd.DataFrame(MNHMM_MC_CH1_ROWS, columns=MNHMM_MC_TIME_COLS)
    ch1.insert(0, "id", ids)
    ch2 = pd.DataFrame(MNHMM_MC_CH2_ROWS, columns=MNHMM_MC_TIME_COLS)
    ch2.insert(0, "id", ids)
    return [
        SequenceData(ch1, time=MNHMM_MC_TIME_COLS, states=["A", "B"], id_col="id"),
        SequenceData(ch2, time=MNHMM_MC_TIME_COLS, states=["X", "Y"], id_col="id"),
    ]


def _all_families_panel():
    return pd.DataFrame(
        {
            "id": np.repeat([f"s{i}" for i in range(5)], 4),
            "time": np.tile(ALL_FAMILIES_TIME_COLS, 5),
            "group": np.repeat([-1.0, -0.5, 0.0, 0.5, 1.0], 4),
            "trend": np.tile(np.linspace(-1.0, 1.0, 4), 5),
            "activity": [
                "A",
                "B",
                "C",
                "A",
                "B",
                "B",
                "C",
                "A",
                "C",
                "C",
                "A",
                "B",
                "A",
                "C",
                "B",
                "B",
                "C",
                "A",
                "B",
                "C",
            ],
        }
    )


def _all_families_seqdata(panel):
    wide = panel.pivot(index="id", columns="time", values="activity").reset_index()
    return SequenceData(
        wide,
        time=ALL_FAMILIES_TIME_COLS,
        states=ALL_FAMILIES_STATES,
        id_col="id",
    )


def _split_reduced(values, n_clusters, shape):
    values = np.asarray(values, dtype=float)
    size = int(np.prod(shape))
    return [
        values[cluster_idx * size : (cluster_idx + 1) * size].reshape(shape, order="F")
        for cluster_idx in range(n_clusters)
    ]


def _load_csv_or_fail(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        pytest.fail(
            f"Missing generated R golden fixture: {path}. "
            "Regenerate it with the matching r_reference_mnhmm_*.R script."
        )


def _max_probability_frame_diff(observed, reference, keys):
    merged = observed.merge(
        reference,
        on=keys,
        suffixes=("_observed", "_reference"),
        validate="one_to_one",
    )
    assert len(merged) == len(observed) == len(reference)
    return float(np.max(np.abs(merged["probability_observed"] - merged["probability_reference"])))


def test_mnhmm_fixed_loglik_matches_seqhmm_golden(record_parity_result):
    seq = _mnhmm_seqdata(MNHMM_ROWS, [f"s{i}" for i in range(4)])
    refs = _load_csv_or_fail(f"{REF_ROOT}/mnhmm_fixed/ref_mnhmm_loglik.csv")
    reference = float(refs.loc[refs["key"] == "loglik", "value"].iloc[0])

    model = build_mnhmm(
        observations=seq,
        n_clusters=2,
        n_states=2,
        initial_probs=[np.array([0.95, 0.05]), np.array([0.05, 0.95])],
        transition_probs=[
            np.array([[0.98, 0.02], [0.10, 0.90]]),
            np.array([[0.90, 0.10], [0.02, 0.98]]),
        ],
        emission_probs=[
            np.array([[0.98, 0.02], [0.30, 0.70]]),
            np.array([[0.70, 0.30], [0.02, 0.98]]),
        ],
        cluster_probs=np.array([0.5, 0.5]),
        cluster_names=["Cluster 1", "Cluster 2"],
    )
    observed = float(model.score())
    diff = abs(observed - reference)
    passed = bool(diff <= LOGLIK_ATOL)

    record_parity_result(
        test_file="tests/seqhmm_parity/test_mnhmm_matrix.py",
        test_function="test_mnhmm_fixed_loglik_matches_seqhmm_golden",
        dataset_id="mnhmm_fixed_seed_1",
        metric="log_likelihood",
        tolerance=LOGLIK_ATOL,
        tolerance_kind="absolute",
        reference_value=reference,
        observed_value=observed,
        observed_diff=diff,
        label_alignment={"applied": False, "method": None, "permutation": None},
        passed=passed,
        prereg_yaml=PREREG,
    )

    assert passed


def test_mnhmm_fixed_cluster_posteriors_match_seqhmm_golden(record_parity_result):
    seq = _mnhmm_seqdata(MNHMM_ROWS, [f"s{i}" for i in range(4)])
    refs = _load_csv_or_fail(f"{REF_ROOT}/mnhmm_fixed/ref_mnhmm_cluster.csv")

    model = build_mnhmm(
        observations=seq,
        n_clusters=2,
        n_states=2,
        initial_probs=[np.array([0.95, 0.05]), np.array([0.05, 0.95])],
        transition_probs=[
            np.array([[0.98, 0.02], [0.10, 0.90]]),
            np.array([[0.90, 0.10], [0.02, 0.98]]),
        ],
        emission_probs=[
            np.array([[0.98, 0.02], [0.30, 0.70]]),
            np.array([[0.70, 0.30], [0.02, 0.98]]),
        ],
        cluster_probs=np.array([0.5, 0.5]),
        cluster_names=["Cluster 1", "Cluster 2"],
    )
    observed = pd.DataFrame(
        model.compute_responsibilities(),
        index=[f"s{i}" for i in range(4)],
        columns=["Cluster 1", "Cluster 2"],
    )
    reference = refs.pivot(index="id", columns="cluster", values="probability")
    reference = reference.loc[observed.index, observed.columns]
    max_diff = float(np.max(np.abs(observed.to_numpy() - reference.to_numpy())))
    passed = bool(max_diff <= LOGLIK_ATOL)

    record_parity_result(
        test_file="tests/seqhmm_parity/test_mnhmm_matrix.py",
        test_function="test_mnhmm_fixed_cluster_posteriors_match_seqhmm_golden",
        dataset_id="mnhmm_fixed_seed_1",
        metric="posterior_cluster_probability",
        tolerance=LOGLIK_ATOL,
        tolerance_kind="max_absolute",
        reference_value=None,
        observed_value=None,
        observed_diff=max_diff,
        label_alignment={"applied": False, "method": None, "permutation": None},
        passed=passed,
        prereg_yaml=PREREG,
    )

    assert passed


def _mnhmm_multichannel_fixed_model():
    return build_mnhmm(
        observations=_mnhmm_multichannel_seqdata(),
        n_clusters=2,
        n_states=2,
        initial_probs=[np.array([0.90, 0.10]), np.array([0.10, 0.90])],
        transition_probs=[
            np.array([[0.85, 0.15], [0.25, 0.75]]),
            np.array([[0.70, 0.30], [0.10, 0.90]]),
        ],
        emission_probs=[
            [
                np.array([[0.98, 0.02], [0.70, 0.30]]),
                np.array([[0.97, 0.03], [0.65, 0.35]]),
            ],
            [
                np.array([[0.30, 0.70], [0.02, 0.98]]),
                np.array([[0.35, 0.65], [0.03, 0.97]]),
            ],
        ],
        cluster_probs=np.array([0.55, 0.45]),
        cluster_names=["Cluster 1", "Cluster 2"],
        state_names=["State 1", "State 2"],
    )


def _mnhmm_multichannel_emission_covariate_model():
    channels = _mnhmm_multichannel_seqdata()
    X_B = np.ones((len(channels[0].sequences), len(MNHMM_MC_TIME_COLS), 2), dtype=float)
    X_B[:, :, 1] = np.linspace(-1.0, 1.0, len(MNHMM_MC_TIME_COLS))
    return build_mnhmm(
        observations=channels,
        n_clusters=2,
        n_states=2,
        initial_probs=[np.array([0.90, 0.10]), np.array([0.10, 0.90])],
        transition_probs=[
            np.array([[0.85, 0.15], [0.25, 0.75]]),
            np.array([[0.70, 0.30], [0.10, 0.90]]),
        ],
        X_B=X_B,
        eta_B_reduced=[
            [
                np.array([[[0.30, -0.10], [0.20, -0.25]]]),
                np.array([[[0.10, 0.25], [-0.15, 0.05]]]),
            ],
            [
                np.array([[[-0.20, 0.35], [0.15, -0.05]]]),
                np.array([[[0.25, -0.30], [-0.10, 0.20]]]),
            ],
        ],
        cluster_probs=np.array([0.55, 0.45]),
        cluster_names=["Cluster 1", "Cluster 2"],
        state_names=["State 1", "State 2"],
    )


def _mnhmm_multichannel_emission_probs_frame(model):
    rows = []
    for cluster_idx, cluster_name in enumerate(model.cluster_names):
        emissions = model._component_probs(cluster_idx)[2]
        for channel_idx, (alphabet, emission) in enumerate(
            zip(model.alphabets, emissions),
            start=1,
        ):
            channel_name = f"Channel {channel_idx}"
            for seq_idx, seq_id in enumerate(model.channels[0].ids):
                for time_idx, time_label in enumerate(model.channels[0].time, start=1):
                    for state_idx, state_name in enumerate(model.state_names[cluster_idx]):
                        for symbol_idx, symbol in enumerate(alphabet):
                            rows.append(
                                {
                                    "channel": channel_name,
                                    "cluster": cluster_name,
                                    "id": seq_id,
                                    "time": time_idx,
                                    "state": state_name,
                                    "symbol": symbol,
                                    "probability": emission[
                                        seq_idx, time_idx - 1, state_idx, symbol_idx
                                    ],
                                }
                            )
    return pd.DataFrame(rows)


def test_mnhmm_multichannel_fixed_loglik_matches_seqhmm_golden(record_parity_result):
    refs = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_multichannel_fixed/ref_mnhmm_multichannel_loglik.csv"
    )
    reference = float(refs.loc[refs["key"] == "loglik", "value"].iloc[0])
    model = _mnhmm_multichannel_fixed_model()
    observed = float(model.score())
    diff = abs(observed - reference)
    passed = bool(diff <= LOGLIK_ATOL)

    record_parity_result(
        test_file="tests/seqhmm_parity/test_mnhmm_matrix.py",
        test_function="test_mnhmm_multichannel_fixed_loglik_matches_seqhmm_golden",
        dataset_id="mnhmm_multichannel_fixed_seed_1",
        metric="log_likelihood",
        tolerance=LOGLIK_ATOL,
        tolerance_kind="absolute",
        reference_value=reference,
        observed_value=observed,
        observed_diff=diff,
        label_alignment={"applied": False, "method": None, "permutation": None},
        passed=passed,
        prereg_yaml=PREREG,
    )

    assert passed


def test_mnhmm_multichannel_fixed_posteriors_match_seqhmm_golden(record_parity_result):
    refs = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_multichannel_fixed/ref_mnhmm_multichannel_posterior_cluster.csv"
    )
    model = _mnhmm_multichannel_fixed_model()
    observed = pd.DataFrame(
        model.compute_responsibilities(),
        index=[f"s{i}" for i in range(4)],
        columns=["Cluster 1", "Cluster 2"],
    )
    reference = refs.pivot(index="id", columns="cluster", values="probability")
    reference = reference.loc[observed.index, observed.columns]
    max_diff = float(np.max(np.abs(observed.to_numpy() - reference.to_numpy())))
    passed = bool(max_diff <= LOGLIK_ATOL)

    record_parity_result(
        test_file="tests/seqhmm_parity/test_mnhmm_matrix.py",
        test_function="test_mnhmm_multichannel_fixed_posteriors_match_seqhmm_golden",
        dataset_id="mnhmm_multichannel_fixed_seed_1",
        metric="posterior_cluster_probability",
        tolerance=LOGLIK_ATOL,
        tolerance_kind="max_absolute",
        reference_value=None,
        observed_value=None,
        observed_diff=max_diff,
        label_alignment={"applied": False, "method": None, "permutation": None},
        passed=passed,
        prereg_yaml=PREREG,
    )

    assert passed


def test_mnhmm_multichannel_fixed_emissions_match_seqhmm_golden(record_parity_result):
    refs = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_multichannel_fixed/ref_mnhmm_multichannel_emission.csv"
    )
    model = _mnhmm_multichannel_fixed_model()
    observed = _mnhmm_multichannel_emission_probs_frame(model)
    key_cols = ["channel", "cluster", "id", "time", "state", "symbol"]
    observed = observed.sort_values(key_cols).reset_index(drop=True)
    reference = refs.sort_values(key_cols).reset_index(drop=True)
    max_diff = float(np.max(np.abs(observed["probability"] - reference["probability"])))
    passed = bool(max_diff <= LOGLIK_ATOL)

    record_parity_result(
        test_file="tests/seqhmm_parity/test_mnhmm_matrix.py",
        test_function="test_mnhmm_multichannel_fixed_emissions_match_seqhmm_golden",
        dataset_id="mnhmm_multichannel_fixed_seed_1",
        metric="emission_probability",
        tolerance=LOGLIK_ATOL,
        tolerance_kind="max_absolute",
        reference_value=None,
        observed_value=None,
        observed_diff=max_diff,
        label_alignment={"applied": False, "method": None, "permutation": None},
        passed=passed,
        prereg_yaml=PREREG,
    )

    assert passed


def test_mnhmm_multichannel_emission_covariate_matches_seqhmm_golden(record_parity_result):
    refs_loglik = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_multichannel_emission_covariate/ref_mnhmm_multichannel_emission_cov_loglik.csv"
    )
    refs_posterior = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_multichannel_emission_covariate/ref_mnhmm_multichannel_emission_cov_posterior_cluster.csv"
    )
    refs_emission = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_multichannel_emission_covariate/ref_mnhmm_multichannel_emission_cov_emission.csv"
    )
    model = _mnhmm_multichannel_emission_covariate_model()
    observed_loglik = float(model.score())
    reference_loglik = float(refs_loglik.loc[refs_loglik["key"] == "loglik", "value"].iloc[0])
    observed_posterior = pd.DataFrame(
        model.compute_responsibilities(),
        index=[f"s{i}" for i in range(4)],
        columns=["Cluster 1", "Cluster 2"],
    )
    reference_posterior = refs_posterior.pivot(
        index="id", columns="cluster", values="probability"
    )
    reference_posterior = reference_posterior.loc[
        observed_posterior.index, observed_posterior.columns
    ]
    key_cols = ["channel", "cluster", "id", "time", "state", "symbol"]
    observed_emission = (
        _mnhmm_multichannel_emission_probs_frame(model)
        .sort_values(key_cols)
        .reset_index(drop=True)
    )
    reference_emission = refs_emission.sort_values(key_cols).reset_index(drop=True)
    max_diff = float(
        max(
            abs(observed_loglik - reference_loglik),
            np.max(np.abs(observed_posterior.to_numpy() - reference_posterior.to_numpy())),
            np.max(np.abs(observed_emission["probability"] - reference_emission["probability"])),
        )
    )
    passed = bool(max_diff <= LOGLIK_ATOL)

    record_parity_result(
        test_file="tests/seqhmm_parity/test_mnhmm_matrix.py",
        test_function="test_mnhmm_multichannel_emission_covariate_matches_seqhmm_golden",
        dataset_id="mnhmm_multichannel_emission_covariate_seed_1",
        metric="multichannel_emission_covariate_bundle",
        tolerance=LOGLIK_ATOL,
        tolerance_kind="max_absolute",
        reference_value=None,
        observed_value=None,
        observed_diff=max_diff,
        label_alignment={"applied": False, "method": None, "permutation": None},
        passed=passed,
        prereg_yaml=PREREG,
    )

    assert passed


def _mnhmm_fixed_start_model():
    seq = _mnhmm_seqdata(MNHMM_ROWS, [f"s{i}" for i in range(4)])
    return build_mnhmm(
        observations=seq,
        n_clusters=2,
        n_states=2,
        initial_probs=[np.array([0.95, 0.05]), np.array([0.05, 0.95])],
        transition_probs=[
            np.array([[0.98, 0.02], [0.10, 0.90]]),
            np.array([[0.90, 0.10], [0.02, 0.98]]),
        ],
        emission_probs=[
            np.array([[0.98, 0.02], [0.30, 0.70]]),
            np.array([[0.70, 0.30], [0.02, 0.98]]),
        ],
        cluster_probs=np.array([0.5, 0.5]),
        cluster_names=["Cluster 1", "Cluster 2"],
        state_names=[["State 1", "State 2"], ["State 1", "State 2"]],
    )


def _mnhmm_cluster_covariate_model(seq):
    return build_mnhmm(
        observations=seq,
        n_clusters=2,
        n_states=2,
        initial_probs=[np.array([0.95, 0.05]), np.array([0.05, 0.95])],
        transition_probs=[
            np.array([[0.98, 0.02], [0.10, 0.90]]),
            np.array([[0.90, 0.10], [0.02, 0.98]]),
        ],
        emission_probs=[
            np.array([[0.98, 0.02], [0.30, 0.70]]),
            np.array([[0.70, 0.30], [0.02, 0.98]]),
        ],
        X_cluster=np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        ),
        eta_omega=np.array(
            [
                [SQRT2_INV, -SQRT2_INV],
                [-SQRT2_INV, SQRT2_INV],
            ]
        ),
        cluster_names=["Cluster 1", "Cluster 2"],
    )


def _mnhmm_emission_covariate_model(seq):
    X_B = np.ones((len(seq.sequences), len(MNHMM_TIME_COLS), 2), dtype=float)
    X_B[:, :, 1] = np.linspace(-1.0, 1.0, len(MNHMM_TIME_COLS))
    return build_mnhmm(
        observations=seq,
        n_clusters=2,
        n_states=2,
        initial_probs=[np.array([0.95, 0.05]), np.array([0.05, 0.95])],
        transition_probs=[
            np.array([[0.98, 0.02], [0.10, 0.90]]),
            np.array([[0.90, 0.10], [0.02, 0.98]]),
        ],
        X_B=X_B,
        eta_B_reduced=[
            np.array([[[0.8, 1.0], [-0.9, -1.1]]]),
            np.array([[[1.2, 1.4], [-1.3, -1.5]]]),
        ],
        eta_omega_reduced=np.array([[0.0]]),
        cluster_names=["Cluster 1", "Cluster 2"],
        state_names=[["State 1", "State 2"], ["State 1", "State 2"]],
    )


def _mnhmm_fitted_component_covariate_model(eta_B_reduced):
    seq = _mnhmm_fitted_component_seqdata()
    X_B = np.ones((len(seq.sequences), len(MNHMM_TIME_COLS), 2), dtype=float)
    X_B[:, :, 1] = np.linspace(-1.0, 1.0, len(MNHMM_TIME_COLS))
    return build_mnhmm(
        observations=seq,
        n_clusters=2,
        n_states=2,
        initial_probs=[np.array([0.50, 0.50]), np.array([0.50, 0.50])],
        transition_probs=[
            np.array([[0.55, 0.45], [0.45, 0.55]]),
            np.array([[0.55, 0.45], [0.45, 0.55]]),
        ],
        X_B=X_B,
        eta_B_reduced=eta_B_reduced,
        cluster_probs=np.array([0.5, 0.5]),
        cluster_names=["Cluster 1", "Cluster 2"],
        state_names=[["State 1", "State 2"], ["State 1", "State 2"]],
    )


def _mnhmm_emission_probs_frame(model):
    rows = []
    for cluster_idx, cluster_name in enumerate(model.cluster_names):
        emission = model._component_probs(cluster_idx)[2]
        for seq_idx, seq_id in enumerate(model.observations.ids):
            for time_idx, time_label in enumerate(MNHMM_TIME_COLS, start=1):
                for state_idx, state_name in enumerate(model.state_names[cluster_idx]):
                    for symbol_idx, symbol in enumerate(model.alphabet):
                        rows.append(
                            {
                                "cluster": cluster_name,
                                "id": seq_id,
                                "time": time_idx,
                                "state": state_name,
                                "activity": symbol,
                                "probability": emission[
                                    seq_idx, time_idx - 1, state_idx, symbol_idx
                                ],
                            }
                        )
    return pd.DataFrame(rows)


def _mnhmm_all_families_model():
    panel = _all_families_panel()
    seq = _all_families_seqdata(panel)
    state_names = ["State 1", "State 2", "State 3"]
    return build_mnhmm(
        observations=seq,
        n_clusters=3,
        n_states=3,
        emission_formula="activity ~ trend",
        initial_formula="~ group",
        transition_formula="~ trend",
        cluster_formula="~ group",
        data=panel,
        id_var="id",
        time_var="time",
        eta_pi_reduced=_split_reduced(np.linspace(-0.55, 0.55, 12), 3, (2, 2)),
        eta_A_reduced=_split_reduced(np.linspace(-0.45, 0.45, 36), 3, (2, 2, 3)),
        eta_B_reduced=_split_reduced(np.linspace(0.50, -0.50, 36), 3, (2, 2, 3)),
        eta_omega_reduced=np.array([[0.25, 0.45], [-0.35, -0.15]]),
        cluster_names=["Cluster 1", "Cluster 2", "Cluster 3"],
        state_names=[state_names, state_names, state_names],
    )


def _mnhmm_objective_gradient_reference(lambda_value: float, kind: str) -> pd.DataFrame:
    refs = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_objective_gradient/ref_mnhmm_objective_gradient.csv"
    )
    return refs[
        (np.isclose(refs["lambda"], lambda_value))
        & (refs["kind"] == kind)
    ].sort_values("index")


def _mnhmm_initial_probs_frame(model):
    rows = []
    for cluster_idx, cluster_name in enumerate(model.cluster_names):
        initial = model._component_probs(cluster_idx)[0]
        for seq_idx, seq_id in enumerate(model.observations.ids):
            for state_idx, state_name in enumerate(model.state_names[cluster_idx]):
                rows.append(
                    {
                        "cluster": cluster_name,
                        "id": seq_id,
                        "state": state_name,
                        "probability": initial[seq_idx, state_idx],
                    }
                )
    return pd.DataFrame(rows)


def _mnhmm_transition_probs_frame(model):
    rows = []
    for cluster_idx, cluster_name in enumerate(model.cluster_names):
        transition = model._component_probs(cluster_idx)[1]
        for seq_idx, seq_id in enumerate(model.observations.ids):
            for time_idx, time_label in enumerate(model.observations.time):
                for from_idx, from_name in enumerate(model.state_names[cluster_idx]):
                    for to_idx, to_name in enumerate(model.state_names[cluster_idx]):
                        rows.append(
                            {
                                "cluster": cluster_name,
                                "id": seq_id,
                                "time": time_label,
                                "state_from": from_name,
                                "state_to": to_name,
                                "probability": transition[seq_idx, time_idx, from_idx, to_idx],
                            }
                        )
    return pd.DataFrame(rows)


def _mnhmm_generic_emission_probs_frame(model):
    rows = []
    for cluster_idx, cluster_name in enumerate(model.cluster_names):
        emission = model._component_probs(cluster_idx)[2]
        for seq_idx, seq_id in enumerate(model.observations.ids):
            for time_idx, time_label in enumerate(model.observations.time):
                for state_idx, state_name in enumerate(model.state_names[cluster_idx]):
                    for symbol_idx, symbol in enumerate(model.alphabet):
                        rows.append(
                            {
                                "cluster": cluster_name,
                                "id": seq_id,
                                "time": time_label,
                                "state": state_name,
                                "activity": symbol,
                                "probability": emission[
                                    seq_idx, time_idx, state_idx, symbol_idx
                                ],
                            }
                        )
    return pd.DataFrame(rows)


def _mnhmm_cluster_probs_frame(model):
    return pd.DataFrame(
        [
            {
                "id": seq_id,
                "cluster": cluster_name,
                "probability": probs[cluster_idx],
            }
            for seq_id, probs in zip(model.observations.ids, model.compute_cluster_probs())
            for cluster_idx, cluster_name in enumerate(model.cluster_names)
        ]
    )


def _mnhmm_responsibility_frame(model):
    return pd.DataFrame(
        [
            {
                "id": seq_id,
                "cluster": cluster_name,
                "probability": probs[cluster_idx],
            }
            for seq_id, probs in zip(model.observations.ids, model.compute_responsibilities())
            for cluster_idx, cluster_name in enumerate(model.cluster_names)
        ]
    )


def test_mnhmm_one_step_em_matches_seqhmm_golden(record_parity_result):
    refs_meta = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_em_one_step/ref_mnhmm_em_one_step_meta.csv"
    )
    refs_initial = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_em_one_step/ref_mnhmm_em_one_step_initial.csv"
    )
    refs_transition = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_em_one_step/ref_mnhmm_em_one_step_transition.csv"
    )
    refs_emission = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_em_one_step/ref_mnhmm_em_one_step_emission.csv"
    )
    refs_prior = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_em_one_step/ref_mnhmm_em_one_step_prior_cluster.csv"
    )
    refs_posterior = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_em_one_step/ref_mnhmm_em_one_step_posterior_cluster.csv"
    )
    refs_transition["time"] = refs_transition["time"].map(lambda value: f"t{int(value)}")
    refs_emission["time"] = refs_emission["time"].map(lambda value: f"t{int(value)}")

    model = _mnhmm_fixed_start_model().fit(n_iter=1, tol=0.0)
    reference_loglik = float(refs_meta.loc[refs_meta["key"] == "loglik", "value"].iloc[0])
    reference_iterations = int(refs_meta.loc[refs_meta["key"] == "iterations", "value"].iloc[0])
    reference_return_code = int(refs_meta.loc[refs_meta["key"] == "return_code", "value"].iloc[0])
    assert reference_iterations == 1
    assert reference_return_code == 5
    assert model.n_iter == 1

    diffs = [
        abs(float(model.log_likelihood) - reference_loglik),
        _max_probability_frame_diff(
            _mnhmm_initial_probs_frame(model),
            refs_initial,
            ["cluster", "id", "state"],
        ),
        _max_probability_frame_diff(
            _mnhmm_transition_probs_frame(model),
            refs_transition,
            ["cluster", "id", "time", "state_from", "state_to"],
        ),
        _max_probability_frame_diff(
            _mnhmm_generic_emission_probs_frame(model),
            refs_emission,
            ["cluster", "id", "time", "state", "activity"],
        ),
        _max_probability_frame_diff(
            _mnhmm_cluster_probs_frame(model),
            refs_prior,
            ["id", "cluster"],
        ),
        _max_probability_frame_diff(
            _mnhmm_responsibility_frame(model),
            refs_posterior,
            ["id", "cluster"],
        ),
    ]
    max_diff = float(max(diffs))
    passed = bool(max_diff <= LOGLIK_ATOL)

    record_parity_result(
        test_file="tests/seqhmm_parity/test_mnhmm_matrix.py",
        test_function="test_mnhmm_one_step_em_matches_seqhmm_golden",
        dataset_id="mnhmm_em_one_step_seed_1",
        metric="one_step_em_probability_bundle",
        tolerance=LOGLIK_ATOL,
        tolerance_kind="max_absolute",
        reference_value=None,
        observed_value=None,
        observed_diff=max_diff,
        label_alignment={"applied": False, "method": None, "permutation": None},
        passed=passed,
        prereg_yaml=PREREG,
    )

    assert passed


def test_mnhmm_cluster_covariate_loglik_matches_seqhmm_golden(record_parity_result):
    seq = _mnhmm_seqdata(MNHMM_ROWS, [f"s{i}" for i in range(4)])
    refs = _load_csv_or_fail(f"{REF_ROOT}/mnhmm_cluster_covariate/ref_mnhmm_cov_loglik.csv")
    reference = float(refs.loc[refs["key"] == "loglik", "value"].iloc[0])

    observed = float(_mnhmm_cluster_covariate_model(seq).score())
    diff = abs(observed - reference)
    passed = bool(diff <= LOGLIK_ATOL)

    record_parity_result(
        test_file="tests/seqhmm_parity/test_mnhmm_matrix.py",
        test_function="test_mnhmm_cluster_covariate_loglik_matches_seqhmm_golden",
        dataset_id="mnhmm_cluster_covariate_seed_1",
        metric="log_likelihood",
        tolerance=LOGLIK_ATOL,
        tolerance_kind="absolute",
        reference_value=reference,
        observed_value=observed,
        observed_diff=diff,
        label_alignment={"applied": False, "method": None, "permutation": None},
        passed=passed,
        prereg_yaml=PREREG,
    )

    assert passed


def test_mnhmm_emission_covariate_loglik_matches_seqhmm_golden(record_parity_result):
    seq = _mnhmm_seqdata(MNHMM_ROWS, [f"s{i}" for i in range(4)])
    refs = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_emission_covariate/ref_mnhmm_emission_cov_loglik.csv"
    )
    reference = float(refs.loc[refs["key"] == "loglik", "value"].iloc[0])

    observed = float(_mnhmm_emission_covariate_model(seq).score())
    diff = abs(observed - reference)
    passed = bool(diff <= LOGLIK_ATOL)

    record_parity_result(
        test_file="tests/seqhmm_parity/test_mnhmm_matrix.py",
        test_function="test_mnhmm_emission_covariate_loglik_matches_seqhmm_golden",
        dataset_id="mnhmm_emission_covariate_seed_1",
        metric="log_likelihood",
        tolerance=LOGLIK_ATOL,
        tolerance_kind="absolute",
        reference_value=reference,
        observed_value=observed,
        observed_diff=diff,
        label_alignment={"applied": False, "method": None, "permutation": None},
        passed=passed,
        prereg_yaml=PREREG,
    )

    assert passed


def test_mnhmm_emission_covariate_posteriors_match_seqhmm_golden(record_parity_result):
    seq = _mnhmm_seqdata(MNHMM_ROWS, [f"s{i}" for i in range(4)])
    refs = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_emission_covariate/ref_mnhmm_emission_cov_posterior_cluster.csv"
    )
    model = _mnhmm_emission_covariate_model(seq)
    observed = pd.DataFrame(
        model.compute_responsibilities(),
        index=[f"s{i}" for i in range(4)],
        columns=["Cluster 1", "Cluster 2"],
    )
    reference = refs.pivot(index="id", columns="cluster", values="probability")
    reference = reference.loc[observed.index, observed.columns]
    max_diff = float(np.max(np.abs(observed.to_numpy() - reference.to_numpy())))
    passed = bool(max_diff <= LOGLIK_ATOL)

    record_parity_result(
        test_file="tests/seqhmm_parity/test_mnhmm_matrix.py",
        test_function="test_mnhmm_emission_covariate_posteriors_match_seqhmm_golden",
        dataset_id="mnhmm_emission_covariate_seed_1",
        metric="posterior_cluster_probability",
        tolerance=LOGLIK_ATOL,
        tolerance_kind="max_absolute",
        reference_value=None,
        observed_value=None,
        observed_diff=max_diff,
        label_alignment={"applied": False, "method": None, "permutation": None},
        passed=passed,
        prereg_yaml=PREREG,
    )

    assert passed


def test_mnhmm_emission_covariate_emissions_match_seqhmm_golden(record_parity_result):
    seq = _mnhmm_seqdata(MNHMM_ROWS, [f"s{i}" for i in range(4)])
    refs = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_emission_covariate/ref_mnhmm_emission_cov_emission.csv"
    )
    model = _mnhmm_emission_covariate_model(seq)
    observed = _mnhmm_emission_probs_frame(model)
    merged = observed.merge(
        refs,
        on=["cluster", "id", "time", "state", "activity"],
        suffixes=("_observed", "_reference"),
        validate="one_to_one",
    )
    assert len(merged) == len(observed) == len(refs)
    max_diff = float(
        np.max(np.abs(merged["probability_observed"] - merged["probability_reference"]))
    )
    passed = bool(max_diff <= LOGLIK_ATOL)

    record_parity_result(
        test_file="tests/seqhmm_parity/test_mnhmm_matrix.py",
        test_function="test_mnhmm_emission_covariate_emissions_match_seqhmm_golden",
        dataset_id="mnhmm_emission_covariate_seed_1",
        metric="emission_probability",
        tolerance=LOGLIK_ATOL,
        tolerance_kind="max_absolute",
        reference_value=None,
        observed_value=None,
        observed_diff=max_diff,
        label_alignment={"applied": False, "method": None, "permutation": None},
        passed=passed,
        prereg_yaml=PREREG,
    )

    assert passed


def test_mnhmm_all_families_reduced_loglik_matches_seqhmm_golden(record_parity_result):
    refs = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_all_families_reduced/ref_mnhmm_all_families_loglik.csv"
    )
    reference = float(refs.loc[refs["key"] == "loglik", "value"].iloc[0])

    observed = float(_mnhmm_all_families_model().score())
    diff = abs(observed - reference)
    passed = bool(diff <= LOGLIK_ATOL)

    record_parity_result(
        test_file="tests/seqhmm_parity/test_mnhmm_matrix.py",
        test_function="test_mnhmm_all_families_reduced_loglik_matches_seqhmm_golden",
        dataset_id="mnhmm_all_families_reduced_seed_1",
        metric="log_likelihood",
        tolerance=LOGLIK_ATOL,
        tolerance_kind="absolute",
        reference_value=reference,
        observed_value=observed,
        observed_diff=diff,
        label_alignment={"applied": False, "method": None, "permutation": None},
        passed=passed,
        prereg_yaml=PREREG,
    )

    assert passed


def test_mnhmm_all_families_reduced_initial_probs_match_seqhmm_golden(record_parity_result):
    refs = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_all_families_reduced/ref_mnhmm_all_families_initial.csv"
    )
    max_diff = _max_probability_frame_diff(
        _mnhmm_initial_probs_frame(_mnhmm_all_families_model()),
        refs,
        ["cluster", "id", "state"],
    )
    passed = bool(max_diff <= LOGLIK_ATOL)

    record_parity_result(
        test_file="tests/seqhmm_parity/test_mnhmm_matrix.py",
        test_function="test_mnhmm_all_families_reduced_initial_probs_match_seqhmm_golden",
        dataset_id="mnhmm_all_families_reduced_seed_1",
        metric="initial_probability",
        tolerance=LOGLIK_ATOL,
        tolerance_kind="max_absolute",
        reference_value=None,
        observed_value=None,
        observed_diff=max_diff,
        label_alignment={"applied": False, "method": None, "permutation": None},
        passed=passed,
        prereg_yaml=PREREG,
    )

    assert passed


def test_mnhmm_all_families_reduced_transitions_match_seqhmm_golden(record_parity_result):
    refs = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_all_families_reduced/ref_mnhmm_all_families_transition.csv"
    )
    max_diff = _max_probability_frame_diff(
        _mnhmm_transition_probs_frame(_mnhmm_all_families_model()),
        refs,
        ["cluster", "id", "time", "state_from", "state_to"],
    )
    passed = bool(max_diff <= LOGLIK_ATOL)

    record_parity_result(
        test_file="tests/seqhmm_parity/test_mnhmm_matrix.py",
        test_function="test_mnhmm_all_families_reduced_transitions_match_seqhmm_golden",
        dataset_id="mnhmm_all_families_reduced_seed_1",
        metric="transition_probability",
        tolerance=LOGLIK_ATOL,
        tolerance_kind="max_absolute",
        reference_value=None,
        observed_value=None,
        observed_diff=max_diff,
        label_alignment={"applied": False, "method": None, "permutation": None},
        passed=passed,
        prereg_yaml=PREREG,
    )

    assert passed


def test_mnhmm_all_families_reduced_emissions_match_seqhmm_golden(record_parity_result):
    refs = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_all_families_reduced/ref_mnhmm_all_families_emission.csv"
    )
    max_diff = _max_probability_frame_diff(
        _mnhmm_generic_emission_probs_frame(_mnhmm_all_families_model()),
        refs,
        ["cluster", "id", "time", "state", "activity"],
    )
    passed = bool(max_diff <= LOGLIK_ATOL)

    record_parity_result(
        test_file="tests/seqhmm_parity/test_mnhmm_matrix.py",
        test_function="test_mnhmm_all_families_reduced_emissions_match_seqhmm_golden",
        dataset_id="mnhmm_all_families_reduced_seed_1",
        metric="emission_probability",
        tolerance=LOGLIK_ATOL,
        tolerance_kind="max_absolute",
        reference_value=None,
        observed_value=None,
        observed_diff=max_diff,
        label_alignment={"applied": False, "method": None, "permutation": None},
        passed=passed,
        prereg_yaml=PREREG,
    )

    assert passed


def test_mnhmm_all_families_reduced_priors_match_seqhmm_golden(record_parity_result):
    refs = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_all_families_reduced/ref_mnhmm_all_families_prior_cluster.csv"
    )
    max_diff = _max_probability_frame_diff(
        _mnhmm_cluster_probs_frame(_mnhmm_all_families_model()),
        refs,
        ["id", "cluster"],
    )
    passed = bool(max_diff <= LOGLIK_ATOL)

    record_parity_result(
        test_file="tests/seqhmm_parity/test_mnhmm_matrix.py",
        test_function="test_mnhmm_all_families_reduced_priors_match_seqhmm_golden",
        dataset_id="mnhmm_all_families_reduced_seed_1",
        metric="prior_cluster_probability",
        tolerance=LOGLIK_ATOL,
        tolerance_kind="max_absolute",
        reference_value=None,
        observed_value=None,
        observed_diff=max_diff,
        label_alignment={"applied": False, "method": None, "permutation": None},
        passed=passed,
        prereg_yaml=PREREG,
    )

    assert passed


def test_mnhmm_all_families_reduced_posteriors_match_seqhmm_golden(record_parity_result):
    refs = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_all_families_reduced/ref_mnhmm_all_families_posterior_cluster.csv"
    )
    max_diff = _max_probability_frame_diff(
        _mnhmm_responsibility_frame(_mnhmm_all_families_model()),
        refs,
        ["id", "cluster"],
    )
    passed = bool(max_diff <= LOGLIK_ATOL)

    record_parity_result(
        test_file="tests/seqhmm_parity/test_mnhmm_matrix.py",
        test_function="test_mnhmm_all_families_reduced_posteriors_match_seqhmm_golden",
        dataset_id="mnhmm_all_families_reduced_seed_1",
        metric="posterior_cluster_probability",
        tolerance=LOGLIK_ATOL,
        tolerance_kind="max_absolute",
        reference_value=None,
        observed_value=None,
        observed_diff=max_diff,
        label_alignment={"applied": False, "method": None, "permutation": None},
        passed=passed,
        prereg_yaml=PREREG,
    )

    assert passed


@pytest.mark.parametrize("lambda_value", [0.0, 0.2])
def test_mnhmm_reduced_objective_matches_seqhmm_golden(lambda_value, record_parity_result):
    objective_ref = _mnhmm_objective_gradient_reference(lambda_value, "objective")
    parameter_ref = _mnhmm_objective_gradient_reference(lambda_value, "parameter")
    model = _mnhmm_all_families_model()

    observed = model.objective_and_gradient(lambda_penalty=lambda_value)
    observed_objective = float(observed["objective"])
    observed_params = np.asarray(observed["parameters"], dtype=float)
    reference_objective = float(objective_ref["value"].iloc[0])
    reference_params = parameter_ref["value"].to_numpy(dtype=float)
    objective_diff = abs(observed_objective - reference_objective)
    parameter_diff = float(np.max(np.abs(observed_params - reference_params)))
    passed = bool(objective_diff <= LOGLIK_ATOL and parameter_diff <= LOGLIK_ATOL)

    record_parity_result(
        test_file="tests/seqhmm_parity/test_mnhmm_matrix.py",
        test_function="test_mnhmm_reduced_objective_matches_seqhmm_golden",
        dataset_id=f"mnhmm_objective_gradient_lambda_{lambda_value:g}_seed_1",
        metric="objective",
        tolerance=LOGLIK_ATOL,
        tolerance_kind="absolute",
        reference_value=reference_objective,
        observed_value=observed_objective,
        observed_diff=max(objective_diff, parameter_diff),
        label_alignment={"applied": False, "method": None, "permutation": None},
        passed=passed,
        prereg_yaml=PREREG,
    )

    assert passed


@pytest.mark.parametrize("lambda_value", [0.0, 0.2])
def test_mnhmm_reduced_objective_gradient_matches_seqhmm_golden(
    lambda_value, record_parity_result
):
    gradient_ref = _mnhmm_objective_gradient_reference(lambda_value, "gradient")
    model = _mnhmm_all_families_model()

    observed = model.objective_and_gradient(lambda_penalty=lambda_value)
    observed_gradient = np.asarray(observed["gradient"], dtype=float)
    reference_gradient = gradient_ref["value"].to_numpy(dtype=float)
    max_diff = float(np.max(np.abs(observed_gradient - reference_gradient)))
    passed = bool(max_diff <= LOGLIK_ATOL)

    record_parity_result(
        test_file="tests/seqhmm_parity/test_mnhmm_matrix.py",
        test_function="test_mnhmm_reduced_objective_gradient_matches_seqhmm_golden",
        dataset_id=f"mnhmm_objective_gradient_lambda_{lambda_value:g}_seed_1",
        metric="objective_gradient",
        tolerance=LOGLIK_ATOL,
        tolerance_kind="max_absolute",
        reference_value=None,
        observed_value=None,
        observed_diff=max_diff,
        label_alignment={"applied": False, "method": None, "permutation": None},
        passed=passed,
        prereg_yaml=PREREG,
    )

    assert passed


def test_mnhmm_cluster_covariate_priors_match_seqhmm_golden(record_parity_result):
    seq = _mnhmm_seqdata(MNHMM_ROWS, [f"s{i}" for i in range(4)])
    refs = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_cluster_covariate/ref_mnhmm_cov_prior_cluster.csv"
    )
    model = _mnhmm_cluster_covariate_model(seq)
    observed = pd.DataFrame(
        model.compute_cluster_probs(),
        index=[f"s{i}" for i in range(4)],
        columns=["Cluster 1", "Cluster 2"],
    )
    reference = refs.pivot(index="id", columns="cluster", values="probability")
    reference = reference.loc[observed.index, observed.columns]
    max_diff = float(np.max(np.abs(observed.to_numpy() - reference.to_numpy())))
    passed = bool(max_diff <= LOGLIK_ATOL)

    record_parity_result(
        test_file="tests/seqhmm_parity/test_mnhmm_matrix.py",
        test_function="test_mnhmm_cluster_covariate_priors_match_seqhmm_golden",
        dataset_id="mnhmm_cluster_covariate_seed_1",
        metric="prior_cluster_probability",
        tolerance=LOGLIK_ATOL,
        tolerance_kind="max_absolute",
        reference_value=None,
        observed_value=None,
        observed_diff=max_diff,
        label_alignment={"applied": False, "method": None, "permutation": None},
        passed=passed,
        prereg_yaml=PREREG,
    )

    assert passed


def test_mnhmm_cluster_covariate_posteriors_match_seqhmm_golden(record_parity_result):
    seq = _mnhmm_seqdata(MNHMM_ROWS, [f"s{i}" for i in range(4)])
    refs = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_cluster_covariate/ref_mnhmm_cov_posterior_cluster.csv"
    )
    model = _mnhmm_cluster_covariate_model(seq)
    observed = pd.DataFrame(
        model.compute_responsibilities(),
        index=[f"s{i}" for i in range(4)],
        columns=["Cluster 1", "Cluster 2"],
    )
    reference = refs.pivot(index="id", columns="cluster", values="probability")
    reference = reference.loc[observed.index, observed.columns]
    max_diff = float(np.max(np.abs(observed.to_numpy() - reference.to_numpy())))
    passed = bool(max_diff <= LOGLIK_ATOL)

    record_parity_result(
        test_file="tests/seqhmm_parity/test_mnhmm_matrix.py",
        test_function="test_mnhmm_cluster_covariate_posteriors_match_seqhmm_golden",
        dataset_id="mnhmm_cluster_covariate_seed_1",
        metric="posterior_cluster_probability",
        tolerance=LOGLIK_ATOL,
        tolerance_kind="max_absolute",
        reference_value=None,
        observed_value=None,
        observed_diff=max_diff,
        label_alignment={"applied": False, "method": None, "permutation": None},
        passed=passed,
        prereg_yaml=PREREG,
    )

    assert passed


def test_mnhmm_fitted_cluster_covariate_matches_seqhmm_golden(record_parity_result):
    seq = _mnhmm_seqdata(MNHMM_ROWS, [f"s{i}" for i in range(4)])
    refs_meta = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_cluster_covariate_fitted/ref_mnhmm_fitted_cov_meta.csv"
    )
    refs_prior = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_cluster_covariate_fitted/ref_mnhmm_fitted_cov_prior_cluster.csv"
    )
    refs_posterior = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_cluster_covariate_fitted/ref_mnhmm_fitted_cov_posterior_cluster.csv"
    )
    refs_eta = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_cluster_covariate_fitted/ref_mnhmm_fitted_cov_eta_omega.csv"
    )

    fitted = estimate_mnhmm(
        observations=seq,
        n_clusters=2,
        n_states=2,
        initial_probs=[np.array([0.95, 0.05]), np.array([0.05, 0.95])],
        transition_probs=[
            np.array([[0.98, 0.02], [0.10, 0.90]]),
            np.array([[0.90, 0.10], [0.02, 0.98]]),
        ],
        emission_probs=[
            np.array([[0.98, 0.02], [0.30, 0.70]]),
            np.array([[0.70, 0.30], [0.02, 0.98]]),
        ],
        X_cluster=np.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 0.0],
            ]
        ),
        eta_omega_reduced=np.array([[0.0, 0.0]]),
        n_iter=400,
        tol=1e-12,
    )

    reference_loglik = float(refs_meta.loc[refs_meta["key"] == "loglik", "value"].iloc[0])
    observed_eta_reduced = (
        (fitted.eta_omega[:, 0] - fitted.eta_omega[:, 1]) * SQRT2_INV
    )
    reference_eta = refs_eta.sort_values("covariate")["value"].to_numpy(dtype=float)
    diffs = [
        abs(float(fitted.log_likelihood) - reference_loglik),
        _max_probability_frame_diff(
            _mnhmm_cluster_probs_frame(fitted),
            refs_prior,
            ["id", "cluster"],
        ),
        _max_probability_frame_diff(
            _mnhmm_responsibility_frame(fitted),
            refs_posterior,
            ["id", "cluster"],
        ),
        float(np.max(np.abs(observed_eta_reduced - reference_eta))),
    ]
    max_diff = float(max(diffs))
    passed = bool(max_diff <= 2e-5)

    record_parity_result(
        test_file="tests/seqhmm_parity/test_mnhmm_matrix.py",
        test_function="test_mnhmm_fitted_cluster_covariate_matches_seqhmm_golden",
        dataset_id="mnhmm_cluster_covariate_fitted_seed_1",
        metric="fitted_cluster_covariate_bundle",
        tolerance=2e-5,
        tolerance_kind="max_absolute",
        reference_value=None,
        observed_value=None,
        observed_diff=max_diff,
        label_alignment={"applied": False, "method": None, "permutation": None},
        passed=passed,
        prereg_yaml=PREREG,
    )

    assert passed


def test_mnhmm_fitted_component_covariate_matches_seqhmm_golden(record_parity_result):
    refs_meta = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_component_covariate_fitted/ref_mnhmm_fitted_component_meta.csv"
    )
    refs_eta = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_component_covariate_fitted/ref_mnhmm_fitted_component_eta_B.csv"
    )
    refs_emission = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_component_covariate_fitted/ref_mnhmm_fitted_component_emission.csv"
    )
    refs_posterior = _load_csv_or_fail(
        f"{REF_ROOT}/mnhmm_component_covariate_fitted/ref_mnhmm_fitted_component_posterior_cluster.csv"
    )

    reference_loglik = float(refs_meta.loc[refs_meta["key"] == "loglik", "value"].iloc[0])
    reference_eta = refs_eta.sort_values("index")["value"].to_numpy(dtype=float)
    seq = _mnhmm_fitted_component_seqdata()
    X_B = np.ones((len(seq.sequences), len(MNHMM_TIME_COLS), 2), dtype=float)
    X_B[:, :, 1] = np.linspace(-1.0, 1.0, len(MNHMM_TIME_COLS))
    fitted = estimate_mnhmm(
        observations=seq,
        n_clusters=2,
        n_states=2,
        initial_probs=[np.array([0.50, 0.50]), np.array([0.50, 0.50])],
        transition_probs=[
            np.array([[0.55, 0.45], [0.45, 0.55]]),
            np.array([[0.55, 0.45], [0.45, 0.55]]),
        ],
        X_B=X_B,
        eta_B_reduced=[np.zeros((1, 2, 2)), np.zeros((1, 2, 2))],
        cluster_probs=np.array([0.5, 0.5]),
        cluster_names=["Cluster 1", "Cluster 2"],
        state_names=[["State 1", "State 2"], ["State 1", "State 2"]],
        n_iter=1000,
        tol=1e-9,
        lambda_penalty=10.0,
    )
    assert fitted.estimation_method == "direct_l_bfgs"
    assert fitted.optimization_result.nfev == fitted.optimization_result.njev
    observed_eta = np.concatenate(
        [
            _reduce_eta_B_full(fitted.eta_B[cluster_idx]).ravel(order="F")
            for cluster_idx in range(fitted.n_clusters)
        ]
    )
    observed_loglik = float(fitted.score())
    diffs = [
        abs(observed_loglik - reference_loglik),
        float(np.max(np.abs(observed_eta - reference_eta))),
        _max_probability_frame_diff(
            _mnhmm_emission_probs_frame(fitted),
            refs_emission,
            ["cluster", "id", "time", "state", "activity"],
        ),
        _max_probability_frame_diff(
            _mnhmm_responsibility_frame(fitted),
            refs_posterior,
            ["id", "cluster"],
        ),
    ]
    max_diff = float(max(diffs))
    passed = bool(max_diff <= 5e-5)

    record_parity_result(
        test_file="tests/seqhmm_parity/test_mnhmm_matrix.py",
        test_function="test_mnhmm_fitted_component_covariate_matches_seqhmm_golden",
        dataset_id="mnhmm_component_covariate_fitted_seed_1",
        metric="fitted_component_covariate_bundle",
        tolerance=5e-5,
        tolerance_kind="max_absolute",
        reference_value=None,
        observed_value=None,
        observed_diff=max_diff,
        label_alignment={"applied": False, "method": None, "permutation": None},
        passed=passed,
        prereg_yaml=PREREG,
    )

    assert passed
