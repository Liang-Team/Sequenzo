"""
@Author  : Yuqi Liang 梁彧祺
@File    : feature_extraction_and_selection_pipeline.py
@Time    : 24/03/2026 22:41
@Desc    :
    End-to-end generic Feature Extraction & Selection pipeline:
    monthly states -> spells -> duration/timing/sequencing features -> Boruta -> final model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score

from sequenzo.define_sequence_data import SequenceData

from .boruta_feature_selection import select_all_relevant_features_boruta


@dataclass(frozen=True)
class FeatureExtractionAndSelectionConfig:
    sequencing_max_k: int = 3
    sequencing_min_support: Union[int, float] = 0.05
    sequencing_top_mined_subsequences: Optional[int] = 1000
    sequencing_count_method: str = "presence"
    sequencing_event_label_mode: str = "state"

    timing_bin_width: float = 12.0
    timing_include_start: bool = True
    timing_include_end: bool = False
    timing_count_method: str = "any"
    timing_bin_include_left: bool = True

    boruta_n_iter: int = 50
    boruta_perc: float = 100.0

    residualize_target_with_controls: bool = True
    include_controls_in_final_model: bool = True


def get_feature_extraction_and_selection_config_preset(
    preset: str,
) -> FeatureExtractionAndSelectionConfig:
    """
    Return a named configuration preset for reproducible FES runs.

    Parameters
    ----------
    preset : str
        Preset name. Currently supported:
        - "unterlerchner2023"

    Returns
    -------
    FeatureExtractionAndSelectionConfig
        A frozen config object with preset parameters.
    """
    key = preset.strip().lower()
    if key == "unterlerchner2023":
        return FeatureExtractionAndSelectionConfig(
            sequencing_max_k=3,
            sequencing_min_support=0.05,
            sequencing_top_mined_subsequences=1000,
            sequencing_count_method="presence",
            sequencing_event_label_mode="state",
            timing_bin_width=12.0,
            timing_include_start=True,
            timing_include_end=False,
            timing_count_method="any",
            timing_bin_include_left=True,
            boruta_n_iter=50,
            boruta_perc=100.0,
            residualize_target_with_controls=True,
            include_controls_in_final_model=True,
        )
    raise ValueError(
        f"Unknown preset '{preset}'. Available presets: unterlerchner2023"
    )


def _infer_problem_type(y: np.ndarray) -> str:
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError("y must be 1D.")
    if np.issubdtype(y_arr.dtype, np.floating):
        uniq = np.unique(y_arr[~np.isnan(y_arr)])
        return "classification" if uniq.size <= 15 else "regression"
    uniq = np.unique(y_arr[~pd.isna(y_arr)])
    return "classification" if uniq.size <= 15 else "regression"


def _prepare_control_matrix(
    controls: Optional[Union[pd.DataFrame, np.ndarray]],
    n: int,
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    if controls is None:
        return None, None
    if isinstance(controls, pd.DataFrame):
        if controls.shape[0] != n:
            raise ValueError("controls rows must match y length.")
        return controls.to_numpy(dtype=float), controls.columns.tolist()
    arr = np.asarray(controls)
    if arr.ndim != 2 or arr.shape[0] != n:
        raise ValueError("controls must be 2D with the same number of rows as y.")
    return arr.astype(float), [f"CTRL_{j}" for j in range(arr.shape[1])]


def run_feature_extraction_and_selection_pipeline(
    seqdata: SequenceData,
    outcome: Union[Sequence[Any], np.ndarray, pd.Series],
    *,
    controls: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    sample_weights: Optional[Union[np.ndarray, pd.Series]] = None,
    state_groups: Optional[Dict[str, List[Any]]] = None,
    problem_type: Optional[str] = None,
    config: Optional[FeatureExtractionAndSelectionConfig] = None,
    preset: Optional[str] = None,
    ids: Optional[Sequence[Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    from .duration_timing_feature_builders import (
        build_duration_features,
        build_timing_features,
    )
    from .monthly_state_to_spells import extract_spells_with_times
    from .sequencing_feature_builders import build_sequencing_features
    from .time_binning_utils import coerce_numeric_time_labels, make_equal_width_bins

    if config is not None and preset is not None:
        raise ValueError("Provide either 'config' or 'preset', not both.")
    cfg = (
        config
        if config is not None
        else (
            get_feature_extraction_and_selection_config_preset(preset)
            if preset is not None
            else FeatureExtractionAndSelectionConfig()
        )
    )

    y_raw = np.asarray(outcome)
    if y_raw.ndim != 1:
        raise ValueError("outcome must be 1D.")
    n = len(y_raw)
    if seqdata.n_sequences != n:
        raise ValueError("Length of outcome must match number of sequences.")

    if problem_type is None or problem_type == "auto":
        problem_type = _infer_problem_type(y_raw)
    if problem_type not in {"regression", "classification"}:
        raise ValueError("problem_type must be 'regression' or 'classification'.")

    if problem_type == "regression":
        y = y_raw.astype(float)
    else:
        if np.issubdtype(y_raw.dtype, np.number):
            if np.issubdtype(y_raw.dtype, np.floating) and np.allclose(y_raw, np.round(y_raw), equal_nan=False):
                y = np.round(y_raw).astype(int)
            else:
                y = pd.Categorical(y_raw).codes
        else:
            y = pd.Categorical(y_raw).codes

    if sample_weights is not None:
        w = np.asarray(sample_weights, dtype=float)
        if w.ndim != 1 or len(w) != n:
            raise ValueError("sample_weights must be a 1D array with length n.")
    else:
        w = None

    X_controls, _ = _prepare_control_matrix(controls, n)
    if verbose:
        print(f"[FES] problem_type={problem_type}, n={n}")

    spells_per_individual = extract_spells_with_times(seqdata)
    if state_groups is None:
        state_groups = {str(s): [s] for s in seqdata.states}

    X_duration, duration_feature_names = build_duration_features(spells_per_individual, state_groups=state_groups)

    numeric_time = coerce_numeric_time_labels(seqdata.time)
    time_bins = make_equal_width_bins(float(np.min(numeric_time)), float(np.max(numeric_time)), cfg.timing_bin_width)
    X_timing, timing_feature_names = build_timing_features(
        spells_per_individual,
        state_groups=state_groups,
        start_bins=time_bins,
        include_start=cfg.timing_include_start,
        include_end=cfg.timing_include_end,
        count_method=cfg.timing_count_method,
        bin_include_left=cfg.timing_bin_include_left,
    )

    X_sequencing, sequencing_feature_names = build_sequencing_features(
        spells_per_individual,
        id_values=ids,
        max_k=cfg.sequencing_max_k,
        min_support=cfg.sequencing_min_support,
        count_method=cfg.sequencing_count_method,
        top_mined_subsequences=cfg.sequencing_top_mined_subsequences,
        event_label_mode=cfg.sequencing_event_label_mode,
        use_start_time=True,
        weighted=(w is not None),
    )

    all_feature_names = duration_feature_names + timing_feature_names + sequencing_feature_names
    X_full = np.hstack([X_duration, X_timing, X_sequencing]).astype(float)

    y_for_selection = y
    if cfg.residualize_target_with_controls and X_controls is not None and problem_type == "regression":
        X_design = sm.add_constant(X_controls, has_constant="add")
        residual_model = sm.WLS(y, X_design, weights=w).fit() if w is not None else sm.OLS(y, X_design).fit()
        y_for_selection = y - residual_model.predict(X_design)

    boruta_result = select_all_relevant_features_boruta(
        X_full,
        y_for_selection.astype(float) if problem_type == "regression" else y_for_selection,
        problem_type=problem_type,
        n_iter=cfg.boruta_n_iter,
        perc=cfg.boruta_perc,
        random_state=42,
        verbose=verbose,
    )
    selected_mask = boruta_result.selected_mask
    selected_feature_names = [all_feature_names[j] for j in range(len(all_feature_names)) if selected_mask[j]]
    if len(selected_feature_names) == 0:
        raise RuntimeError("Boruta selected zero features. Consider relaxing selection parameters.")
    X_selected = X_full[:, selected_mask]

    result: Dict[str, Any] = {
        "problem_type": problem_type,
        "n": n,
        "all_feature_names": all_feature_names,
        "selected_feature_names": selected_feature_names,
        "selected_mask": selected_mask,
        "X_selected": X_selected,
    }

    if problem_type == "regression":
        X_final = X_selected if (not cfg.include_controls_in_final_model or X_controls is None) else np.hstack([X_controls, X_selected])
        X_final_design = sm.add_constant(X_final, has_constant="add")
        final_model = sm.WLS(y, X_final_design, weights=w).fit() if w is not None else sm.OLS(y, X_final_design).fit()
        y_pred = final_model.predict(X_final_design)
        result.update({"final_model": final_model, "y_pred": y_pred, "r2": float(r2_score(y, y_pred)), "bic": float(final_model.bic)})
    else:
        X_final = X_selected if (not cfg.include_controls_in_final_model or X_controls is None) else np.hstack([X_controls, X_selected])
        final_model = LogisticRegression(max_iter=5000, solver="lbfgs")
        final_model.fit(X_final, y)
        y_pred = final_model.predict(X_final)
        result.update({"final_model": final_model, "y_pred": y_pred, "accuracy": float(accuracy_score(y, y_pred))})

    result["X_duration"] = pd.DataFrame(X_duration, columns=duration_feature_names, index=ids)
    result["X_timing"] = pd.DataFrame(X_timing, columns=timing_feature_names, index=ids)
    result["X_sequencing"] = pd.DataFrame(X_sequencing, columns=sequencing_feature_names, index=ids)
    result["X_full"] = pd.DataFrame(X_full, columns=all_feature_names, index=ids)
    return result


