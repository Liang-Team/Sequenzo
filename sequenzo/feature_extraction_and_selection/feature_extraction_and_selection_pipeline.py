"""
@Author  : Yuqi Liang 梁彧祺
@File    : feature_extraction_and_selection_pipeline.py
@Time    : 24/03/2026 22:41
@Desc    :
    FES-inspired automated pipeline (Bolano & Studer 2020; Unterlerchner et al. 2023):
    spells -> duration / timing / sequencing features -> Boruta -> optional exploratory model.

    This is not a byte-for-byte reproduction of the published R workflows; see README.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score

from sequenzo.define_sequence_data import SequenceData

from .boruta_feature_selection import select_all_relevant_features_boruta
from .monthly_state_to_spells import EndTimeMode
from .time_binning_utils import TimeUnitHint, suggest_timing_bin_width

SelectionProblemType = Literal["regression", "classification"]


@dataclass(frozen=True)
class FeatureExtractionAndSelectionConfig:
    sequencing_max_k: int = 3
    sequencing_min_support: Union[int, float] = 0.05
    sequencing_top_mined_subsequences: Optional[int] = 1000
    sequencing_count_method: str = "presence"
    sequencing_event_label_mode: str = "state"

    # Width in the same unit as seqdata.time (see time_unit_hint).
    timing_bin_width: float = 12.0
    time_unit_hint: TimeUnitHint = "same_as_labels"
    timing_include_start: bool = True
    timing_include_end: bool = True
    timing_count_method: str = "any"
    timing_bin_include_left: bool = True
    end_time_mode: EndTimeMode = "last_observed"

    boruta_n_iter: int = 50
    boruta_perc: float = 100.0
    boruta_alpha: float = 0.01
    boruta_two_step: bool = False

    residualize_target_with_controls: bool = True
    include_controls_in_final_model: bool = True
    fit_final_model: bool = False


def get_feature_extraction_and_selection_config_preset(
    preset: str,
) -> FeatureExtractionAndSelectionConfig:
    """
    Return a named configuration preset for reproducible FES runs.

    Parameters
    ----------
    preset : str
        Preset name. Currently supported:
        - ``unterlerchner2023`` — TREE-style monthly grids (month indices 1..T),
          12-month timing bins, spell start **and** end timing, exit-time ends.
    """
    key = preset.strip().lower()
    if key == "unterlerchner2023":
        return FeatureExtractionAndSelectionConfig(
            sequencing_max_k=3,
            sequencing_min_support=0.05,
            sequencing_top_mined_subsequences=1000,
            sequencing_count_method="presence",
            sequencing_event_label_mode="state",
            time_unit_hint="month",
            timing_bin_width=suggest_timing_bin_width("month"),
            timing_include_start=True,
            timing_include_end=True,
            timing_count_method="any",
            timing_bin_include_left=True,
            end_time_mode="exit_time",
            boruta_n_iter=50,
            boruta_perc=100.0,
            boruta_alpha=0.01,
            boruta_two_step=False,
            residualize_target_with_controls=True,
            include_controls_in_final_model=True,
            fit_final_model=False,
        )
    raise ValueError(
        f"Unknown preset '{preset}'. Available presets: unterlerchner2023"
    )


def _infer_problem_type(y: np.ndarray) -> SelectionProblemType:
    """Conservative auto-detection: numeric outcomes default to regression."""
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError("y must be 1D.")
    if np.issubdtype(y_arr.dtype, np.number):
        return "regression"
    return "classification"


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


def _outcome_for_feature_selection(
    y: np.ndarray,
    problem_type: SelectionProblemType,
    X_controls: Optional[np.ndarray],
    w: Optional[np.ndarray],
    *,
    residualize: bool,
) -> Tuple[np.ndarray, SelectionProblemType]:
    """
    Residualize outcome on controls before Boruta (Bolano & Studer 2020).

    Regression: OLS/WLS residuals. Classification: binomial GLM deviance residuals,
    then Boruta runs as regression on those residuals.
    """
    if not residualize or X_controls is None:
        return y, problem_type

    if problem_type == "classification":
        uniq = np.unique(y)
        if uniq.size != 2:
            raise ValueError(
                "Residualization for classification currently supports only binary outcomes. "
                "For multi-class outcomes, set residualize_target_with_controls=False."
            )

    X_design = sm.add_constant(X_controls, has_constant="add")
    if problem_type == "regression":
        model = sm.WLS(y, X_design, weights=w).fit() if w is not None else sm.OLS(y, X_design).fit()
        return (y - model.predict(X_design)).astype(float), "regression"

    if w is not None:
        glm = sm.GLM(y, X_design, family=sm.families.Binomial(), freq_weights=w).fit()
    else:
        glm = sm.GLM(y, X_design, family=sm.families.Binomial()).fit()
    return np.asarray(glm.resid_deviance, dtype=float), "regression"


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
    fit_final_model: Optional[bool] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the FES-inspired pipeline: extract features, Boruta selection, optional model.

    If ``problem_type`` is omitted or ``"auto"``, numeric outcomes default to
    **regression**. Binary ``0/1`` outcomes are still treated as regression unless
    you pass ``problem_type="classification"`` explicitly.
    """
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
    do_fit_final = cfg.fit_final_model if fit_final_model is None else fit_final_model

    y_raw = np.asarray(outcome)
    if y_raw.ndim != 1:
        raise ValueError("outcome must be 1D.")
    n = len(y_raw)
    if seqdata.n_sequences != n:
        raise ValueError("Length of outcome must match number of sequences.")

    if problem_type is None or problem_type == "auto":
        problem_type = _infer_problem_type(y_raw)
    if problem_type not in {"regression", "classification"}:
        raise ValueError("problem_type must be 'regression', 'classification', or 'auto'.")

    if problem_type == "regression":
        y = y_raw.astype(float)
    else:
        y = pd.Categorical(y_raw).codes
        if np.any(y < 0):
            raise ValueError("outcome contains missing values.")

    if sample_weights is not None:
        w = np.asarray(sample_weights, dtype=float)
        if w.ndim != 1 or len(w) != n:
            raise ValueError("sample_weights must be a 1D array with length n.")
    else:
        w = None

    X_controls, _ = _prepare_control_matrix(controls, n)
    if verbose:
        print(f"[FES] problem_type={problem_type}, n={n}, fit_final_model={do_fit_final}")

    spells_per_individual = extract_spells_with_times(
        seqdata,
        end_time_mode=cfg.end_time_mode,
    )
    if state_groups is None:
        state_groups = {str(s): [s] for s in seqdata.states}

    X_duration, duration_feature_names = build_duration_features(
        spells_per_individual,
        state_groups=state_groups,
    )

    numeric_time = coerce_numeric_time_labels(seqdata.time)
    time_bins = make_equal_width_bins(
        float(np.min(numeric_time)),
        float(np.max(numeric_time)),
        cfg.timing_bin_width,
    )
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
        weighted=False,
    )

    all_feature_names = duration_feature_names + timing_feature_names + sequencing_feature_names
    X_full = np.hstack([X_duration, X_timing, X_sequencing]).astype(float)

    y_for_selection, boruta_problem_type = _outcome_for_feature_selection(
        y if problem_type == "regression" else y.astype(float),
        problem_type,
        X_controls,
        w,
        residualize=cfg.residualize_target_with_controls,
    )

    boruta_result = select_all_relevant_features_boruta(
        X_full,
        y_for_selection,
        problem_type=boruta_problem_type,
        n_iter=cfg.boruta_n_iter,
        perc=cfg.boruta_perc,
        boruta_alpha=cfg.boruta_alpha,
        boruta_two_step=cfg.boruta_two_step,
        random_state=42,
        verbose=verbose,
    )
    selected_mask = boruta_result.selected_mask
    selected_feature_names = [
        all_feature_names[j] for j in range(len(all_feature_names)) if selected_mask[j]
    ]
    tentative_feature_names: List[str] = []
    if boruta_result.tentative_mask is not None:
        tentative_feature_names = [
            all_feature_names[j]
            for j, keep in enumerate(boruta_result.tentative_mask)
            if keep
        ]
    if len(selected_feature_names) == 0:
        raise RuntimeError("Boruta selected zero features. Consider relaxing selection parameters.")
    X_selected = X_full[:, selected_mask]

    result: Dict[str, Any] = {
        "problem_type": problem_type,
        "n": n,
        "time_unit_hint": cfg.time_unit_hint,
        "timing_bin_width": cfg.timing_bin_width,
        "end_time_mode": cfg.end_time_mode,
        "all_feature_names": all_feature_names,
        "selected_feature_names": selected_feature_names,
        "selected_mask": selected_mask,
        "selected_indices": boruta_result.selected_indices,
        "tentative_mask": boruta_result.tentative_mask,
        "tentative_indices": boruta_result.tentative_indices,
        "tentative_feature_names": tentative_feature_names,
        "boruta_ranking": boruta_result.ranking,
        "hit_counts": boruta_result.hit_counts,
        "shadow_hit_counts": boruta_result.shadow_hit_counts,
        "X_selected": X_selected,
        "fit_final_model": do_fit_final,
        "final_model_fitted": do_fit_final,
        "final_model_is_exploratory": do_fit_final,
    }

    if do_fit_final:
        if problem_type == "regression":
            X_final = (
                X_selected
                if (not cfg.include_controls_in_final_model or X_controls is None)
                else np.hstack([X_controls, X_selected])
            )
            X_final_design = sm.add_constant(X_final, has_constant="add")
            final_model = (
                sm.WLS(y, X_final_design, weights=w).fit()
                if w is not None
                else sm.OLS(y, X_final_design).fit()
            )
            y_pred = final_model.predict(X_final_design)
            result.update(
                {
                    "final_model": final_model,
                    "y_pred": y_pred,
                    "r2": float(r2_score(y, y_pred)),
                    "bic": float(final_model.bic),
                }
            )
        else:
            X_final = (
                X_selected
                if (not cfg.include_controls_in_final_model or X_controls is None)
                else np.hstack([X_controls, X_selected])
            )
            final_model = LogisticRegression(max_iter=5000, solver="lbfgs")
            if w is not None:
                final_model.fit(X_final, y, sample_weight=w)
            else:
                final_model.fit(X_final, y)
            y_pred = final_model.predict(X_final)
            result.update(
                {
                    "final_model": final_model,
                    "y_pred": y_pred,
                    "accuracy": float(accuracy_score(y, y_pred)),
                }
            )

    result["X_duration"] = pd.DataFrame(X_duration, columns=duration_feature_names, index=ids)
    result["X_timing"] = pd.DataFrame(X_timing, columns=timing_feature_names, index=ids)
    result["X_sequencing"] = pd.DataFrame(X_sequencing, columns=sequencing_feature_names, index=ids)
    result["X_full"] = pd.DataFrame(X_full, columns=all_feature_names, index=ids)
    return result
