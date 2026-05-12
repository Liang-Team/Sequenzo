"""
@Author  : Yuqi Liang 梁彧祺
@File    : rarcat_typology_regression.py
@Time    : 08/05/2025 17:41
@Desc    : 
RARCAT: robust average marginal effects for typology-based regression.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform

from ..k_medoids import KMedoids
from ..sequences_to_variables import medoid_indices_from_kmedoids_result
from .partition_quality import METRIC_ORDER, cluster_range_from_partitions


@dataclass
class RarcatResult:
    arguments: Dict[str, Any]
    formula: str
    factor_names: List[str]
    cluster_names: List[str]
    ame_list: Dict[str, pd.DataFrame]
    pooled_ame: Optional[pd.DataFrame] = None
    standard_error: Optional[pd.DataFrame] = None
    bootstrap_stddev: Optional[pd.DataFrame] = None
    observation_stddev: Optional[pd.DataFrame] = None
    observation_ranef: Optional[pd.DataFrame] = None
    observation_stdranef: Optional[pd.DataFrame] = None
    cluster_solution: Optional[np.ndarray] = None
    optimal_number: Optional[np.ndarray] = None
    bootout: Optional[Dict[str, Any]] = None


def _average_marginal_effects(model) -> pd.DataFrame:
    margeff = model.get_margeff(at="overall", method="dydx")
    return pd.DataFrame(
        {
            "factor": margeff.summary_frame().index.astype(str),
            "AME": margeff.margeff,
            "SE": margeff.margeff_se,
        }
    )


def _kmedoid_labels(diss: np.ndarray, k: int) -> np.ndarray:
    memb = KMedoids(diss, k=k, method="PAMonce", verbose=False)
    medoids = medoid_indices_from_kmedoids_result(memb)
    return np.argmin(diss[:, medoids], axis=1) + 1


def _select_clustering(
    diss: np.ndarray,
    kmedoid: bool,
    hclust_method: str,
    fixed: bool,
    ncluster: int,
    cqi: str,
) -> np.ndarray:
    if fixed or ncluster == 2:
        if kmedoid:
            return _kmedoid_labels(diss, ncluster)
        condensed = squareform(diss, checks=False)
        tree = linkage(condensed, method=hclust_method.lower().replace(".", "_"))
        return cut_tree(tree, n_clusters=ncluster).ravel() + 1

    if kmedoid:
        partitions = np.column_stack([_kmedoid_labels(diss, k) for k in range(2, ncluster + 1)])
    else:
        condensed = squareform(diss, checks=False)
        tree = linkage(condensed, method=hclust_method.lower().replace(".", "_"))
        partitions = np.column_stack([cut_tree(tree, n_clusters=k).ravel() + 1 for k in range(2, ncluster + 1)])
    quality = cluster_range_from_partitions(diss, partitions)
    stats = quality.stats
    if cqi == "HC":
        best = int(np.argmin(stats[cqi].to_numpy()))
    else:
        best = int(np.argmax(stats[cqi].to_numpy()))
    return partitions[:, best]


def _amemat(
    diss: np.ndarray,
    indices: np.ndarray,
    membership_name: str,
    model_df: pd.DataFrame,
    kmedoid: bool,
    hclust_method: str,
    fixed: bool,
    ncluster: int,
    cqi: str,
) -> np.ndarray:
    sample = diss[np.ix_(indices, indices)]
    clustering = _select_clustering(sample, kmedoid, hclust_method, fixed, ncluster, cqi)
    clustering = pd.factorize(clustering)[0] + 1
    boot_df = model_df.iloc[indices].copy()
    boot_df[membership_name] = clustering
    outputs = []
    unique_ids = np.unique(indices)
    duplicated = np.ones_like(indices, dtype=bool)
    duplicated[np.unique(indices, return_index=True)[1]] = False
    for cluster_value in np.unique(clustering):
        boot_df["membership"] = clustering == cluster_value
        y = boot_df["membership"].astype(float)
        predictors = boot_df.drop(columns=[membership_name, "membership"])
        x = sm.add_constant(pd.get_dummies(predictors, drop_first=False))
        model = sm.GLM(y, x, family=sm.families.Binomial()).fit()
        effects = _average_marginal_effects(model)
        selected = indices[clustering == cluster_value]
        selected = selected[~duplicated[clustering == cluster_value]]
        for obs_id in selected:
            row = [obs_id, cluster_value]
            row.extend(effects["AME"].tolist())
            row.extend(effects["SE"].tolist())
            outputs.append(row)
    if not outputs:
        return np.empty((len(indices), 1))
    columns = ["id", "clustering"] + effects["factor"].tolist() + [f"{name} SE" for name in effects["factor"]]
    frame = pd.DataFrame(outputs, columns=columns)
    merged = pd.DataFrame({"position": np.arange(1, len(indices) + 1)})
    merged = merged.merge(frame, left_on="position", right_on="id", how="left")
    return merged.drop(columns=["position", "id"]).to_numpy(dtype=float)


def rarcat(
    formula: str,
    data: pd.DataFrame,
    diss: np.ndarray,
    robust: bool = True,
    n_boot: int = 500,
    kmedoid: bool = False,
    hclust_method: str = "ward",
    fixed: bool = False,
    ncluster: int = 10,
    cqi: str = "HC",
    fisher_transform: bool = False,
    random_state: Optional[int] = None,
) -> RarcatResult:
    """
    Robust AME analysis for typology membership (WeightedCluster ``rarcat``).
    """
    if data.shape[0] != diss.shape[0]:
        raise ValueError("The dissimilarity matrix and data must have the same number of rows.")
    if ncluster < 2:
        raise ValueError("At least two clusters are required.")
    if cqi not in METRIC_ORDER:
        raise ValueError(f"Incorrect evaluation measure: {cqi}")

    lhs, rhs = formula.split("~", maxsplit=1)
    membership_name = lhs.strip()
    model_df = data.copy()
    clustering = model_df[membership_name].to_numpy()
    cluster_names = [str(value) for value in np.unique(clustering)]
    ame_list: Dict[str, pd.DataFrame] = {}
    for cluster_name in cluster_names:
        model_df["membership"] = clustering == cluster_name
        predictors = model_df.drop(columns=[membership_name, "membership"])
        x = sm.add_constant(pd.get_dummies(predictors, drop_first=False))
        y = model_df["membership"].astype(float)
        model = sm.GLM(y, x, family=sm.families.Binomial()).fit()
        ame_list[cluster_name] = _average_marginal_effects(model)

    result = RarcatResult(
        arguments={
            "formula": formula,
            "robust": robust,
            "R": n_boot,
            "kmedoid": kmedoid,
            "hclust.method": hclust_method,
            "fixed": fixed,
            "ncluster": ncluster,
            "cqi": cqi,
            "fisher.transform": fisher_transform,
        },
        formula=formula,
        factor_names=list(ame_list[cluster_names[0]]["factor"]),
        cluster_names=cluster_names,
        ame_list=ame_list,
    )

    if not robust:
        return result

    rng = np.random.default_rng(random_state)
    boot_indices = [rng.integers(0, data.shape[0], size=data.shape[0]) for _ in range(n_boot)]
    boot_outputs = [
        _amemat(
            diss=diss,
            indices=indices,
            membership_name=membership_name,
            model_df=model_df,
            kmedoid=kmedoid,
            hclust_method=hclust_method,
            fixed=fixed,
            ncluster=ncluster,
            cqi=cqi,
        )
        for indices in boot_indices
    ]
    cluster_solution = np.column_stack([output[:, 1] for output in boot_outputs])
    optimal_number = np.apply_along_axis(lambda col: len(np.unique(col[~np.isnan(col)])), 0, cluster_solution)

    factor_names = result.factor_names
    pooled = pd.DataFrame(np.nan, index=factor_names, columns=cluster_names)
    stderr = pooled.copy()
    boot_sd = pooled.copy()
    obs_sd = pooled.copy()
    observation_ranef = pd.DataFrame(np.nan, index=data.index.astype(str), columns=factor_names)
    observation_stdranef = pd.DataFrame(0.0, index=data.index.astype(str), columns=factor_names)

    for factor_idx, factor_name in enumerate(factor_names):
        effects = np.column_stack([output[:, 2 + factor_idx] for output in boot_outputs])
        errors = np.column_stack([output[:, 2 + len(factor_names) + factor_idx] for output in boot_outputs])
        for cluster_idx, cluster_name in enumerate(cluster_names):
            mask = clustering == cluster_name
            ids = np.where(mask)[0]
            prep = pd.DataFrame(
                {
                    "bootstrap": np.repeat(np.arange(n_boot), mask.sum()),
                    "id": np.tile(ids, n_boot),
                    "ame": effects[mask, :].reshape(-1),
                    "sterror": errors[mask, :].reshape(-1),
                }
            )
            prep = prep.dropna(subset=["ame"])
            prep["weight"] = 1.0 / np.square(prep["sterror"].clip(lower=1e-12))
            prep["stweight"] = prep["weight"] / prep["weight"].mean()
            if fisher_transform:
                prep["ame"] = np.arctanh(np.clip(prep["ame"], -0.999999, 0.999999))
            try:
                from statsmodels.regression.mixed_linear_model import MixedLM

                mixed = MixedLM.from_formula(
                    "ame ~ 1",
                    groups="id",
                    vc_formula={"bootstrap": "0 + C(bootstrap)"},
                    data=prep,
                    weights=prep["stweight"],
                ).fit(reml=False, disp=False)
                coef = mixed.params["Intercept"]
                se = mixed.bse["Intercept"]
                pooled.loc[factor_name, cluster_name] = np.tanh(coef) if fisher_transform else coef
                stderr.loc[factor_name, cluster_name] = se * (1 - np.tanh(coef) ** 2) if fisher_transform else se
                boot_sd.loc[factor_name, cluster_name] = np.sqrt(mixed.cov_re.iloc[0, 0])
                obs_sd.loc[factor_name, cluster_name] = np.sqrt(mixed.cov_re.iloc[1, 1]) if mixed.cov_re.shape[0] > 1 else np.nan
                ranef = mixed.random_effects
                for obs_id, values in ranef.items():
                    observation_ranef.loc[str(obs_id), factor_name] = values.get("Group", values.get("id", np.nan))
            except Exception:
                pooled.loc[factor_name, cluster_name] = np.nanmean(effects[mask, :])
                stderr.loc[factor_name, cluster_name] = np.nanstd(effects[mask, :], ddof=1)

    result.pooled_ame = pooled
    result.standard_error = stderr
    result.bootstrap_stddev = boot_sd
    result.observation_stddev = obs_sd
    result.observation_ranef = observation_ranef
    result.observation_stdranef = observation_stdranef
    result.cluster_solution = cluster_solution
    result.optimal_number = optimal_number
    result.bootout = {
        "cluster.solution": cluster_solution,
        "optimal.number": optimal_number,
        "effects": {name: np.column_stack([out[:, 2 + idx] for out in boot_outputs]) for idx, name in enumerate(factor_names)},
        "errors": {name: np.column_stack([out[:, 2 + len(factor_names) + idx] for out in boot_outputs]) for idx, name in enumerate(factor_names)},
    }
    return result
