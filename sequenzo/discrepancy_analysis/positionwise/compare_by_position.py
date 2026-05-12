"""
@Author  : Yuqi Liang 梁彧祺
@File    : compare_by_position.py
@Time    : 2026-02-10 16:35
@Desc    : Position-wise discrepancy analysis between groups of sequences (TraMineR: seqdiff).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union, List
import warnings
import gc

from ..stats.single_factor_association import single_factor_association
from ...dissimilarity_measures.get_distance_matrix import get_distance_matrix
from ...define_sequence_data import SequenceData


def compare_groups_across_positions(
    seqdata: Union[pd.DataFrame, SequenceData],
    group: Union[np.ndarray, pd.Series, pd.DataFrame],
    cmprange: tuple = (0, 1),
    seqdist_args: Optional[dict] = None,
    with_missing: bool = False,
    weighted: bool = True,
    squared: bool = False
) -> dict:
    if seqdist_args is None:
        seqdist_args = {"method": "LCS", "norm": "auto"}
    else:
        seqdist_args = seqdist_args.copy()

    if isinstance(seqdata, SequenceData):
        seqdata_df = seqdata.data[seqdata.time].copy()
    elif isinstance(seqdata, pd.DataFrame):
        seqdata_df = seqdata.copy()
    else:
        raise TypeError(
            "[!] 'seqdata' should be a DataFrame or SequenceData object. "
            f"Got {type(seqdata)}"
        )

    seqdist_args["with_missing"] = with_missing
    slenE = seqdata_df.shape[1]
    startAt = 1
    totrange = range(max(startAt, 1 - cmprange[0]), min(slenE, slenE - cmprange[1]) + 1)
    totrange_list = list(totrange)

    if len(totrange_list) == 0:
        raise ValueError(
            f"[!] Invalid cmprange {cmprange} for sequence length {slenE}. "
            "No valid positions to analyze."
        )

    is_group_seqdata = isinstance(group, pd.DataFrame)
    if is_group_seqdata:
        if hasattr(group, "alphabet"):
            name_column = group.alphabet
        else:
            unique_states = []
            for col in group.columns:
                unique_states.extend(group[col].unique())
            name_column = sorted(set(unique_states))
    else:
        group = pd.Series(group) if not isinstance(group, pd.Series) else group
        group = pd.Categorical(group)
        name_column = group.categories.tolist()

    num_column = len(name_column)
    num_positions = len(totrange_list)
    stat_df = pd.DataFrame(
        np.full((num_positions, 5), np.nan),
        columns=["Pseudo F", "Pseudo Fbf", "Pseudo R2", "Bartlett", "Levene"],
        index=[seqdata_df.columns[i - 1] for i in totrange_list],
    )
    discrepancy_df = pd.DataFrame(
        np.zeros((num_positions, num_column + 1)),
        columns=name_column + ["Total"],
        index=[seqdata_df.columns[i - 1] for i in totrange_list],
    )

    weights = None
    if weighted:
        if isinstance(seqdata, SequenceData) and hasattr(seqdata, "weights"):
            weights = seqdata.weights
        elif hasattr(seqdata, "weights"):
            weights = seqdata.weights

    for idx, i in enumerate(totrange_list):
        gc.collect()
        srange = list(range(i + cmprange[0] - 1, i + cmprange[1]))
        srange = [s for s in srange if 0 <= s < slenE]
        if len(srange) == 0:
            continue

        if is_group_seqdata:
            cmpbase = group.iloc[:, i - 1].copy()
            if hasattr(group, "void"):
                cmpbase[cmpbase == group.void] = np.nan
            if hasattr(group, "nr"):
                cmpbase[cmpbase == group.nr] = np.nan
        else:
            cmpbase = group.copy()

        subseq = seqdata_df.iloc[:, srange]
        if not with_missing:
            subseq2 = subseq.copy()
            if isinstance(seqdata, SequenceData):
                if hasattr(seqdata, "void"):
                    subseq2[subseq2 == seqdata.void] = np.nan
                if hasattr(seqdata, "nr"):
                    subseq2[subseq2 == seqdata.nr] = np.nan
            elif hasattr(seqdata, "void"):
                subseq2[subseq2 == seqdata.void] = np.nan
            elif hasattr(seqdata, "nr"):
                subseq2[subseq2 == seqdata.nr] = np.nan
            seqok = (~subseq2.isna().any(axis=1)) & (~cmpbase.isna())
        else:
            seqok = ~cmpbase.isna()

        if not seqok.any():
            warnings.warn(f"[!] No valid sequences at position {i}. Skipping.", UserWarning)
            continue

        subseq_filtered = subseq[seqok].copy()
        cmpbase_filtered = cmpbase[seqok].copy()
        weights_filtered = weights[seqok] if weights is not None else None

        if isinstance(seqdata, SequenceData):
            temp_states = seqdata.states
        else:
            temp_states = sorted(subseq_filtered.stack().dropna().unique().tolist())

        temp_seqdata = SequenceData(
            subseq_filtered,
            time=list(subseq_filtered.columns),
            states=temp_states,
            weights=weights_filtered,
        )

        seqdist_args_temp = seqdist_args.copy()
        seqdist_args_temp["seqdata"] = temp_seqdata

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sdist = get_distance_matrix(**seqdist_args_temp)
        except Exception as e:
            warnings.warn(f"[!] Failed to compute distance matrix at position {i}: {e}", UserWarning)
            continue

        try:
            if isinstance(cmpbase_filtered, pd.Categorical):
                group_values = np.asarray(cmpbase_filtered)
            else:
                group_values = np.asarray(
                    cmpbase_filtered.values if hasattr(cmpbase_filtered, "values") else cmpbase_filtered
                )

            result = single_factor_association(
                distance_matrix=sdist,
                group=group_values,
                weights=weights_filtered,
                R=0,
                weight_permutation="diss",
                squared=squared,
            )

            if "stat" in result:
                for stat_name in result["stat"].index:
                    if stat_name in stat_df.columns:
                        stat_df.loc[stat_df.index[idx], stat_name] = result["stat"].loc[stat_name, "Value"]
            else:
                stat_df.loc[stat_df.index[idx], "Pseudo F"] = result.get("pseudo_f", np.nan)
                stat_df.loc[stat_df.index[idx], "Pseudo Fbf"] = result.get("pseudo_fbf", np.nan)
                stat_df.loc[stat_df.index[idx], "Pseudo R2"] = result.get("pseudo_r2", np.nan)
                stat_df.loc[stat_df.index[idx], "Bartlett"] = result.get("bartlett", np.nan)
                stat_df.loc[stat_df.index[idx], "Levene"] = result.get("levene", np.nan)

            for group_name in result["groups"].index:
                if group_name in discrepancy_df.columns:
                    discrepancy_df.loc[discrepancy_df.index[idx], group_name] = result["groups"].loc[group_name, "discrepancy"]
        except Exception as e:
            warnings.warn(f"[!] Failed to compute association at position {i}: {e}", UserWarning)
            continue

    return {
        "stat": stat_df,
        "discrepancy": discrepancy_df,
        "xtstep": getattr(seqdata, "xtstep", 1),
        "tick_last": getattr(seqdata, "tick_last", False),
    }


def print_group_differences_across_positions(result: dict) -> None:
    print("\nStatistics:")
    print(result["stat"])
    print("\nDiscrepancies:")
    print(result["discrepancy"])


def plot_group_differences_across_positions(
    result: dict,
    stat: Union[str, List[str]] = "Pseudo R2",
    plot_type: str = "l",
    ylab: Optional[str] = None,
    xlab: str = "",
    legend_pos: str = "upper center",
    ylim: Optional[tuple] = None,
    xaxis: bool = True,
    col: Optional[Union[str, List[str]]] = None,
    xtstep: Optional[int] = None,
    tick_last: Optional[bool] = None,
    figsize: tuple = (10, 6),
    **kwargs
) -> plt.Figure:
    if ylab is None:
        ylab = " / ".join(stat) if isinstance(stat, list) else stat
    if xtstep is None:
        xtstep = result.get("xtstep", 1)
    if tick_last is None:
        tick_last = result.get("tick_last", False)

    fig, ax = plt.subplots(figsize=figsize)
    if isinstance(stat, str) and stat in ["Variance", "discrepancy", "Residuals", "residuals"]:
        discrepancy_df = result["discrepancy"]
        num_groups = discrepancy_df.shape[1]
        if col is None:
            if num_groups <= 8:
                try:
                    from matplotlib import colormaps
                    cmap = colormaps["Accent"]
                except (AttributeError, KeyError, ImportError):
                    from matplotlib.cm import get_cmap
                    cmap = get_cmap("Accent")
                colors = [cmap(i / num_groups) for i in range(num_groups)]
            else:
                colors = plt.cm.Set3(np.linspace(0, 1, num_groups))
        else:
            colors = col if isinstance(col, list) else [col] * num_groups

        toplot = discrepancy_df.values * (1 - result["stat"]["Pseudo R2"].values[:, np.newaxis]) if stat in ["Residuals", "residuals"] else discrepancy_df.values
        if ylim is None:
            ylim = (np.nanmin(toplot), np.nanmax(toplot))

        x_values = np.arange(len(discrepancy_df))
        ax.plot(x_values, toplot[:, -1], color=colors[-1], linestyle="-" if plot_type == "l" else "", marker="o" if plot_type == "p" else "", label=discrepancy_df.columns[-1], **kwargs)
        for i in range(num_groups - 1):
            ax.plot(x_values, toplot[:, i], color=colors[i], linestyle="-" if plot_type == "l" else "", marker="o" if plot_type == "p" else "", label=discrepancy_df.columns[i], **kwargs)
        ax.set_ylim(ylim)
        ax.legend(loc=legend_pos)
    elif isinstance(stat, str):
        if stat not in result["stat"].columns:
            raise ValueError(f"[!] 'stat' argument should be one of {', '.join(['discrepancy'] + result['stat'].columns.tolist())}")
        if col is None:
            col = "black"
        x_values = np.arange(len(result["stat"]))
        y_values = result["stat"][stat].values
        ax.plot(x_values, y_values, color=col, linestyle="-" if plot_type == "l" else "", marker="o" if plot_type == "p" else "", **kwargs)
    elif isinstance(stat, list) and len(stat) == 2:
        if not all(s in result["stat"].columns for s in stat):
            raise ValueError(f"[!] The two values of 'stat' should be one of {', '.join(result['stat'].columns.tolist())}")
        if col is None:
            col = ["red", "blue"]
        x_values = np.arange(len(result["stat"]))
        ax.plot(x_values, result["stat"][stat[0]].values, color=col[0], linestyle="-" if plot_type == "l" else "", marker="o" if plot_type == "p" else "", label=stat[0], **kwargs)
        ax.set_ylabel(stat[0], color=col[0])
        ax.tick_params(axis="y", labelcolor=col[0])
        ax2 = ax.twinx()
        ax2.plot(x_values, result["stat"][stat[1]].values, color=col[1], linestyle="-" if plot_type == "l" else "", marker="o" if plot_type == "p" else "", label=stat[1], **kwargs)
        ax2.set_ylabel(stat[1], color=col[1])
        ax2.tick_params(axis="y", labelcolor=col[1])
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc=legend_pos)
    else:
        raise ValueError("[!] Too many values for 'stat' argument (max 2)")

    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    if xaxis:
        seql = len(result["stat"])
        tpos = list(range(0, seql, xtstep))
        if tick_last and tpos[-1] < seql - 1:
            tpos.append(seql - 1)
        ax.set_xticks(tpos)
        ax.set_xticklabels([result["stat"].index[i] for i in tpos])
    else:
        ax.set_xticks([])
    plt.tight_layout()
    return fig
