"""
@Author  : Yuqi Liang 梁彧祺
@File    : spell_survival_analysis.py
@Time    : 15/05/2026 11:29
@Desc    : 
Spell survival analysis: how long spells in each state last before leaving.

Each *spell* (a consecutive run in the same state) is one row in a Kaplan–Meier model:

- **Duration** = spell length in time steps (``end - begin + 1``).
- **Event** = leaving the state before the sequence ends.
- **Censoring** = last spell of the sequence (still in that state at the end).

R equivalent (TraMineRextras)
-----------------------------
- ``seqsurv()`` — see ``?seqsurv`` and ``R/seqsurv.R``
- Uses TraMineR ``seqformat(STS -> SPELL)``, ``seqlength``, ``seqstatl``, and ``survival::survfit``

Python ↔ R
----------
+----------------------------------+----------------------------------+
| Python (Sequenzo)                | R (TraMineRextras)               |
+----------------------------------+----------------------------------+
| ``get_spell_survival_analysis()`` | ``seqsurv()``                   |
| ``SpellSurvivalResult``          | ``stslist.surv`` + ``survfit``   |
| ``plot_spell_survival_analysis()`` | ``plot()`` on seqsurv result   |
+----------------------------------+----------------------------------+
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.utils.seqdss import seqdss
from sequenzo.dissimilarity_measures.utils.seqdur import seqdur
from sequenzo.dissimilarity_measures.utils.seqlength import seqlength


def _seqstatl(seqdata: SequenceData) -> List[Any]:
    """
    States that appear at least once in ``seqdata`` (TraMineR ``seqstatl``).

    Returned in ``seqdata.states`` order so colours and labels stay aligned.
    """
    vals = seqdata.values
    observed = set(int(v) for v in np.unique(vals) if int(v) > 0)
    return [s for i, s in enumerate(seqdata.states) if i + 1 in observed]


def _sts_to_spell_table(seqdata: SequenceData) -> pd.DataFrame:
    """
    Build a spell-level table like ``seqformat(..., from='STS', to='SPELL')``.

    Columns: id, begin, end, states, dur, weights, length, status, group (filled later).
    Row ``id`` is 1..n (R resets row names before conversion).
    """
    dss = seqdss(seqdata)
    dur = seqdur(seqdata)
    state_list = list(seqdata.states)
    lengths = np.asarray(seqlength(seqdata), dtype=int)
    weights = np.asarray(seqdata.weights, dtype=float)
    n = dss.shape[0]

    rows: List[dict] = []
    for i in range(n):
        seq_id = i + 1  # R uses 1..n after rownames(seqdata) <- 1:n
        pos = 0
        spell_states_i: List[Any] = []
        spell_durs_i: List[float] = []
        for j in range(dss.shape[1]):
            code = int(dss[i, j])
            if code < 0:
                break
            # seqdss stores 1..K state codes (TraMineR-style)
            spell_states_i.append(state_list[code - 1] if code >= 1 else code)
            spell_durs_i.append(float(dur[i, j]))
        for state, dur_val in zip(spell_states_i, spell_durs_i):
            d = int(dur_val)
            if d <= 0:
                break
            begin = pos + 1
            end = pos + d
            rows.append(
                {
                    "id": seq_id,
                    "begin": begin,
                    "end": end,
                    "states": state,
                    "dur": d,
                    "weights": weights[i],
                    "length": int(lengths[i]),
                    "status": bool(end != lengths[i]),
                }
            )
            pos += d
    return pd.DataFrame(rows)


def _brewer_dark2(n: int) -> Optional[List[str]]:
    """
    Match R ``RColorBrewer::brewer.pal(..., 'Dark2')`` logic in ``seqsurv``.
    """
    if n <= 0:
        return None
    cmap = plt.get_cmap("Dark2")
    if n == 1:
        return [cmap(2 / 2.999)]  # brewer.pal(3, "Dark2")[3] in 0-based terms
    if n == 2:
        return [cmap(i / 2.999) for i in (0, 1)]
    if n < 9:
        return [cmap(i / max(n - 1, 1)) for i in range(n)]
    warnings.warn(
        "[!] More than 8 groups: no automatic colour palette (see seqsplot).",
        UserWarning,
        stacklevel=3,
    )
    return None


def _hex_colors(colors: Sequence) -> List[str]:
    out: List[str] = []
    for c in colors:
        if isinstance(c, str):
            out.append(c)
        else:
            rgba = plt.matplotlib.colors.to_rgba(c)
            out.append(plt.matplotlib.colors.to_hex(rgba))
    return out


def _weighted_kaplan_meier(
    duration: np.ndarray,
    event: np.ndarray,
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Weighted Kaplan–Meier curve (one stratum).

    Matches R ``survival::survfit``: rows at every distinct spell duration where
    at least one event or censoring occurs (``n.event`` may be 0 when only
    censored spells end at that time).
    """
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=bool)
    weights = np.asarray(weights, dtype=float)
    if len(duration) == 0:
        return {
            "time": np.array([], dtype=float),
            "n.risk": np.array([], dtype=float),
            "n.event": np.array([], dtype=float),
            "surv": np.array([], dtype=float),
            "std.err": np.array([], dtype=float),
        }

    unique_times = np.sort(np.unique(duration))
    times: List[float] = []
    n_risk: List[float] = []
    n_event: List[float] = []
    surv: List[float] = []
    stderr: List[float] = []

    s = 1.0
    var_greenwood = 0.0
    for ti in unique_times:
        at_risk = weights[duration >= ti].sum()
        at_time = duration == ti
        d_i = weights[at_time & event].sum()
        c_i = weights[at_time & ~event].sum()
        if d_i <= 0 and c_i <= 0:
            continue

        if d_i > 0 and at_risk > 0:
            s *= 1.0 - d_i / at_risk
            if at_risk > d_i:
                var_greenwood += d_i / (at_risk * (at_risk - d_i))

        times.append(float(ti))
        n_risk.append(float(at_risk))
        n_event.append(float(d_i))
        surv.append(s)
        stderr.append(float(s * np.sqrt(max(var_greenwood, 0.0))))

    return {
        "time": np.asarray(times, dtype=float),
        "n.risk": np.asarray(n_risk, dtype=float),
        "n.event": np.asarray(n_event, dtype=float),
        "surv": np.asarray(surv, dtype=float),
        "std.err": np.asarray(stderr, dtype=float),
    }


def _survfit_by_strata(
    spell: pd.DataFrame,
    strata_col: str,
    *,
    subset: Optional[pd.Series] = None,
) -> tuple[dict[str, dict[str, np.ndarray]], List[str]]:
    """Fit KM separately for each level of ``strata_col`` (like ``survfit(~ strata)``)."""
    if subset is not None:
        spell = spell.loc[subset].copy()
    if spell.empty:
        return {}, []

    curves: dict[str, dict[str, np.ndarray]] = {}
    present = spell[strata_col].dropna().unique()
    # Keep factor order stable (alphabet / group levels), not order of appearance.
    levels = [x for x in spell[strata_col].cat.categories if x in present] if hasattr(
        spell[strata_col], "cat"
    ) else sorted(present, key=lambda x: str(x))
    for level in levels:
        sub = spell.loc[spell[strata_col] == level]
        curves[str(level)] = _weighted_kaplan_meier(
            sub["dur"].to_numpy(),
            sub["status"].to_numpy(),
            sub["weights"].to_numpy(),
        )
    return curves, levels


@dataclass
class SpellSurvivalResult:
    """
    Container for spell survival curves (R ``stslist.surv`` + ``survfit``).

    Attributes
    ----------
    curves : dict
        Map stratum name -> arrays ``time``, ``n.risk``, ``n.event``, ``surv``, ``std.err``.
    ltext : list
        Legend text (state labels or group levels).
    cpal : list or None
        Colours for plotting (hex strings when available).
    xtstep, tick_last
        Copied from ``SequenceData`` for axis styling (TraMineR ``seqplot``).
    spell : DataFrame
        Spell table used for fitting (for debugging / extensions).
    """

    curves: dict[str, dict[str, np.ndarray]]
    ltext: List[str]
    cpal: Optional[List[str]]
    xtstep: Any
    tick_last: Any
    spell: pd.DataFrame = field(repr=False)
    per_state: bool = False
    empty: bool = False

    def to_summary_frame(self) -> pd.DataFrame:
        """Long-format table similar to ``summary(survfit)`` in R."""
        parts = []
        for strata, arr in self.curves.items():
            if len(arr["time"]) == 0:
                continue
            parts.append(
                pd.DataFrame(
                    {
                        "strata": strata,
                        "time": arr["time"],
                        "n.risk": arr["n.risk"],
                        "n.event": arr["n.event"],
                        "surv": arr["surv"],
                        "std.err": arr["std.err"],
                    }
                )
            )
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def get_spell_survival_analysis(
    seqdata: SequenceData,
    groups: Optional[Sequence] = None,
    per_state: bool = False,
    state: Optional[Sequence] = None,
    with_missing: bool = False,
) -> SpellSurvivalResult:
    """
    Spell-based Kaplan–Meier survival by state or group.

    R equivalent: TraMineRextras ``seqsurv()``.

    Parameters
    ----------
    seqdata : SequenceData
        Sequence object (TraMineR ``seqdef`` equivalent).
    groups : array-like, optional
        One group label per sequence. Used when ``per_state=True`` to compare groups
        within a chosen state (subset of spells in that state).
    per_state : bool, default False
        If False (default): one curve per **state** (all spells in that state).
        If True: one curve per **group**, keeping only spells in ``state``.
    state : list, optional
        Which states to include. Default: all states observed in the data.
    with_missing : bool, default False
        Not implemented in R either; emits a warning if True.

    Returns
    -------
    SpellSurvivalResult
        Use ``.to_summary_frame()`` for a flat table, or ``plot_spell_survival_analysis()`` to plot.

    Raises
    ------
    TypeError
        If ``seqdata`` is not a ``SequenceData`` object.
    ValueError
        If ``state`` lists only unobserved states, or multiple groups are used with
        ``per_state=False`` (use ``seqsplot`` in R; not ported here).

    Examples
    --------
    >>> from sequenzo import SequenceData, load_dataset
    >>> from sequenzo.with_event_history_analysis import (
    ...     get_spell_survival_analysis, plot_spell_survival_analysis,
    ... )
    >>> df = load_dataset("mvad")
    >>> seq = SequenceData(df, time=list(df.columns[16:86]), id_col="id", states=[...])
    >>> fit = get_spell_survival_analysis(seq)
    >>> plot_spell_survival_analysis(fit)
    """
    if not isinstance(seqdata, SequenceData):
        raise TypeError(
            "[!] seqdata must be a SequenceData object. "
            "Use SequenceData(...) to create one (TraMineR: seqdef)."
        )

    if with_missing:
        warnings.warn(
            "[!] with_missing=True is not implemented yet (same as TraMineRextras).",
            UserWarning,
            stacklevel=2,
        )

    spell = _sts_to_spell_table(seqdata)
    # Factor order = alphabet (state codes in Sequenzo)
    spell["states"] = pd.Categorical(spell["states"], categories=list(seqdata.states))

    obs_states = _seqstatl(seqdata)
    if state is None:
        state_filter = list(obs_states)
    else:
        state_filter = [s for s in state if s in obs_states]
        if len(state_filter) < 1:
            raise ValueError("[!] state contains only unobserved states.")

    n = len(seqdata.values)
    if groups is None:
        groups_arr = pd.Categorical(["1"] * n)
    else:
        if len(groups) != n:
            raise ValueError("[!] groups must have one value per sequence.")
        groups_arr = pd.Categorical(groups)

    spell["group"] = groups_arr[np.asarray(spell["id"], dtype=int) - 1]

    levels_num = len(groups_arr.categories)
    cpal = _brewer_dark2(levels_num)
    if cpal is not None:
        cpal = _hex_colors(cpal)
    ltext: List[str] = list(groups_arr.categories)

    subset = spell["states"].isin(state_filter)
    empty = not bool(subset.any())

    if per_state:
        if empty:
            return SpellSurvivalResult(
                curves={},
                ltext=ltext,
                cpal=cpal,
                xtstep=getattr(seqdata, "xtstep", None),
                tick_last=getattr(seqdata, "tick_last", None),
                spell=spell,
                per_state=True,
                empty=True,
            )
        curves, _ = _survfit_by_strata(spell, "group", subset=subset)
    elif levels_num == 1:
        if empty:
            return SpellSurvivalResult(
                curves={},
                ltext=[],
                cpal=[],
                xtstep=getattr(seqdata, "xtstep", None),
                tick_last=getattr(seqdata, "tick_last", None),
                spell=spell,
                per_state=False,
                empty=True,
            )
        curves, strata_levels = _survfit_by_strata(spell, "states", subset=subset)
        # Legend = state labels for states in the model (R: labels[alphabet %in% state])
        alphabet = list(seqdata.states)
        labels = list(seqdata.labels) if seqdata.labels else alphabet
        idx = [alphabet.index(s) for s in strata_levels if s in alphabet]
        ltext = [labels[i] for i in idx]
        cpal_seq = seqdata.custom_colors
        if cpal_seq is None:
            cpal = None
        else:
            cpal = [_hex_colors([cpal_seq[i]])[0] for i in idx]
    else:
        raise ValueError(
            "[!] With per_state=False, only a single group is supported. "
            "Consider per_state=True or a single group."
        )

    return SpellSurvivalResult(
        curves=curves,
        ltext=ltext,
        cpal=cpal,
        xtstep=getattr(seqdata, "xtstep", None),
        tick_last=getattr(seqdata, "tick_last", None),
        spell=spell,
        per_state=per_state,
        empty=empty,
    )


def plot_spell_survival_analysis(
    result: SpellSurvivalResult,
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xlabel: str = "Spell duration",
    ylabel: str = "Survival probability",
) -> plt.Axes:
    """
    Plot spell survival curves (matplotlib; R: ``plot()`` on a ``seqsurv()`` fit).

    Parameters
    ----------
    result : SpellSurvivalResult
        Object returned by ``get_spell_survival_analysis()``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; created if omitted.
    """
    if result.empty or not result.curves:
        raise ValueError("[!] No survival curves to plot (empty result).")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    names = list(result.curves.keys())
    colors = result.cpal
    if colors is None or len(colors) < len(names):
        colors = plt.cm.Dark2(np.linspace(0, 1, max(len(names), 3)))[: len(names)]
    labels = result.ltext if len(result.ltext) == len(names) else names

    for name, color, lab in zip(names, colors, labels):
        c = result.curves[name]
        if len(c["time"]) == 0:
            continue
        # Step function: survival drops at event times
        t = np.concatenate([[0.0], c["time"]])
        s = np.concatenate([[1.0], c["surv"]])
        ax.step(t, s, where="post", label=lab, color=color, linewidth=1.5)

    ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)
    return ax
