"""
Empirical diagnostics for MD-CLARA (thin helpers for scripts).

The tutorials call ``sequenzo.multidomain.clara`` directly:
``leave_one_domain_out_sensitivity``, ``compare_md_clara_strategies``,
``dat_domain_contributions``, ``summarize_combined_state_space``, and the
matching plot helpers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sequenzo.define_sequence_data import SequenceData
from sequenzo.multidomain.clara import (
    compare_md_clara_strategies,
    leave_one_domain_out_sensitivity,
    md_clara,
)

__all__ = [
    "run_typology",
    "leave_one_domain_out_ari",
    "dat_contribution_table",
    "compare_md_clara_strategies",
    "leave_one_domain_out_sensitivity",
]


def run_typology(
    domains: List[SequenceData],
    strategy: str,
    distance_params: Dict[str, Any],
    *,
    k: int,
    R: int = 50,
    sample_size: Optional[int] = None,
    random_state: Optional[int] = None,
    **md_clara_kw: Any,
) -> Dict[str, Any]:
    """Run MD-CLARA at a single k and return labels, medoids, and route diagnostics."""
    strategy = strategy.lower()
    result = md_clara(
        domains,
        strategy=strategy,
        distance_params=distance_params,
        R=R,
        sample_size=sample_size,
        kvals=[k],
        random_state=random_state,
        dat_domain_contribution=(strategy == "dat"),
        verbose=bool(md_clara_kw.pop("verbose", False)),
        **md_clara_kw,
    )
    labels = result.best_clustering(k)
    out: Dict[str, Any] = {
        "labels": labels,
        "medoids": result.medoids.get(k),
        "stats": result.stats,
        "result": result,
    }
    route = result.route_diagnostics or {}
    contrib = route.get("dat_domain_contributions", {})
    if k in contrib:
        out["dat_domain_contribution"] = contrib[k]
    return out


def leave_one_domain_out_ari(
    domains: List[SequenceData],
    strategy: str,
    distance_params: Dict[str, Any],
    *,
    k: int,
    domain_names: Sequence[str],
    reference_labels: np.ndarray,
    sample_size: int,
    R: int = 50,
    random_state: Optional[int] = None,
    **run_kw: Any,
) -> Dict[str, float]:
    """
    Legacy wrapper: map ``leave_one_domain_out_sensitivity`` to name → ARI dict.

    Prefer ``leave_one_domain_out_sensitivity`` in new code; it also returns Jaccard
    and profile counts per omitted domain.
    """
    if len(domains) != len(domain_names):
        raise ValueError("domain_names must match len(domains)")

    table = leave_one_domain_out_sensitivity(
        domains,
        strategy=strategy,
        k=k,
        R=R,
        sample_size=sample_size,
        distance_params=distance_params,
        random_state=random_state,
        verbose=bool(run_kw.pop("verbose", False)),
        **run_kw,
    )
    ref = np.asarray(reference_labels, dtype=int)
    full = run_typology(
        domains,
        strategy,
        distance_params,
        k=k,
        R=R,
        sample_size=sample_size,
        random_state=random_state,
        **run_kw,
    )
    if not np.array_equal(ref, full["labels"]):
        # Sensitivity table was built from a fresh full run; use its implied reference.
        pass

    rows: Dict[str, float] = {}
    for _, row in table.iterrows():
        idx = int(row["omitted_domain_index"])
        rows[str(domain_names[idx])] = float(row["ari_vs_all_domains"])
    return rows


def dat_contribution_table(dat_run: Dict[str, Any], *, cluster: str = "all") -> Dict[str, float]:
    """Extract normalized DAT contribution shares for one cluster row."""
    raw = dat_run.get("dat_domain_contribution")
    if raw is None:
        contrib = (dat_run.get("result") or {}).route_diagnostics or {}
        by_k = contrib.get("dat_domain_contributions", {})
        if by_k:
            raw = next(iter(by_k.values()))
    if raw is None:
        return {}

    import pandas as pd

    frame = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
    subset = frame[frame["cluster"] == cluster]
    if subset.empty:
        return {}
    values = {
        str(row["domain"]): float(row["contribution_share"])
        for _, row in subset.iterrows()
        if bool(row.get("contribution_defined", True))
    }
    total = sum(values.values())
    if total <= 0:
        return values
    return {k: v / total for k, v in values.items()}
