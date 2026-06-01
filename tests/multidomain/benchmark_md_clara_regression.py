#!/usr/bin/env python3
"""
Small-scale MD-CLARA regression benchmark (N ~ 500–2000).

Compares cached vs uncached medoid-column paths across strategies.
The production engine always uses condensed within-sample distances; unit
tests already verify condensed == square at the provider level.

Run from repo root:
    python3 tests/multidomain/benchmark_md_clara_regression.py
"""

from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sequenzo import SequenceData
from sequenzo.multidomain.clara.diagnostics import dat_domain_contributions
from sequenzo.multidomain.clara.md_clara import md_clara


def _generate_domains(
    n_sequences: int,
    *,
    n_domains: int = 3,
    n_time: int = 8,
    n_states: int = 3,
    random_state: int = 0,
) -> List[SequenceData]:
    rng = np.random.default_rng(random_state)
    time_cols = [f"t{i}" for i in range(1, n_time + 1)]
    domains: List[SequenceData] = []
    state_list = list(range(n_states))

    for _ in range(n_domains):
        block = rng.integers(0, n_states, size=(n_sequences, n_time))
        df = pd.DataFrame(block, columns=time_cols)
        df.insert(0, "id", np.arange(1, n_sequences + 1))
        domains.append(
            SequenceData(
                data=df,
                time=time_cols,
                id_col="id",
                states=state_list,
            )
        )
    return domains


def _distance_params(strategy: str) -> Dict[str, Any]:
    if strategy == "idcd":
        return {"method": "HAM", "sm": "CONSTANT", "indel": 1, "norm": "none"}
    if strategy == "cat":
        return {
            "method": "HAM",
            "sm": ["CONSTANT"] * 3,
            "indel": 1,
            "norm": "none",
        }
    return {
        "method_params": [{"method": "HAM"}] * 3,
        "domain_weights": [0.5, 2.0, 1.0],
        "link": "mean",
        "n_jobs_domains": 1,
    }


@dataclass
class RunResult:
    label: str
    ok: bool
    runtime_s: float
    n_unique_profiles: Optional[int] = None
    cache_hits: Optional[int] = None
    cache_misses: Optional[int] = None
    peak_cached_columns: Optional[int] = None
    error: Optional[str] = None


def _run_md_clara(
    domains: List[SequenceData],
    strategy: str,
    *,
    condensed_subsample: bool,
    use_medoid_cache: bool,
    random_state: int,
    R: int,
    sample_size: int,
    kvals: List[int],
) -> Tuple[Any, RunResult]:
    label = (
        f"{strategy}_condensed={condensed_subsample}_cache={use_medoid_cache}"
    )
    t0 = time.perf_counter()
    try:
        result = md_clara(
            domains,
            strategy=strategy,
            distance_params=_distance_params(strategy),
            R=R,
            sample_size=sample_size,
            kvals=kvals,
            criteria=("distance",),
            stability=False,
            random_state=random_state,
            n_jobs=1,
            verbose=False,
            combined_state_space=False,
            condensed_subsample=condensed_subsample,
            use_medoid_cache=use_medoid_cache,
            dat_domain_contribution=(strategy == "dat"),
        )
        elapsed = time.perf_counter() - t0
        cache = (result.route_diagnostics or {}).get("medoid_cache", {})
        return result, RunResult(
            label=label,
            ok=True,
            runtime_s=elapsed,
            n_unique_profiles=result.settings.get("n_unique_profiles"),
            cache_hits=cache.get("hits"),
            cache_misses=cache.get("misses"),
            peak_cached_columns=cache.get("peak_cached_columns"),
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return None, RunResult(
            label=label,
            ok=False,
            runtime_s=elapsed,
            error=f"{type(exc).__name__}: {exc}",
        )


def _compare_results(
    reference,
    other,
    *,
    strategy: str,
    kvals: List[int],
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> List[str]:
    """Return list of mismatch descriptions (empty if equivalent)."""
    issues: List[str] = []

    if not np.allclose(
        reference.stats["total_diss"].to_numpy(),
        other.stats["total_diss"].to_numpy(),
        rtol=rtol,
        atol=atol,
    ):
        issues.append("total_diss differs between runs")

    if not np.array_equal(
        reference.clustering.to_numpy(),
        other.clustering.to_numpy(),
    ):
        issues.append("case-level clustering labels differ")

    for k in kvals:
        if not np.array_equal(reference.medoids[k], other.medoids[k]):
            issues.append(f"medoids differ at k={k}")

    if strategy == "dat":
        contrib_ref = (reference.route_diagnostics or {}).get(
            "dat_domain_contributions", {}
        )
        contrib_other = (other.route_diagnostics or {}).get(
            "dat_domain_contributions", {}
        )
        for k in kvals:
            if k not in contrib_ref or k not in contrib_other:
                issues.append(f"DAT contributions missing for k={k}")
                continue
            all_ref = contrib_ref[k][contrib_ref[k]["cluster"] == "all"]
            if not np.isclose(all_ref["contribution_share"].sum(), 1.0, atol=1e-5):
                issues.append(f"DAT contribution shares do not sum to 1 (k={k})")

    return issues


# (label, condensed_subsample, use_medoid_cache)
ABLATION_ARMS: Tuple[Tuple[str, bool, bool], ...] = (
    ("square", False, False),
    ("condensed", True, False),
    ("condensed+cache", True, True),
)


def benchmark_n(
    n_sequences: int,
    *,
    strategies: Tuple[str, ...] = ("idcd", "cat", "dat"),
    R: int = 6,
    kvals: Optional[List[int]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    if kvals is None:
        kvals = [2, 3, 4]

    max_k = max(kvals)
    sample_size = min(80 + 2 * max_k, n_sequences)

    domains = _generate_domains(n_sequences, random_state=random_state)
    rows: List[Dict[str, Any]] = []

    print(f"\n=== N = {n_sequences} (sample_size={sample_size}, R={R}, kvals={kvals}) ===")

    for strategy in strategies:
        runs: Dict[str, Tuple[Any, RunResult]] = {}
        for arm_label, condensed, use_cache in ABLATION_ARMS:
            result, meta = _run_md_clara(
                domains,
                strategy,
                condensed_subsample=condensed,
                use_medoid_cache=use_cache,
                random_state=random_state,
                R=R,
                sample_size=sample_size,
                kvals=kvals,
            )
            runs[arm_label] = (result, meta)

        baseline_res, baseline_meta = runs["condensed"]
        strategy_ok = all(meta.ok for _, meta in runs.values())

        strategy_rows: List[Dict[str, Any]] = []
        for arm_label, condensed, use_cache in ABLATION_ARMS:
            result, meta = runs[arm_label]
            equiv_ok = False
            equiv_notes = "run failed"
            if strategy_ok and result is not None and baseline_res is not None:
                if arm_label == "condensed":
                    equiv_ok = True
                    equiv_notes = "reference"
                else:
                    issues = _compare_results(
                        baseline_res,
                        result,
                        strategy=strategy,
                        kvals=kvals,
                    )
                    equiv_ok = len(issues) == 0
                    equiv_notes = "; ".join(issues) if issues else "match baseline"

            row = {
                    "n_sequences": n_sequences,
                    "strategy": strategy,
                    "ablation_arm": arm_label,
                    "condensed_subsample": condensed,
                    "use_medoid_cache": use_cache,
                    "ok": meta.ok,
                    "equivalence_ok": equiv_ok if meta.ok else False,
                    "equivalence_notes": equiv_notes,
                    "runtime_seconds": meta.runtime_s,
                    "n_unique_profiles": meta.n_unique_profiles,
                    "cache_hits": meta.cache_hits,
                    "cache_misses": meta.cache_misses,
                    "peak_cached_columns": meta.peak_cached_columns,
                    "error": meta.error,
                }
            strategy_rows.append(row)
        rows.extend(strategy_rows)

        status = "PASS" if strategy_ok and all(
            r["equivalence_ok"] for r in strategy_rows
        ) else "FAIL"
        parts = "  ".join(
            f"{arm_label}={runs[arm_label][1].runtime_s:.2f}s"
            for arm_label, _, _ in ABLATION_ARMS
        )
        print(f"  {strategy:4s} {status}  {parts}")
        if status == "FAIL":
            for row in strategy_rows:
                if not row["equivalence_ok"]:
                    print(f"    ! {row['ablation_arm']}: {row['equivalence_notes']}")

    return pd.DataFrame(rows)


def main() -> int:
    print("MD-CLARA small-scale regression benchmark")
    print("Ablation arms: square | condensed | condensed+cache")

    all_frames: List[pd.DataFrame] = []
    any_fail = False

    for n in (500, 2000):
        try:
            df = benchmark_n(n, R=6, random_state=100 + n)
            all_frames.append(df)
            if not df["equivalence_ok"].all() or not df["ok"].all():
                any_fail = True
        except Exception:
            any_fail = True
            print(f"  FATAL for N={n}:")
            traceback.print_exc()

    if all_frames:
        summary = pd.concat(all_frames, ignore_index=True)
        out_path = "tests/multidomain/_benchmark_md_clara_regression.csv"
        summary.to_csv(out_path, index=False)
        print(f"\nWrote {out_path}")

    if any_fail:
        print("\nOVERALL: FAIL — fix regressions before large simulations.")
        return 1

    print(
        "\nOVERALL: PASS — all ablation arms match condensed baseline at N=500 and N=2000."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
