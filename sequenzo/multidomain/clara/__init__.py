"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 18/05/2026 11:25
@Desc    : 
Scalable multidomain sequence analysis with CLARA (IDCD, CAT, DAT).
"""

from .distance_providers import (
    CATDistanceProvider,
    DATDistanceProvider,
    DistanceProvider,
    IDCDDistanceProvider,
    make_distance_provider,
)
from .clara_engine import clara_from_distance_provider
from .md_clara import md_clara
from .results import MDClaraResult
from .diagnostics import (
    compare_md_clara_strategies,
    dat_domain_contributions,
    leave_one_domain_out_sensitivity,
    summarize_combined_state_space,
    summarize_subsample_coverage,
)

_LAZY_PLOT_EXPORTS = {
    "plot_md_clara_memory": "plot_md_clara_memory",
    "plot_md_clara_quality": "plot_md_clara_quality",
    "plot_md_clara_runtime": "plot_md_clara_runtime",
    "plot_md_clara_stability": "plot_md_clara_stability",
    "plot_cross_strategy_agreement": "plot_cross_strategy_agreement",
    "plot_dat_domain_contributions": "plot_dat_domain_contributions",
    "plot_leave_one_domain_out_sensitivity": "plot_leave_one_domain_out_sensitivity",
}

__all__ = [
    "md_clara",
    "clara_from_distance_provider",
    "MDClaraResult",
    "DistanceProvider",
    "IDCDDistanceProvider",
    "CATDistanceProvider",
    "DATDistanceProvider",
    "make_distance_provider",
    "plot_md_clara_quality",
    "plot_md_clara_stability",
    "plot_md_clara_runtime",
    "plot_md_clara_memory",
    "compare_md_clara_strategies",
    "summarize_combined_state_space",
    "dat_domain_contributions",
    "leave_one_domain_out_sensitivity",
    "summarize_subsample_coverage",
    "plot_cross_strategy_agreement",
    "plot_dat_domain_contributions",
    "plot_leave_one_domain_out_sensitivity",
]


def __getattr__(name: str):
    if name in _LAZY_PLOT_EXPORTS:
        from . import visualization

        return getattr(visualization, _LAZY_PLOT_EXPORTS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
