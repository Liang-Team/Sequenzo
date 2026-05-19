"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py.py
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
from .visualization import (
    plot_md_clara_memory,
    plot_md_clara_quality,
    plot_md_clara_runtime,
    plot_md_clara_stability,
    plot_md_cluster_by_domain,
)

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
    "plot_md_cluster_by_domain",
]
