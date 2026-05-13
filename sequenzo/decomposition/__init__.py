"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 2026-03-01 10:51
@Desc    :
Group-difference decomposition APIs (KOB / Oaxaca-Blinder / SA-KOB).
"""

from .kob import (
    KOBDecompositionResult,
    KOBBootstrapResult,
    get_kob_decomposition,
    get_kob_decomposition_bootstrap,
)
from .oaxaca import get_oaxaca_blinder_decomposition
from .results import SAKOBDecompositionResult, SAKOBBootstrapResult
from .sa_kob import (
    ClusterCovariates,
    build_cluster_covariates,
    cluster_group_composition_table,
    detect_cluster_coefficient_owners,
    get_sa_kob_decomposition,
    get_sa_kob_decomposition_bootstrap,
)

__all__ = [
    "get_kob_decomposition",
    "get_kob_decomposition_bootstrap",
    "get_oaxaca_blinder_decomposition",
    "get_sa_kob_decomposition",
    "get_sa_kob_decomposition_bootstrap",
    "KOBDecompositionResult",
    "KOBBootstrapResult",
    "SAKOBDecompositionResult",
    "SAKOBBootstrapResult",
    "ClusterCovariates",
    "build_cluster_covariates",
    "cluster_group_composition_table",
    "detect_cluster_coefficient_owners",
]
