"""
EMLT analysis (TraMineRextras ``seqemlt``).
"""

from .results import EMLTResult
from .seqemlt import compute_emlt, seqemlt

__all__ = [
    "EMLTResult",
    "compute_emlt",
    "seqemlt",
]
