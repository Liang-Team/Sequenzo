"""Pytest configuration for multidomain CLARA tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Paper/benchmark scripts live outside the sequenzo package namespace.
_MD_CLARA_ROOT = Path(__file__).resolve().parents[2].parent / "multidomain_clara"
if _MD_CLARA_ROOT.is_dir():
    root = str(_MD_CLARA_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
