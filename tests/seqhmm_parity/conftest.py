import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RECORDS_DIR = REPO_ROOT / "experiments" / "records"
DEFAULT_PARITY_JSONL = RECORDS_DIR / "parity_results.jsonl"


def _cmd_text(*args):
    try:
        result = subprocess.run(
            args,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or result.stderr.strip() or None


def _sequenzo_version():
    try:
        from sequenzo import __version__
    except Exception:
        return "unknown"
    return __version__


def _schema_meta(path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "record_type": "schema_meta",
                "schema_version": "1.0",
                "kit_version": "1.1",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def record_parity_result():
    def _record(**payload):
        if os.environ.get("SEQUENZO_RECORD_PARITY") != "1":
            return
        parity_jsonl = Path(
            os.environ.get("SEQUENZO_PARITY_RECORD_PATH", str(DEFAULT_PARITY_JSONL))
        )
        git_status = _cmd_text("git", "status", "--porcelain")
        base = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": _cmd_text("git", "rev-parse", "HEAD") or "unknown",
            "working_tree_dirty": bool(git_status),
            "python_version": sys.version.split()[0],
            "r_version": _cmd_text("R", "--version").splitlines()[0]
            if _cmd_text("R", "--version")
            else None,
            "sequenzo_version": _sequenzo_version(),
            "seqhmm_version": _cmd_text(
                "Rscript",
                "-e",
                "cat(as.character(utils::packageVersion('seqHMM')))",
            ),
            "os": f"{platform.system()} {platform.release()}",
            "cpu": platform.machine(),
            "notes": "pytest parity assertion",
            "record_type": "parity_result",
        }
        base.update(payload)
        _schema_meta(parity_jsonl)
        with parity_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(base, sort_keys=True) + "\n")

    return _record
