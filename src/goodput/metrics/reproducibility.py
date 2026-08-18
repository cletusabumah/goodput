"""Git SHA, package versions, and config hash for run reports (ticket 3.4).

Logged on every ``build_run_report`` so a JSON artifact can be traced to a
commit + dependency set + training knobs. Paths and provider backends are left
out of the hash so the same experiment YAML hashes the same on two laptops.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from goodput.config import Settings

# Training knobs that change the run. Local paths / sinks stay out so a tmp
# artifacts_dir does not produce a different hash than the committed recipe.
CONFIG_HASH_FIELDS: tuple[str, ...] = (
    "device",
    "num_workers",
    "rank",
    "seed",
    "steps",
    "batch_size",
    "input_size",
    "hidden_size",
    "learning_rate",
    "ckpt_interval",
    "ckpt_mode",
    "ckpt_full_every",
    "fault_mode",
    "fault_at",
    "fault_mean_interval",
    "health_check_timeout_s",
    "bitflip_detect",
    "bitflip_grad_ratio_threshold",
)

# Core deps from pyproject — not a full ``pip freeze`` (editable paths drift).
VERSIONED_PACKAGES: tuple[str, ...] = (
    "goodput",
    "torch",
    "numpy",
    "pydantic",
    "pydantic-settings",
    "PyYAML",
)


def git_sha(*, start: Path | None = None) -> str | None:
    """HEAD SHA, or None when git is missing / this is not a checkout."""
    cwd = start if start is not None else Path(__file__).resolve().parent
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    sha = out.strip()
    return sha or None


def git_dirty(*, start: Path | None = None) -> bool:
    """True when the worktree has uncommitted changes (False if git unavailable)."""
    cwd = start if start is not None else Path(__file__).resolve().parent
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return bool(out.strip())


def package_versions() -> dict[str, str]:
    """Installed versions of the packages that actually train / configure runs."""
    versions: dict[str, str] = {}
    for name in VERSIONED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def versions_hash(versions: dict[str, str] | None = None) -> str:
    """SHA-256 of the canonical package-version mapping (short hex)."""
    payload = versions if versions is not None else package_versions()
    return _sha256_canonical(payload)[:16]


def config_payload(settings: Settings) -> dict[str, Any]:
    dump = settings.model_dump(mode="json")
    return {key: dump[key] for key in CONFIG_HASH_FIELDS}


def config_hash(settings: Settings) -> str:
    """SHA-256 of training knobs (short hex)."""
    return _sha256_canonical(config_payload(settings))[:16]


def reproducibility_fields(settings: Settings) -> dict[str, Any]:
    """Block merged into every run report."""
    versions = package_versions()
    return {
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "config_hash": config_hash(settings),
        "package_versions": versions,
        "versions_hash": versions_hash(versions),
    }


def _sha256_canonical(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
