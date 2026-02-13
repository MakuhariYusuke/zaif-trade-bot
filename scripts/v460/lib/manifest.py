"""
v460 実験 manifest — JSONL 追記型.

001# §4.3 準拠. 既存 ztb/utils/run_manifest.py をベースに
JSONL 形式へ変換し、依存ハッシュ・Python/CUDA バージョン等を追加。

保存先: results/v460/manifest.jsonl (追記専用)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _get_git_sha() -> str:
    """Current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, cwd=str(_PROJECT_ROOT),
        )
        return result.stdout.strip()[:12]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _get_deps_hash() -> str:
    """SHA-256 of `pip freeze` output for reproducibility."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, check=True,
        )
        return hashlib.sha256(result.stdout.encode()).hexdigest()[:16]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _get_cuda_version() -> Optional[str]:
    """CUDA version or None."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.version.cuda
    except ImportError:
        pass
    return None


def compute_file_hash(path: Path) -> str:
    """SHA-256 of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def compute_config_hash(config: dict[str, Any]) -> str:
    """SHA-256 of sorted JSON config."""
    s = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


@dataclass
class ManifestEntry:
    """Single run entry for manifest.jsonl — §4.3 schema."""

    run_id: str
    config_path: str
    config_hash: str
    data_hash: str
    git_commit: str
    gate: str
    seed: int
    python_version: str
    deps_hash: str
    cuda_version: Optional[str]
    started_at: str
    finished_at: Optional[str] = None
    status: str = "running"
    metrics: dict[str, Any] = field(default_factory=dict)
    gate_result: Optional[str] = None
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


class ManifestWriter:
    """JSONL append-only manifest writer."""

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path) if path else _PROJECT_ROOT / "results" / "v460" / "manifest.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def start_run(
        self,
        config_path: str,
        config: dict[str, Any],
        data_path: str,
        gate: str,
        seed: int,
    ) -> ManifestEntry:
        """Create and write a new manifest entry at run start."""
        now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        gate_short = gate.lower().replace("-", "")
        run_id = f"v460_{gate_short}_seed{seed}_{now}"

        data_hash = compute_file_hash(Path(data_path)) if Path(data_path).exists() else "pending"

        entry = ManifestEntry(
            run_id=run_id,
            config_path=config_path,
            config_hash=compute_config_hash(config),
            data_hash=data_hash,
            git_commit=_get_git_sha(),
            gate=gate,
            seed=seed,
            python_version=sys.version.split()[0],
            deps_hash=_get_deps_hash(),
            cuda_version=_get_cuda_version(),
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._append(entry)
        return entry

    def finish_run(
        self,
        entry: ManifestEntry,
        metrics: dict[str, Any],
        gate_result: str,
        artifacts: Optional[list[str]] = None,
        status: str = "completed",
    ) -> None:
        """Write a completed manifest line.

        NOTE (003# #15): By design, each run writes TWO lines to the manifest:
          1. start_run() → status="running" (intent record / crash-recovery marker)
          2. finish_run() → status="completed" (final result with metrics)
        This is an event-log pattern, not a bug. To get the latest state,
        group by run_id and take the last entry for each.
        """
        entry.finished_at = datetime.now(timezone.utc).isoformat()
        entry.status = status
        entry.metrics = metrics
        entry.gate_result = gate_result
        entry.artifacts = artifacts or []
        self._append(entry)

    def _append(self, entry: ManifestEntry) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(entry.to_json() + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        """Read all manifest entries."""
        if not self.path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
