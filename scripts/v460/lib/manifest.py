"""
v460 実験 manifest — JSONL 追記型.

001# §4.3 準拠. 既存 ztb/utils/run_manifest.py をベースに
JSONL 形式へ変換し、依存ハッシュ・Python/CUDA バージョン等を追加。

保存先: results/v460/manifest.jsonl (追記専用)
"""

from __future__ import annotations

import logging
import hashlib
import json
import os
import subprocess
import sys
from functools import lru_cache
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from ztb.types.common import ConfigSection, JSONDict
from ztb.utils.git_utils import get_git_sha as _get_shared_git_sha
from ztb.utils.run_manifest import compute_file_hash as _compute_shared_file_hash


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_git_sha() -> str:
    """Current git commit SHA."""
    return _get_shared_git_sha(cwd=_PROJECT_ROOT)[:12]


@lru_cache(maxsize=1)
def _get_deps_hash() -> str:
    """Dependency fingerprint hash.

    Fast path uses importlib.metadata to avoid subprocess overhead.
    Falls back to `pip freeze` for compatibility.
    """
    try:
        from importlib import metadata

        lines: list[str] = []
        for dist in metadata.distributions():
            name = dist.metadata.get("Name") or dist.metadata.get("name")
            if not name:
                continue
            version = dist.version or "unknown"
            lines.append(f"{name}=={version}")
        if lines:
            lines.sort()
            return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]
    except Exception:
        # metadata API unavailable/broken -> fallback below
        pass

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, check=True,
        )
        return hashlib.sha256(result.stdout.encode()).hexdigest()[:16]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@lru_cache(maxsize=1)
def _get_cuda_version() -> str | None:
    """Best-effort CUDA version or None.

    By default this avoids importing torch (very expensive) unless:
      - torch is already imported in this process, or
      - ZTB_MANIFEST_DETECT_CUDA=1 is set.
    """
    detect_cuda = os.getenv("ZTB_MANIFEST_DETECT_CUDA", "").lower() in {"1", "true", "yes", "on"}

    if not detect_cuda:
        torch_mod = sys.modules.get("torch")
        if torch_mod is None:
            return None
        try:
            if torch_mod.cuda.is_available():  # type: ignore[attr-defined]
                return cast(str | None, torch_mod.version.cuda)  # type: ignore[attr-defined]
        except Exception:
            return None
        return None

    try:
        import torch
        if torch.cuda.is_available():
            return torch.version.cuda
    except ImportError:
        pass
    return None


def compute_file_hash(path: Path) -> str:
    """SHA-256 of a file."""
    return _compute_shared_file_hash(path)


def compute_config_hash(config: ConfigSection) -> str:
    """SHA-256 of sorted JSON config."""
    s = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _compute_data_hash(data_path: str) -> str:
    path = Path(data_path).expanduser()
    if not data_path or not path.exists() or not path.is_file():
        return "pending"
    try:
        return compute_file_hash(path)
    except OSError:
        logger.warning("Failed to hash data file: %s", path)
        return "pending"


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
    cuda_version: str | None
    started_at: str
    finished_at: str | None = None
    status: str = "running"
    metrics: JSONDict = field(default_factory=dict)
    gate_result: str | None = None
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> JSONDict:
        return cast(JSONDict, asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


class ManifestWriter:
    """JSONL append-only manifest writer."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else _PROJECT_ROOT / "results" / "v460" / "manifest.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def start_run(
        self,
        config_path: str,
        config: ConfigSection,
        data_path: str,
        gate: str,
        seed: int,
    ) -> ManifestEntry:
        """Create and write a new manifest entry at run start."""
        now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        gate_short = gate.lower().replace("-", "")
        run_id = f"v460_{gate_short}_seed{seed}_{now}"

        data_hash = _compute_data_hash(data_path)

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
        metrics: JSONDict,
        gate_result: str,
        artifacts: list[str] | None = None,
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
            f.flush()
            os.fsync(f.fileno())  # 032#16: ディスクフル時の部分書き込み防止

    def read_all(self) -> list[JSONDict]:
        """Read all manifest entries, skipping malformed lines."""
        if not self.path.exists():
            return []
        entries: list[JSONDict] = []
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, 1):
                parsed = _parse_manifest_line(
                    line,
                    path=self.path,
                    line_no=line_no,
                )
                if parsed is not None:
                    entries.append(parsed)
        return entries


def _parse_manifest_line(
    line: str,
    *,
    path: Path,
    line_no: int,
) -> JSONDict | None:
    stripped = line.strip()
    if not stripped:
        return None
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        logger.warning("Skipping malformed manifest line: %s:%d", path, line_no)
        return None
    if not isinstance(loaded, dict):
        logger.warning("Skipping non-object manifest line: %s:%d", path, line_no)
        return None
    return cast(JSONDict, loaded)
