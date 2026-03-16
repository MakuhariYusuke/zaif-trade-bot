from __future__ import annotations

from pathlib import Path


def atomic_pickle_tmp_path(path: Path) -> Path:
    """原子置換用の一時 pickle path を返す."""
    return path.with_suffix(".pkl.tmp")


def hash_sidecar_path(path: Path) -> Path:
    """artifact に対応する SHA256 sidecar path を返す."""
    return path.with_suffix(path.suffix + ".sha256")
