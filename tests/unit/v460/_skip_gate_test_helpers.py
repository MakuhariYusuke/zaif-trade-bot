from __future__ import annotations

from pathlib import Path

from scripts.v460.ml.skip_gate import SkipGate


def save_and_load_skip_gate(gate: SkipGate, path: Path) -> SkipGate:
    gate.save(path)
    return SkipGate.load(path)
