from __future__ import annotations

from pathlib import Path

from ztb.ml.skip_gate import SkipGate


class PickleStub:
    """pickle 可能な最小 stub."""

    def __init__(self, name: str) -> None:
        self.name = name


def save_and_load_skip_gate(gate: SkipGate, path: Path) -> SkipGate:
    gate.save(path)
    return SkipGate.load(path)
