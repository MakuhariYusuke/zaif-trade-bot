from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from ztb.utils.dataclass_utils import shallow_asdict


@dataclass
class _NestedConfig:
    threshold: float = 1.0
    params: dict[str, float] = field(default_factory=lambda: {"alpha": 0.5})


def test_shallow_asdict_returns_field_mapping() -> None:
    cfg = _NestedConfig(threshold=2.0, params={"alpha": 0.25})

    result = shallow_asdict(cfg)

    assert result == {"threshold": 2.0, "params": {"alpha": 0.25}}


def test_shallow_asdict_keeps_nested_mapping_identity() -> None:
    params = {"alpha": 0.25}
    cfg = _NestedConfig(params=params)

    result = shallow_asdict(cfg)

    assert result["params"] is params


def test_shallow_asdict_rejects_non_dataclass_instance() -> None:
    with pytest.raises(TypeError):
        shallow_asdict(object())
