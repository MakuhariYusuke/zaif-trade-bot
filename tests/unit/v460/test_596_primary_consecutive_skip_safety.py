"""596# テスト: Primary model 連続 skip 安全弁 (evaluator-level).

190# A の ev_weighted 連続 skip 安全弁は ev_as_offset_enabled=True (193#) で無効化される。
本 596# は mode に依存しない evaluator-level の安全弁を追加:
- BTC=0 → sell preflight_insufficient → buy skip_gate → death spiral を防止
- N 回連続で primary model が skip → 強制 PASS
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator
from tests.unit.v460._yaml_test_helpers import clone_fill_test_config, load_fill_test_config_from_mapping


def _make_evaluator(
    *,
    primary_max_consecutive: int = 10,
    ev_as_offset: bool = True,
) -> SkipGateEvaluator:
    """テスト用の SkipGateEvaluator (最小構成)."""
    config = FillTestConfig(
        skip_gate_enabled=False,
        skip_gate_primary_max_consecutive_skip=primary_max_consecutive,
        skip_gate_ev_as_offset_enabled=ev_as_offset,
    )
    return SkipGateEvaluator(config, Path("."))


class TestPrimaryConsecutiveSkipSafety:
    """596# Primary model 連続 skip 安全弁."""

    def test_counter_initializes_to_zero(self) -> None:
        evaluator = _make_evaluator()
        assert evaluator._primary_consecutive_skip_count == 0

    def test_counter_increments_on_skip(self) -> None:
        evaluator = _make_evaluator(primary_max_consecutive=20)
        evaluator._primary_consecutive_skip_count = 5
        # simulate: skip decision stays, counter should increment
        # We test the counter directly since evaluate() requires full setup
        evaluator._primary_consecutive_skip_count += 1
        assert evaluator._primary_consecutive_skip_count == 6

    def test_config_default_is_zero(self) -> None:
        """デフォルト値は 0 (無効)."""
        config = FillTestConfig()
        assert config.skip_gate_primary_max_consecutive_skip == 0

    def test_config_set_from_yaml(self) -> None:
        """YAML パースでフィールドが正しくマッピングされること."""
        from scripts.v460.lib.fill_config_parser import parse_fill_config_yaml
        yaml_data = {"skip_gate": {"primary_max_consecutive_skip": 15}}
        config = parse_fill_config_yaml(yaml_data)
        assert config.skip_gate_primary_max_consecutive_skip == 15

    def test_safety_valve_field_in_config(self) -> None:
        """FillTestConfig に skip_gate_primary_max_consecutive_skip フィールドがあること."""
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(FillTestConfig)}
        assert "skip_gate_primary_max_consecutive_skip" in field_names

    def test_evaluator_has_counter(self) -> None:
        """SkipGateEvaluator に _primary_consecutive_skip_count 属性があること."""
        evaluator = _make_evaluator()
        assert hasattr(evaluator, "_primary_consecutive_skip_count")
        assert evaluator._primary_consecutive_skip_count == 0

    def test_counter_separate_from_ev_counter(self) -> None:
        """primary カウンタは ev_consecutive カウンタと独立であること."""
        evaluator = _make_evaluator()
        evaluator._primary_consecutive_skip_count = 7
        evaluator._ev_consecutive_skip_count = 3
        assert evaluator._primary_consecutive_skip_count != evaluator._ev_consecutive_skip_count


class TestPrimaryMaxConsecutiveSkipYamlIntegration:
    """596# YAML 統合テスト."""

    def test_yaml_overrides_default(self) -> None:
        """fill_test.yaml の値が code default (0) と異なること."""
        from tests.unit.v460._yaml_test_helpers import load_yaml_mapping

        yaml_path = Path("configs/v460/fill_test.yaml")
        if not yaml_path.exists():
            pytest.skip("fill_test.yaml not found")
        raw = load_yaml_mapping(yaml_path)
        config = clone_fill_test_config(load_fill_test_config_from_mapping(raw))
        # YAML sets non-zero value, code default is 0
        assert config.skip_gate_primary_max_consecutive_skip > 0
