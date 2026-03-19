"""183# ログ分析ベース改善テスト

Changes:
  1. Buy velocity skip 有効化 + 閾値保守化 (8→6 bps)
  2. 時間帯別 skip_gate 閾値オフセット (hour_offsets)
  3. 狭スプレッド adverse guard (skip_gate 閾値加算)
  4. VG/VPIN 感度引上げ
  5. Narrow spread boost 強化
"""

from __future__ import annotations

import copy

import pytest

from scripts.v460.lib.config_hot_reload import _HOT_RELOADABLE_FIELDS
from scripts.v460.lib.fill_config import FillTestConfig
from tests.unit.v460._yaml_test_helpers import parse_yaml_mapping


_BUY_VELOCITY_SKIP_YAML = parse_yaml_mapping("""
skip_gate:
  enabled: true
  mode: pnl
  buy_velocity_skip_enabled: true
  buy_velocity_skip_threshold_bps: -6.0
  sell_velocity_skip_enabled: true
  sell_velocity_skip_threshold_bps: 6.0
""")

_HOUR_OFFSETS_YAML = parse_yaml_mapping("""
skip_gate:
  enabled: true
  mode: pnl
  hour_offsets:
    14: 0.3
    16: 0.5
    18: 0.3
    21: 0.3
    23: 0.2
""")

_NARROW_SPREAD_GUARD_YAML = parse_yaml_mapping("""
skip_gate:
  enabled: true
  mode: pnl
  skip_gate_narrow_spread_threshold_jpy: 2000.0
  skip_gate_narrow_spread_offset: 0.2
""")

_VOLATILITY_GUARD_YAML = parse_yaml_mapping("""
volatility_guard:
  enabled: true
  velocity_threshold_bps: 12.0
  vpin_threshold: 0.60
  offset_boost_factor: 2.0
""")

_VOLATILITY_GUARD_TIGHT_YAML = parse_yaml_mapping("""
volatility_guard:
  enabled: true
  velocity_threshold_bps: 12.0
  vpin_threshold: 0.60
""")

_SPREAD_ADAPTIVE_YAML = parse_yaml_mapping("""
spread_adaptive:
  enabled: true
  narrow_spread_bps: 2.5
  narrow_spread_boost_buy: 2.0
  narrow_spread_boost_sell: 2.5
""")


@pytest.fixture
def buy_velocity_skip_yaml() -> dict[str, object]:
    return copy.deepcopy(_BUY_VELOCITY_SKIP_YAML)


@pytest.fixture
def hour_offsets_yaml() -> dict[str, object]:
    return copy.deepcopy(_HOUR_OFFSETS_YAML)


@pytest.fixture
def narrow_spread_guard_yaml() -> dict[str, object]:
    return copy.deepcopy(_NARROW_SPREAD_GUARD_YAML)


@pytest.fixture
def volatility_guard_yaml() -> dict[str, object]:
    return copy.deepcopy(_VOLATILITY_GUARD_YAML)


@pytest.fixture
def volatility_guard_tight_yaml() -> dict[str, object]:
    return copy.deepcopy(_VOLATILITY_GUARD_TIGHT_YAML)


@pytest.fixture
def spread_adaptive_yaml() -> dict[str, object]:
    return copy.deepcopy(_SPREAD_ADAPTIVE_YAML)


# =====================================================================
# 1. Buy velocity skip enable + threshold tuning
# =====================================================================
class TestBuyVelocitySkipEnable:
    """183# buy_velocity_skip 有効化テスト."""

    def test_buy_velocity_skip_default_false(self) -> None:
        cfg = FillTestConfig()
        assert cfg.buy_velocity_skip_enabled is False

    def test_buy_velocity_skip_from_yaml(self, buy_velocity_skip_yaml: dict[str, object]) -> None:
        cfg = FillTestConfig.from_yaml(buy_velocity_skip_yaml)
        assert cfg.buy_velocity_skip_enabled is True
        assert cfg.buy_velocity_skip_threshold_bps == pytest.approx(-6.0)
        assert cfg.sell_velocity_skip_enabled is True
        assert cfg.sell_velocity_skip_threshold_bps == pytest.approx(6.0)

    def test_velocity_thresholds_symmetric(self) -> None:
        """183# buy/sell 閾値の対称性を確認."""
        cfg = FillTestConfig(
            buy_velocity_skip_enabled=True,
            buy_velocity_skip_threshold_bps=-6.0,
            sell_velocity_skip_enabled=True,
            sell_velocity_skip_threshold_bps=6.0,
        )
        assert abs(cfg.buy_velocity_skip_threshold_bps) == pytest.approx(
            abs(cfg.sell_velocity_skip_threshold_bps)
        )


# =====================================================================
# 2. hour_offsets YAML 統合テスト
# =====================================================================
class TestHourOffsetsYAML:
    """183# 時間帯別閾値オフセットの YAML → Config 統合テスト."""

    def test_hour_offsets_from_full_yaml(self, hour_offsets_yaml: dict[str, object]) -> None:
        cfg = FillTestConfig.from_yaml(hour_offsets_yaml)
        assert cfg.skip_gate_hour_offsets[14] == pytest.approx(0.3)
        assert cfg.skip_gate_hour_offsets[16] == pytest.approx(0.5)
        assert cfg.skip_gate_hour_offsets[18] == pytest.approx(0.3)
        assert cfg.skip_gate_hour_offsets[21] == pytest.approx(0.3)
        assert cfg.skip_gate_hour_offsets[23] == pytest.approx(0.2)
        # 未設定時間帯
        assert cfg.skip_gate_hour_offsets.get(0, 0.0) == 0.0
        assert cfg.skip_gate_hour_offsets.get(12, 0.0) == 0.0

    def test_hour_offsets_worst_hour_highest(self, hour_offsets_yaml: dict[str, object]) -> None:
        """183# 最悪時間帯 (01h JST=16 UTC) が最高オフセット."""
        cfg = FillTestConfig.from_yaml(hour_offsets_yaml)
        max_hour = max(cfg.skip_gate_hour_offsets, key=cfg.skip_gate_hour_offsets.get)  # type: ignore[arg-type]
        assert max_hour == 16  # 01h JST = 16h UTC (AS 64%)

    def test_all_offsets_positive(self, hour_offsets_yaml: dict[str, object]) -> None:
        """183# 厳格化方向 (正) のみであること."""
        cfg = FillTestConfig.from_yaml(hour_offsets_yaml)
        for h, offset in cfg.skip_gate_hour_offsets.items():
            assert offset > 0, f"Hour {h} has non-positive offset {offset}"


# =====================================================================
# 3. Narrow spread adverse guard
# =====================================================================
class TestNarrowSpreadAdverseGuard:
    """183# 狭スプレッド逆選択防御テスト."""

    def test_config_defaults(self) -> None:
        cfg = FillTestConfig()
        assert cfg.skip_gate_narrow_spread_threshold_jpy == 0.0
        assert cfg.skip_gate_narrow_spread_offset == 0.0

    def test_config_explicit(self) -> None:
        cfg = FillTestConfig(
            skip_gate_narrow_spread_threshold_jpy=2000.0,
            skip_gate_narrow_spread_offset=0.2,
        )
        assert cfg.skip_gate_narrow_spread_threshold_jpy == pytest.approx(2000.0)
        assert cfg.skip_gate_narrow_spread_offset == pytest.approx(0.2)

    def test_yaml_parsing(self, narrow_spread_guard_yaml: dict[str, object]) -> None:
        cfg = FillTestConfig.from_yaml(narrow_spread_guard_yaml)
        assert cfg.skip_gate_narrow_spread_threshold_jpy == pytest.approx(2000.0)
        assert cfg.skip_gate_narrow_spread_offset == pytest.approx(0.2)

    def test_disabled_when_threshold_zero(self) -> None:
        """threshold=0.0 のときは無効 (デフォルト)."""
        cfg = FillTestConfig(
            skip_gate_narrow_spread_threshold_jpy=0.0,
            skip_gate_narrow_spread_offset=0.5,
        )
        # threshold=0 なので spread_offset は適用されない
        assert cfg.skip_gate_narrow_spread_threshold_jpy == 0.0

    def test_hot_reload_includes_narrow_spread(self) -> None:
        """183# hot-reload 対象に narrow_spread パラメータが含まれる."""
        assert "skip_gate_narrow_spread_threshold_jpy" in _HOT_RELOADABLE_FIELDS
        assert "skip_gate_narrow_spread_offset" in _HOT_RELOADABLE_FIELDS


# =====================================================================
# 4. VG / VPIN threshold tuning (YAML-only, 構造テスト)
# =====================================================================
class TestVolatilityGuardTuning:
    """183# VG/VPIN 感度引上げテスト (YAML → Config)."""

    def test_vg_threshold_from_yaml(self, volatility_guard_yaml: dict[str, object]) -> None:
        cfg = FillTestConfig.from_yaml(volatility_guard_yaml)
        assert cfg.volatility_guard_velocity_threshold_bps == pytest.approx(12.0)
        assert cfg.volatility_guard_vpin_threshold == pytest.approx(0.60)
        assert cfg.volatility_guard_offset_boost_factor == pytest.approx(2.0)

    def test_vg_threshold_tighter_than_previous(
        self,
        volatility_guard_tight_yaml: dict[str, object],
    ) -> None:
        """183# VG 閾値が以前の 15.0 より厳しい (低い)."""
        cfg = FillTestConfig.from_yaml(volatility_guard_tight_yaml)
        assert cfg.volatility_guard_velocity_threshold_bps < 15.0  # prev: 15.0
        assert cfg.volatility_guard_vpin_threshold < 0.63           # prev: 0.63


# =====================================================================
# 5. Narrow spread boost 強化 (YAML → Config)
# =====================================================================
class TestNarrowSpreadBoostTuning:
    """183# spread_adaptive boost 強化テスト."""

    def test_spread_adaptive_boost_from_yaml(self, spread_adaptive_yaml: dict[str, object]) -> None:
        cfg = FillTestConfig.from_yaml(spread_adaptive_yaml)
        assert cfg.narrow_spread_boost_buy == pytest.approx(2.0)
        assert cfg.narrow_spread_boost_sell == pytest.approx(2.5)

    def test_sell_boost_higher_than_buy(self) -> None:
        """183# sell 側は buy より高い boost (AS 構造差)."""
        cfg = FillTestConfig(
            narrow_spread_boost_buy=2.0,
            narrow_spread_boost_sell=2.5,
        )
        assert cfg.narrow_spread_boost_sell > cfg.narrow_spread_boost_buy


# =====================================================================
# 6. Full integration: fill_test.yaml 読み込みテスト
# =====================================================================
class TestFillTestYAMLIntegration:
    """183# 実 YAML ファイルの読み込みが成功し、183# 設定が反映される."""

    def test_load_fill_test_yaml(self, v460_fill_test_yaml: dict[str, object]) -> None:
        data = v460_fill_test_yaml
        cfg = FillTestConfig.from_yaml(data)

        # 183# velocity skip  (353# buy -6→-4)
        assert cfg.buy_velocity_skip_enabled is True
        assert cfg.buy_velocity_skip_threshold_bps == pytest.approx(-4.0)
        assert cfg.sell_velocity_skip_threshold_bps == pytest.approx(6.0)

        # 183# hour offsets
        assert 16 in cfg.skip_gate_hour_offsets
        assert cfg.skip_gate_hour_offsets[16] == pytest.approx(0.5)

        # 183# narrow spread adverse guard
        assert cfg.skip_gate_narrow_spread_threshold_jpy == pytest.approx(2000.0)
        assert cfg.skip_gate_narrow_spread_offset == pytest.approx(0.2)

        # 183# VG tuning
        assert cfg.volatility_guard_velocity_threshold_bps == pytest.approx(12.0)
        assert cfg.volatility_guard_vpin_threshold == pytest.approx(0.80)

        # 183# narrow spread boost
        assert cfg.narrow_spread_boost_buy == pytest.approx(2.0)
        assert cfg.narrow_spread_boost_sell == pytest.approx(2.5)
