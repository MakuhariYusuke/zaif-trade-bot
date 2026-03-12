"""336# fill_config_parser.py のユニットテスト.

329# で分離された YAML→FillTestConfig パーサーの正当性を検証する。

テスト観点:
  - parse_fill_config_yaml: 空 dict, flat key, ネスト section
  - _parse_trading_features: FFD, imbalance, smart_side
  - _parse_skip_gate_section: ML skip gate params
  - _parse_stopgap_section: dynamic kill, inventory skewing
  - _parse_stale_vg_section: stale order, VG
  - _parse_infra_section: PnL fee, resilience
  - production YAML round-trip: 実 YAML の parse 成功確認
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.fill_config_parser import (
    _parse_infra_section,
    _parse_skip_gate_section,
    _parse_stale_vg_section,
    _parse_stopgap_section,
    _parse_trading_features,
    parse_fill_config_yaml,
)
from tests.unit.v460._yaml_test_helpers import load_yaml_mapping

YAML_PATH = Path("configs/v460/fill_test.yaml")
_DEFAULT_FILL_CONFIG = FillTestConfig()


# ═══════════════════════════════════════════════════════════════════════
# parse_fill_config_yaml — E2E 基本テスト
# ═══════════════════════════════════════════════════════════════════════


class TestParseFillConfigYaml:
    """parse_fill_config_yaml の入出力検証."""

    def test_empty_dict_returns_defaults(self) -> None:
        """空 YAML → 全フィールドがデフォルト値."""
        cfg = parse_fill_config_yaml({})
        assert cfg.symbol == _DEFAULT_FILL_CONFIG.symbol
        assert cfg.cycle_interval_sec == _DEFAULT_FILL_CONFIG.cycle_interval_sec

    def test_flat_keys_applied(self) -> None:
        """フラットキーが正しくマッピングされる."""
        cfg = parse_fill_config_yaml({
            "symbol": "btc_jpy",
            "cycle_interval_sec": 120.0,
            "batch_size": 20,
        })
        assert cfg.symbol == "btc_jpy"
        assert cfg.cycle_interval_sec == 120.0
        assert cfg.batch_size == 20

    def test_adaptation_section(self) -> None:
        """adaptation セクションの解析."""
        cfg = parse_fill_config_yaml({
            "adaptation": {"enabled": True, "interval_cycles": 5},
        })
        assert cfg.enable_auto_adapt is True
        assert cfg.adapt_interval_cycles == 5

    def test_lot_sizing_section(self) -> None:
        """lot_sizing セクションの解析."""
        cfg = parse_fill_config_yaml({
            "lot_sizing": {"enabled": True, "max_lot": 0.01, "interval_cycles": 3},
        })
        assert cfg.enable_dynamic_lot is True
        assert cfg.max_lot == 0.01
        assert cfg.lot_adapt_interval_cycles == 3

    def test_safety_section(self) -> None:
        """safety セクションの解析."""
        cfg = parse_fill_config_yaml({
            "safety": {"loss_cap_jpy": 500, "soft_loss_cap_ratio": 0.7},
        })
        assert cfg.loss_cap_jpy == 500
        assert cfg.soft_loss_cap_ratio == 0.7

    def test_regime_section_basics(self) -> None:
        """regime 基本キーの解析."""
        cfg = parse_fill_config_yaml({
            "regime": {
                "enabled": True,
                "window": 120,
                "trend_threshold_pct": 0.03,
            },
        })
        assert cfg.enable_regime is True
        assert cfg.regime_window == 120
        assert cfg.regime_trend_threshold_pct == 0.03

    def test_regime_lot_multipliers(self) -> None:
        """regime.lot_multipliers dict の解析."""
        cfg = parse_fill_config_yaml({
            "regime": {"lot_multipliers": {"ranging": 1.5, "trending_up": 0.8}},
        })
        assert cfg.regime_lot_multipliers == {"ranging": 1.5, "trending_up": 0.8}

    def test_time_filter_section(self) -> None:
        """time_filter セクションの解析."""
        cfg = parse_fill_config_yaml({
            "time_filter": {"enabled": True, "skip_utc_hours": [1, 2, 3]},
        })
        assert cfg.enable_time_filter is True
        assert cfg.skip_utc_hours == [1, 2, 3]

    def test_offset_ceiling(self) -> None:
        """321# offset ceiling keys."""
        cfg = parse_fill_config_yaml({
            "offset_ceiling_ratio": 0.8,
            "offset_ceiling_ratio_buy": 0.7,
            "offset_ceiling_ratio_sell": 0.9,
        })
        assert cfg.offset_ceiling_ratio == 0.8
        assert cfg.offset_ceiling_ratio_buy == 0.7
        assert cfg.offset_ceiling_ratio_sell == 0.9


# ═══════════════════════════════════════════════════════════════════════
# セクションパーサー — 個別テスト
# ═══════════════════════════════════════════════════════════════════════


class TestParseTradingFeatures:
    """_parse_trading_features: FFD, imbalance, smart_side, early_exit."""

    def test_empty_returns_empty(self) -> None:
        assert _parse_trading_features({}) == {}

    def test_fast_fill_defense(self) -> None:
        result = _parse_trading_features({
            "fast_fill_defense": {
                "enabled": True,
                "threshold_sec": 3.0,
                "offset_boost": 1.5,
            },
        })
        assert result["fast_fill_defense_enabled"] is True
        assert result["fast_fill_threshold_sec"] == 3.0
        assert result["fast_fill_offset_boost"] == 1.5

    def test_imbalance_section(self) -> None:
        result = _parse_trading_features({
            "imbalance": {
                "enabled": True,
                "depth": 5,
                "threshold": 0.3,
            },
        })
        assert result["imbalance_enabled"] is True
        assert result["imbalance_depth"] == 5
        assert result["imbalance_threshold"] == 0.3

    def test_side_offset(self) -> None:
        result = _parse_trading_features({
            "side_offset": {"buy": 1.2, "sell": 0.8},
        })
        assert result["spread_offset_ratio_buy"] == 1.2
        assert result["spread_offset_ratio_sell"] == 0.8

    def test_e3_sampling(self) -> None:
        result = _parse_trading_features({"e3": {"sampling_ratio": 0.1}})
        assert result["e3_sampling_ratio"] == 0.1


class TestParseSkipGateSection:
    """_parse_skip_gate_section: ML skip gate パラメータ."""

    def test_empty_returns_empty(self) -> None:
        assert _parse_skip_gate_section({}) == {}

    def test_skip_gate_basic(self) -> None:
        result = _parse_skip_gate_section({
            "skip_gate": {"enabled": True, "mode": "soft"},
        })
        assert result["skip_gate_enabled"] is True
        assert result["skip_gate_mode"] == "soft"


class TestParseStopgapSection:
    """_parse_stopgap_section: dynamic kill, inv skewing, etc."""

    def test_empty_returns_empty(self) -> None:
        assert _parse_stopgap_section({}) == {}

    def test_sell_dynamic_kill(self) -> None:
        """sell_dynamic_kill は 止血 セクション配下."""
        result = _parse_stopgap_section({
            "止血": {
                "sell_dynamic_kill": {
                    "enabled": True,
                    "window": 100,
                    "threshold_bps": -0.5,
                },
            },
        })
        assert result["sell_dynamic_kill_enabled"] is True
        assert result["sell_dynamic_kill_window"] == 100
        assert result["sell_dynamic_kill_threshold_bps"] == -0.5  # 364# TUNE-3

    def test_buy_dynamic_kill(self) -> None:
        """buy_dynamic_kill は 止血 セクション配下."""
        result = _parse_stopgap_section({
            "止血": {
                "buy_dynamic_kill": {
                    "enabled": True,
                    "threshold_bps": -1.5,
                    "regime_thresholds": {"ranging": -2.0},
                },
            },
        })
        assert result["buy_dynamic_kill_enabled"] is True
        assert result["buy_dynamic_kill_threshold_bps"] == -1.5
        assert result["buy_dynamic_kill_regime_thresholds"] == {"ranging": -2.0}

    def test_buy_dynamic_kill_inv_relaxation(self) -> None:
        """buy_dynamic_kill_inv_relaxation は 止血 セクション配下."""
        result = _parse_stopgap_section({
            "止血": {
                "buy_dynamic_kill_inv_relaxation": {"max_bps": 0.5, "scale": 0.5},
            },
        })
        assert result["buy_dynamic_kill_inv_relaxation_max_bps"] == 0.5

    def test_sell_dynamic_kill_inv_relaxation(self) -> None:
        """337# sell_dynamic_kill_inv_relaxation は 止血 セクション配下."""
        result = _parse_stopgap_section({
            "止血": {
                "sell_dynamic_kill_inv_relaxation": {
                    "enabled": True,
                    "scale": 0.4,
                    "max_bps": 0.3,
                },
            },
        })
        assert result["sell_dynamic_kill_inv_relaxation_enabled"] is True
        assert result["sell_dynamic_kill_inv_relaxation_scale"] == 0.4
        assert result["sell_dynamic_kill_inv_relaxation_max_bps"] == 0.3


class TestParseStaleVgSection:
    """_parse_stale_vg_section: stale order, VG."""

    def test_empty_returns_empty(self) -> None:
        assert _parse_stale_vg_section({}) == {}


class TestParseInfraSection:
    """_parse_infra_section: PnL fee, resilience, A/B test."""

    def test_empty_returns_empty(self) -> None:
        assert _parse_infra_section({}) == {}

    def test_pnl_fee_deduction(self) -> None:
        """止血.pnl_fee_deduction の解析."""
        result = _parse_infra_section({
            "止血": {
                "pnl_fee_deduction": {
                    "enabled": True,
                    "maker_fee_bps": 0.0,
                    "taker_fee_bps": 0.1,
                },
            },
        })
        assert result["pnl_fee_deduction_enabled"] is True
        assert result["maker_fee_bps"] == 0.0
        assert result["taker_fee_bps"] == 0.1


# ═══════════════════════════════════════════════════════════════════════
# Production YAML round-trip
# ═══════════════════════════════════════════════════════════════════════


class TestProductionYamlRoundTrip:
    """実 YAML ファイルのパース成功確認."""

    @pytest.fixture(scope="class")
    def yaml_cfg(self) -> dict:
        """configs/v460/fill_test.yaml をロード."""
        return dict(load_yaml_mapping(YAML_PATH))

    @pytest.fixture(scope="class")
    def direct_cfg(self, yaml_cfg: dict) -> FillTestConfig:
        return parse_fill_config_yaml(copy.deepcopy(yaml_cfg))

    @pytest.fixture(scope="class")
    def from_yaml_cfg(self, yaml_cfg: dict) -> FillTestConfig:
        return FillTestConfig.from_yaml(copy.deepcopy(yaml_cfg))

    def test_parse_succeeds(self, direct_cfg: FillTestConfig) -> None:
        """実 YAML の parse がエラーなく完了する."""
        assert isinstance(direct_cfg, FillTestConfig)

    def test_critical_fields_match_yaml(
        self,
        yaml_cfg: dict,
        direct_cfg: FillTestConfig,
    ) -> None:
        """336# で変更した critical fields が YAML 値と一致."""
        loss_control = yaml_cfg.get("止血", yaml_cfg.get("loss_control", {}))
        # buy_dynamic_kill — 336# T-1
        assert direct_cfg.buy_dynamic_kill_threshold_bps == loss_control["buy_dynamic_kill"]["threshold_bps"]
        # buy_dynamic_kill_inv_relaxation — 336# T-2
        assert direct_cfg.buy_dynamic_kill_inv_relaxation_max_bps == loss_control["buy_dynamic_kill_inv_relaxation"]["max_bps"]
        # 337# sell_dynamic_kill threshold 緩和
        assert direct_cfg.sell_dynamic_kill_threshold_bps == loss_control["sell_dynamic_kill"]["threshold_bps"]
        # 337# sell_dynamic_kill_inv_relaxation
        sell_inv = loss_control["sell_dynamic_kill_inv_relaxation"]
        assert direct_cfg.sell_dynamic_kill_inv_relaxation_enabled == sell_inv["enabled"]
        assert direct_cfg.sell_dynamic_kill_inv_relaxation_scale == sell_inv["scale"]
        assert direct_cfg.sell_dynamic_kill_inv_relaxation_max_bps == sell_inv["max_bps"]

    def test_from_yaml_matches_direct_parse(
        self,
        direct_cfg: FillTestConfig,
        from_yaml_cfg: FillTestConfig,
    ) -> None:
        """FillTestConfig.from_yaml() と直接呼び出しの結果が同一."""
        # dataclass の全フィールドが一致
        from dataclasses import fields

        for f in fields(direct_cfg):
            v1 = getattr(direct_cfg, f.name)
            v2 = getattr(from_yaml_cfg, f.name)
            assert v1 == v2, f"{f.name}: {v1!r} != {v2!r}"
