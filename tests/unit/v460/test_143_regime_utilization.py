"""143# R-1 レジーム別パラメータ適応テスト.

R-1a: offset regime adaptation (maker_price.py)
  - high_vol offset boost
  - ranging offset discount
  - config fields + YAML mapping
R-1b: lot regime adaptation (run_fill_test.py)
  - regime_lot_multipliers
  - 安全クランプ (min_lot, max_lot)
  - YAML mapping
レビュー修正テスト (140#/141#):
  - fill_quality quarantine bypass for cancel_reason
  - online_monitor pre-filter
  - skip_gate side hot-reload independence
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import yaml

# ======================================================================
# R-1a: Offset regime adaptation tests
# ======================================================================

class TestRegimeOffsetBoostConfig:
    """143# R-1a: config フィールドの存在とデフォルト値."""

    def test_high_vol_offset_boost_default(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert hasattr(cfg, "regime_high_vol_offset_boost")
        assert cfg.regime_high_vol_offset_boost == 1.2

    def test_ranging_offset_discount_default(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert hasattr(cfg, "regime_ranging_offset_discount")
        assert cfg.regime_ranging_offset_discount == 1.0

    def test_regime_lot_multipliers_default_empty(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert hasattr(cfg, "regime_lot_multipliers")
        assert cfg.regime_lot_multipliers == {}


class TestRegimeOffsetBoostSource:
    """143# R-1a: maker_price.py にレジーム別 offset ロジックが含まれることをソースで確認."""

    def test_high_vol_offset_boost_in_source(self) -> None:
        from scripts.v460.lib.maker_price import MakerPriceCalculator
        source = inspect.getsource(MakerPriceCalculator.compute)
        assert "regime_high_vol_offset_boost" in source
        assert "high_vol" in source

    def test_ranging_offset_discount_in_source(self) -> None:
        from scripts.v460.lib.maker_price import MakerPriceCalculator
        source = inspect.getsource(MakerPriceCalculator.compute)
        assert "regime_ranging_offset_discount" in source
        assert "ranging" in source


class TestRegimeOffsetBoostFunctional:
    """143# R-1a: MakerPriceCalculator.compute のレジーム別 offset 動作テスト."""

    def _make_calculator(
        self,
        regime_value: str | None = None,
        *,
        high_vol_boost: float = 1.2,
        ranging_discount: float = 1.0,
        trending_boost: float = 1.0,
        base_offset: float = 0.05,
    ) -> tuple:
        """テスト用の MakerPriceCalculator を生成."""
        from scripts.v460.lib.fill_config import FillTestConfig
        from scripts.v460.lib.maker_price import MakerPriceCalculator

        cfg = FillTestConfig(
            regime_high_vol_offset_boost=high_vol_boost,
            regime_ranging_offset_discount=ranging_discount,
            regime_trending_offset_boost=trending_boost,
            spread_offset_ratio=base_offset,
            max_offset_ratio=0.30,
            min_offset_ratio=0.01,
            # 不要な機能を無効化
            spread_adaptive_enabled=False,
            imbalance_enabled=False,
            volatility_guard_enabled=False,
            fast_fill_defense_enabled=False,
            sell_offset_floor=0.0,
            sell_max_spread_jpy=0.0,
        )

        # mock regime detector
        regime_det = None
        if regime_value is not None:
            regime_det = MagicMock()
            regime_det.current_regime = MagicMock()
            regime_det.current_regime.value = regime_value

        ffd = MagicMock()
        ffd.should_boost.return_value = False
        ffd.get_boost_multiplier.return_value = 1.0

        calc = MakerPriceCalculator(
            config=cfg,
            fast_fill_defense=ffd,
            regime_detector=regime_det,
            base_offset_ratio=base_offset,
        )
        return calc, cfg

    def _mock_adapter(self, best_bid: float = 15_000_000, best_ask: float = 15_001_000):
        """best_bid/best_ask を返す mock adapter."""
        adapter = MagicMock()
        ob = MagicMock()
        ob.bids = [(best_bid, 0.1)]
        ob.asks = [(best_ask, 0.1)]
        adapter.get_orderbook = AsyncMock(return_value=ob)
        return adapter

    def test_high_vol_boosts_offset(self) -> None:
        """high_vol 時に offset が boost される."""
        calc_hv, _ = self._make_calculator("high_vol", high_vol_boost=1.2)
        calc_base, _ = self._make_calculator("ranging", high_vol_boost=1.2, ranging_discount=1.0)

        adapter = self._mock_adapter()

        result_hv = asyncio.run(
            calc_hv.compute("buy", adapter, "btc_jpy")
        )
        result_base = asyncio.run(
            calc_base.compute("buy", adapter, "btc_jpy")
        )

        # high_vol の offset は ranging (base) より大きいはず
        assert result_hv.effective_offset_ratio > result_base.effective_offset_ratio

    def test_high_vol_boost_clamped_to_max(self) -> None:
        """high_vol boost が max_offset_ratio を超えない."""
        calc, cfg = self._make_calculator(
            "high_vol", high_vol_boost=10.0, base_offset=0.25,
        )
        adapter = self._mock_adapter()

        result = asyncio.run(
            calc.compute("buy", adapter, "btc_jpy")
        )
        assert result.effective_offset_ratio <= cfg.max_offset_ratio

    def test_ranging_discount_shrinks_offset(self) -> None:
        """ranging 時に discount < 1.0 で offset が縮小される."""
        calc_ranging, _ = self._make_calculator(
            "ranging", ranging_discount=0.8,
        )
        calc_base, _ = self._make_calculator(
            "unknown", ranging_discount=0.8,
        )

        adapter = self._mock_adapter()

        result_ranging = asyncio.run(
            calc_ranging.compute("buy", adapter, "btc_jpy")
        )
        result_base = asyncio.run(
            calc_base.compute("buy", adapter, "btc_jpy")
        )

        # ranging discount 時は offset が基準より小さい
        assert result_ranging.effective_offset_ratio < result_base.effective_offset_ratio

    def test_ranging_discount_clamped_to_min(self) -> None:
        """ranging discount が min_offset_ratio を下回らない."""
        calc, cfg = self._make_calculator(
            "ranging", ranging_discount=0.01, base_offset=0.02,
        )
        adapter = self._mock_adapter()

        result = asyncio.run(
            calc.compute("buy", adapter, "btc_jpy")
        )
        assert result.effective_offset_ratio >= cfg.min_offset_ratio

    def test_no_boost_when_regime_none(self) -> None:
        """regime_detector=None の場合、boost/discount は適用されない."""
        calc, _ = self._make_calculator(
            None, high_vol_boost=1.5, ranging_discount=0.5,
        )
        adapter = self._mock_adapter()

        result = asyncio.run(
            calc.compute("buy", adapter, "btc_jpy")
        )
        # base offset がそのまま
        assert abs(result.effective_offset_ratio - 0.05) < 0.01

    def test_disabled_when_boost_is_1(self) -> None:
        """high_vol_boost=1.0 のとき boost 無効."""
        calc_hv1, _ = self._make_calculator("high_vol", high_vol_boost=1.0)
        calc_none, _ = self._make_calculator("unknown", high_vol_boost=1.0)

        adapter = self._mock_adapter()

        r1 = asyncio.run(
            calc_hv1.compute("buy", adapter, "btc_jpy")
        )
        r2 = asyncio.run(
            calc_none.compute("buy", adapter, "btc_jpy")
        )
        # boost が 1.0 なので差がない (unknown_buy_offset_boost=1.0 のデフォルト)
        assert abs(r1.effective_offset_ratio - r2.effective_offset_ratio) < 1e-6


class TestRegimeOffsetYamlMapping:
    """143# R-1a: YAML → FillTestConfig のマッピング."""

    def test_yaml_high_vol_offset_boost(self, tmp_path: Path) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        yaml_data = {
            "regime": {
                "high_vol_offset_boost": 1.3,
                "ranging_offset_discount": 0.85,
            }
        }

        cfg = FillTestConfig.from_yaml(yaml_data)
        assert cfg.regime_high_vol_offset_boost == 1.3
        assert cfg.regime_ranging_offset_discount == 0.85

    def test_yaml_lot_multipliers(self, tmp_path: Path) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        yaml_data = {
            "regime": {
                "lot_multipliers": {
                    "high_vol": 0.7,
                    "trending": 1.2,
                    "ranging": 1.0,
                }
            }
        }

        cfg = FillTestConfig.from_yaml(yaml_data)
        assert cfg.regime_lot_multipliers == {
            "high_vol": 0.7,
            "trending": 1.2,
            "ranging": 1.0,
        }


# ======================================================================
# R-1b: Lot regime adaptation tests
# ======================================================================

class TestRegimeAdjustedLot:
    """143# R-1b: _regime_adjusted_lot() の動作テスト."""

    def _make_runner_mock(
        self,
        *,
        base_lot: float = 0.005,
        max_lot: float = 0.01,
        multipliers: dict[str, float] | None = None,
        regime_value: str | None = None,
    ) -> Any:
        """FillTestRunner のモック — _regime_adjusted_lot のみテスト."""
        from scripts.v460.lib.fill_config import FillTestConfig

        config = FillTestConfig(
            order_quantity=base_lot,
            max_lot=max_lot,
            regime_lot_multipliers=multipliers or {},
        )

        runner = MagicMock()
        runner.config = config
        runner._current_lot = base_lot

        if regime_value is not None:
            det = MagicMock()
            det.current_regime = MagicMock()
            det.current_regime.value = regime_value
            runner._regime_detector = det
        else:
            runner._regime_detector = None

        # _regime_adjusted_lot を実装からバインド
        from scripts.v460.run_fill_test import FillTestRunner
        import types
        runner._regime_adjusted_lot = types.MethodType(
            FillTestRunner._regime_adjusted_lot, runner,
        )
        return runner

    def test_no_multipliers_returns_base(self) -> None:
        """multipliers が空の場合、base_lot を返す."""
        runner = self._make_runner_mock(base_lot=0.005)
        assert runner._regime_adjusted_lot() == 0.005

    def test_no_detector_returns_base(self) -> None:
        """regime_detector=None の場合、base_lot を返す."""
        runner = self._make_runner_mock(
            base_lot=0.005,
            multipliers={"high_vol": 0.7},
            regime_value=None,
        )
        assert runner._regime_adjusted_lot() == 0.005

    def test_high_vol_shrinks_lot(self) -> None:
        """high_vol で lot が 0.7 倍に縮小."""
        runner = self._make_runner_mock(
            base_lot=0.005,
            multipliers={"high_vol": 0.7},
            regime_value="high_vol",
        )
        result = runner._regime_adjusted_lot()
        assert abs(result - 0.0035) < 1e-8

    def test_trending_expands_lot(self) -> None:
        """trending で lot が 1.2 倍に拡大."""
        runner = self._make_runner_mock(
            base_lot=0.005,
            multipliers={"trending": 1.2},
            regime_value="trending",
        )
        result = runner._regime_adjusted_lot()
        assert abs(result - 0.006) < 1e-8

    def test_clamped_to_min_lot(self) -> None:
        """倍率適用後も min_lot (0.001) 以上."""
        runner = self._make_runner_mock(
            base_lot=0.001,
            multipliers={"high_vol": 0.5},
            regime_value="high_vol",
        )
        result = runner._regime_adjusted_lot()
        assert result == 0.001  # 0.001 * 0.5 = 0.0005 → clamped to 0.001

    def test_clamped_to_max_lot(self) -> None:
        """倍率適用後も max_lot 以下."""
        runner = self._make_runner_mock(
            base_lot=0.008,
            max_lot=0.01,
            multipliers={"trending": 1.5},
            regime_value="trending",
        )
        result = runner._regime_adjusted_lot()
        assert result == 0.01  # 0.008 * 1.5 = 0.012 → clamped to 0.01

    def test_multiplier_1_returns_base(self) -> None:
        """multiplier=1.0 では base_lot をそのまま返す."""
        runner = self._make_runner_mock(
            base_lot=0.005,
            multipliers={"ranging": 1.0},
            regime_value="ranging",
        )
        assert runner._regime_adjusted_lot() == 0.005

    def test_unknown_regime_no_multiplier(self) -> None:
        """multipliers に unknown が無ければ base_lot."""
        runner = self._make_runner_mock(
            base_lot=0.005,
            multipliers={"high_vol": 0.7},
            regime_value="unknown",
        )
        assert runner._regime_adjusted_lot() == 0.005


# ======================================================================
# Review fix tests (140#/141#)
# ======================================================================

class TestQuarantineBypassCancelReason:
    """140# §7-#1: cancel_reason 付き FillRecord の quarantine bypass."""

    def test_cancel_reason_bypasses_price_check(self) -> None:
        """cancel_reason がある場合、order_price=0 でも quarantine されない."""
        from ztb.metrics.fill_quality import FillRecord, _quarantine_reason

        r = FillRecord(
            cycle_id="test_1",
            timestamp=time.time(),
            side="none",
            order_price=0.0,
            order_quantity=0.0,
            cancelled=True,
            cancel_reason="circuit_breaker_open",
            run_id="test_run",
            git_sha="abc1234",
        )
        reason = _quarantine_reason(r)
        assert reason is None, f"Should not quarantine cancel_reason record: {reason}"

    def test_no_cancel_reason_still_quarantined(self) -> None:
        """cancel_reason なしで order_price=0 は従来通り quarantine."""
        from ztb.metrics.fill_quality import FillRecord, _quarantine_reason

        r = FillRecord(
            cycle_id="test_2",
            timestamp=time.time(),
            side="buy",
            order_price=0.0,
            order_quantity=0.001,
            cancelled=True,
            run_id="test_run",
            git_sha="abc1234",
        )
        reason = _quarantine_reason(r)
        assert reason is not None


class TestOnlineMonitorPreFilter:
    """141# A.1-#3: online_monitor pre-filter テスト."""

    def test_pre_filter_in_source(self) -> None:
        """online_monitor.py に skip_gate_skipped/filled の pre-filter がある."""
        from ztb.ml import online_monitor
        source = inspect.getsource(online_monitor)
        assert "skip_gate_skipped" in source
        assert "filled" in source


class TestSkipGateSideReloadIndependence:
    """141# A.1-#1: side hot-reload が unified early-return と独立."""

    def test_side_reload_before_unified_check_in_source(self) -> None:
        """_check_and_reload_model で side 再読込が unified hash チェックより先."""
        from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator
        source = inspect.getsource(SkipGateEvaluator._check_and_reload_model)
        # side reload の呼び出しが unified hash チェック (_model_file_hash) より先
        side_idx = source.index("_check_and_reload_side_models")
        hash_idx = source.index("_model_file_hash")
        assert side_idx < hash_idx, (
            "_check_and_reload_side_models should execute before "
            "unified _model_file_hash check"
        )


class TestEvaluateGuardAllowsSideOnly:
    """141# A.1-#2: evaluate() が side-only models でも動作."""

    def test_evaluate_guard_checks_side_models(self) -> None:
        """evaluate() のガードが _gate_buy/_gate_sell の存在もチェック."""
        from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator
        source = inspect.getsource(SkipGateEvaluator.evaluate)
        assert "_gate_buy" in source
        assert "_gate_sell" in source
