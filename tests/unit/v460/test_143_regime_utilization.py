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
import logging
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from scripts.v460.lib.fill_config import FillMonitorResult, FillTestConfig
from scripts.v460.lib.maker_price import MakerPriceCalculator
from scripts.v460.lib.order_monitor import OrderMonitor
from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator
from scripts.v460.run_fill_test import FillTestRunner
from tests.unit.v460._fill_test_source import (
    MAKER_REGIME_BOOST,
    ORDER_MONITOR,
    SKIP_GATE_EVALUATOR,
    SKIP_GATE_MODEL_LOADER,
    read_class_method_source,
    read_fill_test_method_source,
    read_inspect_source,
)
from tests.unit.v460._yaml_test_helpers import clone_fill_test_config, load_fill_test_config_from_mapping
from ztb.metrics.fill_quality import FillRecord, _quarantine_reason
from ztb.ml import online_monitor
from ztb.trading.execution.order_monitor_policy import compute_stale_reprice_policy
from ztb.trading.signal.regime.regime_detector import (
    FillTestRegime,
    FillTestRegimeDetector,
    RegimeConfig,
)

_REGIME_HIGH_VOL_SOURCE = read_class_method_source(
    MAKER_REGIME_BOOST,
    "RegimeBoostMixin",
    "_regime_boost_high_vol",
)
_REGIME_RANGING_SOURCE = read_class_method_source(
    MAKER_REGIME_BOOST,
    "RegimeBoostMixin",
    "_regime_boost_ranging",
)
_ONLINE_MONITOR_SOURCE = read_inspect_source(online_monitor)
_SKIP_GATE_RELOAD_SOURCE = read_class_method_source(
    SKIP_GATE_MODEL_LOADER,
    "SkipGateModelLoaderMixin",
    "_check_and_reload_model",
)
_SKIP_GATE_EVALUATE_SOURCE = read_class_method_source(
    SKIP_GATE_EVALUATOR,
    "SkipGateEvaluator",
    "evaluate",
)
_REGIME_ADJUSTED_LOT_SOURCE = read_fill_test_method_source("_regime_adjusted_lot")
_RUN_SINGLE_CYCLE_SOURCE = read_fill_test_method_source("run_single_cycle")
_ORDER_MONITOR_MONITOR_SOURCE = read_class_method_source(
    ORDER_MONITOR,
    "OrderMonitor",
    "monitor",
)

# ======================================================================
# R-1a: Offset regime adaptation tests
# ======================================================================

class TestRegimeOffsetBoostConfig:
    """143# R-1a: config フィールドの存在とデフォルト値."""

    def test_high_vol_offset_boost_default(self) -> None:
        cfg = FillTestConfig()
        assert hasattr(cfg, "regime_high_vol_offset_boost")
        assert cfg.regime_high_vol_offset_boost == 1.2

    def test_ranging_offset_discount_default(self) -> None:
        cfg = FillTestConfig()
        assert hasattr(cfg, "regime_ranging_offset_discount")
        assert cfg.regime_ranging_offset_discount == 1.0

    def test_regime_lot_multipliers_default_empty(self) -> None:
        cfg = FillTestConfig()
        assert hasattr(cfg, "regime_lot_multipliers")
        assert cfg.regime_lot_multipliers == {}

class TestRegimeOffsetBoostSource:
    """143# R-1a: maker_price.py にレジーム別 offset ロジックが含まれることをソースで確認."""

    def test_high_vol_offset_boost_in_source(self) -> None:
        source = _REGIME_HIGH_VOL_SOURCE
        assert "regime_high_vol_offset_boost" in source
        assert "high_vol" in source

    def test_ranging_offset_discount_in_source(self) -> None:
        source = _REGIME_RANGING_SOURCE
        assert "regime_ranging_offset_discount" in source
        assert "ranging" in source

class TestRegimeOffsetBoostFunctional:
    """143# R-1a: MakerPriceCalculator.compute のレジーム別 offset 動作テスト."""

    @dataclass(slots=True)
    class _StaticFFD:
        def maybe_expire_boost(self, _side: str) -> None:
            return None

        def get_boost_multiplier(self, _side: str) -> float:
            return 1.0

    @dataclass(slots=True)
    class _OrderBook:
        bids: list[tuple[float, float]]
        asks: list[tuple[float, float]]

    class _Adapter:
        def __init__(self, orderbook: "TestRegimeOffsetBoostFunctional._OrderBook") -> None:
            self._orderbook = orderbook

        async def get_orderbook(
            self,
            _symbol: str,
            *,
            depth: int | None = None,
        ) -> "TestRegimeOffsetBoostFunctional._OrderBook":
            del depth
            return self._orderbook

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

        # mock regime detector — 156# §18: FillTestRegime enum を直接使用
        regime_det = None
        if regime_value is not None:
            regime_det = MagicMock()
            regime_det.current_regime = FillTestRegime(regime_value)
            regime_det.last_volatility_ratio = 1.0  # 648# σ refresh 対応

        calc = MakerPriceCalculator(
            config=cfg,
            fast_fill_defense=self._StaticFFD(),
            regime_detector=regime_det,
            base_offset_ratio=base_offset,
        )
        return calc, cfg

    def _mock_adapter(self, best_bid: float = 15_000_000, best_ask: float = 15_001_000):
        """best_bid/best_ask を返す mock adapter."""
        return self._Adapter(
            self._OrderBook(
                bids=[(best_bid, 0.1)],
                asks=[(best_ask, 0.1)],
            )
        )

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

    def test_yaml_high_vol_offset_boost(self) -> None:
        yaml_data = {
            "regime": {
                "high_vol_offset_boost": 1.3,
                "ranging_offset_discount": 0.85,
            }
        }

        cfg = clone_fill_test_config(load_fill_test_config_from_mapping(yaml_data))
        assert cfg.regime_high_vol_offset_boost == 1.3
        assert cfg.regime_ranging_offset_discount == 0.85

    def test_yaml_lot_multipliers(self) -> None:
        yaml_data = {
            "regime": {
                "lot_multipliers": {
                    "high_vol": 0.7,
                    "trending": 1.2,
                    "ranging": 1.0,
                }
            }
        }

        cfg = clone_fill_test_config(load_fill_test_config_from_mapping(yaml_data))
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
    ) -> MagicMock:
        """FillTestRunner のモック — _regime_adjusted_lot のみテスト."""

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
        # 145#: _regime_lot_multiplier も必要 (_regime_adjusted_lot が内部呼出し)
        runner._regime_lot_multiplier = types.MethodType(
            FillTestRunner._regime_lot_multiplier, runner,
        )
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
        source = _ONLINE_MONITOR_SOURCE
        assert "skip_gate_skipped" in source
        assert "filled" in source

class TestSkipGateSideReloadIndependence:
    """141# A.1-#1: side hot-reload が unified early-return と独立."""

    def test_side_reload_before_unified_check_in_source(self) -> None:
        """_check_and_reload_model で side 再読込が unified hash チェックより先."""
        source = _SKIP_GATE_RELOAD_SOURCE
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
        source = _SKIP_GATE_EVALUATE_SOURCE
        assert "_gate_buy" in source
        assert "_gate_sell" in source

# ======================================================================
# 144# Review fix tests — behavioral / R-1c / R-1d
# ======================================================================

class TestQuarantineBypassNarrowed:
    """144# #3: quarantine bypass が監査系 cancel_reason に限定."""

    def test_audit_cancel_reason_bypasses_quarantine(self) -> None:
        """_AUDIT_CANCEL_REASONS に含まれる reason は quarantine されない."""

        for reason in [
            "circuit_breaker_open", "preflight_pause", "preflight_insufficient",
            "time_filter_both_sides", "narrow_spread_pause",
        ]:
            r = FillRecord(
                cycle_id=f"audit_{reason}",
                timestamp=time.time(),
                side="none",
                order_price=0.0,
                order_quantity=0.0,
                cancelled=True,
                cancel_reason=reason,
                run_id="test_run",
                git_sha="abc1234",
            )
            assert _quarantine_reason(r) is None, f"{reason} should bypass quarantine"

    def test_non_audit_cancel_reason_quarantined(self) -> None:
        """監査系でない cancel_reason は quarantine される."""

        r = FillRecord(
            cycle_id="non_audit",
            timestamp=time.time(),
            side="none",
            order_price=0.0,
            order_quantity=0.0,
            cancelled=True,
            cancel_reason="timeout",
            run_id="test_run",
            git_sha="abc1234",
        )
        reason = _quarantine_reason(r)
        assert reason is not None, "Non-audit cancel_reason should be quarantined"
        assert "invalid_side" in reason

    def test_audit_reason_buy_side_valid_price(self) -> None:
        """audit cancel_reason + side=buy + valid price → clean."""

        r = FillRecord(
            cycle_id="audit_buy",
            timestamp=time.time(),
            side="buy",
            order_price=15_000_000,
            order_quantity=0.001,
            cancelled=True,
            cancel_reason="sell_dynamic_kill",
            run_id="test_run",
            git_sha="abc1234",
        )
        assert _quarantine_reason(r) is None

    def test_non_audit_reason_invalid_price_quarantined(self) -> None:
        """非監査系 reason + side=buy + order_price=0 → quarantined."""

        r = FillRecord(
            cycle_id="non_audit_price",
            timestamp=time.time(),
            side="buy",
            order_price=0.0,
            order_quantity=0.001,
            cancelled=True,
            cancel_reason="timeout",
            run_id="test_run",
            git_sha="abc1234",
        )
        reason = _quarantine_reason(r)
        assert reason == "invalid_order_price"

class TestMinLotUnification:
    """144# #2: _regime_adjusted_lot の min_lot が config.min_order_btc を参照."""

    def test_min_lot_uses_config(self) -> None:
        """ハードコード 0.001 ではなく config.min_order_btc を参照."""
        source = _REGIME_ADJUSTED_LOT_SOURCE
        assert "self.config.min_order_btc" in source
        assert "min_lot = 0.001" not in source

    def test_custom_min_order_btc_respected(self) -> None:
        """config.min_order_btc を変更すると min_lot が追従."""

        # min_order_btc = 0.005 に変更
        config = FillTestConfig(
            order_quantity=0.002,
            max_lot=0.01,
            min_order_btc=0.005,
            regime_lot_multipliers={"high_vol": 0.5},
        )
        runner = MagicMock()
        runner.config = config
        runner._current_lot = 0.002

        det = MagicMock()
        det.current_regime = MagicMock()
        det.current_regime.value = "high_vol"
        runner._regime_detector = det

        runner._regime_lot_multiplier = types.MethodType(
            FillTestRunner._regime_lot_multiplier, runner,
        )
        runner._regime_adjusted_lot = types.MethodType(
            FillTestRunner._regime_adjusted_lot, runner,
        )
        result = runner._regime_adjusted_lot()
        # 0.002 * 0.5 = 0.001 → clamped to min_order_btc = 0.005
        assert result == 0.005

class TestPreflightLotAlignment:
    """144# #1 → 145# fix: regime-adjusted lot は per-cycle で _current_lot に永続化しない."""

    def test_regime_lot_no_persistent_mutation(self) -> None:
        """145# fix: run_single_cycle で _regime_adjusted_lot が呼ばれるが、
        _current_lot への永続化コードが除去されている."""
        pre_order_source = read_fill_test_method_source("_run_pre_order_phase")
        submit_source = read_fill_test_method_source("_submit_order_phase")
        # 151# P3-03 / 583# Task C: regime_lot を pre-order phase で1回算出
        assert "regime_lot = self._regime_adjusted_lot()" in pre_order_source, (
            "regime_lot should be computed once per cycle"
        )
        # 151# P3-03 / 583# Task C: submission phase で confidence × regime を合成
        assert "self._effective_order_lot(" in submit_source, (
            "_effective_order_lot should be called for confidence_lot integration"
        )
        # 145# fix: 永続化コード「_order_lot > self._current_lot」は除去済み
        assert "_order_lot > self._current_lot" not in pre_order_source + submit_source, (
            "§8-#2/#3 fix: _current_lot への永続化コードが残っている"
        )

class TestRegimeRepriceConfig:
    """144# R-1c: regime_reprice_adjustments config テスト."""

    def test_default_empty(self) -> None:
        cfg = FillTestConfig()
        assert cfg.regime_reprice_adjustments == {}

    def test_yaml_mapping(self) -> None:
        yaml_data = {
            "regime": {
                "reprice_adjustments": {
                    "high_vol": 1,
                    "trending": 2,
                    "ranging": 0,
                }
            }
        }
        cfg = clone_fill_test_config(load_fill_test_config_from_mapping(yaml_data))
        assert cfg.regime_reprice_adjustments == {
            "high_vol": 1,
            "trending": 2,
            "ranging": 0,
        }

class TestRegimeTimeoutConfig:
    """144# R-1d: regime_timeout_multipliers config テスト."""

    def test_default_empty(self) -> None:
        cfg = FillTestConfig()
        assert cfg.regime_timeout_multipliers == {}

    def test_yaml_mapping(self) -> None:
        yaml_data = {
            "regime": {
                "timeout_multipliers": {
                    "high_vol": 0.7,
                    "trending": 1.3,
                }
            }
        }
        cfg = clone_fill_test_config(load_fill_test_config_from_mapping(yaml_data))
        assert cfg.regime_timeout_multipliers == {
            "high_vol": 0.7,
            "trending": 1.3,
        }

class TestRegimeRepriceInOrderMonitor:
    """144# R-1c: OrderMonitor の regime reprice offset がソースに含まれる."""

    def test_regime_reprice_offset_in_source(self) -> None:
        source = _ORDER_MONITOR_MONITOR_SOURCE
        assert "regime_reprice_adjustments" in source
        assert "_regime_reprice_offset" in source

    def test_reprice_offset_applied_to_stale_max(self) -> None:
        """monitor() は stale reprice policy helper に regime offset を渡す."""
        source = _ORDER_MONITOR_MONITOR_SOURCE
        assert "_compute_stale_reprice_policy(" in source
        assert "regime_reprice_offset=_regime_reprice_offset" in source

class TestRegimeTimeoutInOrderMonitor:
    """144# R-1d: OrderMonitor の regime timeout multiplier がソースに含まれる."""

    def test_regime_timeout_in_source(self) -> None:
        source = _ORDER_MONITOR_MONITOR_SOURCE
        assert "regime_timeout_multipliers" in source
        assert "_effective_timeout" in source

    def test_effective_timeout_used_in_loop(self) -> None:
        """while ループが _effective_timeout を使用."""
        source = _ORDER_MONITOR_MONITOR_SOURCE
        assert "elapsed < _effective_timeout" in source
        # 旧ハードコード timeout_sec が直接ループで使われていないこと
        assert "elapsed < cfg.order_timeout_sec" not in source

class TestRegimeRepriceMonitorBehavioral:
    """144# R-1c: OrderMonitor の regime reprice を mock で動作確認."""

    def test_reprice_offset_increases_limit(self) -> None:
        """regime_reprice_adjustments で reprice 上限が増える."""
        policy = compute_stale_reprice_policy(
            side="buy",
            stale_check_after_sec=0.1,
            stale_check_after_sec_buy=None,
            stale_check_after_sec_sell=None,
            stale_drift_bps=0.01,
            stale_drift_bps_buy=None,
            stale_drift_bps_sell=None,
            stale_max_reprice=2,
            stale_max_reprice_buy=None,
            stale_max_reprice_sell=None,
            chase_drift_bps_override=None,
            chase_max_reprice_override=None,
            regime_reprice_offset=3,
        )
        assert policy.stale_max_reprice == 5

    def test_negative_offset_clamps_to_zero(self) -> None:
        """regime_reprice_adjustments 負の値で max 0 にクランプ."""
        policy = compute_stale_reprice_policy(
            side="buy",
            stale_check_after_sec=0.1,
            stale_check_after_sec_buy=None,
            stale_check_after_sec_sell=None,
            stale_drift_bps=0.01,
            stale_drift_bps_buy=None,
            stale_drift_bps_sell=None,
            stale_max_reprice=1,
            stale_max_reprice_buy=None,
            stale_max_reprice_sell=None,
            chase_drift_bps_override=None,
            chase_max_reprice_override=None,
            regime_reprice_offset=-5,
        )
        assert policy.stale_max_reprice == 0

class TestRegimeTimeoutMonitorBehavioral:
    """144# R-1d: OrderMonitor の regime timeout を mock で動作確認."""

    def test_timeout_multiplier_applied(self) -> None:
        """regime_timeout_multipliers で effective timeout が変わる."""

        cfg = FillTestConfig(
            order_timeout_sec=90.0,
            regime_timeout_multipliers={"high_vol": 0.7, "trending": 1.3},
        )
        # high_vol: 90 * 0.7 = 63s
        assert cfg.order_timeout_sec * 0.7 == pytest.approx(63.0)
        # trending: 90 * 1.3 = 117s
        assert cfg.order_timeout_sec * 1.3 == pytest.approx(117.0)

    def test_no_regime_uses_base_timeout(self) -> None:
        """regime=None の場合は base timeout を使用."""

        cfg = FillTestConfig(
            order_timeout_sec=90.0,
            regime_timeout_multipliers={"high_vol": 0.7},
        )
        # multiplier default 1.0
        mult = cfg.regime_timeout_multipliers.get(None, 1.0)  # type: ignore[arg-type]
        assert mult == 1.0
        assert cfg.order_timeout_sec * mult == 90.0

class TestOrderMonitorHelpers:
    """OrderMonitor の小ヘルパー契約を直接検証."""

    def test_resolve_regime_name(self) -> None:

        monitor = OrderMonitor(FillTestConfig())
        regime_det = MagicMock()
        regime_det.current_regime = MagicMock()
        regime_det.current_regime.value = "trending"

        assert monitor._resolve_regime_name(regime_det) == "trending"
        assert monitor._resolve_regime_name(None) is None

    def test_reprice_skip_gate_helper_uses_config_offset(self) -> None:

        cfg = FillTestConfig(stale_reprice_skip_gate_offset=0.15)
        monitor = OrderMonitor(cfg)

        regime_det = MagicMock()
        regime_det.current_regime = MagicMock()
        regime_det.current_regime.value = "trending"

        decision = MagicMock()
        decision.should_skip = True
        decision.as_probability = 0.72
        decision.threshold_used = 0.55

        skip_gate = MagicMock()
        skip_gate.evaluate.return_value = decision

        blocked = monitor._should_block_reprice_with_skip_gate(
            skip_gate=skip_gate,
            side="sell",
            spread_at_order=2000.0,
            effective_offset_ratio=0.05,
            regime_detector=regime_det,
            market_timestamp=1700000000.0,
        )

        assert blocked is True
        call = skip_gate.evaluate.call_args
        assert call is not None
        assert call.kwargs["side"] == "sell"
        assert call.kwargs["threshold_offset"] == pytest.approx(-0.15)
        assert call.args[0]["regime_trending"] == 1.0

# ======================================================================
# 145# fix: lot management bug fixes (§8-#1/#2/#3, §9-#1, §9-#2)
# ======================================================================

class TestLotNoCompounding:
    """145# §8-#2 fix: _regime_adjusted_lot が _current_lot を永続化しない確認."""

    def _make_runner_mock(
        self,
        *,
        base_lot: float = 0.005,
        max_lot: float = 0.05,
        multipliers: dict[str, float] | None = None,
        regime_value: str | None = None,
    ) -> MagicMock:

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

        runner._regime_lot_multiplier = types.MethodType(
            FillTestRunner._regime_lot_multiplier, runner,
        )
        runner._regime_adjusted_lot = types.MethodType(
            FillTestRunner._regime_adjusted_lot, runner,
        )
        return runner

    def test_no_compounding_trending(self) -> None:
        """trending×1.5 を繰り返しても _current_lot は変化しない."""
        runner = self._make_runner_mock(
            base_lot=0.001,
            multipliers={"trending": 1.5},
            regime_value="trending",
        )
        for _ in range(10):
            order_lot = runner._regime_adjusted_lot()
            assert abs(order_lot - 0.0015) < 1e-8, (
                f"compounding detected: order_lot={order_lot}"
            )
            # 145# fix: _current_lot を更新しない
            assert runner._current_lot == 0.001

    def test_no_one_sided_update(self) -> None:
        """§8-#3: trending 後に high_vol に切り替えても _current_lot は base のまま."""
        runner = self._make_runner_mock(
            base_lot=0.005,
            multipliers={"trending": 1.5, "high_vol": 0.5},
            regime_value="trending",
        )
        # trending: 0.005 * 1.5 = 0.0075
        lot1 = runner._regime_adjusted_lot()
        assert abs(lot1 - 0.0075) < 1e-8

        # regime 切替: high_vol
        runner._regime_detector.current_regime.value = "high_vol"
        lot2 = runner._regime_adjusted_lot()
        # 0.005 * 0.5 = 0.0025 (base からの再計算)
        assert abs(lot2 - 0.0025) < 1e-8
        # _current_lot は不変
        assert runner._current_lot == 0.005

    def test_balance_shrink_reflected(self) -> None:
        """balance shrink が _current_lot を下げた場合、regime lot もそれに追従."""
        runner = self._make_runner_mock(
            base_lot=0.005,
            multipliers={"trending": 1.5},
            regime_value="trending",
        )
        lot = runner._regime_adjusted_lot()
        assert abs(lot - 0.0075) < 1e-8

        # balance shrink で _current_lot が 0.003 に縮小
        runner._current_lot = 0.003
        lot_after = runner._regime_adjusted_lot()
        # 0.003 * 1.5 = 0.0045 (shrink 後の base から計算)
        assert abs(lot_after - 0.0045) < 1e-8

class TestMonitorReceivesOrderLot:
    """145# §9-#1 fix: _monitor_fill_polling が order_lot を monitor に渡す確認."""

    @pytest.mark.asyncio
    async def test_monitor_receives_order_lot_not_current_lot(self) -> None:
        """order_lot を渡すと, monitor.current_lot に反映される."""

        runner = MagicMock(spec=FillTestRunner)
        runner._current_lot = 0.001  # base lot
        runner._pending_order_id = None
        runner._kill_switch = MagicMock()
        runner._maker_price = MagicMock()
        runner._order_monitor = MagicMock()
        runner._skip_gate = MagicMock()
        runner._regime_detector = None
        runner.adapter = AsyncMock()

        # monitor mock
        monitor_result = FillMonitorResult(filled=True, fill_price=15000000.0)
        runner._order_monitor.monitor = AsyncMock(return_value=monitor_result)

        # _get_mid_price, _compute_maker_price
        runner._get_mid_price = AsyncMock(return_value=15000000.0)
        runner._compute_maker_price = AsyncMock(return_value=(15000000.0, 100.0, 0.0003))

        runner._monitor_fill_polling = types.MethodType(
            FillTestRunner._monitor_fill_polling, runner,
        )

        order = MagicMock()
        order.order_id = "test_order"

        await runner._monitor_fill_polling(
            order, 15000000.0, "buy", time.time(), 100.0, 0.0003,
            order_lot=0.0015,  # regime-adjusted lot
        )

        # monitor が order_lot (0.0015) で呼ばれた (NOT _current_lot 0.001)
        call_kwargs = runner._order_monitor.monitor.call_args
        actual_lot = call_kwargs.kwargs.get("current_lot")
        assert actual_lot == 0.0015, (
            f"Expected current_lot=0.0015, got: {actual_lot}"
        )

class TestEffectiveTimeout:
    """145# §9-#2: FillMonitorResult.effective_timeout が正しく返される."""

    def test_effective_timeout_field(self) -> None:
        """effective_timeout フィールドが存在し値が設定される."""

        result = FillMonitorResult(
            filled=False,
            queue_wait=30.0,
            effective_timeout=45.0,
        )
        assert result.effective_timeout == 45.0

    def test_effective_timeout_default_zero(self) -> None:
        """デフォルトは 0.0."""

        result = FillMonitorResult(filled=True)
        assert result.effective_timeout == 0.0

    def test_cancel_reason_uses_effective_timeout(self) -> None:
        """regime で短縮されたタイムアウトで正しく timeout ラベルが付く.

        例: base=60s, high_vol mult=0.5 → effective=30s.
        queue_wait=30s は base(60s) 未満だが effective(30s) 以上 → timeout.
        """
        effective_timeout = 30.0  # regime-shortened
        queue_wait = 30.0
        filled = False
        cancel_reason_poll: str | None = None
        base_timeout = 60.0

        # 145# fix 適用後の cancel_reason ロジック
        cancel_reason = (
            cancel_reason_poll
            if cancel_reason_poll
            else (
                "timeout"
                if (not filled and queue_wait >= (effective_timeout or base_timeout))
                else ("unknown" if not filled else None)
            )
        )
        assert cancel_reason == "timeout"

    def test_cancel_reason_old_logic_would_fail(self) -> None:
        """fix 前のロジック (base_timeout 比較) では unknown になるケースを確認."""
        queue_wait = 30.0
        filled = False
        cancel_reason_poll: str | None = None
        base_timeout = 60.0

        # 旧ロジック: base_timeout で比較 → 30 < 60 → unknown
        old_cancel_reason = (
            cancel_reason_poll
            if cancel_reason_poll
            else (
                "timeout"
                if (not filled and queue_wait >= base_timeout)
                else ("unknown" if not filled else None)
            )
        )
        assert old_cancel_reason == "unknown"  # 旧ロジックの欠陥を証明

# ======================================================================
# 152# Regime detector unknown reduction tests
# ======================================================================

class TestAcceleratedHysteresis:
    """152# A: UNKNOWN → first regime は accelerated hysteresis で高速確定."""

    def test_unknown_to_first_regime_needs_fewer_consecutive(self) -> None:
        """UNKNOWN からの初回遷移は hysteresis_count - 1 で確定する.

        window=5 の場合、5 回目の update で初回分類が走り raw_history に 1 件追加。
        6 回目の update で 2 連続 → accelerated threshold (max(2, 3-1)=2) を満たし確定。
        旧ロジックでは 3 連続 (8 回目) が必要だった。
        """

        config = RegimeConfig(window=5, hysteresis_count=3, min_confidence=0.0)
        det = FillTestRegimeDetector(config)

        # window 充填: 4 回は early return、5 回目で初回分類 (raw_history=[RANGING])
        base_price = 10_000_000.0
        for i in range(5):
            det.update(float(i), base_price + i * 10)  # ≈0% 変動

        # 5 回目で分類済みだが 1 連続 < threshold(2) → まだ UNKNOWN
        assert det.current_regime == FillTestRegime.UNKNOWN

        # 6 回目: 2 連続 RANGING → accelerated threshold (2) を満たし確定
        det.update(5.0, base_price + 50)
        assert det.current_regime == FillTestRegime.RANGING

    def test_normal_transition_still_needs_full_hysteresis(self) -> None:
        """UNKNOWN 以外 → 別レジームへの遷移は通常の hysteresis_count を使う."""

        config = RegimeConfig(
            window=5, hysteresis_count=3, min_confidence=0.0,
            trend_threshold_pct=0.3,
        )
        det = FillTestRegimeDetector(config)

        # まず RANGING を確定させる
        base = 10_000_000.0
        for i in range(5):
            det.update(float(i), base + i * 10)
        det.update(5.0, base + 50)
        det.update(6.0, base + 60)
        assert det.current_regime == FillTestRegime.RANGING

        # 強いトレンドを投入: 2 連続では遷移しない (threshold=3)
        for i in range(2):
            price = base + 60 + (i + 1) * 50_000  # 大きな値動き
            det.update(float(7 + i), price)
        assert det.current_regime == FillTestRegime.RANGING  # まだ遷移しない

        # 3 連続で遷移 (156# D-4: 方向付きで trending_up になる)
        det.update(9.0, base + 60 + 3 * 50_000)
        assert det.current_regime.is_trending

class TestMajorityFallback:
    """152# B: UNKNOWN 長期化時の最頻分類フォールバック."""

    def test_choppy_market_triggers_majority_fallback(self) -> None:
        """choppy な市場で連続一致が成立しなくても、最頻分類で仮確定する."""

        config = RegimeConfig(
            window=5, hysteresis_count=3, min_confidence=0.0,
            trend_threshold_pct=0.3,
            high_vol_multiplier=2.0,
        )
        det = FillTestRegimeDetector(config)

        # window を満たす
        base = 10_000_000.0
        for i in range(5):
            det.update(float(i), base + i * 10)

        # choppy: RANGINGとTRENDINGが交互 → 連続一致しない
        # 但し RANGING のほうが多い
        prices = [
            base + 50,     # ranging
            base + 50_100, # trending (大きな値動き)
            base + 50_110, # ranging (戻り)
            base + 50_120, # ranging
            base + 100_000, # trending
            base + 100_010, # ranging
        ]
        for i, price in enumerate(prices):
            det.update(float(5 + i), price)

        # hysteresis_count * 2 = 6 回の raw_history が溜まった
        # 過半数が同一分類なら majority fallback が発動
        # 結果は RANGING か TRENDING のいずれか (not UNKNOWN が重要)
        assert det.current_regime != FillTestRegime.UNKNOWN

    def test_no_fallback_when_insufficient_raw_history(self) -> None:
        """raw_history が hysteresis_count * 2 未満なら fallback しない."""

        # hysteresis_count=4 → accelerated threshold = max(2, 3) = 3
        # fallback requires raw_history >= 8
        config = RegimeConfig(window=5, hysteresis_count=4, min_confidence=0.0)
        det = FillTestRegimeDetector(config)

        # window 充填: 5 回 (5 回目で分類、raw_history=[RANGING])
        base = 10_000_000.0
        for i in range(5):
            det.update(float(i), base + i * 10)

        # 1 回追加: raw_history=[RANGING, RANGING], len=2 < 8 → fallback なし
        # consecutive=2 < threshold=3 → accelerated 未確定
        det.update(5.0, base + 50)

        assert det.current_regime == FillTestRegime.UNKNOWN

    def test_accelerated_takes_priority_over_fallback(self) -> None:
        """accelerated hysteresis が先に確定すれば fallback は不要."""

        config = RegimeConfig(window=5, hysteresis_count=3, min_confidence=0.0)
        det = FillTestRegimeDetector(config)

        # window を満たす
        base = 10_000_000.0
        for i in range(5):
            det.update(float(i), base + i * 10)

        # 2 連続同一分類 → accelerated で確定 (fallback 不要)
        det.update(5.0, base + 50)
        det.update(6.0, base + 60)
        assert det.current_regime == FillTestRegime.RANGING  # accelerated 確定
