"""145# 構造修正テスト (144# レビュー §8/§9 残項目).

対象:
  - §8-#1: preflight-lot alignment (regime_mult 付き残高チェック)
  - §8-#6: config value range validation (regime fields)
  - §9-#3: OB format normalization (ob_utils)
  - §9-#4: SkipGate lot consistency (regime-adjusted lot 伝搬)
  - §9-#5/7: _make_skip_record / _new_cycle_id DRY ヘルパ
  - §9-#6: cancel_reasons 定数モジュール
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest


# ======================================================================
# §9-#3: ob_utils テスト
# ======================================================================


class TestObUtilsExtractPrice:
    """ob_utils.extract_price — tuple / object 両対応."""

    def test_tuple_format(self) -> None:
        from scripts.v460.lib.ob_utils import extract_price
        assert extract_price((1234.5, 0.1)) == 1234.5

    def test_list_format(self) -> None:
        from scripts.v460.lib.ob_utils import extract_price
        assert extract_price([9800000.0, 0.005]) == 9800000.0

    def test_object_format(self) -> None:
        from scripts.v460.lib.ob_utils import extract_price

        @dataclass
        class Level:
            price: float
            quantity: float

        assert extract_price(Level(price=100.0, quantity=1.0)) == 100.0


class TestObUtilsExtractSize:
    """ob_utils.extract_size — tuple / object 両対応."""

    def test_tuple_format(self) -> None:
        from scripts.v460.lib.ob_utils import extract_size
        assert extract_size((1000.0, 0.5)) == 0.5

    def test_list_format(self) -> None:
        from scripts.v460.lib.ob_utils import extract_size
        assert extract_size([1000.0, 2.0]) == 2.0

    def test_object_quantity(self) -> None:
        from scripts.v460.lib.ob_utils import extract_size

        @dataclass
        class Level:
            price: float
            quantity: float

        assert extract_size(Level(price=100.0, quantity=3.0)) == 3.0

    def test_object_size_fallback(self) -> None:
        from scripts.v460.lib.ob_utils import extract_size

        class Level:
            def __init__(self) -> None:
                self.price = 100.0
                self.size = 4.0

        assert extract_size(Level()) == 4.0


class TestObUtilsBestBidAsk:
    """ob_utils.best_bid_ask — OrderBookSnapshot からの安全抽出."""

    def test_normal_tuple_ob(self) -> None:
        from scripts.v460.lib.ob_utils import best_bid_ask

        ob = MagicMock()
        ob.bids = [(9800000.0, 0.1), (9799000.0, 0.2)]
        ob.asks = [(9801000.0, 0.3)]
        bid, ask = best_bid_ask(ob)
        assert bid == 9800000.0
        assert ask == 9801000.0

    def test_empty_ob(self) -> None:
        from scripts.v460.lib.ob_utils import best_bid_ask

        ob = MagicMock()
        ob.bids = []
        ob.asks = []
        bid, ask = best_bid_ask(ob)
        assert bid is None
        assert ask is None

    def test_none_ob(self) -> None:
        from scripts.v460.lib.ob_utils import best_bid_ask
        bid, ask = best_bid_ask(None)
        assert bid is None
        assert ask is None


class TestObUtilsDepthVolume:
    """ob_utils.depth_volume — 深さ指定の合計出来高."""

    def test_full_depth(self) -> None:
        from scripts.v460.lib.ob_utils import depth_volume

        levels = [(100.0, 1.0), (99.0, 2.0), (98.0, 3.0)]
        assert depth_volume(levels, depth=3) == pytest.approx(6.0)

    def test_partial_depth(self) -> None:
        from scripts.v460.lib.ob_utils import depth_volume

        levels = [(100.0, 1.0), (99.0, 2.0), (98.0, 3.0)]
        assert depth_volume(levels, depth=2) == pytest.approx(3.0)

    def test_default_depth_5(self) -> None:
        from scripts.v460.lib.ob_utils import depth_volume

        levels = [(i, 1.0) for i in range(10)]
        assert depth_volume(levels) == pytest.approx(5.0)


# ======================================================================
# §9-#6: cancel_reasons 定数テスト
# ======================================================================


class TestCancelReasons:
    """cancel_reasons モジュールの定数と frozenset の整合性."""

    def test_audit_reasons_frozenset_matches_constants(self) -> None:
        from scripts.v460.lib import cancel_reasons as CR

        expected = {
            CR.CIRCUIT_BREAKER_OPEN,
            CR.PREFLIGHT_PAUSE,
            CR.PREFLIGHT_INSUFFICIENT,
            CR.TIME_FILTER_BOTH_SIDES,
            CR.TIME_FILTER_086_DEADLOCK,
            CR.NARROW_SPREAD_PAUSE,
            CR.BALANCE_FORCED_SKIP,
            CR.UNKNOWN_REGIME_BUY_SKIP,
            CR.UNKNOWN_REGIME_SELL_SKIP,
            CR.SELL_DYNAMIC_KILL,
            CR.BUY_DYNAMIC_KILL,       # 157# §19
            CR.TRENDING_SELL_SKIP,
            CR.DAILY_DRAWDOWN_HALT,     # 168# §4.1 #3
            CR.RANGING_LOW_VOL_SKIP,   # 169# B1'
            CR.SKIP_GATE,                      # 174#
            CR.SKIP_GATE_RULE_VELOCITY_SELL,   # 174#
            CR.SKIP_GATE_RULE_VELOCITY_BUY,    # 174#
            CR.POSTONLY_CROSSING_SKIP,          # 200# B/I
        }
        assert CR.AUDIT_CANCEL_REASONS == expected

    def test_audit_is_frozenset(self) -> None:
        from scripts.v460.lib import cancel_reasons as CR
        assert isinstance(CR.AUDIT_CANCEL_REASONS, frozenset)

    def test_exec_constants_exist(self) -> None:
        from scripts.v460.lib import cancel_reasons as CR
        for name in ["POST_ONLY_REJECT", "INSUFFICIENT_FUNDS", "MINIMUM_SIZE",
                      "API_ERROR", "TIMEOUT", "UNKNOWN"]:
            assert hasattr(CR, name)

    def test_guard_constants_exist(self) -> None:
        from scripts.v460.lib import cancel_reasons as CR
        assert hasattr(CR, "STALE_SKIP_GATE_BLOCKED")
        assert hasattr(CR, "STALE_REPRICE_FAILED")

    def test_orderbook_constants_exist(self) -> None:
        from scripts.v460.lib import cancel_reasons as CR
        for name in ["ORDERBOOK_ERROR", "ORDERBOOK_TIMEOUT",
                      "ORDERBOOK_RATE_LIMIT", "ORDERBOOK_EMPTY",
                      "SELL_GUARD_REJECT"]:
            assert hasattr(CR, name)

    def test_fill_quality_uses_shared_constants(self) -> None:
        """fill_quality.py が cancel_reasons.AUDIT_CANCEL_REASONS を使っていることを確認."""
        import inspect
        from ztb.metrics import fill_quality

        source = inspect.getsource(fill_quality)
        assert "AUDIT_CANCEL_REASONS" in source
        assert "cancel_reasons" in source


# ======================================================================
# §8-#6: config value range validation テスト
# ======================================================================


class TestRegimeConfigValidation:
    """FillTestConfig.__post_init__ で regime 関連フィールドの値域を検証."""

    def test_default_config_passes_validation(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg is not None

    def test_regime_timeout_multiplier_zero_raises(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        with pytest.raises(ValueError, match="regime_timeout_multipliers"):
            FillTestConfig(regime_timeout_multipliers={"trending": 0.0})

    def test_regime_timeout_multiplier_negative_raises(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        with pytest.raises(ValueError, match="regime_timeout_multipliers"):
            FillTestConfig(regime_timeout_multipliers={"trending": -1.0})

    def test_regime_lot_multiplier_zero_raises(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        with pytest.raises(ValueError, match="regime_lot_multipliers"):
            FillTestConfig(regime_lot_multipliers={"trending": 0.0})

    def test_regime_lot_multiplier_negative_raises(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        with pytest.raises(ValueError, match="regime_lot_multipliers"):
            FillTestConfig(regime_lot_multipliers={"trending": -0.5})

    def test_regime_reprice_adjustment_too_large_raises(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        with pytest.raises(ValueError, match="regime_reprice_adjustments"):
            FillTestConfig(regime_reprice_adjustments={"trending": 15.0})

    def test_regime_reprice_adjustment_too_negative_raises(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        with pytest.raises(ValueError, match="regime_reprice_adjustments"):
            FillTestConfig(regime_reprice_adjustments={"trending": -11.0})

    def test_valid_regime_values_pass(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig(
            regime_timeout_multipliers={"trending": 1.5},
            regime_lot_multipliers={"ranging": 0.8},
            regime_reprice_adjustments={"high_vol": -5.0},
        )
        assert cfg.regime_timeout_multipliers["trending"] == 1.5
        assert cfg.regime_lot_multipliers["ranging"] == 0.8
        assert cfg.regime_reprice_adjustments["high_vol"] == -5.0


# ======================================================================
# §8-#1: preflight-lot alignment テスト (BalanceChecker + regime_mult)
# ======================================================================


class TestPreflightRegimeMult:
    """BalanceChecker.check() が regime_mult を正しく反映するか."""

    def _make_checker(self, *, lot: float = 0.005) -> object:
        from scripts.v460.lib.fill_config import FillTestConfig
        from scripts.v460.lib.balance_checker import BalanceChecker

        config = FillTestConfig(
            order_quantity=lot,
            min_order_btc=0.001,
            balance_margin_ratio=1.01,
        )
        return BalanceChecker(config)

    def _mock_adapter_sell(self, btc_free: float) -> AsyncMock:
        adapter = AsyncMock()
        balance = MagicMock()
        balance.free = btc_free
        adapter.get_balance.return_value = [balance]
        return adapter

    def _mock_adapter_buy(self, jpy_free: float, price: float = 10_000_000) -> AsyncMock:
        adapter = AsyncMock()
        adapter.get_current_price.return_value = price
        balance = MagicMock()
        balance.free = jpy_free
        adapter.get_balance.return_value = [balance]
        return adapter

    @pytest.mark.asyncio
    async def test_sell_regime_mult_1_passes(self) -> None:
        """regime_mult=1.0: btc_free >= lot → pass."""
        checker = self._make_checker(lot=0.005)
        adapter = self._mock_adapter_sell(0.006)
        result = await checker.check("sell", adapter, "BTC_JPY", regime_mult=1.0)
        assert result is False  # pass

    @pytest.mark.asyncio
    async def test_sell_regime_mult_makes_insufficient(self) -> None:
        """regime_mult=1.5: btc_free(0.006) < lot(0.005)*1.5(0.0075) → insufficient."""
        checker = self._make_checker(lot=0.005)
        adapter = self._mock_adapter_sell(0.006)
        # Without regime_mult this would pass (0.006 >= 0.005)
        # With regime_mult=1.5: effective = 0.0075, 0.006 < 0.0075 → shrink
        result = await checker.check("sell", adapter, "BTC_JPY", regime_mult=1.5)
        # Should shrink to floor(0.006/1.5 / 0.001) * 0.001 = floor(0.004/0.001)*0.001 = 0.004
        assert result is False
        assert checker.current_lot == 0.004

    @pytest.mark.asyncio
    async def test_sell_regime_mult_truly_insufficient(self) -> None:
        """regime_mult=2.0: btc_free(0.0008) → max_base = 0.0004 < min_order → True."""
        checker = self._make_checker(lot=0.001)
        adapter = self._mock_adapter_sell(0.0008)
        result = await checker.check("sell", adapter, "BTC_JPY", regime_mult=2.0)
        # effective = 0.001 * 2.0 = 0.002, btc_free=0.0008 < 0.002
        # max_base = 0.0008 / 2.0 = 0.0004, floor(0.0004/0.001)*0.001 = 0 < 0.001 → insufficient
        assert result is True

    @pytest.mark.asyncio
    async def test_buy_regime_mult_makes_insufficient(self) -> None:
        """buy 側: regime_mult により JPY 不足に."""
        checker = self._make_checker(lot=0.005)
        price = 10_000_000.0
        margin = 1.01
        # Without mult: needed = 0.005 * 10M * 1.01 = 50500
        # With mult=1.5: needed = 0.0075 * 10M * 1.01 = 75750
        jpy_free = 60000.0  # > 50500 but < 75750
        adapter = self._mock_adapter_buy(jpy_free, price)
        result = await checker.check("buy", adapter, "BTC_JPY", regime_mult=1.5)
        # Should shrink: affordable_effective = 60000 / (10M * 1.01) = 0.00594
        # affordable_base = 0.00594 / 1.5 = 0.00396, floor(0.00396/0.001)*0.001 = 0.003
        assert result is False
        assert checker.current_lot == 0.003

    @pytest.mark.asyncio
    async def test_sell_restore_considers_regime_mult(self) -> None:
        """復元ロジック: 復元後の実効ロットが残高内であること."""
        checker = self._make_checker(lot=0.005)
        # Shrink to 0.003
        checker._current_lot = 0.003
        checker._pre_shrink_lot = 0.005
        # btc_free = 0.006: pre_shrink*mult(0.005*1.5=0.0075) > 0.006 → 復元しない
        adapter = self._mock_adapter_sell(0.006)
        result = await checker.check("sell", adapter, "BTC_JPY", regime_mult=1.5)
        assert result is False
        # Should NOT restore (0.005 * 1.5 = 0.0075 > 0.006)
        assert checker.current_lot == 0.003

    @pytest.mark.asyncio
    async def test_sell_restore_when_balance_sufficient_for_regime(self) -> None:
        """残高がレジーム込みで十分なら復元する."""
        checker = self._make_checker(lot=0.005)
        checker._current_lot = 0.003
        checker._pre_shrink_lot = 0.005
        # btc_free = 0.01 > 0.005 * 1.5 = 0.0075 → 復元可能
        adapter = self._mock_adapter_sell(0.01)
        result = await checker.check("sell", adapter, "BTC_JPY", regime_mult=1.5)
        assert result is False
        assert checker.current_lot == 0.005

    @pytest.mark.asyncio
    async def test_default_regime_mult_is_1(self) -> None:
        """regime_mult 省略時はデフォルト 1.0 (後方互換)."""
        checker = self._make_checker(lot=0.005)
        adapter = self._mock_adapter_sell(0.006)
        result = await checker.check("sell", adapter, "BTC_JPY")
        assert result is False
        assert checker.current_lot == 0.005


# ======================================================================
# §8-#1: _regime_lot_multiplier 抽出テスト
# ======================================================================


class TestRegimeLotMultiplier:
    """_regime_lot_multiplier() 単体テスト."""

    def _make_runner_mock(
        self,
        multipliers: dict[str, float] | None = None,
        regime_value: str | None = None,
    ) -> MagicMock:
        import types
        from scripts.v460.run_fill_test import FillTestRunner
        from scripts.v460.lib.fill_config import FillTestConfig

        runner = MagicMock()
        runner.config = FillTestConfig(
            regime_lot_multipliers=multipliers or {},
        )
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
        return runner

    def test_no_multipliers(self) -> None:
        runner = self._make_runner_mock()
        assert runner._regime_lot_multiplier() == 1.0

    def test_no_detector(self) -> None:
        runner = self._make_runner_mock(multipliers={"trending": 1.5})
        assert runner._regime_lot_multiplier() == 1.0

    def test_trending(self) -> None:
        runner = self._make_runner_mock(
            multipliers={"trending": 1.5},
            regime_value="trending",
        )
        assert runner._regime_lot_multiplier() == 1.5

    def test_unknown_regime_falls_back_to_1(self) -> None:
        runner = self._make_runner_mock(
            multipliers={"trending": 1.5},
            regime_value="unknown",
        )
        assert runner._regime_lot_multiplier() == 1.0


# ======================================================================
# §9-#5/7: _make_skip_record / _new_cycle_id テスト
# ======================================================================


class TestNewCycleId:
    """_new_cycle_id() の形式テスト."""

    def test_format_without_prefix(self) -> None:
        from scripts.v460.run_fill_test import FillTestRunner
        cid = FillTestRunner._new_cycle_id()
        parts = cid.split("_")
        assert len(parts) == 2
        # timestamp part should be numeric
        assert parts[0].isdigit()
        # uuid part should be 8 hex chars
        assert len(parts[1]) == 8

    def test_format_with_prefix(self) -> None:
        from scripts.v460.run_fill_test import FillTestRunner
        cid = FillTestRunner._new_cycle_id(prefix="test")
        assert cid.startswith("test_")
        parts = cid.split("_")
        assert len(parts) == 3

    def test_uniqueness(self) -> None:
        from scripts.v460.run_fill_test import FillTestRunner
        ids = {FillTestRunner._new_cycle_id() for _ in range(100)}
        assert len(ids) == 100


class TestMakeSkipRecord:
    """_make_skip_record() ヘルパのフィールド検証."""

    def _make_runner_mock(self) -> MagicMock:
        import types
        from scripts.v460.run_fill_test import FillTestRunner

        runner = MagicMock()
        runner._run_id = "test_run_001"
        runner._git_sha = "abc1234"
        runner._current_lot = 0.005
        runner._regime_detector = None
        runner._make_skip_record = types.MethodType(
            FillTestRunner._make_skip_record, runner,
        )
        runner._new_cycle_id = FillTestRunner._new_cycle_id
        return runner

    def test_basic_skip_record(self) -> None:
        runner = self._make_runner_mock()
        rec = runner._make_skip_record(
            side="buy",
            cancel_reason="test_reason",
        )
        assert rec.side == "buy"
        assert rec.cancel_reason == "test_reason"
        assert rec.cancelled is True
        assert rec.run_id == "test_run_001"
        assert rec.git_sha == "abc1234"
        assert rec.order_price == 0.0

    def test_auto_cycle_id(self) -> None:
        runner = self._make_runner_mock()
        rec = runner._make_skip_record(side="sell", cancel_reason="x")
        assert rec.cycle_id is not None
        assert len(rec.cycle_id) > 0

    def test_custom_cycle_id(self) -> None:
        runner = self._make_runner_mock()
        rec = runner._make_skip_record(
            side="buy", cancel_reason="x", cycle_id="custom_123",
        )
        assert rec.cycle_id == "custom_123"

    def test_default_order_quantity_is_current_lot(self) -> None:
        runner = self._make_runner_mock()
        runner._current_lot = 0.003
        rec = runner._make_skip_record(side="buy", cancel_reason="x")
        assert rec.order_quantity == 0.003

    def test_custom_order_quantity(self) -> None:
        runner = self._make_runner_mock()
        rec = runner._make_skip_record(
            side="buy", cancel_reason="x", order_quantity=0.01,
        )
        assert rec.order_quantity == 0.01

    def test_extra_kwargs_passed(self) -> None:
        runner = self._make_runner_mock()
        rec = runner._make_skip_record(
            side="buy",
            cancel_reason="balance_forced_skip",
            balance_forced_switch=True,
            regime="unknown",
        )
        assert rec.balance_forced_switch is True
        assert rec.regime == "unknown"

    def test_custom_timestamp(self) -> None:
        runner = self._make_runner_mock()
        rec = runner._make_skip_record(
            timestamp=12345.0,
            side="buy",
            cancel_reason="x",
        )
        assert rec.timestamp == 12345.0

    def test_count_trailing_cancel_reason(self) -> None:
        from scripts.v460.run_fill_test import FillTestRunner
        from ztb.metrics.fill_quality import FillRecord

        records = [
            FillRecord(
                cycle_id="1",
                timestamp=1.0,
                side="buy",
                order_price=100.0,
                order_quantity=0.001,
                cancelled=True,
                cancel_reason="x",
            ),
            FillRecord(
                cycle_id="2",
                timestamp=2.0,
                side="buy",
                order_price=100.0,
                order_quantity=0.001,
                cancelled=True,
                cancel_reason="y",
            ),
            FillRecord(
                cycle_id="3",
                timestamp=3.0,
                side="buy",
                order_price=100.0,
                order_quantity=0.001,
                cancelled=True,
                cancel_reason="y",
            ),
        ]
        assert FillTestRunner._count_trailing_cancel_reason(records, "y") == 2
        assert FillTestRunner._count_trailing_cancel_reason(records, "x") == 0


# ======================================================================
# §9-#4: SkipGate lot consistency — ソースレベル検証
# ======================================================================


class TestSkipGateLotConsistency:
    """_evaluate_skip_gate が order_lot パラメータを受け取ることを確認."""

    def test_evaluate_skip_gate_accepts_order_lot(self) -> None:
        """_evaluate_skip_gate のシグネチャに order_lot kwarg がある."""
        import inspect
        from scripts.v460.run_fill_test import FillTestRunner

        sig = inspect.signature(FillTestRunner._evaluate_skip_gate)
        params = sig.parameters
        assert "order_lot" in params, (
            "_evaluate_skip_gate should accept order_lot parameter"
        )
        # デフォルト値は None
        assert params["order_lot"].default is None

    def test_skip_gate_call_passes_regime_lot(self) -> None:
        """run_single_cycle 内の SkipGate 呼出しで order_lot= が渡されていることを確認.

        151# P3-03: regime_lot を1回算出し、SkipGate/発注/記録へ共通引き回し.
        """
        import inspect
        from scripts.v460.run_fill_test import FillTestRunner

        source = inspect.getsource(FillTestRunner.run_single_cycle)
        # 151# P3-03: 単一算出した _regime_lot を SkipGate に渡す
        assert "order_lot=_regime_lot" in source, (
            "SkipGate call should pass pre-computed _regime_lot (151# §10 #4)"
        )


class TestFillRecordBuilderIntegration:
    """FillRecord 組み立てが共通 builder に寄っていることを確認."""

    def test_build_fill_record_is_used(self) -> None:
        import inspect
        from scripts.v460.run_fill_test import FillTestRunner

        source = inspect.getsource(FillTestRunner._build_fill_record)
        assert "build_fill_record(" in source
        assert "_build_fill_measurement_fields(" in source
        assert "_build_fill_market_fields(" in source
        assert "_build_fill_strategy_fields(" in source
        measurement_source = inspect.getsource(FillTestRunner._build_fill_measurement_fields)
        market_source = inspect.getsource(FillTestRunner._build_fill_market_fields)
        assert "_resolve_fill_cancel_reason(" in measurement_source
        assert "_compute_fill_spread_bps(" in market_source


class TestCheckBalanceAcceptsRegimeMult:
    """_check_balance_for_side が regime_mult パラメータを受け取ることを確認."""

    def test_signature_has_regime_mult(self) -> None:
        import inspect
        from scripts.v460.run_fill_test import FillTestRunner

        sig = inspect.signature(FillTestRunner._check_balance_for_side)
        params = sig.parameters
        assert "regime_mult" in params
        assert params["regime_mult"].default == 1.0

    def test_run_continuous_passes_regime_mult(self) -> None:
        """run_continuous 内で regime_mult= が preflight に渡されていることを確認."""
        import inspect
        from scripts.v460.run_fill_test import FillTestRunner

        source = inspect.getsource(FillTestRunner.run_continuous)
        assert "_regime_lot_multiplier()" in source
        assert "regime_mult=_regime_mult" in source


# ======================================================================
# §9-#3: SkipGate OB 形式修正のソース検証
# ======================================================================


class TestSkipGateObFormat:
    """skip_gate_evaluator.py が ob_utils を使っていることを確認."""

    def test_uses_extract_price(self) -> None:
        import inspect
        from scripts.v460.lib import skip_gate_evaluator

        source = inspect.getsource(skip_gate_evaluator)
        assert "extract_price" in source
        assert "depth_volume" in source

    def test_no_dot_price_access(self) -> None:
        """OB レベルに .price でアクセスしていないことを確認."""
        import inspect
        from scripts.v460.lib import skip_gate_evaluator

        source = inspect.getsource(skip_gate_evaluator)
        # _build_ob_features 内で .price/.quantity を直接使わない
        # (extract_price / depth_volume 経由のみ)
        lines = source.split("\n")
        ob_feature_area = False
        for line in lines:
            if "_build_ob_features" in line:
                ob_feature_area = True
            if ob_feature_area and "return" in line and "ob_features" in line:
                ob_feature_area = False
            if ob_feature_area:
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                assert ".price" not in stripped or "extract_price" in stripped or "order_price" in stripped, (
                    f"Direct .price access found in _build_ob_features: {stripped}"
                )
