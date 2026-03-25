"""239# InfeasibleQuoteError + 制約前方移動テスト.

232# §1.5 [P1] 対応: feasible quote proactive calculation.
- InfeasibleQuoteError の属性・互換性テスト
- sell_max_spread の offset 計算前チェック (早期離脱) テスト
- executor の InfeasibleQuoteError 型安全 catch テスト
- _make_price_error_skip ヘルパーの重複排除テスト
"""

from __future__ import annotations

import asyncio
import inspect
import re
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.fill_cycle_executor import FillCycleExecutorMixin
from scripts.v460.lib.maker_price import (
    InfeasibleQuoteError,
    MakerPriceCalculator,
    MakerPriceResult,
)
from tests.unit.v460._fill_test_source import (
    FILL_CYCLE_EXECUTOR,
    MAKER_PRICE,
    read_class_method_source,
    read_source_text,
)
from ztb.trading.risk.fast_fill_defense import FastFillDefense

_MAKE_PRICE_ERROR_SKIP_SIG = inspect.signature(FillCycleExecutorMixin._make_price_error_skip)


# ─────────────────────────────────────────────────────
# Fixtures & Helpers
# ─────────────────────────────────────────────────────


def _make_config(**overrides: Any) -> Any:
    """テスト用 FillTestConfig を生成."""
    return FillTestConfig(**overrides)


@dataclass
class _MockOB:
    bids: list[list[float]]
    asks: list[list[float]]


def _make_adapter(best_bid: float = 10_000_000.0, best_ask: float = 10_005_000.0) -> MagicMock:
    """OB モック adapter."""
    adapter = MagicMock()
    ob = _MockOB(bids=[[best_bid, 1.0]], asks=[[best_ask, 1.0]])
    adapter.get_orderbook = AsyncMock(return_value=ob)
    return adapter


def _make_calculator(**cfg_overrides: Any) -> MakerPriceCalculator:
    """テスト用 MakerPriceCalculator を最小構成で生成."""
    cfg = _make_config(**cfg_overrides)
    ffd = FastFillDefense(cfg, base_offset_ratio=cfg.spread_offset_ratio)
    return MakerPriceCalculator(
        config=cfg,
        fast_fill_defense=ffd,
        regime_detector=None,
        base_offset_ratio=cfg.spread_offset_ratio,
    )


# ═══════════════════════════════════════════════════════════════════════
# A. InfeasibleQuoteError — 属性・互換性テスト
# ═══════════════════════════════════════════════════════════════════════


class TestInfeasibleQuoteError:
    """InfeasibleQuoteError の型構造・互換性検証."""

    def test_is_value_error_subclass(self) -> None:
        """ValueError のサブクラスであること (後方互換)."""
        err = InfeasibleQuoteError(reason="spread_too_narrow", msg="test")
        assert isinstance(err, ValueError)

    def test_reason_attribute(self) -> None:
        """reason 属性が設定されること."""
        err = InfeasibleQuoteError(reason="sell_guard_reject", msg="detail")
        assert err.reason == "sell_guard_reject"

    def test_message_preserved(self) -> None:
        """str(err) でメッセージが取得できること."""
        err = InfeasibleQuoteError(reason="spread_too_narrow", msg="Spread too narrow: 500 JPY")
        assert "Spread too narrow" in str(err)
        assert "500" in str(err)

    def test_except_value_error_catches(self) -> None:
        """既存の except ValueError で捕捉できること (後方互換)."""
        with pytest.raises(ValueError):
            raise InfeasibleQuoteError(reason="test", msg="backward compat")

    def test_except_exception_catches(self) -> None:
        """既存の except Exception で捕捉できること."""
        with pytest.raises(Exception):
            raise InfeasibleQuoteError(reason="test", msg="generic catch")

    def test_slots_defined(self) -> None:
        """__slots__ が定義されていること (メモリ効率)."""
        assert hasattr(InfeasibleQuoteError, "__slots__")
        assert "reason" in InfeasibleQuoteError.__slots__


# ═══════════════════════════════════════════════════════════════════════
# B. compute() — sell_max_spread 前方移動テスト
# ═══════════════════════════════════════════════════════════════════════


class TestSellMaxSpreadEarlyBailout:
    """239# sell_max_spread チェックが offset 計算前に実行されること."""

    def test_sell_wide_spread_raises_infeasible(self) -> None:
        """sell + spread > sell_max_spread_jpy → InfeasibleQuoteError."""
        calc = _make_calculator(
            sell_max_spread_jpy=3000.0,
            min_spread_jpy=100.0,
        )
        # spread = 5000 > sell_max_spread=3000
        adapter = _make_adapter(best_bid=10_000_000.0, best_ask=10_005_000.0)
        with pytest.raises(InfeasibleQuoteError) as exc_info:
            asyncio.run(calc.compute("sell", adapter, "btc_jpy"))
        assert exc_info.value.reason == "sell_guard_reject"
        assert "sell_guard" in str(exc_info.value)

    def test_buy_wide_spread_not_rejected(self) -> None:
        """buy 側は sell_max_spread_jpy に影響されない."""
        calc = _make_calculator(
            sell_max_spread_jpy=3000.0,
            min_spread_jpy=100.0,
        )
        adapter = _make_adapter(best_bid=10_000_000.0, best_ask=10_005_000.0)
        # buy 側: spread=5000 > sell_max=3000 でも通過
        result = asyncio.run(calc.compute("buy", adapter, "btc_jpy"))
        assert isinstance(result, MakerPriceResult)
        assert result.price > 0

    def test_narrow_spread_raises_infeasible(self) -> None:
        """spread < min_spread_jpy → InfeasibleQuoteError(spread_too_narrow)."""
        calc = _make_calculator(min_spread_jpy=10000.0)
        # spread = 5000 < min_spread=10000
        adapter = _make_adapter(best_bid=10_000_000.0, best_ask=10_005_000.0)
        with pytest.raises(InfeasibleQuoteError) as exc_info:
            asyncio.run(calc.compute("buy", adapter, "btc_jpy"))
        assert exc_info.value.reason == "spread_too_narrow"

    def test_sell_max_spread_zero_means_unlimited(self) -> None:
        """sell_max_spread_jpy=0 → 無制限 (sell でも通過)."""
        calc = _make_calculator(
            sell_max_spread_jpy=0,
            min_spread_jpy=100.0,
        )
        adapter = _make_adapter(best_bid=10_000_000.0, best_ask=10_005_000.0)
        result = asyncio.run(calc.compute("sell", adapter, "btc_jpy"))
        assert isinstance(result, MakerPriceResult)


# ═══════════════════════════════════════════════════════════════════════
# C. compute() — offset 重複排除テスト (sell_max_spread が1箇所のみ)
# ═══════════════════════════════════════════════════════════════════════


class TestSellGuardSingleLocation:
    """旧コードでは sell_max_spread チェックが offset 計算後にもあった。
    239# で前方移動 + 旧位置削除を確認。
    """

    def test_sell_guard_raise_count_in_compute_source(self) -> None:
        """_enforce_spread_guards() ソース内で sell_guard の raise は 1 箇所のみ.

        239# で前方移動後、_enforce_spread_guards に集約された。
        """
        source = read_class_method_source(MAKER_PRICE, "MakerPriceCalculator", "_enforce_spread_guards")
        # InfeasibleQuoteError で sell_guard_reject を raise する箇所 (複数行にまたがる)
        matches = re.findall(r'raise\s+InfeasibleQuoteError', source)
        sell_guard_count = sum(1 for m in re.finditer(r'sell_guard_reject', source))
        assert len(matches) >= 1
        assert sell_guard_count == 1, f"Expected 1 sell_guard_reject, found {sell_guard_count}"

    def test_no_old_valueerror_sell_guard_in_compute(self) -> None:
        """compute() に旧 ValueError("sell_guard: ...") が残っていないこと."""
        source = read_class_method_source(MAKER_PRICE, "MakerPriceCalculator", "compute")
        # ValueError(...sell_guard...) パターンが存在しないこと
        old_pattern = re.findall(r'raise\s+ValueError[^)]*sell_guard', source, re.DOTALL)
        assert len(old_pattern) == 0, "Old ValueError sell_guard still present"


# ═══════════════════════════════════════════════════════════════════════
# D. executor — InfeasibleQuoteError 型安全 catch テスト
# ═══════════════════════════════════════════════════════════════════════


class TestExecutorInfeasibleCatch:
    """executor の except InfeasibleQuoteError が存在し、
    string match ベースの sell_guard/spread_too_narrow が除去されていること。
    """

    def test_infeasible_catch_in_source(self) -> None:
        """FillCycleExecutorMixin に except InfeasibleQuoteError が存在."""
        source = read_source_text(FILL_CYCLE_EXECUTOR)
        assert "except InfeasibleQuoteError" in source

    def test_no_string_match_sell_guard_in_executor(self) -> None:
        """executor に旧 string match ("sell_guard" in err_msg) が残っていないこと."""
        source = read_class_method_source(
            FILL_CYCLE_EXECUTOR,
            "FillCycleExecutorMixin",
            "run_single_cycle",
        )
        assert '"sell_guard" in err_msg' not in source

    def test_no_string_match_spread_narrow_in_executor(self) -> None:
        """executor に旧 string match ("spread too narrow" in err_msg) が残っていないこと."""
        source = read_class_method_source(
            FILL_CYCLE_EXECUTOR,
            "FillCycleExecutorMixin",
            "run_single_cycle",
        )
        assert '"spread too narrow" in err_msg' not in source

    def test_infeasible_import_exists(self) -> None:
        """fill_cycle_executor が InfeasibleQuoteError をインポートしていること."""
        import scripts.v460.lib.fill_cycle_executor as mod

        assert hasattr(mod, "InfeasibleQuoteError")


# ═══════════════════════════════════════════════════════════════════════
# E. _make_price_error_skip ヘルパー存在テスト
# ═══════════════════════════════════════════════════════════════════════


class TestMakePriceErrorSkipHelper:
    """239# で抽出した _make_price_error_skip の存在・シグネチャ確認."""

    def test_method_exists(self) -> None:
        """FillCycleExecutorMixin に _make_price_error_skip が存在."""
        assert hasattr(FillCycleExecutorMixin, "_make_price_error_skip")

    def test_method_signature(self) -> None:
        """キーワード引数 side, cancel_reason, cycle_id, error を取ること."""
        params = list(_MAKE_PRICE_ERROR_SKIP_SIG.parameters.keys())
        assert "side" in params
        assert "cancel_reason" in params
        assert "cycle_id" in params
        assert "error" in params

    def test_fallback_dedup_no_duplicate_in_run_single_cycle(self) -> None:
        """pre-order phase の except ブロックが _make_price_error_skip を呼び出していること."""
        source = read_class_method_source(
            FILL_CYCLE_EXECUTOR,
            "FillCycleExecutorMixin",
            "_run_pre_order_phase",
        )
        assert "_make_price_error_skip" in source, (
            "_run_pre_order_phase should call _make_price_error_skip for error handling"
        )


# ═══════════════════════════════════════════════════════════════════════
# F. 市場理論: 制約集合崩壊の早期検知の意義
# ═══════════════════════════════════════════════════════════════════════


class TestFeasibleQuoteTheory:
    """232# §1.5 の市場理論的正当性を構造テストで確認.

    Avellaneda-Stoikov: maker の quote は best bid/ask 内に制約される。
    min_spread_jpy と sell_max_spread_jpy は feasible set の境界条件。
    両制約が同時に active な場合、feasible set が空集合になりうる。
    → 早期検知がサイクル計算コスト削減に寄与。
    """

    def test_narrow_and_wide_constraint_non_overlap(self) -> None:
        """min_spread > sell_max_spread → 構造的に sell 不可能."""
        calc = _make_calculator(
            min_spread_jpy=6000.0,
            sell_max_spread_jpy=3000.0,
        )
        # spread=5000: min_spread(6000)で先にreject
        adapter = _make_adapter(best_bid=10_000_000.0, best_ask=10_005_000.0)
        with pytest.raises(InfeasibleQuoteError) as exc_info:
            asyncio.run(calc.compute("sell", adapter, "btc_jpy"))
        assert exc_info.value.reason == "spread_too_narrow"

    def test_feasible_window_exists(self) -> None:
        """min_spread < spread < sell_max_spread → feasible."""
        calc = _make_calculator(
            min_spread_jpy=1000.0,
            sell_max_spread_jpy=10000.0,
        )
        # spread=5000: within [1000, 10000]
        adapter = _make_adapter(best_bid=10_000_000.0, best_ask=10_005_000.0)
        result = asyncio.run(calc.compute("sell", adapter, "btc_jpy"))
        assert isinstance(result, MakerPriceResult)
        assert result.spread == 5000.0

    def test_early_bailout_before_offset_keyword(self) -> None:
        """_enforce_spread_guards() が compute() 内で offset 計算より前に呼ばれること."""
        source = read_class_method_source(MAKER_PRICE, "MakerPriceCalculator", "compute")
        # _enforce_spread_guards 呼出位置
        guard_pos = source.find("_enforce_spread_guards")
        # offset 決定ロジック開始位置
        offset_logic_pos = source.find("offset 決定ロジック")
        assert guard_pos > 0, "_enforce_spread_guards not found in compute"
        assert offset_logic_pos > 0, "offset logic not found"
        assert guard_pos < offset_logic_pos, (
            "_enforce_spread_guards should appear BEFORE offset calculations"
        )


# ═══════════════════════════════════════════════════════════════════════
# G. 632# ATR floor cap テスト
# ═══════════════════════════════════════════════════════════════════════


class TestATRFloorCap:
    """632# ATR floor cap (min_spread_atr_cap_bps) の動作検証."""

    def test_atr_cap_limits_floor(self) -> None:
        """ATR floor が cap_bps を超えない — cap が有効な場合."""
        calc = _make_calculator(
            min_spread_jpy=100.0,
            min_spread_floor_bps=0.38,
            min_spread_atr_enabled=True,
            min_spread_atr_mult=2.0,
            min_spread_atr_cap_bps=2.0,  # 低めの cap
        )
        # σ を直接セットして ATR floor が cap を超えるケースをテスト
        calc._last_sigma = 0.001  # 高い σ → ATR=0.001*10M*2.0=20,000 JPY
        # spread=5000 < ATR(20000) だが cap 2.0bps → 10M*2.0/10000=2,000 JPY
        # effective_min = max(100, 380, 2000) = 2000 < 5000 → OK
        adapter = _make_adapter(best_bid=10_000_000.0, best_ask=10_005_000.0)
        result = asyncio.run(calc.compute("buy", adapter, "btc_jpy"))
        assert isinstance(result, MakerPriceResult)

    def test_atr_no_cap_blocks(self) -> None:
        """ATR floor cap=0 (無制限) で高 σ 時にブロックされること."""
        calc = _make_calculator(
            min_spread_jpy=100.0,
            min_spread_atr_enabled=True,
            min_spread_atr_mult=2.0,
            min_spread_atr_cap_bps=0.0,  # 無制限
        )
        calc._last_sigma = 0.001  # ATR=0.001*10M*2.0=20,000 JPY > spread=5000
        adapter = _make_adapter(best_bid=10_000_000.0, best_ask=10_005_000.0)
        with pytest.raises(InfeasibleQuoteError) as exc_info:
            asyncio.run(calc.compute("buy", adapter, "btc_jpy"))
        assert exc_info.value.reason == "spread_too_narrow"

    def test_atr_cap_config_field_exists(self) -> None:
        """FillTestConfig に min_spread_atr_cap_bps フィールドが存在."""
        cfg = FillTestConfig()
        assert hasattr(cfg, "min_spread_atr_cap_bps")

    def test_sigma_in_error_message(self) -> None:
        """632# spread_too_narrow エラーメッセージに σ 値が含まれること."""
        calc = _make_calculator(
            min_spread_jpy=100000.0,  # 強制ブロック
        )
        adapter = _make_adapter(best_bid=10_000_000.0, best_ask=10_005_000.0)
        with pytest.raises(InfeasibleQuoteError) as exc_info:
            asyncio.run(calc.compute("buy", adapter, "btc_jpy"))
        assert "σ=" in str(exc_info.value)
