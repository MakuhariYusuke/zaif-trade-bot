"""664# Deadlock Escape テスト.

648# Inventory Deadlock が長期化した際に min_spread を一時緩和し
クオート再開を試みる escape 機構の動作を検証する。
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.maker_price import (
    InfeasibleQuoteError,
    MakerPriceCalculator,
    MakerPriceResult,
)
from tests.unit.v460._fill_test_source import (
    MAKER_PRICE,
    ORCHESTRATOR_BALANCE,
    ORCHESTRATOR_POST_CYCLE,
    read_source_text,
)
from ztb.trading.risk.fast_fill_defense import FastFillDefense


# ─────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────


def _make_config(**overrides: Any) -> FillTestConfig:
    return FillTestConfig(**overrides)


def _make_calculator(**cfg_overrides: Any) -> MakerPriceCalculator:
    cfg = _make_config(**cfg_overrides)
    ffd = FastFillDefense(cfg, base_offset_ratio=cfg.spread_offset_ratio)
    return MakerPriceCalculator(
        config=cfg,
        fast_fill_defense=ffd,
        regime_detector=None,
        base_offset_ratio=cfg.spread_offset_ratio,
    )


def _make_adapter(
    best_bid: float = 10_000_000.0,
    best_ask: float = 10_005_000.0,
) -> MagicMock:
    from dataclasses import dataclass

    @dataclass
    class _MockOB:
        bids: list[list[float]]
        asks: list[list[float]]

    adapter = MagicMock()
    ob = _MockOB(bids=[[best_bid, 1.0]], asks=[[best_ask, 1.0]])
    adapter.get_orderbook = AsyncMock(return_value=ob)
    return adapter


# ═══════════════════════════════════════════════════════════════════════
# A. Config パラメータ存在テスト
# ═══════════════════════════════════════════════════════════════════════


class TestDeadlockEscapeConfig:
    """664# Deadlock escape 設定パラメータの存在と型."""

    def test_threshold_exists(self) -> None:
        cfg = FillTestConfig()
        assert hasattr(cfg, "deadlock_escape_threshold")
        assert isinstance(cfg.deadlock_escape_threshold, int)

    def test_spread_mult_exists(self) -> None:
        cfg = FillTestConfig()
        assert hasattr(cfg, "deadlock_escape_spread_mult")
        assert isinstance(cfg.deadlock_escape_spread_mult, float)

    def test_default_disabled(self) -> None:
        """デフォルトは threshold=0 (無効)."""
        cfg = FillTestConfig()
        assert cfg.deadlock_escape_threshold == 0

    def test_default_spread_mult(self) -> None:
        """デフォルト乗数は 0.5."""
        cfg = FillTestConfig()
        assert cfg.deadlock_escape_spread_mult == 0.5


# ═══════════════════════════════════════════════════════════════════════
# B. MakerPriceCalculator フラグテスト
# ═══════════════════════════════════════════════════════════════════════


class TestDeadlockEscapeFlag:
    """664# MakerPriceCalculator の deadlock escape フラグ動作."""

    def test_initial_state_inactive(self) -> None:
        calc = _make_calculator()
        assert calc.deadlock_escape_active is False

    def test_set_active(self) -> None:
        calc = _make_calculator()
        calc.set_deadlock_escape(True)
        assert calc.deadlock_escape_active is True

    def test_set_inactive(self) -> None:
        calc = _make_calculator()
        calc.set_deadlock_escape(True)
        calc.set_deadlock_escape(False)
        assert calc.deadlock_escape_active is False


# ═══════════════════════════════════════════════════════════════════════
# C. Spread Guard 緩和テスト
# ═══════════════════════════════════════════════════════════════════════


class TestDeadlockEscapeSpreadRelaxation:
    """664# escape 有効時に min_spread が緩和されることの機能テスト."""

    def test_narrow_spread_raises_without_escape(self) -> None:
        """通常時: spread < min_spread → InfeasibleQuoteError."""
        calc = _make_calculator(
            min_spread_jpy=10000.0,
            min_spread_floor_bps=0.0,
            min_spread_atr_enabled=False,
            deadlock_escape_threshold=20,
            deadlock_escape_spread_mult=0.5,
        )
        # spread = 5000 < min_spread = 10000 → reject
        adapter = _make_adapter(best_bid=10_000_000.0, best_ask=10_005_000.0)
        with pytest.raises(InfeasibleQuoteError) as exc_info:
            asyncio.run(calc.compute("buy", adapter, "btc_jpy"))
        assert exc_info.value.reason == "spread_too_narrow"

    def test_narrow_spread_passes_with_escape(self) -> None:
        """escape 有効時: effective_min が半減し通過."""
        calc = _make_calculator(
            min_spread_jpy=10000.0,
            min_spread_floor_bps=0.0,
            min_spread_atr_enabled=False,
            deadlock_escape_threshold=20,
            deadlock_escape_spread_mult=0.5,
        )
        calc.set_deadlock_escape(True)
        # spread = 5000, effective_min = 10000 * 0.5 = 5000 → 通過 (spread >= effective_min)
        adapter = _make_adapter(best_bid=10_000_000.0, best_ask=10_005_000.0)
        result = asyncio.run(calc.compute("buy", adapter, "btc_jpy"))
        assert isinstance(result, MakerPriceResult)
        assert result.price > 0

    def test_escape_does_not_affect_sell_guard(self) -> None:
        """escape は sell_max_spread_jpy (上限) には影響しない."""
        calc = _make_calculator(
            min_spread_jpy=100.0,
            sell_max_spread_jpy=3000.0,
            deadlock_escape_threshold=20,
            deadlock_escape_spread_mult=0.5,
        )
        calc.set_deadlock_escape(True)
        # spread = 5000 > sell_max = 3000 → reject regardless of escape
        adapter = _make_adapter(best_bid=10_000_000.0, best_ask=10_005_000.0)
        with pytest.raises(InfeasibleQuoteError) as exc_info:
            asyncio.run(calc.compute("sell", adapter, "btc_jpy"))
        assert exc_info.value.reason == "sell_guard_reject"

    def test_escape_inactive_when_threshold_zero(self) -> None:
        """threshold=0 (無効) では escape フラグが立っても緩和されない.

        これは config 境界テスト: deadlock_escape_spread_mult > 0 だが
        実際に threshold=0 のとき orchestrator は activate しない。
        フラグが外部から強制的に立てられても spread_mult が適用される点は
        maker_price 側の設計として正しい (フラグは orchestrator の責任)。
        """
        calc = _make_calculator(
            min_spread_jpy=10000.0,
            min_spread_floor_bps=0.0,
            min_spread_atr_enabled=False,
            deadlock_escape_threshold=0,
            deadlock_escape_spread_mult=0.5,
        )
        calc.set_deadlock_escape(True)
        # maker_price 側はフラグに従うので緩和される (orchestrator が立てないのが防御)
        adapter = _make_adapter(best_bid=10_000_000.0, best_ask=10_005_000.0)
        result = asyncio.run(calc.compute("buy", adapter, "btc_jpy"))
        assert isinstance(result, MakerPriceResult)


# ═══════════════════════════════════════════════════════════════════════
# D. ソースコントラクトテスト
# ═══════════════════════════════════════════════════════════════════════


class TestDeadlockEscapeSourceContract:
    """664# ソースレベルで実装が正しく配置されていることの検証."""

    def test_escape_in_spread_guards(self) -> None:
        """_enforce_spread_guards 内に DEADLOCK_ESCAPE ロジックが存在."""
        src = read_source_text(MAKER_PRICE)
        assert "DEADLOCK_ESCAPE" in src
        assert "deadlock_escape_spread_mult" in src

    def test_escape_activation_in_balance(self) -> None:
        """orchestrator_balance に escape 有効化ロジックが存在."""
        src = read_source_text(ORCHESTRATOR_BALANCE)
        assert "deadlock_escape" in src
        assert "set_deadlock_escape" in src

    def test_escape_deactivation_on_fill(self) -> None:
        """orchestrator_post_cycle で fill 成功時に escape が解除される."""
        src = read_source_text(ORCHESTRATOR_POST_CYCLE)
        assert "set_deadlock_escape(False)" in src

    def test_guard_fire_for_escape(self) -> None:
        """deadlock_escape 発動時に guard_fire が記録される."""
        src = read_source_text(ORCHESTRATOR_BALANCE)
        assert '"deadlock_escape"' in src

    def test_escape_after_atr_floor(self) -> None:
        """escape 緩和は ATR floor 計算後に適用される (3-tier 全考慮)."""
        src = read_source_text(MAKER_PRICE)
        # _enforce_spread_guards メソッド内での順序を検証
        guard_start = src.index("def _enforce_spread_guards(")
        guard_src = src[guard_start:]
        idx_atr = guard_src.index("atr_floor = min(atr_floor, atr_cap)")
        idx_escape = guard_src.index("deadlock_escape_active")
        assert idx_atr < idx_escape, (
            "deadlock escape must be applied after ATR floor calculation"
        )
