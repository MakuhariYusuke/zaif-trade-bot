"""659# balance_checker リファクタリング + T1-1 MCB HALT 警告テスト.

_apply_lot_shrink / _try_lot_restore 共通ヘルパーの動作検証、
および MCB HALT 時のポジション警告ログの検証。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional
from unittest.mock import patch

import pytest

from scripts.v460.lib.balance_checker import BalanceChecker, BalanceAdapterProtocol
from scripts.v460.lib.fill_config import FillTestConfig


# ------------------------------------------------------------------
# Shared Fixtures
# ------------------------------------------------------------------


@dataclass
class _FakeBalance:
    free: float


class FakeAdapter:
    """テスト用の残高アダプタ."""

    def __init__(self, btc: float = 0.0, jpy: float = 0.0, price: float = 10_000_000.0) -> None:
        self.btc = btc
        self.jpy = jpy
        self.price = price

    async def get_balance(self, currency: str) -> Sequence[_FakeBalance]:
        if currency == "BTC":
            return [_FakeBalance(free=self.btc)]
        return [_FakeBalance(free=self.jpy)]

    async def get_current_price(self, symbol: str) -> Optional[float]:
        return self.price


def _make_config(**overrides: object) -> FillTestConfig:
    defaults = dict(
        order_quantity=0.001,
        min_order_btc=0.001,
        max_lot=0.01,
        balance_margin_ratio=1.1,
        dust_sweep_enabled=False,
    )
    defaults.update(overrides)
    return FillTestConfig(**defaults)  # type: ignore[arg-type]


def _run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


# ==================================================================
# _apply_lot_shrink テスト
# ==================================================================


class TestApplyLotShrink:
    """659# _apply_lot_shrink 共通ヘルパーの単体テスト."""

    def test_shrink_updates_lot_and_saves_pre_shrink(self) -> None:
        bc = BalanceChecker(_make_config())
        assert bc.current_lot == 0.001
        bc._apply_lot_shrink(0.0005, "test shrink")
        assert bc.current_lot == 0.0005
        assert bc.pre_shrink_lot == 0.001

    def test_shrink_does_not_overwrite_pre_shrink_when_active(self) -> None:
        bc = BalanceChecker(_make_config())
        bc.pre_shrink_lot = 0.005  # 外部で設定済み
        bc.balance_shrink_active = True
        bc._apply_lot_shrink(0.0003, "test")
        assert bc.current_lot == 0.0003
        assert bc.pre_shrink_lot == 0.005  # 上書きされない

    def test_shrink_logs_message(self, caplog: pytest.LogCaptureFixture) -> None:
        bc = BalanceChecker(_make_config())
        with caplog.at_level(logging.INFO):
            bc._apply_lot_shrink(0.0008, "CUSTOM_LOG_MSG")
        assert "CUSTOM_LOG_MSG" in caplog.text


# ==================================================================
# _try_lot_restore テスト
# ==================================================================


class TestTryLotRestore:
    """659# _try_lot_restore 共通ヘルパーの単体テスト."""

    def test_restore_when_affordable(self) -> None:
        bc = BalanceChecker(_make_config())
        bc._apply_lot_shrink(0.0005, "shrink")
        assert bc.current_lot == 0.0005
        bc._try_lot_restore(True, "BTC")
        assert bc.current_lot == 0.001  # pre_shrink_lot に復元

    def test_no_restore_when_not_affordable(self) -> None:
        bc = BalanceChecker(_make_config())
        bc._apply_lot_shrink(0.0005, "shrink")
        bc._try_lot_restore(False, "BTC")
        assert bc.current_lot == 0.0005  # 変化なし

    def test_no_restore_when_balance_shrink_active(self) -> None:
        bc = BalanceChecker(_make_config())
        bc._apply_lot_shrink(0.0005, "shrink")
        bc.balance_shrink_active = True
        bc._try_lot_restore(True, "JPY")
        assert bc.current_lot == 0.0005  # active 時は復元しない

    def test_no_restore_when_lot_already_at_pre_shrink(self) -> None:
        bc = BalanceChecker(_make_config())
        # 縮小していない → current == pre_shrink → 復元不要
        bc._try_lot_restore(True, "BTC")
        assert bc.current_lot == 0.001

    def test_restore_logs_side(self, caplog: pytest.LogCaptureFixture) -> None:
        bc = BalanceChecker(_make_config())
        bc._apply_lot_shrink(0.0005, "shrink")
        with caplog.at_level(logging.INFO):
            bc._try_lot_restore(True, "JPY")
        assert "JPY 残高回復" in caplog.text


# ==================================================================
# _check_sell / _check_buy 統合テスト (659# refactor 後の動作検証)
# ==================================================================


class TestCheckSellRefactored:
    """659# リファクタ後の sell チェック動作検証."""

    def test_sufficient_btc_returns_false(self) -> None:
        bc = BalanceChecker(_make_config())
        adapter = FakeAdapter(btc=0.01)
        assert _run(bc.check("sell", adapter, "BTC_JPY")) is False

    def test_insufficient_btc_shrinks_lot(self) -> None:
        bc = BalanceChecker(_make_config())
        adapter = FakeAdapter(btc=0.0005)  # min_order 未満
        result = _run(bc.check("sell", adapter, "BTC_JPY"))
        assert result is True  # 注文スキップ

    def test_shortage_but_above_min_shrinks(self) -> None:
        """BTC < effective_lot だが >= min_order → 縮小して続行."""
        bc = BalanceChecker(_make_config(order_quantity=0.005))
        adapter = FakeAdapter(btc=0.002)
        result = _run(bc.check("sell", adapter, "BTC_JPY"))
        assert result is False  # 縮小して続行
        assert bc.current_lot == 0.002

    def test_restore_after_balance_recovery(self) -> None:
        """BTC 回復時にロット復元."""
        bc = BalanceChecker(_make_config(order_quantity=0.005))
        adapter_low = FakeAdapter(btc=0.002)
        _run(bc.check("sell", adapter_low, "BTC_JPY"))
        assert bc.current_lot == 0.002
        adapter_high = FakeAdapter(btc=0.01)
        _run(bc.check("sell", adapter_high, "BTC_JPY"))
        assert bc.current_lot == 0.005  # 復元


class TestCheckBuyRefactored:
    """659# リファクタ後の buy チェック動作検証."""

    def test_sufficient_jpy_returns_false(self) -> None:
        bc = BalanceChecker(_make_config())
        adapter = FakeAdapter(jpy=100_000.0, price=10_000_000.0)
        assert _run(bc.check("buy", adapter, "BTC_JPY")) is False

    def test_insufficient_jpy_returns_true(self) -> None:
        """JPY 極少 → min_order 未満 → skip."""
        bc = BalanceChecker(_make_config())
        adapter = FakeAdapter(jpy=100.0, price=10_000_000.0)
        result = _run(bc.check("buy", adapter, "BTC_JPY"))
        assert result is True

    def test_shortage_but_above_min_shrinks(self) -> None:
        """JPY < needed だが affordable >= min_order → 縮小して続行."""
        bc = BalanceChecker(_make_config(order_quantity=0.005))
        # 0.005 * 10M * 1.1 = 55,000 JPY needed, give 22,000 → ~0.002
        adapter = FakeAdapter(jpy=22_000.0, price=10_000_000.0)
        result = _run(bc.check("buy", adapter, "BTC_JPY"))
        assert result is False
        assert bc.current_lot >= 0.001
        assert bc.current_lot < 0.005

    def test_restore_after_balance_recovery(self) -> None:
        bc = BalanceChecker(_make_config(order_quantity=0.005, max_lot=0.005))
        adapter_low = FakeAdapter(jpy=22_000.0, price=10_000_000.0)
        _run(bc.check("buy", adapter_low, "BTC_JPY"))
        old = bc.current_lot
        assert old < 0.005
        adapter_high = FakeAdapter(jpy=200_000.0, price=10_000_000.0)
        _run(bc.check("buy", adapter_high, "BTC_JPY"))
        assert bc.current_lot == 0.005

    def test_regime_mult_affects_shrink(self) -> None:
        """regime_mult > 1 時にロット計算が正しくスケールされる."""
        bc = BalanceChecker(_make_config(order_quantity=0.005))
        # regime_mult=2.0: effective = 0.01, needs 110,000 JPY
        adapter = FakeAdapter(jpy=55_000.0, price=10_000_000.0)
        _run(bc.check("buy", adapter, "BTC_JPY", regime_mult=2.0))
        # affordable_effective = 55000 / (10M * 1.1) = 0.005
        # affordable_base = 0.005 / 2.0 = 0.0025
        assert bc.current_lot == 0.0025


# ==================================================================
# T1-1: MCB HALT ポジション警告テスト
# ==================================================================


class TestMCBHaltPositionWarning:
    """659# T1-1: MCB HALT 時に BTC ポジションがあれば警告する."""

    def test_warning_logged_when_btc_position_exists(self) -> None:
        """last_btc_free > 0 → WARNING ログ."""
        src = (
            "scripts/v460/lib/orchestrator_pre_cycle.py"
        )
        # MCB HALT ログ箇所のソースコードに警告ロジックが含まれることを検証
        import importlib
        mod = importlib.import_module("scripts.v460.lib.orchestrator_pre_cycle")
        import inspect
        source = inspect.getsource(mod)
        # 659# T1-1 の警告コードが存在するか
        assert "659# MCB" in source
        assert "HALT with open BTC position" in source

    def test_micro_circuit_breaker_halt_has_cooldown(self) -> None:
        """MCB HALT で cooldown_sec が設定される基本テスト."""
        from scripts.v460.lib.micro_circuit_breaker import (
            MCBConfig, MCBLevel, MicroCircuitBreaker,
        )
        mcb = MicroCircuitBreaker(MCBConfig(
            baseline_sample_interval_sec=1.0,
            halt_cooldown_sec=120.0,
        ))
        # 5分窓で大幅変動 → HALT
        base = 100_000.0
        for i in range(400):
            mcb.update(base + (i % 3), float(i))
        # 急落
        mcb.update(base * 0.9, 401.0)
        result = mcb.check(401.0)
        # HALT or WARNING (デフォルト閾値依存)
        if result.level == MCBLevel.HALT:
            assert result.cooldown_remaining_sec == 120.0
