"""128# dust sweep テスト.

BalanceChecker の端数BTC一掃機能をテスト:
- dust 検出して全額売却ロットに拡張
- apply_lot_floor が dust sweep 中にスキップされる
- サイクル後のロット復元
- dust_sweep_enabled=false で無効化
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.v460.lib.balance_checker import BalanceChecker
from scripts.v460.lib.fill_config import FillTestConfig


# ---- テスト用ヘルパー ----


@dataclass
class MockBalance:
    """残高モック."""

    free: float
    locked: float = 0.0


def _make_config(**overrides: object) -> FillTestConfig:
    """テスト用 FillTestConfig を生成."""
    defaults = {
        "order_quantity": 0.001,
        "min_order_btc": 0.001,
        "dust_sweep_enabled": True,
        "balance_margin_ratio": 1.01,
    }
    defaults.update(overrides)
    return FillTestConfig(**defaults)


def _make_adapter(btc_free: float = 0.001, jpy_free: float = 20000.0) -> AsyncMock:
    """exchange adapter モック."""
    adapter = AsyncMock()
    adapter.get_balance = AsyncMock(
        side_effect=lambda currency: [MockBalance(free=btc_free)]
        if currency == "BTC"
        else [MockBalance(free=jpy_free)]
    )
    adapter.get_current_price = AsyncMock(return_value=10_500_000.0)
    return adapter


# ---- テストケース ----


class TestDustSweepDetection:
    """128# dust 検出テスト."""

    @pytest.mark.asyncio
    async def test_dust_detected_on_sell(self) -> None:
        """BTC 残高に dust がある場合、全額売却ロットに拡張."""
        config = _make_config()
        checker = BalanceChecker(config)
        adapter = _make_adapter(btc_free=0.00199707)

        skip = await checker.check("sell", adapter, "btc_jpy")

        assert not skip, "dust がある sell はスキップされない"
        assert checker.dust_sweep_active
        assert checker.current_lot == pytest.approx(0.00199707, abs=1e-9)

    @pytest.mark.asyncio
    async def test_no_dust_no_sweep(self) -> None:
        """dust がない場合は通常ロットのまま."""
        config = _make_config()
        checker = BalanceChecker(config)
        adapter = _make_adapter(btc_free=0.002)

        skip = await checker.check("sell", adapter, "btc_jpy")

        assert not skip
        assert not checker.dust_sweep_active
        assert checker.current_lot == 0.001  # 変わらない

    @pytest.mark.asyncio
    async def test_dust_disabled(self) -> None:
        """dust_sweep_enabled=false なら dust があっても通常ロット."""
        config = _make_config(dust_sweep_enabled=False)
        checker = BalanceChecker(config)
        adapter = _make_adapter(btc_free=0.00199707)

        skip = await checker.check("sell", adapter, "btc_jpy")

        assert not skip
        assert not checker.dust_sweep_active
        assert checker.current_lot == 0.001  # 通常ロットのまま

    @pytest.mark.asyncio
    async def test_insufficient_btc_no_sweep(self) -> None:
        """BTC < min_order_btc の場合は sell スキップ (dust sweep 不可)."""
        config = _make_config()
        checker = BalanceChecker(config)
        adapter = _make_adapter(btc_free=0.00099707)

        skip = await checker.check("sell", adapter, "btc_jpy")

        assert skip, "min_order_btc 未満は sell スキップ"
        assert not checker.dust_sweep_active

    @pytest.mark.asyncio
    async def test_buy_side_unaffected(self) -> None:
        """buy 側は dust sweep に影響されない."""
        config = _make_config()
        checker = BalanceChecker(config)
        adapter = _make_adapter(btc_free=0.00199707, jpy_free=20000.0)

        skip = await checker.check("buy", adapter, "btc_jpy")

        assert not skip
        assert not checker.dust_sweep_active


class TestDustSweepLotFloor:
    """128# apply_lot_floor が dust sweep 中にスキップされるテスト."""

    @pytest.mark.asyncio
    async def test_lot_floor_skipped_during_dust_sweep(self) -> None:
        """dust sweep 中は lot floor が適用されない."""
        config = _make_config()
        checker = BalanceChecker(config)
        adapter = _make_adapter(btc_free=0.00199707)

        await checker.check("sell", adapter, "btc_jpy")
        assert checker.dust_sweep_active

        # apply_lot_floor を呼んでも dust 量が維持される
        checker.apply_lot_floor()
        assert checker.current_lot == pytest.approx(0.00199707, abs=1e-9)

    def test_lot_floor_normal_without_dust(self) -> None:
        """dust sweep 非アクティブ時は通常の lot floor が機能する."""
        config = _make_config()
        checker = BalanceChecker(config)
        checker.current_lot = 0.00150  # 端数なロット

        checker.apply_lot_floor()
        assert checker.current_lot == 0.001  # min_order_btc 単位にフロア


class TestDustSweepRestore:
    """128# dust sweep 後のロット復元テスト."""

    @pytest.mark.asyncio
    async def test_restore_after_sweep(self) -> None:
        """dust sweep 後にロットが元に戻る."""
        config = _make_config()
        checker = BalanceChecker(config)
        adapter = _make_adapter(btc_free=0.00199707)

        await checker.check("sell", adapter, "btc_jpy")
        assert checker.current_lot == pytest.approx(0.00199707, abs=1e-9)
        assert checker.dust_sweep_active

        checker.restore_lot_after_dust_sweep()
        assert checker.current_lot == 0.001
        assert not checker.dust_sweep_active

    def test_restore_noop_when_inactive(self) -> None:
        """dust sweep 非アクティブ時の restore は no-op."""
        config = _make_config()
        checker = BalanceChecker(config)
        original_lot = checker.current_lot

        checker.restore_lot_after_dust_sweep()
        assert checker.current_lot == original_lot
        assert not checker.dust_sweep_active

    @pytest.mark.asyncio
    async def test_restore_lot_after_exception(self) -> None:
        """例外時もロットが復元される (run_fill_test の finally 相当)."""
        config = _make_config()
        checker = BalanceChecker(config)
        adapter = _make_adapter(btc_free=0.00299707)

        await checker.check("sell", adapter, "btc_jpy")
        assert checker.dust_sweep_active

        # 例外が起きたとして...
        checker.restore_lot_after_dust_sweep()
        assert checker.current_lot == 0.001
        assert not checker.dust_sweep_active


class TestDustSweepEdgeCases:
    """128# dust sweep エッジケーステスト."""

    @pytest.mark.asyncio
    async def test_multiple_lots_with_dust(self) -> None:
        """複数ロット分 + dust の場合も全額売却."""
        config = _make_config()
        checker = BalanceChecker(config)
        # 通常 0.001 ロットだが残高 0.00299707 (2ロット + dust)
        adapter = _make_adapter(btc_free=0.00299707)

        skip = await checker.check("sell", adapter, "btc_jpy")

        assert not skip
        assert checker.dust_sweep_active
        assert checker.current_lot == pytest.approx(0.00299707, abs=1e-9)

    @pytest.mark.asyncio
    async def test_tiny_dust_below_epsilon(self) -> None:
        """極小の丸め誤差 (< 1e-9) は dust とみなさない."""
        config = _make_config()
        checker = BalanceChecker(config)
        adapter = _make_adapter(btc_free=0.001)  # 正確に 1 mBTC

        skip = await checker.check("sell", adapter, "btc_jpy")

        assert not skip
        assert not checker.dust_sweep_active
        assert checker.current_lot == 0.001

    @pytest.mark.asyncio
    async def test_dust_sweep_with_shrunk_lot(self) -> None:
        """ロット縮小中に dust がある場合もsweep が機能."""
        config = _make_config()
        checker = BalanceChecker(config)
        # btc_free < current_lot だが >= min_order_btc → 縮小 → dust sweep
        adapter = _make_adapter(btc_free=0.00199707)
        checker.current_lot = 0.003  # 縮小が必要

        skip = await checker.check("sell", adapter, "btc_jpy")

        assert not skip
        # 縮小で 0.001 にした後、dust sweep で 0.00199707 に拡張
        assert checker.dust_sweep_active
        assert checker.current_lot == pytest.approx(0.00199707, abs=1e-9)

    @pytest.mark.asyncio
    async def test_sequential_dust_sweeps(self) -> None:
        """連続して dust sweep → restore → 再 sweep が正常動作."""
        config = _make_config()
        checker = BalanceChecker(config)

        # 1回目
        adapter1 = _make_adapter(btc_free=0.00199707)
        await checker.check("sell", adapter1, "btc_jpy")
        assert checker.dust_sweep_active
        checker.restore_lot_after_dust_sweep()
        assert checker.current_lot == 0.001

        # 2回目 (異なる dust 量)
        adapter2 = _make_adapter(btc_free=0.00199500)
        await checker.check("sell", adapter2, "btc_jpy")
        assert checker.dust_sweep_active
        assert checker.current_lot == pytest.approx(0.001995, abs=1e-9)
        checker.restore_lot_after_dust_sweep()
        assert checker.current_lot == 0.001
