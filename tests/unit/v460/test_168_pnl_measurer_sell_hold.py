"""168# §4.1 #1: PnlMeasurer sell 保持期間延長テスト.

post_fill_wait_sec_sell が設定されている場合、sell 側で sellspecific の待機時間を使用することを検証。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.pnl_measurer import PnlMeasurer


def _make_config(**overrides: object) -> FillTestConfig:
    """テスト用 FillTestConfig."""
    defaults = dict(
        post_fill_wait_sec=30.0,
        early_exit_enabled=False,
        e3_sampling_ratio=0.0,  # E3 計測無効
        pnl_fee_deduction_enabled=False,
        as_deadzone_bps=0.5,
    )
    defaults.update(overrides)
    return FillTestConfig(**defaults)  # type: ignore[arg-type]


def _make_mid_price_mock(price: float = 10_000_000.0) -> AsyncMock:
    """一定価格を返す mid_price mock."""
    return AsyncMock(return_value=price)


class TestSellHoldPeriodExtension:
    """168# sell 保持期間延長テスト."""

    @pytest.mark.asyncio
    async def test_default_no_sell_override(self) -> None:
        """post_fill_wait_sec_sell=None の場合、通常の wait_sec を使用."""
        config = _make_config(post_fill_wait_sec=0.05, post_fill_wait_sec_sell=None)
        measurer = PnlMeasurer(config)
        get_mid = _make_mid_price_mock()

        t0 = time.monotonic()
        result = await measurer.measure(
            filled=True,
            fill_price=10_000_000.0,
            side="sell",
            get_mid_price=get_mid,
        )
        elapsed = time.monotonic() - t0

        # 0.05s ± tolerance (sell override なしなので通常値を使用)
        assert 0.04 <= elapsed < 0.5
        assert result.actual_measurement_sec is not None
        assert result.actual_measurement_sec >= 0.04

    @pytest.mark.asyncio
    async def test_sell_uses_sell_specific_wait(self) -> None:
        """sell 側は post_fill_wait_sec_sell (長い) を使用."""
        config = _make_config(
            post_fill_wait_sec=0.05,
            post_fill_wait_sec_sell=0.15,
        )
        measurer = PnlMeasurer(config)
        get_mid = _make_mid_price_mock()

        t0 = time.monotonic()
        result = await measurer.measure(
            filled=True,
            fill_price=10_000_000.0,
            side="sell",
            get_mid_price=get_mid,
        )
        elapsed = time.monotonic() - t0

        # sell は 0.15s 待機 (通常の 0.05s ではない)
        assert elapsed >= 0.12, f"sell should wait ~0.15s, got {elapsed:.3f}s"
        assert result.actual_measurement_sec is not None
        assert result.actual_measurement_sec >= 0.12

    @pytest.mark.asyncio
    async def test_buy_ignores_sell_override(self) -> None:
        """buy 側は post_fill_wait_sec_sell を無視し、通常の wait_sec を使用."""
        config = _make_config(
            post_fill_wait_sec=0.05,
            post_fill_wait_sec_sell=0.15,
        )
        measurer = PnlMeasurer(config)
        get_mid = _make_mid_price_mock()

        t0 = time.monotonic()
        await measurer.measure(
            filled=True,
            fill_price=10_000_000.0,
            side="buy",
            get_mid_price=get_mid,
        )
        elapsed = time.monotonic() - t0

        # buy は 0.05s (sell の 0.15s ではない)
        assert elapsed < 0.12, f"buy should wait ~0.05s, got {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_sell_pnl_computed_correctly(self) -> None:
        """sell 保持期間延長時の PnL 計算が正しいことを検証."""
        fill_price = 10_000_000.0
        # 価格が下がる → sell は利益
        prices = iter([fill_price, fill_price - 100])

        async def declining_price() -> float:
            return next(prices)

        config = _make_config(
            post_fill_wait_sec=0.02,
            post_fill_wait_sec_sell=0.05,
        )
        measurer = PnlMeasurer(config)

        result = await measurer.measure(
            filled=True,
            fill_price=fill_price,
            side="sell",
            get_mid_price=declining_price,
        )

        # sell: (mid_at_fill - mid_after) / mid_at_fill * 10000
        # = (10M - (10M-100)) / 10M * 10000 = 100/10M * 10000 = 0.1 bps
        assert result.post_fill_pnl is not None
        assert result.post_fill_pnl > 0  # sell with price drop = profit

    @pytest.mark.asyncio
    async def test_unfilled_returns_empty(self) -> None:
        """未約定の場合は空の PnlMeasurement を返す."""
        config = _make_config(post_fill_wait_sec_sell=0.15)
        measurer = PnlMeasurer(config)
        get_mid = _make_mid_price_mock()

        result = await measurer.measure(
            filled=False,
            fill_price=None,
            side="sell",
            get_mid_price=get_mid,
        )

        assert result.post_fill_pnl is None
        assert result.actual_measurement_sec is None


class TestSellHoldWithEarlyExit:
    """168# sell 保持 + early exit の組合せテスト."""

    @pytest.mark.asyncio
    async def test_sell_early_exit_uses_sell_wait(self) -> None:
        """early exit 有効時も sell 用の wait_sec が適用される."""
        config = _make_config(
            post_fill_wait_sec=0.05,
            post_fill_wait_sec_sell=0.15,
            early_exit_enabled=True,
            early_exit_monitor_interval_sec=0.03,
            early_exit_threshold_bps=100.0,  # 高閾値 → トリガーしない
        )
        measurer = PnlMeasurer(config)
        get_mid = _make_mid_price_mock(10_000_000.0)

        t0 = time.monotonic()
        result = await measurer.measure(
            filled=True,
            fill_price=10_000_000.0,
            side="sell",
            get_mid_price=get_mid,
        )
        elapsed = time.monotonic() - t0

        # sell は 0.15s 待機 (early exit 未トリガー)
        assert elapsed >= 0.12, f"sell should wait ~0.15s, got {elapsed:.3f}s"
        assert result.early_exit_triggered is False


class TestFillTestConfigSellHold:
    """FillTestConfig の sell hold パラメータテスト."""

    def test_default_is_none(self) -> None:
        """デフォルトでは post_fill_wait_sec_sell は None."""
        config = FillTestConfig()
        assert config.post_fill_wait_sec_sell is None

    def test_from_yaml_with_sell_hold(self) -> None:
        """YAML に post_fill_wait_sec_sell が指定されている場合."""
        yaml_cfg = {
            "post_fill_wait_sec": 30.0,
            "post_fill_wait_sec_sell": 90.0,
        }
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.post_fill_wait_sec == 30.0
        assert config.post_fill_wait_sec_sell == 90.0

    def test_from_yaml_without_sell_hold(self) -> None:
        """YAML に post_fill_wait_sec_sell がない場合は None."""
        yaml_cfg = {
            "post_fill_wait_sec": 30.0,
        }
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.post_fill_wait_sec == 30.0
        assert config.post_fill_wait_sec_sell is None
