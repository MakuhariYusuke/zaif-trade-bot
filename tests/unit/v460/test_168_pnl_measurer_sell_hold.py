"""168# §4.1 #1: PnlMeasurer sell 保持期間延長テスト.

post_fill_wait_sec_sell が設定されている場合、sell 側で sellspecific の待機時間を使用することを検証。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.pnl_measurer import PnlMeasurer
from tests.unit.v460._yaml_test_helpers import clone_fill_test_config, load_fill_test_config_from_mapping


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


def _make_measurer(**overrides: object) -> PnlMeasurer:
    return PnlMeasurer(_make_config(**overrides))


def _make_mid_price_mock(price: float = 10_000_000.0) -> AsyncMock:
    """一定価格を返す mid_price mock."""
    return AsyncMock(return_value=price)


class _FakeClock:
    """PnlMeasurer の sleep/time を進めるだけの仮想時計."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    async def sleep(self, seconds: float) -> None:
        self.now += seconds

    def time(self) -> float:
        return self.now


@pytest.fixture
def fake_pnl_clock() -> _FakeClock:
    """PnlMeasurer の待機を実時間なしで進める."""
    clock = _FakeClock()
    with patch("scripts.v460.lib.pnl_measurer.asyncio.sleep", new=clock.sleep), patch(
        "scripts.v460.lib.pnl_measurer.time.time",
        new=clock.time,
    ):
        yield clock


class TestSellHoldPeriodExtension:
    """168# sell 保持期間延長テスト."""

    @pytest.mark.asyncio
    async def test_default_no_sell_override(self, fake_pnl_clock: _FakeClock) -> None:
        """post_fill_wait_sec_sell=None の場合、通常の wait_sec を使用."""
        measurer = _make_measurer(post_fill_wait_sec=0.05, post_fill_wait_sec_sell=None)
        get_mid = _make_mid_price_mock()

        result = await measurer.measure(
            filled=True,
            fill_price=10_000_000.0,
            side="sell",
            get_mid_price=get_mid,
        )

        assert result.actual_measurement_sec is not None
        assert result.actual_measurement_sec == pytest.approx(0.05, abs=1e-9)

    @pytest.mark.asyncio
    async def test_sell_uses_sell_specific_wait(self, fake_pnl_clock: _FakeClock) -> None:
        """sell 側は post_fill_wait_sec_sell (長い) を使用."""
        measurer = _make_measurer(
            post_fill_wait_sec=0.05,
            post_fill_wait_sec_sell=0.15,
        )
        get_mid = _make_mid_price_mock()

        result = await measurer.measure(
            filled=True,
            fill_price=10_000_000.0,
            side="sell",
            get_mid_price=get_mid,
        )

        assert result.actual_measurement_sec is not None
        assert result.actual_measurement_sec == pytest.approx(0.15, abs=1e-9)

    @pytest.mark.asyncio
    async def test_buy_ignores_sell_override(self, fake_pnl_clock: _FakeClock) -> None:
        """buy 側は post_fill_wait_sec_sell を無視し、通常の wait_sec を使用."""
        measurer = _make_measurer(
            post_fill_wait_sec=0.05,
            post_fill_wait_sec_sell=0.15,
        )
        get_mid = _make_mid_price_mock()

        result = await measurer.measure(
            filled=True,
            fill_price=10_000_000.0,
            side="buy",
            get_mid_price=get_mid,
        )

        assert result.actual_measurement_sec is not None
        assert result.actual_measurement_sec == pytest.approx(0.05, abs=1e-9)

    @pytest.mark.asyncio
    async def test_sell_pnl_computed_correctly(self, fake_pnl_clock: _FakeClock) -> None:
        """sell 保持期間延長時の PnL 計算が正しいことを検証."""
        fill_price = 10_000_000.0
        # 価格が下がる → sell は利益
        prices = iter([fill_price, fill_price - 100])

        async def declining_price() -> float:
            return next(prices)

        measurer = _make_measurer(
            post_fill_wait_sec=0.02,
            post_fill_wait_sec_sell=0.05,
        )

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
        measurer = _make_measurer(post_fill_wait_sec_sell=0.15)
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
    async def test_sell_early_exit_uses_sell_wait(self, fake_pnl_clock: _FakeClock) -> None:
        """early exit 有効時も sell 用の wait_sec が適用される."""
        measurer = _make_measurer(
            post_fill_wait_sec=0.05,
            post_fill_wait_sec_sell=0.15,
            early_exit_enabled=True,
            early_exit_monitor_interval_sec=0.03,
            early_exit_threshold_bps=100.0,  # 高閾値 → トリガーしない
        )
        get_mid = _make_mid_price_mock(10_000_000.0)

        result = await measurer.measure(
            filled=True,
            fill_price=10_000_000.0,
            side="sell",
            get_mid_price=get_mid,
        )

        assert result.early_exit_triggered is False
        assert result.actual_measurement_sec == pytest.approx(0.15, abs=1e-9)


class TestFillTestConfigSellHold:
    """FillTestConfig の sell hold パラメータテスト."""

    def test_default_is_none(self) -> None:
        """デフォルトでは post_fill_wait_sec_sell は None."""
        config = FillTestConfig()
        assert config.post_fill_wait_sec_sell is None

    def test_from_yaml_with_sell_hold(self) -> None:
        """YAML に post_fill_wait_sec_sell が指定されている場合."""
        config = clone_fill_test_config(
            load_fill_test_config_from_mapping(
                {
                    "post_fill_wait_sec": 30.0,
                    "post_fill_wait_sec_sell": 90.0,
                }
            )
        )
        assert config.post_fill_wait_sec == 30.0
        assert config.post_fill_wait_sec_sell == 90.0

    def test_from_yaml_without_sell_hold(self) -> None:
        """YAML に post_fill_wait_sec_sell がない場合は None."""
        config = clone_fill_test_config(load_fill_test_config_from_mapping({"post_fill_wait_sec": 30.0}))
        assert config.post_fill_wait_sec == 30.0
        assert config.post_fill_wait_sec_sell is None
