from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import ztb.trading.common.cancel_reasons as CR
from scripts.v460.lib.cross_venue_lead_lag import CrossVenueLeadLagHint
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.maker_price import InfeasibleQuoteError, MakerPriceCalculator
from tests.unit.v460._yaml_test_helpers import load_yaml_mapping
from ztb.trading.risk.fast_fill_defense import FastFillDefense, FastFillDefenseConfig


class _OB:
    def __init__(
        self,
        timestamp: float,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        exchange: str,
    ) -> None:
        self.timestamp = timestamp
        self.bids = bids
        self.asks = asks
        self.exchange = exchange


class _OrderbookAdapter:
    def __init__(self, orderbooks: list[_OB]) -> None:
        self._orderbooks = list(orderbooks)

    async def get_orderbook(self, symbol: str, depth: int = 1) -> _OB:
        assert symbol == "btc_jpy"
        if len(self._orderbooks) > 1:
            return self._orderbooks.pop(0)
        return self._orderbooks[0]


def _make_calc(config: FillTestConfig) -> MakerPriceCalculator:
    ffd = FastFillDefense(
        config=FastFillDefenseConfig(
            enabled=False,
            threshold_sec=5.0,
            offset_boost=1.0,
            max_offset_ratio=config.max_offset_ratio,
            min_offset_ratio=config.min_offset_ratio,
        ),
        base_offset_ratio=config.spread_offset_ratio,
    )
    return MakerPriceCalculator(
        config=config,
        fast_fill_defense=ffd,
        regime_detector=None,
        base_offset_ratio=config.spread_offset_ratio,
    )


def _buy_hint(
    *,
    spread_bps: float,
    confidence: float = 1.0,
) -> CrossVenueLeadLagHint:
    return CrossVenueLeadLagHint(
        direction="down",
        adverse_side="buy",
        spread_bps=spread_bps,
        reference_velocity_bps=-3.0,
        age_sec=0.5,
        reference_exchange="bitflyer",
        confidence=confidence,
    )


def _adapter() -> _OrderbookAdapter:
    return _OrderbookAdapter(
        [_OB(100.0, [(10_000_000.0, 1.0)], [(10_002_000.0, 1.0)], "coincheck")]
    )


class TestCrossVenueBuyProtect:
    def test_buy_veto_when_spread_exceeds_threshold(self) -> None:
        config = FillTestConfig(
            min_spread_jpy=1.0,
            cross_venue_lead_lag_enabled=True,
            cross_venue_buy_protect_enabled=True,
            cross_venue_buy_veto_spread_bps=5.0,
        )
        calc = _make_calc(config)
        calc.set_cross_venue_lead_lag_hint(_buy_hint(spread_bps=6.0))

        with pytest.raises(InfeasibleQuoteError) as exc_info:
            asyncio.run(calc.compute("buy", _adapter(), "btc_jpy"))

        assert exc_info.value.reason == CR.CROSS_VENUE_BUY_VETO

    def test_buy_boost_when_spread_moderate(self) -> None:
        config = FillTestConfig(
            min_spread_jpy=1.0,
            cross_venue_lead_lag_enabled=True,
            cross_venue_buy_protect_enabled=True,
            cross_venue_buy_veto_spread_bps=5.0,
            cross_venue_buy_boost_spread_bps=3.0,
            cross_venue_buy_offset_boost_factor=1.3,
        )
        baseline = _make_calc(config)
        guarded = _make_calc(config)
        guarded.set_cross_venue_lead_lag_hint(_buy_hint(spread_bps=4.0))

        base_result = asyncio.run(baseline.compute("buy", _adapter(), "btc_jpy"))
        guarded_result = asyncio.run(guarded.compute("buy", _adapter(), "btc_jpy"))

        assert guarded_result.effective_offset_ratio > base_result.effective_offset_ratio
        assert guarded._cross_venue_buy_offset_mult == pytest.approx(1.3)

    def test_buy_no_action_below_threshold(self) -> None:
        config = FillTestConfig(
            min_spread_jpy=1.0,
            cross_venue_lead_lag_enabled=True,
            cross_venue_buy_protect_enabled=True,
            cross_venue_buy_boost_spread_bps=3.0,
        )
        baseline = _make_calc(config)
        guarded = _make_calc(config)
        baseline.set_cross_venue_lead_lag_hint(_buy_hint(spread_bps=2.0))
        guarded.set_cross_venue_lead_lag_hint(_buy_hint(spread_bps=2.0))

        base_result = asyncio.run(baseline.compute("buy", _adapter(), "btc_jpy"))
        guarded_result = asyncio.run(guarded.compute("buy", _adapter(), "btc_jpy"))

        assert guarded_result.effective_offset_ratio == pytest.approx(base_result.effective_offset_ratio)
        assert guarded._cross_venue_buy_offset_mult is None

    def test_sell_not_affected_by_buy_config(self) -> None:
        config = FillTestConfig(
            min_spread_jpy=1.0,
            cross_venue_lead_lag_enabled=True,
            cross_venue_buy_protect_enabled=True,
            cross_venue_buy_boost_spread_bps=3.0,
            cross_venue_buy_offset_boost_factor=1.3,
        )
        baseline = _make_calc(config)
        guarded = _make_calc(config)
        guarded.set_cross_venue_lead_lag_hint(_buy_hint(spread_bps=4.0))

        base_result = asyncio.run(baseline.compute("sell", _adapter(), "btc_jpy"))
        guarded_result = asyncio.run(guarded.compute("sell", _adapter(), "btc_jpy"))

        assert guarded_result.effective_offset_ratio == pytest.approx(base_result.effective_offset_ratio)

    def test_disabled_config_no_action(self) -> None:
        config = FillTestConfig(
            min_spread_jpy=1.0,
            cross_venue_lead_lag_enabled=True,
            cross_venue_buy_protect_enabled=False,
            cross_venue_buy_boost_spread_bps=3.0,
        )
        baseline = _make_calc(
            FillTestConfig(
                min_spread_jpy=1.0,
                cross_venue_lead_lag_enabled=True,
                cross_venue_buy_protect_enabled=False,
                cross_venue_buy_boost_spread_bps=3.0,
            )
        )
        calc = _make_calc(config)
        baseline.set_cross_venue_lead_lag_hint(_buy_hint(spread_bps=8.0))
        calc.set_cross_venue_lead_lag_hint(_buy_hint(spread_bps=8.0))

        base_result = asyncio.run(baseline.compute("buy", _adapter(), "btc_jpy"))
        result = asyncio.run(calc.compute("buy", _adapter(), "btc_jpy"))

        assert result.effective_offset_ratio == pytest.approx(base_result.effective_offset_ratio)

    def test_no_hint_no_action(self) -> None:
        config = FillTestConfig(
            min_spread_jpy=1.0,
            cross_venue_lead_lag_enabled=True,
            cross_venue_buy_protect_enabled=True,
        )
        calc = _make_calc(config)
        result = asyncio.run(calc.compute("buy", _adapter(), "btc_jpy"))
        assert result.effective_offset_ratio == pytest.approx(config.spread_offset_ratio)

    def test_adverse_side_mismatch(self) -> None:
        config = FillTestConfig(
            min_spread_jpy=1.0,
            cross_venue_lead_lag_enabled=True,
            cross_venue_buy_protect_enabled=True,
        )
        calc = _make_calc(config)
        calc.set_cross_venue_lead_lag_hint(
            CrossVenueLeadLagHint(
                direction="up",
                adverse_side="sell",
                spread_bps=8.0,
                reference_velocity_bps=3.0,
                age_sec=0.5,
                reference_exchange="bitflyer",
            )
        )

        result = asyncio.run(calc.compute("buy", _adapter(), "btc_jpy"))

        assert result.effective_offset_ratio == pytest.approx(config.spread_offset_ratio)

    def test_cancel_reason_in_taxonomy(self) -> None:
        assert CR.CROSS_VENUE_BUY_VETO == "cross_venue_buy_veto"

    def test_config_yaml_roundtrip(self) -> None:
        yaml_cfg = load_yaml_mapping(Path("configs/v460/fill_test.yaml"))
        cross_venue = yaml_cfg["cross_venue_lead_lag"]
        assert cross_venue["buy_protect_enabled"] is False
        assert cross_venue["buy_veto_spread_bps"] == pytest.approx(5.0)
        assert cross_venue["buy_boost_spread_bps"] == pytest.approx(3.0)
        assert cross_venue["buy_offset_boost_factor"] == pytest.approx(1.3)

    def test_fill_record_builder_preserves_buy_offset_mult(self) -> None:
        from ztb.metrics.fill_quality import build_fill_record

        record = build_fill_record(
            cycle_id="cv_buy_1",
            timestamp=1.0,
            side="buy",
            order_price=100.0,
            order_quantity=0.01,
            cross_venue_buy_offset_mult=1.3,
        )

        assert record.cross_venue_buy_offset_mult == pytest.approx(1.3)
