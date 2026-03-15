from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from scripts.v460.lib import cancel_reasons as CR
from scripts.v460.lib.cross_venue_lead_lag import (
    CrossVenueLeadLagHint,
    VenueMidSnapshot,
    build_reference_adapter,
    compute_cross_venue_lead_lag_hint,
)
from scripts.v460.lib.fast_fill_defense import FastFillDefense, FastFillDefenseConfig
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.fill_cycle_executor import FillCycleExecutorMixin
from scripts.v460.lib.fill_record_builder import FillRecordBuilderMixin
from scripts.v460.lib.maker_price import InfeasibleQuoteError, MakerPriceCalculator
from ztb.metrics.fill_quality import FillRecord


@dataclass
class _OB:
    timestamp: float
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
    exchange: str


class _RegistryStub:
    def __init__(self, adapter: object | None = None) -> None:
        self.adapter = adapter
        self.calls: list[tuple[str, bool]] = []

    def has_broker(self, name: str) -> bool:
        return name == "bitflyer"

    def create_adapter(self, name: str, *, dry_run: bool) -> object:
        self.calls.append((name, dry_run))
        return self.adapter if self.adapter is not None else object()


class _OrderbookAdapter:
    def __init__(self, orderbooks: list[_OB]) -> None:
        self._orderbooks = list(orderbooks)

    async def get_orderbook(self, symbol: str, depth: int = 1) -> _OB:
        assert symbol == "btc_jpy"
        assert depth == 1
        if len(self._orderbooks) > 1:
            return self._orderbooks.pop(0)
        return self._orderbooks[0]


class _FailingOrderbookAdapter:
    async def get_orderbook(self, symbol: str, depth: int = 1) -> _OB:
        raise RuntimeError("reference orderbook unavailable")


class _DummyExecutor(FillCycleExecutorMixin):
    pass


class _DummyFillRecordBuilder(FillRecordBuilderMixin):
    pass


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


def _make_hint(
    *,
    adverse_side: str = "sell",
    spread_bps: float = 8.0,
    velocity_bps: float = 3.0,
    age_sec: float = 0.5,
    direction: str = "up",
) -> CrossVenueLeadLagHint:
    return CrossVenueLeadLagHint(
        direction=direction,
        adverse_side=adverse_side,
        spread_bps=spread_bps,
        reference_velocity_bps=velocity_bps,
        age_sec=age_sec,
        reference_exchange="bitflyer",
    )


class TestCrossVenueLeadLagHelper:
    def test_upward_move_suppresses_sell(self) -> None:
        hint = compute_cross_venue_lead_lag_hint(
            local_snapshot=VenueMidSnapshot("coincheck", 100.0, 100.0),
            reference_snapshot=VenueMidSnapshot("bitflyer", 100.6, 100.5),
            previous_reference_snapshot=VenueMidSnapshot("bitflyer", 100.2, 99.5),
            max_age_sec=3.0,
            spread_bps_threshold=2.0,
            velocity_bps_threshold=1.0,
        )
        assert hint is not None
        assert hint.direction == "up"
        assert hint.adverse_side == "sell"

    def test_disagreeing_signs_return_none(self) -> None:
        hint = compute_cross_venue_lead_lag_hint(
            local_snapshot=VenueMidSnapshot("coincheck", 100.0, 100.0),
            reference_snapshot=VenueMidSnapshot("bitflyer", 100.6, 100.5),
            previous_reference_snapshot=VenueMidSnapshot("bitflyer", 100.8, 99.5),
            max_age_sec=3.0,
            spread_bps_threshold=2.0,
            velocity_bps_threshold=1.0,
        )
        assert hint is None

    def test_stale_hint_returns_none(self) -> None:
        hint = compute_cross_venue_lead_lag_hint(
            local_snapshot=VenueMidSnapshot("coincheck", 100.0, 100.0),
            reference_snapshot=VenueMidSnapshot("bitflyer", 100.6, 104.5),
            previous_reference_snapshot=VenueMidSnapshot("bitflyer", 100.2, 103.5),
            max_age_sec=3.0,
            spread_bps_threshold=2.0,
            velocity_bps_threshold=1.0,
        )
        assert hint is None

    def test_build_reference_adapter_reuses_primary_dry_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        registry = _RegistryStub(adapter="ref-adapter")
        monkeypatch.setattr(
            "ztb.trading.live.registry.broker_registry.get_broker_registry",
            lambda: registry,
        )
        primary = SimpleNamespace(dry_run=False)
        result = build_reference_adapter("bitflyer", primary_adapter=primary)
        assert result == "ref-adapter"
        assert registry.calls == [("bitflyer", False)]


class TestCrossVenueLeadLagGuard:
    def test_adverse_side_gets_offset_boost(self) -> None:
        config = FillTestConfig(
            min_spread_jpy=1.0,
            cross_venue_lead_lag_enabled=True,
            cross_venue_lead_lag_offset_boost=1.5,
        )
        baseline = _make_calc(config)
        guarded = _make_calc(config)
        adapter = _OrderbookAdapter([
            _OB(100.0, [(10_000_000.0, 1.0)], [(10_002_000.0, 1.0)], "coincheck"),
        ])
        base_result = asyncio.run(baseline.compute("sell", adapter, "btc_jpy"))
        guarded.set_cross_venue_lead_lag_hint(_make_hint())
        guarded_result = asyncio.run(guarded.compute("sell", adapter, "btc_jpy"))
        assert guarded_result.effective_offset_ratio > base_result.effective_offset_ratio

    def test_safe_side_is_unchanged(self) -> None:
        config = FillTestConfig(
            min_spread_jpy=1.0,
            cross_venue_lead_lag_enabled=True,
            cross_venue_lead_lag_offset_boost=1.5,
        )
        baseline = _make_calc(config)
        guarded = _make_calc(config)
        adapter = _OrderbookAdapter([
            _OB(100.0, [(10_000_000.0, 1.0)], [(10_002_000.0, 1.0)], "coincheck"),
        ])
        base_result = asyncio.run(baseline.compute("buy", adapter, "btc_jpy"))
        guarded.set_cross_venue_lead_lag_hint(_make_hint())
        guarded_result = asyncio.run(guarded.compute("buy", adapter, "btc_jpy"))
        assert guarded_result.effective_offset_ratio == pytest.approx(
            base_result.effective_offset_ratio
        )

    def test_veto_raises_infeasible_quote(self) -> None:
        config = FillTestConfig(
            min_spread_jpy=1.0,
            cross_venue_lead_lag_enabled=True,
            cross_venue_lead_lag_veto_enabled=True,
            cross_venue_lead_lag_veto_threshold_bps=6.0,
        )
        calc = _make_calc(config)
        calc.set_cross_venue_lead_lag_hint(
            _make_hint(spread_bps=8.0, velocity_bps=2.0)
        )
        adapter = _OrderbookAdapter([
            _OB(100.0, [(10_000_000.0, 1.0)], [(10_002_000.0, 1.0)], "coincheck"),
        ])
        with pytest.raises(InfeasibleQuoteError) as exc_info:
            asyncio.run(calc.compute("sell", adapter, "btc_jpy"))
        assert exc_info.value.reason == CR.CROSS_VENUE_LEAD_LAG_VETO


class TestCrossVenueLeadLagExecutorInjection:
    def test_hint_is_injected_from_reference_orderbook(self) -> None:
        config = FillTestConfig(
            symbol="btc_jpy",
            cross_venue_lead_lag_enabled=True,
            cross_venue_lead_lag_spread_bps_threshold=2.0,
            cross_venue_lead_lag_velocity_bps_threshold=1.0,
        )
        runner = _DummyExecutor()
        runner.config = config
        runner.adapter = SimpleNamespace()
        runner._maker_price = _make_calc(config)
        runner._maker_price._last_ob_snapshot = _OB(
            timestamp=100.0,
            bids=[(100.0, 1.0)],
            asks=[(101.0, 1.0)],
            exchange="coincheck",
        )
        runner._cross_venue_reference_adapter = _OrderbookAdapter([
            _OB(100.5, [(101.0, 1.0)], [(102.0, 1.0)], "bitflyer"),
        ])
        runner._cross_venue_prev_reference_snapshot = VenueMidSnapshot(
            exchange="bitflyer",
            mid_price=100.8,
            timestamp=99.5,
        )

        asyncio.run(runner._update_cross_venue_lead_lag_hint())

        hint = runner._maker_price._cross_venue_lead_lag_hint
        assert hint is not None
        assert hint.adverse_side == "sell"
        assert runner._cross_venue_prev_reference_snapshot is not None

    def test_reference_failure_fails_open(self) -> None:
        config = FillTestConfig(
            symbol="btc_jpy",
            cross_venue_lead_lag_enabled=True,
        )
        runner = _DummyExecutor()
        runner.config = config
        runner.adapter = SimpleNamespace()
        runner._maker_price = _make_calc(config)
        runner._maker_price._last_ob_snapshot = _OB(
            timestamp=100.0,
            bids=[(100.0, 1.0)],
            asks=[(101.0, 1.0)],
            exchange="coincheck",
        )
        runner._cross_venue_reference_adapter = _FailingOrderbookAdapter()
        runner._cross_venue_prev_reference_snapshot = None

        asyncio.run(runner._update_cross_venue_lead_lag_hint())

        assert runner._maker_price._cross_venue_lead_lag_hint is None


class TestCrossVenueLeadLagFillRecordObservability:
    def test_fill_record_round_trip_preserves_cross_venue_fields(self) -> None:
        record = FillRecord(
            cycle_id="cv-1",
            timestamp=100.0,
            side="sell",
            order_price=1_000_000.0,
            order_quantity=0.01,
            cross_venue_reference_exchange="bitflyer",
            cross_venue_lead_lag_direction="up",
            cross_venue_lead_lag_adverse_side="sell",
            cross_venue_lead_lag_spread_bps=8.5,
            cross_venue_lead_lag_velocity_bps=2.25,
            cross_venue_lead_lag_age_sec=0.4,
            cross_venue_lead_lag_applied=True,
            cross_venue_lead_lag_vetoed=False,
        )
        restored = FillRecord.from_dict(record.to_dict())
        assert restored.cross_venue_reference_exchange == "bitflyer"
        assert restored.cross_venue_lead_lag_direction == "up"
        assert restored.cross_venue_lead_lag_adverse_side == "sell"
        assert restored.cross_venue_lead_lag_spread_bps == pytest.approx(8.5)
        assert restored.cross_venue_lead_lag_velocity_bps == pytest.approx(2.25)
        assert restored.cross_venue_lead_lag_age_sec == pytest.approx(0.4)
        assert restored.cross_venue_lead_lag_applied is True
        assert restored.cross_venue_lead_lag_vetoed is False

    def test_fill_record_builder_exports_cross_venue_fields(self) -> None:
        runner = _DummyFillRecordBuilder()
        runner.config = FillTestConfig(cross_venue_lead_lag_enabled=True)
        runner._maker_price = SimpleNamespace(
            _cross_venue_lead_lag_hint=_make_hint(
                adverse_side="sell",
                spread_bps=7.0,
                velocity_bps=2.5,
                age_sec=0.3,
            ),
            _cross_venue_lead_lag_vetoed=False,
        )

        fields = runner._build_fill_cross_venue_fields(side="sell")

        assert fields["cross_venue_reference_exchange"] == "bitflyer"
        assert fields["cross_venue_lead_lag_direction"] == "up"
        assert fields["cross_venue_lead_lag_adverse_side"] == "sell"
        assert fields["cross_venue_lead_lag_spread_bps"] == pytest.approx(7.0)
        assert fields["cross_venue_lead_lag_velocity_bps"] == pytest.approx(2.5)
        assert fields["cross_venue_lead_lag_age_sec"] == pytest.approx(0.3)
        assert fields["cross_venue_lead_lag_applied"] is True
        assert fields["cross_venue_lead_lag_vetoed"] is False

    def test_fill_record_builder_marks_safe_side_as_not_applied(self) -> None:
        runner = _DummyFillRecordBuilder()
        runner.config = FillTestConfig(cross_venue_lead_lag_enabled=True)
        runner._maker_price = SimpleNamespace(
            _cross_venue_lead_lag_hint=_make_hint(adverse_side="sell"),
            _cross_venue_lead_lag_vetoed=False,
        )

        fields = runner._build_fill_cross_venue_fields(side="buy")

        assert fields["cross_venue_lead_lag_applied"] is False
        assert fields["cross_venue_lead_lag_vetoed"] is False
