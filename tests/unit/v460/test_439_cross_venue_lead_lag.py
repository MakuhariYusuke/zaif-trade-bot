from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.v460.lib import cancel_reasons as CR
from scripts.v460.lib.cross_venue_lead_lag import (
    CrossVenueEMAState,
    CrossVenueLeadLagHint,
    VenueMidSnapshot,
    build_reference_adapter,
    build_cross_venue_event_details,
    build_cross_venue_fill_fields,
    compute_cross_venue_lead_lag_hint,
    update_cross_venue_ema,
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


@dataclass
class _CrossVenueState:
    cross_venue_lead_lag_hint: CrossVenueLeadLagHint | None
    cross_venue_lead_lag_vetoed: bool = False


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
    def test_build_event_details(self) -> None:
        hint = _make_hint(spread_bps=7.5, velocity_bps=2.0, age_sec=0.4)
        details = build_cross_venue_event_details(hint)
        assert details["reference_exchange"] == "bitflyer"
        assert details["direction"] == "up"
        assert details["adverse_side"] == "sell"
        assert details["spread_bps"] == 7.5
        assert details["velocity_bps"] == 2.0
        assert details["age_sec"] == 0.4
        # 442# 新フィールド
        assert "microprice_spread_bps" in details
        assert "depth_imbalance" in details

    def test_build_fill_fields_disabled_returns_empty_payload(self) -> None:
        fields = build_cross_venue_fill_fields(
            enabled=False,
            hint=None,
            side="buy",
            vetoed=False,
        )
        assert fields["cross_venue_reference_exchange"] is None
        assert fields["cross_venue_lead_lag_direction"] is None
        assert fields["cross_venue_lead_lag_applied"] is None
        assert fields["cross_venue_lead_lag_vetoed"] is None
        assert fields["cross_venue_microprice_spread_bps"] is None
        assert fields["cross_venue_depth_imbalance"] is None

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
            results_dir="results/v460/fill_test",
        )
        runner = _DummyExecutor()
        runner.config = config
        runner.adapter = SimpleNamespace()
        runner._run_id = "run-439"
        runner._git_sha = "abc1234"
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
        runner._cross_venue_ema_state = None

        with patch("scripts.v460.lib.event_logger.log_event") as mock_log_event:
            asyncio.run(runner._update_cross_venue_lead_lag_hint())

        hint = runner._maker_price.cross_venue_lead_lag_hint
        assert hint is not None
        assert hint.adverse_side == "sell"
        assert runner._cross_venue_prev_reference_snapshot is not None
        mock_log_event.assert_called_once()
        assert mock_log_event.call_args.args == (
            "cross_venue_hint",
            "results/v460/fill_test",
        )
        assert mock_log_event.call_args.kwargs["run_id"] == "run-439"
        assert mock_log_event.call_args.kwargs["git_sha"] == "abc1234"
        assert mock_log_event.call_args.kwargs["details"]["adverse_side"] == "sell"

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
        runner._cross_venue_ema_state = None

        asyncio.run(runner._update_cross_venue_lead_lag_hint())

        assert runner._maker_price.cross_venue_lead_lag_hint is None


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
        runner._maker_price = _CrossVenueState(
            cross_venue_lead_lag_hint=_make_hint(
                adverse_side="sell",
                spread_bps=7.0,
                velocity_bps=2.5,
                age_sec=0.3,
            ),
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
        runner._maker_price = _CrossVenueState(
            cross_venue_lead_lag_hint=_make_hint(adverse_side="sell"),
        )

        fields = runner._build_fill_cross_venue_fields(side="buy")

        assert fields["cross_venue_lead_lag_applied"] is False
        assert fields["cross_venue_lead_lag_vetoed"] is False

    def test_fill_record_builder_returns_empty_fields_when_disabled(self) -> None:
        runner = _DummyFillRecordBuilder()
        runner.config = FillTestConfig(cross_venue_lead_lag_enabled=False)
        runner._maker_price = _CrossVenueState(cross_venue_lead_lag_hint=None)

        fields = runner._build_fill_cross_venue_fields(side="buy")

        assert fields["cross_venue_reference_exchange"] is None
        assert fields["cross_venue_lead_lag_applied"] is None
        assert fields["cross_venue_lead_lag_vetoed"] is None
        assert fields["cross_venue_microprice_spread_bps"] is None
        assert fields["cross_venue_depth_imbalance"] is None


# ---- 445# EMA + Confidence Scoring Tests ----


class TestUpdateCrossVenueEMA:
    """445# update_cross_venue_ema() のユニットテスト."""

    def test_first_call_initializes_state(self) -> None:
        state = update_cross_venue_ema(None, ref_mid=100.0, spread_bps=2.0, timestamp=1.0, alpha=0.3)
        assert state.ema_ref_mid == 100.0
        assert state.ema_spread_bps == 2.0
        assert state.n_updates == 1

    def test_subsequent_call_blends_ema(self) -> None:
        s1 = update_cross_venue_ema(None, ref_mid=100.0, spread_bps=2.0, timestamp=1.0, alpha=0.3)
        s2 = update_cross_venue_ema(s1, ref_mid=110.0, spread_bps=5.0, timestamp=2.0, alpha=0.3)
        assert s2.ema_ref_mid == pytest.approx(0.3 * 110.0 + 0.7 * 100.0)
        assert s2.ema_spread_bps == pytest.approx(0.3 * 5.0 + 0.7 * 2.0)
        assert s2.n_updates == 2


class TestConfidenceModeCompute:
    """445# confidence mode (ema_spread_bps 指定時) のテスト."""

    _LOCAL = VenueMidSnapshot("coincheck", 100.0, 100.0)

    def test_sign_disagree_fires_with_reduced_confidence(self) -> None:
        """Legacy mode では sign_disagree で None 扱いだった状況が、
        confidence mode では hint 発火 (confidence<1.0) になることを確認."""
        hint = compute_cross_venue_lead_lag_hint(
            local_snapshot=self._LOCAL,
            reference_snapshot=VenueMidSnapshot("bitflyer", 100.6, 100.5),
            previous_reference_snapshot=VenueMidSnapshot("bitflyer", 100.8, 99.5),
            max_age_sec=3.0,
            spread_bps_threshold=2.0,
            velocity_bps_threshold=1.0,
            ema_spread_bps=60.0,   # large positive EMA
            min_confidence=0.0,
            confidence_reference_spread_bps=3.0,
        )
        assert hint is not None
        assert hint.direction == "up"
        assert hint.adverse_side == "sell"
        # velocity disagrees → vel_factor=0.5, base_conf=1.0 → confidence=0.5
        assert hint.confidence == pytest.approx(0.5)

    def test_velocity_agree_gives_full_confidence(self) -> None:
        hint = compute_cross_venue_lead_lag_hint(
            local_snapshot=self._LOCAL,
            reference_snapshot=VenueMidSnapshot("bitflyer", 100.6, 100.5),
            previous_reference_snapshot=VenueMidSnapshot("bitflyer", 100.2, 99.5),
            max_age_sec=3.0,
            spread_bps_threshold=2.0,
            velocity_bps_threshold=1.0,
            ema_spread_bps=60.0,
            min_confidence=0.0,
            confidence_reference_spread_bps=3.0,
        )
        assert hint is not None
        assert hint.confidence == pytest.approx(1.0)

    def test_small_ema_spread_returns_none(self) -> None:
        hint = compute_cross_venue_lead_lag_hint(
            local_snapshot=self._LOCAL,
            reference_snapshot=VenueMidSnapshot("bitflyer", 100.6, 100.5),
            previous_reference_snapshot=VenueMidSnapshot("bitflyer", 100.2, 99.5),
            max_age_sec=3.0,
            spread_bps_threshold=2.0,
            velocity_bps_threshold=1.0,
            ema_spread_bps=0.5,  # below threshold
            min_confidence=0.0,
            confidence_reference_spread_bps=3.0,
        )
        assert hint is None

    def test_min_confidence_gate(self) -> None:
        """confidence < min_confidence → None."""
        hint = compute_cross_venue_lead_lag_hint(
            local_snapshot=self._LOCAL,
            reference_snapshot=VenueMidSnapshot("bitflyer", 100.6, 100.5),
            previous_reference_snapshot=VenueMidSnapshot("bitflyer", 100.8, 99.5),
            max_age_sec=3.0,
            spread_bps_threshold=2.0,
            velocity_bps_threshold=1.0,
            ema_spread_bps=60.0,
            min_confidence=0.6,  # 0.5 < 0.6 → filtered
            confidence_reference_spread_bps=3.0,
        )
        assert hint is None

    def test_small_ema_spread_reduces_base_confidence(self) -> None:
        """ema_spread が reference の 50% → base_conf ≈ 0.5."""
        hint = compute_cross_venue_lead_lag_hint(
            local_snapshot=self._LOCAL,
            reference_snapshot=VenueMidSnapshot("bitflyer", 100.6, 100.5),
            previous_reference_snapshot=VenueMidSnapshot("bitflyer", 100.2, 99.5),
            max_age_sec=3.0,
            spread_bps_threshold=1.0,
            velocity_bps_threshold=1.0,
            ema_spread_bps=1.5,  # 1.5 / 3.0 = 0.5, clamped to max(0.33, 0.5) = 0.5
            min_confidence=0.0,
            confidence_reference_spread_bps=3.0,
        )
        assert hint is not None
        # velocity agrees → vel_factor=1.0 → confidence=0.5
        assert hint.confidence == pytest.approx(0.5)

    def test_microprice_disagree_halves_confidence(self) -> None:
        """microprice 方向が EMA spread 方向と逆 → mp_factor=0.5."""
        hint = compute_cross_venue_lead_lag_hint(
            local_snapshot=VenueMidSnapshot("coincheck", 100.0, 100.0, microprice=100.0),
            reference_snapshot=VenueMidSnapshot("bitflyer", 100.6, 100.5, microprice=99.5),
            previous_reference_snapshot=VenueMidSnapshot("bitflyer", 100.2, 99.5),
            max_age_sec=3.0,
            spread_bps_threshold=2.0,
            velocity_bps_threshold=1.0,
            ema_spread_bps=60.0,  # direction=up
            min_confidence=0.0,
            confidence_reference_spread_bps=3.0,
        )
        assert hint is not None
        # microprice_spread_bps = (99.5 - 100.0)/100.0 * 10000 = -50bps (disagrees with "up")
        # base=1.0, vel agrees=1.0, mp_factor=0.5 → confidence=0.5
        assert hint.confidence == pytest.approx(0.5)

    def test_direction_from_ema_not_point_spread(self) -> None:
        """EMA spread が負 → direction=down (point spread が正でも)."""
        hint = compute_cross_venue_lead_lag_hint(
            local_snapshot=self._LOCAL,
            reference_snapshot=VenueMidSnapshot("bitflyer", 100.6, 100.5),
            previous_reference_snapshot=VenueMidSnapshot("bitflyer", 100.2, 99.5),
            max_age_sec=3.0,
            spread_bps_threshold=2.0,
            velocity_bps_threshold=1.0,
            ema_spread_bps=-3.0,  # negative → down
            min_confidence=0.0,
            confidence_reference_spread_bps=3.0,
        )
        assert hint is not None
        assert hint.direction == "down"
        assert hint.adverse_side == "buy"

    def test_confidence_in_event_details(self) -> None:
        hint = _make_hint()
        details = build_cross_venue_event_details(hint)
        assert "confidence" in details
        assert details["confidence"] == 1.0

    def test_confidence_in_fill_fields(self) -> None:
        fields = build_cross_venue_fill_fields(
            enabled=True, hint=_make_hint(), side="sell", vetoed=False,
        )
        assert "cross_venue_confidence" in fields
        assert fields["cross_venue_confidence"] == 1.0

    def test_fill_fields_disabled_includes_confidence_none(self) -> None:
        fields = build_cross_venue_fill_fields(
            enabled=False, hint=None, side="buy", vetoed=False,
        )
        assert fields["cross_venue_confidence"] is None


class TestConfidenceProportionalBoost:
    """445# confidence-proportional boost のテスト."""

    def test_full_confidence_gets_full_boost(self) -> None:
        config = FillTestConfig(
            min_spread_jpy=1.0,
            cross_venue_lead_lag_enabled=True,
            cross_venue_lead_lag_offset_boost=1.5,
        )
        calc = _make_calc(config)
        adapter = _OrderbookAdapter([
            _OB(100.0, [(10_000_000.0, 1.0)], [(10_002_000.0, 1.0)], "coincheck"),
        ])
        calc.set_cross_venue_lead_lag_hint(
            _make_hint(spread_bps=8.0, velocity_bps=3.0)  # confidence=1.0
        )
        result = asyncio.run(calc.compute("sell", adapter, "btc_jpy"))
        full_ratio = result.effective_offset_ratio

        # Same config without hint (baseline)
        calc2 = _make_calc(config)
        baseline = asyncio.run(calc2.compute("sell", adapter, "btc_jpy"))

        # full confidence → full boost (1.5x)
        assert full_ratio == pytest.approx(baseline.effective_offset_ratio * 1.5, rel=0.01)

    def test_half_confidence_gets_half_boost(self) -> None:
        config = FillTestConfig(
            min_spread_jpy=1.0,
            cross_venue_lead_lag_enabled=True,
            cross_venue_lead_lag_offset_boost=1.5,
        )
        calc = _make_calc(config)
        adapter = _OrderbookAdapter([
            _OB(100.0, [(10_000_000.0, 1.0)], [(10_002_000.0, 1.0)], "coincheck"),
        ])
        hint = CrossVenueLeadLagHint(
            direction="up", adverse_side="sell", spread_bps=8.0,
            reference_velocity_bps=3.0, age_sec=0.5, reference_exchange="bitflyer",
            confidence=0.5,
        )
        calc.set_cross_venue_lead_lag_hint(hint)
        result = asyncio.run(calc.compute("sell", adapter, "btc_jpy"))

        calc2 = _make_calc(config)
        baseline = asyncio.run(calc2.compute("sell", adapter, "btc_jpy"))

        # confidence=0.5 → boost = 1 + (1.5-1)*0.5 = 1.25
        expected_ratio = baseline.effective_offset_ratio * 1.25
        assert result.effective_offset_ratio == pytest.approx(expected_ratio, rel=0.01)


# ---- 449# DRY / config 拡張テスト ----


class TestPrecomputedSpreadBps:
    """449# precomputed_point_spread_bps パラメータの動作確認."""

    _LOCAL = VenueMidSnapshot("coincheck", 100.0, 100.0)

    def test_precomputed_value_used_instead_of_recompute(self) -> None:
        """caller が渡した point_spread_bps が関数内部で再利用されることを確認."""
        ref = VenueMidSnapshot("bitflyer", 100.6, 100.5)
        prev = VenueMidSnapshot("bitflyer", 100.2, 99.5)
        # point_spread_bps を明示的に指定
        hint = compute_cross_venue_lead_lag_hint(
            local_snapshot=self._LOCAL,
            reference_snapshot=ref,
            previous_reference_snapshot=prev,
            max_age_sec=3.0,
            spread_bps_threshold=1.0,
            velocity_bps_threshold=1.0,
            precomputed_point_spread_bps=42.0,  # 実際の値とは異なる値を注入
        )
        assert hint is not None
        # legacy mode: spread_bps = precomputed value
        assert hint.spread_bps == pytest.approx(42.0)

    def test_precomputed_none_falls_back_to_calculation(self) -> None:
        """precomputed=None の場合、従来通り内部計算が行われる."""
        ref = VenueMidSnapshot("bitflyer", 100.6, 100.5)
        prev = VenueMidSnapshot("bitflyer", 100.2, 99.5)
        hint_auto = compute_cross_venue_lead_lag_hint(
            local_snapshot=self._LOCAL,
            reference_snapshot=ref,
            previous_reference_snapshot=prev,
            max_age_sec=3.0,
            spread_bps_threshold=1.0,
            velocity_bps_threshold=1.0,
        )
        hint_explicit = compute_cross_venue_lead_lag_hint(
            local_snapshot=self._LOCAL,
            reference_snapshot=ref,
            previous_reference_snapshot=prev,
            max_age_sec=3.0,
            spread_bps_threshold=1.0,
            velocity_bps_threshold=1.0,
            precomputed_point_spread_bps=None,
        )
        assert hint_auto is not None
        assert hint_explicit is not None
        assert hint_auto.spread_bps == pytest.approx(hint_explicit.spread_bps)


class TestConfidenceFloorParam:
    """449# confidence_floor パラメータの動作確認."""

    _LOCAL = VenueMidSnapshot("coincheck", 100.0, 100.0)

    def test_custom_confidence_floor(self) -> None:
        """confidence_floor=0.5 → base_conf が 0.5 以上にクランプされる."""
        hint = compute_cross_venue_lead_lag_hint(
            local_snapshot=self._LOCAL,
            reference_snapshot=VenueMidSnapshot("bitflyer", 100.6, 100.5),
            previous_reference_snapshot=VenueMidSnapshot("bitflyer", 100.2, 99.5),
            max_age_sec=3.0,
            spread_bps_threshold=1.0,
            velocity_bps_threshold=1.0,
            ema_spread_bps=1.5,  # 1.5 / 3.0 = 0.5, threshold OK
            min_confidence=0.0,
            confidence_reference_spread_bps=3.0,
            confidence_floor=0.6,  # 0.5 < 0.6 → floor にクランプ
        )
        assert hint is not None
        # base_conf = max(0.6, 0.5) = 0.6, vel agrees → 1.0 → confidence = 0.6
        assert hint.confidence == pytest.approx(0.6)

    def test_default_confidence_floor_is_033(self) -> None:
        """デフォルト floor=0.33 の動作確認 (既存動作の後方互換)."""
        hint = compute_cross_venue_lead_lag_hint(
            local_snapshot=self._LOCAL,
            reference_snapshot=VenueMidSnapshot("bitflyer", 100.6, 100.5),
            previous_reference_snapshot=VenueMidSnapshot("bitflyer", 100.2, 99.5),
            max_age_sec=3.0,
            spread_bps_threshold=0.1,
            velocity_bps_threshold=1.0,
            ema_spread_bps=0.5,  # 0.5 / 3.0 = 0.167
            min_confidence=0.0,
            confidence_reference_spread_bps=3.0,
            # confidence_floor=0.33 (default)
        )
        assert hint is not None
        # base_conf = max(0.33, 0.167) = 0.33, vel agrees → 1.0 → confidence = 0.33
        assert hint.confidence == pytest.approx(0.33)


# ---- 450# P0: FillRecord スキーマ統合テスト ----


class TestFillRecordSchemaIncludesNewFields:
    """450# P0: builder が生成する 448# 新フィールドが FillRecord まで到達するか検証.

    450# レビューで指摘された「コードはある → JSONL に落ちない」穴を防ぐ。
    """

    def test_point_spread_bps_round_trip(self) -> None:
        """cross_venue_lead_lag_point_spread_bps が FillRecord→dict→FillRecord で保持される."""
        record = FillRecord(
            cycle_id="schema-1",
            timestamp=100.0,
            side="sell",
            order_price=1_000_000.0,
            order_quantity=0.01,
            cross_venue_lead_lag_point_spread_bps=5.5,
        )
        d = record.to_dict()
        assert d["cross_venue_lead_lag_point_spread_bps"] == pytest.approx(5.5)
        restored = FillRecord.from_dict(d)
        assert restored.cross_venue_lead_lag_point_spread_bps == pytest.approx(5.5)

    def test_cap_hit_and_offsets_round_trip(self) -> None:
        """pre_offset / post_offset / cap_hit が FillRecord round-trip で保持される."""
        record = FillRecord(
            cycle_id="schema-2",
            timestamp=100.0,
            side="sell",
            order_price=1_000_000.0,
            order_quantity=0.01,
            cross_venue_lead_lag_pre_offset=0.15,
            cross_venue_lead_lag_post_offset=0.15,
            cross_venue_lead_lag_cap_hit=True,
        )
        d = record.to_dict()
        assert d["cross_venue_lead_lag_pre_offset"] == pytest.approx(0.15)
        assert d["cross_venue_lead_lag_post_offset"] == pytest.approx(0.15)
        assert d["cross_venue_lead_lag_cap_hit"] is True
        restored = FillRecord.from_dict(d)
        assert restored.cross_venue_lead_lag_pre_offset == pytest.approx(0.15)
        assert restored.cross_venue_lead_lag_post_offset == pytest.approx(0.15)
        assert restored.cross_venue_lead_lag_cap_hit is True

    def test_builder_fields_accepted_by_fill_record(self) -> None:
        """fill_record_builder が生成する dict が FillRecord に通ることを確認."""
        runner = _DummyFillRecordBuilder()
        runner.config = FillTestConfig(cross_venue_lead_lag_enabled=True)
        hint = _make_hint(adverse_side="sell", spread_bps=7.0)
        state = _CrossVenueState(cross_venue_lead_lag_hint=hint)
        # 448# no-op 可視化フィールドをスタブに追加
        state._cross_venue_lead_lag_pre_offset = 0.20  # type: ignore[attr-defined]
        state._cross_venue_lead_lag_post_offset = 0.25  # type: ignore[attr-defined]
        state._cross_venue_lead_lag_cap_hit = False  # type: ignore[attr-defined]
        runner._maker_price = state

        fields = runner._build_fill_cross_venue_fields(side="sell")

        # 全フィールドが FillRecord のフィールド名と一致する (sanitize で落ちない)
        from ztb.metrics.fill_quality import _FILL_RECORD_FIELD_NAMES
        for key in fields:
            assert key in _FILL_RECORD_FIELD_NAMES, (
                f"builder field '{key}' is NOT in FillRecord schema — "
                f"it will be silently dropped by _sanitize_fill_record_fields"
            )
