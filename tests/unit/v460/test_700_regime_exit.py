from __future__ import annotations

from scripts.v460.lib.regime_exit_strategy import RegimeExitConfig, RegimeExitTracker
from ztb.metrics.fill_quality import build_skip_fill_record


def test_enabled_false_noop() -> None:
    tracker = RegimeExitTracker(RegimeExitConfig(enabled=False))

    result = tracker.evaluate(regime="trending_down", imbalance=0.5, now=100.0)

    assert result.reason == "disabled"
    assert result.should_escalate_skewing is False


def test_buy_fill_tracking_and_window_expiry() -> None:
    tracker = RegimeExitTracker(
        RegimeExitConfig(enabled=True, max_trending_down_buy_fills=1, tracking_window_sec=10.0)
    )
    tracker.record_fill("buy", 0.0)
    tracker.record_fill("buy", 5.0)

    result = tracker.evaluate(regime="trending_down", imbalance=0.2, now=8.0)
    assert result.buy_count_in_window == 2
    assert result.should_escalate_skewing is True

    expired = tracker.evaluate(regime="trending_down", imbalance=0.2, now=20.0)
    assert expired.buy_count_in_window == 0
    assert expired.should_escalate_skewing is False


def test_no_escalation_other_regimes() -> None:
    tracker = RegimeExitTracker(
        RegimeExitConfig(enabled=True, max_trending_down_buy_fills=1, tracking_window_sec=100.0)
    )
    tracker.record_fill("buy", 1.0)
    tracker.record_fill("buy", 2.0)

    result = tracker.evaluate(regime="ranging", imbalance=0.4, now=3.0)

    assert result.reason == "regime_inactive"
    assert result.should_escalate_skewing is False


def test_nfq_trigger_on_high_imbalance() -> None:
    tracker = RegimeExitTracker(
        RegimeExitConfig(
            enabled=True,
            max_trending_down_buy_fills=1,
            tracking_window_sec=100.0,
            escalated_max_factor=0.7,
            nfq_trigger_imbalance=0.3,
        )
    )
    tracker.record_fill("buy", 1.0)
    tracker.record_fill("buy", 2.0)

    result = tracker.evaluate(regime="trending_down", imbalance=0.4, now=3.0)

    assert result.should_escalate_skewing is True
    assert result.should_trigger_nfq is True
    assert result.effective_max_factor == 0.7
    assert result.reason == "nfq"


def test_fill_record_accepts_regime_exit_fields() -> None:
    record = build_skip_fill_record(
        cycle_id="c1",
        timestamp=1.0,
        side="buy",
        order_price=100.0,
        order_quantity=0.1,
        cancel_reason="no_feasible_quote",
        run_id="run",
        git_sha="sha",
        regime_exit_escalated=True,
        regime_exit_buy_count=12,
        regime_exit_reason="nfq",
        regime_exit_triggered_nfq=True,
    )

    assert record.regime_exit_escalated is True
    assert record.regime_exit_buy_count == 12
    assert record.regime_exit_reason == "nfq"
    assert record.regime_exit_triggered_nfq is True
