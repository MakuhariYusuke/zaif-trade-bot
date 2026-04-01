from __future__ import annotations

from scripts.v460.lib.fill_config import FillTestConfig
from ztb.types.common import ConfigSection
from tests.unit.v460._fill_test_source import (
    FILL_CYCLE_EXECUTOR,
    SKIP_GATE_EVALUATOR,
    read_source_text,
)


class TestTimeoutOverrideResolution:
    def test_fill_test_yaml_exposes_regime_timeout_overrides(
        self,
        v460_fill_test_yaml_base: ConfigSection,
        v460_fill_test_config_base: FillTestConfig,
    ) -> None:
        regime_cfg = v460_fill_test_yaml_base["regime"]
        timeout_cfg = regime_cfg["timeout_overrides"]

        assert timeout_cfg["strong_up"]["sell"] == 20.0
        assert timeout_cfg["strong_down"]["buy"] == 30.0
        assert v460_fill_test_config_base.regime_timeout_overrides == {
            "strong_up": {"sell": 20.0, "buy": 120.0},
            "strong_down": {"sell": 90.0, "buy": 30.0},
        }

    def test_regime_timeout_override_sell_strong_up(self) -> None:
        cfg = FillTestConfig(
            order_timeout_sec=90.0,
            order_timeout_sec_sell=75.0,
            regime_timeout_overrides={"strong_up": {"sell": 20.0}},
        )

        timeout_sec, reason = cfg.get_timeout_with_reason("sell", "strong_up")

        assert timeout_sec == 20.0
        assert reason == "regime_strong_up_sell"

    def test_regime_timeout_override_buy_strong_down(self) -> None:
        cfg = FillTestConfig(
            order_timeout_sec=90.0,
            regime_timeout_overrides={"strong_down": {"buy": 30.0}},
        )

        timeout_sec, reason = cfg.get_timeout_with_reason("buy", "strong_down")

        assert timeout_sec == 30.0
        assert reason == "regime_strong_down_buy"

    def test_legacy_macro_sell_timeout_remains_fallback(self) -> None:
        cfg = FillTestConfig(
            order_timeout_sec=90.0,
            order_timeout_sec_sell=75.0,
            macro_sell_timeout_strong_up=6.0,
            macro_sell_timeout_weak_up=12.0,
        )

        timeout_sec, reason = cfg.get_timeout_with_reason("sell", "strong_up")
        assert timeout_sec == 6.0
        assert reason == "legacy_macro_strong_up_sell"

        timeout_sec, reason = cfg.get_timeout_with_reason("sell", "weak_up")
        assert timeout_sec == 12.0
        assert reason == "legacy_macro_weak_up_sell"

    def test_neutral_regime_falls_back_to_side_default(self) -> None:
        cfg = FillTestConfig(
            order_timeout_sec=90.0,
            order_timeout_sec_sell=75.0,
        )

        timeout_sec, reason = cfg.get_timeout_with_reason("sell", "neutral")

        assert timeout_sec == 75.0
        assert reason == "sell_default"


class TestDecisionTraceSourceAudit:
    def test_fill_cycle_executor_logs_timeout_and_outcome_with_trace_id(self) -> None:
        source = read_source_text(FILL_CYCLE_EXECUTOR)

        assert "[dt=%s] timeout: value=%.1fs reason=%s regime=%s side=%s" in source
        assert "[dt=%s] outcome: filled=%s reason=%s timeout_reason=%s" in source
        assert "decision_trace_id = self._new_decision_trace_id()" in source

    def test_skip_gate_evaluator_logs_with_trace_id(self) -> None:
        source = read_source_text(SKIP_GATE_EVALUATOR)

        assert "[dt=%s] [skip_gate]" in source
        assert "decision_trace_id: str | None = None" in source
