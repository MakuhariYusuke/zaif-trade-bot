from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from scripts.v460.lib.config_hot_reload import ConfigHotReloader
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.fill_cycle_executor import FillCycleExecutorMixin


class TestTimeoutPriorityChain:
    def test_regime_override_highest_priority(self) -> None:
        config = FillTestConfig(
            order_timeout_sec=30.0,
            order_timeout_sec_sell=20.0,
            macro_sell_timeout_strong_up=10.0,
            regime_timeout_overrides={"strong_up": {"sell": 5.0}},
        )

        timeout, reason = config.get_timeout_with_reason("sell", "strong_up")

        assert timeout == 5.0
        assert reason == "regime_override_strong_up_sell"

    def test_legacy_macro_sell_second_priority(self) -> None:
        config = FillTestConfig(
            order_timeout_sec=30.0,
            macro_sell_timeout_strong_up=10.0,
            regime_timeout_overrides={},
        )

        timeout, reason = config.get_timeout_with_reason("sell", "strong_up")

        assert timeout == 10.0
        assert reason == "legacy_macro_strong_up_sell"

    def test_side_specific_third_priority(self) -> None:
        config = FillTestConfig(
            order_timeout_sec=30.0,
            order_timeout_sec_sell=20.0,
        )

        timeout, reason = config.get_timeout_with_reason("sell", "ranging")

        assert timeout == 20.0
        assert reason == "sell_default"

    def test_global_fallback(self) -> None:
        config = FillTestConfig(order_timeout_sec=30.0)

        timeout, reason = config.get_timeout_with_reason("buy", "ranging")

        assert timeout == 30.0
        assert reason == "base_default"

    def test_buy_side_ignores_macro_sell_timeout(self) -> None:
        config = FillTestConfig(
            order_timeout_sec=30.0,
            macro_sell_timeout_strong_up=10.0,
        )

        timeout, reason = config.get_timeout_with_reason("buy", "strong_up")

        assert timeout == 30.0
        assert reason == "base_default"

    def test_override_buy_in_strong_down(self) -> None:
        config = FillTestConfig(
            order_timeout_sec=30.0,
            regime_timeout_overrides={"strong_down": {"buy": 3.0}},
        )

        timeout, reason = config.get_timeout_with_reason("buy", "strong_down")

        assert timeout == 3.0
        assert reason == "regime_override_strong_down_buy"

    def test_override_missing_side_falls_through(self) -> None:
        config = FillTestConfig(
            order_timeout_sec=30.0,
            regime_timeout_overrides={"strong_up": {"buy": 5.0}},
        )

        timeout, reason = config.get_timeout_with_reason("sell", "strong_up")

        assert timeout == 30.0
        assert reason == "base_default"

    def test_none_macro_trend_uses_fallback(self) -> None:
        config = FillTestConfig(
            order_timeout_sec=30.0,
            regime_timeout_overrides={"strong_up": {"sell": 5.0}},
        )

        timeout, reason = config.get_timeout_with_reason("sell", None)

        assert timeout == 30.0
        assert reason == "base_default"


class TestTimeoutRegimeNames:
    @pytest.mark.parametrize("regime", ["strong_up", "weak_up", "ranging", "weak_down", "strong_down"])
    @pytest.mark.parametrize("side", ["buy", "sell"])
    def test_all_regime_side_combinations(self, regime: str, side: str) -> None:
        config = FillTestConfig(
            order_timeout_sec=30.0,
            regime_timeout_overrides={regime: {side: 7.0}},
        )

        timeout, reason = config.get_timeout_with_reason(side, regime)

        assert timeout == 7.0
        assert reason == f"regime_override_{regime}_{side}"


class TestTimeoutYamlParsing:
    def test_parse_valid_overrides(self) -> None:
        cfg = FillTestConfig.from_yaml(
            {
                "regime_timeout_overrides": {
                    "Strong_Up": {"SELL": 5.0},
                    "strong_down": {"buy": 7.0},
                }
            }
        )

        assert cfg.regime_timeout_overrides == {
            "strong_up": {"sell": 5.0},
            "strong_down": {"buy": 7.0},
        }

    def test_parse_empty_overrides(self) -> None:
        cfg = FillTestConfig.from_yaml({"regime_timeout_overrides": {}})
        assert cfg.regime_timeout_overrides == {}

    def test_parse_nested_structure_from_regime_section(self) -> None:
        cfg = FillTestConfig.from_yaml(
            {
                "regime": {
                    "timeout_overrides": {
                        "strong_up": {"sell": 20.0},
                        "strong_down": {"buy": 30.0},
                    }
                }
            }
        )
        assert cfg.regime_timeout_overrides == {
            "strong_up": {"sell": 20.0},
            "strong_down": {"buy": 30.0},
        }

    def test_invalid_timeout_value_raises(self) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            FillTestConfig(regime_timeout_overrides={"strong_up": {"sell": -1.0}})


class _TimeoutRunner(FillCycleExecutorMixin):
    def __init__(self, config: FillTestConfig, macro_trend: str | None) -> None:
        self.config = config
        self._last_macro_trend = macro_trend


class TestResolveCycleTimeoutPolicy:
    def test_macro_trend_from_runner_is_used(self) -> None:
        runner = _TimeoutRunner(
            FillTestConfig(
                order_timeout_sec=30.0,
                regime_timeout_overrides={"strong_up": {"sell": 5.0}},
            ),
            "strong_up",
        )

        timeout, reason, regime = runner._resolve_cycle_timeout_policy(side="sell")

        assert timeout == 5.0
        assert reason == "regime_override_strong_up_sell"
        assert regime == "strong_up"

    def test_timeout_policy_falls_back_when_macro_trend_is_none(self) -> None:
        runner = _TimeoutRunner(FillTestConfig(order_timeout_sec=30.0), None)

        timeout, reason, regime = runner._resolve_cycle_timeout_policy(side="buy")

        assert timeout == 30.0
        assert reason == "base_default"
        assert regime is None


class TestTimeoutHotReload:
    def _make_runner(self, config: FillTestConfig) -> SimpleNamespace:
        return SimpleNamespace(
            config=config,
            _time_filter=MagicMock(),
            _daily_drawdown_guard=MagicMock(export_state=MagicMock(return_value={})),
            _sell_kill_mgr=MagicMock(),
            _buy_kill_mgr=MagicMock(),
            _fast_fill_defense=MagicMock(),
            _config_reloader=MagicMock(),
            _maker_price=MagicMock(),
            _git_sha="test",
            _reset_entry_gate_guard=MagicMock(),
        )

    def test_override_change_reflected_immediately(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "fill_test.yaml"
        yaml_path.write_text(
            "order_timeout_sec: 30.0\nregime_timeout_overrides:\n  strong_up:\n    sell: 5.0\n",
            encoding="utf-8",
        )
        config = FillTestConfig(order_timeout_sec=30.0)
        runner = self._make_runner(config)
        reloader = ConfigHotReloader(config=config, yaml_path=yaml_path, yaml_cfg={}, check_interval_sec=0.0)

        assert reloader._do_reload(runner) is True
        assert runner.config.get_timeout_with_reason("sell", "strong_up")[0] == 5.0

        yaml_path.write_text("order_timeout_sec: 30.0\nregime_timeout_overrides:\n  strong_up:\n    sell: 9.0\n", encoding="utf-8")
        reloader._last_mtime = 0.0
        assert reloader._do_reload(runner) is True
        assert runner.config.get_timeout_with_reason("sell", "strong_up")[0] == 9.0

    def test_override_removal_falls_through(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "fill_test.yaml"
        yaml_path.write_text("order_timeout_sec: 30.0\nregime_timeout_overrides:\n  strong_up:\n    sell: 5.0\n", encoding="utf-8")
        config = FillTestConfig(order_timeout_sec=30.0)
        runner = self._make_runner(config)
        reloader = ConfigHotReloader(config=config, yaml_path=yaml_path, yaml_cfg={}, check_interval_sec=0.0)
        assert reloader._do_reload(runner) is True

        yaml_path.write_text("order_timeout_sec: 30.0\n", encoding="utf-8")
        reloader._last_mtime = 0.0
        assert reloader._do_reload(runner) is True
        assert runner.config.get_timeout_with_reason("sell", "strong_up") == (30.0, "base_default")

    def test_global_timeout_change_reflected(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "fill_test.yaml"
        yaml_path.write_text("order_timeout_sec: 30.0\n", encoding="utf-8")
        config = FillTestConfig(order_timeout_sec=30.0)
        runner = self._make_runner(config)
        reloader = ConfigHotReloader(config=config, yaml_path=yaml_path, yaml_cfg={}, check_interval_sec=0.0)
        assert reloader._do_reload(runner) is False

        yaml_path.write_text("order_timeout_sec: 45.0\n", encoding="utf-8")
        reloader._last_mtime = 0.0
        assert reloader._do_reload(runner) is True
        assert runner.config.get_timeout_with_reason("buy", None) == (45.0, "base_default")

