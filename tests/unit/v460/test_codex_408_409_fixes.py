from __future__ import annotations

import ast
import gc
import logging
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from scripts.v460.lib.fill_test_cli import _wait_for_process_start
from scripts.v460.lib.tasks.sac_train import _validate_oos_eval_requirements
from ztb.trading.environment import _load_environment_exports
from ztb.trading.environment.components.calculators.reward_calculator import (
    RewardCalculator,
)
from ztb.trading.environment.components.rewards.forced_balance import (
    ForcedBalanceReward,
)
from ztb.trading.environment.heavy_env.core import _apply_terminal_reward_adjustments
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.trading.environment.utils.config import RewardSettings
from ztb.trading.environment.utils.gc_guard import maybe_collect_garbage
from ztb.trading.live.core.health_monitor import HealthMonitor
from ztb.trading.live.core.idempotency_store import IdempotencyStore
from ztb.trading.live.core.service_runner import TradingService
from ztb.trading.live.simulation.replay_market import ReplayMarket


REPO_ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=None)
def _read_repo_text(*parts: str) -> str:
    return REPO_ROOT.joinpath(*parts).read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _parse_repo_python(*parts: str) -> ast.AST:
    return ast.parse(_read_repo_text(*parts))


def _make_reward_calculator() -> RewardCalculator:
    config = EnvironmentConfig()
    config.max_position_size = 1.0
    config.reward_settings = RewardSettings()
    config.venue_settings = {}
    return RewardCalculator(
        config=config,
        reward_settings=config.reward_settings,
        initial_portfolio_value=100000.0,
    )


class _FakeProcess:
    def __init__(self, poll_results: list[int | None]) -> None:
        self._poll_results = poll_results
        self.returncode: int | None = None

    def poll(self) -> int | None:
        if self._poll_results:
            self.returncode = self._poll_results.pop(0)
        return self.returncode


class _FakePsutil:
    def __init__(self, process: object) -> None:
        self._process = process
        self.cpu_intervals: list[float | None] = []

    def cpu_percent(self, interval: float | None = None) -> float:
        self.cpu_intervals.append(interval)
        return 12.5

    @staticmethod
    def virtual_memory() -> object:
        return SimpleNamespace(percent=42.0, used=2 * 1024**3, total=8 * 1024**3)

    @staticmethod
    def disk_usage(_path: str) -> object:
        return SimpleNamespace(percent=11.0, free=100 * 1024**3)


class _FakeHealthProcess:
    def __init__(self) -> None:
        self.cpu_intervals: list[float | None] = []

    @staticmethod
    def memory_info() -> object:
        return SimpleNamespace(rss=32 * 1024**2, vms=64 * 1024**2)

    def cpu_percent(self, interval: float | None = None) -> float:
        self.cpu_intervals.append(interval)
        return 7.5

    @staticmethod
    def num_threads() -> int:
        return 4


class TestT1IdempotencyLock:
    def test_process_lock_is_exclusive(self, tmp_path: Path) -> None:
        db_path = tmp_path / "orders.sqlite"
        owner = IdempotencyStore(str(db_path), lock_timeout_sec=0.05, lock_retry_interval_sec=0.01)
        waiter = IdempotencyStore(str(db_path), lock_timeout_sec=0.05, lock_retry_interval_sec=0.01)

        with owner._process_lock():
            with pytest.raises(TimeoutError):
                with waiter._process_lock():
                    pass

    def test_process_lock_releases_on_exit(self, tmp_path: Path) -> None:
        db_path = tmp_path / "orders.sqlite"
        store = IdempotencyStore(str(db_path))
        assert store._lock_file is not None

        with store._process_lock():
            assert store._lock_file.exists()

        assert not store._lock_file.exists()

    def test_stale_lock_recovery(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        db_path = tmp_path / "orders.sqlite"
        store = IdempotencyStore(str(db_path), lock_timeout_sec=0.05, lock_retry_interval_sec=0.01)
        assert store._lock_file is not None
        store._lock_file.write_text("999999", encoding="utf-8")
        monkeypatch.setattr(store, "_pid_exists", lambda _pid: False)

        with store._process_lock():
            assert store._lock_file.exists()
            assert store._lock_file.read_text(encoding="utf-8").strip() == str(os.getpid())


class TestT2RewardComponents:
    def test_reward_components_include_bankruptcy_penalty(self) -> None:
        config = EnvironmentConfig(reward_scaling=2.0)
        config.bankruptcy_threshold = 2000.0  # type: ignore[attr-defined]
        config.bankruptcy_penalty = 1000.0  # type: ignore[attr-defined]
        config.drawdown_penalty_threshold = 1.0  # type: ignore[attr-defined]

        info: dict[str, object] = {}
        reward, components = _apply_terminal_reward_adjustments(
            5.0,
            info=info,
            portfolio_value=1000.0,
            initial_portfolio_value=10000.0,
            config=config,
        )

        assert reward == components["final_reward"]
        assert components["bankruptcy_penalty"] == -2000.0
        assert info["bankruptcy"] is True

    def test_reward_components_include_drawdown_penalty(self) -> None:
        config = EnvironmentConfig(reward_scaling=2.0)
        config.bankruptcy_threshold = 500.0  # type: ignore[attr-defined]
        config.drawdown_penalty_threshold = 0.2  # type: ignore[attr-defined]
        config.drawdown_penalty_factor = 0.1  # type: ignore[attr-defined]

        info: dict[str, object] = {}
        reward, components = _apply_terminal_reward_adjustments(
            1.0,
            info=info,
            portfolio_value=7000.0,
            initial_portfolio_value=10000.0,
            config=config,
        )

        assert reward == pytest.approx(0.98)
        assert components["drawdown_penalty"] == pytest.approx(-0.02)
        assert info["drawdown_penalty"] == pytest.approx(0.02)

    def test_reward_components_match_final_reward(self) -> None:
        config = EnvironmentConfig(reward_scaling=1.0)
        config.bankruptcy_threshold = 500.0  # type: ignore[attr-defined]
        config.drawdown_penalty_threshold = 0.9  # type: ignore[attr-defined]

        reward, components = _apply_terminal_reward_adjustments(
            3.5,
            info={},
            portfolio_value=9000.0,
            initial_portfolio_value=10000.0,
            config=config,
        )

        assert reward == 3.5
        assert components == {"final_reward": 3.5}


class TestT3ReplayMarketProgress:
    def test_replay_progress_none_data(self) -> None:
        market = ReplayMarket.__new__(ReplayMarket)
        market._data = None
        market._current_index = 0
        assert market.get_progress() == 0.0

    def test_replay_progress_empty_dataframe(self) -> None:
        market = ReplayMarket.__new__(ReplayMarket)
        market._data = pd.DataFrame()
        market._current_index = 0
        assert market.get_progress() == 0.0

    def test_replay_progress_normal(self) -> None:
        market = ReplayMarket.__new__(ReplayMarket)
        market._data = pd.DataFrame({"x": [1, 2, 3, 4]})
        market._current_index = 2
        assert 0.0 < market.get_progress() <= 1.0


class TestT4ServiceRunnerRestart:
    def _make_runner(self) -> TradingService:
        runner = TradingService.__new__(TradingService)
        runner.logger = logging.getLogger("test_service_runner")
        runner.restart_count = 0
        runner.max_restarts = 2
        return runner

    def test_should_restart_false_after_success(self) -> None:
        runner = self._make_runner()
        assert runner._should_restart(True) is False

    def test_should_restart_true_after_failure_within_limit(self) -> None:
        runner = self._make_runner()
        assert runner._should_restart(False) is True

    def test_should_restart_false_when_max_exceeded(self) -> None:
        runner = self._make_runner()
        runner.restart_count = 1
        assert runner._should_restart(False) is False


class TestT5HealthMonitor:
    def test_health_status_is_non_blocking(self) -> None:
        monitor = HealthMonitor()
        fake_process = _FakeHealthProcess()
        fake_psutil = _FakePsutil(fake_process)
        monitor._psutil = fake_psutil
        monitor._process = fake_process

        start = time.monotonic()
        status = monitor.get_health_status()
        elapsed = time.monotonic() - start

        assert elapsed < 0.5
        assert status["status"] == "healthy"
        assert fake_psutil.cpu_intervals == [None]
        assert fake_process.cpu_intervals == [None]

    def test_process_handle_reused(self) -> None:
        monitor = HealthMonitor()
        assert hasattr(monitor, "_process")


class TestT6EnvironmentImportHandling:
    def test_import_error_emits_warning(self) -> None:
        def _raise_import_error(_name: str, _package: str) -> object:
            raise ImportError("missing optional dependency")

        with pytest.warns(ImportWarning):
            flip_env, heavy_env = _load_environment_exports(_raise_import_error)

        assert flip_env is None
        assert heavy_env is None

    def test_runtime_error_is_not_silenced(self) -> None:
        def _raise_runtime_error(_name: str, _package: str) -> object:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            _load_environment_exports(_raise_runtime_error)


class TestT7SacTrainOOSValidation:
    def test_oos_validation_requires_env(self) -> None:
        with pytest.raises(ValueError, match="oos_eval_env"):
            _validate_oos_eval_requirements(True, None, Path("best.zip"))

    def test_oos_validation_requires_best_model_path(self) -> None:
        with pytest.raises(ValueError, match="best_model_path"):
            _validate_oos_eval_requirements(True, object(), None)

    def test_oos_validation_allows_disabled(self) -> None:
        _validate_oos_eval_requirements(False, None, None)


class TestT8RetrainSchedulerStartupWait:
    def test_wait_for_process_start_exits_early_on_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        slept: list[float] = []
        monkeypatch.setattr(time, "sleep", lambda value: slept.append(value))

        process = _FakeProcess([None, 1])

        assert _wait_for_process_start(process, max_wait_sec=10.0, poll_interval_sec=0.5) is False
        assert sum(slept) == pytest.approx(0.5)

    def test_wait_for_process_start_returns_true_after_short_grace(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        slept: list[float] = []
        monkeypatch.setattr(time, "sleep", lambda value: slept.append(value))

        process = _FakeProcess([None, None, None, None])

        assert _wait_for_process_start(
            process,
            max_wait_sec=10.0,
            poll_interval_sec=0.5,
            success_grace_sec=1.0,
        ) is True
        assert sum(slept) == pytest.approx(1.0)


class TestT9ConftestCatchNarrowing:
    def test_conftest_early_section_has_no_broad_exception_handlers(self) -> None:
        tree = _parse_repo_python("tests", "conftest.py")

        broad_handlers = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if node.lineno > 55:
                continue
            if node.type is None:
                broad_handlers.append(node.lineno)
            elif isinstance(node.type, ast.Name) and node.type.id == "Exception":
                broad_handlers.append(node.lineno)

        assert broad_handlers == []


class TestT10BehaviorOptimizationMapping:
    def test_behavior_optimization_known_keys_are_mapped(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            cfg = EnvironmentConfig.from_dict(
                {
                    "environment": {
                        "behavior_optimization": {
                            "trading_bonus": "1.25",
                            "use_simple_reward": "true",
                        }
                    }
                }
            )

        assert cfg.reward_settings is not None
        assert cfg.reward_settings.trading_bonus == pytest.approx(1.25)
        assert cfg.reward_settings.use_simple_reward is True
        assert "Unknown behavior_optimization key ignored" not in caplog.text

    def test_behavior_optimization_unknown_keys_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            cfg = EnvironmentConfig.from_dict(
                {
                    "environment": {
                        "behavior_optimization": {
                            "unknown_reward_key": 123,
                        }
                    }
                }
            )

        assert cfg.reward_settings is not None
        assert "Unknown behavior_optimization key ignored" in caplog.text


class TestT11ArchivedDeadFiles:
    def test_archived_files_exist_and_live_paths_are_removed(self) -> None:
        live_paths = [
            REPO_ROOT / "ztb/trading/environment/components/calculators/simplified_reward_calculator.py",
            REPO_ROOT / "ztb/trading/environment/components/reward/metrics.py",
            REPO_ROOT / "ztb/trading/environment/bridge.py",
        ]
        archived_paths = [
            REPO_ROOT / "ztb/trading/environment/archived/reward/simplified_reward_calculator.py",
            REPO_ROOT / "ztb/trading/environment/archived/reward/metrics.py",
            REPO_ROOT / "ztb/trading/environment/archived/bridge/bridge.py",
        ]

        assert all(not path.exists() for path in live_paths)
        assert all(path.exists() for path in archived_paths)


class TestT12RewardCalculatorSelfTestRemoval:
    def test_reward_calculator_has_no_embedded_self_test(self) -> None:
        assert not hasattr(RewardCalculator, "test_reward_calculation")

    def test_reward_calculation_external_smoke(self) -> None:
        with (
            patch("ztb.trading.environment.components.calculators.reward_calculator.BehavioralPenaltyCalculator") as mock_behavioral_penalty,
            patch("ztb.trading.environment.components.calculators.reward_calculator.AsymmetricRewardScaler") as mock_asymmetric_scaler,
            patch("ztb.trading.environment.components.calculators.reward_calculator.DynamicRewardShaper") as mock_dynamic_shaper,
            patch("ztb.trading.environment.components.calculators.reward_calculator.SignalIntegrator") as mock_signal_integrator,
            patch("ztb.trading.environment.components.calculators.reward_calculator.OpportunityCostPenaltyCalculator"),
            patch("ztb.trading.environment.components.calculators.reward_calculator.UnrealizedLossPenaltyCalculator"),
        ):
            mock_behavioral_penalty.return_value.record_action = Mock()
            mock_behavioral_penalty.return_value._get_recent_counts = Mock(return_value=[0, 0, 0])
            mock_asymmetric_scaler.return_value.scale_reward = lambda reward, _position, _pnl: reward
            mock_dynamic_shaper.return_value.shape_reward = lambda reward, _price, _step, _pnl: reward
            mock_signal_integrator.return_value.enabled = False
            mock_signal_integrator.return_value.integrate_signal.return_value = 0.0

            calculator = _make_reward_calculator()
            reward = calculator.calculate_reward(
                action=1,
                current_price=5_000_000.0,
                position=0.02,
                portfolio_value=100_100.0,
                atr=500.0,
                transaction_cost=10.0,
                reward_scaling=1.0,
                pnl=100.0,
                old_position=0.0,
                step=1,
                observation=None,
                reward_history=[],
                portfolio_value_history=[100000.0] * 30,
            )

        assert isinstance(reward, float)


class TestT13DeprecatedRewardCalculatorMethods:
    def test_get_current_regime_warns(self) -> None:
        calculator = RewardCalculator.__new__(RewardCalculator)
        calculator.trend_detector = None

        with pytest.warns(DeprecationWarning):
            regime = calculator.get_current_regime(100.0, 1)

        assert regime == "sideways"

    def test_reset_episode_state_warns(self) -> None:
        calculator = RewardCalculator.__new__(RewardCalculator)
        calculator.reset = Mock()

        with pytest.warns(DeprecationWarning):
            calculator.reset_episode_state()

        calculator.reset.assert_called_once_with()


class TestT14ForcedBalanceCanonicalMapping:
    def test_penalty_mapping_uses_canonical_helper(self) -> None:
        calculator = RewardCalculator.__new__(RewardCalculator)
        calculator.get_setting_float = lambda key, default: {
            "forced_balance.penalty.scale": 1.5,
            "forced_balance.penalty.threshold_small": 0.05,
            "forced_balance.penalty.threshold_medium": 0.1,
            "forced_balance.penalty.threshold_large": 0.2,
            "forced_balance.penalty.value_small_deviation": 1.0,
            "forced_balance.penalty.value_medium_deviation": 2.5,
            "forced_balance.penalty.value_large_deviation": 5.0,
            "forced_balance.penalty.value_very_large_deviation": 10.0,
        }.get(key, default)

        actual = calculator._map_forced_balance_penalty(0.08, 0.4)
        expected = ForcedBalanceReward._map_forced_balance_penalty_static(
            deviation=0.08,
            severity=0.4,
            penalty_scale=1.5,
            thresh_small=0.05,
            thresh_medium=0.1,
            thresh_large=0.2,
            penalty_small=1.0,
            penalty_medium=2.5,
            penalty_large=5.0,
            penalty_very_large=10.0,
        )

        assert actual == expected

    def test_bonus_mapping_uses_canonical_helper(self) -> None:
        calculator = RewardCalculator.__new__(RewardCalculator)
        calculator.get_setting_float = lambda key, default: {
            "forced_balance.bonus.scale": 1.25,
            "forced_balance.bonus.threshold_small": 0.05,
            "forced_balance.bonus.threshold_medium": 0.1,
            "forced_balance.bonus.value_small_deviation": 6.0,
            "forced_balance.bonus.value_medium_deviation": 12.0,
            "forced_balance.bonus.value_large_deviation": 20.0,
        }.get(key, default)

        actual = calculator._map_forced_balance_bonus(0.08, 0.4)
        expected = ForcedBalanceReward._map_forced_balance_bonus_static(
            deviation=0.08,
            severity=0.4,
            bonus_scale=1.25,
            thresh_small=0.05,
            thresh_medium=0.1,
            bonus_small=6.0,
            bonus_medium=12.0,
            bonus_large=20.0,
        )

        assert actual == expected


class TestT15ConditionalGC:
    def test_maybe_collect_garbage_triggers_only_under_pressure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        collected: list[bool] = []
        fake_psutil = SimpleNamespace(
            virtual_memory=lambda: SimpleNamespace(percent=91.0)
        )

        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
        monkeypatch.setattr(gc, "collect", lambda: collected.append(True) or 0)

        assert maybe_collect_garbage(85.0) is True
        assert collected == [True]


class TestT16IntegrationAssertionCleanup:
    def test_no_assert_true_in_v459_phase0_integration(self) -> None:
        tree = _parse_repo_python(
            "tests",
            "integration",
            "test_v459_phase0_integration.py",
        )
        assert_true_nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assert)
            and isinstance(node.test, ast.Constant)
            and node.test.value is True
        ]
        assert assert_true_nodes == []
