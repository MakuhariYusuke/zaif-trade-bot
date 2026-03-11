"""365# P6 テスト: SAC Sidecar retrain scheduler.

SB3 / HeavyTradingEnv への依存をモックで切り離し、
スケジューラのロジック層を徹底テスト。

カバレッジ対象:
  - SACRetrainConfig (dataclass, from_yaml_dict)
  - SACRetrainTrigger (should_retrain, record_result, backoff)
  - RetrainResult (to_dict)
  - retrain_once (モック model/env)
  - run_scheduler (1 iteration モック)
  - _atomic_deploy_model / _update_sidecar_signal / _append_history
  - load_config (YAML)
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ════════════════════════════════════════════════════════════════
# §1 SACRetrainConfig
# ════════════════════════════════════════════════════════════════


class TestSACRetrainConfig:
    """SACRetrainConfig dataclass + from_yaml_dict."""

    def test_defaults(self) -> None:
        from scripts.v460.ml.sac_retrain_scheduler import SACRetrainConfig

        cfg = SACRetrainConfig()
        assert cfg.total_timesteps == 50_000
        assert cfg.incremental_timesteps == 15_000
        assert cfg.gamma == 0.80
        assert cfg.retrain_interval_sec == 7200
        assert cfg.rolling_window_days == 7

    def test_from_yaml_dict_minimal(self) -> None:
        from scripts.v460.ml.sac_retrain_scheduler import SACRetrainConfig

        raw = {
            "data": {"ohlcv_path": "test.parquet"},
            "sac_hyperparameters": {"gamma": 0.95},
            "training": {"total_timesteps": 30000},
            "features": {"selected": ["price_velocity", "micro_trend"]},
        }
        cfg = SACRetrainConfig.from_yaml_dict(raw)
        assert cfg.ohlcv_path == "test.parquet"
        assert cfg.gamma == 0.95
        assert cfg.total_timesteps == 30000
        assert cfg.feature_columns == ["price_velocity", "micro_trend"]

    def test_from_yaml_dict_retrain_section(self) -> None:
        from scripts.v460.ml.sac_retrain_scheduler import SACRetrainConfig

        raw = {
            "sac_retrain": {
                "rolling_window_days": 14,
                "incremental_timesteps": 20000,
                "check_interval_sec": 120,
                "min_gross_roi": 0.01,
            },
        }
        cfg = SACRetrainConfig.from_yaml_dict(raw)
        assert cfg.rolling_window_days == 14
        assert cfg.incremental_timesteps == 20000
        assert cfg.check_interval_sec == 120
        assert cfg.min_gross_roi == 0.01

    def test_from_yaml_dict_empty(self) -> None:
        from scripts.v460.ml.sac_retrain_scheduler import SACRetrainConfig

        cfg = SACRetrainConfig.from_yaml_dict({})
        # Should use all defaults without error
        assert cfg.total_timesteps == 50_000

    def test_from_yaml_dict_confidence_roi_full(self) -> None:
        """372# audit: confidence_roi_full が YAML からパースされる."""
        from scripts.v460.ml.sac_retrain_scheduler import SACRetrainConfig

        raw = {"sac_retrain": {"confidence_roi_full": 0.01}}
        cfg = SACRetrainConfig.from_yaml_dict(raw)
        assert cfg.confidence_roi_full == pytest.approx(0.01)

    def test_from_yaml_dict_min_trade_count(self) -> None:
        """372# audit: min_trade_count が YAML からパースされる."""
        from scripts.v460.ml.sac_retrain_scheduler import SACRetrainConfig

        raw = {"sac_retrain": {"min_trade_count": 5}}
        cfg = SACRetrainConfig.from_yaml_dict(raw)
        assert cfg.min_trade_count == 5

    def test_from_yaml_dict_372_fields_defaults(self) -> None:
        """372# audit: 未指定時はデフォルト値が使われる."""
        from scripts.v460.ml.sac_retrain_scheduler import SACRetrainConfig

        cfg = SACRetrainConfig.from_yaml_dict({})
        assert cfg.confidence_roi_full == pytest.approx(0.005)
        assert cfg.min_trade_count == 3


# ════════════════════════════════════════════════════════════════
# §2 SACRetrainTrigger
# ════════════════════════════════════════════════════════════════


class TestSACRetrainTrigger:
    """再訓練トリガー判定."""

    def _make_trigger(self, tmp_path: Path, **overrides):
        from scripts.v460.ml.sac_retrain_scheduler import (
            SACRetrainConfig,
            SACRetrainTrigger,
        )

        data_file = tmp_path / "data.parquet"
        data_file.write_bytes(b"dummy")
        cfg = SACRetrainConfig(
            ohlcv_path=str(data_file),
            retrain_interval_sec=60,
            retrain_interval_max_sec=600,
            **overrides,
        )
        return SACRetrainTrigger(cfg=cfg), data_file

    def test_first_run_should_retrain(self, tmp_path: Path) -> None:
        trigger, _ = self._make_trigger(tmp_path)
        should, reason = trigger.should_retrain()
        assert should is True
        assert "data_updated" in reason

    def test_interval_wait(self, tmp_path: Path) -> None:
        trigger, _ = self._make_trigger(tmp_path)
        trigger._last_retrain_time = time.time()  # just ran
        should, reason = trigger.should_retrain()
        assert should is False
        assert "interval_wait" in reason

    def test_data_unchanged(self, tmp_path: Path) -> None:
        trigger, data_file = self._make_trigger(tmp_path)
        trigger._last_retrain_time = time.time() - 120  # interval passed
        trigger._last_data_mtime = data_file.stat().st_mtime
        should, reason = trigger.should_retrain()
        assert should is False
        assert "data_unchanged" in reason

    def test_data_not_found(self, tmp_path: Path) -> None:
        from scripts.v460.ml.sac_retrain_scheduler import (
            SACRetrainConfig,
            SACRetrainTrigger,
        )

        cfg = SACRetrainConfig(ohlcv_path=str(tmp_path / "nonexistent.parquet"))
        trigger = SACRetrainTrigger(cfg=cfg)
        should, reason = trigger.should_retrain()
        assert should is False
        assert "data_not_found" in reason

    def test_backoff_on_failure(self, tmp_path: Path) -> None:
        trigger, _ = self._make_trigger(tmp_path)
        trigger.record_result("error")
        trigger.record_result("error")
        assert trigger._consecutive_failures == 2
        # backoff: 60 * 2^2 = 240
        assert trigger.effective_interval == 240.0

    def test_backoff_capped(self, tmp_path: Path) -> None:
        trigger, _ = self._make_trigger(tmp_path)
        for _ in range(10):
            trigger.record_result("error")
        # max 600
        assert trigger.effective_interval <= 600.0

    def test_success_resets_failures(self, tmp_path: Path) -> None:
        trigger, _ = self._make_trigger(tmp_path)
        trigger.record_result("error")
        trigger.record_result("error")
        assert trigger._consecutive_failures == 2
        trigger.record_result("deployed")
        assert trigger._consecutive_failures == 0


# ════════════════════════════════════════════════════════════════
# §3 RetrainResult
# ════════════════════════════════════════════════════════════════


class TestRetrainResult:
    """RetrainResult dataclass."""

    def test_to_dict(self) -> None:
        from scripts.v460.ml.sac_retrain_scheduler import RetrainResult

        r = RetrainResult(
            status="deployed",
            timestamp="2026-03-10T12:00:00",
            model_version="test_v1",
            training_time_sec=42.5678,
            total_timesteps=15000,
            warm_start=True,
            gross_roi=0.0312345,
        )
        d = r.to_dict()
        assert d["status"] == "deployed"
        assert d["training_time_sec"] == 42.6  # rounded
        assert d["gross_roi"] == pytest.approx(0.031234, abs=1e-5)  # 6dp rounded
        assert d["warm_start"] is True


# ════════════════════════════════════════════════════════════════
# §4 retrain_once (モック)
# ════════════════════════════════════════════════════════════════


def _make_mock_env():
    """Mock training env."""
    env = MagicMock()
    env.observation_space = MagicMock()
    env.observation_space.shape = (12,)
    env.action_space = MagicMock()
    env.action_space.shape = (1,)
    env.reset.return_value = (np.zeros(12), {})
    env.step.return_value = (np.zeros(12), 0.1, True, False, {})
    env.portfolio_value = 10_100_000.0
    env.initial_portfolio_value = 10_000_000.0
    env.trades_count = 5
    return env


def _make_sidecar_env() -> SimpleNamespace:
    """_update_sidecar_signal 用の最小 env."""
    return SimpleNamespace(
        df=[0] * 12,
        current_step=0,
        trades_count=5,
        portfolio_value=10_100_000.0,
        initial_portfolio_value=10_000_000.0,
        _get_observation=lambda: np.zeros(12),
    )


class _EvalEnv:
    """_evaluate_model 用の最小 1-step env."""

    def __init__(
        self,
        *,
        episode_trade_counts: list[int] | tuple[int, ...],
        episode_portfolio_values: list[float] | tuple[float, ...],
        initial_portfolio_value: float = 10_000_000.0,
        reward: float = 0.1,
    ) -> None:
        self._episode_trade_counts = list(episode_trade_counts)
        self._episode_portfolio_values = list(episode_portfolio_values)
        self.initial_portfolio_value = initial_portfolio_value
        self.portfolio_value = initial_portfolio_value
        self.trades_count = 0
        self._episode_index = -1
        self._reward = reward

    def reset(self) -> tuple[np.ndarray, dict[str, object]]:
        self._episode_index += 1
        self.trades_count = 0
        return np.zeros(12), {}

    def step(self, action: object) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        self.trades_count = self._episode_trade_counts[self._episode_index]
        self.portfolio_value = self._episode_portfolio_values[self._episode_index]
        return np.zeros(12), self._reward, True, False, {}


class _PredictOnlyModel:
    """_evaluate_model 用の最小 predict stub."""

    def __init__(self, action: float = 0.0) -> None:
        self._action = np.array([action], dtype=float)

    def predict(self, observation: object, deterministic: bool = True) -> tuple[np.ndarray, None]:
        del observation, deterministic
        return self._action, None


def _make_mock_model():
    """Mock SB3 SAC model."""
    model = MagicMock()
    model.learn.return_value = None
    model.predict.return_value = (np.array([0.42]), None)
    model.save.return_value = None
    model.save_replay_buffer.return_value = None
    model.load_replay_buffer.return_value = None
    return model


@contextmanager
def _mock_sb3_import(mock_model: MagicMock) -> Iterator[MagicMock]:
    """retrain_once() の import_real_sb3 を fake module に置き換える."""
    fake_sac_cls = MagicMock()
    fake_sac_cls.return_value = mock_model
    fake_sac_cls.load.return_value = mock_model

    fake_sb3 = ModuleType("stable_baselines3")
    fake_sb3.__version__ = "test"
    fake_sb3.__file__ = "fake_stable_baselines3.py"
    fake_sb3.SAC = fake_sac_cls

    with patch(
        "scripts.v460.ml.sac_retrain_scheduler.import_real_sb3",
        return_value=fake_sb3,
    ):
        yield fake_sac_cls


class TestRetrainOnce:
    """retrain_once() のテスト (SB3/env をモック化)."""

    @patch("scripts.v460.ml.sac_retrain_scheduler._create_env")
    @patch("scripts.v460.ml.sac_retrain_scheduler._atomic_deploy_model")
    @patch("scripts.v460.ml.sac_retrain_scheduler._update_sidecar_signal")
    def test_cold_start_success(
        self,
        mock_update_signal: MagicMock,
        mock_deploy: MagicMock,
        mock_create_env: MagicMock,
        tmp_path: Path,
    ) -> None:
        from scripts.v460.ml.sac_retrain_scheduler import (
            SACRetrainConfig,
            retrain_once,
        )

        mock_env = _make_mock_env()
        mock_create_env.return_value = mock_env
        mock_model = _make_mock_model()

        cfg = SACRetrainConfig(
            ohlcv_path=str(tmp_path / "data.parquet"),
            model_path=tmp_path / "not_exists.zip",  # cold start
        )

        # Create mock data file
        data_file = tmp_path / "data.parquet"

        with patch("scripts.v460.lib.data_loader.load_parquet") as mock_load:
            import pandas as pd

            mock_load.return_value = pd.DataFrame({"close": range(1000)})
            with _mock_sb3_import(mock_model) as mock_sac_cls:
                mock_sac_cls.return_value = mock_model
                result = retrain_once(cfg)

        assert result.status == "deployed"
        assert result.warm_start is False
        assert result.gross_roi > 0
        mock_deploy.assert_called_once()
        mock_update_signal.assert_called_once()

    @patch("scripts.v460.ml.sac_retrain_scheduler._create_env")
    def test_warm_start(
        self,
        mock_create_env: MagicMock,
        tmp_path: Path,
    ) -> None:
        from scripts.v460.ml.sac_retrain_scheduler import (
            SACRetrainConfig,
            retrain_once,
        )

        mock_env = _make_mock_env()
        mock_create_env.return_value = mock_env
        mock_model = _make_mock_model()

        model_path = tmp_path / "model.zip"
        model_path.write_bytes(b"fake")
        buffer_path = tmp_path / "model.buffer.pkl"
        buffer_path.write_bytes(b"fake")

        cfg = SACRetrainConfig(
            ohlcv_path=str(tmp_path / "data.parquet"),
            model_path=model_path,
            buffer_path=buffer_path,
            signal_path=tmp_path / "signal.json",
        )

        with patch("scripts.v460.lib.data_loader.load_parquet") as mock_load:
            import pandas as pd

            mock_load.return_value = pd.DataFrame({"close": range(1000)})
            with _mock_sb3_import(mock_model) as mock_sac_cls:
                mock_sac_cls.load.return_value = mock_model
                result = retrain_once(cfg)

        assert result.status == "deployed"
        assert result.warm_start is True
        mock_sac_cls.load.assert_called_once()
        mock_model.load_replay_buffer.assert_called_once()

    @patch("scripts.v460.ml.sac_retrain_scheduler._push_neutral_fallback")
    @patch("scripts.v460.ml.sac_retrain_scheduler._create_env")
    def test_oos_failed(
        self,
        mock_create_env: MagicMock,
        mock_push_neutral: MagicMock,
        tmp_path: Path,
    ) -> None:
        from scripts.v460.ml.sac_retrain_scheduler import (
            SACRetrainConfig,
            retrain_once,
        )

        mock_env = _make_mock_env()
        # OOS failure: negative ROI
        mock_env.portfolio_value = 9_000_000.0
        mock_create_env.return_value = mock_env

        cfg = SACRetrainConfig(
            ohlcv_path=str(tmp_path / "data.parquet"),
            model_path=tmp_path / "not_exists.zip",
            min_gross_roi=0.0,  # > 0 required
        )

        with patch("scripts.v460.lib.data_loader.load_parquet") as mock_load:
            import pandas as pd

            mock_load.return_value = pd.DataFrame({"close": range(1000)})
            mock_model = _make_mock_model()
            with _mock_sb3_import(mock_model) as mock_sac_cls:
                mock_sac_cls.return_value = mock_model
                result = retrain_once(cfg)

        assert result.status == "oos_failed"
        assert result.gross_roi < 0
        # 379# P3-C: neutral fallback が呼ばれることを検証
        mock_push_neutral.assert_called_once()

    def test_data_load_error(self, tmp_path: Path) -> None:
        from scripts.v460.ml.sac_retrain_scheduler import (
            SACRetrainConfig,
            retrain_once,
        )

        cfg = SACRetrainConfig(
            ohlcv_path=str(tmp_path / "nonexistent.parquet"),
        )

        with patch(
            "scripts.v460.lib.data_loader.load_parquet",
            side_effect=FileNotFoundError("not found"),
        ):
            result = retrain_once(cfg)

        assert result.status == "error"
        assert "data_load" in result.error_message


# ════════════════════════════════════════════════════════════════
# §5 Atomic deploy + sidecar signal update
# ════════════════════════════════════════════════════════════════


class TestAtomicDeploy:
    """_atomic_deploy_model のテスト."""

    def test_deploy_creates_files(self, tmp_path: Path) -> None:
        from scripts.v460.ml.sac_retrain_scheduler import (
            SACRetrainConfig,
            _atomic_deploy_model,
        )

        model_path = tmp_path / "model.zip"
        buffer_path = tmp_path / "model.buffer.pkl"
        cfg = SACRetrainConfig(model_path=model_path, buffer_path=buffer_path)

        mock_model = MagicMock()

        def save_side_effect(path: str) -> None:
            Path(path).write_bytes(b"model_data")

        def save_buffer_side_effect(path: str) -> None:
            Path(path).write_bytes(b"buffer_data")

        mock_model.save.side_effect = save_side_effect
        mock_model.save_replay_buffer.side_effect = save_buffer_side_effect

        _atomic_deploy_model(mock_model, cfg, "test_v1")

        assert model_path.exists()
        assert buffer_path.exists()


class TestUpdateSidecarSignal:
    """_update_sidecar_signal のテスト."""

    def test_writes_signal_file(self, tmp_path: Path) -> None:
        from scripts.v460.ml.sac_retrain_scheduler import (
            SACRetrainConfig,
            _update_sidecar_signal,
        )

        signal_path = tmp_path / "signal.json"
        cfg = SACRetrainConfig(signal_path=signal_path)

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.42]), None)

        mock_env = _make_sidecar_env()

        _update_sidecar_signal(
            mock_model, mock_env, cfg, "test_v1",
            {"gross_roi": 0.01, "trade_count": 5},
        )

        assert signal_path.exists()
        data = json.loads(signal_path.read_text(encoding="utf-8"))
        assert data["directional_bias"] == pytest.approx(0.42)
        assert data["model_version"] == "test_v1"


# ════════════════════════════════════════════════════════════════
# §6 History append
# ════════════════════════════════════════════════════════════════


class TestEvaluateModel:
    """372# audit: _evaluate_model の複数エピソード集約テスト."""

    def test_multi_episode_aggregation(self) -> None:
        """複数エピソードの trade_count が累積、ROI が平均される."""
        from scripts.v460.ml.sac_retrain_scheduler import (
            SACRetrainConfig,
            _evaluate_model,
        )

        mock_env = _EvalEnv(
            episode_trade_counts=[3, 5, 7],
            episode_portfolio_values=[10_100_000.0, 10_100_000.0, 10_100_000.0],
        )

        mock_model = _PredictOnlyModel()

        cfg = SACRetrainConfig(n_eval_episodes=3)
        result = _evaluate_model(mock_model, mock_env, cfg)

        # trade_count は累積 (3 + 5 + 7 = 15)
        assert result["trade_count"] == 15
        # gross_roi は平均 (各エピソードとも同じ ROI = 0.01)
        assert result["gross_roi"] == pytest.approx(0.01)


class TestAppendHistory:
    """_append_history のテスト."""

    def test_append(self, tmp_path: Path) -> None:
        from scripts.v460.ml.sac_retrain_scheduler import (
            RetrainResult,
            _append_history,
        )

        path = tmp_path / "history.jsonl"
        r1 = RetrainResult(status="deployed", gross_roi=0.01)
        r2 = RetrainResult(status="oos_failed", gross_roi=-0.02)

        _append_history(path, r1)
        _append_history(path, r2)

        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["status"] == "deployed"
        assert json.loads(lines[1])["status"] == "oos_failed"


# ════════════════════════════════════════════════════════════════
# §7 load_config (YAML)
# ════════════════════════════════════════════════════════════════


class TestLoadConfig:
    """load_config YAML loader."""

    def test_load_valid_yaml(
        self,
        write_yaml_file: "Callable[[str | Path, str | dict[str, object]], Path]",
    ) -> None:
        from scripts.v460.ml.sac_retrain_scheduler import load_config

        yaml_content = """
data:
  ohlcv_path: "test_data.parquet"
sac_hyperparameters:
  gamma: 0.90
training:
  total_timesteps: 30000
features:
  selected:
    - price_velocity
"""
        config_path = write_yaml_file("test.yaml", yaml_content)

        cfg = load_config(config_path)
        assert cfg.ohlcv_path == "test_data.parquet"
        assert cfg.gamma == 0.90
        assert cfg.total_timesteps == 30000

    def test_missing_yaml(self, tmp_path: Path) -> None:
        from scripts.v460.ml.sac_retrain_scheduler import load_config

        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")


# ════════════════════════════════════════════════════════════════
# §8 _evaluate_model
# ════════════════════════════════════════════════════════════════


class TestEvaluateModel:
    """_evaluate_model のテスト."""

    def test_positive_roi(self) -> None:
        from scripts.v460.ml.sac_retrain_scheduler import (
            SACRetrainConfig,
            _evaluate_model,
        )

        mock_model = _PredictOnlyModel(0.42)
        mock_env = _EvalEnv(
            episode_trade_counts=[5],
            episode_portfolio_values=[10_100_000.0],
        )
        # PV = 10.1M, IPV = 10M → ROI = 1%
        cfg = SACRetrainConfig(n_eval_episodes=1)

        result = _evaluate_model(mock_model, mock_env, cfg)
        assert result["gross_roi"] == pytest.approx(0.01)
        assert result["trade_count"] == 5

    def test_negative_roi(self) -> None:
        from scripts.v460.ml.sac_retrain_scheduler import (
            SACRetrainConfig,
            _evaluate_model,
        )

        mock_model = _PredictOnlyModel(0.42)
        mock_env = _EvalEnv(
            episode_trade_counts=[5],
            episode_portfolio_values=[9_500_000.0],
        )

        cfg = SACRetrainConfig(n_eval_episodes=1)
        result = _evaluate_model(mock_model, mock_env, cfg)
        assert result["gross_roi"] == pytest.approx(-0.05)


# ════════════════════════════════════════════════════════════════
# §9 run_scheduler (1 iteration)
# ════════════════════════════════════════════════════════════════


class TestRunScheduler:
    """run_scheduler の 1 イテレーション制御テスト."""

    @patch("scripts.v460.ml.sac_retrain_scheduler.retrain_once")
    @patch("scripts.v460.ml.sac_retrain_scheduler._install_signal_handlers")
    def test_single_iteration_then_shutdown(
        self,
        mock_signals: MagicMock,
        mock_retrain: MagicMock,
        tmp_path: Path,
    ) -> None:
        from scripts.v460.ml.sac_retrain_scheduler import (
            SACRetrainConfig,
            RetrainResult,
            run_scheduler,
            _shutdown_event,
        )

        # データファイル
        data_file = tmp_path / "data.parquet"
        data_file.write_bytes(b"dummy")

        cfg = SACRetrainConfig(
            ohlcv_path=str(data_file),
            check_interval_sec=1,
            retrain_interval_sec=1,
            history_path=tmp_path / "history.jsonl",
        )

        mock_retrain.return_value = RetrainResult(status="deployed")

        # 1回実行したら shutdown
        call_count = 0

        original_wait = _shutdown_event.wait

        def limited_wait(timeout: float | None = None) -> bool:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                _shutdown_event.set()
                return True
            return False

        _shutdown_event.clear()

        with patch.object(_shutdown_event, "wait", side_effect=limited_wait):
            with patch.object(_shutdown_event, "is_set", side_effect=[False, False, True]):
                run_scheduler(cfg)

        mock_retrain.assert_called_once()
        _shutdown_event.clear()  # cleanup


# ════════════════════════════════════════════════════════════════
# §10 _push_neutral_fallback + sidecar IO cache
# ════════════════════════════════════════════════════════════════


class TestPushNeutralFallback:
    """379# P3-C: _push_neutral_fallback のテスト."""

    def test_writes_neutral_signal(self, tmp_path: Path) -> None:
        from scripts.v460.ml.sac_retrain_scheduler import _push_neutral_fallback

        signal_path = tmp_path / "signal.json"
        with patch(
            "scripts.v460.lib.sidecar_signal_io.write_sidecar_signal"
        ) as mock_write:
            _push_neutral_fallback()
            mock_write.assert_called_once()
            sig = mock_write.call_args[0][0]
            assert sig.directional_bias == 0.0
            assert sig.confidence == 0.0
            assert sig.model_version == "neutral"


class TestReadSidecarCache:
    """379# P3-C: sidecar_signal_io の mtime キャッシュテスト."""

    def test_cache_hit_on_same_mtime(self, tmp_path: Path) -> None:
        from scripts.v460.lib.sidecar_signal_io import (
            read_sidecar_signal,
            write_sidecar_signal,
            create_neutral_signal,
            _SIDECAR_CACHE,
        )

        signal_path = tmp_path / "signal.json"
        sig = create_neutral_signal()
        write_sidecar_signal(sig, signal_path)

        # 初回読み込み (キャッシュミス)
        result1 = read_sidecar_signal(signal_path, ttl_sec=0)
        assert result1 is not None
        abs_path = str(signal_path.absolute())
        assert abs_path in _SIDECAR_CACHE

        # 同じ mtime で再読み込み → キャッシュヒット
        result2 = read_sidecar_signal(signal_path, ttl_sec=0)
        assert result2 is not None
        assert result2.directional_bias == result1.directional_bias

    def test_cache_invalidated_on_new_write(self, tmp_path: Path) -> None:
        from scripts.v460.lib.sidecar_signal_io import (
            read_sidecar_signal,
            write_sidecar_signal,
            _SIDECAR_CACHE,
        )
        from scripts.v460.lib.sidecar_types import SidecarSignal

        signal_path = tmp_path / "signal.json"

        sig1 = SidecarSignal(
            timestamp="2026-03-11T00:00:00+00:00",
            directional_bias=0.5,
            confidence=1.0,
            model_version="v1",
        )
        write_sidecar_signal(sig1, signal_path)
        result1 = read_sidecar_signal(signal_path, ttl_sec=0)
        assert result1 is not None
        assert result1.directional_bias == pytest.approx(0.5)

        # 新しいシグナルを書き込み → mtime 変更 → キャッシュ無効化
        import time
        time.sleep(0.05)  # Windows mtime 精度確保
        sig2 = SidecarSignal(
            timestamp="2026-03-11T00:00:01+00:00",
            directional_bias=-0.3,
            confidence=0.8,
            model_version="v2",
        )
        write_sidecar_signal(sig2, signal_path)
        result2 = read_sidecar_signal(signal_path, ttl_sec=0)
        assert result2 is not None
        assert result2.directional_bias == pytest.approx(-0.3)

    def test_cache_cleared_on_file_deleted(self, tmp_path: Path) -> None:
        from scripts.v460.lib.sidecar_signal_io import (
            read_sidecar_signal,
            write_sidecar_signal,
            create_neutral_signal,
            _SIDECAR_CACHE,
        )

        signal_path = tmp_path / "signal.json"
        write_sidecar_signal(create_neutral_signal(), signal_path)
        read_sidecar_signal(signal_path, ttl_sec=0)

        abs_path = str(signal_path.absolute())
        assert abs_path in _SIDECAR_CACHE

        # ファイル削除 → キャッシュからも消える
        signal_path.unlink()
        result = read_sidecar_signal(signal_path, ttl_sec=0)
        assert result is None
        assert abs_path not in _SIDECAR_CACHE
