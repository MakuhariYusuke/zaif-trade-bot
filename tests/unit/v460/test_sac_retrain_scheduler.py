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
import pandas as pd
import pytest

from scripts.v460.lib.sidecar_signal_io import (
    _SIDECAR_CACHE,
    create_neutral_signal,
    read_sidecar_signal,
    write_sidecar_signal,
)
from scripts.v460.lib.sidecar_types import SidecarSignal
from scripts.v460.ml.sac_retrain_scheduler import (
    SACRetrainConfig,
    SACRetrainTrigger,
    RetrainResult,
    _build_training_debug_details,
    _append_history,
    _atomic_deploy_model,
    _evaluate_model,
    _MAX_AUTO_RESTARTS,
    _post_cycle_memory_check,
    _push_neutral_fallback,
    _RESTART_BACKOFF_SEC,
    _RSS_WARNING_MB,
    _TRAINING_TIMEOUT_SEC,
    _shutdown_event,
    _update_sidecar_signal,
    load_config,
    main,
    retrain_once,
    run_scheduler,
)


# ════════════════════════════════════════════════════════════════
# §1 SACRetrainConfig
# ════════════════════════════════════════════════════════════════


class TestSACRetrainConfig:
    """SACRetrainConfig dataclass + from_yaml_dict."""

    def test_defaults(self) -> None:
        cfg = SACRetrainConfig()
        assert cfg.total_timesteps == 50_000
        assert cfg.incremental_timesteps == 15_000
        assert cfg.gamma == 0.80
        assert cfg.retrain_interval_sec == 7200
        assert cfg.rolling_window_days == 7

    def test_from_yaml_dict_minimal(self) -> None:
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
        cfg = SACRetrainConfig.from_yaml_dict({})
        # Should use all defaults without error
        assert cfg.total_timesteps == 50_000

    def test_from_yaml_dict_confidence_roi_full(self) -> None:
        """372# audit: confidence_roi_full が YAML からパースされる."""
        raw = {"sac_retrain": {"confidence_roi_full": 0.01}}
        cfg = SACRetrainConfig.from_yaml_dict(raw)
        assert cfg.confidence_roi_full == pytest.approx(0.01)

    def test_from_yaml_dict_min_trade_count(self) -> None:
        """372# audit: min_trade_count が YAML からパースされる."""
        raw = {"sac_retrain": {"min_trade_count": 5}}
        cfg = SACRetrainConfig.from_yaml_dict(raw)
        assert cfg.min_trade_count == 5

    def test_from_yaml_dict_372_fields_defaults(self) -> None:
        """372# audit: 未指定時はデフォルト値が使われる."""
        cfg = SACRetrainConfig.from_yaml_dict({})
        assert cfg.confidence_roi_full == pytest.approx(0.005)
        assert cfg.min_trade_count == 3


# ════════════════════════════════════════════════════════════════
# §2 SACRetrainTrigger
# ════════════════════════════════════════════════════════════════


class TestSACRetrainTrigger:
    """再訓練トリガー判定."""

    def _make_trigger(self, tmp_path: Path, **overrides):
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


_MOCK_OHLCV_DF = pd.DataFrame({"close": np.arange(12, dtype=float)})
_EVAL_RESULT_PASS = {"gross_roi": 0.01, "trade_count": 5}
_EVAL_RESULT_FAIL = {"gross_roi": -0.05, "trade_count": 5}


@contextmanager
def _mock_sb3_import(mock_model: MagicMock) -> Iterator[MagicMock]:
    """retrain_once() の SB3 SAC クラスを fake に置き換える.

    384#: import_real_sb3 廃止 → 関数内 `from stable_baselines3 import SAC`
    をモック化。sys.modules に fake SB3 を注入して制御する。
    """
    fake_sac_cls = MagicMock()
    fake_sac_cls.return_value = mock_model
    fake_sac_cls.load.return_value = mock_model

    fake_sb3 = ModuleType("stable_baselines3")
    fake_sb3.__version__ = "test"
    fake_sb3.__file__ = "fake_stable_baselines3.py"
    fake_sb3.SAC = fake_sac_cls

    with patch.dict("sys.modules", {"stable_baselines3": fake_sb3}):
        yield fake_sac_cls


@contextmanager
def _run_retrain_once_with_patches(
    cfg: SACRetrainConfig,
    *,
    mock_model: MagicMock,
    mock_env: MagicMock,
    eval_result: dict[str, float | int],
) -> Iterator[tuple[MagicMock, MagicMock]]:
    """retrain_once() の主要 patch を束ねる."""
    with (
        patch("scripts.v460.ml.sac_retrain_scheduler._create_env", return_value=mock_env) as mock_create_env,
        patch("scripts.v460.ml.sac_retrain_scheduler._evaluate_model", return_value=eval_result) as mock_evaluate_model,
        patch("scripts.v460.lib.data_loader.load_parquet", return_value=_MOCK_OHLCV_DF),
        _mock_sb3_import(mock_model) as mock_sac_cls,
    ):
        yield mock_sac_cls, mock_evaluate_model


class TestRetrainOnce:
    """retrain_once() のテスト (SB3/env をモック化)."""

    @patch("scripts.v460.ml.sac_retrain_scheduler._atomic_deploy_model")
    @patch("scripts.v460.ml.sac_retrain_scheduler._update_sidecar_signal")
    def test_cold_start_success(
        self,
        mock_update_signal: MagicMock,
        mock_deploy: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_env = _make_mock_env()
        mock_model = _make_mock_model()

        cfg = SACRetrainConfig(
            ohlcv_path=str(tmp_path / "data.parquet"),
            model_path=tmp_path / "not_exists.zip",  # cold start
        )

        with _run_retrain_once_with_patches(
            cfg,
            mock_model=mock_model,
            mock_env=mock_env,
            eval_result=_EVAL_RESULT_PASS,
        ) as (mock_sac_cls, _):
            mock_sac_cls.return_value = mock_model
            result = retrain_once(cfg)

        assert result.status == "deployed"
        assert result.warm_start is False
        assert result.gross_roi > 0
        mock_deploy.assert_called_once()
        mock_update_signal.assert_called_once()

    def test_warm_start(
        self,
        tmp_path: Path,
    ) -> None:
        mock_env = _make_mock_env()
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

        with _run_retrain_once_with_patches(
            cfg,
            mock_model=mock_model,
            mock_env=mock_env,
            eval_result=_EVAL_RESULT_PASS,
        ) as (mock_sac_cls, _):
            mock_sac_cls.load.return_value = mock_model
            result = retrain_once(cfg)

        assert result.status == "deployed"
        assert result.warm_start is True
        mock_sac_cls.load.assert_called_once()
        mock_model.load_replay_buffer.assert_called_once()

    @patch("scripts.v460.ml.sac_retrain_scheduler._push_neutral_fallback")
    def test_oos_failed(
        self,
        mock_push_neutral: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_env = _make_mock_env()

        cfg = SACRetrainConfig(
            ohlcv_path=str(tmp_path / "data.parquet"),
            model_path=tmp_path / "not_exists.zip",
            min_gross_roi=0.0,  # > 0 required
        )

        mock_model = _make_mock_model()
        with _run_retrain_once_with_patches(
            cfg,
            mock_model=mock_model,
            mock_env=mock_env,
            eval_result=_EVAL_RESULT_FAIL,
        ) as (mock_sac_cls, _):
            mock_sac_cls.return_value = mock_model
            result = retrain_once(cfg)

        assert result.status == "oos_failed"
        assert result.gross_roi < 0
        # 379# P3-C: neutral fallback が呼ばれることを検証
        mock_push_neutral.assert_called_once()

    def test_data_load_error(self, tmp_path: Path) -> None:
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

    def test_training_debug_details_contains_shapes(self) -> None:
        train_df = pd.DataFrame({
            "timestamp": [1.0, 2.0, 3.0],
            "close": [100.0, 101.0, 102.0],
        })
        val_df = pd.DataFrame({
            "timestamp": [4.0, 5.0],
            "close": [103.0, 104.0],
        })
        cfg = SACRetrainConfig(feature_columns=["close", "volume"])

        details = _build_training_debug_details(
            train_df,
            val_df,
            cfg,
            env=_make_mock_env(),
        )

        assert details["train_rows"] == 3
        assert details["val_rows"] == 2
        assert details["feature_columns_configured"] == 2
        assert details["observation_shape"] == [12]
        assert details["action_shape"] == [1]
        assert "process_rss_mb" in details


# ════════════════════════════════════════════════════════════════
# §5 Atomic deploy + sidecar signal update
# ════════════════════════════════════════════════════════════════


class TestAtomicDeploy:
    """_atomic_deploy_model のテスト."""

    def test_deploy_creates_files(self, tmp_path: Path) -> None:
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
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")


# ════════════════════════════════════════════════════════════════
# §8 _evaluate_model
# ════════════════════════════════════════════════════════════════


class TestEvaluateModel:
    """_evaluate_model のテスト."""

    def test_positive_roi(self) -> None:
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

        # 新しいシグナルを書き込み → mtime を明示更新してキャッシュ無効化
        sig2 = SidecarSignal(
            timestamp="2026-03-11T00:00:01+00:00",
            directional_bias=-0.3,
            confidence=0.8,
            model_version="v2",
        )
        write_sidecar_signal(sig2, signal_path)
        current_mtime = signal_path.stat().st_mtime
        os.utime(signal_path, (current_mtime + 1.0, current_mtime + 1.0))
        result2 = read_sidecar_signal(signal_path, ttl_sec=0)
        assert result2 is not None
        assert result2.directional_bias == pytest.approx(-0.3)

    def test_cache_cleared_on_file_deleted(self, tmp_path: Path) -> None:
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


# ════════════════════════════════════════════════════════════════
# §12 495# Crash Resilience Tests
# ════════════════════════════════════════════════════════════════


class TestCrashResilience495:
    """495# 再学習スケジューラのクラッシュ耐性テスト."""

    @patch("scripts.v460.ml.sac_retrain_scheduler.retrain_once")
    @patch("scripts.v460.ml.sac_retrain_scheduler._install_signal_handlers")
    def test_trigger_exception_does_not_kill_loop(
        self,
        mock_signals: MagicMock,
        mock_retrain: MagicMock,
        tmp_path: Path,
    ) -> None:
        """trigger.should_retrain() が例外を投げてもループが継続."""
        data_file = tmp_path / "data.parquet"
        data_file.write_bytes(b"dummy")

        cfg = SACRetrainConfig(
            ohlcv_path=str(data_file),
            check_interval_sec=1,
            retrain_interval_sec=1,
            history_path=tmp_path / "history.jsonl",
        )

        call_count = 0
        _shutdown_event.clear()

        def limited_wait(timeout: float | None = None) -> bool:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                _shutdown_event.set()
                return True
            return False

        # should_retrain が 1回目で例外→ 2回目の wait で shutdown
        with (
            patch.object(_shutdown_event, "wait", side_effect=limited_wait),
            patch.object(_shutdown_event, "is_set", side_effect=[False, False, True]),
            patch.object(
                SACRetrainTrigger,
                "should_retrain",
                side_effect=RuntimeError("disk error"),
            ),
        ):
            # 例外が run_scheduler から漏れないことを確認
            run_scheduler(cfg)

        # retrain_once は呼ばれない (trigger が失敗するため)
        mock_retrain.assert_not_called()
        _shutdown_event.clear()

    @patch("scripts.v460.ml.sac_retrain_scheduler.retrain_once")
    @patch("scripts.v460.ml.sac_retrain_scheduler._install_signal_handlers")
    def test_record_result_exception_does_not_kill_loop(
        self,
        mock_signals: MagicMock,
        mock_retrain: MagicMock,
        tmp_path: Path,
    ) -> None:
        """trigger.record_result() が例外を投げてもループが継続."""
        data_file = tmp_path / "data.parquet"
        data_file.write_bytes(b"dummy")

        cfg = SACRetrainConfig(
            ohlcv_path=str(data_file),
            check_interval_sec=1,
            retrain_interval_sec=1,
            history_path=tmp_path / "history.jsonl",
        )

        mock_retrain.return_value = RetrainResult(status="deployed")
        call_count = 0
        _shutdown_event.clear()

        def limited_wait(timeout: float | None = None) -> bool:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                _shutdown_event.set()
                return True
            return False

        with (
            patch.object(_shutdown_event, "wait", side_effect=limited_wait),
            patch.object(_shutdown_event, "is_set", side_effect=[False, False, True]),
            patch.object(
                SACRetrainTrigger,
                "record_result",
                side_effect=RuntimeError("record error"),
            ),
        ):
            run_scheduler(cfg)

        # ループが record_result 例外で死んでいないことを確認
        # (record_result が _last_retrain_time を更新できなかったため 2 回呼ばれる)
        assert mock_retrain.call_count >= 1
        _shutdown_event.clear()

    def test_main_auto_restart_on_scheduler_crash(self, tmp_path: Path) -> None:
        """run_scheduler が例外で死んでも main() が再起動."""
        data_file = tmp_path / "data.parquet"
        data_file.write_bytes(b"dummy")
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            f"data:\n  ohlcv_path: {data_file}\nsac_retrain:\n"
            f"  check_interval_sec: 1\n  retrain_interval_sec: 1\n"
            f"  history_path: {tmp_path / 'history.jsonl'}\n",
            encoding="utf-8",
        )

        crash_count = 0

        def crashing_scheduler(cfg: SACRetrainConfig) -> None:
            nonlocal crash_count
            crash_count += 1
            if crash_count <= 2:
                raise RuntimeError(f"crash #{crash_count}")
            # 3回目は正常終了 (graceful shutdown をシミュレート)

        _shutdown_event.clear()

        with (
            patch(
                "scripts.v460.ml.sac_retrain_scheduler.run_scheduler",
                side_effect=crashing_scheduler,
            ),
            patch(
                "sys.argv",
                ["sac_retrain_scheduler.py", "--config", str(config_file)],
            ),
            patch.object(
                _shutdown_event,
                "wait",
                return_value=False,  # backoff wait → 即復帰
            ),
        ):
            main()

        assert crash_count == 3  # 2回クラッシュ + 1回正常終了
        _shutdown_event.clear()

    def test_main_auto_restart_limit(self, tmp_path: Path) -> None:
        """自動リスタートが _MAX_AUTO_RESTARTS で打ち切られる."""
        data_file = tmp_path / "data.parquet"
        data_file.write_bytes(b"dummy")
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            f"data:\n  ohlcv_path: {data_file}\nsac_retrain:\n"
            f"  check_interval_sec: 1\n  retrain_interval_sec: 1\n"
            f"  history_path: {tmp_path / 'history.jsonl'}\n",
            encoding="utf-8",
        )

        crash_count = 0

        def always_crashing(cfg: SACRetrainConfig) -> None:
            nonlocal crash_count
            crash_count += 1
            raise RuntimeError(f"persistent crash #{crash_count}")

        _shutdown_event.clear()

        with (
            patch(
                "scripts.v460.ml.sac_retrain_scheduler.run_scheduler",
                side_effect=always_crashing,
            ),
            patch(
                "sys.argv",
                ["sac_retrain_scheduler.py", "--config", str(config_file)],
            ),
            patch.object(
                _shutdown_event,
                "wait",
                return_value=False,
            ),
        ):
            main()

        # _MAX_AUTO_RESTARTS + 1 回 (limit check が > なので)
        assert crash_count == _MAX_AUTO_RESTARTS + 1
        _shutdown_event.clear()

    def test_main_fatal_config_error_logged(self, tmp_path: Path) -> None:
        """load_config が失敗しても main() が例外で死なない (ログ出力して終了)."""
        _shutdown_event.clear()

        with (
            patch(
                "sys.argv",
                ["sac_retrain_scheduler.py", "--config", "/nonexistent/path.yaml"],
            ),
        ):
            # FileNotFoundError は try/except で捕捉される → main() は正常終了
            main()

        _shutdown_event.clear()

    def test_training_timeout_raises(self, tmp_path: Path) -> None:
        """model.learn() がタイムアウトした場合、TimeoutError で retrain_once がエラー返却."""
        data_file = tmp_path / "data.parquet"
        data_file.write_bytes(b"dummy")

        cfg = SACRetrainConfig(
            ohlcv_path=str(data_file),
            model_path=tmp_path / "model.zip",
            buffer_path=tmp_path / "buffer.pkl",
            signal_path=tmp_path / "signal.json",
            history_path=tmp_path / "history.jsonl",
        )

        mock_env = _make_mock_env()
        mock_model = _make_mock_model()

        # model.learn を長時間ブロックに置き換え
        def slow_learn(**kwargs: object) -> None:
            time.sleep(10)

        mock_model.learn.side_effect = slow_learn

        with (
            patch("scripts.v460.ml.sac_retrain_scheduler._create_env", return_value=mock_env),
            patch("scripts.v460.ml.sac_retrain_scheduler._evaluate_model"),
            patch("scripts.v460.lib.data_loader.load_parquet", return_value=_MOCK_OHLCV_DF),
            _mock_sb3_import(mock_model),
            # タイムアウトを極短に設定
            patch(
                "scripts.v460.ml.sac_retrain_scheduler._TRAINING_TIMEOUT_SEC",
                0.5,
            ),
        ):
            result = retrain_once(cfg)

        assert result.status == "error"
        assert "timeout" in result.error_message.lower()

    def test_post_cycle_memory_check_runs(self) -> None:
        """_post_cycle_memory_check が例外なく完了."""
        import scripts.v460.ml.sac_retrain_scheduler as mod

        # RSS 追跡状態をリセット
        saved = mod._last_cycle_rss_mb
        try:
            mod._last_cycle_rss_mb = 0.0
            _post_cycle_memory_check()
            # 呼出後 RSS が記録される
            assert mod._last_cycle_rss_mb > 0
        finally:
            mod._last_cycle_rss_mb = saved

    def test_retrain_once_cleans_up_on_error(self, tmp_path: Path) -> None:
        """retrain_once が例外時も cleanup_training_resources を呼ぶ."""
        data_file = tmp_path / "data.parquet"
        data_file.write_bytes(b"dummy")

        cfg = SACRetrainConfig(
            ohlcv_path=str(data_file),
            model_path=tmp_path / "model.zip",
            buffer_path=tmp_path / "buffer.pkl",
            signal_path=tmp_path / "signal.json",
            history_path=tmp_path / "history.jsonl",
        )

        mock_env = _make_mock_env()
        mock_model = _make_mock_model()
        mock_model.learn.side_effect = RuntimeError("training crash")

        with (
            patch("scripts.v460.ml.sac_retrain_scheduler._create_env", return_value=mock_env),
            patch("scripts.v460.lib.data_loader.load_parquet", return_value=_MOCK_OHLCV_DF),
            _mock_sb3_import(mock_model),
            patch(
                "scripts.v460.ml.sac_retrain_scheduler.cleanup_training_resources"
            ) as mock_cleanup,
        ):
            result = retrain_once(cfg)

        assert result.status == "error"
        mock_cleanup.assert_called_once()
