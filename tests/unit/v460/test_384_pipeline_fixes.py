"""384# Pipeline fixes: CRITICAL-1/2, HIGH-1/2 の単体テスト.

382#/383# Codex/Gemini レビューで指摘された問題の修正を検証。
- CRITICAL-2: OOS 評価の max_steps 撤廃、gross_pnl 集約修正
- HIGH-2: scaler 転送 (_build_val_env_config)
- HIGH-1: seed crash → ERROR gate (test_356 で検証済み)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest


# ════════════════════════════════════════════════════════════════
# CRITICAL-2: OOS 評価修正
# ════════════════════════════════════════════════════════════════


class TestEvaluateModelOOS:
    """evaluate_model_oos の修正検証."""

    def _make_mock_env(
        self,
        n_steps: int = 100,
        portfolio_value: float = 10_500_000.0,
        initial_value: float = 10_000_000.0,
        total_pnl: float = 500_000.0,
        trades_count: int = 42,
    ) -> MagicMock:
        env = MagicMock()
        env.portfolio_value = portfolio_value
        env.initial_portfolio_value = initial_value
        env.total_pnl = total_pnl
        env.trades_count = trades_count

        # reset() → (obs, info)
        env.reset.return_value = (np.zeros(10), {})

        # step() — terminate after n_steps
        call_count = 0

        def step_fn(action: object) -> tuple:
            nonlocal call_count
            call_count += 1
            if call_count >= n_steps:
                return np.zeros(10), 1.0, True, False, {}
            return np.zeros(10), 1.0, False, False, {}

        env.step.side_effect = step_fn
        return env

    def test_default_no_max_steps_limit(self) -> None:
        """384# CRITICAL-2: max_steps_per_episode のデフォルトは None (全走査)."""
        from scripts.v460.lib.sac_common import evaluate_model_oos

        model = MagicMock()
        model.predict.return_value = (np.array([0.0]), None)

        # 500 ステップまで走る env
        env = self._make_mock_env(n_steps=500)

        result = evaluate_model_oos(model, env, n_episodes=1)

        # 全 500 ステップ走査されるべき (旧デフォルト 10K ではなく)
        assert env.step.call_count == 500
        assert result["n_episodes"] == 1

    def test_max_steps_explicit(self) -> None:
        """max_steps_per_episode を明示すれば制限される."""
        from scripts.v460.lib.sac_common import evaluate_model_oos

        model = MagicMock()
        model.predict.return_value = (np.array([0.0]), None)

        env = self._make_mock_env(n_steps=10_000)

        result = evaluate_model_oos(model, env, n_episodes=1, max_steps_per_episode=100)
        assert env.step.call_count == 100

    def test_gross_pnl_aggregated(self) -> None:
        """384# CRITICAL-2: gross_pnl が全エピソードで集約される."""
        from scripts.v460.lib.sac_common import evaluate_model_oos

        model = MagicMock()
        model.predict.return_value = (np.array([0.0]), None)

        env = self._make_mock_env(n_steps=10, total_pnl=100_000.0)
        result = evaluate_model_oos(model, env, n_episodes=2, max_steps_per_episode=10)

        # 2 エピソードの avg_pnl
        assert result["gross_pnl"] == 100_000.0  # averaged
        assert result["n_episodes"] == 2

    def test_single_episode_roi(self) -> None:
        """1 エピソードでの ROI 算出."""
        from scripts.v460.lib.sac_common import evaluate_model_oos

        model = MagicMock()
        model.predict.return_value = (np.array([0.0]), None)

        env = self._make_mock_env(
            n_steps=10,
            portfolio_value=10_500_000.0,
            initial_value=10_000_000.0,
        )
        result = evaluate_model_oos(model, env, n_episodes=1, max_steps_per_episode=10)

        assert abs(result["gross_roi"] - 0.05) < 1e-6

    def test_multi_slice_metrics_present(self) -> None:
        """425# multi-slice: 十分なステップがある場合 slice_metrics が出力される."""
        from scripts.v460.lib.sac_common import evaluate_model_oos

        model = MagicMock()
        model.predict.return_value = (np.array([0.0]), None)

        # 4320 steps で 3分割が有効 (1440×3 以上)
        env = self._make_mock_env(n_steps=4320)
        result = evaluate_model_oos(model, env, n_episodes=1)

        assert "slice_metrics" in result
        slices = result["slice_metrics"]
        assert len(slices) == 3
        labels = [s["label"] for s in slices]
        assert labels == ["early", "mid", "late"]
        # 全スライスに pf, max_drawdown キーがある
        for s in slices:
            assert "pf" in s
            assert "max_drawdown" in s

    def test_multi_slice_not_present_short_data(self) -> None:
        """425# multi-slice: ステップが少ない場合は slice_metrics がない."""
        from scripts.v460.lib.sac_common import evaluate_model_oos

        model = MagicMock()
        model.predict.return_value = (np.array([0.0]), None)

        env = self._make_mock_env(n_steps=100)
        result = evaluate_model_oos(model, env, n_episodes=1)

        assert "slice_metrics" not in result


# ════════════════════════════════════════════════════════════════
# HIGH-2: scaler 転送
# ════════════════════════════════════════════════════════════════


class TestBuildValEnvConfig:
    """_build_val_env_config が train env の scaler を正しく注入すること."""

    def test_scaler_injected(self) -> None:
        """train env の scaler_mean/std が val env config に注入される."""
        from scripts.v460.lib.tasks.sac_train import _build_val_env_config

        mock_train_env = SimpleNamespace(
            scaler_mean=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            scaler_std=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        )

        cfg: dict = {
            "environment": {"transaction_cost": 0.001},
            "features": {"selected": ["f1", "f2", "f3"]},
        }

        val_cfg = _build_val_env_config(mock_train_env, cfg)

        env_section = val_cfg.get("environment", {})
        assert "scaler_mean" in env_section
        assert "scaler_std" in env_section
        assert len(env_section["scaler_mean"]) == 3
        assert len(env_section["scaler_std"]) == 3
        assert abs(env_section["scaler_mean"][0] - 1.0) < 1e-6
        assert abs(env_section["scaler_std"][1] - 0.2) < 1e-6

    def test_no_scaler_in_train_env(self) -> None:
        """train env に scaler がない場合は警告のみ."""
        from scripts.v460.lib.tasks.sac_train import _build_val_env_config

        mock_train_env = SimpleNamespace()  # no scaler_mean/std

        cfg: dict = {"environment": {}, "features": {}}

        val_cfg = _build_val_env_config(mock_train_env, cfg)

        env_section = val_cfg.get("environment", {})
        assert "scaler_mean" not in env_section
        assert "scaler_std" not in env_section

    def test_original_cfg_not_mutated(self) -> None:
        """元の cfg が変更されないこと (deepcopy)."""
        from scripts.v460.lib.tasks.sac_train import _build_val_env_config

        mock_train_env = SimpleNamespace(
            scaler_mean=np.array([1.0], dtype=np.float32),
            scaler_std=np.array([0.5], dtype=np.float32),
        )

        cfg: dict = {"environment": {"transaction_cost": 0.001}, "features": {}}

        _build_val_env_config(mock_train_env, cfg)

        # 元の cfg に scaler が注入されていないこと
        assert "scaler_mean" not in cfg.get("environment", {})
