#!/usr/bin/env python3
"""
Phase C ランナーのユニットテスト

compute_gate2_metrics のロジックと実験定義の整合性を検証。
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestComputeGate2Metrics:
    """compute_gate2_metrics の入力→出力を検証"""

    def _make_mock_env(self, balances: list) -> MagicMock:
        """balance履歴を持つモックenv"""
        from collections import deque
        env = MagicMock()
        env.portfolio_value_history = deque(balances, maxlen=len(balances))
        # statistics_calculator は無し
        env.statistics_calculator = None
        # unwrap_env が通る経路を設定
        env.envs = None  # VecEnvでない
        env.env = None  # ラッパーでない
        return env

    def test_profitable_series_passes_gate2(self, monkeypatch):
        """利益が出るbalance列 → Gate2 PASS"""
        from scripts.v459 import run_phase_c

        # 100000 → 106000 (6% ROI) で安定上昇
        n = 500
        balances = [100000 + i * 12 for i in range(n)]
        mock_env = self._make_mock_env(balances)
        
        # unwrap_envをバイパス
        monkeypatch.setattr(run_phase_c, "unwrap_env", lambda env: mock_env)

        result = run_phase_c.compute_gate2_metrics(mock_env)

        assert result["gate2_available"] is True
        assert result["mtm_roi"] > 5.0
        assert result["profit_factor"] > 1.0
        assert result["win_rate"] > 0.5

    def test_losing_series_fails_gate2(self, monkeypatch):
        """損失balance列 → Gate2 FAIL"""
        from scripts.v459 import run_phase_c

        n = 500
        balances = [100000 - i * 40 for i in range(n)]  # -20000
        mock_env = self._make_mock_env(balances)
        monkeypatch.setattr(run_phase_c, "unwrap_env", lambda env: mock_env)

        result = run_phase_c.compute_gate2_metrics(mock_env)

        assert result["gate2_available"] is True
        assert result["gate2_pass"] is False
        assert result["mtm_roi"] < 0

    def test_flat_series(self, monkeypatch):
        """横ばいbalance → PF/Sharpe低い"""
        from scripts.v459 import run_phase_c

        n = 500
        rng = np.random.RandomState(42)
        noise = rng.normal(0, 10, n)
        balances = [100000 + noise[i] for i in range(n)]
        mock_env = self._make_mock_env(balances)
        monkeypatch.setattr(run_phase_c, "unwrap_env", lambda env: mock_env)

        result = run_phase_c.compute_gate2_metrics(mock_env)

        assert result["gate2_available"] is True
        assert result["gate2_pass"] is False
        assert abs(result["mtm_roi"]) < 5.0

    def test_insufficient_data(self, monkeypatch):
        """データ不足 → gate2_available=False"""
        from scripts.v459 import run_phase_c

        mock_env = self._make_mock_env([100000, 100001])  # 2点のみ
        monkeypatch.setattr(run_phase_c, "unwrap_env", lambda env: mock_env)
        
        result = run_phase_c.compute_gate2_metrics(mock_env)
        assert result["gate2_available"] is False


class TestExperimentConfigs:
    """実験定義の整合性チェック"""

    def test_all_configs_have_required_keys(self):
        from scripts.v459.run_phase_c import get_experiment_configs

        configs = get_experiment_configs()
        assert len(configs) > 0

        for name, cfg in configs.items():
            assert "description" in cfg, f"{name}: description missing"
            assert "sac_overrides" in cfg, f"{name}: sac_overrides missing"
            assert "reward_overrides" in cfg, f"{name}: reward_overrides missing"
            assert "env_overrides" in cfg, f"{name}: env_overrides missing"

    def test_gamma_experiments_have_correct_values(self):
        from scripts.v459.run_phase_c import get_experiment_configs

        configs = get_experiment_configs()

        assert configs["c1_gamma_080"]["sac_overrides"]["gamma"] == 0.80
        assert configs["c1_gamma_090"]["sac_overrides"]["gamma"] == 0.90
        assert configs["c1_gamma_095"]["sac_overrides"]["gamma"] == 0.95

    def test_v451_golden_uses_pnl_centered(self):
        from scripts.v459.run_phase_c import get_experiment_configs

        configs = get_experiment_configs()
        golden = configs["c1_v451_golden"]

        assert golden["sac_overrides"]["gamma"] == 0.80
        assert golden["reward_overrides"]["reward_scale"] == 1.0
        assert golden["reward_overrides"]["custom_reward_params"]["type"] == "pnl_centered"

    def test_threshold_experiments_range(self):
        from scripts.v459.run_phase_c import get_experiment_configs

        configs = get_experiment_configs()

        for t in [50, 60, 70]:
            name = f"c1_threshold_{t}"
            assert name in configs
            threshold = configs[name]["env_overrides"]["continuous_to_discrete_threshold"]
            assert 0.4 <= threshold <= 0.8

    def test_build_config_structure(self):
        from scripts.v459.run_phase_c import build_config, get_experiment_configs

        configs = get_experiment_configs()
        cfg = build_config("test", 42, configs["c0_baseline_p1"])

        assert cfg["training"]["algorithm"] == "SAC"
        assert cfg["training"]["seed"] == 42
        assert cfg["training"]["sac_hyperparameters"]["gamma"] == 0.99
        assert cfg["training"]["environment"]["transaction_cost"] == 0.001

    def test_build_config_gamma_override(self):
        from scripts.v459.run_phase_c import build_config, get_experiment_configs

        configs = get_experiment_configs()
        cfg = build_config("test", 42, configs["c1_gamma_080"])

        assert cfg["training"]["sac_hyperparameters"]["gamma"] == 0.80

    def test_batch_c0_c1_has_all_experiments(self):
        from scripts.v459.run_phase_c import BATCHES, get_experiment_configs

        configs = get_experiment_configs()
        for exp_name in BATCHES["c0_c1"]:
            assert exp_name in configs, f"{exp_name} not in experiment configs"


class TestSubprocessRunner:
    """サブプロセスランナーの定義チェック"""

    def test_c0_c1_screening_matches_batch(self):
        from scripts.v459.run_phase_c_subprocess import C0_C1_SCREENING
        from scripts.v459.run_phase_c import BATCHES

        assert set(C0_C1_SCREENING) == set(BATCHES["c0_c1"])

    def test_screening_count(self):
        from scripts.v459.run_phase_c_subprocess import C0_C1_SCREENING

        assert len(C0_C1_SCREENING) == 14
