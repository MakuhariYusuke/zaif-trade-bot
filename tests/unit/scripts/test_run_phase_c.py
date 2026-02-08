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
        """balance履歴を持つモックenv (unwrap済み相当)"""
        from collections import deque
        env = MagicMock()
        # statistics_calculator.portfolio_value_history (優先パス)
        env.statistics_calculator = MagicMock()
        env.statistics_calculator.portfolio_value_history = deque(balances)
        return env

    def test_profitable_series_passes_gate2(self):
        """利益が出るbalance列 → Gate2 PASS"""
        from scripts.v459.run_phase_c import compute_gate2_metrics

        n = 500
        balances = [100000 + i * 12 for i in range(n)]
        mock_env = self._make_mock_env(balances)

        result = compute_gate2_metrics(mock_env)

        assert result["gate2_available"] is True
        assert result["mtm_roi"] > 5.0
        assert result["profit_factor"] > 1.0
        assert result["win_rate"] > 0.5

    def test_losing_series_fails_gate2(self):
        """損失balance列 → Gate2 FAIL"""
        from scripts.v459.run_phase_c import compute_gate2_metrics

        n = 500
        balances = [100000 - i * 40 for i in range(n)]
        mock_env = self._make_mock_env(balances)

        result = compute_gate2_metrics(mock_env)

        assert result["gate2_available"] is True
        assert result["gate2_pass"] is False
        assert result["mtm_roi"] < 0

    def test_flat_series(self):
        """横ばいbalance → PF/Sharpe低い"""
        from scripts.v459.run_phase_c import compute_gate2_metrics

        n = 500
        rng = np.random.RandomState(42)
        noise = rng.normal(0, 10, n)
        balances = [100000 + noise[i] for i in range(n)]
        mock_env = self._make_mock_env(balances)

        result = compute_gate2_metrics(mock_env)

        assert result["gate2_available"] is True
        assert result["gate2_pass"] is False
        assert abs(result["mtm_roi"]) < 5.0

    def test_insufficient_data(self):
        """データ不足 → gate2_available=False"""
        from scripts.v459.run_phase_c import compute_gate2_metrics

        mock_env = self._make_mock_env([100000, 100001])
        result = compute_gate2_metrics(mock_env)
        assert result["gate2_available"] is False

    def test_none_env(self):
        """env=None → gate2_available=False"""
        from scripts.v459.run_phase_c import compute_gate2_metrics

        result = compute_gate2_metrics(None)
        assert result["gate2_available"] is False


class TestComputeGate2MetricsFromBalances:
    """compute_gate2_metrics_from_balances (新しいbalance直接入力API) の検証"""

    def test_profitable_balances(self):
        from scripts.v459.run_phase_c import compute_gate2_metrics_from_balances

        n = 500
        balances = np.array([100000 + i * 12 for i in range(n)], dtype=np.float64)
        result = compute_gate2_metrics_from_balances(balances)

        assert result["gate2_available"] is True
        assert result["mtm_roi"] > 5.0
        assert result["profit_factor"] > 1.0
        assert result["win_rate"] > 0.5
        assert result["balance_samples"] == 500

    def test_losing_balances(self):
        from scripts.v459.run_phase_c import compute_gate2_metrics_from_balances

        n = 500
        balances = np.array([100000 - i * 40 for i in range(n)], dtype=np.float64)
        result = compute_gate2_metrics_from_balances(balances)

        assert result["gate2_available"] is True
        assert result["gate2_pass"] is False
        assert result["mtm_roi"] < 0

    def test_insufficient_data(self):
        from scripts.v459.run_phase_c import compute_gate2_metrics_from_balances

        result = compute_gate2_metrics_from_balances(np.array([100000.0, 100001.0]))
        assert result["gate2_available"] is False

    def test_none_input(self):
        from scripts.v459.run_phase_c import compute_gate2_metrics_from_balances

        result = compute_gate2_metrics_from_balances(None)
        assert result["gate2_available"] is False

    def test_max_drawdown_is_negative(self):
        """max_drawdownは負値を返す"""
        from scripts.v459.run_phase_c import compute_gate2_metrics_from_balances

        balances = np.array([100000 + i * 12 - (50 if i % 10 == 5 else 0) for i in range(500)], dtype=np.float64)
        result = compute_gate2_metrics_from_balances(balances)

        assert result["gate2_available"] is True
        assert result["max_drawdown"] <= 0


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
        assert cfg["training"]["environment"]["feature_set"] == "v451"  # MTF無効化
        assert cfg["training"]["environment"]["correlation_reduction"] is False

    def test_build_config_gamma_override(self):
        from scripts.v459.run_phase_c import build_config, get_experiment_configs

        configs = get_experiment_configs()
        cfg = build_config("test", 42, configs["c1_gamma_080"])

        assert cfg["training"]["sac_hyperparameters"]["gamma"] == 0.80

    def test_eval_steps_capped_by_total_timesteps(self):
        """_deterministic_eval_gate2のmax_eval_stepsがtotal_timestepsでキャップされる"""
        from scripts.v459.run_phase_c import build_config, get_experiment_configs, TOTAL_TIMESTEPS

        configs = get_experiment_configs()
        cfg = build_config("test", 42, configs["c0_baseline_p1"])

        # build_config出力のtotal_timestepsはTOTAL_TIMESTEPS定数に一致
        assert cfg["training"]["total_timesteps"] == TOTAL_TIMESTEPS
        # eval側のキャップはこの値を使う（n_stepsは1.2Mだがcapされる）
        assert TOTAL_TIMESTEPS <= 100000  # 合理的上限

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
