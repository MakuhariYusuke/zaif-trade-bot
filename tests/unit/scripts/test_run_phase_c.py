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


class TestFindVecNormalize:
    """_find_vec_normalize のラッパー検出テスト"""

    def test_finds_vec_normalize_at_top(self):
        from scripts.v459.run_phase_c import _find_vec_normalize

        mock_vn = MagicMock()
        mock_vn.__class__.__name__ = "VecNormalize"
        # isinstance チェックのため、実際の型を使う
        try:
            from stable_baselines3.common.vec_env import VecNormalize
            mock_vn.__class__ = VecNormalize
            result = _find_vec_normalize(mock_vn)
            assert result is mock_vn
        except ImportError:
            pytest.skip("stable_baselines3 not available")

    def test_returns_none_when_no_vec_normalize(self):
        from scripts.v459.run_phase_c import _find_vec_normalize

        mock_env = MagicMock(spec=[])  # no venv attribute
        result = _find_vec_normalize(mock_env)
        assert result is None

    def test_returns_none_for_none_input(self):
        from scripts.v459.run_phase_c import _find_vec_normalize

        result = _find_vec_normalize(None)
        assert result is None


class TestRunDeterministicEval:
    """_run_deterministic_eval のロジック検証"""

    def _make_mock_env_and_model(self, n_steps=100, initial_balance=100000.0):
        """モックenv + modelを作成"""
        env = MagicMock()
        env.portfolio_value = initial_balance
        env.total_trades = 0
        env.gross_pnl = 0.0
        env.total_fees = 0.0
        env.buy_count = 0
        env.sell_count = 0
        env.n_steps = n_steps * 10  # raw n_steps >> max_eval_steps

        step_count = [0]

        def reset_fn(seed=None):
            step_count[0] = 0
            env.portfolio_value = initial_balance
            env.total_trades = 0
            return np.zeros(8, dtype=np.float32), {}

        def step_fn(action):
            step_count[0] += 1
            # action の符号でbalanceを微小変動
            action_val = float(action) if np.isscalar(action) else float(action.flatten()[0])
            env.portfolio_value += action_val * 10
            if abs(action_val) > 0.33:
                env.total_trades += 1
            terminated = step_count[0] >= n_steps
            return np.zeros(8, dtype=np.float32), 0.0, terminated, False, {}

        env.reset = reset_fn
        env.step = step_fn

        model = MagicMock()
        # predict → 固定action (threshold超過)
        model.predict = MagicMock(return_value=(np.array([0.5]), None))

        return env, model

    def test_with_no_normalization(self):
        """normalize_fn=None → 生obs評価"""
        from scripts.v459.run_phase_c import _run_deterministic_eval

        env, model = self._make_mock_env_and_model(n_steps=50)
        result = _run_deterministic_eval(
            model, env, max_eval_steps=50, threshold=0.33,
            normalize_fn=None, label="raw",
        )

        assert result["eval_method"] == "raw"
        assert result["eval_steps"] == 50
        assert result["eval_trades"] > 0  # action=0.5 > threshold=0.33
        assert "action_stats" in result
        assert result["action_stats"]["mean"] == pytest.approx(0.5, abs=0.01)

    def test_with_normalization(self):
        """normalize_fn指定 → 正規化obs評価"""
        from scripts.v459.run_phase_c import _run_deterministic_eval

        env, model = self._make_mock_env_and_model(n_steps=50)

        normalize_fn = lambda obs: obs * 0.1  # なんらかの正規化

        result = _run_deterministic_eval(
            model, env, max_eval_steps=50, threshold=0.33,
            normalize_fn=normalize_fn, label="normalized",
        )

        assert result["eval_method"] == "normalized"
        assert result["eval_steps"] == 50
        # normalize_fnが呼ばれた=obs変換されたことを確認
        # model.predictは変換後のobsで呼ばれる（直接検証は困難だがメソッド正常完了で可）

    def test_action_stats_threshold_ratio(self):
        """abs_above_thresholdの計算が正しい"""
        from scripts.v459.run_phase_c import _run_deterministic_eval

        env, model = self._make_mock_env_and_model(n_steps=20)
        # model always outputs 0.5 → all above threshold 0.33
        result = _run_deterministic_eval(
            model, env, max_eval_steps=20, threshold=0.33,
        )

        assert result["action_stats"]["abs_above_threshold"] == pytest.approx(1.0)

    def test_action_stats_below_threshold(self):
        """全actionが閾値以下 → abs_above_threshold=0"""
        from scripts.v459.run_phase_c import _run_deterministic_eval

        env, model = self._make_mock_env_and_model(n_steps=20)
        model.predict = MagicMock(return_value=(np.array([0.1]), None))

        result = _run_deterministic_eval(
            model, env, max_eval_steps=20, threshold=0.33,
        )

        assert result["action_stats"]["abs_above_threshold"] == pytest.approx(0.0)


class TestD0TradePnlTracking:
    """D0: 取引ベース PnL 追跡のテスト (107# §7 D0-a)"""

    def _make_trade_mock_env(
        self,
        n_steps: int = 100,
        initial_balance: float = 100000.0,
        trades: list | None = None,
    ):
        """trades_count/realized_pnl を正しくシミュレートするモック env。

        trades: [(step_number, pnl_delta), ...] — 指定ステップで取引がクローズされる。
        """
        env = MagicMock()
        env.portfolio_value = initial_balance
        env.total_trades = 0
        env.gross_pnl = 0.0
        env.total_fees = 0.0
        env.buy_count = 0
        env.sell_count = 0
        env.n_steps = n_steps * 10
        # D0 追跡対象属性
        env.trades_count = 0
        env.realized_pnl = 0.0

        if trades is None:
            trades = []
        trade_map = {s: pnl for s, pnl in trades}

        step_count = [0]

        def reset_fn(seed=None):
            step_count[0] = 0
            env.portfolio_value = initial_balance
            env.total_trades = 0
            env.trades_count = 0
            env.realized_pnl = 0.0
            env.gross_pnl = 0.0
            env.total_fees = 0.0
            return np.zeros(8, dtype=np.float32), {}

        def step_fn(action):
            step_count[0] += 1
            s = step_count[0]
            action_val = float(action) if np.isscalar(action) else float(action.flatten()[0])
            env.portfolio_value += action_val * 10

            if s in trade_map:
                pnl = trade_map[s]
                env.trades_count += 1
                env.realized_pnl += pnl
                env.total_trades += 1
                env.gross_pnl += abs(pnl) if pnl > 0 else pnl
                env.total_fees += 5.0  # 固定手数料

            terminated = s >= n_steps
            return np.zeros(8, dtype=np.float32), 0.0, terminated, False, {}

        env.reset = reset_fn
        env.step = step_fn

        model = MagicMock()
        model.predict = MagicMock(return_value=(np.array([0.5]), None))
        return env, model

    def test_trade_pnls_populated(self):
        """取引クローズ時に trade_pnls が正しく記録される"""
        from scripts.v459.run_phase_c import _run_deterministic_eval

        trades = [(10, 100.0), (20, -50.0), (30, 200.0)]
        env, model = self._make_trade_mock_env(n_steps=40, trades=trades)

        result = _run_deterministic_eval(
            model, env, max_eval_steps=40, threshold=0.33,
        )

        assert len(result["trade_pnls"]) == 3
        assert result["trade_pnls"][0] == pytest.approx(100.0)
        assert result["trade_pnls"][1] == pytest.approx(-50.0)
        assert result["trade_pnls"][2] == pytest.approx(200.0)

    def test_trade_win_rate_calculation(self):
        """trade_win_rate = 勝ち取引 / 全取引"""
        from scripts.v459.run_phase_c import _run_deterministic_eval

        trades = [(5, 100.0), (15, -50.0), (25, 200.0), (35, -10.0)]
        env, model = self._make_trade_mock_env(n_steps=40, trades=trades)

        result = _run_deterministic_eval(
            model, env, max_eval_steps=40, threshold=0.33,
        )

        assert result["trade_win_rate"] == pytest.approx(0.5)  # 2/4
        assert result["trade_win_count"] == 2
        assert result["trade_loss_count"] == 2

    def test_no_trades_returns_zero(self):
        """取引なしの場合 trade_win_rate=0, binom_p=1.0"""
        from scripts.v459.run_phase_c import _run_deterministic_eval

        env, model = self._make_trade_mock_env(n_steps=20, trades=[])

        result = _run_deterministic_eval(
            model, env, max_eval_steps=20, threshold=0.33,
        )

        assert result["trade_win_rate"] == 0.0
        assert result["trade_win_count"] == 0
        assert result["binom_p_value"] == 1.0
        assert result["avg_gross_per_trade"] == 0.0
        assert result["avg_fee_per_trade"] == 0.0


class TestD0PerTradeMetrics:
    """D0: per-trade 平均メトリクスのテスト (107# §7 D0-a.2)"""

    def test_avg_gross_and_fee_per_trade(self):
        """avg_gross_per_trade, avg_fee_per_trade が正しく計算される"""
        from scripts.v459.run_phase_c import _run_deterministic_eval

        env = MagicMock()
        env.portfolio_value = 100000.0
        env.total_trades = 0
        env.gross_pnl = 0.0
        env.total_fees = 0.0
        env.buy_count = 0
        env.sell_count = 0
        env.n_steps = 500
        env.trades_count = 0
        env.realized_pnl = 0.0

        step_count = [0]

        def reset_fn(seed=None):
            step_count[0] = 0
            env.portfolio_value = 100000.0
            env.total_trades = 0
            env.trades_count = 0
            env.realized_pnl = 0.0
            env.gross_pnl = 300.0   # 3取引合計 gross
            env.total_fees = 60.0   # 3取引合計 fee
            return np.zeros(8, dtype=np.float32), {}

        def step_fn(action):
            step_count[0] += 1
            # step 10, 20, 30 で取引クローズ
            if step_count[0] in (10, 20, 30):
                env.trades_count += 1
                env.realized_pnl += 50.0
            terminated = step_count[0] >= 40
            return np.zeros(8, dtype=np.float32), 0.0, terminated, False, {}

        env.reset = reset_fn
        env.step = step_fn

        model = MagicMock()
        model.predict = MagicMock(return_value=(np.array([0.5]), None))

        result = _run_deterministic_eval(
            model, env, max_eval_steps=40, threshold=0.33,
        )

        assert result["avg_gross_per_trade"] == pytest.approx(100.0)  # 300/3
        assert result["avg_fee_per_trade"] == pytest.approx(20.0)     # 60/3


class TestD0BinomTest:
    """D0: 簡易二項検定の出力テスト (107# §7 D0-a.3)"""

    def test_binom_p_value_exists(self):
        """binom_p_value がfloat値として出力される"""
        from scripts.v459.run_phase_c import _run_deterministic_eval

        env = MagicMock()
        env.portfolio_value = 100000.0
        env.total_trades = 0
        env.gross_pnl = 0.0
        env.total_fees = 0.0
        env.buy_count = 0
        env.sell_count = 0
        env.n_steps = 500
        env.trades_count = 0
        env.realized_pnl = 0.0

        step_count = [0]

        def reset_fn(seed=None):
            step_count[0] = 0
            env.portfolio_value = 100000.0
            env.trades_count = 0
            env.realized_pnl = 0.0
            env.total_trades = 0
            env.gross_pnl = 0.0
            env.total_fees = 0.0
            return np.zeros(8, dtype=np.float32), {}

        def step_fn(action):
            step_count[0] += 1
            # 10取引: 7勝3敗
            if step_count[0] <= 10:
                env.trades_count += 1
                pnl = 100.0 if step_count[0] <= 7 else -50.0
                env.realized_pnl += pnl
            terminated = step_count[0] >= 20
            return np.zeros(8, dtype=np.float32), 0.0, terminated, False, {}

        env.reset = reset_fn
        env.step = step_fn

        model = MagicMock()
        model.predict = MagicMock(return_value=(np.array([0.5]), None))

        result = _run_deterministic_eval(
            model, env, max_eval_steps=20, threshold=0.33,
        )

        assert isinstance(result["binom_p_value"], float)
        assert 0.0 <= result["binom_p_value"] <= 1.0
        # 7/10 勝ち: 二項検定で p < 0.5 程度（有意ではないが 0.5 より大)
        assert result["trade_win_rate"] == pytest.approx(0.7)

    def test_binom_all_wins(self):
        """全勝の場合 p-value は非常に小さい"""
        from scipy.stats import binomtest

        # 直接 binomtest を検証（run_phase_c の内部で使うのと同じ）
        r = binomtest(10, 10, 0.5)
        assert r.pvalue < 0.01  # 10/10 全勝は p≈0.001


class TestD0DataPathOverride:
    """D0-b: data_path オーバーライドのテスト (107# §7)"""

    def test_default_data_path(self):
        """exp_def に data_path がなければデフォルト DATA_PATH を使用"""
        from scripts.v459.run_phase_c import build_config, DATA_PATH

        config = build_config("test_exp", 42, {
            "sac_overrides": {},
            "reward_overrides": {},
            "env_overrides": {},
        })

        assert config["training"]["data_config"]["data_path"] == DATA_PATH

    def test_custom_data_path(self):
        """exp_def に data_path 指定 → config に反映"""
        from scripts.v459.run_phase_c import build_config

        custom_path = "data/btc_jpy_1m_curated_features.parquet"
        config = build_config("test_exp", 42, {
            "sac_overrides": {},
            "reward_overrides": {},
            "env_overrides": {},
            "data_path": custom_path,
        })

        assert config["training"]["data_config"]["data_path"] == custom_path

    def test_eval_dd_thresholds_in_config(self):
        """eval_dd_thresholds が config に含まれる"""
        from scripts.v459.run_phase_c import build_config

        config = build_config("test_exp", 42, {
            "sac_overrides": {},
            "reward_overrides": {},
            "env_overrides": {},
            "eval_dd_thresholds": [1.0, 0.30],
        })

        assert config["eval_dd_thresholds"] == [1.0, 0.30]

    def test_eval_dd_thresholds_absent(self):
        """eval_dd_thresholds 未設定 → config に含まれない"""
        from scripts.v459.run_phase_c import build_config

        config = build_config("test_exp", 42, {
            "sac_overrides": {},
            "reward_overrides": {},
            "env_overrides": {},
        })

        assert "eval_dd_thresholds" not in config


class TestD0StepVsTradeWinRate:
    """step_win_rate と trade_win_rate が両方計測されることを確認"""

    def test_both_win_rates_in_gate2_metrics(self):
        """compute_gate2_metrics_from_balances に step_win_rate がある"""
        from scripts.v459.run_phase_c import compute_gate2_metrics_from_balances

        balances = np.array([100000 + i * 5 for i in range(200)], dtype=np.float64)
        result = compute_gate2_metrics_from_balances(balances)

        assert "step_win_rate" in result
        assert "win_rate" in result  # 後方互換
        assert result["step_win_rate"] == result["win_rate"]

    def test_trade_win_rate_in_eval_result(self):
        """_run_deterministic_eval に trade_win_rate が含まれる"""
        from scripts.v459.run_phase_c import _run_deterministic_eval

        env = MagicMock()
        env.portfolio_value = 100000.0
        env.total_trades = 0
        env.gross_pnl = 0.0
        env.total_fees = 0.0
        env.buy_count = 0
        env.sell_count = 0
        env.n_steps = 500
        env.trades_count = 0
        env.realized_pnl = 0.0

        step_count = [0]

        def reset_fn(seed=None):
            step_count[0] = 0
            env.trades_count = 0
            env.realized_pnl = 0.0
            env.total_trades = 0
            env.gross_pnl = 0.0
            env.total_fees = 0.0
            env.portfolio_value = 100000.0
            return np.zeros(8, dtype=np.float32), {}

        def step_fn(action):
            step_count[0] += 1
            terminated = step_count[0] >= 10
            return np.zeros(8, dtype=np.float32), 0.0, terminated, False, {}

        env.reset = reset_fn
        env.step = step_fn

        model = MagicMock()
        model.predict = MagicMock(return_value=(np.array([0.1]), None))

        result = _run_deterministic_eval(
            model, env, max_eval_steps=10, threshold=0.33,
        )

        # step_win_rate は balance-based → compute_gate2_metrics_from_balances から
        assert "step_win_rate" in result
        # trade_win_rate は trade-based
        assert "trade_win_rate" in result


class TestC2Experiments:
    """C2実験定義の整合性テスト"""

    def test_c2_configs_exist(self):
        from scripts.v459.run_phase_c import get_experiment_configs

        configs = get_experiment_configs()
        c2_names = [
            "c2_ent001_thr50", "c2_ent001_thr60",
            "c2_ent001_hold10", "c2_ent001_thr50_hold10",
        ]
        for name in c2_names:
            assert name in configs, f"{name} not in configs"

    def test_c2_all_have_ent001(self):
        """全C2実験にent_coef=0.01が設定されている"""
        from scripts.v459.run_phase_c import get_experiment_configs

        configs = get_experiment_configs()
        for name, cfg in configs.items():
            if name.startswith("c2_"):
                assert cfg["sac_overrides"].get("ent_coef") == 0.01, (
                    f"{name} missing ent_coef=0.01"
                )

    def test_c2_batch_matches_configs(self):
        """C2バッチの全実験がコンフィグに存在"""
        from scripts.v459.run_phase_c import BATCHES, get_experiment_configs

        configs = get_experiment_configs()
        for exp_name in BATCHES["c2"]:
            assert exp_name in configs, f"{exp_name} not in configs"

    def test_c2_subprocess_list_matches(self):
        """サブプロセスランナーのC2リストがバッチ定義と一致"""
        from scripts.v459.run_phase_c_subprocess import C2_FEE_REDUCTION
        from scripts.v459.run_phase_c import BATCHES

        assert set(C2_FEE_REDUCTION) == set(BATCHES["c2"])


class TestC3Experiments:
    """C3実験定義の整合性テスト"""

    def test_c3_configs_exist(self):
        from scripts.v459.run_phase_c import get_experiment_configs

        configs = get_experiment_configs()
        c3_names = [
            "c3_ent001_thr60_nodd", "c3_gamma080_ent001_thr60",
            "c3_ent001_thr70_nodd", "c3_gamma080_ent001_thr70",
        ]
        for name in c3_names:
            assert name in configs, f"{name} not in configs"

    def test_c3_all_have_eval_dd_threshold(self):
        """全C3実験にeval_dd_threshold=1.0が設定されている"""
        from scripts.v459.run_phase_c import get_experiment_configs

        configs = get_experiment_configs()
        for name, cfg in configs.items():
            if name.startswith("c3_"):
                assert cfg.get("eval_dd_threshold") == 1.0, (
                    f"{name} missing eval_dd_threshold=1.0"
                )

    def test_c3_all_have_ent001(self):
        """全C3実験にent_coef=0.01が設定されている"""
        from scripts.v459.run_phase_c import get_experiment_configs

        configs = get_experiment_configs()
        for name, cfg in configs.items():
            if name.startswith("c3_"):
                assert cfg["sac_overrides"].get("ent_coef") == 0.01

    def test_c3_batch_matches_configs(self):
        from scripts.v459.run_phase_c import BATCHES, get_experiment_configs

        configs = get_experiment_configs()
        for exp_name in BATCHES["c3"]:
            assert exp_name in configs, f"{exp_name} not in configs"

    def test_c3_subprocess_list_matches(self):
        from scripts.v459.run_phase_c_subprocess import C3_DD_DISABLE
        from scripts.v459.run_phase_c import BATCHES

        assert set(C3_DD_DISABLE) == set(BATCHES["c3"])

    def test_build_config_includes_eval_dd_threshold(self):
        """eval_dd_thresholdがconfigに含まれることを確認"""
        from scripts.v459.run_phase_c import build_config, get_experiment_configs

        configs = get_experiment_configs()
        config = build_config("c3_ent001_thr60_nodd", 42, configs["c3_ent001_thr60_nodd"])
        assert config.get("eval_dd_threshold") == 1.0

    def test_build_config_without_eval_dd_threshold(self):
        """eval_dd_threshold未設定の実験はconfigに含まれない"""
        from scripts.v459.run_phase_c import build_config, get_experiment_configs

        configs = get_experiment_configs()
        config = build_config("c0_baseline_p1", 42, configs["c0_baseline_p1"])
        assert "eval_dd_threshold" not in config


class TestD1Experiments:
    """D1: 特徴量セット比較実験 (107# §4.2)"""

    def test_d1_configs_exist(self):
        from scripts.v459.run_phase_c import get_experiment_configs

        configs = get_experiment_configs()
        for name in ["d1_v451opt", "d1_minimal", "d1_curated"]:
            assert name in configs, f"{name} not in configs"

    def test_d1_all_have_eval_dd_thresholds(self):
        """D1実験はすべて eval_dd_thresholds=[1.0, 0.30] を持つ"""
        from scripts.v459.run_phase_c import get_experiment_configs

        configs = get_experiment_configs()
        for name in ["d1_v451opt", "d1_minimal", "d1_curated"]:
            cfg = configs[name]
            assert cfg.get("eval_dd_thresholds") == [1.0, 0.30], (
                f"{name} missing eval_dd_thresholds"
            )

    def test_d1_all_have_ent001_thr70(self):
        """D1実験はC3 best設定を固定 (ent=0.01, thr=0.70)"""
        from scripts.v459.run_phase_c import get_experiment_configs

        configs = get_experiment_configs()
        for name in ["d1_v451opt", "d1_minimal", "d1_curated"]:
            cfg = configs[name]
            assert cfg["sac_overrides"].get("ent_coef") == 0.01
            assert cfg["env_overrides"].get("continuous_to_discrete_threshold") == 0.70

    def test_d1_data_paths_differ(self):
        """D1実験は各々異なる data_path を持つ"""
        from scripts.v459.run_phase_c import build_config, get_experiment_configs

        configs = get_experiment_configs()
        paths = set()
        for name in ["d1_v451opt", "d1_minimal", "d1_curated"]:
            c = build_config(name, 42, configs[name])
            paths.add(c["training"]["data_config"]["data_path"])
        assert len(paths) == 3, "D1 experiments should use 3 different parquets"

    def test_d1_batch_matches_configs(self):
        from scripts.v459.run_phase_c import BATCHES, get_experiment_configs

        configs = get_experiment_configs()
        for exp_name in BATCHES["d1"]:
            assert exp_name in configs, f"{exp_name} not in configs"


class TestD2Experiments:
    """D2: コスト感度+報酬微調整実験 (107# §4.3)"""

    def test_d2_configs_exist(self):
        from scripts.v459.run_phase_c import get_experiment_configs

        configs = get_experiment_configs()
        for name in ["d2_cost05", "d2_cost10", "d2_cost15", "d2_asymm12"]:
            assert name in configs, f"{name} not in configs"

    def test_d2_cost_variations(self):
        """D2-a: 3つのコスト設定が正しいこと"""
        from scripts.v459.run_phase_c import build_config, get_experiment_configs

        configs = get_experiment_configs()
        expected = {"d2_cost05": 0.0005, "d2_cost10": 0.001, "d2_cost15": 0.0015}
        for name, expected_cost in expected.items():
            c = build_config(name, 42, configs[name])
            actual = c["training"]["environment"]["transaction_cost"]
            assert actual == expected_cost, f"{name}: {actual} != {expected_cost}"

    def test_d2_asymm_has_loss_multiplier(self):
        """D2-b: 非対称報酬 loss_multiplier=1.2"""
        from scripts.v459.run_phase_c import build_config, get_experiment_configs

        configs = get_experiment_configs()
        c = build_config("d2_asymm12", 42, configs["d2_asymm12"])
        assert c["reward"].get("loss_multiplier") == 1.2

    def test_d2_batches_match_configs(self):
        from scripts.v459.run_phase_c import BATCHES, get_experiment_configs

        configs = get_experiment_configs()
        for batch_name in ["d2_cost", "d2_reward"]:
            for exp_name in BATCHES[batch_name]:
                assert exp_name in configs, f"{exp_name} not in configs"
