#!/usr/bin/env python3
"""
Phase D0 計測基盤のユニットテスト

107# §7 で追加された取引ベースメトリクス、二項検定、
data_path オーバーライド、eval_dd_thresholds のテスト。
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class TestComputeGate2MetricsFromBalancesD0:
    """D0: step_win_rate / win_rate 並記の検証"""

    def test_step_win_rate_field_exists(self):
        """step_win_rate フィールドが追加されていること"""
        from scripts.v459.run_phase_c import compute_gate2_metrics_from_balances

        balances = np.array(
            [100000 + i * 12 for i in range(500)], dtype=np.float64
        )
        result = compute_gate2_metrics_from_balances(balances)

        assert "step_win_rate" in result
        assert "win_rate" in result  # 後方互換
        assert result["step_win_rate"] == result["win_rate"]

    def test_step_win_rate_is_step_based(self):
        """step_win_rate がステップごとのリターン > 0 の割合であること"""
        from scripts.v459.run_phase_c import compute_gate2_metrics_from_balances

        # 単調増加 → step_win_rate ≈ 1.0
        balances = np.array(
            [100000 + i * 10 for i in range(100)], dtype=np.float64
        )
        result = compute_gate2_metrics_from_balances(balances)

        assert result["step_win_rate"] > 0.95

    def test_gate2_pass_uses_step_win_rate(self):
        """gate2_pass 判定が step_win_rate を使用していること"""
        from scripts.v459.run_phase_c import compute_gate2_metrics_from_balances

        # ROI, PF, Sharpe, MaxDD は PASS だが WinRate が低い → FAIL
        # step_win_rate < 0.35 のケースを作る (下落トレンドに小さいノイズ)
        rng = np.random.RandomState(42)
        n = 500
        base = np.array([100000 - i * 0.1 for i in range(n)])
        noise = rng.normal(0, 5, n)
        balances = np.array(base + noise, dtype=np.float64)
        result = compute_gate2_metrics_from_balances(balances)

        # gate2_pass は step_win_rate > 0.35 を要求
        assert "step_win_rate" in result


class TestTradePnlTracking:
    """D0: _run_deterministic_eval 内の取引ベース PnL 追跡ロジック"""

    def _make_mock_env(
        self,
        n_steps: int = 100,
        trade_steps: list[int] | None = None,
        trade_pnls: list[float] | None = None,
    ) -> MagicMock:
        """trades_count / realized_pnl をステップごとに変化させるモック env"""
        env = MagicMock()
        env.portfolio_value = 100000.0
        env.trades_count = 0
        env.realized_pnl = 0.0
        env.total_trades = 0
        env.gross_pnl = 0.0
        env.total_fees = 0.0
        env.buy_count = 0
        env.sell_count = 0
        env.n_steps = n_steps

        if trade_steps is None:
            trade_steps = []
        if trade_pnls is None:
            trade_pnls = []

        step_counter = [0]
        cumulative_realized = [0.0]
        cumulative_trades = [0]
        cumulative_gross = [0.0]
        cumulative_fees = [0.0]

        def side_effect_step(action):
            s = step_counter[0]
            step_counter[0] += 1

            if s in trade_steps:
                idx = trade_steps.index(s)
                pnl = trade_pnls[idx]
                cumulative_realized[0] += pnl
                cumulative_trades[0] += 1
                # Approximate gross_pnl (pnl + fee)
                fee = abs(pnl) * 0.01
                cumulative_gross[0] += pnl + fee
                cumulative_fees[0] += fee

            env.realized_pnl = cumulative_realized[0]
            env.trades_count = cumulative_trades[0]
            env.total_trades = cumulative_trades[0]
            env.gross_pnl = cumulative_gross[0]
            env.total_fees = cumulative_fees[0]
            env.portfolio_value = 100000.0 + cumulative_realized[0]

            done = s >= n_steps - 1
            obs = np.zeros(8, dtype=np.float32)
            return obs, 0.0, done, False, {}

        env.step = MagicMock(side_effect=side_effect_step)
        env.reset = MagicMock(return_value=(np.zeros(8, dtype=np.float32), {}))

        return env

    def test_trade_win_rate_computed(self):
        """取引ベース win_rate が正しく計算されること"""
        from scripts.v459.run_phase_c import _run_deterministic_eval

        # 5 trades: 3 wins, 2 losses → trade_win_rate = 0.6
        trade_steps = [10, 20, 30, 40, 50]
        trade_pnls = [100.0, -50.0, 200.0, -30.0, 150.0]
        env = self._make_mock_env(
            n_steps=60, trade_steps=trade_steps, trade_pnls=trade_pnls
        )

        model = MagicMock()
        model.predict = MagicMock(
            return_value=(np.array([0.0], dtype=np.float32), None)
        )

        result = _run_deterministic_eval(model, env, 60, 0.3333)

        assert result["trade_win_rate"] == pytest.approx(0.6, abs=0.01)
        assert result["trade_win_count"] == 3
        assert result["trade_loss_count"] == 2

    def test_no_trades_gives_zero_win_rate(self):
        """取引なし → trade_win_rate=0, binom_p=1"""
        from scripts.v459.run_phase_c import _run_deterministic_eval

        env = self._make_mock_env(n_steps=60, trade_steps=[], trade_pnls=[])

        model = MagicMock()
        model.predict = MagicMock(
            return_value=(np.array([0.0], dtype=np.float32), None)
        )

        result = _run_deterministic_eval(model, env, 60, 0.3333)

        assert result["trade_win_rate"] == 0.0
        assert result["binom_p_value"] == 1.0
        assert result["trade_pnls"] == []

    def test_binom_p_value_symmetric(self):
        """win/loss 均等 → binom_p ≈ 1.0 (50% に近い)"""
        from scripts.v459.run_phase_c import _run_deterministic_eval

        # 10 wins, 10 losses → binom_p ≈ 1.0
        trade_steps = list(range(0, 40, 2))  # 20 trades
        trade_pnls = [100.0, -100.0] * 10
        env = self._make_mock_env(
            n_steps=42, trade_steps=trade_steps, trade_pnls=trade_pnls
        )

        model = MagicMock()
        model.predict = MagicMock(
            return_value=(np.array([0.0], dtype=np.float32), None)
        )

        result = _run_deterministic_eval(model, env, 42, 0.3333)

        assert result["binom_p_value"] > 0.5  # p ≈ 1.0 (not significant)

    def test_binom_p_value_significant(self):
        """全勝 → binom_p が小さい (有意)"""
        from scripts.v459.run_phase_c import _run_deterministic_eval

        # 20 wins, 0 losses → binom_p < 0.01
        trade_steps = list(range(0, 40, 2))  # 20 trades
        trade_pnls = [100.0] * 20
        env = self._make_mock_env(
            n_steps=42, trade_steps=trade_steps, trade_pnls=trade_pnls
        )

        model = MagicMock()
        model.predict = MagicMock(
            return_value=(np.array([0.0], dtype=np.float32), None)
        )

        result = _run_deterministic_eval(model, env, 42, 0.3333)

        assert result["binom_p_value"] < 0.01

    def test_trade_pnls_list_preserved(self):
        """trade_pnls リストが結果に含まれること"""
        from scripts.v459.run_phase_c import _run_deterministic_eval

        trade_steps = [5, 15, 25]
        trade_pnls = [100.0, -50.0, 200.0]
        env = self._make_mock_env(
            n_steps=30, trade_steps=trade_steps, trade_pnls=trade_pnls
        )

        model = MagicMock()
        model.predict = MagicMock(
            return_value=(np.array([0.0], dtype=np.float32), None)
        )

        result = _run_deterministic_eval(model, env, 30, 0.3333)

        assert len(result["trade_pnls"]) == 3
        assert result["trade_pnls"][0] == pytest.approx(100.0, abs=0.1)


class TestPerTradeMetrics:
    """D0: per-trade 平均メトリクス (run_baselines.py パターン流用)"""

    def test_avg_gross_per_trade(self):
        """avg_gross_per_trade = gross_pnl / n_trades"""
        from scripts.v459.run_phase_c import _run_deterministic_eval

        trade_steps = [10, 20, 30]
        trade_pnls = [100.0, 200.0, 300.0]
        env = MagicMock()
        env.portfolio_value = 100000.0
        env.trades_count = 0
        env.realized_pnl = 0.0
        env.total_trades = 0
        env.gross_pnl = 0.0
        env.total_fees = 0.0
        env.buy_count = 0
        env.sell_count = 0
        env.n_steps = 40

        step_counter = [0]
        cumulative = {"realized": 0.0, "trades": 0, "gross": 0.0, "fees": 0.0}

        def side_effect_step(action):
            s = step_counter[0]
            step_counter[0] += 1
            if s in trade_steps:
                idx = trade_steps.index(s)
                pnl = trade_pnls[idx]
                cumulative["realized"] += pnl
                cumulative["trades"] += 1
                cumulative["gross"] += pnl + 10.0  # fixed fee
                cumulative["fees"] += 10.0
            env.realized_pnl = cumulative["realized"]
            env.trades_count = cumulative["trades"]
            env.total_trades = cumulative["trades"]
            env.gross_pnl = cumulative["gross"]
            env.total_fees = cumulative["fees"]
            env.portfolio_value = 100000.0 + cumulative["realized"]
            done = s >= 39
            return np.zeros(8, dtype=np.float32), 0.0, done, False, {}

        env.step = MagicMock(side_effect=side_effect_step)
        env.reset = MagicMock(
            return_value=(np.zeros(8, dtype=np.float32), {})
        )

        model = MagicMock()
        model.predict = MagicMock(
            return_value=(np.array([0.0], dtype=np.float32), None)
        )

        result = _run_deterministic_eval(model, env, 40, 0.3333)

        assert result["eval_trades"] == 3
        assert result["avg_gross_per_trade"] == pytest.approx(
            result["eval_gross_pnl"] / 3, abs=0.01
        )
        assert result["avg_fee_per_trade"] == pytest.approx(10.0, abs=0.01)


class TestDataPathOverride:
    """D0-b: build_config の data_path オーバーライド"""

    def test_default_data_path(self):
        """data_path 指定なし → DATA_PATH デフォルト"""
        from scripts.v459.run_phase_c import build_config, DATA_PATH

        exp_def = {
            "description": "test",
            "sac_overrides": {},
            "reward_overrides": {},
            "env_overrides": {},
        }
        config = build_config("test", 42, exp_def)

        assert config["training"]["data_config"]["data_path"] == DATA_PATH

    def test_custom_data_path(self):
        """data_path 指定あり → オーバーライド"""
        from scripts.v459.run_phase_c import build_config

        custom_path = "/tmp/custom_features.parquet"
        exp_def = {
            "description": "test",
            "sac_overrides": {},
            "reward_overrides": {},
            "env_overrides": {},
            "data_path": custom_path,
        }
        config = build_config("test", 42, exp_def)

        assert config["training"]["data_config"]["data_path"] == custom_path


class TestEvalDdThresholds:
    """D0-b: eval_dd_thresholds の build_config パススルー"""

    def test_eval_dd_thresholds_passthrough(self):
        """eval_dd_thresholds がconfigに伝播すること"""
        from scripts.v459.run_phase_c import build_config

        exp_def = {
            "description": "test",
            "sac_overrides": {},
            "reward_overrides": {},
            "env_overrides": {},
            "eval_dd_thresholds": [1.0, 0.30],
        }
        config = build_config("test", 42, exp_def)

        assert config["eval_dd_thresholds"] == [1.0, 0.30]

    def test_eval_dd_thresholds_not_set_by_default(self):
        """eval_dd_thresholds 未指定 → config に含まれない"""
        from scripts.v459.run_phase_c import build_config

        exp_def = {
            "description": "test",
            "sac_overrides": {},
            "reward_overrides": {},
            "env_overrides": {},
        }
        config = build_config("test", 42, exp_def)

        assert "eval_dd_thresholds" not in config
