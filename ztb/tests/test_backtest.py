#!/usr/bin/env python3
"""
バックテスト機能テストスクリプト
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# PYTHONPATH設定
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def calculate_sharpe_ratio(returns: list, risk_free_rate: float = 0.0) -> float:
    """シャープレシオを計算"""
    if len(returns) < 2:
        return 0.0

    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0.0

    return np.mean(excess_returns) / np.std(excess_returns)


def calculate_max_drawdown(values: list) -> float:
    """最大ドローダウンを計算"""
    if not values:
        return 0.0

    values_array = np.array(values)
    peak = np.maximum.accumulate(values_array)
    drawdown = (values_array - peak) / peak
    max_drawdown = np.min(drawdown)

    return abs(max_drawdown) * 100


def run_trading_backtest(
    model_path: str, data_path: str, output_dir: str = "backtest_results"
) -> dict:
    """取引バックテストを実行"""
    try:
        print(f"取引バックテスト開始: {model_path}")

        # 出力ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)

        # モデル読み込み確認（モック）
        if not os.path.exists(model_path):
            print(f"モデルファイルが見つかりません: {model_path} - モックデータを使用")

            # モックモデルを使用
            class MockModel:
                def predict(self, obs, deterministic=True):
                    return np.random.randn(1), None

            model = MockModel()
        else:
            try:
                from stable_baselines3 import SAC

                model = SAC.load(model_path)
                print("モデル読み込み完了")
            except ImportError:
                print("stable_baselines3が利用できません。モックモデルを使用")

                class MockModel:
                    def predict(self, obs, deterministic=True):
                        return np.random.randn(1), None

                model = MockModel()

        # テストデータ読み込み
        if not os.path.exists(data_path):
            print(f"テストデータが見つかりません: {data_path} - サンプルデータを作成")
            # サンプルデータ作成
            dates = pd.date_range("2023-01-01", periods=100, freq="H")
            test_data = pd.DataFrame(
                {
                    "timestamp": dates,
                    "open": np.random.uniform(3000000, 5000000, 100),
                    "high": np.random.uniform(3000000, 5000000, 100),
                    "low": np.random.uniform(3000000, 5000000, 100),
                    "close": np.random.uniform(3000000, 5000000, 100),
                    "volume": np.random.uniform(1000, 10000, 100),
                }
            )
        else:
            test_data = pd.read_csv(data_path)
        print(f"テストデータ読み込み完了: {len(test_data)} 行")

        # 環境作成（簡易実装）
        try:
            import gymnasium as gym
            from stable_baselines3.common.vec_env import DummyVecEnv
        except ImportError:
            print("必要なライブラリが利用できません。バックテストをスキップします。")
            return {"error": "必要なライブラリが利用できません"}

        class SimpleTradingEnv(gym.Env):
            def __init__(self, data):
                super().__init__()
                self.data = data
                self.current_step = 0
                self.action_space = gym.spaces.Box(
                    low=-1, high=1, shape=(1,), dtype=np.float32
                )
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
                )
                self.portfolio_value = 10000.0

            def reset(self, seed=None, options=None):
                self.current_step = 0
                self.portfolio_value = 10000.0
                return self._get_obs(), {}

            def step(self, action):
                self.current_step += 1
                reward = action * 0.01  # actionはスカラー
                self.portfolio_value += reward * 100

                done = self.current_step >= len(self.data) - 1
                return (
                    self._get_obs(),
                    reward,
                    done,
                    False,
                    {
                        "portfolio_value": self.portfolio_value,
                        "trade_executed": abs(action) > 0.1,
                    },
                )

            def _get_obs(self):
                if self.current_step >= len(self.data):
                    return np.zeros(4)
                row = self.data.iloc[self.current_step]
                return np.array(
                    [
                        row.get("close", 0),
                        row.get("volume", 0),
                        self.portfolio_value,
                        self.current_step,
                    ]
                )

        env = DummyVecEnv([lambda: SimpleTradingEnv(test_data)])

        # バックテスト実行
        obs = env.reset()
        total_reward = 0
        trades = []
        portfolio_values = []

        print("バックテスト実行中...")
        for step in range(min(len(test_data), 1000)):  # 最大1000ステップ
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            total_reward += reward[0]
            portfolio_values.append(info[0].get("portfolio_value", 0))

            if info[0].get("trade_executed", False):
                trades.append(
                    {
                        "step": step,
                        "action": action[0],
                        "reward": reward[0],
                        "portfolio_value": info[0].get("portfolio_value", 0),
                    }
                )

            if done:
                break

        # 結果計算
        results = {
            "total_reward": float(total_reward),
            "total_trades": len(trades),
            "final_portfolio_value": portfolio_values[-1] if portfolio_values else 0,
            "sharpe_ratio": calculate_sharpe_ratio(portfolio_values)
            if len(portfolio_values) > 1
            else 0,
            "max_drawdown": calculate_max_drawdown(portfolio_values)
            if portfolio_values
            else 0,
            "win_rate": len([t for t in trades if t["reward"] > 0]) / len(trades)
            if trades
            else 0,
        }

        # 結果保存
        results_path = os.path.join(output_dir, "backtest_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # 取引履歴保存
        if trades:
            trades_path = os.path.join(output_dir, "trades_history.csv")
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv(trades_path, index=False)

        # ポートフォリオ推移保存
        if portfolio_values:
            portfolio_path = os.path.join(output_dir, "portfolio_values.csv")
            pd.DataFrame(
                {"step": range(len(portfolio_values)), "value": portfolio_values}
            ).to_csv(portfolio_path, index=False)

        print(f"バックテスト完了。結果保存: {results_path}")
        print(
            f"収益化指標: 総報酬={results['total_reward']:.2f}, 取引回数={results['total_trades']}, 勝率={results['win_rate']:.2%}"
        )

        return results

    except Exception as e:
        print(f"バックテスト実行中にエラー発生: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    # テスト実行
    result = run_trading_backtest(
        model_path="models/sac_v434_2_integrated/sac_v434_2_simple.zip",
        data_path="data/btc_jpy_featured_dataset.csv",
        output_dir="test_backtest_results",
    )
    print("テスト結果:", result)
