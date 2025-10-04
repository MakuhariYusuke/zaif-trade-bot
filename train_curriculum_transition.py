#!/usr/bin/env python3
"""
カリキュラム学習 Stage 2: バランス維持しながら通常報酬関数へ移行
"""

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.trading.environment import HeavyTradingEnv


class TrainingCallback(BaseCallback):
    """トレーニングコールバック - 行動分布とバランススコアを追跡"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.action_counts = [0, 0, 0]  # [HOLD, BUY, SELL]
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        # 行動をカウント（DummyVecEnvから取得）
        if hasattr(self.locals, 'actions'):
            actions = self.locals['actions']
            if len(actions.shape) > 0:
                for action in actions:
                    if action < 3:  # 有効な行動のみ
                        self.action_counts[action] += 1
            else:
                action = actions.item()
                if action < 3:
                    self.action_counts[action] += 1

        # エピソード報酬を追跡
        if 'rewards' in self.locals:
            rewards = self.locals['rewards']
            if len(rewards.shape) > 0:
                self.current_episode_reward += rewards[0]
            else:
                self.current_episode_reward += rewards

        # エピソード終了時に統計を記録
        if 'dones' in self.locals:
            dones = self.locals['dones']
            done = dones[0] if len(dones.shape) > 0 else dones
            if done:
                self.episode_rewards.append(self.current_episode_reward)
                self.current_episode_reward = 0

        return True

    def _on_training_end(self) -> None:
        """トレーニング終了時に最終統計を表示"""
        total_actions = sum(self.action_counts)
        if total_actions > 0:
            action_dist = [count / total_actions * 100 for count in self.action_counts]

            # バランススコア計算（低いほどバランスが良い）
            target_ratio = 1.0 / 3.0
            balance_score = sum(abs(ratio/100 - target_ratio) for ratio in action_dist)

            print("\n=== カリキュラム学習 Stage 2: バランス維持しながら通常報酬関数へ移行 ===")
            print(f"総行動数: {total_actions}")
            print(f"バランススコア: {balance_score:.4f}")
            print(f"HOLD: {action_dist[0]:.1f}%")
            print(f"BUY: {action_dist[1]:.1f}%")
            print(f"SELL: {action_dist[2]:.1f}%")
            print(f"バランススコア: {balance_score:.4f}")

            if self.episode_rewards:
                avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
                print(f"平均エピソード報酬: {avg_reward:.3f}")


def main():
    """メイン実行関数"""
    print("=== カリキュラム学習 Stage 2: バランス維持しながら通常報酬関数へ移行 ===")

    # データ読み込み
    print("データを読み込み中...")
    df = pd.read_csv("ml-dataset-enhanced.csv")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"データサイズ: {len(df)} 行")

    # 環境設定
    config = {
        "reward_scaling": 6.0,  # 最適化されたスケーリング
        "curriculum_stage": "balanced_transition",  # 新しい移行ステージ
        "max_position_size": 1.0,
        "transaction_cost": 0.0,
        "timeframe": "1m",
        "feature_set": "full",
        "initial_portfolio_value": 1_000_000.0,
        # 報酬設定
        "reward_position_soft_cap": 0.8,
        "reward_position_penalty_scale": 0.5,
        "reward_position_penalty_exponent": 4.0,
        "reward_inventory_window": 128,
        "reward_inventory_penalty_scale": 0.1,
        "reward_trade_frequency_penalty": 0.2,
        "reward_trade_frequency_halflife": 8.0,
        "reward_trade_cooldown_steps": 2,
        "reward_trade_cooldown_penalty": 0.2,
        "reward_max_consecutive_trades": 5,
        "reward_consecutive_trade_penalty": 0.1,
        "reward_volatility_window": 32,
        "reward_volatility_penalty_scale": 0.05,
        "reward_sharpe_bonus_scale": 0.02,
        "reward_clip_value": 2.0,
        "reward_profit_bonus_multipliers": [1.1, 1.15, 0.8],  # BUY, SELL, HOLD
        "enable_forced_diversity": False,  # 移行ステージでは無効
    }

    # 環境作成
    print("環境を作成中...")
    env = HeavyTradingEnv(df=df, config=config)
    env = DummyVecEnv([lambda: env])

    # PPOモデル設定（最適化されたハイパーパラメータ）
    print("PPOモデルを作成中...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-4,      # 最適化済み
        gamma=0.95,              # 最適化済み
        gae_lambda=0.8,          # 最適化済み
        clip_range=0.3,          # 最適化済み
        vf_coef=0.5,             # 最適化済み
        max_grad_norm=1.0,       # 最適化済み
        target_kl=0.005,         # 最適化済み
        ent_coef=0.05,           # 最適化済み
        batch_size=64,           # 最適化済み
        n_epochs=10,
        verbose=1,
        tensorboard_log="./tensorboard/",
    )

    # コールバック設定
    callback = TrainingCallback()

    # トレーニング実行
    print("トレーニングを開始します...")
    print("目標: バランスを維持しながら通常の利益ベース報酬を学習")
    print("バランスペナルティ: 行動分布が33%から大きく外れるとペナルティ")

    total_timesteps = 50_000  # 移行ステージのトレーニング
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    # モデル保存
    model_path = "models/curriculum_transition.zip"
    model.save(model_path)
    print(f"\nモデルを保存しました: {model_path}")

    # 最終評価
    print("\n=== 最終評価 ===")
    obs = env.reset()
    episode_reward = 0
    done = False
    step_count = 0

    while not done and step_count < 1000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward[0] if hasattr(reward, '__len__') else reward
        step_count += 1

        if done:
            break

    print(f"評価エピソード報酬: {episode_reward:.3f}")
    print(f"ステップ数: {step_count}")

    env.close()


if __name__ == "__main__":
    main()