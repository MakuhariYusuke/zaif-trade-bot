#!/usr/bin/env python3
"""
カリキュラム学習 Stage 2: バランス維持しながら通常報酬関数へ移行
"""

from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.training.training_utils import setup_project_path, create_ppo_model, load_training_data, save_model_with_path, evaluate_model
from ztb.training.ppo_config import DEFAULT_REWARD_SCALING, DEFAULT_INITIAL_PORTFOLIO_VALUE
from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.training.callbacks import SimpleTrainingCallback


def main() -> None:
    """メイン実行関数"""
    print("=== カリキュラム学習 Stage 2: バランス維持しながら通常報酬関数へ移行 ===")

    # Setup project path
    setup_project_path()

    # データ読み込み
    print("データを読み込み中...")
    df = load_training_data()
    print(f"データサイズ: {len(df)} 行")

    # 環境設定
    config = {
        "reward_scaling": DEFAULT_REWARD_SCALING,  # 最適化されたスケーリング
        "curriculum_stage": "balanced_transition",  # 新しい移行ステージ
        "max_position_size": 1.0,
        "transaction_cost": 0.0,
        "timeframe": "1m",
        "feature_set": "full",
        "initial_portfolio_value": DEFAULT_INITIAL_PORTFOLIO_VALUE,
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
    env = DummyVecEnv([lambda: env])  # type: ignore[assignment]

    # PPOモデル設定（最適化されたハイパーパラメータ）
    print("PPOモデルを作成中...")
    model_config = {
        "learning_rate": 5e-4,  # 最適化済み
        "gamma": 0.95,  # 最適化済み
        "gae_lambda": 0.8,  # 最適化済み
        "clip_range": 0.3,  # 最適化済み
        "vf_coef": 0.5,  # 最適化済み
        "max_grad_norm": 1.0,  # 最適化済み
        "target_kl": 0.005,  # 最適化済み
        "ent_coef": 0.05,  # 最適化済み
        "batch_size": 64,  # 最適化済み
        "n_epochs": 10,
        "verbose": 1,
    }
    model = create_ppo_model(env, model_config)

    # コールバック設定
    callback = SimpleTrainingCallback()

    # トレーニング実行
    print("トレーニングを開始します...")
    print("目標: バランスを維持しながら通常の利益ベース報酬を学習")
    print("バランスペナルティ: 行動分布が33%から大きく外れるとペナルティ")

    total_timesteps = 50_000  # 移行ステージのトレーニング
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    # モデル保存
    model_path = save_model_with_path(model, "curriculum_transition")
    print(f"\nモデルを保存しました: {model_path}")

    # 最終評価
    print("\n=== 最終評価 ===")
    episode_reward, step_count = evaluate_model(model, env)

    print(f"評価エピソード報酬: {episode_reward:.3f}")
    print(f"ステップ数: {step_count}")

    env.close()


if __name__ == "__main__":
    main()
