#!/usr/bin/env python3
"""
Phase 3: Time-Series Split 訓練・評価（簡略版）

分割データで訓練し、OOS評価を実施
既存の train_mlp_v456_phase2_complete.py をベースにしたシンプル実装
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ztb.training.action_converter_v456 import ActionAnalyzer
from ztb.config.environment_config import get_training_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Phase3Callback(BaseCallback):
    """訓練監視"""

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.milestones_hit = []

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", 0)
        done = self.locals.get("dones", False)

        self.current_episode_reward += float(reward) if isinstance(reward, np.ndarray) else reward
        self.current_episode_length += 1

        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0.0
            self.current_episode_length = 0

        # Milestones
        milestones = [1000, 5000, 10000]
        for m in milestones:
            if self.num_timesteps == m and m not in self.milestones_hit:
                avg = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                logger.info(
                    f"⏱️  Milestone {m:,} steps | Episodes: {len(self.episode_rewards)} | "
                    f"Avg Reward: {avg:.6f}"
                )
                self.milestones_hit.append(m)

        return True


def load_and_split_data(csv_path: Path) -> Dict[str, pd.DataFrame]:
    """データをロードして分割"""
    logger.info(f"📥 Loading data from {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # タイムスタンプを処理
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
    
    logger.info(f"✓ Loaded {len(df)} bars")
    
    # Simple 70/15/15 split
    n = len(df)
    train_size = int(n * 0.70)
    val_size = int(n * 0.15)
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size : train_size + val_size]
    test_df = df.iloc[train_size + val_size :]
    
    logger.info(f"  Train: {len(train_df)} bars")
    logger.info(f"  Val: {len(val_df)} bars")
    logger.info(f"  Test: {len(test_df)} bars\n")
    
    return {"train": train_df, "val": val_df, "test": test_df}


def create_environment_wrapper(df: pd.DataFrame, config):
    """
    簡略環境ラッパーを作成（FastIntradayEnvV456の複雑性を回避）
    """
    from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
    from scripts.v456.feature_calculator_v456 import calculate_all_features
    from ztb.config.environment_config import TrainingConfig
    
    # 特徴量計算
    logger.info(f"  Calculating features for {len(df)} bars...")
    df_features = calculate_all_features(df.copy())
    logger.info(f"  ✓ Features: {df_features.shape[1]} columns")
    
    # Base特徴量の定義（OHLCV + 追加ダミー特徴量）
    base_cols = ["open", "high", "low", "close", "volume"]
    base_cols = [c for c in base_cols if c in df_features.columns]
    
    # 必要なベース特徴量の数は30
    while len(base_cols) < 30:
        dummy_col = f"base_dummy_{len(base_cols) - 5}"
        base_cols.append(dummy_col)
        if dummy_col not in df_features.columns:
            df_features[dummy_col] = 0.0
    
    # MTF特徴量（計算されたもの）
    mtf_cols = [c for c in df_features.columns if "mtf_" in c.lower()][:27]
    if len(mtf_cols) < 27:
        # ダミー特徴量を追加
        for i in range(len(mtf_cols), 27):
            mtf_cols.append(f"dummy_mtf_{i}")
            if f"dummy_mtf_{i}" not in df_features.columns:
                df_features[f"dummy_mtf_{i}"] = 0.0
    else:
        mtf_cols = mtf_cols[:27]
    
    # Regime特徴量
    regime_cols = [c for c in df_features.columns if "regime_" in c.lower()][:13]
    if len(regime_cols) < 13:
        for i in range(len(regime_cols), 13):
            regime_cols.append(f"dummy_regime_{i}")
            if f"dummy_regime_{i}" not in df_features.columns:
                df_features[f"dummy_regime_{i}"] = 0.0
    else:
        regime_cols = regime_cols[:13]
    
    logger.info(f"  Base cols: {len(base_cols)}, MTF: {len(mtf_cols)}, Regime: {len(regime_cols)}")
    
    # 環境作成
    env = FastIntradayEnvV456(
        df=df_features,
        base_feature_columns=base_cols,
        mtf_feature_columns=mtf_cols,
        regime_feature_columns=regime_cols,
        initial_balance=TrainingConfig.INITIAL_BALANCE,
        max_position=TrainingConfig.MAX_POSITION,
        max_steps=TrainingConfig.MAX_STEPS,
        max_ttl_steps=TrainingConfig.MAX_TTL_STEPS,
        cooldown_steps=TrainingConfig.COOLDOWN_STEPS,
        max_delta_per_step=TrainingConfig.MAX_DELTA_PER_STEP,
        min_delta=TrainingConfig.MIN_DELTA,
        drawdown_limit=TrainingConfig.DRAWDOWN_LIMIT,
    )
    
    return env


def train_phase3(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    timesteps: int = 10000,
) -> SAC:
    """
    訓練実行
    """
    from ztb.config.environment_config import TrainingConfig
    
    logger.info("\n" + "=" * 70)
    logger.info("🚀 Phase 3 Training")
    logger.info("=" * 70)
    
    # 環境作成
    logger.info("[Step 1] Creating training environment...")
    train_env = create_environment_wrapper(train_df, None)
    
    logger.info(f"  Obs space: {train_env.observation_space}")
    logger.info(f"  Action space: {train_env.action_space}")
    
    # モデル作成
    logger.info("\n[Step 2] Creating SAC model...")
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=TrainingConfig.LEARNING_RATE,
        batch_size=256,
        buffer_size=TrainingConfig.BUFFER_SIZE,
        tau=0.005,
        gamma=0.99,
        verbose=0,
    )
    
    # 訓練
    logger.info(f"\n[Step 3] Training for {timesteps:,} timesteps...")
    callback = Phase3Callback()
    model.learn(total_timesteps=timesteps, callback=callback)
    
    # 結果報告
    logger.info(f"\n[Step 4] Training completed")
    logger.info(f"  Total episodes: {len(callback.episode_rewards)}")
    if callback.episode_rewards:
        avg_last_100 = np.mean(callback.episode_rewards[-100:])
        logger.info(f"  Avg reward (last 100): {avg_last_100:.6f}")
        logger.info(f"  Max reward: {np.max(callback.episode_rewards):.6f}")
        logger.info(f"  Min reward: {np.min(callback.episode_rewards):.6f}")
    
    return model, callback


def evaluate_phase3(
    model: SAC,
    test_df: pd.DataFrame,
    label: str = "val",
) -> Dict:
    """
    評価実行
    """
    logger.info(f"\n" + "=" * 70)
    logger.info(f"📊 {label.upper()} Evaluation")
    logger.info("=" * 70)
    
    # 環境作成
    logger.info(f"[Step 1] Creating {label} environment...")
    eval_env = create_environment_wrapper(test_df, None)
    
    # 評価実行
    logger.info(f"[Step 2] Evaluating on {len(test_df)} bars...")
    obs, _ = eval_env.reset()
    done = False
    episode_reward = 0.0
    steps = 0
    actions = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        
        episode_reward += float(reward)
        steps += 1
        actions.append(float(action[0]) if isinstance(action, np.ndarray) else action)
    
    # 結果
    final_balance = eval_env.balance
    total_pnl = final_balance - eval_env.initial_balance
    roi = (total_pnl / eval_env.initial_balance) * 100
    
    logger.info(f"\n[Step 3] Evaluation completed")
    logger.info(f"  Episode reward: {episode_reward:.6f}")
    logger.info(f"  Steps: {steps}")
    logger.info(f"  Initial balance: {eval_env.initial_balance:,.0f} JPY")
    logger.info(f"  Final balance: {final_balance:,.0f} JPY")
    logger.info(f"  Total P&L: {total_pnl:,.0f} JPY")
    logger.info(f"  ROI: {roi:.2f}%")
    
    return {
        "episode_reward": float(episode_reward),
        "steps": steps,
        "initial_balance": float(eval_env.initial_balance),
        "final_balance": float(final_balance),
        "total_pnl": float(total_pnl),
        "roi": float(roi),
        "action_mean": float(np.mean(actions)),
        "action_std": float(np.std(actions)),
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 3 Train & Evaluate")
    parser.add_argument("--timesteps", type=int, default=10000, help="Training timesteps")
    parser.add_argument("--data", type=str, default=None, help="Data CSV path")
    args = parser.parse_args()
    
    # データロード・分割
    if args.data:
        data_path = Path(args.data)
    else:
        # デフォルトパスを探す
        candidates = [
            Path("test_synthetic_dataset.csv"),
            Path("data/datasets/test_synthetic_dataset.csv"),
        ]
        data_path = None
        for c in candidates:
            if c.exists():
                data_path = c
                break
        
        if not data_path:
            logger.error("No data file found. Use --data to specify path.")
            return
    
    data_splits = load_and_split_data(data_path)
    
    # 訓練
    model, callback = train_phase3(
        data_splits["train"],
        data_splits["val"],
        timesteps=args.timesteps,
    )
    
    # Val評価
    val_stats = evaluate_phase3(model, data_splits["val"], label="val")
    
    # Test評価
    test_stats = evaluate_phase3(model, data_splits["test"], label="test")
    
    # モデル保存
    logger.info("\n" + "=" * 70)
    logger.info("💾 Saving model...")
    model_dir = Path("models/phase3")
    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"sac_v456_phase3_{timestamp}"
    model.save(str(model_path))
    logger.info(f"✓ Model saved to {model_path}")
    
    # 最終レポート
    logger.info("=" * 70)
    logger.info("📈 Phase 3 Summary")
    logger.info("=" * 70)
    logger.info(f"Training timesteps: {args.timesteps:,}")
    logger.info(f"Total episodes: {len(callback.episode_rewards)}")
    logger.info(f"\nVal results:")
    for k, v in val_stats.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}" if abs(v) < 100 else f"  {k}: {v:,.2f}")
    logger.info(f"\nTest results:")
    for k, v in test_stats.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}" if abs(v) < 100 else f"  {k}: {v:,.2f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
