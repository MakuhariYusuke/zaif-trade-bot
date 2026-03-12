#!/usr/bin/env python3
"""
Phase 3: Time-Series Split 訓練・評価統合スクリプト

Train/Val データセットで訓練し、Test セットで OOS 評価を実施
"""
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List
import argparse

import numpy as np
import pandas as pd
from gymnasium import Env
import matplotlib.pyplot as plt

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))
sys.path.insert(0, str(workspace_root / "src"))

from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
from ztb.training.action_converter_v456 import ActionConverterV456, ActionAnalyzer
from ztb.config.environment_config import TrainingConfig

# v456スクリプト内のモジュール
sys.path.insert(0, str(workspace_root / "scripts" / "v456"))
from feature_calculator_v456 import (
    MTFFeatureCalculator,
    RegimeFeatureCalculator,
)
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Phase3Callback(BaseCallback):
    """Phase 3 訓練監視コールバック"""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.milestone_steps = [1000, 5000, 10000, 20000, 50000]
        self.next_milestone_idx = 0
    
    def _on_step(self) -> bool:
        # Episode情報の更跡
        reward = self.locals.get("rewards", 0)
        done = self.locals.get("dones", False)
        
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        # Milestone報告
        if self.next_milestone_idx < len(self.milestone_steps):
            if self.num_timesteps >= self.milestone_steps[self.next_milestone_idx]:
                avg_reward = (
                    np.mean(self.episode_rewards[-100:])
                    if self.episode_rewards
                    else 0
                )
                logger.info(
                    f"⏱️  Milestone {self.milestone_steps[self.next_milestone_idx]:,} steps | "
                    f"Episodes: {len(self.episode_rewards):,} | "
                    f"Avg Reward (100 episodes): {avg_reward:.6f}"
                )
                self.next_milestone_idx += 1
        
        return True


class Phase3TrainingEnvironment:
    """Phase 3 統合訓練環境"""
    
    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        config: TrainingConfig = None,
        use_action_analyzer: bool = True,
    ):
        self.train_df = train_df
        self.val_df = val_df
        self.config = config or TrainingConfig()
        self.use_action_analyzer = use_action_analyzer
        
        # 特徴量計算器の初期化
        self.mtf_calc = MTFFeatureCalculator()
        self.regime_calc = RegimeFeatureCalculator()
        
        # ActionAnalyzerの初期化
        self.action_analyzer = ActionAnalyzer() if use_action_analyzer else None
        self.converter = ActionConverterV456()
        
        logger.info("✓ Phase 3 TrainingEnvironment initialized")
    
    def create_env(self, subset: str = "train") -> FastIntradayEnvV456:
        """訓練環境を作成"""
        data_df = self.train_df if subset == "train" else self.val_df
        
        # 特徴量を計算
        data_with_features = self._calculate_features(data_df)
        
        env = FastIntradayEnvV456(
            data=data_with_features,
            initial_balance=TrainingConfig.INITIAL_BALANCE,
            max_position_size=TrainingConfig.MAX_POSITION,
            max_drawdown_pct=TrainingConfig.DRAWDOWN_LIMIT,
            verbose=0,
        )
        
        logger.info(
            f"✓ Environment created for {subset}: {len(data_with_features)} bars"
        )
        return env
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量を計算"""
        df_copy = df.copy()
        
        # MTF特徴量
        mtf_features = self.mtf_calc.calculate_all_mtf_features(df_copy)
        for col in mtf_features.columns:
            if col not in df_copy.columns:
                df_copy[col] = mtf_features[col]
        
        # Regime特徴量 (クラス関数から直接呼び出し)
        regime_features = RegimeFeatureCalculator.calculate_regime_features(df_copy)
        for col in regime_features.columns:
            if col not in df_copy.columns:
                df_copy[col] = regime_features[col]
        
        logger.info(
            f"  Features calculated: {len(df_copy.columns)} columns, {len(df_copy)} rows"
        )
        return df_copy
    
    def train(
        self,
        total_timesteps: int = 10000,
        learning_rate: float = 3e-4,
        batch_size: int = 256,
        buffer_size: int = 100000,
    ) -> Tuple[SAC, Dict]:
        """訓練を実行"""
        logger.info("\n" + "=" * 70)
        logger.info("🚀 Phase 3 Training Start")
        logger.info("=" * 70)
        
        # 環境作成
        train_env = self.create_env(subset="train")
        
        # SAC モデル作成
        logger.info(f"\n[訓練準備]")
        logger.info(f"  Observation Space: {train_env.observation_space}")
        logger.info(f"  Action Space: {train_env.action_space}")
        logger.info(f"  Learning Rate: {learning_rate}")
        logger.info(f"  Batch Size: {batch_size}")
        logger.info(f"  Buffer Size: {buffer_size}")
        
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            buffer_size=buffer_size,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            verbose=0,
        )
        
        # 訓練
        logger.info(f"\n[訓練開始]")
        callback = Phase3Callback(verbose=0)
        model.learn(total_timesteps=total_timesteps, callback=callback)
        
        logger.info(f"\n[訓練完了]")
        logger.info(f"  Total Episodes: {len(callback.episode_rewards)}")
        logger.info(f"  Mean Reward (last 100): {np.mean(callback.episode_rewards[-100:]):.6f}")
        logger.info(f"  Max Reward: {np.max(callback.episode_rewards):.6f}")
        logger.info(f"  Min Reward: {np.min(callback.episode_rewards):.6f}")
        
        # ActionAnalyzer の統計
        if self.action_analyzer:
            stats = self.action_analyzer.get_statistics()
            logger.info(f"\n[アクション分析]")
            logger.info(f"  BUY: {stats['buy_ratio']:.2%}")
            logger.info(f"  SELL: {stats['sell_ratio']:.2%}")
            logger.info(f"  HOLD: {stats['hold_ratio']:.2%}")
        
        return model, {
            "total_timesteps": total_timesteps,
            "total_episodes": len(callback.episode_rewards),
            "mean_reward": float(np.mean(callback.episode_rewards[-100:])),
            "action_stats": self.action_analyzer.get_statistics() if self.action_analyzer else {},
        }
    
    def evaluate(self, model: SAC, subset: str = "val") -> Dict:
        """評価を実行"""
        logger.info(f"\n" + "=" * 70)
        logger.info(f"📊 {subset.upper()} Evaluation Start")
        logger.info("=" * 70)
        
        # 評価環境作成
        eval_env = self.create_env(subset=subset)
        
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        trades = []
        actions_taken = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_steps += 1
            actions_taken.append(action[0] if isinstance(action, np.ndarray) else action)
            
            if "trade_executed" in info and info["trade_executed"]:
                trades.append(info)
        
        # 環境から最終統計を取得
        final_balance = eval_env.balance
        total_pnl = final_balance - eval_env.initial_balance
        roi = (total_pnl / eval_env.initial_balance) * 100
        
        logger.info(f"\n[{subset.upper()} 評価完了]")
        logger.info(f"  Episode Reward: {episode_reward:.6f}")
        logger.info(f"  Episode Steps: {episode_steps}")
        logger.info(f"  Total Trades: {len(trades)}")
        logger.info(f"  Initial Balance: {eval_env.initial_balance:,.0f} JPY")
        logger.info(f"  Final Balance: {final_balance:,.0f} JPY")
        logger.info(f"  Total P&L: {total_pnl:,.0f} JPY")
        logger.info(f"  ROI: {roi:.2f}%")
        
        # Win Rate計算
        if trades:
            winning_trades = sum(1 for t in trades if t.get("profit", 0) > 0)
            win_rate = winning_trades / len(trades)
            logger.info(f"  Win Rate: {win_rate:.2%}")
        
        return {
            "episode_reward": float(episode_reward),
            "episode_steps": episode_steps,
            "total_trades": len(trades),
            "initial_balance": float(eval_env.initial_balance),
            "final_balance": float(final_balance),
            "total_pnl": float(total_pnl),
            "roi": float(roi),
            "actions_mean": float(np.mean(actions_taken)),
            "actions_std": float(np.std(actions_taken)),
        }


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Integrated Training")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10000,
        help="Total timesteps for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="SAC learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training",
    )
    args = parser.parse_args()
    
    # データロード
    logger.info("📥 Loading data...")
    # CSV形式を優先的に試す
    data_paths = [
        workspace_root / "test_synthetic_dataset.csv",
        workspace_root / "data" / "datasets" / "test_synthetic_dataset.csv",
    ]
    
    df = None
    for path in data_paths:
        if path.exists():
            df = pd.read_csv(path)
            logger.info(f"✓ {len(df)} bars loaded from {path}")
            break
    
    if df is None:
        logger.error("No data file found")
        return
    
    # timestamp列をインデックスに設定
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
    
    # Phase 3 分割を実行（phase3_oos_evaluation.pyから）
    from scripts.v456.phase3_oos_evaluation import TimeSeriesSplitter
    
    splitter = TimeSeriesSplitter(df, train_ratio=0.70, embargo_days=7)
    train_df, val_df, test_df = splitter.split()
    
    # 訓練環境作成
    phase3_env = Phase3TrainingEnvironment(
        train_df=train_df,
        val_df=val_df,
        use_action_analyzer=True,
    )
    
    # 訓練実行
    model, train_stats = phase3_env.train(
        total_timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )
    
    # Val 評価
    val_stats = phase3_env.evaluate(model, subset="val")
    
    # モデル保存
    model_dir = Path(__file__).parent.parent.parent / "models" / "phase3"
    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"sac_v456_phase3_{timestamp}"
    model.save(str(model_path))
    logger.info(f"\n✓ Model saved to {model_path}")
    
    # 統計出力
    logger.info("\n" + "=" * 70)
    logger.info("📈 Phase 3 Training Summary")
    logger.info("=" * 70)
    logger.info(f"Train Stats: {train_stats}")
    logger.info(f"Val Stats: {val_stats}")
    logger.info(f"Model: {model_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
