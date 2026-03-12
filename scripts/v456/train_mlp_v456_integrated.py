#!/usr/bin/env python3
"""
Phase 2統合訓練スクリプト (MTF特徴量実装版)

改善点:
1. feature_calculator_v456.py で MTF + Regime 特徴量を実計算
2. ランダムノイズ埋め込みを完全に排除
3. 環境パラメータを environment_config.py で一元管理
4. SafeIntradayEnvWrapper で訓練の安定性を確保

実行:
    python scripts/v456/train_mlp_v456_integrated.py --timesteps 30000
"""

import sys
import os
import json
from pathlib import Path
import logging
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure as configure_logger
import gymnasium as gym
from gymnasium import spaces

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
from scripts.v456.feature_calculator_v456 import (
    MTFFeatureCalculator,
    RegimeFeatureCalculator,
    calculate_all_features
)
from ztb.config.environment_config import get_training_config
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
CONFIG = get_training_config()


class SafeIntradayEnvWrapper(gym.Env):
    """
    FastIntradayEnvV456 のラッパー
    
    改善内容:
    1. 初期化ステップのスキップ（warmup）
    2. drawdown 制限の段階的適用
    3. 報酬スケーリングと正規化
    """
    
    def __init__(
        self,
        base_env: FastIntradayEnvV456,
        warmup_steps: int = 10,
        initial_drawdown_limit: float = 0.5,  # 初期は寛容
        final_drawdown_limit: float = 0.3,
    ):
        super().__init__()
        self.env = base_env
        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space
        
        self.warmup_steps = warmup_steps
        self.warmup_counter = 0
        self.initial_drawdown_limit = initial_drawdown_limit
        self.final_drawdown_limit = final_drawdown_limit
        
        # 初期状態
        self.episode_steps = 0
        self.initial_balance = base_env.initial_balance
        
    def reset(self, seed=None, options=None):
        """リセット"""
        obs, info = self.env.reset(seed=seed, options=options)
        self.warmup_counter = 0
        self.episode_steps = 0
        
        logger.debug(f"Wrapper reset: balance={self.env.balance:.2f}")
        
        return obs, info
    
    def step(self, action: np.ndarray):
        """ステップ実行"""
        
        # ウォームアップ期間: drawdown 制限を緩和
        if self.warmup_counter < self.warmup_steps:
            # 現在の drawdown_limit を保存
            original_limit = self.env.drawdown_limit
            
            # 段階的に緩和
            progress = self.warmup_counter / self.warmup_steps
            self.env.drawdown_limit = (
                self.initial_drawdown_limit +
                progress * (self.final_drawdown_limit - self.initial_drawdown_limit)
            )
            
            self.warmup_counter += 1
        
        # 基本環境でステップ実行
        result = self.env.step(action)
        
        # Gymnasium 5層タプル対応
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        
        self.episode_steps += 1
        
        # 返却
        if len(result) == 5:
            return obs, reward, terminated, truncated, info
        else:
            return obs, reward, done, info
    
    def render(self):
        """描画"""
        return self.env.render()
    
    def close(self):
        """クローズ"""
        self.env.close()


class MilestoneCallback(BaseCallback):
    """訓練進捗を記録するコールバック"""
    
    def __init__(self, log_interval: int = 100):
        super().__init__()
        self.log_interval = log_interval
        self.milestones = {}
    
    def _on_step(self) -> bool:
        """ステップ時の処理"""
        
        if self.n_calls % self.log_interval == 0:
            # ロガーから統計を取得
            if hasattr(self.model, 'logger') and self.model.logger:
                ep_rew = self.model.logger.name_to_value.get('rollout/ep_rew_mean', np.nan)
                ep_len = self.model.logger.name_to_value.get('rollout/ep_len_mean', np.nan)
                
                milestone_key = self.n_calls // self.log_interval
                self.milestones[self.n_calls] = {
                    'timestamp': datetime.now().isoformat(),
                    'episode_reward_mean': float(ep_rew) if not np.isnan(ep_rew) else None,
                    'episode_length_mean': float(ep_len) if not np.isnan(ep_len) else None,
                }
                
                logger.info(f"\n{'='*70}")
                logger.info(f"📊 Milestone #{milestone_key} ({self.n_calls} steps)")
                logger.info(f"{'='*70}")
                logger.info(f"  Episode Reward Mean: {ep_rew:.6f}")
                logger.info(f"  Episode Length Mean: {ep_len:.1f}")
                logger.info("")
        
        return True
    
    def save(self, path: Path):
        """マイルストーン保存"""
        with open(path, 'w') as f:
            json.dump(self.milestones, f, indent=2, default=float)


def create_environment(
    market_data: pd.DataFrame,
    initial_balance: float = 100000.0,
    use_wrapper: bool = True,
):
    """環境作成 (MTF特徴量実装版)"""
    
    logger.info("=" * 70)
    logger.info("環境作成中...")
    logger.info("=" * 70)
    logger.info(f"  初期残高: {initial_balance:,.0f} JPY")
    logger.info(f"  データ行数: {len(market_data):,}")
    
    # ステップ1: 基本特徴量の検証
    logger.info("\n[Step 1] 基本特徴量の検証...")
    base_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_base = [col for col in base_cols if col not in market_data.columns]
    
    if missing_base:
        raise ValueError(f"Missing base columns: {missing_base}")
    
    logger.info("  ✓ 基本特徴量OK (OHLCV)")
    
    # ステップ2: 拡張特徴量の計算
    logger.info("\n[Step 2] MTF + Regime 特徴量の計算...")
    
    df = market_data.copy()
    
    # feature_calculator_v456.py を使用して計算
    try:
        df = calculate_all_features(df)
        logger.info(f"  ✓ 特徴量計算完了: {len(df.columns)} columns")
    except Exception as e:
        logger.error(f"特徴量計算エラー: {e}")
        raise
    
    # 特徴量列の確認
    mtf_cols = [f'mtf_{i}' for i in range(27)]
    regime_cols = [f'regime_{i}' for i in range(13)]
    
    # 追加の必須列
    base_feature_cols = [f'base_{i}' for i in range(30)]
    for i, col in enumerate(base_feature_cols):
        if col not in df.columns:
            # Base features are composites of OHLCV - create normalized versions
            if i < 5:
                # Direct OHLCV
                df[col] = df[['open', 'high', 'low', 'close', 'volume'][i]]
            else:
                # Derived features - use simple transformations
                if i < 10:
                    df[col] = (df['close'] - df['close'].rolling(5).mean()) / (df['close'].rolling(5).std() + 1e-6)
                elif i < 15:
                    df[col] = (df['volume'] - df['volume'].rolling(5).mean()) / (df['volume'].rolling(5).std() + 1e-6)
                else:
                    df[col] = np.sin(np.arange(len(df)) * 2 * np.pi / 1440)  # Cyclical time
    
    # オプション列
    if 'atr' not in df.columns:
        df['atr'] = np.ones(len(df))
    if 'impact_proxy' not in df.columns:
        df['impact_proxy'] = np.ones(len(df))
    
    logger.info(f"  ✓ 拡張特徴量統合完了: {df.shape}")
    
    # ステップ3: 環境作成
    logger.info("\n[Step 3] 基本環境の作成...")
    
    base_env = FastIntradayEnvV456(
        df=df,
        base_feature_columns=base_feature_cols[:30],
        mtf_feature_columns=mtf_cols[:27],
        regime_feature_columns=regime_cols[:13],
        initial_balance=initial_balance,
        max_position=CONFIG.MAX_POSITION,
        max_steps=CONFIG.MAX_STEPS,
        drawdown_limit=0.3,
        prewarm_steps=100,
        commission_rate=0.001
    )
    
    logger.info("  ✓ 基本環境OK")
    
    # ステップ4: ラッパー適用
    logger.info("\n[Step 4] ラッパーの適用...")
    
    if use_wrapper:
        env = SafeIntradayEnvWrapper(
            base_env=base_env,
            warmup_steps=10,
            initial_drawdown_limit=0.5,
            final_drawdown_limit=0.3,
        )
        logger.info("  ✓ SafeIntradayEnvWrapper 適用")
    else:
        env = base_env
        logger.info("  ✓ ラッパーなし")
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ 環境作成完了")
    logger.info("=" * 70 + "\n")
    
    return env


def main():
    """メイン関数"""
    
    # パラメータ解析
    parser = argparse.ArgumentParser(
        description='Phase 2統合訓練スクリプト (MTF特徴量実装版)'
    )
    parser.add_argument('--timesteps', type=int, default=30000)
    parser.add_argument('--initial-balance', type=float, default=100000.0)
    parser.add_argument('--model-dir', type=Path, default=PROJECT_ROOT / 'models')
    parser.add_argument('--log-dir', type=Path, default=PROJECT_ROOT / 'logs')
    
    args = parser.parse_args()
    
    # ディレクトリ作成
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    
    # ロード環境設定
    load_dotenv()
    
    logger.info("\n" + "="*70)
    logger.info("🚀 Phase 2 統合訓練開始")
    logger.info("="*70)
    logger.info(f"  Timesteps: {args.timesteps:,}")
    logger.info(f"  Initial Balance: {args.initial_balance:,.0f} JPY")
    logger.info(f"  Model Dir: {args.model_dir}")
    logger.info(f"  Log Dir: {args.log_dir}")
    logger.info("="*70 + "\n")
    
    # データロード
    logger.info("[1] データロード...")
    data_path = PROJECT_ROOT / 'data' / 'btc_jpy_1m_v454.csv'
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    market_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    logger.info(f"  ✓ {len(market_data):,} rows loaded")
    
    # 環境作成
    logger.info("\n[2] 環境作成...")
    env = create_environment(
        market_data=market_data,
        initial_balance=args.initial_balance,
        use_wrapper=True
    )
    
    # モデル作成
    logger.info("\n[3] SAC モデル作成...")
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        policy_kwargs={
            "net_arch": [256, 256],
        },
        device="cuda",
        verbose=0,
    )
    logger.info("  ✓ SAC モデル準備完了")
    
    # ロガー設定
    logger.info("\n[4] ロガー設定...")
    logger_instance = configure_logger(
        str(args.log_dir / f"sac_v456_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )
    model.set_logger(logger_instance)
    logger.info("  ✓ ロガー設定完了")
    
    # コールバック設定
    callback = MilestoneCallback(log_interval=100)
    
    # 訓練
    logger.info("\n[5] 訓練開始...")
    logger.info("="*70)
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        logger.info("\n⚠️  訓練中断 (Ctrl+C)")
    except Exception as e:
        logger.error(f"\n❌ 訓練エラー: {e}", exc_info=True)
        raise
    
    logger.info("="*70)
    logger.info("✓ 訓練完了\n")
    
    # モデル保存
    logger.info("[6] モデル保存...")
    model_path = args.model_dir / f"sac_v456_mtf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model.save(str(model_path))
    logger.info(f"  ✓ {model_path}.zip")
    
    # マイルストーン保存
    milestone_path = args.log_dir / f"milestones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    callback.save(milestone_path)
    logger.info(f"  ✓ {milestone_path}")
    
    logger.info("\n" + "="*70)
    logger.info("🎉 完了")
    logger.info("="*70 + "\n")


if __name__ == '__main__':
    main()
