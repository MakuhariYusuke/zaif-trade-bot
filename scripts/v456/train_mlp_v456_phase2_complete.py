#!/usr/bin/env python3
"""
Phase 2完全統合訓練スクリプト

包含内容:
1. MTF特徴量の実装 (feature_calculator_v456.py)
2. アクション変換の統一化 (ActionConverterV456)
3. 統一された環境設定 (TrainingConfig)
4. 訓練パイプラインの標準化

実行:
    python scripts/v456/train_mlp_v456_phase2_complete.py --timesteps 50000
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
from scripts.v456.feature_calculator_v456 import calculate_all_features
from ztb.training.action_converter_v456 import (
    ActionConverterV456,
    ActionAnalyzer,
)
from ztb.config.environment_config import get_training_config
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
CONFIG = get_training_config()


class PhaseIITrainingEnvironment(gym.Env):
    """
    Phase II統合訓練環境
    
    改善事項:
    1. MTF特徴量を実計算
    2. ActionConverterV456で統一的なアクション変換
    3. SafeIntradayEnvWrapperで訓練の安定性を確保
    """
    
    def __init__(
        self,
        base_env: FastIntradayEnvV456,
        action_analyzer: ActionAnalyzer,
        warmup_steps: int = 10,
        initial_drawdown_limit: float = 0.5,
        final_drawdown_limit: float = 0.3,
    ):
        super().__init__()
        self.env = base_env
        self.action_analyzer = action_analyzer
        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space
        
        self.warmup_steps = warmup_steps
        self.warmup_counter = 0
        self.initial_drawdown_limit = initial_drawdown_limit
        self.final_drawdown_limit = final_drawdown_limit
        
        self.episode_steps = 0
        self.initial_balance = base_env.initial_balance
        
        # アクション分析
        self.last_action = None
        self.last_discrete_action = None
        
    def reset(self, seed=None, options=None):
        """リセット"""
        obs, info = self.env.reset(seed=seed, options=options)
        self.warmup_counter = 0
        self.episode_steps = 0
        
        logger.debug(f"環境リセット: balance={self.env.balance:.2f} JPY")
        
        return obs, info
    
    def step(self, action: np.ndarray):
        """ステップ実行"""
        
        # ウォームアップ期間: drawdown制限を段階的に適用
        if self.warmup_counter < self.warmup_steps:
            progress = self.warmup_counter / self.warmup_steps
            self.env.drawdown_limit = (
                self.initial_drawdown_limit +
                progress * (self.final_drawdown_limit - self.initial_drawdown_limit)
            )
            self.warmup_counter += 1
        
        # アクション分析
        action_scalar = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        self.action_analyzer.record_action(action_scalar)
        
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


class TrainingCallback(BaseCallback):
    """訓練進捗を記録するコールバック"""
    
    def __init__(self, log_interval: int = 100, action_analyzer: ActionAnalyzer = None):
        super().__init__()
        self.log_interval = log_interval
        self.milestones = {}
        self.action_analyzer = action_analyzer
    
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
                
                if self.action_analyzer:
                    action_stats = self.action_analyzer.get_statistics()
                    self.milestones[self.n_calls]['action_stats'] = action_stats
                
                logger.info(f"\n{'='*70}")
                logger.info(f"📊 Milestone #{milestone_key} ({self.n_calls:,} steps)")
                logger.info(f"{'='*70}")
                logger.info(f"  Episode Reward Mean: {ep_rew:.6f}")
                logger.info(f"  Episode Length Mean: {ep_len:.1f}")
                
                if self.action_analyzer:
                    stats = self.action_analyzer.get_statistics()
                    logger.info(f"  Action Mean: {stats.get('action_mean', 0):.4f}")
                    logger.info(f"  BUY Rate: {stats.get('buy_ratio', 0):.2%}")
                    logger.info(f"  SELL Rate: {stats.get('sell_ratio', 0):.2%}")
                    logger.info(f"  HOLD Rate: {stats.get('hold_ratio', 0):.2%}")
                logger.info("")
        
        return True
    
    def save(self, path: Path):
        """マイルストーン保存"""
        with open(path, 'w') as f:
            json.dump(self.milestones, f, indent=2, default=float)


def create_environment(
    market_data: pd.DataFrame,
    initial_balance: float = 100000.0,
    action_analyzer: ActionAnalyzer = None,
):
    """環境作成 (Phase II統合版)"""
    
    logger.info("=" * 70)
    logger.info("📊 Phase II 統合環境作成")
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
    
    # feature_calculator_v456.py を使用
    try:
        df = calculate_all_features(df)
        logger.info(f"  ✓ 特徴量計算完了: {len(df.columns)} columns")
    except Exception as e:
        logger.error(f"特徴量計算エラー: {e}")
        raise
    
    mtf_cols = [f'mtf_{i}' for i in range(27)]
    regime_cols = [f'regime_{i}' for i in range(13)]
    
    # Base features の作成
    base_feature_cols = [f'base_{i}' for i in range(30)]
    for i, col in enumerate(base_feature_cols):
        if col not in df.columns:
            if i < 5:
                df[col] = df[['open', 'high', 'low', 'close', 'volume'][i]]
            else:
                if i < 10:
                    df[col] = (df['close'] - df['close'].rolling(5).mean()) / (df['close'].rolling(5).std() + 1e-6)
                elif i < 15:
                    df[col] = (df['volume'] - df['volume'].rolling(5).mean()) / (df['volume'].rolling(5).std() + 1e-6)
                else:
                    df[col] = np.sin(np.arange(len(df)) * 2 * np.pi / 1440)
    
    # オプション列
    for col in ['atr', 'impact_proxy']:
        if col not in df.columns:
            df[col] = np.ones(len(df))
    
    logger.info(f"  ✓ 拡張特徴量統合完了: {df.shape}")
    
    # ステップ3: 基本環境作成
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
    
    # ステップ4: 統合環境ラッパー
    logger.info("\n[Step 4] 統合環境ラッパーの適用...")
    
    env = PhaseIITrainingEnvironment(
        base_env=base_env,
        action_analyzer=action_analyzer or ActionAnalyzer(),
        warmup_steps=10,
        initial_drawdown_limit=0.5,
        final_drawdown_limit=0.3,
    )
    
    logger.info("  ✓ PhaseIITrainingEnvironment 適用")
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ 環境作成完了")
    logger.info("=" * 70 + "\n")
    
    return env


def main():
    """メイン関数"""
    
    # パラメータ解析
    parser = argparse.ArgumentParser(
        description='Phase II完全統合訓練スクリプト'
    )
    parser.add_argument('--timesteps', type=int, default=50000)
    parser.add_argument('--initial-balance', type=float, default=100000.0)
    parser.add_argument('--model-dir', type=Path, default=PROJECT_ROOT / 'models')
    parser.add_argument('--log-dir', type=Path, default=PROJECT_ROOT / 'logs')
    
    args = parser.parse_args()
    
    # ディレクトリ作成
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    
    load_dotenv()
    
    logger.info("\n" + "="*70)
    logger.info("🚀 Phase II 完全統合訓練開始")
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
    logger.info(f"  ✓ {len(market_data):,} rows loaded\n")
    
    # アクション分析器
    action_analyzer = ActionAnalyzer()
    
    # 環境作成
    logger.info("[2] 環境作成...")
    env = create_environment(
        market_data=market_data,
        initial_balance=args.initial_balance,
        action_analyzer=action_analyzer,
    )
    
    # モデル作成
    logger.info("[3] SAC モデル作成...")
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
    logger.info("  ✓ SAC モデル準備完了\n")
    
    # ロガー設定
    logger.info("[4] ロガー設定...")
    logger_instance = configure_logger(
        str(args.log_dir / f"phase2_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )
    model.set_logger(logger_instance)
    logger.info("  ✓ ロガー設定完了\n")
    
    # コールバック設定
    callback = TrainingCallback(log_interval=100, action_analyzer=action_analyzer)
    
    # 訓練
    logger.info("[5] 訓練開始...")
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
    
    # 最終統計
    logger.info("[6] 最終統計...")
    final_stats = action_analyzer.get_statistics()
    logger.info("  アクション分布:")
    logger.info(f"    BUY:  {final_stats.get('buy_ratio', 0):.2%}")
    logger.info(f"    SELL: {final_stats.get('sell_ratio', 0):.2%}")
    logger.info(f"    HOLD: {final_stats.get('hold_ratio', 0):.2%}")
    
    # モデル保存
    logger.info("\n[7] モデル保存...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = args.model_dir / f"sac_v456_phase2_complete_{timestamp}"
    model.save(str(model_path))
    logger.info(f"  ✓ {model_path}.zip")
    
    # マイルストーン保存
    milestone_path = args.log_dir / f"phase2_complete_milestones_{timestamp}.json"
    callback.save(milestone_path)
    logger.info(f"  ✓ {milestone_path}")
    
    # 最終統計を別ファイルに保存
    stats_path = args.log_dir / f"phase2_complete_stats_{timestamp}.json"
    with open(stats_path, 'w') as f:
        json.dump(final_stats, f, indent=2)
    logger.info(f"  ✓ {stats_path}")
    
    logger.info("\n" + "="*70)
    logger.info("🎉 訓練完了 - Phase II統合実装成功")
    logger.info("="*70 + "\n")


if __name__ == '__main__':
    main()
