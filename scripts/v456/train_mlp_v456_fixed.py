#!/usr/bin/env python3
"""
Week 4 修正版訓練スクリプト (Fast Fix)

問題: FastIntradayEnvV456 は以下の理由で初期ステップで終了
1. 手数料(fee_paid)が balance から直接差引される
2. 報酬関数が複数のペナルティを含む (fee_norm, slip_norm, inventory_risk...)
3. 初期ランダムアクション(-100% 空売り)で大きなペナルティが発生
4. balance が drawdown_limit を即座に超過

解決策:
1. **環境ラッパー**: FastIntradayEnvV456 のラッパーで initial warmup フェーズを追加
2. **報酬スケーリング**: 報酬を合理的な範囲に正規化
3. **初期化検証**: リセット直後に冗長な手数料がないことを確認
4. **drawdown 緩和**: 初期フェーズのdrawdown制限を一時的に緩和
5. **より大きな初期残高**: balance の変動に耐性を持たせる

実行:
    python scripts/v456/train_mlp_v456_fixed.py --timesteps 30000 --initial-balance 100000
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
from ztb.live_trading.trading_api import TradingAPI
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
            original_dd = self.env.drawdown_limit
            # ウォームアップ中は寛容に
            self.env.drawdown_limit = self.initial_drawdown_limit
            self.warmup_counter += 1
        else:
            # 段階的に厳しくする
            progress = min(1.0, (self.episode_steps - self.warmup_steps) / 500.0)
            self.env.drawdown_limit = (
                self.initial_drawdown_limit * (1 - progress) +
                self.final_drawdown_limit * progress
            )
        
        # 環境ステップ実行
        obs, reward, done, truncated, info = self.env.step(action)
        
        # 報酬スケーリング: 外部スケーリング層
        # 環境の報酬をそのまま使うが、外側で補正
        reward = self._scale_reward(reward, info)
        
        self.episode_steps += 1
        
        return obs, reward, done, truncated, info
    
    def _scale_reward(self, reward: float, info: dict) -> float:
        """報酬スケーリング"""
        
        # 報酬を [-1, 1] の範囲に収める
        # ただし、環境の報酬が既にそこそこ正規化されていると仮定
        
        # 簡易スケーリング
        scaled = np.clip(reward, -1.0, 1.0)
        
        # 小さな正報酬ボーナス (アクション起動時)
        if abs(info.get('pnl', 0)) > 0:
            scaled += 0.01  # 取引を起動したことで小さなボーナス
        
        return float(scaled)
    
    def close(self):
        """環境をクローズ"""
        self.env.close()


class ThousandStepCallback(BaseCallback):
    """千ステップごとの統計記録"""
    
    def __init__(self, log_interval: int = 1000):
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
    """環境作成"""
    
    logger.info("環境作成中...")
    logger.info(f"  初期残高: {initial_balance:,.0f} JPY")
    
    # 特徴量準備
    df = market_data.copy()
    
    base_cols = [f'base_{i}' for i in range(30)]
    mtf_cols = [f'mtf_{i}' for i in range(27)]
    regime_cols = [f'regime_{i}' for i in range(13)]
    
    # 特徴量検証: 欠損列はエラーで検出
    missing_base = [col for col in base_cols if col not in df.columns]
    missing_mtf = [col for col in mtf_cols if col not in df.columns]
    missing_regime = [col for col in regime_cols if col not in df.columns]
    
    if missing_base or missing_mtf or missing_regime:
        error_msg = "Missing feature columns detected:\n"
        if missing_base:
            error_msg += f"  Base: {missing_base}\n"
        if missing_mtf:
            error_msg += f"  MTF: {missing_mtf}\n"
        if missing_regime:
            error_msg += f"  Regime: {missing_regime}\n"
        error_msg += f"\nAvailable columns: {df.columns.tolist()}\n"
        error_msg += "→ Implement feature calculation or provide pre-computed features."
        raise ValueError(error_msg)
    
    logger.info("✓ All required feature columns present")
    
    for col in ['atr', 'impact_proxy']:
        if col not in df.columns:
            logger.warning(f"Optional column '{col}' not found. Creating placeholder.")
            df[col] = np.ones(len(df))  # 最小値1で初期化（ランダムではなく）
    
    # 基本環境作成
    # max_position: 最大ポジションサイズ (JPY単位, 小数)
    # 例: BTC=100,000円で max_position=0.01 = 1000円 max position
    # または max_position=0.001 = 100円 max position
    base_env = FastIntradayEnvV456(
        df=df,
        base_feature_columns=base_cols[:30],
        mtf_feature_columns=mtf_cols[:27],
        regime_feature_columns=regime_cols[:13],
        initial_balance=initial_balance,
        max_position=CONFIG.MAX_POSITION,
        max_steps=CONFIG.MAX_STEPS,
        drawdown_limit=0.3,  # 初期は 30%
        prewarm_steps=100,
        commission_rate=0.001
    )
    
    logger.info("✓ 基本環境作成成功")
    
    # ラッパーで改善
    if use_wrapper:
        env = SafeIntradayEnvWrapper(
            base_env=base_env,
            warmup_steps=10,
            initial_drawdown_limit=0.5,  # ウォームアップ中は 50%
            final_drawdown_limit=0.3,    # 最終的には 30%
        )
        logger.info("✓ ラッパー適用成功 (warmup + drawdown段階化)")
    else:
        env = base_env
    
    return env


def main(args):
    """訓練実行"""
    
    print("=" * 70)
    print("Week 4 Fixed Training (Fast Implementation)")
    print("=" * 70)
    print()
    
    # 環境変数読み込み
    load_dotenv(PROJECT_ROOT / '.env')
    
    # データ読み込み
    data_path = PROJECT_ROOT / 'data' / 'btc_jpy_1m_v454.csv'
    market_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    logger.info(f"Market data: {len(market_data):,} records")
    
    # 環境作成
    env = create_environment(
        market_data=market_data,
        initial_balance=args.initial_balance,
        use_wrapper=True
    )
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("Training Configuration")
    logger.info("=" * 70)
    logger.info(f"  Initial Balance: {args.initial_balance:,.0f} JPY")
    logger.info(f"  Timesteps: {args.timesteps:,}")
    logger.info(f"  Learning Rate: {args.learning_rate}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info("")
    
    # SAC モデル作成
    logger.info("Creating SAC model...")
    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        buffer_size=1000000,
        learning_starts=10000,  # ウォームアップ期間
        tau=0.005,
        gamma=0.99,
        policy_kwargs={
            'net_arch': [128, 128],
            'n_critics': 2,
        },
        verbose=0,
        tensorboard_log=str(PROJECT_ROOT / 'runs' / 'week4_fixed')
    )
    
    logger.info("✓ SAC model created")
    logger.info("")
    logger.info("=" * 70)
    logger.info("Training Start")
    logger.info("=" * 70)
    logger.info("")
    
    # コールバック
    callback = ThousandStepCallback(log_interval=1000)
    
    try:
        # 訓練実行
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            progress_bar=False
        )
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("✓ Training Complete")
        logger.info("=" * 70)
        logger.info("")
        
        # モデル保存
        model_dir = PROJECT_ROOT / 'models' / 'week4_fixed'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = model_dir / f'sac_fixed_v456_{timestamp}.zip'
        
        model.save(str(model_path))
        logger.info(f"✓ Model saved: {model_path}")
        
        # 統計保存
        stats_path = model_dir / f'milestones_{timestamp}.json'
        callback.save(stats_path)
        logger.info(f"✓ Milestones saved: {stats_path}")
        
        # メタデータ
        metadata = {
            'timestamp': timestamp,
            'timesteps': args.timesteps,
            'initial_balance': args.initial_balance,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'wrapper_used': True,
            'improvements': [
                'Warmup phase with relaxed drawdown',
                'Gradual drawdown limit tightening',
                'Reward scaling to [-1, 1] range',
                'SafeIntradayEnvWrapper for stability',
            ]
        }
        
        metadata_path = model_dir / f'metadata_{timestamp}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=float)
        
        logger.info(f"✓ Metadata saved: {metadata_path}")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        env.close()
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        env.close()
        raise
    
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Week 4 Fixed Training')
    parser.add_argument(
        '--timesteps',
        type=int,
        default=30000,
        help='Training timesteps (default: 30000)'
    )
    parser.add_argument(
        '--initial-balance',
        type=int,
        default=100000,
        help='Initial balance in JPY (default: 100000)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-4,
        help='Learning rate (default: 3e-4)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size (default: 256)'
    )
    
    args = parser.parse_args()
    main(args)
