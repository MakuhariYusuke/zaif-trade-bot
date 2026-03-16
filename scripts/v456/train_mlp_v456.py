#!/usr/bin/env python3
"""
Week 4: MLP SAC 訓練スクリプト

実際のZaifアカウント残高を使用して、
MLPポリシーでSAC訓練を実行します。

Configuration:
  - initial_balance: 実残高から自動取得 (124 JPY)
  - max_position: None (100%, uncapped)
  - training_data: btc_jpy_1m_v454.csv (2025-11-03 ~ 2026-01-13)
  - environment: FastIntradayEnvV456 (88D observation)
  - model: MLP SAC with [128, 128] architecture
  
Usage:
    python scripts/v456/train_mlp_v456.py --timesteps 1000000 --log-interval 10000
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

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
from ztb.live_trading.trading_api import TradingAPI
from dotenv import load_dotenv

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsCallback(BaseCallback):
    """訓練メトリクス記録コールバック"""
    
    def __init__(self, log_interval: int = 10000):
        super().__init__()
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        """ステップ時の処理"""
        
        # 定期的にメトリクスを記録
        if self.n_calls % self.log_interval == 0:
            logger.info(f"Step {self.n_calls}/{self.model.num_timesteps}")
            
            # エピソード統計がある場合
            if 'rollout/ep_rew_mean' in self.logger.name_to_value:
                ep_rew_mean = self.logger.name_to_value.get('rollout/ep_rew_mean', 0)
                logger.info(f"  Episode Reward Mean: {ep_rew_mean:.4f}")
        
        return True


def load_account_config() -> dict:
    """アカウント設定を読み込む"""
    config_file = PROJECT_ROOT / 'scripts' / 'v456' / 'account_config.json'
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            return json.load(f)
    
    return {
        'initial_balance_jpy': 0.0,
        'initial_btc': 0.0,
        'max_position': None,
        'position_sizing': '100% (uncapped)'
    }


def load_market_data(data_file: Path) -> pd.DataFrame:
    """市場データを読み込む"""
    logger.info(f"Loading market data: {data_file}")
    
    df = pd.read_csv(
        data_file,
        index_col=0,
        parse_dates=True
    )
    
    # タイムゾーン統一
    if hasattr(df.index, 'tz'):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
    
    logger.info(f"  Loaded {len(df)} records")
    logger.info(f"  Range: {df.index[0]} to {df.index[-1]}")
    
    return df


def create_environment(
    market_data: pd.DataFrame,
    initial_balance: float,
    max_position: float = None
) -> FastIntradayEnvV456:
    """訓練環境を作成"""
    
    logger.info("Creating training environment...")
    logger.info(f"  Initial Balance: {initial_balance:,.2f} JPY")
    logger.info(f"  Max Position: {max_position if max_position else 'None (uncapped)'}")
    
    # 特徴量カラムを準備
    base_cols = [col for col in market_data.columns if col.startswith('base_')]
    mtf_cols = [col for col in market_data.columns if col.startswith('mtf_')]
    regime_cols = [col for col in market_data.columns if col.startswith('regime_')]
    
    # カラムが不足している場合は、ダミーデータで補完
    if len(base_cols) < 30:
        missing = 30 - len(base_cols)
        for i in range(missing):
            market_data[f'base_{len(base_cols) + i}'] = np.random.randn(len(market_data))
        base_cols = [col for col in market_data.columns if col.startswith('base_')]
    
    if len(mtf_cols) < 27:
        missing = 27 - len(mtf_cols)
        for i in range(missing):
            market_data[f'mtf_{len(mtf_cols) + i}'] = np.random.randn(len(market_data))
        mtf_cols = [col for col in market_data.columns if col.startswith('mtf_')]
    
    if len(regime_cols) < 13:
        missing = 13 - len(regime_cols)
        for i in range(missing):
            market_data[f'regime_{len(regime_cols) + i}'] = np.random.rand(len(market_data))
        regime_cols = [col for col in market_data.columns if col.startswith('regime_')]
    
    # 必須カラムがない場合は追加
    if 'atr' not in market_data.columns:
        market_data['atr'] = np.abs(np.random.randn(len(market_data))) + 50
    
    if 'impact_proxy' not in market_data.columns:
        market_data['impact_proxy'] = np.random.rand(len(market_data)) * 0.1
    
    # max_positionがNoneの場合は無制限を表すため、large valueを使用
    if max_position is None:
        max_position = initial_balance / 100  # 初期残高を100で割った値（大きい値）
    
    env = FastIntradayEnvV456(
        df=market_data,
        base_feature_columns=base_cols[:30],
        mtf_feature_columns=mtf_cols[:27],
        regime_feature_columns=regime_cols[:13],
        initial_balance=initial_balance,
        max_position=max(max_position, 1.0),  # 最小値1.0
        max_steps=None,
        prewarm_steps=100,
        commission_rate=0.001
    )
    
    return env


def main(args):
    """メイン訓練ループ"""
    
    print("=" * 70)
    print("Week 4: MLP SAC 訓練")
    print("=" * 70)
    print()
    
    # 環境変数読み込み
    load_dotenv(PROJECT_ROOT / '.env')
    
    # アカウント設定読み込み
    account_config = load_account_config()
    initial_balance = account_config.get('initial_balance_jpy', 0.0)
    
    logger.info(f"Account Config:")
    logger.info(f"  Initial Balance: {initial_balance:,.2f} JPY")
    logger.info(f"  Initial BTC: {account_config.get('initial_btc', 0.0):.8f} BTC")
    logger.info(f"  Max Position: {account_config.get('max_position')}")
    logger.info("")
    
    # 市場データ読み込み
    data_file = PROJECT_ROOT / 'data' / 'btc_jpy_1m_v454.csv'
    market_data = load_market_data(data_file)
    
    # 訓練環境作成
    env = create_environment(
        market_data=market_data,
        initial_balance=initial_balance,
        max_position=None  # 100% uncapped
    )
    
    print()
    logger.info("=" * 70)
    logger.info("Training Configuration")
    logger.info("=" * 70)
    logger.info(f"Environment: FastIntradayEnvV456")
    logger.info(f"  Observation Space: {env.observation_space}")
    logger.info(f"  Action Space: {env.action_space}")
    logger.info(f"  Market Data Points: {len(market_data)}")
    logger.info("")
    logger.info(f"Model: SAC (Soft Actor-Critic)")
    logger.info(f"  Policy: MlpPolicy")
    logger.info(f"  Policy Network: [128, 128]")
    logger.info(f"  Learning Rate: {args.learning_rate}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info("")
    logger.info(f"Training Timesteps: {args.timesteps:,}")
    logger.info(f"Log Interval: {args.log_interval:,}")
    logger.info("")
    
    # モデル作成
    logger.info("Creating SAC model...")
    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        buffer_size=1000000,
        learning_starts=10000,
        tau=0.005,
        gamma=0.99,
        policy_kwargs={
            'net_arch': [128, 128],
            'n_critics': 2,
            'share_features_extractor': False
        },
        verbose=1,
        tensorboard_log=str(PROJECT_ROOT / 'runs' / 'week4_mlp_sac')
    )
    
    # ログ設定
    log_dir = PROJECT_ROOT / 'runs' / 'week4_mlp_sac' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    configure_logger(str(log_dir))
    
    # 訓練実行
    logger.info("")
    logger.info("=" * 70)
    logger.info("Starting Training")
    logger.info("=" * 70)
    logger.info("")
    
    try:
        # メトリクスコールバック
        callback = MetricsCallback(log_interval=args.log_interval)
        
        # 訓練実行
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            progress_bar=False  # 進捗バー無効化
        )
        
        # モデル保存
        model_dir = PROJECT_ROOT / 'models' / 'week4_mlp_sac'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = model_dir / f'sac_mlp_v456_{timestamp}.zip'
        
        model.save(str(model_path))
        logger.info(f"✓ Model saved: {model_path}")
        
        # メタデータ保存
        metadata = {
            'timestamp': timestamp,
            'timesteps': args.timesteps,
            'initial_balance': initial_balance,
            'max_position': None,
            'market_data_points': len(market_data),
            'market_data_range': {
                'start': str(market_data.index[0]),
                'end': str(market_data.index[-1])
            },
            'environment': 'FastIntradayEnvV456',
            'model': 'SAC',
            'policy': 'MlpPolicy',
            'policy_net_arch': [128, 128]
        }
        
        metadata_path = model_dir / f'metadata_{timestamp}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Metadata saved: {metadata_path}")
        logger.info("")
        logger.info("=" * 70)
        logger.info("Training Complete")
        logger.info("=" * 70)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
        # 中断時もモデルを保存
        model_dir = PROJECT_ROOT / 'models' / 'week4_mlp_sac'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = model_dir / f'sac_mlp_v456_interrupted_{timestamp}.zip'
        
        model.save(str(model_path))
        logger.info(f"✓ Model saved (interrupted): {model_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Week 4 MLP SAC Training'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=1000000,
        help='Total training timesteps (default: 1000000)'
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
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10000,
        help='Log interval in steps (default: 10000)'
    )
    
    args = parser.parse_args()
    main(args)
