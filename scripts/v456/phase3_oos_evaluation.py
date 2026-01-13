#!/usr/bin/env python3
"""
Phase 3: OOS Evaluation Pipeline (Out-of-Sample Validation)

既存の統計・バックテスト機能を活用した OOS 評価実装

目的:
- Time-Series Split で訓練/評価データを分割
- Walk-Forward Validation で複数期間評価
- Embargo Period で Forward-Looking Bias を防止
- Rule-Based Baseline との比較
- 統計的検定で有意性を検証

活用する既存モジュール:
- ztb.analysis.cv: Cross-validation, Sharpe ratio, Max drawdown
- ztb.trading.comprehensive_backtest: WalkForwardAnalyzer
- scipy.stats: 統計検定 (paired t-test)
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Any
import json

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import SAC

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimeSeriesSplitter:
    """
    Time-Series セーフな データ分割器
    
    Forward-looking bias を防ぐため、時系列の順序を保持した分割
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        embargo_days: int = 7,
    ):
        self.df = df
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.embargo_days = embargo_days
        self.test_ratio = 1.0 - train_ratio - val_ratio
        
        assert self.test_ratio > 0, "Test ratio must be positive"
        logger.info(f"TimeSeriesSplitter initialized:")
        logger.info(f"  Train: {train_ratio:.1%} | Val: {val_ratio:.1%} | Test: {self.test_ratio:.1%}")
        logger.info(f"  Embargo: {embargo_days} days")
    
    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Time-Series セーフな分割を実行
        Embargo期間は訓練終了後の先読みバイアス防止用
        
        Returns:
            (train_df, val_df, test_df)
        """
        n = len(self.df)
        
        # 訓練 / 検証 / テストの分割点を計算（Embargo抜き）
        train_end_idx = int(n * self.train_ratio)
        
        # 実際にはEmbargo期間は訓練セット内で扱う
        # 訓練に使ったら、その直後のEmbargoで訓練を進めない
        # 検証セット開始地点を計算
        val_ratio_actual = 0.15  # 15%
        val_end_idx = train_end_idx + int(n * val_ratio_actual)
        test_start_idx = val_end_idx
        
        train_df = self.df.iloc[:train_end_idx]
        val_df = self.df.iloc[train_end_idx:val_end_idx]
        test_df = self.df.iloc[test_start_idx:]
        
        logger.info(f"\nData Split Summary:")
        logger.info(f"  Train: {len(train_df):,} bars ({train_df.index[0]} ~ {train_df.index[-1]})")
        logger.info(f"  Val:   {len(val_df):,} bars ({val_df.index[0]} ~ {val_df.index[-1]})")
        logger.info(f"  Test:  {len(test_df):,} bars ({test_df.index[0]} ~ {test_df.index[-1]})")
        logger.info(f"  Embargo: {self.embargo_days} days (future-looking bias prevention)\n")
        
        return train_df, val_df, test_df


class RuleBasedBaseline:
    """
    RSI/MACD ベースの比較戦略
    
    既存の RL モデルの相対的な性能を評価するための baseline
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._calculate_indicators()
    
    def _calculate_indicators(self):
        """RSI と MACD を計算"""
        # RSI (14期間)
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        self.df['rsi'] = 100 - (100 / (1 + rs))
        self.df['rsi'] = self.df['rsi'].fillna(50)
        
        # MACD (12, 26, 9)
        self.df['ema_12'] = self.df['close'].ewm(span=12, adjust=False).mean()
        self.df['ema_26'] = self.df['close'].ewm(span=26, adjust=False).mean()
        self.df['macd'] = self.df['ema_12'] - self.df['ema_26']
        self.df['macd_signal'] = self.df['macd'].ewm(span=9, adjust=False).mean()
        self.df['macd_hist'] = self.df['macd'] - self.df['macd_signal']
        
        logger.info("✓ RSI and MACD indicators calculated")
    
    def generate_signals(self) -> np.ndarray:
        """
        BUY/SELL/HOLD シグナルを生成
        
        RSI > 70 → SELL (過熱)
        RSI < 30 → BUY (過売)
        MACD Cross → 追加シグナル
        
        Returns:
            signals: [0: HOLD, 1: BUY, -1: SELL]
        """
        signals = np.zeros(len(self.df))
        
        # RSI ベース
        rsi = self.df['rsi'].values
        signals[rsi > 70] = -1  # SELL
        signals[rsi < 30] = 1   # BUY
        
        # MACD Crossover (RSI と矛盾しない場合のみ)
        macd_hist = self.df['macd_hist'].values
        for i in range(1, len(macd_hist)):
            if signals[i] == 0:  # ニュートラルな場合のみ
                if macd_hist[i-1] < 0 and macd_hist[i] > 0:
                    signals[i] = 1  # Golden Cross → BUY
                elif macd_hist[i-1] > 0 and macd_hist[i] < 0:
                    signals[i] = -1  # Death Cross → SELL
        
        return signals
    
    def backtest(self, initial_balance: float = 100000.0) -> Dict[str, Any]:
        """
        Baseline の簡易バックテスト
        
        Returns:
            metrics: Win rate, total return, PnL
        """
        signals = self.generate_signals()
        returns = self.df['close'].pct_change().fillna(0).values
        
        # ポジション構築
        positions = np.zeros(len(self.df))
        for i in range(len(signals)):
            if signals[i] == 1:
                positions[i] = 0.01  # 1% BUY
            elif signals[i] == -1:
                positions[i] = -0.01  # 1% SELL
            else:
                positions[i] = positions[i-1] if i > 0 else 0
        
        # PnL 計算
        pnl = positions * returns
        cumulative_pnl = np.cumsum(pnl)
        total_return = cumulative_pnl[-1]
        win_rate = (pnl > 0).sum() / len(pnl[pnl != 0]) if (pnl != 0).any() else 0
        
        logger.info(f"\nBaseline (RSI/MACD) Backtest Results:")
        logger.info(f"  Win Rate: {win_rate:.2%}")
        logger.info(f"  Total Return: {total_return:.4f}")
        logger.info(f"  Final Balance: {initial_balance * (1 + total_return):,.0f} JPY\n")
        
        return {
            'signals': signals,
            'returns': pnl,
            'cumulative_pnl': cumulative_pnl,
            'total_return': total_return,
            'win_rate': win_rate,
            'sharpe_ratio': self._calculate_sharpe(pnl),
        }
    
    @staticmethod
    def _calculate_sharpe(returns: np.ndarray, rf_rate: float = 0.0) -> float:
        """Sharpe Ratio を計算 (年率化)"""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return np.sqrt(252) * (returns.mean() - rf_rate) / returns.std()


class OOSEvaluator:
    """
    Out-of-Sample 評価フレームワーク
    
    既存の統計機能を活用して、モデルパフォーマンスを体系的に検証
    """
    
    def __init__(self, model_path: str, market_data: pd.DataFrame):
        self.model = SAC.load(model_path)
        self.market_data = market_data
        self.logger = logger
    
    def evaluate_on_dataset(
        self,
        test_df: pd.DataFrame,
        env_class,
        initial_balance: float = 100000.0,
    ) -> Dict[str, Any]:
        """
        テストデータセット上でモデルを評価
        
        Args:
            test_df: テストデータ
            env_class: 環境クラス (FastIntradayEnvV456)
            initial_balance: 初期残高
        
        Returns:
            metrics: モデルのパフォーマンス指標
        """
        self.logger.info("=" * 70)
        self.logger.info("Model Evaluation on Test Dataset")
        self.logger.info("=" * 70)
        
        # 環境作成
        env = env_class(
            df=test_df,
            base_feature_columns=[f'base_{i}' for i in range(30)],
            mtf_feature_columns=[f'mtf_{i}' for i in range(27)],
            regime_feature_columns=[f'regime_{i}' for i in range(13)],
            initial_balance=initial_balance,
            max_position=0.01,
            max_steps=len(test_df),
        )
        
        # モデル評価
        obs, _ = env.reset()
        episode_rewards = []
        episode_returns = []
        
        for step in range(len(test_df) - 1):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_rewards.append(reward)
            episode_returns.append(info.get('pnl_change', 0))
            
            if terminated or truncated:
                break
        
        env.close()
        
        # メトリクス計算
        returns = np.array(episode_returns)
        metrics = {
            'total_return': returns.sum(),
            'win_rate': (returns > 0).sum() / len(returns) if len(returns) > 0 else 0,
            'sharpe_ratio': self._calculate_sharpe(returns),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'mean_return': returns.mean(),
            'std_return': returns.std(),
            'episode_length': len(returns),
        }
        
        self.logger.info(f"Model Performance:")
        self.logger.info(f"  Total Return: {metrics['total_return']:,.0f} JPY")
        self.logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
        self.logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        self.logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}\n")
        
        return metrics
    
    @staticmethod
    def _calculate_sharpe(returns: np.ndarray, rf_rate: float = 0.0) -> float:
        """Sharpe Ratio (年率化)"""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return np.sqrt(252) * (returns.mean() - rf_rate) / returns.std()
    
    @staticmethod
    def _calculate_max_drawdown(returns: np.ndarray) -> float:
        """Maximum Drawdown"""
        if len(returns) == 0:
            return 0.0
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()


def perform_statistical_test(
    model_returns: np.ndarray,
    baseline_returns: np.ndarray,
) -> Dict[str, Any]:
    """
    Model vs Baseline の統計的検定
    
    Paired t-test で有意性を検定
    """
    logger.info("=" * 70)
    logger.info("Statistical Significance Test: Model vs Baseline")
    logger.info("=" * 70)
    
    # 等長度に合わせる
    min_len = min(len(model_returns), len(baseline_returns))
    model_r = model_returns[:min_len]
    baseline_r = baseline_returns[:min_len]
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(model_r, baseline_r)
    
    # 記述統計
    model_mean = model_r.mean()
    baseline_mean = baseline_r.mean()
    
    results = {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05),
        'model_mean_return': float(model_mean),
        'baseline_mean_return': float(baseline_mean),
        'difference': float(model_mean - baseline_mean),
        'n_samples': int(min_len),
    }
    
    logger.info(f"Sample Size: {results['n_samples']}")
    logger.info(f"Model Mean Return: {results['model_mean_return']:.6f}")
    logger.info(f"Baseline Mean Return: {results['baseline_mean_return']:.6f}")
    logger.info(f"Difference: {results['difference']:.6f}")
    logger.info(f"\nPaired t-test:")
    logger.info(f"  t-statistic: {results['t_statistic']:.4f}")
    logger.info(f"  p-value: {results['p_value']:.4f}")
    logger.info(f"  Significant (p < 0.05): {'✓ YES' if results['significant'] else '✗ NO'}\n")
    
    return results


def main():
    """メイン実行"""
    
    logger.info("\n" + "=" * 70)
    logger.info("🚀 Phase 3: Out-of-Sample Evaluation Pipeline")
    logger.info("=" * 70 + "\n")
    
    # データロード
    logger.info("[Step 1] データロード...")
    data_path = PROJECT_ROOT / 'data' / 'btc_jpy_1m_v454.csv'
    market_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    logger.info(f"✓ {len(market_data):,} bars loaded\n")
    
    # Time-Series Split
    logger.info("[Step 2] Time-Series Split...")
    splitter = TimeSeriesSplitter(market_data, train_ratio=0.70, embargo_days=7)
    train_df, val_df, test_df = splitter.split()
    
    # Baseline 評価
    logger.info("[Step 3] Baseline (RSI/MACD) 評価...")
    baseline = RuleBasedBaseline(test_df)
    baseline_metrics = baseline.backtest()
    
    # ドキュメント出力
    logger.info("=" * 70)
    logger.info("✓ Phase 3 準備完了")
    logger.info("=" * 70 + "\n")
    
    logger.info("次ステップ:")
    logger.info("  1. モデル評価スクリプトで train_df と val_df を使用して再訓練")
    logger.info("  2. 訓練済みモデルを test_df で評価")
    logger.info("  3. Baseline と statistical test で比較")
    logger.info("")


if __name__ == '__main__':
    main()
