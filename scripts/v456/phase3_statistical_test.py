#!/usr/bin/env python3
"""
Phase 3: モデル vs ベースラインの有意性検定

既存インフラの活用:
- scipy.stats: paired t-test
- 時系列分割による unbiased 評価
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from stable_baselines3 import SAC

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.v456.phase3_oos_evaluation import (
    TimeSeriesSplitter,
    RuleBasedBaseline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(csv_path: Path = None) -> pd.DataFrame:
    """データをロード"""
    if csv_path is None:
        candidates = [
            Path("test_synthetic_dataset.csv"),
            Path("data/datasets/test_synthetic_dataset.csv"),
        ]
        for c in candidates:
            if c.exists():
                csv_path = c
                break
    
    if csv_path is None:
        raise FileNotFoundError("No data file found")
    
    logger.info(f"📥 Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # タイムスタンプ処理
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
    
    logger.info(f"✓ Loaded {len(df)} bars\n")
    return df


def evaluate_baseline(test_df: pd.DataFrame) -> Dict:
    """ベースライン（RSI/MACD）の評価"""
    logger.info("=" * 70)
    logger.info("📊 Baseline (RSI/MACD) Evaluation")
    logger.info("=" * 70)
    
    baseline = RuleBasedBaseline(test_df.copy())
    
    # シグナル生成
    signals = baseline.generate_signals()
    logger.info(f"Signals generated: {len(signals)} steps")
    
    # バックテスト実行
    results = baseline.backtest()
    logger.info(f"\nBaseline Results:")
    logger.info(f"  Win Rate: {results['win_rate']:.2%}")
    logger.info(f"  Total Return: {results['total_return']:.6f}")
    
    # オプショナルなキーを安全にアクセス
    if "sharpe_ratio" in results:
        logger.info(f"  Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    if "final_balance" in results:
        logger.info(f"  Final Balance: {results['final_balance']:,.0f} JPY")
    
    return results


def evaluate_model(model: SAC, test_df: pd.DataFrame, label: str = "test") -> Dict:
    """訓練済みモデルの評価"""
    from train_and_evaluate_v456_phase3 import create_environment_wrapper
    
    logger.info(f"\n📊 Model Evaluation ({label})")
    
    # 環境作成
    eval_env = create_environment_wrapper(test_df, None)
    
    # 評価実行
    obs, _ = eval_env.reset()
    done = False
    episode_reward = 0.0
    steps = 0
    actions = []
    trades = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        
        episode_reward += float(reward)
        steps += 1
        actions.append(float(action[0]))
        
        if info.get("trade_executed"):
            trades += 1
    
    final_balance = eval_env.balance
    total_pnl = final_balance - eval_env.initial_balance
    roi = (total_pnl / eval_env.initial_balance)
    
    logger.info(f"  Episode Reward: {episode_reward:.6f}")
    logger.info(f"  Steps: {steps}")
    logger.info(f"  Trades: {trades}")
    logger.info(f"  Final Balance: {final_balance:,.0f} JPY")
    logger.info(f"  Total P&L: {total_pnl:,.0f} JPY")
    logger.info(f"  ROI: {roi:.4f}")
    
    return {
        "episode_reward": episode_reward,
        "steps": steps,
        "trades": trades,
        "final_balance": final_balance,
        "total_pnl": total_pnl,
        "roi": roi,
        "action_mean": float(np.mean(actions)),
        "action_std": float(np.std(actions)),
    }


def perform_significance_test(
    model_returns: np.ndarray,
    baseline_returns: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """
    Paired t-test で有意性を検定
    
    H0: Model returns = Baseline returns
    H1: Model returns != Baseline returns
    """
    logger.info("\n" + "=" * 70)
    logger.info("📈 Statistical Significance Test")
    logger.info("=" * 70)
    
    # データの基本統計量
    logger.info(f"\nBasic Statistics:")
    logger.info(f"  Model:    mean={np.mean(model_returns):.6f}, std={np.std(model_returns):.6f}")
    logger.info(f"  Baseline: mean={np.mean(baseline_returns):.6f}, std={np.std(baseline_returns):.6f}")
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(model_returns, baseline_returns)
    
    logger.info(f"\nPaired t-test Results:")
    logger.info(f"  t-statistic: {t_stat:.6f}")
    logger.info(f"  p-value: {p_value:.6f}")
    logger.info(f"  Alpha (significance level): {alpha}")
    
    is_significant = p_value < alpha
    
    if is_significant:
        if t_stat > 0:
            logger.info(f"  ✓ Model SIGNIFICANTLY BETTER than baseline (p < {alpha})")
        else:
            logger.info(f"  ✗ Model SIGNIFICANTLY WORSE than baseline (p < {alpha})")
    else:
        logger.info(f"  ○ No significant difference (p >= {alpha})")
    
    # Effect size (Cohen's d)
    mean_diff = np.mean(model_returns - baseline_returns)
    std_diff = np.std(model_returns - baseline_returns, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0
    
    logger.info(f"\nEffect Size (Cohen's d): {cohens_d:.4f}")
    if abs(cohens_d) < 0.2:
        effect = "negligible"
    elif abs(cohens_d) < 0.5:
        effect = "small"
    elif abs(cohens_d) < 0.8:
        effect = "medium"
    else:
        effect = "large"
    logger.info(f"  Interpretation: {effect}")
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "is_significant": bool(is_significant),
        "alpha": alpha,
        "cohens_d": float(cohens_d),
        "effect_size": effect,
        "mean_difference": float(mean_diff),
        "std_difference": float(std_diff),
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 3: Statistical Significance Test")
    parser.add_argument("--data", type=str, default=None, help="Data CSV path")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    args = parser.parse_args()
    
    # データロード
    df = load_data(Path(args.data) if args.data else None)
    
    # Time-Series split
    logger.info("=" * 70)
    logger.info("🔀 Time-Series Split")
    logger.info("=" * 70)
    splitter = TimeSeriesSplitter(df, train_ratio=0.70, embargo_days=7)
    _, _, test_df = splitter.split()
    
    # モデルロード
    logger.info("\n📦 Loading model...")
    model = SAC.load(args.model)
    logger.info(f"✓ Model loaded from {args.model}")
    
    # ベースライン評価
    baseline_results = evaluate_baseline(test_df)
    
    # モデル評価
    model_results = evaluate_model(model, test_df, label="test")
    
    # 有意性検定
    # 簡略版: ROI を比較
    # 実運用ではトレードごとのリターンで検定すべき
    model_returns = np.array([model_results["roi"]])
    baseline_returns = np.array([baseline_results["total_return"]])
    
    # より詳細な検定用にシミュレーションデータを使用
    logger.info("\n" + "=" * 70)
    logger.info("⚠️  Note: Using single-sample comparison for demonstration")
    logger.info("In production, aggregate returns across rolling windows")
    logger.info("=" * 70)
    
    # 最終レポート
    logger.info("\n" + "=" * 70)
    logger.info("📋 Final Comparison")
    logger.info("=" * 70)
    logger.info(f"\nModel:")
    for k, v in model_results.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.6f}")
    
    logger.info(f"\nBaseline:")
    for k, v in baseline_results.items():
        logger.info(f"  {k}: {v}")
    
    # 結論
    if model_results["roi"] > baseline_results["total_return"]:
        logger.info(f"\n✓ Model OUTPERFORMS baseline by {(model_results['roi'] - baseline_results['total_return']):.4f}")
    else:
        logger.info(f"\n✗ Model UNDERPERFORMS baseline by {(baseline_results['total_return'] - model_results['roi']):.4f}")
    
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
