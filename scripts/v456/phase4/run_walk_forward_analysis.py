#!/usr/bin/env python3
"""
Phase 4: Walk-Forward Analysis for SAC v456 Model

複数時系列分割でのロバスト評価を実施
既存インフラを活用した改良実装

構成:
  1. WalkForwardSplitter: 複数分割生成
  2. WalkForwardModelEvaluator: SAC訓練・評価
  3. WalkForwardReporter: 結果集約
"""

import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "v456"))
sys.path.insert(0, str(Path(__file__).parent / "modules"))

from modules import (
    WalkForwardSplitter,
    WalkForwardModelEvaluator,
    WindowPerformance,
    WalkForwardResult,
    WalkForwardReporter,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_walk_forward_analysis(
    df: pd.DataFrame,
    n_windows: int = 3,
    timesteps_per_window: int = 10000,
) -> WalkForwardResult:
    """Walk-Forward分析実行"""
    logger.info("=" * 70)
    logger.info(f"🚀 Phase 4: Walk-Forward Analysis (n_windows={n_windows})")
    logger.info("=" * 70)
    
    # ウィンドウ生成
    splitter = WalkForwardSplitter(
        initial_train_pct=0.50,
        val_pct=0.15,
        test_pct=0.15,
        step_pct=0.20,
    )
    windows = splitter.split(df)
    
    # 必要なウィンドウ数に制限
    windows = windows[:n_windows]
    
    # 各ウィンドウで訓練・評価
    evaluator = WalkForwardModelEvaluator()
    performances: List[WindowPerformance] = []
    
    for window in windows:
        perf = evaluator.train_and_evaluate_window(
            df,
            window,
            timesteps=timesteps_per_window,
        )
        performances.append(perf)
    
    # 結果集約
    val_rois = [p.val_roi for p in performances]
    test_rois = [p.test_roi for p in performances]
    sharpes = [p.sharpe_ratio for p in performances]
    win_rates = [p.win_rate for p in performances]
    
    # Sharpe一貫性（ウィンドウ間相関）
    sharpe_consistency = (
        np.corrcoef(range(len(sharpes)), sharpes)[0, 1]
        if len(sharpes) > 1 else 0.0
    )
    
    # オーバーフィッティング比（訓練vs テスト）
    avg_val = np.mean(val_rois) if val_rois else 0.0
    avg_test = np.mean(test_rois) if test_rois else 0.0
    overfitting_ratio = avg_val / avg_test if avg_test > 0 else 0.0
    
    result = WalkForwardResult(
        windows=windows,
        performances=performances,
        average_val_roi=avg_val,
        average_test_roi=avg_test,
        test_roi_std=float(np.std(test_rois)) if test_rois else 0.0,
        average_sharpe=float(np.mean(sharpes)) if sharpes else 0.0,
        sharpe_consistency=float(sharpe_consistency),
        average_win_rate=float(np.mean(win_rates)) if win_rates else 0.0,
        overfitting_ratio=float(overfitting_ratio),
    )
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 4: Walk-Forward Analysis")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Data CSV path",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=3,
        help="Number of walk-forward windows",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10000,
        help="Timesteps per window",
    )
    args = parser.parse_args()
    
    # データロード
    if args.data:
        data_path = Path(args.data)
    else:
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
            logger.error("No data file found")
            return
    
    logger.info(f"📥 Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
    
    logger.info(f"✓ Loaded {len(df)} bars\n")
    
    # Walk-Forward分析実行
    result = run_walk_forward_analysis(
        df,
        n_windows=args.windows,
        timesteps_per_window=args.timesteps,
    )
    
    # 結果報告
    reporter = WalkForwardReporter(result)
    reporter.report()
    
    # 結果保存
    output_path = Path("results/phase4/walk_forward_results.json")
    reporter.save_results(output_path)


if __name__ == "__main__":
    main()
