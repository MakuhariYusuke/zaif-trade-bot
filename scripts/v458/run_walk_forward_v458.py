#!/usr/bin/env python3
"""
Phase 5: Walk-Forward Analysis for SAC v458 Model

複数時系列分割でのロバスト評価を実施
v458環境/設定を活用した改良実装

構成:
  1. WalkForwardSplitter: 複数分割生成
  2. WalkForwardModelEvaluator: SAC訓練・評価 (v458対応)
  3. WalkForwardReporter: 結果集約
"""

import logging
import sys
from pathlib import Path
from typing import Callable, List, Optional

try:
    import numpy as np
    import pandas as pd
    from stable_baselines3 import SAC
    from ztb.training.utils.v457_config_utils import load_config_dict, extract_env_config
    from ztb.trading.environment.utils.fast_intraday_env_v456_utils import create_fast_intraday_env_v456
except ImportError as e:
    # Fallback logging for import errors
    print(f"Import error: {e}")
    print("Please ensure all required packages are installed")
    raise

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# BacktestReporter import
scripts_path = PROJECT_ROOT / "scripts" / "v457"
if str(scripts_path) not in sys.path:
    sys.path.insert(0, str(scripts_path))
from backtest_v457 import BacktestReporter

from ztb.evaluation.walk_forward import (
    WalkForwardSplitter,
    WalkForwardModelEvaluator,
    WindowPerformance,
    WalkForwardResult,
    WalkForwardReporter,
)
from ztb.analysis.evaluation.walk_forward_integration_pipeline import WalkForwardEvaluationPipeline
from ztb.analysis.baseline_comparison import BaselineComparisonEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_walk_forward_analysis(
    df: pd.DataFrame,
    n_windows: int = 3,
    timesteps_per_window: int = 10000,
    env_factory: Optional[Callable] = None,
    model_seed: int = 42,
) -> WalkForwardResult:
    """Walk-Forward分析実行"""
    try:
        logger.info("=" * 70)
        logger.info(f"🚀 Phase 4: Walk-Forward Analysis (n_windows={n_windows})")
        logger.info("=" * 70)
        
        # ウィンドウ生成
        splitter = WalkForwardSplitter(
            initial_train_pct=0.50,
            val_pct=0.15,
            test_pct=0.15,
            step_pct=0.20,  # 増加して重なり防止
            embargo_days=0,
        )
        windows = splitter.split(df)
        
        # 必要なウィンドウ数に制限
        windows = windows[:n_windows]
        
        # 各ウィンドウで訓練・評価
        evaluator = WalkForwardModelEvaluator(env_factory=env_factory)
        performances: List[WindowPerformance] = []
        reporters: List[BacktestReporter] = []
        
        for i, window in enumerate(windows):
            try:
                model_path = Path(f"models/v458/sac_v458_seed_{model_seed}.zip")
                if model_path.exists():
                    logger.info(f"Loading pre-trained model: {model_path}")
                    model = SAC.load(str(model_path))
                    perf, reporter = evaluator.evaluate_window_with_model(df, window, model)
                    if perf is not None:
                        performances.append(perf)
                        reporters.append(reporter)
                else:
                    logger.warning(f"Model not found: {model_path}, skipping")
            except Exception as e:
                logger.error(f"Error in window {i}: {e}")
                raise  # 再スローして全体を失敗させる
        
        # 結果集約
        val_rois = [p.val_roi for p in performances]
        test_rois = [p.test_roi for p in performances]
        sharpes = [p.sharpe_ratio for p in performances]
        win_rates = [p.win_rate for p in performances]
        overfitting_ratios = [p.overfitting_ratio for p in performances]
        profit_factors = [p.profit_factor for p in performances]
        expectancies = [p.expectancy for p in performances]
        avg_wins = [p.avg_win for p in performances]
        avg_losses = [p.avg_loss for p in performances]
        
        # Sharpe一貫性（ウィンドウ間相関）→ Doc09: 1 - (std/mean)
        mean_sharpe = np.mean(sharpes) if sharpes else 0.0
        sharpe_consistency = (
            1 - (np.std(sharpes) / abs(mean_sharpe))
            if sharpes and mean_sharpe != 0 else 0.0
        )
        
        # オーバーフィッティング比（WindowPerformance定義に統一）
        avg_val = np.mean(val_rois) if val_rois else 0.0
        avg_test = np.mean(test_rois) if test_rois else 0.0
        average_overfitting_ratio = np.mean(overfitting_ratios) if overfitting_ratios else 0.0
        
        result = WalkForwardResult(
            windows=windows,
            performances=performances,
            average_val_roi=avg_val,
            average_test_roi=avg_test,
            test_roi_std=float(np.std(test_rois)) if test_rois else 0.0,
            average_sharpe=float(np.mean(sharpes)) if sharpes else 0.0,
            sharpe_consistency=float(sharpe_consistency),
            average_win_rate=float(np.mean(win_rates)) if win_rates else 0.0,
            overfitting_ratio=float(average_overfitting_ratio),
            profit_factor=float(np.mean(profit_factors)) if profit_factors else 0.0,
            expectancy=float(np.mean(expectancies)) if expectancies else 0.0,
            avg_win=float(np.mean(avg_wins)) if avg_wins else 0.0,
            avg_loss=float(np.mean(avg_losses)) if avg_losses else 0.0,
            reporters=reporters,
        )
        
        return result
    except Exception as e:
        logger.error(f"Walk-Forward analysis failed: {e}")
        raise


def main():
    try:
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
        parser.add_argument(
            "--config",
            type=str,
            default="config/v458/base/config.yaml",
            help="Config YAML path",
        )
        parser.add_argument(
            "--seeds",
            type=str,
            default="42,123",
            help="Comma-separated list of seeds for pre-trained models",
        )
        args = parser.parse_args()
        
        # データロード
        if args.data:
            data_path = Path(args.data)
        else:
            candidates = [
                Path("test_synthetic_dataset.csv"),
                Path("data/btc_jpy_training_data.csv"),
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
        
        # v458 config ロード
        logger.info(f"📋 Loading v458 config from {args.config}")
        full_config = load_config_dict(Path(args.config))
        env_config = extract_env_config(full_config)
        logger.info(f"✓ Config loaded: {len(env_config)} env params\n")
        
        # v458 env factory 作成
        def v458_env_factory(data: pd.DataFrame, **kwargs):
            return create_fast_intraday_env_v456(data, env_config=env_config, **kwargs)
        
        # Parse model seeds
        model_seeds = [int(s.strip()) for s in args.seeds.split(',')]
        
        # Walk-Forward分析実行 for each seed
        results = []
        for seed in model_seeds:
            result = run_walk_forward_analysis(
                df,
                n_windows=args.windows,
                timesteps_per_window=args.timesteps,
                env_factory=v458_env_factory,
                model_seed=seed,
            )
            results.append(result)
        
        # Doc14: Baseline比較とABテスト実行
        logger.info("\n" + "="*70)
        logger.info("📊 Doc14: Baseline Comparisons & AB Testing")
        logger.info("="*70)
        
        try:
            baseline_comparison_engine = BaselineComparisonEngine(price_data=df)
            
            for i, result in enumerate(results):
                seed = model_seeds[i]
                logger.info(f"\n🔍 Baseline Comparisons for seed {seed}:")
                
                # 結果報告
                reporter = WalkForwardReporter(result)
                reporter.report()
                
                # パイプライン統合
                pipeline = WalkForwardEvaluationPipeline()
                evaluation_result = pipeline.integrate_walk_forward_results(
                    [p for p in result.performances], [p.test_reporter for p in result.performances if p.test_reporter]
                )
                
                # Baseline比較
                model_result = pipeline._extract_model_result_from_evaluation(evaluation_result)
                baseline_results = baseline_comparison_engine.generate_baseline_results(df)
                baseline_result = baseline_comparison_engine.compare_with_baselines(model_result, baseline_results)
                
                logger.info(f"  Model: ROI {model_result.total_return:.4f}, Sharpe {model_result.sharpe_ratio:.4f}")
                for br in baseline_results:
                    logger.info(f"  {br.strategy_name}: ROI {br.total_return:.4f}, Sharpe {br.sharpe_ratio:.4f}")
                logger.info(f"  Superiority: {baseline_result.superiority_metrics}")
            
            # ABテスト (複数seed間比較)
            if len(results) > 1:
                logger.info("\n🅰️🅱️ AB Testing between seeds:")
                ab_results = baseline_comparison_engine.compare_multiple_evaluations(results)
                for comparison in ab_results:
                    logger.info(f"  {comparison}")
            
            logger.info("✅ Doc14 features validation completed")
            
        except Exception as e:
            logger.error(f"Doc14 validation failed: {e}")
            raise
        
        # 結果保存
        output_path = Path("results/phase4/walk_forward_results.json")
        reporter.save_results(output_path)
        
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
