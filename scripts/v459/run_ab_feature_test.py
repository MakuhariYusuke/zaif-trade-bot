"""
Phase 4 Week 1 Day 5: 8特徴 vs フル特徴 A/B検証実験

目的:
- 8特徴削減が収益性に与える影響を実測
- 収益率の分散を測定し、統計検定の検出力を評価
- Week 2の判定材料を提供

実験設計:
- 2 seeds × 2 configs (8特徴 vs フル特徴) = 4 experiments
- 測定: Net ROI, Sharpe Ratio, 総時間、収益率分散
"""
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer
from ztb.training.unified_trainer.trainer import UnifiedTrainer
from ztb.training.utils.env_metrics import (
    compute_balance_roi,
    extract_trainer_env_metrics,
)
from ztb.utils.logging_utils import get_logger

logger = get_logger("ab_feature_test")


def create_experiment_config(
    data_path: str, seed: int, experiment_name: str
) -> dict:
    """実験設定を生成（run_ab_reward_experiments.pyと同じフォーマット）"""
    return {
        "experiment_name": experiment_name,
        "training": {
            "algorithm": "SAC",
            "total_timesteps": 50000,  # Phase 3.5と同じ
            "eval_freq": 5000,
            "n_eval_episodes": 3,
            "log_interval": 100,
            "seed": seed,
            "sac_hyperparameters": {
                "learning_rate": 0.0003,
                "buffer_size": 100000,
                "learning_starts": 1000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1,
                "ent_coef": "auto",
                "target_update_interval": 1,
                "target_entropy": "auto",
            },
            "data_config": {
                "data_path": data_path,
                "window_size": 60,
            },
            "environment": {
                "use_continuous_actions": True,
                "action_space_type": "continuous",
                "initial_portfolio_value": 100000.0,
                "transaction_cost": 0.001,
            },
            "walk_forward": {
                "enabled": True,
                "n_splits": 4,
                "train_size": 0.6,
                "validation_size": 0.2,
                "test_size": 0.2,
            },
        },
    }


def run_single_experiment(config: dict, output_dir: Path) -> dict:
    """1回の実験を実行"""
    start_time = time.time()
    experiment_name = config["experiment_name"]

    logger.info(f"\n{'='*60}")
    logger.info(f"実験開始: {experiment_name}")
    logger.info(f"{'='*60}")

    try:
        # SACTrainerを直接使用してトレーニング結果を取得
        trainer = SACTrainer(config=config, logger=logger)
        
        # トレーニング実行
        result = trainer.train()
        
        elapsed_time = time.time() - start_time

        # 結果を整形
        data_path = config.get("training", {}).get("data_config", {}).get("data_path", "unknown")
        training_result = {
            "experiment_name": experiment_name,
            "data_path": data_path,
            "seed": config["training"]["seed"],
            "total_time_seconds": elapsed_time,
            "success": True,
            "timestamp": datetime.now().isoformat(),
        }

        # トレーニング結果を統合
        if result:
            if isinstance(result, dict):
                training_result.update(result)
            elif hasattr(result, "__dict__"):
                training_result.update(vars(result))
        
        # 環境からメトリクスを抽出
        try:
            metrics = extract_trainer_env_metrics(trainer, include_optional=True)
            if metrics:
                if "final_balance" in metrics and "initial_balance" not in metrics:
                    metrics["initial_balance"] = 100000.0

                for key, value in metrics.items():
                    if key not in training_result:
                        training_result[key] = value

                roi = compute_balance_roi(metrics)
                if roi is not None:
                    training_result["final_roi"] = roi
                    training_result["net_roi"] = roi

                logger.info("環境から取得したメトリクス:")
                logger.info(
                    f"  final_balance: {training_result.get('final_balance', 'N/A')}"
                )
                logger.info(f"  ROI: {training_result.get('final_roi', 'N/A')}")
                logger.info(
                    f"  total_trades: {training_result.get('total_trades', 'N/A')}"
                )
            else:
                logger.warning("環境へのアクセスに失敗しました")
        except Exception as env_error:
            logger.error(f"環境メトリクス取得エラー: {env_error}", exc_info=True)
        
        # 最終評価メトリクスを取得（trainer属性がある場合）
        if hasattr(trainer, "final_balance") and "final_balance" not in training_result:
            training_result["final_balance"] = trainer.final_balance
        if hasattr(trainer, "final_roi") and "final_roi" not in training_result:
            training_result["final_roi"] = trainer.final_roi
        if hasattr(trainer, "sharpe_ratio") and "sharpe_ratio" not in training_result:
            training_result["sharpe_ratio"] = trainer.sharpe_ratio
        if hasattr(trainer, "max_drawdown"):
            training_result["max_drawdown"] = trainer.max_drawdown
        if hasattr(trainer, "profit_factor"):
            training_result["profit_factor"] = trainer.profit_factor

        logger.info(f"✅ 実験完了: {experiment_name} ({elapsed_time:.1f}秒)")
        logger.info(f"   ROI: {training_result.get('final_roi', 'N/A')}")
        logger.info(f"   Sharpe: {training_result.get('sharpe_ratio', 'N/A')}")
        logger.info(f"   取引回数: {training_result.get('total_trades', 'N/A')}")
        return training_result

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"❌ 実験失敗: {experiment_name} - {e}", exc_info=True)
        data_path = config.get("training", {}).get("data_config", {}).get("data_path", "unknown")
        return {
            "experiment_name": experiment_name,
            "data_path": data_path,
            "seed": config["training"]["seed"],
            "total_time_seconds": elapsed_time,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def analyze_results(results: list, output_dir: Path) -> dict:
    """実験結果を分析"""
    logger.info(f"\n{'='*60}")
    logger.info("結果分析")
    logger.info(f"{'='*60}")

    # データタイプ別に分類
    feature8_results = [r for r in results if "optimized_features" in r["data_path"]]
    full_results = [r for r in results if "optimized_features" not in r["data_path"]]

    def extract_metrics(results_list):
        """メトリクスを抽出"""
        times = [r["total_time_seconds"] for r in results_list if r["success"]]
        # 収益率などの指標を抽出（実際のキー名は結果に依存）
        rois = []
        sharpes = []
        final_balances = []
        
        for r in results_list:
            if r["success"]:
                # 結果から収益率を抽出（複数のキー名を試行）
                roi = r.get("net_roi") or r.get("roi") or r.get("mean_reward", 0)
                if roi == 0 and "final_balance" in r:
                    roi = (r["final_balance"] / 100000) - 1
                rois.append(roi * 100 if abs(roi) < 1 else roi)  # パーセント調整
                final_balances.append(r.get("final_balance", 100000))

                sharpe = r.get("sharpe_ratio") or r.get("sharpe", 0)
                sharpes.append(sharpe)

        return {
            "time_mean": np.mean(times) if times else 0,
            "time_std": np.std(times) if times else 0,
            "roi_mean": np.mean(rois) if rois else 0,
            "roi_std": np.std(rois) if rois else 0,
            "sharpe_mean": np.mean(sharpes) if sharpes else 0,
            "sharpe_std": np.std(sharpes) if sharpes else 0,
            "final_balance_mean": np.mean(final_balances) if final_balances else 100000,
            "final_balance_std": np.std(final_balances) if final_balances else 0,
            "success_count": len([r for r in results_list if r["success"]]),
            "total_count": len(results_list),
        }

    feature8_metrics = extract_metrics(feature8_results)
    full_metrics = extract_metrics(full_results)

    analysis = {
        "feature8_results": {
            "description": "8特徴Parquet (相関0.95削減)",
            "metrics": feature8_metrics,
        },
        "full_results": {
            "description": "フル特徴CSV",
            "metrics": full_metrics,
        },
        "comparison": {
            "time_reduction_pct": (
                (1 - feature8_metrics["time_mean"] / full_metrics["time_mean"]) * 100
                if full_metrics["time_mean"] > 0
                else 0
            ),
            "roi_difference_pct": feature8_metrics["roi_mean"]
            - full_metrics["roi_mean"],
            "sharpe_difference": feature8_metrics["sharpe_mean"]
            - full_metrics["sharpe_mean"],
            "variance_ratio": (
                feature8_metrics["roi_std"] / full_metrics["roi_std"]
                if full_metrics["roi_std"] > 0
                else 1.0
            ),
        },
    }

    # 結果表示
    logger.info("\n8特徴Parquet:")
    logger.info(f"  平均時間: {feature8_metrics['time_mean']:.1f}秒 (±{feature8_metrics['time_std']:.1f})")
    logger.info(f"  平均ROI: {feature8_metrics['roi_mean']:.2f}% (±{feature8_metrics['roi_std']:.2f})")
    logger.info(f"  平均Sharpe: {feature8_metrics['sharpe_mean']:.3f} (±{feature8_metrics['sharpe_std']:.3f})")
    logger.info(f"  成功率: {feature8_metrics['success_count']}/{feature8_metrics['total_count']}")

    logger.info("\nフル特徴CSV:")
    logger.info(f"  平均時間: {full_metrics['time_mean']:.1f}秒 (±{full_metrics['time_std']:.1f})")
    logger.info(f"  平均ROI: {full_metrics['roi_mean']:.2f}% (±{full_metrics['roi_std']:.2f})")
    logger.info(f"  平均Sharpe: {full_metrics['sharpe_mean']:.3f} (±{full_metrics['sharpe_std']:.3f})")
    logger.info(f"  成功率: {full_metrics['success_count']}/{full_metrics['total_count']}")

    logger.info("\n比較:")
    logger.info(f"  時間削減: {analysis['comparison']['time_reduction_pct']:.1f}%")
    logger.info(f"  ROI差: {analysis['comparison']['roi_difference_pct']:+.2f}%")
    logger.info(f"  Sharpe差: {analysis['comparison']['sharpe_difference']:+.3f}")
    logger.info(f"  分散比: {analysis['comparison']['variance_ratio']:.2f}x")

    return analysis


def main():
    """メイン実行"""
    output_dir = Path("results/phase4_day5_ab_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    log_file = output_dir / f"ab_test_{timestamp}.log"

    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logger.info("Phase 4 Week 1 Day 5: 8特徴 vs フル特徴 A/B検証")
    logger.info(f"ログファイル: {log_file}")

    # 実験設定
    seeds = [42, 123]  # 2 seeds
    data_configs = [
        {
            "name": "8features",
            "path": "data/btc_jpy_1m_v451_optimized_features.parquet",
        },
        {
            "name": "full_features",
            "path": "data/btc_jpy_1m_v451.csv",
        },
    ]

    # 実験実行
    all_results = []
    for seed in seeds:
        for data_config in data_configs:
            experiment_name = f"{data_config['name']}_seed{seed}"
            config = create_experiment_config(
                data_path=data_config["path"],
                seed=seed,
                experiment_name=experiment_name,
            )

            result = run_single_experiment(config, output_dir)
            all_results.append(result)

            # 結果をJSON保存
            result_file = output_dir / f"{experiment_name}_{timestamp}.json"
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)

    # 統合分析
    analysis = analyze_results(all_results, output_dir)

    # 統合結果を保存
    summary_file = output_dir / f"ab_test_summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(
            {"results": all_results, "analysis": analysis}, f, indent=2
        )

    logger.info(f"\n✅ A/B検証完了")
    logger.info(f"結果: {summary_file}")


if __name__ == "__main__":
    main()
