#!/usr/bin/env python3
"""
Phase 4.5 P1: 基準モデル作成実験（96#改訂版）

89#に基づき、ペナルティなしのPnLのみ報酬で基準モデルを作成。
95#レビュー反映: use_simple_reward=True で複合報酬経路をバイパス。
96#計画: 4 seeds × 50K steps で統計的信頼性を確保。

実験:
- P1-1: 真のPnLのみ（use_simple_reward=True、ペナルティ/シェーピング全無効）
- P1-3: 現行設定（Day11再現・比較用）

判断基準:
- P1-1 > 0%: 取引自体は利益 → コスト/ペナルティ調整で改善可能
- P1-1 < 0%: 取引戦略自体が損失 → 学習設計根本見直し必要
"""

import gc
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer
from ztb.training.utils.env_metrics import (
    extract_trainer_env_metrics,
    compute_balance_roi,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ============================================================================
# 設定
# ============================================================================

DATA_PATH = str(project_root / "data" / "btc_jpy_1m_v451_optimized_features.parquet")
OUTPUT_DIR = project_root / "results" / "phase45_p1_baseline"

# SAC基本設定（Day11/45# Day5と同一）
SAC_DEFAULT = {
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
}

# P1-1: 真のPnLのみ（95#レビュー反映版）
# NOTE: use_simple_reward=True で複合報酬経路をバイパス
# 95#指摘: confidence_penalty, balance_shaping, entropy_shaping の暗黙動作を排除
REWARD_PARAMS_P1_1 = {
    # ★ 最重要: simple_reward経路を使用（複合経路の汚染を回避）
    "use_simple_reward": True,
    # ペナルティ関連を全て0に
    "balance_penalty": 0.0,
    "balance_penalty_tolerance": 1.0,  # 大きな許容で無効化
    "position_penalty_scale": 0.0,
    "position_penalty_exponent": 1.0,  # 指数1で線形化（ペナルティなしには影響しない）
    "inventory_penalty_scale": 0.0,
    "trade_frequency_penalty": 0.0,
    "trade_cooldown_penalty": 0.0,
    "consecutive_trade_penalty": 0.0,
    "hold_penalty_multiplier": 1.0,  # 98#修正: 0.0はHOLD報酬消去、1.0はPnL保持
    "volatility_penalty_scale": 0.0,
    "consistency_penalty": 0.0,
    "redundant_trade_penalty": 0.0,
    # ボーナス関連は1.0（デフォルト）に
    "profit_weight": 1.0,
    "reward_scale": 100.0,
    # 95#指摘: 暗黙動作する追加コンポーネントを明示的に無効化
    "confidence_penalty_factor": 0.0,
    "balance_shaping_enabled": False,
    "action_entropy_shaping_enabled": False,
    # 非対称報酬スケーリング無効化
    "long_position_reward_multiplier": 1.0,
    "short_position_reward_multiplier": 1.0,
    "long_position_penalty_multiplier": 1.0,
    "short_position_penalty_multiplier": 1.0,
}

# P1-3: 現行設定（デフォルト）
REWARD_PARAMS_P1_3 = {}  # デフォルト値使用

# 96#計画: 4 seeds × 50K steps（0番§5.6準拠、統計的信頼性確保）
SEEDS = [42, 123, 456, 789]
TOTAL_TIMESTEPS = 50000


def create_experiment_config(
    experiment_name: str,
    seed: int,
    reward_params: dict,
    total_timesteps: int = 50000,
) -> dict:
    """実験設定を生成（Day11ベース）
    
    NOTE: Gate 0検証のため、reward_paramsは以下2箇所に配置:
    1. config["reward"] - _extract_expected_reward_params で読み取り
    2. config["training"]["environment"]["reward_settings"] - 環境に伝播
    """
    config = {
        "experiment_name": experiment_name,
        "training": {
            "algorithm": "SAC",
            "total_timesteps": total_timesteps,
            "eval_freq": 5000,
            "n_eval_episodes": 3,
            "log_interval": 100,
            "seed": seed,
            "sac_hyperparameters": SAC_DEFAULT.copy(),
            "data_config": {
                "data_path": DATA_PATH,
                "window_size": 60,
            },
            "environment": {
                "use_continuous_actions": True,
                "action_space_type": "continuous",
                "initial_portfolio_value": 100000.0,
                "transaction_cost": 0.001,
                # Gate 0: reward_settings として環境に伝播
                "reward_settings": reward_params.copy() if reward_params else {},
            },
            "walk_forward": {
                "enabled": False,  # P1は簡略化のためwalk-forward無効
            },
        },
        # Gate 0: 検証用にもrewardブロック保持
        "reward": reward_params,
    }
    return config


def run_single_experiment(config: dict) -> Dict[str, Any]:
    """1回の実験を実行（Day11と同じロジック）"""
    start_time = time.time()
    experiment_name = config["experiment_name"]

    logger.warning(f"\n{'='*60}")
    logger.warning(f"🔬 実験開始: {experiment_name}")
    logger.warning(f"{'='*60}")
    logger.warning(f"  reward_params: {config.get('reward', {})}")
    logger.warning(f"  total_timesteps: {config['training']['total_timesteps']}")
    logger.warning(f"  seed: {config['training']['seed']}")

    trainer = None
    try:
        # SACTrainerを直接使用（Day11と同じ）
        trainer = SACTrainer(config=config, logger=logger)
        
        # トレーニング実行
        result = trainer.train()
        
        elapsed_time = time.time() - start_time

        # 結果を整形
        training_result = {
            "experiment_name": experiment_name,
            "seed": config["training"]["seed"],
            "total_time_seconds": elapsed_time,
            "reward_params": config.get("reward", {}),
            "total_timesteps": config["training"]["total_timesteps"],
            "success": True,
            "timestamp": datetime.now().isoformat(),
        }

        # トレーニング結果を統合
        if result:
            if isinstance(result, dict):
                training_result.update(result)
            elif hasattr(result, "__dict__"):
                training_result.update(vars(result))
        
        # 環境からメトリクスを抽出（Day11と同じ）
        try:
            metrics = extract_trainer_env_metrics(trainer, include_optional=True)
            if metrics:
                if "final_balance" in metrics:
                    logger.info(f"  ✓ Got balance: {metrics['final_balance']:.2f}")

                if "final_balance" in metrics and "initial_balance" not in metrics:
                    metrics["initial_balance"] = 100000.0

                # 全メトリクスを統合
                for key, value in metrics.items():
                    if key not in training_result:
                        training_result[key] = value

                # ROI計算
                roi = compute_balance_roi(metrics)
                if roi is not None:
                    training_result["final_roi"] = roi
                    training_result["net_roi"] = roi
                    training_result["balance_roi"] = roi

                # Gross/Net ROI計算
                if 'gross_pnl' in metrics and metrics.get('initial_balance', 0) > 0:
                    gross_roi = (metrics['gross_pnl'] / metrics['initial_balance']) * 100
                    training_result["gross_roi"] = gross_roi
                    logger.info(f"  Gross ROI: {gross_roi:+.2f}%")

                logger.info("環境から取得したメトリクス:")
                logger.info(f"  final_balance: {training_result.get('final_balance', 'N/A')}")
                logger.info(f"  ROI: {training_result.get('final_roi', 'N/A')}")
                logger.info(f"  total_trades: {training_result.get('total_trades', 'N/A')}")
                logger.info(f"  gross_pnl: {training_result.get('gross_pnl', 'N/A')}")
                logger.info(f"  net_pnl: {training_result.get('net_pnl', 'N/A')}")
                logger.info(f"  total_fees: {training_result.get('total_fees', 'N/A')}")
            else:
                logger.warning("環境へのアクセスに失敗しました")
        except Exception as env_error:
            logger.error(f"環境メトリクス取得エラー: {env_error}", exc_info=True)

        logger.warning(f"✅ 実験完了: {experiment_name} ({elapsed_time:.1f}秒)")
        logger.warning(f"   ROI: {training_result.get('final_roi', training_result.get('balance_roi', 'N/A'))}")
        logger.warning(f"   Final Balance: {training_result.get('final_balance', 'N/A')}")
        logger.warning(f"   取引回数: {training_result.get('total_trades', 'N/A')}")
        return training_result

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"❌ 実験失敗: {experiment_name} - {e}", exc_info=True)
        return {
            "experiment_name": experiment_name,
            "seed": config["training"]["seed"],
            "total_time_seconds": elapsed_time,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
    finally:
        # メモリリーク防止: 既存cleanup + 明示的解放
        if trainer is not None:
            try:
                trainer.cleanup_training_environment()
            except Exception:
                pass
            del trainer
        gc.collect()


def main():
    """P1実験を実行"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.warning("=" * 70)
    logger.warning("Phase 4.5 P1: 基準モデル作成実験")
    logger.warning(f"出力ディレクトリ: {OUTPUT_DIR}")
    logger.warning("=" * 70)
    
    all_results = []
    
    # P1-1: PnLのみ（ペナルティ全無効）
    logger.warning("\n" + "="*60)
    logger.warning("P1-1: PnLのみ（ペナルティ全無効）")
    logger.warning("="*60)
    
    for seed in SEEDS:
        config = create_experiment_config(
            experiment_name=f"P1-1_pnl_only_seed{seed}",
            seed=seed,
            reward_params=REWARD_PARAMS_P1_1,
            total_timesteps=TOTAL_TIMESTEPS,
        )
        result = run_single_experiment(config)
        result["experiment_category"] = "P1-1"
        all_results.append(result)
        gc.collect()  # seed間でメモリ解放
    
    # P1-3: 現行設定（比較用）
    logger.warning("\n" + "="*60)
    logger.warning("P1-3: 現行設定（Day11再現・比較用）")
    logger.warning("="*60)
    
    for seed in SEEDS:
        config = create_experiment_config(
            experiment_name=f"P1-3_default_seed{seed}",
            seed=seed,
            reward_params=REWARD_PARAMS_P1_3,
            total_timesteps=TOTAL_TIMESTEPS,
        )
        result = run_single_experiment(config)
        result["experiment_category"] = "P1-3"
        all_results.append(result)
        gc.collect()  # seed間でメモリ解放
    
    # 結果分析
    logger.warning("\n" + "="*70)
    logger.warning("📊 P1 RESULTS SUMMARY")
    logger.warning("="*70)
    
    # カテゴリ別集計
    p1_1_results = [r for r in all_results if r.get("experiment_category") == "P1-1"]
    p1_3_results = [r for r in all_results if r.get("experiment_category") == "P1-3"]
    
    def summarize(results: list, name: str) -> dict:
        rois = [r.get("balance_roi", r.get("final_roi")) for r in results if r.get("success")]
        rois = [r for r in rois if r is not None]
        balances = [r.get("final_balance") for r in results if r.get("success") and r.get("final_balance")]
        trades = [r.get("total_trades") for r in results if r.get("success") and r.get("total_trades")]
        gross_pnls = [r.get("gross_pnl") for r in results if r.get("success") and r.get("gross_pnl") is not None]
        total_fees_list = [r.get("total_fees") for r in results if r.get("success") and r.get("total_fees") is not None]
        
        summary = {
            "name": name,
            "count": len(results),
            "success": len([r for r in results if r.get("success")]),
            "roi_mean": np.mean(rois) if rois else None,
            "roi_std": np.std(rois) if len(rois) > 1 else 0.0,
            "balance_mean": np.mean(balances) if balances else None,
            "trades_mean": np.mean(trades) if trades else None,
            "gross_pnl_mean": np.mean(gross_pnls) if gross_pnls else None,
            "total_fees_mean": np.mean(total_fees_list) if total_fees_list else None,
        }
        
        logger.warning(f"\n{name}:")
        logger.warning(f"  成功: {summary['success']}/{summary['count']}")
        if summary["roi_mean"] is not None:
            logger.warning(f"  ROI: {summary['roi_mean']:.2f}% ± {summary['roi_std']:.2f}%")
        if summary["balance_mean"] is not None:
            logger.warning(f"  Balance: {summary['balance_mean']:,.0f}")
        if summary["trades_mean"] is not None:
            logger.warning(f"  Trades: {summary['trades_mean']:.0f}")
        if summary["gross_pnl_mean"] is not None:
            logger.warning(f"  Gross PnL: {summary['gross_pnl_mean']:+,.0f}")
        if summary["total_fees_mean"] is not None:
            logger.warning(f"  Total Fees: {summary['total_fees_mean']:,.0f}")
        
        return summary
    
    p1_1_summary = summarize(p1_1_results, "P1-1 (PnLのみ)")
    p1_3_summary = summarize(p1_3_results, "P1-3 (現行設定)")
    
    # 判断基準
    logger.warning("\n" + "-"*60)
    logger.warning("INTERPRETATION:")
    
    if p1_1_summary["roi_mean"] is not None:
        if p1_1_summary["roi_mean"] > 0:
            logger.warning("✅ P1-1 (PnLのみ) > 0%: 取引自体は利益")
            logger.warning("   → コスト/ペナルティ調整で改善可能")
        elif p1_1_summary["roi_mean"] > -3:
            logger.warning("⚠️ P1-1 (PnLのみ) ≈ 0%: 取引自体は損益なし")
            logger.warning("   → コスト削減で黒字化可能性あり")
        else:
            logger.warning("❌ P1-1 (PnLのみ) < -3%: 取引戦略自体が損失")
            logger.warning("   → 学習設計根本見直し必要")
    
    if p1_1_summary["roi_mean"] is not None and p1_3_summary["roi_mean"] is not None:
        diff = p1_1_summary["roi_mean"] - p1_3_summary["roi_mean"]
        logger.warning(f"\nP1-1 vs P1-3 差分: {diff:+.2f}%")
        if diff > 0:
            logger.warning("   → ペナルティ除去で改善 = ペナルティが過剰")
        else:
            logger.warning("   → ペナルティ除去で悪化 = ペナルティは有効")
    
    # 結果保存
    results_file = OUTPUT_DIR / f"p1_results_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "all_results": all_results,
            "summary": {
                "P1-1": p1_1_summary,
                "P1-3": p1_3_summary,
            }
        }, f, indent=2, ensure_ascii=False)
    
    logger.warning(f"\n✅ Results saved: {results_file}")
    logger.warning("="*70 + "\n")


def run_single(category: str, seed: int):
    """サブプロセスから呼ばれる単一実験モード。結果JSONをstdoutに出力。"""
    reward_params = REWARD_PARAMS_P1_1 if category == "P1-1" else REWARD_PARAMS_P1_3
    config = create_experiment_config(
        experiment_name=f"{category}_seed{seed}",
        seed=seed,
        reward_params=reward_params,
        total_timesteps=TOTAL_TIMESTEPS,
    )
    result = run_single_experiment(config)
    result["experiment_category"] = category
    # 結果JSONを stdout の最終行に出力（subprocess runner が読み取る）
    import json as _json
    print(_json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-run", action="store_true", help="単一実験モード")
    parser.add_argument("--category", type=str, default="P1-1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.single_run:
        run_single(args.category, args.seed)
    else:
        main()
