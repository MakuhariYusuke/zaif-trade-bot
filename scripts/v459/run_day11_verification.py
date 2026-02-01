#!/usr/bin/env python3
"""
Day 11: 評価基盤修正後の再実験

84# 対応:
1. 45# Day5設定の再現（SAC_DEFAULT, 50k, walk-forward有効）
2. 修正版環境アクセスでの検証
3. A/B: walk-forward有効 vs 無効の比較

目的:
- ROI=-5%の再現可能性確認
- walk-forward有効/無効の影響測定
- final_balance取得の検証
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer
from ztb.utils.logging_utils import get_logger

logger = get_logger("day11_verification")

# ============================================================================
# Constants
# ============================================================================

DATA_PATH = str(project_root / "data" / "btc_jpy_1m_v451_optimized_features.parquet")
OUTPUT_DIR = project_root / "results" / "phase4_day11_verification"

# 45# Day5: SAC_DEFAULT設定（ROI=-5%を達成）
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

SEEDS = [42, 123]


def create_experiment_config(
    experiment_name: str,
    seed: int,
    total_timesteps: int = 50000,
    walk_forward_enabled: bool = True,
) -> dict:
    """実験設定を生成（45# Day5ベース）"""
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
            },
            "walk_forward": {
                "enabled": walk_forward_enabled,
                "n_splits": 4,
                "train_size": 0.6,
                "validation_size": 0.2,
                "test_size": 0.2,
            } if walk_forward_enabled else {
                "enabled": False,
            },
        },
    }
    return config


def run_single_experiment(config: dict) -> Dict[str, Any]:
    """1回の実験を実行（45# run_ab_feature_test.pyベース）"""
    start_time = time.time()
    experiment_name = config["experiment_name"]

    logger.warning(f"\n{'='*60}")
    logger.warning(f"🔬 実験開始: {experiment_name}")
    logger.warning(f"{'='*60}")
    logger.warning(f"  walk_forward: {config['training']['walk_forward']['enabled']}")
    logger.warning(f"  total_timesteps: {config['training']['total_timesteps']}")
    logger.warning(f"  seed: {config['training']['seed']}")

    try:
        # SACTrainerを直接使用（45# Day5と同じ）
        trainer = SACTrainer(config=config, logger=logger)
        
        # トレーニング実行
        result = trainer.train()
        
        elapsed_time = time.time() - start_time

        # 結果を整形
        training_result = {
            "experiment_name": experiment_name,
            "seed": config["training"]["seed"],
            "total_time_seconds": elapsed_time,
            "walk_forward_enabled": config["training"]["walk_forward"]["enabled"],
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
        
        # 環境からメトリクスを抽出（84#修正版）
        try:
            env = None
            # SACTrainerはtrainer.model.envでアクセス可能
            if hasattr(trainer, 'model') and trainer.model is not None:
                if hasattr(trainer.model, 'env') and trainer.model.env is not None:
                    env = trainer.model.env
                elif hasattr(trainer.model, 'get_env'):
                    env = trainer.model.get_env()
            
            if env is not None:
                # VecEnvをunwrap
                actual_env = env
                if hasattr(env, 'envs') and len(env.envs) > 0:
                    actual_env = env.envs[0]
                
                # さらにMonitor等をunwrap
                unwrapped_env = actual_env
                max_unwrap = 10
                for _ in range(max_unwrap):
                    if hasattr(unwrapped_env, 'env'):
                        unwrapped_env = unwrapped_env.env
                    else:
                        break
                
                # 84# Fix: Check for 'balance' first
                if hasattr(unwrapped_env, 'balance'):
                    training_result["final_balance"] = float(unwrapped_env.balance)
                    logger.info(f"  ✓ Got balance: {training_result['final_balance']:.2f}")
                elif hasattr(unwrapped_env, 'portfolio_value'):
                    training_result["final_balance"] = float(unwrapped_env.portfolio_value)
                    logger.info(f"  ✓ Got portfolio_value: {training_result['final_balance']:.2f}")
                
                # 84# Fix: Check for 'initial_balance' first
                initial_balance = 100000.0
                if hasattr(unwrapped_env, 'initial_balance'):
                    initial_balance = float(unwrapped_env.initial_balance)
                elif hasattr(unwrapped_env, 'initial_portfolio_value'):
                    initial_balance = float(unwrapped_env.initial_portfolio_value)
                training_result["initial_balance"] = initial_balance
                
                # ROI計算
                if "final_balance" in training_result:
                    roi = (training_result["final_balance"] - initial_balance) / initial_balance * 100
                    training_result["final_roi"] = roi
                    training_result["net_roi"] = roi
                
                # 取引回数
                if hasattr(unwrapped_env, 'total_trades'):
                    training_result["total_trades"] = int(unwrapped_env.total_trades)
                if hasattr(unwrapped_env, 'buy_count'):
                    training_result["buy_count"] = int(unwrapped_env.buy_count)
                if hasattr(unwrapped_env, 'sell_count'):
                    training_result["sell_count"] = int(unwrapped_env.sell_count)
                
                # Sharpe ratio
                if hasattr(unwrapped_env, 'get_sharpe_ratio'):
                    training_result["sharpe_ratio"] = float(unwrapped_env.get_sharpe_ratio())
                elif hasattr(unwrapped_env, 'sharpe_ratio'):
                    training_result["sharpe_ratio"] = float(unwrapped_env.sharpe_ratio)
                
                # 84# Fix: Log effective reward settings
                if hasattr(unwrapped_env, 'reward_scale'):
                    training_result["effective_reward_scale"] = float(unwrapped_env.reward_scale)
                    logger.info(f"  Effective reward_scale: {unwrapped_env.reward_scale}")
                    
                logger.info(f"環境から取得したメトリクス:")
                logger.info(f"  final_balance: {training_result.get('final_balance', 'N/A')}")
                logger.info(f"  ROI: {training_result.get('final_roi', 'N/A')}")
                logger.info(f"  total_trades: {training_result.get('total_trades', 'N/A')}")
            else:
                logger.warning("環境へのアクセスに失敗しました")
        except Exception as env_error:
            logger.error(f"環境メトリクス取得エラー: {env_error}", exc_info=True)
        
        # trainer属性からも取得（フォールバック）
        if hasattr(trainer, "final_balance") and "final_balance" not in training_result:
            training_result["final_balance"] = trainer.final_balance
        if hasattr(trainer, "final_roi") and "final_roi" not in training_result:
            training_result["final_roi"] = trainer.final_roi

        logger.warning(f"✅ 実験完了: {experiment_name} ({elapsed_time:.1f}秒)")
        logger.warning(f"   ROI: {training_result.get('final_roi', training_result.get('net_roi', 'N/A'))}")
        logger.warning(f"   Final Balance: {training_result.get('final_balance', 'N/A')}")
        logger.warning(f"   Sharpe: {training_result.get('sharpe_ratio', 'N/A')}")
        logger.warning(f"   取引回数: {training_result.get('total_trades', 'N/A')}")
        return training_result

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"❌ 実験失敗: {experiment_name} - {e}", exc_info=True)
        return {
            "experiment_name": experiment_name,
            "seed": config["training"]["seed"],
            "total_time_seconds": elapsed_time,
            "walk_forward_enabled": config["training"]["walk_forward"]["enabled"],
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def analyze_results(results: List[Dict], category: str) -> Dict[str, Any]:
    """カテゴリ別結果分析"""
    successful = [r for r in results if r.get("success", False)]
    
    if not successful:
        return {"n_experiments": 0, "all_failed": True}
    
    rois = []
    balances = []
    trades = []
    
    for r in successful:
        roi = r.get("final_roi") or r.get("net_roi")
        if roi is not None:
            rois.append(float(roi))
        
        balance = r.get("final_balance")
        if balance is not None:
            balances.append(float(balance))
        
        trade_count = r.get("total_trades")
        if trade_count is not None:
            trades.append(int(trade_count))
    
    return {
        "category": category,
        "n_experiments": len(successful),
        "n_failed": len(results) - len(successful),
        "roi_mean": float(np.mean(rois)) if rois else None,
        "roi_std": float(np.std(rois, ddof=1)) if len(rois) > 1 else 0.0,
        "roi_min": float(np.min(rois)) if rois else None,
        "roi_max": float(np.max(rois)) if rois else None,
        "balance_mean": float(np.mean(balances)) if balances else None,
        "balance_std": float(np.std(balances, ddof=1)) if len(balances) > 1 else 0.0,
        "trades_mean": float(np.mean(trades)) if trades else None,
        "balance_available": len(balances) > 0,
        "roi_source": "balance" if balances else "unknown",
    }


def main():
    """メイン実行"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.warning("\n" + "="*80)
    logger.warning("🧪 DAY 11: 評価基盤修正後の再実験")
    logger.warning("="*80)
    logger.warning("目的:")
    logger.warning("  1. 45# Day5設定の再現（ROI=-5%の再現可能性確認）")
    logger.warning("  2. walk-forward有効/無効の影響測定")
    logger.warning("  3. final_balance取得の検証")
    logger.warning(f"出力先: {OUTPUT_DIR}")
    logger.warning("="*80 + "\n")
    
    all_results = []
    interim_path = OUTPUT_DIR / "day11_verification_interim.json"
    
    # =========================================================================
    # Category A: 45# Day5完全再現（walk-forward有効, 50k）
    # =========================================================================
    logger.warning("\n" + "="*60)
    logger.warning("📊 Category A: 45# Day5完全再現 (walk-forward有効)")
    logger.warning("="*60)
    
    category_a_results = []
    for seed in SEEDS:
        exp_name = f"A_day5_wf_enabled_seed{seed}"
        config = create_experiment_config(
            experiment_name=exp_name,
            seed=seed,
            total_timesteps=50000,
            walk_forward_enabled=True,
        )
        result = run_single_experiment(config)
        result["category"] = "A_wf_enabled"
        category_a_results.append(result)
        all_results.append(result)
        
        # 中間保存
        with open(interim_path, "w") as f:
            json.dump(all_results, f, indent=2)
    
    # =========================================================================
    # Category B: walk-forward無効（Day10と同条件）
    # =========================================================================
    logger.warning("\n" + "="*60)
    logger.warning("📊 Category B: walk-forward無効 (Day10条件)")
    logger.warning("="*60)
    
    category_b_results = []
    for seed in SEEDS:
        exp_name = f"B_day5_wf_disabled_seed{seed}"
        config = create_experiment_config(
            experiment_name=exp_name,
            seed=seed,
            total_timesteps=50000,
            walk_forward_enabled=False,
        )
        result = run_single_experiment(config)
        result["category"] = "B_wf_disabled"
        category_b_results.append(result)
        all_results.append(result)
        
        # 中間保存
        with open(interim_path, "w") as f:
            json.dump(all_results, f, indent=2)
    
    # =========================================================================
    # Category C: 25kステップ比較（崩壊前の安定性確認）
    # =========================================================================
    logger.warning("\n" + "="*60)
    logger.warning("📊 Category C: 25kステップ (崩壊前確認)")
    logger.warning("="*60)
    
    category_c_results = []
    for seed in SEEDS:
        exp_name = f"C_25k_wf_disabled_seed{seed}"
        config = create_experiment_config(
            experiment_name=exp_name,
            seed=seed,
            total_timesteps=25000,
            walk_forward_enabled=False,
        )
        result = run_single_experiment(config)
        result["category"] = "C_25k"
        category_c_results.append(result)
        all_results.append(result)
        
        # 中間保存
        with open(interim_path, "w") as f:
            json.dump(all_results, f, indent=2)
    
    # =========================================================================
    # 分析
    # =========================================================================
    analysis = {
        "A_wf_enabled": analyze_results(category_a_results, "A_wf_enabled"),
        "B_wf_disabled": analyze_results(category_b_results, "B_wf_disabled"),
        "C_25k": analyze_results(category_c_results, "C_25k"),
    }
    
    # 解釈
    interpretation = []
    
    # A vs B: walk-forward影響
    if analysis["A_wf_enabled"]["roi_mean"] is not None and analysis["B_wf_disabled"]["roi_mean"] is not None:
        wf_diff = analysis["A_wf_enabled"]["roi_mean"] - analysis["B_wf_disabled"]["roi_mean"]
        interpretation.append(f"walk-forward影響: {wf_diff:+.2f}% (有効 - 無効)")
        if wf_diff > 5:
            interpretation.append("→ walk-forward有効が大幅に優位")
        elif wf_diff < -5:
            interpretation.append("→ walk-forward無効が優位（意外な結果）")
        else:
            interpretation.append("→ walk-forwardの影響は小さい")
    
    # A vs 45# Day5比較
    if analysis["A_wf_enabled"]["roi_mean"] is not None:
        day5_diff = analysis["A_wf_enabled"]["roi_mean"] - (-5.07)
        interpretation.append(f"45# Day5との差: {day5_diff:+.2f}% (今回 vs -5.07%)")
        if abs(day5_diff) < 3:
            interpretation.append("→ ✅ 45# Day5を再現成功")
        else:
            interpretation.append("→ ⚠️ 45# Day5との乖離あり")
    
    # B vs Day10比較
    if analysis["B_wf_disabled"]["roi_mean"] is not None:
        day10_diff = analysis["B_wf_disabled"]["roi_mean"] - (-36.04)
        interpretation.append(f"Day10(A1)との差: {day10_diff:+.2f}% (今回 vs -36.04%)")
    
    # balance取得成功率
    balance_ok = sum(1 for r in all_results if r.get("final_balance") is not None)
    interpretation.append(f"final_balance取得成功: {balance_ok}/{len(all_results)}")
    
    # 最終分析を保存
    final_analysis = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(all_results),
            "completed": sum(1 for r in all_results if r.get("success", False)),
            "balance_available": balance_ok,
        },
        "category_results": analysis,
        "interpretation": interpretation,
        "reference": {
            "45_day5_roi": -5.07,
            "day10_a1_roi": -36.04,
            "day10_c1_roi": -5.71,
        }
    }
    
    analysis_path = OUTPUT_DIR / f"day11_verification_analysis_{timestamp}.json"
    with open(analysis_path, "w") as f:
        json.dump(final_analysis, f, indent=2)
    
    results_path = OUTPUT_DIR / f"day11_verification_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # サマリー出力
    logger.warning("\n" + "="*80)
    logger.warning("📊 DAY 11 VERIFICATION RESULTS SUMMARY")
    logger.warning("="*80)
    
    for cat_name, cat_stats in analysis.items():
        if cat_stats.get("roi_mean") is not None:
            logger.warning(f"\n{cat_name}:")
            logger.warning(f"  ROI: {cat_stats['roi_mean']:.2f}% ± {cat_stats['roi_std']:.2f}%")
            logger.warning(f"  Balance: {cat_stats.get('balance_mean', 'N/A')}")
            logger.warning(f"  Trades: {cat_stats.get('trades_mean', 'N/A')}")
            logger.warning(f"  Source: {cat_stats.get('roi_source', 'N/A')}")
    
    logger.warning("\n" + "-"*40)
    logger.warning("INTERPRETATION:")
    for line in interpretation:
        logger.warning(f"  {line}")
    
    logger.warning(f"\n✅ Analysis saved: {analysis_path}")
    logger.warning(f"✅ Results saved: {results_path}")
    logger.warning("="*80 + "\n")


if __name__ == "__main__":
    main()
