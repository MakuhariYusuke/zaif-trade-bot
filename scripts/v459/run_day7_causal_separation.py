#!/usr/bin/env python3
"""
Phase 4 Day 7: 因果分離検証実験
66番計画: 0番整合性を確保した検証実験

実験構成（因果分離）:
  S1_default: 純PnL報酬 + SAC_DEFAULT → ベースライン
  S1_tuned: 純PnL報酬 + SAC_TUNED → SAC効果の分離
  S2_default: E報酬 + SAC_DEFAULT → 報酬効果の分離
  S2_tuned: E報酬 + SAC_TUNED → Day 6 E相当（再現）

追加メトリクス:
  - ROI (%)
  - Sharpe Ratio
  - Max Drawdown (%)
  - Win Rate (%)
  - Profit Factor
  - Trade Count

Usage:
    python scripts/v459/run_day7_causal_separation.py [--limit N] [--quick]
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import numpy as np

# Project root setup
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Environment setup
os.environ.setdefault("ZTB_SIGINT_POLICY", "ignore" if os.name == "nt" else "default")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("ZTB_SAFE_DATETIME", "1")
os.environ.setdefault("ZTB_SKIP_SCIPY", "1")
os.environ.setdefault("ZTB_SKIP_SKLEARN", "1")
os.environ.setdefault("SKIP_HEAVY_IMPORTS", "1")

from ztb.training.unified_trainer.trainer import UnifiedTrainer
from ztb.training.reward_config_schema import RewardConfigSchema

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 実験設定 ====================

# シード（0番 §5.6: n≥16要件に向けて4 seeds）
SEEDS = [42, 123, 456, 789]
SEEDS_QUICK = [42, 123]  # --quick用

DATA_PATH = str(project_root / "data" / "btc_jpy_1m_v451_optimized_features.parquet")
OUTPUT_DIR = str(project_root / "results" / "phase4_day7_causal_separation")

# SAC設定
SAC_DEFAULT = {
    "learning_rate": 0.0003,
    "buffer_size": 25000,
    "learning_starts": 500,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "target_update_interval": 1,
    "target_entropy": "auto"
}

SAC_TUNED = {
    "learning_rate": 0.0005,
    "buffer_size": 25000,
    "learning_starts": 500,
    "batch_size": 128,
    "tau": 0.005,
    "gamma": 0.95,
    "train_freq": 1,
    "gradient_steps": 2,
    "ent_coef": 0.01,  # 固定
    "target_update_interval": 1,
    "target_entropy": "auto"
}

# ==================== 報酬設定 ====================

# S1: 純PnL報酬（0番 Stage 1 準拠）
REWARD_S1_PURE_PNL = {
    "name": "pure_pnl_stage1",
    "description": "Pure PnL reward as per Doc00 Stage 1",
    "curriculum_stage": "simple",
    "use_simple_reward": True,  # 純PnL
    "reward_scale": 1.0,  # スケールなし
    "profit_weight": 1.0,
    "risk_weight": 0.0,  # リスクペナルティなし
    "consistency_weight": 0.0,
    "trading_bonus": 0.0,  # Holdボーナスなし
    "trade_frequency_penalty": 0.0,  # 取引ペナルティなし
    "reward_clip_min": -100.0,
    "reward_clip_max": 100.0,
}

# S2: E報酬（Day 6の最良設定）
REWARD_S2_EXPLORATION = "configs/rewards/stage1_exploration_tuned.yaml"

# 実験定義
EXPERIMENTS = [
    {
        "name": "S1_default",
        "description": "純PnL + SAC_DEFAULT（ベースライン）",
        "reward": REWARD_S1_PURE_PNL,
        "sac": SAC_DEFAULT,
    },
    {
        "name": "S1_tuned",
        "description": "純PnL + SAC_TUNED（SACの効果分離）",
        "reward": REWARD_S1_PURE_PNL,
        "sac": SAC_TUNED,
    },
    {
        "name": "S2_default",
        "description": "E報酬 + SAC_DEFAULT（報酬の効果分離）",
        "reward": REWARD_S2_EXPLORATION,
        "sac": SAC_DEFAULT,
    },
    {
        "name": "S2_tuned",
        "description": "E報酬 + SAC_TUNED（Day 6 E再現）",
        "reward": REWARD_S2_EXPLORATION,
        "sac": SAC_TUNED,
    },
]


# ==================== メトリクス計算 ====================

def calculate_extended_metrics(
    portfolio_values: List[float],
    trade_returns: List[float],
    total_trades: int,
    initial_value: float = 100000.0
) -> Dict[str, float]:
    """
    拡張メトリクスを計算（0番 §5.2 準拠）
    
    Args:
        portfolio_values: 各ステップのポートフォリオ価値
        trade_returns: 各取引のリターン
        total_trades: 総取引数
        initial_value: 初期ポートフォリオ価値
    
    Returns:
        0番基準のメトリクス辞書
    """
    if not portfolio_values:
        return {
            "net_roi_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
            "avg_pnl_per_trade": 0.0,
        }
    
    pv = np.array(portfolio_values)
    
    # Net ROI (%)
    final_value = pv[-1] if len(pv) > 0 else initial_value
    net_roi_pct = ((final_value - initial_value) / initial_value) * 100
    
    # Daily returns for Sharpe calculation
    returns = np.diff(pv) / pv[:-1] if len(pv) > 1 else np.array([0.0])
    returns = returns[~np.isnan(returns)]  # NaN除去
    
    # Sharpe Ratio (年率換算、1分足=525600分/年)
    if len(returns) > 0 and np.std(returns) > 1e-10:
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(525600)
    else:
        sharpe_ratio = 0.0
    
    # Max Drawdown (%)
    running_max = np.maximum.accumulate(pv)
    drawdowns = (running_max - pv) / running_max
    max_drawdown_pct = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0.0
    
    # Win Rate (%)
    if trade_returns:
        wins = sum(1 for r in trade_returns if r > 0)
        win_rate_pct = (wins / len(trade_returns)) * 100
    else:
        win_rate_pct = 0.0
    
    # Profit Factor
    if trade_returns:
        gross_profit = sum(r for r in trade_returns if r > 0)
        gross_loss = abs(sum(r for r in trade_returns if r < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    else:
        profit_factor = 0.0
    
    # Average PnL per trade
    avg_pnl_per_trade = np.mean(trade_returns) if trade_returns else 0.0
    
    return {
        "net_roi_pct": float(net_roi_pct),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown_pct": float(max_drawdown_pct),
        "win_rate_pct": float(win_rate_pct),
        "profit_factor": float(min(profit_factor, 999.99)),  # inf対策
        "total_trades": int(total_trades),
        "avg_pnl_per_trade": float(avg_pnl_per_trade),
    }


# ==================== 実験実行 ====================

def create_experiment_config(
    experiment_name: str,
    seed: int,
    reward_config: Dict[str, Any] | str,
    sac_params: Dict[str, Any],
    total_timesteps: int = 50000,
) -> Dict[str, Any]:
    """実験設定を作成"""
    config = {
        "training": {
            "algorithm": "SAC",
            "total_timesteps": total_timesteps,
            "eval_freq": 10000,
            "n_eval_episodes": 5,
            "log_interval": 500,
            "seed": seed,
            "sac_hyperparameters": sac_params,
            "data_config": {
                "data_path": DATA_PATH,
                "window_size": 60
            },
            "environment": {
                "use_continuous_actions": True,
                "action_space_type": "continuous",
                "initial_portfolio_value": 100000.0,
                "transaction_cost": 0.001,  # 0.1%
                "use_precomputed_features": True,
                "feature_set": "minimal"
            },
            "walk_forward": {
                "enabled": False
            }
        },
        "experiment_name": f"{experiment_name}_seed{seed}",
        "output_dir": OUTPUT_DIR
    }
    
    # 報酬設定を注入
    if isinstance(reward_config, str):
        # YAMLファイルパス
        reward_dict = RewardConfigSchema.load_and_validate(str(project_root / reward_config))
        behavior_opt = reward_dict.pop("behavior_optimization", None)
        config["training"]["environment"]["reward_settings"] = reward_dict
        if behavior_opt:
            config["training"]["environment"]["behavior_optimization"] = behavior_opt
    else:
        # 直接dict
        config["training"]["environment"]["reward_settings"] = reward_config
    
    return config


def run_single_experiment(
    exp_def: Dict[str, Any],
    seed: int,
    total_timesteps: int = 50000,
) -> Dict[str, Any]:
    """単一実験を実行し、拡張メトリクスを収集"""
    exp_name = exp_def["name"]
    exp_id = f"{exp_name}_seed{seed}"
    
    logger.info("=" * 80)
    logger.info(f"実験開始: {exp_id}")
    logger.info(f"説明: {exp_def['description']}")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # 設定作成
        config = create_experiment_config(
            exp_name,
            seed,
            exp_def["reward"],
            exp_def["sac"],
            total_timesteps
        )
        
        # Trainer作成・実行
        trainer = UnifiedTrainer(config)
        success = trainer.run()
        
        elapsed = time.time() - start_time
        
        if not success:
            raise ValueError("Training failed")
        
        # レポート取得
        report = trainer.get_training_report()
        
        # 基本メトリクス抽出
        training_stats = report.get("training_stats", {})
        action_dist = training_stats.get("action_distribution", {})
        
        # numpy型変換
        def to_python(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: to_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_python(item) for item in obj]
            return obj
        
        # 基本メトリクス
        basic_metrics = {
            "final_reward": to_python(training_stats.get("final_reward", 0)),
            "steps_per_second": to_python(training_stats.get("steps_per_second", 0)),
            "total_timesteps": to_python(training_stats.get("total_timesteps", 0)),
            "training_time": to_python(training_stats.get("training_time", 0)),
            "action_distribution": to_python(action_dist),
            "elapsed_seconds": elapsed,
        }
        
        # 拡張メトリクス（環境からポートフォリオ履歴を取得）
        # NOTE: UnifiedTrainerが履歴を保持していない場合はダミー値
        portfolio_history = training_stats.get("portfolio_history", [])
        trade_returns = training_stats.get("trade_returns", [])
        total_trades = training_stats.get("total_trades", 0)
        
        if not portfolio_history:
            # 履歴がない場合、Final Rewardから推定
            # 仮定: final_reward ≈ 正規化されたROI
            initial_value = 100000.0
            estimated_roi = basic_metrics["final_reward"] * 100  # 仮推定
            portfolio_history = [initial_value, initial_value * (1 + estimated_roi / 100)]
            logger.warning("ポートフォリオ履歴なし、Final Rewardから推定")
        
        extended_metrics = calculate_extended_metrics(
            portfolio_history,
            trade_returns,
            total_trades
        )
        
        # 全メトリクス統合
        all_metrics = {**basic_metrics, **extended_metrics}
        
        logger.info(f"実験完了: {exp_id} ({elapsed:.1f}秒)")
        logger.info(f"  Final Reward: {all_metrics['final_reward']:.6e}")
        logger.info(f"  Net ROI: {all_metrics['net_roi_pct']:.2f}%")
        logger.info(f"  Sharpe: {all_metrics['sharpe_ratio']:.4f}")
        logger.info(f"  MaxDD: {all_metrics['max_drawdown_pct']:.2f}%")
        logger.info(f"  Win Rate: {all_metrics['win_rate_pct']:.1f}%")
        
        return {
            "experiment_name": exp_id,
            "experiment_type": exp_name,
            "description": exp_def["description"],
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            "config": {
                "sac": exp_def["sac"],
                "reward_type": "pure_pnl" if isinstance(exp_def["reward"], dict) else "exploration_tuned",
            },
            "metrics": all_metrics,
            "report": to_python(report)
        }
        
    except Exception as e:
        import traceback
        logger.error(f"実験失敗: {exp_id} - {e}")
        logger.error(traceback.format_exc())
        
        return {
            "experiment_name": exp_id,
            "experiment_type": exp_name,
            "status": "failed",
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            "error": str(e)
        }


def run_all_experiments(
    seeds: List[int],
    limit: Optional[int] = None,
    total_timesteps: int = 50000,
) -> List[Dict[str, Any]]:
    """全実験を実行"""
    results = []
    
    total_experiments = len(EXPERIMENTS) * len(seeds)
    if limit:
        total_experiments = min(total_experiments, limit)
    
    logger.info("=" * 80)
    logger.info("Phase 4 Day 7: 因果分離検証実験")
    logger.info("=" * 80)
    logger.info(f"総実験数: {total_experiments}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"実験タイプ: {[e['name'] for e in EXPERIMENTS]}")
    logger.info(f"Total timesteps: {total_timesteps}")
    logger.info("=" * 80)
    
    experiment_count = 0
    
    for exp_def in EXPERIMENTS:
        for seed in seeds:
            if limit and experiment_count >= limit:
                logger.info(f"実験数制限に到達: {limit}")
                break
            
            result = run_single_experiment(exp_def, seed, total_timesteps)
            results.append(result)
            
            experiment_count += 1
            
            # 進捗保存（途中経過）
            save_results(results, "day7_causal_partial.json")
        
        if limit and experiment_count >= limit:
            break
    
    return results


def save_results(results: List[Dict[str, Any]], filename: str):
    """結果を保存"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = Path(OUTPUT_DIR) / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"結果保存: {output_path}")


def analyze_causal_separation(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """因果分離分析"""
    completed = [r for r in results if r["status"] == "completed"]
    
    if not completed:
        return {"error": "No completed experiments"}
    
    # タイプ別に集計
    by_type = {}
    for r in completed:
        exp_type = r["experiment_type"]
        if exp_type not in by_type:
            by_type[exp_type] = []
        by_type[exp_type].append(r["metrics"])
    
    # 平均計算
    summary = {}
    for exp_type, metrics_list in by_type.items():
        if not metrics_list:
            continue
        
        avg_metrics = {}
        for key in metrics_list[0].keys():
            if isinstance(metrics_list[0][key], (int, float)):
                values = [m[key] for m in metrics_list]
                avg_metrics[f"{key}_mean"] = np.mean(values)
                avg_metrics[f"{key}_std"] = np.std(values)
        
        summary[exp_type] = avg_metrics
    
    # 因果分離分析
    analysis = {
        "summary_by_type": summary,
        "causal_effects": {},
    }
    
    # SAC効果 = S1_tuned - S1_default
    if "S1_default" in summary and "S1_tuned" in summary:
        analysis["causal_effects"]["sac_effect"] = {
            "final_reward_delta": summary["S1_tuned"]["final_reward_mean"] - summary["S1_default"]["final_reward_mean"],
            "roi_delta": summary["S1_tuned"].get("net_roi_pct_mean", 0) - summary["S1_default"].get("net_roi_pct_mean", 0),
        }
    
    # 報酬効果 = S2_default - S1_default
    if "S1_default" in summary and "S2_default" in summary:
        analysis["causal_effects"]["reward_effect"] = {
            "final_reward_delta": summary["S2_default"]["final_reward_mean"] - summary["S1_default"]["final_reward_mean"],
            "roi_delta": summary["S2_default"].get("net_roi_pct_mean", 0) - summary["S1_default"].get("net_roi_pct_mean", 0),
        }
    
    # 交互作用 = S2_tuned - (S1_default + SAC効果 + 報酬効果)
    if all(k in summary for k in ["S1_default", "S1_tuned", "S2_default", "S2_tuned"]):
        expected = (
            summary["S1_default"]["final_reward_mean"]
            + analysis["causal_effects"]["sac_effect"]["final_reward_delta"]
            + analysis["causal_effects"]["reward_effect"]["final_reward_delta"]
        )
        actual = summary["S2_tuned"]["final_reward_mean"]
        analysis["causal_effects"]["interaction"] = {
            "expected": expected,
            "actual": actual,
            "interaction_delta": actual - expected,
        }
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Day 7: 因果分離検証実験")
    parser.add_argument("--limit", type=int, help="実験数制限")
    parser.add_argument("--quick", action="store_true", help="クイックモード（2 seeds）")
    parser.add_argument("--steps", type=int, default=50000, help="Total timesteps")
    args = parser.parse_args()
    
    seeds = SEEDS_QUICK if args.quick else SEEDS
    
    # 実験実行
    results = run_all_experiments(
        seeds=seeds,
        limit=args.limit,
        total_timesteps=args.steps
    )
    
    # 結果保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, f"day7_causal_{timestamp}.json")
    
    # 因果分離分析
    analysis = analyze_causal_separation(results)
    save_results(analysis, f"day7_analysis_{timestamp}.json")
    
    # サマリー出力
    logger.info("=" * 80)
    logger.info("因果分離分析結果")
    logger.info("=" * 80)
    
    if "summary_by_type" in analysis:
        for exp_type, metrics in analysis["summary_by_type"].items():
            logger.info(f"\n{exp_type}:")
            logger.info(f"  Final Reward: {metrics.get('final_reward_mean', 0):.6e} ± {metrics.get('final_reward_std', 0):.6e}")
            logger.info(f"  Net ROI: {metrics.get('net_roi_pct_mean', 0):.2f}% ± {metrics.get('net_roi_pct_std', 0):.2f}%")
    
    if "causal_effects" in analysis:
        effects = analysis["causal_effects"]
        if "sac_effect" in effects:
            logger.info(f"\nSAC効果（純報酬下）: Final Reward Δ = {effects['sac_effect']['final_reward_delta']:.6e}")
        if "reward_effect" in effects:
            logger.info(f"報酬効果（デフォルトSAC下）: Final Reward Δ = {effects['reward_effect']['final_reward_delta']:.6e}")
        if "interaction" in effects:
            logger.info(f"交互作用: Δ = {effects['interaction']['interaction_delta']:.6e}")
    
    logger.info("=" * 80)
    logger.info("実験完了")


if __name__ == "__main__":
    main()
