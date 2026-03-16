#!/usr/bin/env python3
"""
Day 8: スケール交絡除去実験（68番レビュー対応）

68番レビューの指摘事項：
1. reward_scale差（1.0 vs 100.0）が巨大な交絡因子
2. SAC_TUNED単独の因果効果は未確定
3. 2 seedsは不十分

実験設計：
- スケールを100.0に統一し、報酬設計の純粋効果を分離
- 4 seeds（42, 123, 456, 789）で統計的信頼性向上
- 0番 §5.2準拠のメトリクス: ROI/Sharpe/MaxDD/WinRate/ProfitFactor

Usage:
    python scripts/v459/run_day8_scale_deconfounding.py
    python scripts/v459/run_day8_scale_deconfounding.py --quick  # 2 seeds, 25k steps
    python scripts/v459/run_day8_scale_deconfounding.py --phase A  # 核心実験のみ
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 環境変数設定（最優先）
os.environ.setdefault("ZTB_SIGINT_POLICY", "ignore" if os.name == "nt" else "default")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("ZTB_SAFE_DATETIME", "1")
os.environ.setdefault("ZTB_SKIP_SCIPY", "1")
os.environ.setdefault("ZTB_SKIP_SKLEARN", "1")
os.environ.setdefault("SKIP_HEAVY_IMPORTS", "1")

import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.unified_trainer.trainer import UnifiedTrainer
from ztb.metrics import calculate_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# パス設定
# =============================================================================
DATA_PATH = str(project_root / "data" / "btc_jpy_1m_v451_optimized_features.parquet")
OUTPUT_DIR = str(project_root / "results" / "phase4_day8_scale_deconfounding")

# =============================================================================
# SAC設定（Day 7と同一）
# =============================================================================
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

# =============================================================================
# 報酬設定（スケール100.0統一）
# =============================================================================

# S1': 純PnL + scale=100（Day 7のS1と比較用）
# 目的: スケール交絡を除去し、SAC設定の純粋効果を測定
REWARD_S1_SCALED = {
    "name": "pure_pnl_scaled",
    "description": "Pure PnL with scale=100 (deconfounding)",
    "curriculum_stage": "simple",
    "use_simple_reward": True,
    "reward_scale": 100.0,  # Day 7は1.0 → 100.0に統一
    "reward_scaling": 100.0,  # 両方設定（実装依存対策）
    "profit_weight": 1.0,
    "risk_weight": 0.0,
    "consistency_weight": 0.0,
    "trading_bonus": 0.0,
    "trade_frequency_penalty": 0.0,
    "action_smoothing": 0.0,
    "reward_clip_min": -100.0,
    "reward_clip_max": 100.0,
}

# S2: E報酬（Day 7と同一）
# 目的: Day 7結果との再現性確認
REWARD_S2_E = {
    "name": "exploration_reward",
    "description": "E reward (Day 6 winner)",
    "curriculum_stage": "exploration",
    "use_simple_reward": False,
    "reward_scale": 100.0,
    "reward_scaling": 100.0,
    "profit_weight": 1.0,
    "risk_weight": 0.0,
    "consistency_weight": 0.0,
    "trading_bonus": 0.0,
    "trade_frequency_penalty": 0.01,
    "action_smoothing": 0.01,
    "reward_clip_min": -1.0,
    "reward_clip_max": 1.0,
}

# S3: 純PnL + scale=100 + E報酬のclip（クリップ効果分離）
# 目的: reward_clip [-1,1] の効果を分離測定
REWARD_S3_SCALED_CLIPPED = {
    "name": "pure_pnl_scaled_clipped",
    "description": "Pure PnL with scale=100 and E-style clip",
    "curriculum_stage": "simple",
    "use_simple_reward": True,
    "reward_scale": 100.0,
    "reward_scaling": 100.0,
    "profit_weight": 1.0,
    "risk_weight": 0.0,
    "consistency_weight": 0.0,
    "trading_bonus": 0.0,
    "trade_frequency_penalty": 0.0,
    "action_smoothing": 0.0,
    "reward_clip_min": -1.0,  # E報酬と同じclip
    "reward_clip_max": 1.0,
}

# S4: E報酬 - ペナルティなし（ペナルティ効果分離）
# 目的: trade_frequency_penalty/action_smoothing の効果を分離測定
REWARD_S4_E_NO_PENALTY = {
    "name": "exploration_no_penalty",
    "description": "E reward without penalties (ablation)",
    "curriculum_stage": "exploration",
    "use_simple_reward": False,
    "reward_scale": 100.0,
    "reward_scaling": 100.0,
    "profit_weight": 1.0,
    "risk_weight": 0.0,
    "consistency_weight": 0.0,
    "trading_bonus": 0.0,
    "trade_frequency_penalty": 0.0,  # ペナルティ除去
    "action_smoothing": 0.0,  # スムージング除去
    "reward_clip_min": -1.0,
    "reward_clip_max": 1.0,
}

# =============================================================================
# 実験定義
# =============================================================================
EXPERIMENTS = [
    # Phase A: スケール交絡除去（核心実験）
    {
        "name": "S1_scaled_default",
        "description": "純PnL(scale=100) + SAC_DEFAULT: スケール統一ベースライン",
        "reward": REWARD_S1_SCALED,
        "sac": SAC_DEFAULT,
        "phase": "A",
    },
    {
        "name": "S1_scaled_tuned",
        "description": "純PnL(scale=100) + SAC_TUNED: 真のSAC効果測定",
        "reward": REWARD_S1_SCALED,
        "sac": SAC_TUNED,
        "phase": "A",
    },
    
    # Phase B: E報酬再現（Day 7との整合性確認）
    {
        "name": "S2_E_default",
        "description": "E報酬 + SAC_DEFAULT: 報酬効果ベースライン",
        "reward": REWARD_S2_E,
        "sac": SAC_DEFAULT,
        "phase": "B",
    },
    {
        "name": "S2_E_tuned",
        "description": "E報酬 + SAC_TUNED: Day 6/7 E設定再現",
        "reward": REWARD_S2_E,
        "sac": SAC_TUNED,
        "phase": "B",
    },
    
    # Phase C: 因子分離（Ablation）
    {
        "name": "S3_clipped_default",
        "description": "純PnL(scale=100,clip=[-1,1]) + SAC_DEFAULT: クリップ効果",
        "reward": REWARD_S3_SCALED_CLIPPED,
        "sac": SAC_DEFAULT,
        "phase": "C",
    },
    {
        "name": "S3_clipped_tuned",
        "description": "純PnL(scale=100,clip=[-1,1]) + SAC_TUNED: クリップ+SAC",
        "reward": REWARD_S3_SCALED_CLIPPED,
        "sac": SAC_TUNED,
        "phase": "C",
    },
    {
        "name": "S4_nopen_default",
        "description": "E報酬(ペナルティなし) + SAC_DEFAULT: ペナルティ効果",
        "reward": REWARD_S4_E_NO_PENALTY,
        "sac": SAC_DEFAULT,
        "phase": "C",
    },
    {
        "name": "S4_nopen_tuned",
        "description": "E報酬(ペナルティなし) + SAC_TUNED: ペナルティ+SAC",
        "reward": REWARD_S4_E_NO_PENALTY,
        "sac": SAC_TUNED,
        "phase": "C",
    },
]

# シード設定
SEEDS_FULL = [42, 123, 456, 789]
SEEDS_QUICK = [42, 123]

# 1分足の年間期間数（365日 × 24時間 × 60分）
MINUTES_PER_YEAR = 525600


# =============================================================================
# メトリクス計算（ztb.metrics使用、0番 §5.2 準拠）
# =============================================================================

def calculate_extended_metrics(
    portfolio_values: List[float],
    trade_returns: List[float],
    total_trades: int,
    initial_value: float = 100000.0
) -> Dict[str, float]:
    """
    拡張メトリクスを計算（ztb.metrics使用、0番 §5.2 準拠）
    
    Returns:
        net_roi_pct: 純ROI (%)
        sharpe_ratio: シャープレシオ（年率換算）
        max_drawdown_pct: 最大ドローダウン (%)
        win_rate_pct: 勝率 (%)
        profit_factor: プロフィットファクター
        total_trades: 総取引数
        avg_pnl_per_trade: 1取引あたり平均損益
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
    
    # Step returns for metrics calculation
    returns = np.diff(pv) / pv[:-1] if len(pv) > 1 else np.array([0.0])
    returns = returns[np.isfinite(returns)]  # NaN/Inf除去
    
    # ztb.metricsを使用（1分足 = 525600期間/年）
    all_metrics = calculate_all_metrics(returns, rf=0.0, period_per_year=MINUTES_PER_YEAR)
    
    # Max Drawdown (ztb.metricsは負値で返すので絶対値×100)
    mdd_pct = abs(all_metrics["max_drawdown"]) * 100
    
    # Win Rate（ztb.metricsは0-1で返すので×100）
    win_rate_pct = all_metrics["win_rate"] * 100
    
    # Profit Factor
    pf = all_metrics["profit_factor"]
    # inf対策
    pf = min(pf, 999.99) if pf != float('inf') else 999.99
    
    # Average PnL per trade（trade_returnsから計算）
    avg_pnl_per_trade = np.mean(trade_returns) if trade_returns else 0.0
    
    return {
        "net_roi_pct": float(net_roi_pct),
        "sharpe_ratio": float(all_metrics["sharpe_ratio"]),
        "max_drawdown_pct": float(mdd_pct),
        "win_rate_pct": float(win_rate_pct),
        "profit_factor": float(pf),
        "total_trades": int(total_trades),
        "avg_pnl_per_trade": float(avg_pnl_per_trade),
        # 追加メトリクス（ztb.metricsから）
        "sortino_ratio": float(all_metrics["sortino_ratio"]),
        "calmar_ratio": float(all_metrics["calmar_ratio"]),
        "volatility": float(all_metrics["volatility"]),
        "expected_value": float(all_metrics["expected_value"]),
    }


def to_python(obj):
    """NumPy型をPython標準型に変換"""
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


# =============================================================================
# 実験実行
# =============================================================================

def create_experiment_config(
    experiment_name: str,
    seed: int,
    reward_config: Dict[str, Any],
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
                "feature_set": "minimal",
                "reward_settings": reward_config,
            },
            "walk_forward": {
                "enabled": False
            }
        },
        "experiment_name": f"{experiment_name}_seed{seed}",
        "output_dir": OUTPUT_DIR
    }
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
    logger.info(f"報酬設定: {exp_def['reward']['name']}")
    logger.info(f"SAC設定: {'DEFAULT' if exp_def['sac'] is SAC_DEFAULT else 'TUNED'}")
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
        portfolio_history = training_stats.get("portfolio_history", [])
        trade_returns = training_stats.get("trade_returns", [])
        total_trades = training_stats.get("total_trades", 0)
        
        if not portfolio_history:
            # 履歴がない場合、Final Rewardから推定
            logger.warning("⚠️ Portfolio履歴取得失敗 - UnifiedTrainerからの返却なし")
            logger.warning(f"  training_stats keys: {list(training_stats.keys())[:10]}...")  # 最初10個のみ
            initial_value = 100000.0
            estimated_roi = basic_metrics["final_reward"] * 100
            portfolio_history = [initial_value, initial_value * (1 + estimated_roi / 100)]
            logger.warning(f"  Final Reward ({basic_metrics['final_reward']:.6f}) → 推定ROI: {estimated_roi:.2f}%")
            logger.warning("  ⚠️ Sharpe/MaxDD/WinRateは信頼性なし（後続backtestで再評価推奨）")
        
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
        logger.info(f"  Profit Factor: {all_metrics['profit_factor']:.2f}")
        
        return {
            "experiment_name": exp_id,
            "experiment_type": exp_name,
            "description": exp_def["description"],
            "phase": exp_def["phase"],
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            "config": {
                "sac": "DEFAULT" if exp_def["sac"] is SAC_DEFAULT else "TUNED",
                "reward_name": exp_def["reward"]["name"],
                "reward_scale": exp_def["reward"].get("reward_scale", 1.0),
                "reward_clip": [exp_def["reward"].get("reward_clip_min", -100),
                               exp_def["reward"].get("reward_clip_max", 100)],
                "trade_frequency_penalty": exp_def["reward"].get("trade_frequency_penalty", 0),
                "action_smoothing": exp_def["reward"].get("action_smoothing", 0),
            },
            "metrics": all_metrics,
        }
        
    except Exception as e:
        logger.error(f"実験失敗: {exp_id} - {e}")
        logger.error(traceback.format_exc())
        
        return {
            "experiment_name": exp_id,
            "experiment_type": exp_name,
            "phase": exp_def["phase"],
            "status": "failed",
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            "error": str(e)
        }


def save_results(results: List[Dict[str, Any]], filepath: Path):
    """結果を保存"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"結果保存: {filepath}")


# =============================================================================
# 分析
# =============================================================================

def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """結果を分析し因果効果を計算"""
    
    # 成功した実験のみ
    successful = [r for r in results if r.get("status") == "completed"]
    
    if not successful:
        return {"error": "No successful experiments"}
    
    # 設定別に集計
    grouped = {}
    for r in successful:
        exp_type = r["experiment_type"]
        if exp_type not in grouped:
            grouped[exp_type] = []
        grouped[exp_type].append(r)
    
    # 統計計算
    summary = {}
    for name, group in grouped.items():
        metrics_list = [r["metrics"] for r in group]
        
        rois = [m["net_roi_pct"] for m in metrics_list]
        sharpes = [m["sharpe_ratio"] for m in metrics_list]
        sortinos = [m.get("sortino_ratio", 0) for m in metrics_list]
        calmars = [m.get("calmar_ratio", 0) for m in metrics_list]
        maxdds = [m["max_drawdown_pct"] for m in metrics_list]
        winrates = [m["win_rate_pct"] for m in metrics_list]
        pfs = [m["profit_factor"] for m in metrics_list]
        final_rewards = [m["final_reward"] for m in metrics_list]
        holds = [m["action_distribution"].get("HOLD", 0) for m in metrics_list]
        
        summary[name] = {
            "n_seeds": len(group),
            "phase": group[0]["phase"],
            # ROI
            "roi_mean": np.mean(rois),
            "roi_std": np.std(rois),
            "roi_min": np.min(rois),
            "roi_max": np.max(rois),
            # Sharpe
            "sharpe_mean": np.mean(sharpes),
            "sharpe_std": np.std(sharpes),
            # Sortino
            "sortino_mean": np.mean(sortinos),
            # Calmar
            "calmar_mean": np.mean(calmars),
            # MaxDD
            "maxdd_mean": np.mean(maxdds),
            "maxdd_std": np.std(maxdds),
            # Win Rate
            "winrate_mean": np.mean(winrates),
            # Profit Factor
            "pf_mean": np.mean(pfs),
            # Final Reward
            "final_reward_mean": np.mean(final_rewards),
            "final_reward_std": np.std(final_rewards),
            # HOLD ratio
            "hold_ratio_mean": np.mean(holds),
        }
    
    # 因果効果計算
    causal_effects = compute_causal_effects(summary)
    
    # 解釈生成
    interpretations = generate_interpretation(summary, causal_effects)
    
    # Day 7との比較
    day7_comparison = compare_with_day7(summary)
    
    return {
        "summary": summary,
        "causal_effects": causal_effects,
        "interpretation": interpretations,
        "day7_comparison": day7_comparison,
    }


def compute_causal_effects(summary: Dict) -> Dict:
    """因果効果を計算"""
    effects = {}
    
    # Phase A: スケール統一後のSAC効果
    if "S1_scaled_default" in summary and "S1_scaled_tuned" in summary:
        effects["sac_effect_roi"] = (
            summary["S1_scaled_tuned"]["roi_mean"] - 
            summary["S1_scaled_default"]["roi_mean"]
        )
        effects["sac_effect_sharpe"] = (
            summary["S1_scaled_tuned"]["sharpe_mean"] - 
            summary["S1_scaled_default"]["sharpe_mean"]
        )
        effects["sac_effect_maxdd"] = (
            summary["S1_scaled_tuned"]["maxdd_mean"] - 
            summary["S1_scaled_default"]["maxdd_mean"]
        )
    
    # Phase B: 報酬効果（SAC_DEFAULT下）
    if "S1_scaled_default" in summary and "S2_E_default" in summary:
        effects["reward_effect_roi"] = (
            summary["S2_E_default"]["roi_mean"] - 
            summary["S1_scaled_default"]["roi_mean"]
        )
        effects["reward_effect_sharpe"] = (
            summary["S2_E_default"]["sharpe_mean"] - 
            summary["S1_scaled_default"]["sharpe_mean"]
        )
    
    # 交互作用（スケール統一後）
    if all(k in summary for k in ["S1_scaled_default", "S1_scaled_tuned", 
                                   "S2_E_default", "S2_E_tuned"]):
        base = summary["S1_scaled_default"]["roi_mean"]
        sac_effect = summary["S1_scaled_tuned"]["roi_mean"] - base
        reward_effect = summary["S2_E_default"]["roi_mean"] - base
        expected = base + sac_effect + reward_effect
        actual = summary["S2_E_tuned"]["roi_mean"]
        
        effects["interaction_roi"] = actual - expected
        effects["additive_expected"] = expected
        effects["actual_combined"] = actual
    
    # Phase C: クリップ効果
    if "S1_scaled_default" in summary and "S3_clipped_default" in summary:
        effects["clip_effect_roi"] = (
            summary["S3_clipped_default"]["roi_mean"] - 
            summary["S1_scaled_default"]["roi_mean"]
        )
    
    # Phase C: ペナルティ効果
    if "S2_E_default" in summary and "S4_nopen_default" in summary:
        effects["penalty_effect_roi"] = (
            summary["S2_E_default"]["roi_mean"] - 
            summary["S4_nopen_default"]["roi_mean"]
        )
    
    return effects


def generate_interpretation(summary: Dict, causal_effects: Dict) -> List[str]:
    """結果の解釈を生成"""
    interpretations = []
    
    # SAC効果の解釈
    if "sac_effect_roi" in causal_effects:
        effect = causal_effects["sac_effect_roi"]
        if effect < -10:
            interpretations.append(
                f"⚠️ SAC_TUNEDはスケール統一後も有害（ROI: {effect:+.1f}%）。"
                "SAC設定自体に問題がある可能性。"
            )
        elif effect > 10:
            interpretations.append(
                f"✅ SAC_TUNEDはスケール統一後に有効（ROI: {effect:+.1f}%）。"
                "Day 7の失敗はスケール交絡が原因。"
            )
        else:
            interpretations.append(
                f"△ SAC_TUNEDの効果は限定的（ROI: {effect:+.1f}%）。"
                "スケール統一後も明確な優位性なし。"
            )
    
    # クリップ効果
    if "clip_effect_roi" in causal_effects:
        effect = causal_effects["clip_effect_roi"]
        if abs(effect) > 5:
            interpretations.append(
                f"🔬 reward_clip [-1,1] の効果: ROI {effect:+.1f}%"
            )
    
    # ペナルティ効果
    if "penalty_effect_roi" in causal_effects:
        effect = causal_effects["penalty_effect_roi"]
        if abs(effect) > 5:
            interpretations.append(
                f"🔬 ペナルティ（trade_freq+action_smooth）の効果: ROI {effect:+.1f}%"
            )
    
    # 交互作用
    if "interaction_roi" in causal_effects:
        interaction = causal_effects["interaction_roi"]
        if abs(interaction) > 10:
            interpretations.append(
                f"🔍 交互作用が大きい（ROI: {interaction:+.1f}%）。"
                "SAC設定と報酬設計の相乗効果が存在。"
            )
        else:
            interpretations.append(
                f"交互作用は小さい（ROI: {interaction:+.1f}%）。効果は加法的。"
            )
    
    return interpretations


def compare_with_day7(summary: Dict) -> Dict:
    """Day 7結果との比較（手動入力値）"""
    
    # Day 7の参照値（67番ドキュメントより）
    day7_reference = {
        "S1_default": {"roi": -2.5, "hold": 31.9},
        "S1_tuned": {"roi": -134.9, "hold": 48.3},
        "S2_default": {"roi": 0.04, "hold": 32.2},
        "S2_tuned": {"roi": 0.14, "hold": 35.0},
    }
    
    comparison = {}
    
    # S1_scaled_default vs Day7 S1_default
    if "S1_scaled_default" in summary:
        d7 = day7_reference["S1_default"]
        d8 = summary["S1_scaled_default"]
        comparison["scale_effect_on_S1_default"] = {
            "day7_roi": d7["roi"],
            "day8_roi": d8["roi_mean"],
            "diff": d8["roi_mean"] - d7["roi"],
            "interpretation": "スケール100.0統一による変化"
        }
    
    # S1_scaled_tuned vs Day7 S1_tuned
    if "S1_scaled_tuned" in summary:
        d7 = day7_reference["S1_tuned"]
        d8 = summary["S1_scaled_tuned"]
        comparison["scale_effect_on_S1_tuned"] = {
            "day7_roi": d7["roi"],
            "day8_roi": d8["roi_mean"],
            "diff": d8["roi_mean"] - d7["roi"],
            "interpretation": "スケール100.0統一でSAC_TUNEDの暴走が改善されたか"
        }
    
    # S2_E_tuned vs Day7 S2_tuned（再現性確認）
    if "S2_E_tuned" in summary:
        d7 = day7_reference["S2_tuned"]
        d8 = summary["S2_E_tuned"]
        comparison["E_tuned_reproducibility"] = {
            "day7_roi": d7["roi"],
            "day8_roi": d8["roi_mean"],
            "diff": d8["roi_mean"] - d7["roi"],
            "interpretation": "E設定の再現性確認"
        }
    
    return comparison


# =============================================================================
# メイン
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Day 8: スケール交絡除去実験")
    parser.add_argument("--quick", action="store_true", 
                        help="クイックモード（2 seeds, 25k steps）")
    parser.add_argument("--phase", choices=["A", "B", "C", "all"], default="all",
                        help="実行フェーズ（A:スケール交絡, B:E報酬再現, C:Ablation, all:全て）")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="学習ステップ数（デフォルト: quick=25000, full=50000）")
    args = parser.parse_args()
    
    # パラメータ設定
    if args.quick:
        seeds = SEEDS_QUICK
        total_timesteps = args.timesteps or 25000
        mode = "quick"
    else:
        seeds = SEEDS_FULL
        total_timesteps = args.timesteps or 50000
        mode = "full"
    
    # フェーズフィルタ
    if args.phase == "all":
        selected_experiments = EXPERIMENTS
    else:
        selected_experiments = [e for e in EXPERIMENTS if e["phase"] == args.phase]
    
    # 実行情報表示
    n_experiments = len(selected_experiments) * len(seeds)
    est_minutes = n_experiments * (40 if args.quick else 55)
    
    logger.info("=" * 80)
    logger.info("Day 8: スケール交絡除去実験（68番レビュー対応）")
    logger.info("=" * 80)
    logger.info(f"モード: {mode}")
    logger.info(f"フェーズ: {args.phase}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Timesteps: {total_timesteps}")
    logger.info(f"実験数: {n_experiments}（{len(selected_experiments)}設定 × {len(seeds)} seeds）")
    logger.info(f"推定時間: {est_minutes // 60}時間{est_minutes % 60}分")
    logger.info("=" * 80)
    
    # データ存在確認
    if not Path(DATA_PATH).exists():
        logger.error(f"データファイルが見つかりません: {DATA_PATH}")
        sys.exit(1)
    
    # 出力ディレクトリ
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 実験実行
    results = []
    start_time = time.time()
    
    for exp_def in selected_experiments:
        for seed in seeds:
            result = run_single_experiment(exp_def, seed, total_timesteps)
            results.append(result)
            
            # 中間保存
            save_results(results, output_dir / f"day8_interim_{timestamp}.json")
    
    total_time = time.time() - start_time
    
    # 最終結果保存
    save_results(results, output_dir / f"day8_scale_{timestamp}.json")
    
    # 分析
    analysis = analyze_results(results)
    analysis["metadata"] = {
        "timestamp": timestamp,
        "mode": mode,
        "phase": args.phase,
        "total_timesteps": total_timesteps,
        "seeds": seeds,
        "total_time_minutes": total_time / 60,
        "n_experiments": len(results),
        "n_successful": len([r for r in results if r.get("status") == "completed"]),
    }
    
    analysis_path = output_dir / f"day8_analysis_{timestamp}.json"
    save_results(analysis, analysis_path)
    
    # 結果表示
    print("\n" + "=" * 80)
    print("Day 8 スケール交絡除去実験 完了")
    print("=" * 80)
    print(f"実行時間: {total_time / 60:.1f}分")
    print(f"成功: {analysis['metadata']['n_successful']}/{analysis['metadata']['n_experiments']}")
    
    print("\n📊 設定別サマリ（0番 §5.2 メトリクス）:")
    print("-" * 120)
    print(f"{'設定':<22} {'ROI%':>8} {'±':>6} {'Sharpe':>8} {'Sortino':>8} {'MaxDD%':>8} {'WinRate%':>8} {'PF':>6} {'HOLD%':>8}")
    print("-" * 120)
    for name in sorted(analysis.get("summary", {}).keys()):
        s = analysis["summary"][name]
        print(f"{name:<22} {s['roi_mean']:>+8.2f} {s['roi_std']:>6.2f} "
              f"{s['sharpe_mean']:>8.4f} {s['sortino_mean']:>8.4f} {s['maxdd_mean']:>8.2f} "
              f"{s['winrate_mean']:>8.1f} {s['pf_mean']:>6.2f} {s['hold_ratio_mean']*100:>8.1f}")
    
    print("\n🔬 因果効果:")
    for key, value in analysis.get("causal_effects", {}).items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:+.2f}")
    
    print("\n💡 解釈:")
    for interp in analysis.get("interpretation", []):
        print(f"  {interp}")
    
    if analysis.get("day7_comparison"):
        print("\n📈 Day 7との比較:")
        for key, comp in analysis["day7_comparison"].items():
            print(f"  {key}:")
            print(f"    Day7: {comp['day7_roi']:+.2f}% → Day8: {comp['day8_roi']:+.2f}% "
                  f"(差: {comp['diff']:+.2f}%)")
    
    print("\n✅ Day 8完了。69番ドキュメントに結果を纏めてください。")
    print(f"結果: {output_dir / f'day8_scale_{timestamp}.json'}")
    print(f"分析: {analysis_path}")


if __name__ == "__main__":
    main()
