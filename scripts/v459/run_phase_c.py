#!/usr/bin/env python3
"""
Phase C 統一実験ランナー — Gate 2 KPI全収集版

0番: Gate 2基準 (ROI>5%, PF>1.20, Sharpe>1.0, MaxDD<15%, WinRate>35%)
66番: 計測基盤が一度も測定していなかった → 本スクリプトで解消
91番: γ=0.80最優先、コスト負け(H2)対策、v451 Golden Era回帰
100# §12: C0計測統一 + C1コスト圧縮を統合実行

Usage:
  # 単一実験
  python scripts/v459/run_phase_c.py --single-run --experiment gamma_080 --seed 42

  # バッチ実行（C0+C1統合）
  python scripts/v459/run_phase_c.py --batch c0_c1
"""

import gc
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.metrics.metrics import (
    calculate_all_metrics,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    win_rate,
)
from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer
from ztb.training.utils.env_metrics import (
    compute_balance_roi,
    extract_trainer_env_metrics,
)
from ztb.utils.env_metrics import resolve_env, unwrap_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ============================================================================
# 定数
# ============================================================================

DATA_PATH = str(project_root / "data" / "btc_jpy_1m_v451_optimized_features.parquet")
OUTPUT_DIR = project_root / "results" / "phase_c"

INITIAL_BALANCE = 100000.0
TOTAL_TIMESTEPS = 50000

# ============================================================================
# SAC基本設定（P1-1ベース）
# ============================================================================

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

# P1-1ベース報酬設定（use_simple_reward=True, ペナルティ全無効）
REWARD_BASE = {
    "use_simple_reward": True,
    "balance_penalty": 0.0,
    "balance_penalty_tolerance": 1.0,
    "position_penalty_scale": 0.0,
    "position_penalty_exponent": 1.0,
    "inventory_penalty_scale": 0.0,
    "trade_frequency_penalty": 0.0,
    "trade_cooldown_penalty": 0.0,
    "consecutive_trade_penalty": 0.0,
    "hold_penalty_multiplier": 1.0,
    "volatility_penalty_scale": 0.0,
    "consistency_penalty": 0.0,
    "redundant_trade_penalty": 0.0,
    "profit_weight": 1.0,
    "reward_scale": 100.0,
    "confidence_penalty_factor": 0.0,
    "balance_shaping_enabled": False,
    "action_entropy_shaping_enabled": False,
    "long_position_reward_multiplier": 1.0,
    "short_position_reward_multiplier": 1.0,
    "long_position_penalty_multiplier": 1.0,
    "short_position_penalty_multiplier": 1.0,
}


# ============================================================================
# 実験定義 — C0+C1+91# H1統合
# ============================================================================

def get_experiment_configs() -> Dict[str, Dict[str, Any]]:
    """Phase C 全実験定義。
    
    命名規則: {phase}_{variable}_{value}
    91#優先順: H1(gamma) ⭐⭐⭐ → H2(cost/threshold) ⭐⭐⭐
    """
    configs = {}

    # --- C0: P1-1再現 (Gate 2 KPI計測付き、ベースライン) ---
    configs["c0_baseline_p1"] = {
        "description": "P1-1再現 + Gate2 KPI全計測",
        "sac_overrides": {},
        "reward_overrides": {},
        "env_overrides": {},
    }

    # --- C1-H1: γ感度 (91# 最優先仮説) ---
    for gamma in [0.80, 0.90, 0.95]:
        gamma_key = f"{gamma:.2f}".replace('.', '')
        configs[f"c1_gamma_{gamma_key}"] = {
            "description": f"γ={gamma} (91# H1: v451={0.80})",
            "sac_overrides": {"gamma": gamma},
            "reward_overrides": {},
            "env_overrides": {},
        }

    # --- C1-H2: threshold感度 (取引コスト削減) ---
    for threshold in [0.50, 0.60, 0.70]:
        configs[f"c1_threshold_{int(threshold*100)}"] = {
            "description": f"threshold={threshold} (H2: 過剰取引抑制)",
            "sac_overrides": {},
            "reward_overrides": {},
            "env_overrides": {"continuous_to_discrete_threshold": threshold},
        }

    # --- C1-H1+H2: γ=0.80 + best threshold (組合せ) ---
    for threshold in [0.50, 0.60, 0.70]:
        configs[f"c1_gamma080_threshold_{int(threshold*100)}"] = {
            "description": f"γ=0.80 + threshold={threshold}",
            "sac_overrides": {"gamma": 0.80},
            "reward_overrides": {},
            "env_overrides": {"continuous_to_discrete_threshold": threshold},
        }

    # --- C1: min_holding_period (現行デフォルト=3) ---
    for mhp in [5, 10, 15]:
        configs[f"c1_holding_{mhp}"] = {
            "description": f"min_holding_period={mhp} (現行=3)",
            "sac_overrides": {},
            "reward_overrides": {},
            "env_overrides": {"min_holding_period": mhp},
        }

    # --- C1: v451復元 (91# Golden Era) ---
    configs["c1_v451_golden"] = {
        "description": "v451 Golden Era: γ=0.80, scale=1.0, loss_mult=1.2",
        "sac_overrides": {"gamma": 0.80},
        "reward_overrides": {
            "reward_scale": 1.0,  # P1-1は100.0
            "custom_reward_params": {"type": "pnl_centered"},  # V457RewardCalculator
        },
        "env_overrides": {},
    }

    return configs


# ============================================================================
# バッチ定義
# ============================================================================

BATCHES = {
    # C0+C1統合: seed=42でスクリーニング → 最良条件を4seedsで展開
    "c0_c1": [
        "c0_baseline_p1",
        "c1_gamma_080", "c1_gamma_090", "c1_gamma_095",
        "c1_threshold_50", "c1_threshold_60", "c1_threshold_70",
        "c1_gamma080_threshold_50", "c1_gamma080_threshold_60", "c1_gamma080_threshold_70",
        "c1_holding_5", "c1_holding_10", "c1_holding_15",
        "c1_v451_golden",
    ],
    # screening後のフルseed展開（実行時に動的指定）
    "full_seeds": [],
}


# ============================================================================
# 実験実行
# ============================================================================

def build_config(
    experiment_name: str,
    seed: int,
    exp_def: Dict[str, Any],
) -> Dict[str, Any]:
    """実験設定dict を構築"""
    sac_params = SAC_DEFAULT.copy()
    sac_params.update(exp_def.get("sac_overrides", {}))

    reward_params = REWARD_BASE.copy()
    reward_params.update(exp_def.get("reward_overrides", {}))

    env_overrides = exp_def.get("env_overrides", {})

    env_config = {
        "use_continuous_actions": True,
        "action_space_type": "continuous",
        "initial_portfolio_value": INITIAL_BALANCE,
        "transaction_cost": 0.001,
        "reward_settings": reward_params,
    }
    env_config.update(env_overrides)

    config = {
        "experiment_name": experiment_name,
        "training": {
            "algorithm": "SAC",
            "total_timesteps": TOTAL_TIMESTEPS,
            "eval_freq": 5000,
            "n_eval_episodes": 3,
            "log_interval": 100,
            "seed": seed,
            "sac_hyperparameters": sac_params,
            "data_config": {
                "data_path": DATA_PATH,
                "window_size": 60,
            },
            "environment": env_config,
            "walk_forward": {"enabled": False},
        },
        "reward": reward_params,
    }
    return config


def compute_gate2_metrics(env: Any) -> Dict[str, Any]:
    """Gate 2 KPI を環境のportfolio_value_historyから計算。
    
    0番 §5.2 基準:
    - Net ROI > 5%
    - PF > 1.20
    - Sharpe > 1.0
    - MaxDD < 15%
    - WinRate > 35%
    """
    # portfolio_value_history は deque(maxlen=512) → 全ステップ不足
    # statistics_calculator.portfolio_value_history は deque(maxlen=None) → 全ステップあり
    balances: Optional[np.ndarray] = None
    
    unwrapped = unwrap_env(env)
    if unwrapped is None:
        return {"gate2_available": False, "gate2_error": "env unwrap failed"}
    
    # 優先1: statistics_calculator (全ステップ保持, maxlen=None)
    sc = getattr(unwrapped, "statistics_calculator", None)
    if sc is not None:
        pvh = getattr(sc, "portfolio_value_history", None)
        if pvh is not None and len(pvh) > 10:
            balances = np.array(pvh, dtype=np.float64)
    
    # フォールバック: core.py の portfolio_value_history (最後512ステップ)
    if balances is None:
        pvh = getattr(unwrapped, "portfolio_value_history", None)
        if pvh is not None and len(pvh) > 10:
            balances = np.array(pvh, dtype=np.float64)
    
    if balances is None or len(balances) < 10:
        return {"gate2_available": False, "gate2_error": "insufficient balance history"}
    
    # balance → step returns
    returns = np.diff(balances) / np.maximum(balances[:-1], 1e-10)
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Gate 2 KPI計算
    gate2: Dict[str, Any] = {"gate2_available": True}
    
    try:
        gate2["sharpe"] = float(sharpe_ratio(returns, period_per_year=525600))  # 1分足→年換算
    except Exception:
        gate2["sharpe"] = 0.0
    
    try:
        gate2["max_drawdown"] = float(max_drawdown(balances))
    except Exception:
        gate2["max_drawdown"] = 0.0
    
    try:
        gate2["profit_factor"] = float(profit_factor(returns))
    except Exception:
        gate2["profit_factor"] = 0.0
    
    try:
        gate2["win_rate"] = float(win_rate(returns))
    except Exception:
        gate2["win_rate"] = 0.0
    
    # ROI (mark-to-market)
    gate2["mtm_roi"] = float((balances[-1] - balances[0]) / balances[0] * 100)
    gate2["balance_samples"] = len(balances)
    gate2["final_balance"] = float(balances[-1])
    gate2["initial_balance_sampled"] = float(balances[0])
    
    # Gate 2 判定
    gate2["gate2_pass"] = (
        gate2["mtm_roi"] > 5.0
        and gate2["profit_factor"] > 1.20
        and gate2["sharpe"] > 1.0
        and abs(gate2["max_drawdown"]) < 15.0
        and gate2["win_rate"] > 0.35
    )
    
    return gate2


def run_single_experiment(
    experiment_name: str,
    seed: int,
    exp_def: Dict[str, Any],
) -> Dict[str, Any]:
    """1回の実験: 学習→Gate2 KPI収集→結果返却"""
    start_time = time.time()
    
    logger.warning(f"\n{'='*60}")
    logger.warning(f"実験: {experiment_name} (seed={seed})")
    logger.warning(f"  {exp_def.get('description', '')}")
    logger.warning(f"{'='*60}")
    
    config = build_config(experiment_name, seed, exp_def)
    trainer = None
    
    try:
        trainer = SACTrainer(config=config, logger=logger)
        _result = trainer.train()
        elapsed = time.time() - start_time
        
        # 基本メトリクス
        result: Dict[str, Any] = {
            "experiment": experiment_name,
            "seed": seed,
            "description": exp_def.get("description", ""),
            "elapsed_seconds": round(elapsed, 1),
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "gamma": config["training"]["sac_hyperparameters"]["gamma"],
                "reward_scale": config["reward"].get("reward_scale", 100.0),
                "threshold": config["training"]["environment"].get(
                    "continuous_to_discrete_threshold", 0.3333
                ),
                "min_holding_period": config["training"]["environment"].get(
                    "min_holding_period", "default(3)"
                ),
                "transaction_cost": config["training"]["environment"]["transaction_cost"],
            },
        }
        
        # P1互換メトリクス
        metrics = extract_trainer_env_metrics(trainer, include_optional=True)
        if metrics:
            result.update(metrics)
            roi = compute_balance_roi(metrics)
            if roi is not None:
                result["net_roi"] = roi
            if "gross_pnl" in metrics and metrics.get("initial_balance", 0) > 0:
                result["gross_roi"] = metrics["gross_pnl"] / metrics["initial_balance"] * 100
        
        # ★ Gate 2 KPI (0番§5.2, 66番指摘)

        env = resolve_env(trainer)
        gate2 = compute_gate2_metrics(env)
        result["gate2"] = gate2
        
        # サマリログ
        logger.warning(f"  完了: {elapsed:.0f}秒")
        logger.warning(f"  Net ROI: {result.get('net_roi', 'N/A')}")
        logger.warning(f"  Trades: {result.get('total_trades', 'N/A')}")
        logger.warning(f"  Gross PnL: {result.get('gross_pnl', 'N/A')}")
        logger.warning(f"  Fees: {result.get('total_fees', 'N/A')}")
        if gate2.get("gate2_available"):
            logger.warning(f"  [Gate2] PF={gate2['profit_factor']:.3f} "
                         f"Sharpe={gate2['sharpe']:.3f} "
                         f"MaxDD={gate2['max_drawdown']:.2f}% "
                         f"WinRate={gate2['win_rate']:.1%} "
                         f"{'PASS' if gate2['gate2_pass'] else 'FAIL'}")
        
        return result
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"実験失敗: {experiment_name} - {e}", exc_info=True)
        return {
            "experiment": experiment_name,
            "seed": seed,
            "elapsed_seconds": round(elapsed, 1),
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
    finally:
        if trainer is not None:
            try:
                trainer.cleanup_training_environment()
            except Exception:
                pass
            del trainer
        gc.collect()


def run_batch(batch_name: str, seeds: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """バッチ実行"""
    if seeds is None:
        seeds = [42]  # スクリーニングはseed=42のみ
    
    experiments = get_experiment_configs()
    batch_exps = BATCHES.get(batch_name, [])
    
    if not batch_exps:
        logger.error(f"バッチ '{batch_name}' が見つかりません")
        return []
    
    all_results: List[Dict[str, Any]] = []
    total = len(batch_exps) * len(seeds)
    
    logger.warning(f"\n{'='*70}")
    logger.warning(f"Phase C バッチ: {batch_name}")
    logger.warning(f"  実験数: {len(batch_exps)} × {len(seeds)} seeds = {total} runs")
    logger.warning(f"{'='*70}")
    
    for i, exp_name in enumerate(batch_exps, 1):
        if exp_name not in experiments:
            logger.warning(f"スキップ: {exp_name} (定義なし)")
            continue
        exp_def = experiments[exp_name]
        
        for seed in seeds:
            logger.warning(f"\n[{i}/{len(batch_exps)}] {exp_name} seed={seed}")
            result = run_single_experiment(exp_name, seed, exp_def)
            all_results.append(result)
    
    return all_results


def print_summary_table(results: List[Dict[str, Any]]) -> None:
    """結果サマリテーブルを出力"""
    logger.warning(f"\n{'='*120}")
    logger.warning("Phase C RESULTS SUMMARY")
    logger.warning(f"{'='*120}")
    
    header = (
        f"{'Experiment':<35} {'γ':>5} {'Thr':>5} {'ROI%':>8} "
        f"{'GrossPnL':>10} {'Fees':>8} {'Trades':>7} "
        f"{'PF':>6} {'Sharpe':>7} {'MaxDD%':>7} {'WinR%':>6} {'G2':>4}"
    )
    logger.warning(header)
    logger.warning("-" * 120)
    
    for r in results:
        if not r.get("success"):
            logger.warning(f"{r['experiment']:<35} FAILED: {r.get('error', '?')[:60]}")
            continue
        
        g2 = r.get("gate2", {})
        cfg = r.get("config", {})
        
        line = (
            f"{r['experiment']:<35} "
            f"{cfg.get('gamma', '?'):>5} "
            f"{cfg.get('threshold', 0.33):>5.2f} "
            f"{r.get('net_roi', 0):>7.2f}% "
            f"{r.get('gross_pnl', 0):>+10.0f} "
            f"{r.get('total_fees', 0):>8.0f} "
            f"{r.get('total_trades', 0):>7} "
            f"{g2.get('profit_factor', 0):>6.3f} "
            f"{g2.get('sharpe', 0):>7.3f} "
            f"{g2.get('max_drawdown', 0):>6.2f}% "
            f"{g2.get('win_rate', 0)*100:>5.1f}% "
            f"{'OK' if g2.get('gate2_pass') else 'NG':>4}"
        )
        logger.warning(line)
    
    # Gate 2 基準リマインダ
    logger.warning(f"\n{'─'*60}")
    logger.warning("Gate 2 基準 (0番§5.2):")
    logger.warning("  ROI>5% | PF>1.20 | Sharpe>1.0 | MaxDD<15% | WinRate>35%")
    logger.warning(f"{'─'*60}")


def save_results(results: List[Dict[str, Any]], batch_name: str) -> Path:
    """結果をJSON保存"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = OUTPUT_DIR / f"{batch_name}_{timestamp}.json"
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({
            "batch": batch_name,
            "timestamp": timestamp,
            "results": results,
            "gate2_criteria": {
                "roi": "> 5%",
                "profit_factor": "> 1.20",
                "sharpe": "> 1.0",
                "max_drawdown": "< 15%",
                "win_rate": "> 35%",
            },
        }, f, indent=2, ensure_ascii=False, default=str)
    
    return filepath


# ============================================================================
# メイン
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase C 統一実験ランナー")
    parser.add_argument("--single-run", action="store_true")
    parser.add_argument("--experiment", type=str, default="c0_baseline_p1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="42",
                       help="カンマ区切りseed例: 42,123,456,789")
    args = parser.parse_args()
    
    if args.single_run:
        experiments = get_experiment_configs()
        if args.experiment not in experiments:
            logger.error(f"実験 '{args.experiment}' が見つかりません")
            logger.error(f"利用可能: {list(experiments.keys())}")
            sys.exit(1)
        
        result = run_single_experiment(
            args.experiment, args.seed, experiments[args.experiment]
        )
        # stdout最終行にJSON出力（subprocess対応）
        print(json.dumps(result, ensure_ascii=False, default=str))
    
    elif args.batch:
        seeds = [int(s) for s in args.seeds.split(",")]
        results = run_batch(args.batch, seeds)
        print_summary_table(results)
        filepath = save_results(results, args.batch)
        logger.warning(f"\n結果保存: {filepath}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
