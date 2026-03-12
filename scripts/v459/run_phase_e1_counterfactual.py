#!/usr/bin/env python3
"""
Phase E1 Counterfactual — SAC 学習障害の原因切り分け

各実験を subprocess で完全プロセス分離し、メモリリークを根本回避。

  CF1: 手数料 0 環境で訓練 → IC 改善するか？
  CF3: 手数料 0 + 低 threshold → 純粋な方向予測力テスト
  Oracle: 正解方向 action の理論上限

Usage:
  python scripts/v459/run_phase_e1_counterfactual.py
  python scripts/v459/run_phase_e1_counterfactual.py --experiment cf1
  python scripts/v459/run_phase_e1_counterfactual.py --experiment oracle_normal
"""

import argparse
import gc
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import spearmanr

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

OUTPUT_DIR = project_root / "results" / "phase_e1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_TIMESTEPS = 50_000
BUFFER_SIZE = 50_000  # メモリ削減: 100K → 50K
IC_HORIZONS = [1, 5, 15, 60]


def _setup_logger(name: str = "e1") -> logging.Logger:
    log_file = OUTPUT_DIR / f"e1_{name}.log"
    logger = logging.getLogger(f"e1.{name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ============================================================================
# 単一実験実行 (subprocess から呼ばれる)
# ============================================================================

def _run_single_cf(label: str, transaction_cost: float, threshold: float) -> Dict[str, Any]:
    """CF 実験1本: 訓練 → eval → IC計算 → 結果返却 → 全解放。"""
    logger = _setup_logger(label)

    from scripts.v459.run_phase_c import (
        build_config, get_experiment_configs,
        _find_vec_normalize, _reset_risk_controllers,
    )
    from scripts.v459.run_phase_e0_diagnostic import (
        run_diagnostic_eval, compute_ic_multi_horizon, compute_ic_by_action_bin,
    )
    from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer
    from ztb.utils.env_metrics import resolve_env, unwrap_env

    exp_configs = get_experiment_configs()
    exp_def = exp_configs["d2_thr80"].copy()
    ov = exp_def.get("env_overrides", {}).copy()
    ov["transaction_cost"] = transaction_cost
    ov["continuous_to_discrete_threshold"] = threshold
    exp_def["env_overrides"] = ov

    config = build_config(f"e1_{label}", 42, exp_def)
    config["training"]["total_timesteps"] = TOTAL_TIMESTEPS
    config["training"]["sac_hyperparameters"]["buffer_size"] = BUFFER_SIZE
    config["model_name"] = f"e1_{label}_model"

    logger.info(f"[{label}] 訓練開始 (cost={transaction_cost}, thr={threshold}, buffer={BUFFER_SIZE})")
    t0 = time.time()

    trainer = SACTrainer(config=config, logger=logger)
    trainer.train()
    train_sec = time.time() - t0
    logger.info(f"[{label}] 訓練完了 ({train_sec:.0f}秒)")

    # env / model 取得
    vec_env = resolve_env(trainer)
    if vec_env is None and hasattr(trainer, "model") and hasattr(trainer.model, "get_env"):
        vec_env = trainer.model.get_env()
    raw_env = unwrap_env(vec_env) if vec_env is not None else None
    if raw_env is None:
        logger.error(f"[{label}] env 取得失敗")
        del trainer
        gc.collect()
        return {"label": label, "error": "env not found"}

    vec_normalize = _find_vec_normalize(vec_env)
    if vec_normalize is not None:
        vec_normalize.training = False
    norm_fn = vec_normalize.normalize_obs if vec_normalize else None
    model = trainer.model if hasattr(trainer, "model") else trainer.algorithm_trainer.model

    max_steps = min(getattr(raw_env, "n_steps", TOTAL_TIMESTEPS), TOTAL_TIMESTEPS)

    diag = run_diagnostic_eval(
        model, raw_env, max_steps, threshold,
        normalize_fn=norm_fn, eval_dd_threshold=1.0,
    )
    ic = compute_ic_multi_horizon(diag["actions"], diag["prices"], IC_HORIZONS)
    bins = compute_ic_by_action_bin(diag["actions"], diag["prices"])

    result = {
        "label": label,
        "config": {"transaction_cost": transaction_cost, "threshold": threshold},
        "train_seconds": round(train_sec, 1),
        "eval": {
            "steps": diag["eval_steps"],
            "trades": diag["eval_trades"],
            "net_roi": round(diag["eval_net_roi"], 4),
            "gross_pnl": round(diag["eval_gross_pnl"], 2),
            "total_fees": round(diag["eval_total_fees"], 2),
        },
        "ic": ic,
        "action_bins": bins,
        "best_abs_ic": round(
            max((abs(v.get("spearman", 0)) for v in ic.values()), default=0.0), 4
        ),
    }

    logger.info(
        f"[{label}] eval: trades={diag['eval_trades']} "
        f"ROI={diag['eval_net_roi']:.2f}% IC_best={result['best_abs_ic']:.4f}"
    )

    # 即時解放
    del model, trainer, vec_normalize, norm_fn, vec_env, raw_env, diag
    gc.collect()

    return result


def _run_oracle(transaction_cost: float) -> Dict[str, Any]:
    """Oracle: 1-step先の価格方向を覗き見て action を決定。"""
    label = f"oracle_cost{transaction_cost}"
    logger = _setup_logger(label)

    from scripts.v459.run_phase_c import (
        build_config, get_experiment_configs, _reset_risk_controllers,
    )
    from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer
    from ztb.utils.env_metrics import resolve_env, unwrap_env

    exp_configs = get_experiment_configs()
    exp_def = exp_configs["d2_thr80"].copy()
    ov = exp_def.get("env_overrides", {}).copy()
    ov["transaction_cost"] = transaction_cost
    ov["continuous_to_discrete_threshold"] = 0.80
    exp_def["env_overrides"] = ov

    # env 構築のための最小訓練
    config = build_config(f"e1_{label}", 42, exp_def)
    config["training"]["total_timesteps"] = 200
    if "sac_hyperparameters" in config["training"]:
        config["training"]["sac_hyperparameters"]["buffer_size"] = 1000
    elif "sac" in config["training"]:
        config["training"]["sac"]["buffer_size"] = 1000
    config["model_name"] = f"e1_{label}_model"

    logger.info(f"[Oracle] env構築 (cost={transaction_cost})")
    trainer = SACTrainer(config=config, logger=logger)
    trainer.train()

    vec_env = resolve_env(trainer)
    if vec_env is None and hasattr(trainer, "model") and hasattr(trainer.model, "get_env"):
        vec_env = trainer.model.get_env()
    raw_env = unwrap_env(vec_env) if vec_env is not None else None
    if raw_env is None:
        logger.error("Oracle: env 取得失敗")
        del trainer
        gc.collect()
        return {"label": label, "error": "env not found"}

    # trainer を先に解放 → Oracle eval はモデル不要
    del trainer
    gc.collect()

    _reset_risk_controllers(raw_env, eval_dd_threshold=1.0)
    if hasattr(raw_env, "action_threshold"):
        raw_env.action_threshold = 0.80
        raw_env.negative_action_threshold = -0.80

    obs, _ = raw_env.reset(seed=42, options={"random_start": False})
    done = False
    balances = [float(raw_env.portfolio_value)]
    max_steps = min(getattr(raw_env, "n_steps", TOTAL_TIMESTEPS), TOTAL_TIMESTEPS)
    step_count = 0
    prev_tc = int(raw_env.trades_count)
    prev_rp = float(raw_env.realized_pnl)
    trade_pnls: List[float] = []

    while not done and step_count < max_steps - 1:
        cur_price = raw_env._resolve_price()
        nxt_price = raw_env._resolve_price(raw_env.current_step + 1)
        diff = nxt_price - cur_price

        if diff > 0:
            action = np.array([0.99], dtype=np.float32)
        elif diff < 0:
            action = np.array([-0.99], dtype=np.float32)
        else:
            action = np.array([0.0], dtype=np.float32)

        obs, _, terminated, truncated, info = raw_env.step(action)
        done = terminated or truncated
        balances.append(float(raw_env.portfolio_value))
        step_count += 1

        cur_tc = int(raw_env.trades_count)
        cur_rp = float(raw_env.realized_pnl)
        if cur_tc > prev_tc:
            trade_pnls.append(cur_rp - prev_rp)
        prev_tc = cur_tc
        prev_rp = cur_rp

    result = {
        "label": label,
        "config": {"transaction_cost": transaction_cost, "oracle": True},
        "eval": {
            "steps": step_count,
            "trades": int(raw_env.total_trades),
            "net_roi": round(
                (balances[-1] - balances[0]) / balances[0] * 100 if balances[0] > 0 else 0.0, 4
            ),
            "gross_pnl": round(float(getattr(raw_env, "gross_pnl", 0.0)), 2),
            "total_fees": round(float(getattr(raw_env, "total_fees", 0.0)), 2),
        },
        "avg_pnl_per_trade": round(float(np.mean(trade_pnls)), 2) if trade_pnls else 0.0,
    }

    logger.info(
        f"[Oracle] cost={transaction_cost}: trades={result['eval']['trades']} "
        f"ROI={result['eval']['net_roi']:.2f}% gross={result['eval']['gross_pnl']:.0f}"
    )

    del raw_env, vec_env
    gc.collect()
    return result


# ============================================================================
# 実験ディスパッチ
# ============================================================================

EXPERIMENTS = {
    "cf1": ("CF1: 手数料0, thr=0.80", lambda: _run_single_cf("cf1_zero_cost", 0.0, 0.80)),
    "cf3": ("CF3: 手数料0, thr=0.30", lambda: _run_single_cf("cf3_zero_cost_low_thr", 0.0, 0.30)),
    "oracle_normal": ("Oracle: 手数料0.001", lambda: _run_oracle(0.001)),
    "oracle_zero": ("Oracle: 手数料0", lambda: _run_oracle(0.0)),
}


def _save_result(name: str, data: Dict[str, Any]) -> Path:
    """numpy 安全な JSON 保存。"""
    def _clean(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"e1_{name}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(data), f, indent=2, ensure_ascii=False)
    return path


def run_single_experiment(name: str) -> Optional[Dict[str, Any]]:
    """指定された実験を1本実行し結果を保存。"""
    if name not in EXPERIMENTS:
        print(f"Unknown experiment: {name}. Available: {list(EXPERIMENTS.keys())}")
        return None

    desc, fn = EXPERIMENTS[name]
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")

    result = fn()
    path = _save_result(name, result)
    print(f"保存: {path}")
    return result


def run_all_via_subprocess() -> Dict[str, Any]:
    """全実験を subprocess で逐次実行。プロセス分離でメモリリーク防止。"""
    logger = _setup_logger("orchestrator")
    report: Dict[str, Any] = {
        "phase": "E1",
        "timestamp": datetime.now().isoformat(),
        "experiments": {},
    }
    start_time = time.time()

    python_exe = sys.executable
    script_path = str(Path(__file__).resolve())

    for name, (desc, _) in EXPERIMENTS.items():
        logger.info(f"{'='*60}")
        logger.info(f"Starting: {desc}")
        logger.info(f"{'='*60}")

        t0 = time.time()
        proc = subprocess.run(
            [python_exe, script_path, "--experiment", name],
            capture_output=False,
            timeout=2400,  # 40分タイムアウト
        )
        elapsed = time.time() - t0
        logger.info(f"Completed: {name} ({elapsed:.0f}s, exit={proc.returncode})")

        # 結果ファイルを読み込み
        result_files = sorted(OUTPUT_DIR.glob(f"e1_{name}_*.json"), reverse=True)
        if result_files:
            with open(result_files[0], "r", encoding="utf-8") as f:
                report["experiments"][name] = json.load(f)
        else:
            report["experiments"][name] = {"error": f"exit_code={proc.returncode}"}

    # 総合診断
    elapsed_total = time.time() - start_time
    report["elapsed_seconds"] = round(elapsed_total, 1)

    # E0 baseline IC を読み込み
    e0_files = sorted((project_root / "results" / "phase_e0").glob("e0_diagnostic_*.json"), reverse=True)
    bl_ic = 0.0
    if e0_files:
        with open(e0_files[0], "r", encoding="utf-8") as f:
            e0_data = json.load(f)
            bl_ic = e0_data.get("best_abs_ic", 0.0)
            report["e0_baseline_ic"] = bl_ic

    cf1 = report["experiments"].get("cf1", {})
    cf3 = report["experiments"].get("cf3", {})
    cf1_ic = cf1.get("best_abs_ic", 0.0)
    cf3_ic = cf3.get("best_abs_ic", 0.0)

    if cf1_ic > 0.05 or cf3_ic > 0.05:
        verdict = "COST_IS_BOTTLENECK"
        explanation = "手数料除去でIC改善 → コスト構造が学習を阻害"
    elif cf1_ic > bl_ic + 0.01:
        verdict = "COST_PARTIALLY_BLOCKING"
        explanation = "手数料除去で微改善 → コストと特徴量の両方に問題"
    else:
        verdict = "FEATURES_OR_ALGORITHM_INSUFFICIENT"
        explanation = "手数料除去でもIC改善なし → 特徴量/アルゴリズムが方向予測に不適"

    report["diagnosis"] = {
        "verdict": verdict,
        "explanation": explanation,
        "e0_baseline_ic": bl_ic,
        "cf1_best_ic": cf1_ic,
        "cf3_best_ic": cf3_ic,
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"E1 Counterfactual 完了 ({elapsed_total:.0f}秒)")
    logger.info(f"診断: {verdict}")
    logger.info(f"説明: {explanation}")
    logger.info(f"IC: E0_BL={bl_ic:.4f}  CF1={cf1_ic:.4f}  CF3={cf3_ic:.4f}")
    logger.info(f"{'='*60}")

    path = _save_result("summary", report)
    logger.info(f"総合結果: {path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase E1 Counterfactual")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Run a single experiment (cf1, cf3, oracle_normal, oracle_zero)")
    args = parser.parse_args()

    try:
        if args.experiment:
            run_single_experiment(args.experiment)
        else:
            run_all_via_subprocess()
    except Exception:
        logging.exception("E1 failed")
        sys.exit(1)
