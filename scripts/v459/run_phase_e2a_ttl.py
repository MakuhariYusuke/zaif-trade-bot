#!/usr/bin/env python3
"""
Phase E2α — Cost-Free 訓練 + 最小保有期間 (TTL) 実験

E1 の発見:
  - cost=0 + threshold=0.30 で SAC は +2.82% ROI を達成 (CF3)
  - Oracle: cost=0.1% では完全予測でも -18.25%
  - 軸B+D: cost=0 で方向学習 → 保有期間延長で 1取引の swing > fee

実験:
  e2a_base:    cost=0, thr=0.30, min_hold=0   (CF3 再現 baseline)
  e2a_hold5:   cost=0, thr=0.30, min_hold=5   (5分保有)
  e2a_hold15:  cost=0, thr=0.30, min_hold=15  (15分保有)
  e2a_hold30:  cost=0, thr=0.30, min_hold=30  (30分保有)
  e2a_eval:    最良 min_hold で cost=0.001 eval (実コスト適用評価)

Usage:
  python scripts/v459/run_phase_e2a_ttl.py --experiment e2a_hold5
  python scripts/v459/run_phase_e2a_ttl.py  (全実験 subprocess 逐次)
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
from typing import Any, Dict, Optional

import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

OUTPUT_DIR = project_root / "results" / "phase_e2a"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_TIMESTEPS = 50_000
BUFFER_SIZE = 50_000


def _setup_logger(name: str) -> logging.Logger:
    log_file = OUTPUT_DIR / f"e2a_{name}.log"
    logger = logging.getLogger(f"e2a.{name}")
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
# 訓練 + Eval (1実験分)
# ============================================================================

def _run_experiment(
    label: str,
    transaction_cost: float,
    threshold: float,
    min_holding_period: int,
    eval_cost: Optional[float] = None,
) -> Dict[str, Any]:
    """
    1実験: 訓練(cost) → eval(eval_cost or cost) → IC → 結果 → 全解放。

    eval_cost: 推論時のみ別 cost を適用する場合に指定。
    """
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
    ov["min_holding_period"] = min_holding_period
    exp_def["env_overrides"] = ov

    config = build_config(f"e2a_{label}", 42, exp_def)
    config["training"]["total_timesteps"] = TOTAL_TIMESTEPS
    if "sac_hyperparameters" in config["training"]:
        config["training"]["sac_hyperparameters"]["buffer_size"] = BUFFER_SIZE
    config["model_name"] = f"e2a_{label}_model"

    logger.info(
        f"[{label}] 訓練開始 (cost={transaction_cost}, thr={threshold}, "
        f"min_hold={min_holding_period})"
    )
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

    # eval 時に別 cost を適用する場合
    if eval_cost is not None and eval_cost != transaction_cost:
        logger.info(f"[{label}] eval時 cost を {transaction_cost} → {eval_cost} に切替")
        if hasattr(raw_env, "config") and hasattr(raw_env.config, "transaction_cost"):
            raw_env.config.transaction_cost = eval_cost
        if hasattr(raw_env, "fee_model"):
            raw_env.fee_model.fee_rate = eval_cost

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
    ic = compute_ic_multi_horizon(diag["actions"], diag["prices"], [1, 5, 15, 60])
    bins = compute_ic_by_action_bin(diag["actions"], diag["prices"])

    actual_cost = eval_cost if eval_cost is not None else transaction_cost
    result: Dict[str, Any] = {
        "label": label,
        "config": {
            "train_cost": transaction_cost,
            "eval_cost": actual_cost,
            "threshold": threshold,
            "min_holding_period": min_holding_period,
        },
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

    # Gate2 判定 (cost > 0 のeval時のみ)
    if actual_cost > 0 and diag["eval_trades"] > 0:
        roi = diag["eval_net_roi"]
        gross = diag["eval_gross_pnl"]
        fees = diag["eval_total_fees"]
        trades = diag["eval_trades"]
        pf = gross / abs(fees) if abs(fees) > 0 else 0.0
        avg_net = (gross - abs(fees)) / trades if trades > 0 else 0.0
        result["gate2"] = {
            "roi": round(roi, 4),
            "pf": round(pf, 4),
            "avg_net_per_trade": round(avg_net, 2),
            "pass": roi > 0 and pf > 1.05,
        }

    logger.info(
        f"[{label}] eval: trades={diag['eval_trades']} "
        f"ROI={diag['eval_net_roi']:.2f}% gross={diag['eval_gross_pnl']:.0f} "
        f"fees={diag['eval_total_fees']:.0f} IC={result['best_abs_ic']:.4f}"
    )

    # 即時解放
    del model, trainer, vec_normalize, norm_fn, vec_env, raw_env, diag
    gc.collect()

    return result


# ============================================================================
# 実験定義
# ============================================================================

EXPERIMENTS: Dict[str, tuple] = {
    "e2a_base": (
        "Baseline: cost=0, thr=0.30, hold=0 (CF3再現)",
        lambda: _run_experiment("base", 0.0, 0.30, 0),
    ),
    "e2a_hold5": (
        "TTL=5: cost=0, min_hold=5min",
        lambda: _run_experiment("hold5", 0.0, 0.30, 5),
    ),
    "e2a_hold15": (
        "TTL=15: cost=0, min_hold=15min",
        lambda: _run_experiment("hold15", 0.0, 0.30, 15),
    ),
    "e2a_hold30": (
        "TTL=30: cost=0, min_hold=30min",
        lambda: _run_experiment("hold30", 0.0, 0.30, 30),
    ),
    # 実コスト評価 — 最良の hold を --best-hold で指定
    "e2a_realcost": (
        "実コスト評価: cost=0訓練 → cost=0.1% eval, hold=0",
        lambda: _run_experiment("realcost", 0.0, 0.30, 0, eval_cost=0.001),
    ),
}


def _save_result(name: str, data: Dict[str, Any]) -> Path:
    """JSON でNaN/inf安全に保存。"""
    def _clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return round(obj, 6)
        if isinstance(obj, np.generic):
            return _clean(float(obj))
        return obj

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"e2a_{name}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(data), f, indent=2, ensure_ascii=False)
    return path


def run_single(name: str) -> Optional[Dict[str, Any]]:
    """指定された実験を 1 本実行し保存。"""
    if name not in EXPERIMENTS:
        print(f"Unknown: {name}. Available: {list(EXPERIMENTS.keys())}")
        return None

    desc, fn = EXPERIMENTS[name]
    print(f"\n{'='*60}\n  {desc}\n{'='*60}")
    result = fn()
    path = _save_result(name, result)
    print(f"保存: {path}")
    return result


def run_all_subprocess() -> None:
    """全実験を subprocess で逐次。e2a_realcost は最後。"""
    logger = _setup_logger("orchestrator")
    python_exe = sys.executable
    script_path = str(Path(__file__).resolve())

    order = ["e2a_base", "e2a_hold5", "e2a_hold15", "e2a_hold30"]
    for name in order:
        desc, _ = EXPERIMENTS[name]
        logger.info(f"Starting: {desc}")
        proc = subprocess.run(
            [python_exe, script_path, "--experiment", name],
            capture_output=False, timeout=2400,
        )
        logger.info(f"Done: {name} (exit={proc.returncode})")

    # 最良 hold を判定
    best_hold = _find_best_hold()
    logger.info(f"最良 min_holding_period = {best_hold}")

    # 実コスト評価 (best_hold で上書き)
    EXPERIMENTS["e2a_realcost"] = (
        f"実コスト: cost=0訓練→cost=0.1%eval, hold={best_hold}",
        lambda: _run_experiment("realcost", 0.0, 0.30, best_hold, eval_cost=0.001),
    )
    proc = subprocess.run(
        [python_exe, script_path, "--experiment", "e2a_realcost"],
        capture_output=False, timeout=2400,
    )
    logger.info(f"Done: e2a_realcost (exit={proc.returncode})")


def _find_best_hold() -> int:
    """保存済み結果から最良 ROI の min_holding_period を返す。"""
    best_roi = -999.0
    best_hold = 15  # default fallback

    for f in OUTPUT_DIR.glob("e2a_*.json"):
        try:
            data = json.loads(f.read_text("utf-8"))
            hold = data.get("config", {}).get("min_holding_period", 0)
            roi = data.get("eval", {}).get("net_roi", -999)
            if roi > best_roi:
                best_roi = roi
                best_hold = hold
        except Exception:
            continue

    return best_hold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase E2α: Cost-Free + TTL")
    parser.add_argument("--experiment", type=str, default=None)
    args = parser.parse_args()

    try:
        if args.experiment:
            run_single(args.experiment)
        else:
            run_all_subprocess()
    except Exception:
        logging.exception("E2α failed")
        sys.exit(1)
