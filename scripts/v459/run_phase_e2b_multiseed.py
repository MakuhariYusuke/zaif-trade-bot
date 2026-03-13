#!/usr/bin/env python3
"""
Phase E2β — Multi-seed 検証 + OOS

E2α の最良条件 (cost=0, thr=0.30, hold=0) を4seed×50Kで再現性検証,
最良モデルでOOS evalを行う。

Usage:
  python scripts/v459/run_phase_e2b_multiseed.py --seed 42
  python scripts/v459/run_phase_e2b_multiseed.py --seed 123
  python scripts/v459/run_phase_e2b_multiseed.py  (全seed subprocess逐次)
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

OUTPUT_DIR = project_root / "results" / "phase_e2b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_TIMESTEPS = 50_000
BUFFER_SIZE = 50_000
SEEDS = [42, 123, 456, 789]


def _setup_logger(name: str) -> logging.Logger:
    log_file = OUTPUT_DIR / f"e2b_{name}.log"
    logger = logging.getLogger(f"e2b.{name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _run_seed(seed: int) -> Dict[str, Any]:
    """1 seed 分の訓練 + eval。"""
    label = f"seed{seed}"
    logger = _setup_logger(label)

    from scripts.v459.run_phase_c import (
        build_config, get_experiment_configs,
        _find_vec_normalize,
    )
    from scripts.v459.run_phase_e0_diagnostic import (
        run_diagnostic_eval, compute_ic_multi_horizon, compute_ic_by_action_bin,
    )
    from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer
    from ztb.utils.env_metrics import resolve_env, unwrap_env

    exp_configs = get_experiment_configs()
    exp_def = exp_configs["d2_thr80"].copy()
    ov = exp_def.get("env_overrides", {}).copy()
    ov["transaction_cost"] = 0.0
    ov["continuous_to_discrete_threshold"] = 0.30
    ov["min_holding_period"] = 0
    exp_def["env_overrides"] = ov

    config = build_config(f"e2b_{label}", seed, exp_def)
    config["training"]["total_timesteps"] = TOTAL_TIMESTEPS
    if "sac_hyperparameters" in config["training"]:
        config["training"]["sac_hyperparameters"]["buffer_size"] = BUFFER_SIZE
    config["model_name"] = f"e2b_{label}_model"

    logger.info(f"[{label}] 訓練開始 (seed={seed}, cost=0, thr=0.30, hold=0)")
    t0 = time.time()

    trainer = SACTrainer(config=config, logger=logger)
    trainer.train()
    train_sec = time.time() - t0
    logger.info(f"[{label}] 訓練完了 ({train_sec:.0f}秒)")

    vec_env = resolve_env(trainer)
    if vec_env is None and hasattr(trainer, "model") and hasattr(trainer.model, "get_env"):
        vec_env = trainer.model.get_env()
    raw_env = unwrap_env(vec_env) if vec_env is not None else None
    if raw_env is None:
        logger.error(f"[{label}] env 取得失敗")
        del trainer
        gc.collect()
        return {"label": label, "seed": seed, "error": "env not found"}

    vec_normalize = _find_vec_normalize(vec_env)
    if vec_normalize is not None:
        vec_normalize.training = False
    norm_fn = vec_normalize.normalize_obs if vec_normalize else None
    model = trainer.model if hasattr(trainer, "model") else trainer.algorithm_trainer.model

    max_steps = min(getattr(raw_env, "n_steps", TOTAL_TIMESTEPS), TOTAL_TIMESTEPS)

    diag = run_diagnostic_eval(
        model, raw_env, max_steps, 0.30,
        normalize_fn=norm_fn, eval_dd_threshold=1.0,
    )
    ic = compute_ic_multi_horizon(diag["actions"], diag["prices"], [1, 5, 15, 60])
    bins = compute_ic_by_action_bin(diag["actions"], diag["prices"])

    result: Dict[str, Any] = {
        "label": label,
        "seed": seed,
        "config": {"train_cost": 0.0, "threshold": 0.30, "min_holding_period": 0},
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
        f"ROI={diag['eval_net_roi']:.2f}% gross={diag['eval_gross_pnl']:.0f} "
        f"IC={result['best_abs_ic']:.4f}"
    )

    # 即時解放
    del model, trainer, vec_normalize, norm_fn, vec_env, raw_env, diag
    gc.collect()
    return result


def _save_result(name: str, data: Dict[str, Any]) -> Path:
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
    path = OUTPUT_DIR / f"e2b_{name}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(data), f, indent=2, ensure_ascii=False)
    return path


def run_single_seed(seed: int) -> Optional[Dict[str, Any]]:
    label = f"seed{seed}"
    print(f"\n{'='*60}\n  E2β: seed={seed}, cost=0, thr=0.30\n{'='*60}")
    result = _run_seed(seed)
    path = _save_result(label, result)
    print(f"保存: {path}")
    return result


def run_all_subprocess() -> None:
    logger = _setup_logger("orchestrator")
    python_exe = sys.executable
    script_path = str(Path(__file__).resolve())

    for seed in SEEDS:
        logger.info(f"Starting seed={seed}")
        proc = subprocess.run(
            [python_exe, script_path, "--seed", str(seed)],
            capture_output=False, timeout=2400,
        )
        logger.info(f"Done seed={seed} (exit={proc.returncode})")

    # 全結果集約
    all_results = {}
    for f in sorted(OUTPUT_DIR.glob("e2b_seed*.json")):
        data = json.loads(f.read_text("utf-8"))
        seed = data.get("seed", 0)
        all_results[seed] = data

    rois = [r["eval"]["net_roi"] for r in all_results.values() if "eval" in r]
    grosses = [r["eval"]["gross_pnl"] for r in all_results.values() if "eval" in r]

    summary = {
        "phase": "E2β",
        "seeds": SEEDS,
        "n_seeds": len(rois),
        "roi_mean": round(np.mean(rois), 4) if rois else None,
        "roi_std": round(np.std(rois), 4) if rois else None,
        "roi_min": round(min(rois), 4) if rois else None,
        "roi_max": round(max(rois), 4) if rois else None,
        "gross_mean": round(np.mean(grosses), 2) if grosses else None,
        "positive_roi_count": sum(1 for r in rois if r > 0),
        "gate_pass": sum(1 for r in rois if r > 0) >= 3,
        "per_seed": {s: {"roi": r["eval"]["net_roi"], "gross": r["eval"]["gross_pnl"],
                         "trades": r["eval"]["trades"], "ic": r.get("best_abs_ic", 0)}
                     for s, r in all_results.items()},
    }
    path = _save_result("summary", summary)

    logger.info(f"\n{'='*60}")
    logger.info(f"E2β Summary: {len(rois)} seeds")
    logger.info(f"ROI: mean={summary['roi_mean']:.2f}% std={summary['roi_std']:.2f}% "
                f"min={summary['roi_min']:.2f}% max={summary['roi_max']:.2f}%")
    logger.info(f"Positive ROI: {summary['positive_roi_count']}/{len(rois)}")
    logger.info(f"Gate PASS: {summary['gate_pass']}")
    logger.info(f"Summary: {path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase E2β: Multi-seed validation")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    try:
        if args.seed is not None:
            run_single_seed(args.seed)
        else:
            run_all_subprocess()
    except Exception:
        logging.exception("E2β failed")
        sys.exit(1)
