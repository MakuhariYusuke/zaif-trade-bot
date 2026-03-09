#!/usr/bin/env python3
"""
v460 唯一のランナー (orchestrator 専任).

001# §4.1 / §6.2 準拠.
責務: config 読込 → task ディスパッチ → 結果保存.
ビジネスロジックは lib/ に委譲.

003# レビュー反映:
  #1: baseline を XGB vs Logistic/Ridge ペアに修正
  #3: XGB パラメータ除外を _RESERVED_XGB_KEYS に委譲
  #7: _evaluate_gate が gate_thresholds.yaml を参照
  #16: task_feature_info を lib/tasks/ に分離

Usage:
  python scripts/v460/run_experiment.py --config configs/v460/experiments/g1_xgb_h5_direction.yaml
  python scripts/v460/run_experiment.py --config configs/v460/experiments/g1_xgb_h5_direction.yaml --seed 123
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import cast

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.v460.lib.config_loader import load_config, load_gate_thresholds
from scripts.v460.lib.evaluator import (
    WalkForwardResult,
    make_logistic,
    make_ridge,
    make_xgboost,
    make_xgboost_classifier,
    make_xgboost_regressor,
)
from scripts.v460.lib.manifest import ManifestWriter
from scripts.v460.lib.tasks.feature_info import task_feature_info
from scripts.v460.lib.tasks.sac_train import task_sac_train
from ztb.io.json_io import write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================
# Orchestrator
# ======================================================================

MODEL_FACTORIES = {
    "XGBoost": make_xgboost,
    "XGBClassifier": make_xgboost_classifier,
    "XGBRegressor": make_xgboost_regressor,
    "Logistic": make_logistic,
    "Ridge": make_ridge,
}

TASK_DISPATCH = {
    "feature_info": task_feature_info,
    "sac_train": task_sac_train,      # 001# P3-1 / 017# P0
    # "backtest": task_backtest,      # P4-1 で追加
}


def run(config_path: str, seed_override: int | None = None) -> dict:
    """Main orchestrator: load config → dispatch task → save results."""
    cfg = load_config(config_path)
    gate = cfg.get("_gate", "unknown")
    task_name = cfg.get("_task", "feature_info")

    if seed_override is not None:
        cfg["seed"] = seed_override

    seed = cfg.get("seed", 42)

    logger.info(f"Gate: {gate} | Task: {task_name} | Seed: {seed}")
    logger.info(f"Config: {config_path}")

    # Manifest start
    data_path = cfg["data"].get("v460_features_path") or cfg["data"].get("ohlcv_path", "")
    manifest = ManifestWriter()
    entry = manifest.start_run(
        config_path=config_path,
        config=cfg,
        data_path=data_path,
        gate=gate,
        seed=seed,
    )

    # Dispatch
    try:
        task_fn = TASK_DISPATCH.get(task_name)
        if task_fn is None:
            raise ValueError(f"Unknown task: {task_name}")

        # 356a# B4: G2 multi-seed execution
        seeds = cfg.get("seeds", [])
        if "G2" in gate and len(seeds) > 1:
            results = _run_multi_seed(cfg, seeds, task_fn)
        else:
            results = task_fn(cfg)

        # Determine gate result using thresholds from config
        # 007# F5: Must run BEFORE save so g1_judgment_cache is included in JSON
        gate_result = _evaluate_gate(gate, results, cfg)

        # Save results
        # 007# F4: fold_results 生配列を保存用に差し替え (JSON 肥大化回避)
        results_to_save = dict(results)
        if "fold_results_saved" in results_to_save:
            results_to_save["fold_results"] = results_to_save.pop("fold_results_saved")
        elif "fold_results" in results_to_save:
            # Legacy: fold_results_saved がない場合はキーだけ残す
            results_to_save["fold_results"] = {
                k: f"{len(v)} folds (raw arrays omitted)"
                for k, v in results_to_save["fold_results"].items()
            }

        results_dir = Path(cfg.get("output", {}).get("results_dir", "results/v460"))
        if not results_dir.is_absolute():
            results_dir = _PROJECT_ROOT / results_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        out_path = results_dir / f"{entry.run_id}.json"
        write_json(out_path, results_to_save, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Results saved: {out_path}")

        manifest.finish_run(
            entry, metrics=results.get("xgboost", {}),
            gate_result=gate_result,
            artifacts=[str(out_path)],
        )

        return results

    except Exception as e:
        logger.error(f"Task failed: {e}")
        manifest.finish_run(
            entry, metrics={}, gate_result="ERROR",
            status="failed",
        )
        raise


def _evaluate_gate(gate: str, results: dict, cfg: dict) -> str:
    """Gate evaluation using thresholds from config.

    003# #7: gate_thresholds.yaml の閾値を g1_judgment に渡す.
    007# F5: g1_judgment 結果を results にキャッシュ (JSON 保存用).
    007# F6: IC/accuracy/sig_folds の any() チェックを追加 (判定統一).
    """
    if "G1" in gate:
        from ztb.metrics.gate_checks import g1_judgment
        fold_results = results.get("fold_results", {})
        if fold_results:
            # Load gate thresholds from standalone YAML (not experiment config)
            try:
                gate_cfg = load_gate_thresholds()
                g1_cfg = gate_cfg.get("g1_info", {})
                alpha = g1_cfg.get("p_alpha", 0.05)
                min_effect = g1_cfg.get("min_cliff_d", 0.33)
            except Exception as e:
                logger.warning("gate_thresholds.yaml not found, using defaults: %s", e)
                alpha = 0.05
                min_effect = 0.33
                g1_cfg = {}

            judgment = g1_judgment(
                fold_results,
                alpha=alpha,
                min_effect=min_effect,
            )

            # 007# F5: Cache judgment result for run_gate_check.py re-use
            results["g1_judgment_cache"] = judgment

            # 007# F6: Extra threshold checks (unified with run_gate_check.py)
            min_ic = g1_cfg.get("min_ic", 0.02)
            min_accuracy = g1_cfg.get("min_accuracy", 0.51)
            min_sig_folds = g1_cfg.get("min_significant_folds", 2)

            xgb_results = results.get("xgboost", {})
            extra_any_pass = any(
                td.get("ic_mean", 0.0) >= min_ic
                and td.get("accuracy_mean", 0.0) >= min_accuracy
                and td.get("ic_significant_count", 0) >= min_sig_folds
                for td in xgb_results.values()
            ) if xgb_results else False

            final_pass = judgment["g1_pass"] and extra_any_pass
            return "PASS" if final_pass else "FAIL"

    # 356a# B4: G2 gate evaluation
    if "G2" in gate:
        seed_results = results.get("seed_results", [])
        if seed_results:
            from scripts.v460.run_gate_check import run_g2_judgment

            # run_g2_judgment expects a file path — use dict-based evaluation
            try:
                gate_cfg = load_gate_thresholds()
                thresholds = gate_cfg.get("g2_train", {})
            except Exception as e:
                logger.warning("gate_thresholds.yaml not found for G2: %s", e)
                thresholds = {}

            judgment = _evaluate_g2_from_results(results, thresholds)
            results["g2_judgment_cache"] = judgment
            return "PASS" if judgment["gate_result"] == "PASS" else "FAIL"

    return "PENDING"


# ======================================================================
# G2 Multi-seed Helpers (356a# B4)
# ======================================================================


def _run_multi_seed(
    cfg: dict,
    seeds: list[int],
    task_fn: object,
) -> dict:
    """Run SAC training across multiple seeds and aggregate for G2 gate.

    356a# B4: 4-seed 訓練実行 + 結果集約.
    """
    from statistics import stdev
    from typing import Callable

    task_callable = cast(Callable[[dict], dict], task_fn)
    seed_results: list[dict[str, object]] = []
    all_checkpoint_metrics: list[list[dict[str, int]]] = []
    raw_results: dict[int, dict[str, object]] = {}

    for i, seed in enumerate(seeds):
        logger.info(f"=== Multi-seed [{i + 1}/{len(seeds)}] seed={seed} ===")
        seed_cfg = dict(cfg)
        seed_cfg["seed"] = seed

        result = task_callable(seed_cfg)

        # Extract eval metrics
        eval_metrics = result.get("eval_metrics", {})
        if isinstance(eval_metrics, dict):
            gross_roi = float(eval_metrics.get("gross_roi", eval_metrics.get("mean_reward", 0.0)))
            ic_mean = float(eval_metrics.get("ic_mean", 0.0))
        else:
            gross_roi = 0.0
            ic_mean = 0.0

        seed_results.append({
            "seed": seed,
            "gross_roi": gross_roi,
            "ic_mean": ic_mean,
        })

        checkpoint_metrics = result.get("checkpoint_metrics", [])
        all_checkpoint_metrics.append(
            checkpoint_metrics if isinstance(checkpoint_metrics, list) else []
        )
        raw_results[seed] = result

    # Convergence 計算: 30K step 以降の ROI 変動
    convergence = _compute_convergence(all_checkpoint_metrics, window_start=30000)

    aggregated: dict[str, object] = {
        "seed_results": seed_results,
        "convergence": convergence,
        "raw_results": raw_results,
        "seeds": seeds,
        "algorithm": "sac",
    }

    return aggregated


def _compute_convergence(
    all_checkpoint_metrics: list[list[dict[str, int]]],
    window_start: int = 30000,
) -> dict[str, float]:
    """30K step 以降の ROI 変動を算出.

    356a# §5.2: convergence 計算.
    """
    roi_values: list[float] = []
    for cp_list in all_checkpoint_metrics:
        for cp in cp_list:
            timestep = cp.get("timesteps", 0)
            if isinstance(timestep, (int, float)) and timestep >= window_start:
                roi = cp.get("roi", cp.get("mean_reward", 0.0))
                if isinstance(roi, (int, float)):
                    roi_values.append(float(roi))

    if len(roi_values) < 2:
        return {"roi_variance_pct_after_30k": 0.0}

    roi_range = max(roi_values) - min(roi_values)
    return {"roi_variance_pct_after_30k": round(roi_range * 100, 4)}


def _evaluate_g2_from_results(
    results: dict,
    thresholds: dict,
) -> dict[str, object]:
    """G2 gate evaluation from in-memory results (dict-based).

    356a# B4: run_g2_judgment のロジックを dict 入力で再現.
    """
    from statistics import stdev

    seed_results = results.get("seed_results", [])
    if not seed_results:
        return {"gate": "G2-train", "gate_result": "FAIL", "checks": {}, "reason": "no seed_results"}

    checks: dict[str, dict[str, object]] = {}

    # E1: gross > 0 の seed 比率 >= 75%
    min_ratio = float(thresholds.get("min_positive_seed_ratio", 0.75))
    positive_seeds = sum(1 for s in seed_results if float(s.get("gross_roi", 0)) > 0)
    ratio = positive_seeds / len(seed_results)
    checks["positive_seed_ratio"] = {
        "value": ratio, "threshold": min_ratio, "pass": ratio >= min_ratio,
    }

    # E2: IC の seed 間標準偏差 <= 0.03
    max_ic_std = float(thresholds.get("max_ic_seed_std", 0.03))
    ic_values = [float(s.get("ic_mean", 0)) for s in seed_results]
    ic_std = stdev(ic_values) if len(ic_values) >= 2 else 0.0
    checks["ic_seed_std"] = {
        "value": ic_std, "threshold": max_ic_std, "pass": ic_std <= max_ic_std,
    }

    # E3: 30K以降の ROI 変動 <= 5%
    max_roi_var = float(thresholds.get("max_roi_variance_pct", 5.0))
    convergence = results.get("convergence", {})
    roi_var = float(convergence.get("roi_variance_pct_after_30k", 0.0)) if isinstance(convergence, dict) else 0.0
    checks["convergence"] = {
        "value": roi_var, "threshold": max_roi_var, "pass": roi_var <= max_roi_var,
    }

    # E4: worst-seed ROI > -2%
    worst_min = float(thresholds.get("worst_seed_min_roi", -0.02))
    roi_list = [float(s.get("gross_roi", 0)) for s in seed_results]
    worst_roi = min(roi_list) if roi_list else 0.0
    checks["worst_seed_roi"] = {
        "value": worst_roi, "threshold": worst_min, "pass": worst_roi > worst_min,
    }

    all_pass = all(c["pass"] for c in checks.values())
    return {
        "gate": "G2-train",
        "gate_result": "PASS" if all_pass else "FAIL",
        "checks": checks,
    }


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="v460 Experiment Runner")
    parser.add_argument("--config", required=True, help="Experiment YAML path")
    parser.add_argument("--seed", type=int, default=None, help="Seed override")
    args = parser.parse_args()

    results = run(args.config, args.seed)

    # Print summary
    print("\n" + "=" * 60)
    print("  v460 Experiment Summary")
    print("=" * 60)
    if "xgboost" in results:
        for target, data in results["xgboost"].items():
            print(f"  {target}: IC={data.get('ic_mean', 'N/A')} "
                  f"acc={data.get('accuracy_mean', 'N/A')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
