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
from typing import Callable

# [WinError 1114] DLL load error workaround (load PyTorch before pandas / large models fragment memory)
try:
    import torch
except ImportError:
    pass

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
from scripts.v460.lib.gate_judgment_core import evaluate_g2_checks, evaluate_g3_checks
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

        # 356# B4: G2 multi-seed execution
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

    # 356# B4: G2 gate evaluation
    if "G2" in gate:
        seed_results = results.get("seed_results", [])
        if seed_results:
            # run_g2_judgment expects a file path — use dict-based evaluation
            try:
                gate_cfg = load_gate_thresholds()
                thresholds = gate_cfg.get("g2_train", {})
            except Exception as e:
                logger.warning("gate_thresholds.yaml not found for G2: %s", e)
                thresholds = {}

            convergence = results.get("convergence", {})
            judgment = evaluate_g2_checks(seed_results, convergence, thresholds)
            results["g2_judgment_cache"] = judgment

            # 396# G3 auto-evaluation: seed_metrics があれば G3 も判定
            seed_metrics_list = results.get("seed_metrics", [])
            if seed_metrics_list:
                try:
                    g3_thresholds = gate_cfg.get("g3_pnl", {})
                except Exception:
                    g3_thresholds = {}
                g3_judgment = evaluate_g3_checks(seed_metrics_list, g3_thresholds)
                if g3_judgment:
                    results["g3_judgment_cache"] = g3_judgment
                    logger.info(f"G3-pnl auto-evaluation: {g3_judgment.get('gate_result', '?')}")

            return "PASS" if judgment["gate_result"] == "PASS" else "FAIL"

    return "PENDING"


# ======================================================================
# G2 Multi-seed Helpers (356# B4)
# ======================================================================


def _run_multi_seed(
    cfg: dict,
    seeds: list[int],
    task_fn: Callable[[dict], dict],
) -> dict:
    """Run SAC training across multiple seeds and aggregate for G2 gate.

    356# B4: 4-seed 訓練実行 + 結果集約.
    """
    seed_results: list[dict[str, object]] = []
    all_checkpoint_metrics: list[list[dict[str, int | float]]] = []
    raw_results: dict[int, dict[str, object]] = {}

    for i, seed in enumerate(seeds):
        logger.info(f"=== Multi-seed [{i + 1}/{len(seeds)}] seed={seed} ===")
        seed_cfg = dict(cfg)
        seed_cfg["seed"] = seed

        try:
            result = task_fn(seed_cfg)
        except Exception as e:
            logger.error(f"Seed {seed} failed: {e}")
            seed_results.append({
                "seed": seed,
                "gross_roi": 0.0,
                "error": str(e),
            })
            continue

        eval_metrics = result.get("eval_metrics", {})
        if isinstance(eval_metrics, dict):
            gross_roi = float(eval_metrics.get("gross_roi", eval_metrics.get("mean_reward", 0.0)))
        else:
            gross_roi = 0.0

        seed_entry: dict[str, object] = {
            "seed": seed,
            "gross_roi": gross_roi,
        }

        # 396# G3 指標を seed_results に追加 (389# P1-1: run_g3_judgment 入力形式)
        if isinstance(eval_metrics, dict):
            for g3_key in ("pf", "sharpe_annual", "max_drawdown",
                           "avg_gross_per_trade", "avg_fee_per_trade",
                           "reward_profit_corr"):
                if g3_key in eval_metrics:
                    seed_entry[g3_key] = float(eval_metrics[g3_key])
            # 425# multi-slice 結果を seed_entry に追加
            if "slice_metrics" in eval_metrics:
                seed_entry["slice_metrics"] = eval_metrics["slice_metrics"]

        # 425# best_model eval 結果を seed_entry に追加
        best_eval = result.get("best_model_eval_metrics")
        if isinstance(best_eval, dict):
            seed_entry["best_model_eval"] = {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in best_eval.items()
            }

        seed_results.append(seed_entry)

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

    # 396# G3 seed_metrics: run_g3_judgment() が直接消費できる形式
    # seed_results に G3 指標が含まれていれば seed_metrics にコピー
    seed_metrics = [
        sr for sr in seed_results
        if "pf" in sr and "sharpe_annual" in sr and "max_drawdown" in sr
    ]
    if seed_metrics:
        aggregated["seed_metrics"] = seed_metrics

    return aggregated


def _compute_convergence(
    all_checkpoint_metrics: list[list[dict[str, int | float]]],
    window_start: int = 30000,
) -> dict[str, float]:
    """30K step 以降の ROI 変動を算出.

    356# §5.2: convergence 計算.
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


# 399# G2/G3 判定は gate_judgment_core に統合済み。
# evaluate_g2_checks, evaluate_g3_checks を直接使用。


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
