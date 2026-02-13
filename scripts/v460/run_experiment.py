#!/usr/bin/env python3
"""
v460 唯一のランナー (orchestrator 専任).

001# §4.1 / §6.2 準拠.
責務: config 読込 → task ディスパッチ → 結果保存.
ビジネスロジックは lib/ に委譲.

Usage:
  python scripts/v460/run_experiment.py --config configs/v460/experiments/g1_xgb_h5_direction.yaml
  python scripts/v460/run_experiment.py --config configs/v460/experiments/g1_xgb_h5_direction.yaml --seed 123
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.v460.lib.config_loader import load_config
from scripts.v460.lib.data_loader import generate_targets, load_parquet, split_train_eval
from scripts.v460.lib.evaluator import (
    WalkForwardResult,
    evaluate_multi_target,
    make_logistic,
    make_xgboost,
)
from scripts.v460.lib.manifest import ManifestWriter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================
# Task: feature_info (G1)
# ======================================================================

def task_feature_info(cfg: dict) -> dict:
    """G1 feature information test — XGBoost walk-forward.

    Returns results dict with fold_results for g1_judgment.
    """
    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]
    wf_cfg = cfg.get("walk_forward", {})
    xgb_cfg = cfg.get("xgboost", {})
    seed = cfg.get("seed", 42)

    # Load data
    data_path = data_cfg.get("v460_features_path") or data_cfg.get("ohlcv_path")
    feature_cols = feat_cfg["selected"]

    df = load_parquet(data_path, feature_cols)

    # Generate targets
    g1_cfg = cfg.get("g1", {})
    horizons = g1_cfg.get("horizons", [1, 5, 15])
    target_types = g1_cfg.get("targets", ["direction", "magnitude", "volatility"])

    df = generate_targets(df, horizons, target_types)

    # Ensure features are float32
    for col in feature_cols:
        df[col] = df[col].astype("float32")

    # XGBoost factory
    def xgb_factory():
        return make_xgboost(seed=seed, **{
            k: v for k, v in xgb_cfg.items()
            if k not in ("seed",)
        })

    # Walk-forward evaluation over all targets
    n_folds = wf_cfg.get("n_folds", 5)
    train_ratio = wf_cfg.get("train_ratio", 0.80)

    multi_results = evaluate_multi_target(
        df, feature_cols, horizons, target_types,
        xgb_factory, "XGBoost", n_folds, train_ratio,
    )

    # Also run logistic baseline
    baseline_results = evaluate_multi_target(
        df, feature_cols, horizons, target_types,
        lambda: make_logistic(seed=seed), "Logistic", n_folds, train_ratio,
    )

    # Build fold_results structure for g1_judgment
    fold_results: dict[str, list[tuple[list[float], list[float]]]] = {}
    for target_name, wf_result in multi_results.items():
        fold_pairs: list[tuple[list[float], list[float]]] = []
        for fold in wf_result.folds:
            fold_pairs.append((fold.model_scores, fold.baseline_scores))
        fold_results[target_name] = fold_pairs

    # Summary metrics
    summary: dict = {
        "xgboost": {k: v.to_dict() for k, v in multi_results.items()},
        "logistic": {k: v.to_dict() for k, v in baseline_results.items()},
        "fold_results": fold_results,
    }

    return summary


# ======================================================================
# Orchestrator
# ======================================================================

MODEL_FACTORIES = {
    "XGBoost": make_xgboost,
    "Logistic": make_logistic,
}

TASK_DISPATCH = {
    "feature_info": task_feature_info,
    # "sac_train": task_sac_train,    # P3-1 で追加
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

        results = task_fn(cfg)

        # Save results
        results_dir = Path(cfg.get("output", {}).get("results_dir", "results/v460"))
        if not results_dir.is_absolute():
            results_dir = _PROJECT_ROOT / results_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        out_path = results_dir / f"{entry.run_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Results saved: {out_path}")

        # Determine gate result
        gate_result = _evaluate_gate(gate, results)

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


def _evaluate_gate(gate: str, results: dict) -> str:
    """Quick gate evaluation from results."""
    if "G1" in gate:
        from ztb.metrics.gate_checks import g1_judgment
        fold_results = results.get("fold_results", {})
        if fold_results:
            judgment = g1_judgment(fold_results)
            return "PASS" if judgment["g1_pass"] else "FAIL"
    return "PENDING"


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
