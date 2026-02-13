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
import json
import logging
import sys
from pathlib import Path

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.v460.lib.config_loader import load_config
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

        # Determine gate result using thresholds from config
        gate_result = _evaluate_gate(gate, results, cfg)

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
    """
    if "G1" in gate:
        from ztb.metrics.gate_checks import g1_judgment
        fold_results = results.get("fold_results", {})
        if fold_results:
            # Load gate thresholds from config
            thresholds_path = _PROJECT_ROOT / "configs/v460/gate_thresholds.yaml"
            try:
                gate_cfg = load_config(str(thresholds_path))
                g1_cfg = gate_cfg.get("g1_info", {})
                alpha = g1_cfg.get("p_alpha", 0.05)
                min_effect = g1_cfg.get("min_cliff_d", 0.33)
            except Exception:
                logger.warning("gate_thresholds.yaml not found, using defaults")
                alpha = 0.05
                min_effect = 0.33

            judgment = g1_judgment(
                fold_results,
                alpha=alpha,
                min_effect=min_effect,
            )
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
