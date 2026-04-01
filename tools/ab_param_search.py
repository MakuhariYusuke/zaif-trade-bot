#!/usr/bin/env python3
"""
Simple hyperparameter grid search for AB tests.

This script generates a grid of configs from a template and runs AB evaluation
for each combination using `tools/ab_test_runner` parallel runner.

Example:
  python tools/ab_param_search.py --template config/v447/sac_v447_1m_multiframe_config.json \
    --grid config/ab_grid.json --seeds 3 --jobs 2 --objective balance

Grid format (JSON):
{
  "training.sac_hyperparameters.learning_rate": [0.0003, 0.0001],
  "environment.behavioral_penalty.balance_penalty": [0.01, 0.05]
}

Objective:
  balance: minimize |BUY - SELL| (reduce directional skew)
  min_sell: minimize SELL fraction

"""
import argparse
import itertools
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

# Ensure tools package path is importable when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ztb.utils.parallel_experiments import run_parallel_experiments

import importlib.util

module_path = Path(__file__).parent / "ab_test_runner.py"
spec = importlib.util.spec_from_file_location("tools.ab_test_runner", str(module_path))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load tools.ab_test_runner from {module_path}")
ab_mod = importlib.util.module_from_spec(spec)
sys.modules["tools.ab_test_runner"] = ab_mod
spec.loader.exec_module(ab_mod)
ABTrainingExperiment = ab_mod.ABTrainingExperiment
import uuid

from ztb.utils.report_utils import extract_action_distribution, find_reports_for_model
from ztb.trading.environment.components.rewards.utils import RewardUtils


def set_nested(dictionary: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    d = dictionary
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def generate_grid(
    template: Dict[str, Any], grid: Dict[str, List[Any]]
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combinations = list(itertools.product(*values))

    results = []
    for comb in combinations:
        conf_copy = json.loads(json.dumps(template))  # deep copy
        param_values = {}
        for k, v in zip(keys, comb):
            set_nested(conf_copy, k, v)
            param_values[k] = v
        results.append((conf_copy, param_values))
    return results


def score_distribution(dist: Dict[str, float], objective: str) -> float:
    buy = dist.get("BUY", 0.0)
    sell = dist.get("SELL", 0.0)
    if objective == "min_sell":
        return -sell
    # balance: use canonical deviation helper (target 50/50 for BUY/SELL)
    return float(
        -RewardUtils.calculate_balance_deviation_from_ratios([buy, sell], [0.5, 0.5])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--method", choices=["grid", "bayesian", "random"], default="grid"
    )
    parser.add_argument(
        "--objective", choices=["balance", "min_sell"], default="balance"
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Use fast-mode defaults to speed up AB jobs (minimal features, skip quality filtering)",
    )
    parser.add_argument(
        "--out", default="reports/ab_searches/ab_param_search_summary.json"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=2000,
        help="Total timesteps per run (passes to ab_test_runner)",
    )
    args = parser.parse_args()

    template = json.loads(Path(args.template).read_text(encoding="utf-8"))
    grid = json.loads(Path(args.grid).read_text(encoding="utf-8"))

    combos = generate_grid(template, grid)
    print(f"Generated {len(combos)} configs from grid")

    tmp_dir = Path("config/ab_search")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Build tasks for seeds
    tasks = []
    cfg_paths = []
    for i, (conf, params) in enumerate(combos):
        # Ensure unique model_name per grid combination so reports don't mix
        base_name = None
        try:
            base_name = conf["training"]["model_name"]
        except Exception:
            base_name = None
        if base_name is None:
            base_name = f"absearch_{i}"
        # ensure training map exists
        conf.setdefault("training", {})
        # append a short unique suffix and set ab_tag so reports are clearly labelled
        conf["training"]["model_name"] = f"{base_name}__ab_{i}"
        conf["ab_tag"] = f"ab_balance_small_{i}"
        name = "absearch_" + "_".join(
            [f"{k.split('.')[-1]}={str(v)}" for k, v in params.items()]
        )
        cfg_path = tmp_dir / f"{name}_{i}.json"
        # Apply fast-mode defaults if requested - reduce feature set and disable expensive quality filtering
        if args.fast_mode:
            conf.setdefault("features", {})
            conf["features"]["feature_set"] = "minimal"
            conf["features"]["skip_quality_filtering"] = True
            # also reduce dataset rows if not specified
            conf.setdefault("data_rows_limit", 5000)
            # limit features globally
            conf.setdefault("max_features", 128)

        cfg_path.write_text(json.dumps(conf, ensure_ascii=False, indent=2))
        cfg_paths.append((cfg_path, params))
        if args.seeds > 0:
            for seed in range(1, args.seeds + 1):
                tasks.append(
                    {
                        "config_path": cfg_path.as_posix(),
                        "seed": seed,
                        "timesteps": args.timesteps,
                        "fast_mode": args.fast_mode,
                    }
                )
    # --timesteps handled earlier in parser

    # Run AB runs (only if not using the UnifiedOptimizer for grid search)
    use_unified_optimizer = True

    if not use_unified_optimizer:
        if args.jobs > 1 and tasks:
            run_parallel_experiments(ABTrainingExperiment, tasks, max_workers=args.jobs)
        else:
            # run sequentially using same helper
            run_parallel_experiments(ABTrainingExperiment, tasks, max_workers=1)

    # Use UnifiedOptimizer for grid search if configured
    if use_unified_optimizer:
        # 687#: config generation path のみを使うテストで torch 依存を踏まないようにする
        from ztb.training.unified_optimizer import OptimizationConfig, UnifiedOptimizer

        # convert grid JSON to search_space format used by UnifiedOptimizer.GridOptimizer
        optimizer_config = OptimizationConfig()
        optimizer = UnifiedOptimizer(optimizer_config)

        search_space = {}
        for key, values in grid.items():
            # If grid key is 'param.name' and values list is list of values -> categorical
            if isinstance(values, list):
                search_space[key] = {"type": "categorical", "choices": values}
            else:
                # else assume low/high
                search_space[key] = values

        def objective_wrapper(params: dict[str, object]) -> float:
            # params is dict of param_name: value; convert keys into config
            run_cfg = json.loads(json.dumps(template))
            for k, v in params.items():
                set_nested(run_cfg, k, v)

            # save to temp config and run with seeds
            # attach unique suffix to model name so we don't mix reports
            base = None
            try:
                base = run_cfg["training"]["model_name"]
            except Exception:
                base = None
            if base is None:
                base = "ab_temp"
            run_cfg.setdefault("training", {})
            suffix = uuid.uuid4().hex[:8]
            run_cfg["training"]["model_name"] = f"{base}__ab_{suffix}"
            run_cfg["ab_tag"] = f"ab_balance_search_{suffix}"

            # save to unique temp config for this trial
            tmp = Path("config") / f"ab_search_temp_{suffix}.json"
            if args.fast_mode:
                run_cfg.setdefault("features", {})
                run_cfg["features"]["feature_set"] = "minimal"
                run_cfg["features"]["skip_quality_filtering"] = True
                run_cfg.setdefault("data_rows_limit", 5000)
            tmp.write_text(json.dumps(run_cfg, ensure_ascii=False, indent=2))
            dists = []
            if args.seeds > 0:
                # build tasks for seeds and run with parallel_experiments utility
                tasks_local = []
                for seed in range(1, args.seeds + 1):
                    tasks_local.append(
                        {
                            "config_path": tmp.as_posix(),
                            "seed": seed,
                            "timesteps": args.timesteps,
                            "fast_mode": args.fast_mode,
                        }
                    )

                run_parallel_experiments(
                    ABTrainingExperiment, tasks_local, max_workers=args.jobs
                )

                for seed in range(1, args.seeds + 1):
                    reports = find_reports_for_model(run_cfg["training"]["model_name"])
                    if reports:
                        dists.append(extract_action_distribution(reports[-1]))
            else:
                # No seeds, use dummy 0 distribution for test
                dists.append({"HOLD": 0.0, "BUY": 0.0, "SELL": 0.0})

            # average
            avg = {"HOLD": 0.0, "BUY": 0.0, "SELL": 0.0}
            if dists:
                for dd in dists:
                    for k in avg.keys():
                        avg[k] += float(dd.get(k, 0.0))
                for k in avg.keys():
                    avg[k] /= len(dists)

            score = score_distribution(avg, args.objective)
            return float(score)

        # run grid optimization via unified optimizer
        print("Running search via UnifiedOptimizer (grid)")
        param_space = {}
        for k, v in grid.items():
            if isinstance(v, list):
                param_space[k] = {"type": "categorical", "choices": v}
            else:
                param_space[k] = v

        res = optimizer.optimize_hyperparameters(
            objective_wrapper, param_space, method=args.method
        )
        # Save optimization result summary
        Path(args.out).write_text(
            json.dumps(
                {"best": res.best_params, "score": res.best_score},
                ensure_ascii=False,
                indent=2,
            )
        )
        print(f"Saved unified optimizer result to {args.out}")
        # also include full results
        return
    all_records = []
    for cfg_path, params in cfg_paths:
        from ztb.utils.config_utils import read_model_name_from_config

        model_name = read_model_name_from_config(cfg_path)
        reports = find_reports_for_model(model_name)
        all_dists = []
        for rp in reports:
            all_dists.append(extract_action_distribution(rp))

        # average dist
        avg = {"HOLD": 0.0, "BUY": 0.0, "SELL": 0.0}
        if all_dists:
            for d in all_dists:
                for k in avg.keys():
                    avg[k] += float(d.get(k, 0.0))
            for k in avg.keys():
                avg[k] /= len(all_dists)

        score = score_distribution(avg, args.objective)
        all_records.append({"params": params, "avg_distribution": avg, "score": score})

    # sort best
    all_records.sort(key=lambda x: cast(float, x["score"]), reverse=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(all_records, ensure_ascii=False, indent=2))
    print(f"Wrote summary to {args.out}")


if __name__ == "__main__":
    main()
