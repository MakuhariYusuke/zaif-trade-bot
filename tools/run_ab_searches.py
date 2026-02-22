#!/usr/bin/env python3
"""
AB Search Runner - Execute comprehensive AB grid searches with reward_components analysis.

This script runs multiple AB searches with different grids and analyzes reward_components
to understand which parameters contribute to better action balance.
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ztb.io.json_io import read_json_object, write_json
from ztb.reporting.services.catalog import (
    extract_action_distribution_from_payload,
    extract_reward_components_from_payload,
    list_training_reports,
    load_training_report,
)
from ztb.trading.environment.components.rewards.utils import RewardUtils
from ztb.utils.safety import ensure_dict, safe_to_float


def run_ab_search(
    template_path: str,
    grid_path: str,
    timesteps: int,
    seeds: int,
    jobs: int,
    objective: str,
    fast_mode: bool = False,
    output_path: Optional[str] = None,
) -> dict[str, object]:
    """Run a single AB parameter search."""
    cmd = [
        sys.executable,
        "tools/ab_param_search.py",
        "--template", template_path,
        "--grid", grid_path,
        "--timesteps", str(timesteps),
        "--seeds", str(seeds),
        "--jobs", str(jobs),
        "--objective", objective,
    ]
    
    if fast_mode:
        cmd.append("--fast-mode")
    
    if output_path:
        cmd.extend(["--out", output_path])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running AB search: {result.stderr}")
        return {"success": False, "error": result.stderr}
    
    # Load results if output path specified
    if output_path and Path(output_path).exists():
        return read_json_object(Path(output_path))
    
    return {"success": True}


def analyze_reward_components(reports_dir: Path) -> dict[str, object]:
    """Analyze reward_components from training reports."""
    reports = list_training_reports(reports_dir=reports_dir)

    with_reward_components = 0
    component_values: dict[str, list[float]] = {}
    best_balanced_config: dict[str, object] | None = None
    best_balance_score = float("inf")
    
    for report_path in reports:
        report = load_training_report(report_path)
        if report is None:
            print(f"Error analyzing {report_path}: could not load JSON")
            continue

        components = extract_reward_components_from_payload(report)
        if not components:
            continue

        with_reward_components += 1
        for key, value in components.items():
            component_values.setdefault(key, []).append(safe_to_float(value, 0.0))

        action_dist = extract_action_distribution_from_payload(report)
        if action_dist:
            buy = safe_to_float(action_dist.get("BUY"), 0.0)
            sell = safe_to_float(action_dist.get("SELL"), 0.0)
            # Use canonical deviation helper (target 50/50 for BUY/SELL)
            balance_score = RewardUtils.calculate_balance_deviation_from_ratios(
                [buy, sell], [0.5, 0.5]
            )
            if balance_score < best_balance_score:
                best_balance_score = balance_score
                best_balanced_config = {
                    "report": report_path.name,
                    "action_distribution": action_dist,
                    "reward_components": components,
                    "config": ensure_dict(report.get("configuration")),
                }

    component_stats: dict[str, dict[str, object]] = {}
    for key, values in component_values.items():
        if values:
            component_stats[key] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }

    return {
        "total_reports": len(reports),
        "with_reward_components": with_reward_components,
        "component_stats": component_stats,
        "best_balanced_config": best_balanced_config,
        "best_balance_score": best_balance_score if best_balanced_config else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive AB searches with reward_components analysis")
    parser.add_argument("--template", required=True, help="Template config path")
    parser.add_argument("--timesteps", type=int, default=5000, help="Timesteps per run")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs")
    parser.add_argument("--objective", default="balance", choices=["balance", "min_sell"])
    parser.add_argument("--fast-mode", action="store_true", help="Enable fast mode")
    parser.add_argument("--grids", nargs="+", help="Grid config paths (default: all in config/ab/)")
    parser.add_argument("--output-dir", default="reports/ab_searches", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine grids to run
    if args.grids:
        grid_paths = [Path(g) for g in args.grids]
    else:
        ab_config_dir = Path("config/ab")
        grid_paths = list(ab_config_dir.glob("ab_grid_*.json"))
    
    print(f"Running {len(grid_paths)} AB searches...")
    
    # Run each grid search
    results: dict[str, dict[str, object]] = {}
    for grid_path in grid_paths:
        grid_name = grid_path.stem
        output_path = output_dir / f"{grid_name}_results.json"
        
        print(f"\n{'='*60}")
        print(f"Running: {grid_name}")
        print(f"{'='*60}")
        
        result = run_ab_search(
            template_path=args.template,
            grid_path=str(grid_path),
            timesteps=args.timesteps,
            seeds=args.seeds,
            jobs=args.jobs,
            objective=args.objective,
            fast_mode=args.fast_mode,
            output_path=str(output_path),
        )
        
        results[grid_name] = result
    
    # Analyze reward_components across all reports
    print(f"\n{'='*60}")
    print("Analyzing reward_components...")
    print(f"{'='*60}")
    
    analysis = analyze_reward_components(Path("reports"))
    
    # Save analysis
    analysis_path = output_dir / "reward_components_analysis.json"
    write_json(analysis_path, analysis, indent=2, ensure_ascii=False)
    
    print(f"\nAnalysis saved to: {analysis_path}")
    
    # Print summary
    print("\nSummary:")
    print(f"  Total reports: {analysis['total_reports']}")
    print(f"  With reward_components: {analysis['with_reward_components']}")
    print(f"  Best balance score: {analysis['best_balance_score']:.4f}")
    
    if analysis["best_balanced_config"]:
        print("\nBest balanced configuration:")
        best = analysis["best_balanced_config"]
        print(f"  Report: {best['report']}")
        print(f"  Action distribution: {best['action_distribution']}")
        print("  Key reward_components:")
        for key, value in best.get("reward_components", {}).items():
            if key in ["balance_penalty", "balance_shaping", "skew_penalty", "action_bonus"]:
                print(f"    {key}: {value:.6f}")


if __name__ == "__main__":
    main()
