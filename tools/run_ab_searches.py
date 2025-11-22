#!/usr/bin/env python3
"""
AB Search Runner - Execute comprehensive AB grid searches with reward_components analysis.

This script runs multiple AB searches with different grids and analyzes reward_components
to understand which parameters contribute to better action balance.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any


def run_ab_search(
    template_path: str,
    grid_path: str,
    timesteps: int,
    seeds: int,
    jobs: int,
    objective: str,
    fast_mode: bool = False,
    output_path: str = None,
) -> Dict[str, Any]:
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
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    return {"success": True}


def analyze_reward_components(reports_dir: Path) -> Dict[str, Any]:
    """Analyze reward_components from training reports."""
    reports = list(reports_dir.glob("training_report_*.json"))
    
    analysis = {
        "total_reports": len(reports),
        "with_reward_components": 0,
        "component_stats": {},
        "best_balanced_config": None,
        "best_balance_score": float("inf"),
    }
    
    for report_path in reports:
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            
            # Extract reward_components
            components = report.get("reward_components")
            if not components:
                components = report.get("training_stats", {}).get("reward_components")
            
            if components:
                analysis["with_reward_components"] += 1
                
                # Aggregate component statistics
                for key, value in components.items():
                    if key not in analysis["component_stats"]:
                        analysis["component_stats"][key] = []
                    analysis["component_stats"][key].append(float(value))
                
                # Calculate balance score
                action_dist = report.get("training_stats", {}).get("action_distribution", {})
                if action_dist:
                    buy = action_dist.get("BUY", 0)
                    sell = action_dist.get("SELL", 0)
                    balance_score = abs(buy - sell)
                    
                    if balance_score < analysis["best_balance_score"]:
                        analysis["best_balance_score"] = balance_score
                        analysis["best_balanced_config"] = {
                            "report": report_path.name,
                            "action_distribution": action_dist,
                            "reward_components": components,
                            "config": report.get("configuration", {})
                        }
        
        except Exception as e:
            print(f"Error analyzing {report_path}: {e}")
    
    # Calculate averages
    for key, values in analysis["component_stats"].items():
        if values:
            analysis["component_stats"][key] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }
    
    return analysis


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
    results = {}
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
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"\nAnalysis saved to: {analysis_path}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total reports: {analysis['total_reports']}")
    print(f"  With reward_components: {analysis['with_reward_components']}")
    print(f"  Best balance score: {analysis['best_balance_score']:.4f}")
    
    if analysis["best_balanced_config"]:
        print(f"\nBest balanced configuration:")
        best = analysis["best_balanced_config"]
        print(f"  Report: {best['report']}")
        print(f"  Action distribution: {best['action_distribution']}")
        print(f"  Key reward_components:")
        for key, value in best.get("reward_components", {}).items():
            if key in ["balance_penalty", "balance_shaping", "skew_penalty", "action_bonus"]:
                print(f"    {key}: {value:.6f}")


if __name__ == "__main__":
    main()
