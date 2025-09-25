#!/usr/bin/env python3
"""
Integrated comprehensive evaluation runner.

This script runs the complete evaluation pipeline:
1. Feature evaluation (--evaluate)
2. Correlation analysis (--correlate)
3. Lag correlation analysis (--lag)
4. Benchmark generation (--benchmark)
5. Weekly report generation (--report)

Use --all to run the complete pipeline.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def run_evaluation():
    """Run feature evaluation."""
    print("🔍 Running feature evaluation...")
    from ztb.evaluation.re_evaluate_features import ComprehensiveFeatureReEvaluator

    evaluator = ComprehensiveFeatureReEvaluator()
    results = evaluator.evaluate_all_experimental()

    # Count statuses
    success_count = sum(1 for r in results.values() if isinstance(r, dict) and r.get('status') == 'success')
    error_count = sum(1 for r in results.values() if isinstance(r, dict) and r.get('status') == 'error')
    insufficient_count = sum(1 for r in results.values() if isinstance(r, dict) and r.get('status') == 'insufficient')

    print(f" Evaluation complete: {len(results)} features, {success_count} success, {error_count} errors, {insufficient_count} insufficient")
    return results

def run_correlation_analysis():
    """Run correlation analysis."""
    print(" Running correlation analysis...")
    from ztb.evaluation.re_evaluate_features import ComprehensiveFeatureReEvaluator
    from ztb.analysis.correlation import compute_correlations

    evaluator = ComprehensiveFeatureReEvaluator()
    exp_results = evaluator.evaluate_all_experimental(collect_frames=True)
    frames = exp_results.get('_success_frames', {})

    if not frames:
        print("  No frames collected for correlation analysis")
        return None

    results = compute_correlations(frames)

    # Save results
    reports_dir = PROJECT_ROOT / "python" / "reports"
    reports_dir.mkdir(exist_ok=True)

    if results['pearson'] is not None:
        results['pearson'].to_csv(reports_dir / "correlation_pearson.csv")
        print(f" Pearson correlation saved to {reports_dir / 'correlation_pearson.csv'}")

    if results['spearman'] is not None:
        results['spearman'].to_csv(reports_dir / "correlation_spearman.csv")
        print(f" Spearman correlation saved to {reports_dir / 'correlation_spearman.csv'}")

    return results

def run_lag_correlation_analysis():
    """Run lag correlation analysis."""
    print(" Running lag correlation analysis...")
    from ztb.evaluation.re_evaluate_features import ComprehensiveFeatureReEvaluator
    from ztb.analysis.timeseries import compute_lag_correlations

    evaluator = ComprehensiveFeatureReEvaluator()
    exp_results = evaluator.evaluate_all_experimental(collect_frames=True)
    frames = exp_results.get('_success_frames', {})

    if not frames:
        print("  No frames collected for lag correlation analysis")
        return None

    results = compute_lag_correlations(frames)

    # Save results
    reports_dir = PROJECT_ROOT / "python" / "reports"
    reports_dir.mkdir(exist_ok=True)

    import json
    with open(reports_dir / "lag_correlations.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f" Lag correlations saved to {reports_dir / 'lag_correlations.json'}")
    return results

def run_benchmark():
    """Run benchmark generation."""
    print(" Running benchmark generation...")
    from ztb.evaluation.re_evaluate_features import ComprehensiveFeatureReEvaluator, generate_benchmark_output

    evaluator = ComprehensiveFeatureReEvaluator()
    results = evaluator.evaluate_all_experimental()

    # Generate benchmark
    benchmark_data = generate_benchmark_output(results)

    print(" Benchmark generation complete")
    return benchmark_data

def run_weekly_report():
    """Run weekly report generation."""
    print(" Running weekly report generation...")
    from ztb.evaluation.experimental_weekly_report import ExperimentalWeeklyReporter

    generator = ExperimentalWeeklyReporter()
    report_path = generator.generate_weekly_report()

    print(f" Weekly report generated: {report_path}")
    return report_path

def main():
    parser = argparse.ArgumentParser(description="Integrated comprehensive evaluation runner")
    parser.add_argument('--all', action='store_true', help='Run complete pipeline')
    parser.add_argument('--evaluate', action='store_true', help='Run feature evaluation')
    parser.add_argument('--correlate', action='store_true', help='Run correlation analysis')
    parser.add_argument('--lag', action='store_true', help='Run lag correlation analysis')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark generation')
    parser.add_argument('--report', action='store_true', help='Run weekly report generation')

    args = parser.parse_args()

    # If --all or no specific args, run everything
    if args.all or not any([args.evaluate, args.correlate, args.lag, args.benchmark, args.report]):
        args.evaluate = args.correlate = args.lag = args.benchmark = args.report = True

    start_time = time.time()

    try:
        if args.evaluate:
            run_evaluation()

        if args.correlate:
            run_correlation_analysis()

        if args.lag:
            run_lag_correlation_analysis()

        if args.benchmark:
            run_benchmark()

        if args.report:
            run_weekly_report()

        elapsed = time.time() - start_time
        print(f" Complete pipeline finished in {elapsed:.2f} seconds")

    except Exception as e:
        print(f" Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
