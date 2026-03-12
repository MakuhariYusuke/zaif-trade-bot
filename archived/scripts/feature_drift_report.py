#!/usr/bin/env python3
"""
Feature drift detection report generator.

Detects distribution shifts between training and evaluation datasets using:
- PSI (Population Stability Index): Measures overall distribution shift
- KS (Kolmogorov-Smirnov) test: Statistical test for distribution difference

Usage:
    python scripts/feature_drift_report.py \\
        --train-features data/train_features.parquet \\
        --eval-features data/eval_features.parquet \\
        --output artifacts/drift/

Thresholds:
    - PSI > 0.2: Significant drift
    - KS p-value < 0.01: Significantly different distributions

Exit Codes:
    0: No drift detected
    1: Drift detected (fails CI)
    2: Error in processing
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.utils.data_utils import load_csv_data_optimized
from ztb.utils.drift_detection import (
    detect_drift_all_features,
    generate_drift_report_html,
)


def main():
    parser = argparse.ArgumentParser(
        description="Feature drift detection between train and eval datasets"
    )
    parser.add_argument(
        "--train-features",
        type=str,
        required=True,
        help="Path to training features (parquet/csv)",
    )
    parser.add_argument(
        "--eval-features",
        type=str,
        required=True,
        help="Path to evaluation features (parquet/csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/drift",
        help="Output directory for drift reports",
    )
    parser.add_argument(
        "--psi-threshold",
        type=float,
        default=0.2,
        help="PSI threshold for drift detection (default: 0.2)",
    )
    parser.add_argument(
        "--ks-p-threshold",
        type=float,
        default=0.01,
        help="KS p-value threshold for drift detection (default: 0.01)",
    )
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit with code 1 if drift detected (for CI)",
    )

    args = parser.parse_args()

    # Load data
    train_path = Path(args.train_features)
    eval_path = Path(args.eval_features)

    if not train_path.exists():
        print(f"Error: Training features not found: {train_path}")
        sys.exit(2)

    if not eval_path.exists():
        print(f"Error: Evaluation features not found: {eval_path}")
        sys.exit(2)

    print(f"Loading training features from: {train_path}")
    if train_path.suffix == ".parquet":
        train_df = pd.read_parquet(train_path)
    elif train_path.suffix == ".csv":
        train_df = load_csv_data_optimized(train_path)
    else:
        print(f"Error: Unsupported file format: {train_path.suffix}")
        sys.exit(2)

    print(f"Loading evaluation features from: {eval_path}")
    if eval_path.suffix == ".parquet":
        eval_df = pd.read_parquet(eval_path)
    elif eval_path.suffix == ".csv":
        eval_df = load_csv_data_optimized(eval_path)
    else:
        print(f"Error: Unsupported file format: {eval_path.suffix}")
        sys.exit(2)

    print(f"\nTrain features: {train_df.shape}")
    print(f"Eval features: {eval_df.shape}")

    # Detect drift
    print(
        f"\nDetecting drift (PSI>{args.psi_threshold} or KS p<{args.ks_p_threshold})..."
    )
    drift_df = detect_drift_all_features(
        train_df,
        eval_df,
        psi_threshold=args.psi_threshold,
        ks_p_threshold=args.ks_p_threshold,
    )

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    psi_csv = output_dir / "psi.csv"
    ks_csv = output_dir / "ks.csv"
    summary_csv = output_dir / "summary.csv"
    html_report = output_dir / "drift_report.html"

    # PSI results
    psi_results = drift_df[["feature_name", "psi", "psi_drift"]].copy()
    psi_results.to_csv(psi_csv, index=False)
    print(f"PSI results saved to: {psi_csv}")

    # KS results
    ks_results = drift_df[
        ["feature_name", "ks_statistic", "ks_p_value", "ks_drift"]
    ].copy()
    ks_results.to_csv(ks_csv, index=False)
    print(f"KS results saved to: {ks_csv}")

    # Full summary
    drift_df.to_csv(summary_csv, index=False)
    print(f"Full summary saved to: {summary_csv}")

    # HTML report
    generate_drift_report_html(drift_df, html_report)
    print(f"HTML report saved to: {html_report}")

    # Print summary
    total_features = len(drift_df)
    drift_count = drift_df["drift_detected"].sum()
    drift_pct = (drift_count / total_features * 100) if total_features > 0 else 0

    print(f"\n{'='*60}")
    print("DRIFT DETECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total features: {total_features}")
    print(f"Features with drift: {drift_count} ({drift_pct:.1f}%)")
    print(f"PSI threshold: {args.psi_threshold}")
    print(f"KS p-value threshold: {args.ks_p_threshold}")

    if drift_count > 0:
        print(f"\n⚠️  DRIFT DETECTED in {drift_count} features:")
        drifted_features = drift_df[drift_df["drift_detected"]]
        for _, row in drifted_features.iterrows():
            reasons = []
            if row["psi_drift"]:
                reasons.append(f"PSI={row['psi']:.4f}")
            if row["ks_drift"]:
                reasons.append(f"KS p={row['ks_p_value']:.4f}")
            print(f"  - {row['feature_name']}: {', '.join(reasons)}")

        print(f"\n{'='*60}")

        if args.fail_on_drift:
            print("❌ FAIL: Drift detected (exit code 1)")
            sys.exit(1)
        else:
            print("⚠️  WARNING: Drift detected but --fail-on-drift not set")
            sys.exit(0)
    else:
        print("\n✅ PASS: No drift detected")
        print(f"{'='*60}")
        sys.exit(0)


if __name__ == "__main__":
    main()
