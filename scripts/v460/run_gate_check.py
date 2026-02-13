#!/usr/bin/env python3
"""
v460 Gate Check Runner — G0/G1 閾値照合ユーティリティ.

001# §4.1 / 000# §3 準拠.

Usage:
  python scripts/v460/run_gate_check.py --gate G0 --data-path data/v460/features/btc_jpy_1m_v460_features.parquet
  python scripts/v460/run_gate_check.py --gate G1 --results-path results/v460/g1_results.json
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

from scripts.v460.lib.config_loader import load_gate_thresholds
from scripts.v460.lib.data_loader import check_nan_ratio, compute_data_hash, load_parquet
from scripts.v460.lib.manifest import ManifestWriter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================
# G0-data
# ======================================================================

def run_g0(
    data_path: str,
    expected_hash: str | None = None,
    thresholds: dict | None = None,
) -> dict:
    """G0-data チェック.

    000# §3.1:
      - データハッシュ一致
      - 特徴量カラム数 ≥ 4
      - NaN 比率 ≤ 1%
      - manifest.jsonl 記録 (存在チェックのみ)
    """
    if thresholds is None:
        thresholds = load_gate_thresholds().get("g0_data", {})

    min_cols = thresholds.get("min_feature_columns", 4)
    max_nan = thresholds.get("max_nan_ratio", 0.01)

    results: dict = {"gate": "G0-data", "checks": {}}

    # Hash
    actual_hash = compute_data_hash(data_path)
    if expected_hash:
        hash_ok = actual_hash == expected_hash
    else:
        hash_ok = True  # No expected hash → skip (record for manifest)
        logger.warning("No expected hash provided. Recording actual hash only.")
    results["checks"]["data_hash"] = {
        "actual": actual_hash[:16],
        "expected": (expected_hash or "N/A")[:16],
        "pass": hash_ok,
    }

    # Column count
    df = load_parquet(data_path)
    n_cols = len(df.columns)
    results["checks"]["column_count"] = {
        "actual": n_cols,
        "threshold": min_cols,
        "pass": n_cols >= min_cols,
    }

    # NaN ratio
    nan_info = check_nan_ratio(df, max_nan)
    results["checks"]["nan_ratio"] = nan_info

    # Manifest existence
    mw = ManifestWriter()
    manifest_exists = mw.path.exists()
    results["checks"]["manifest_exists"] = {
        "path": str(mw.path),
        "pass": manifest_exists,
    }

    # Overall
    all_pass = all(c["pass"] for c in results["checks"].values())
    results["gate_result"] = "PASS" if all_pass else "FAIL"

    return results


# ======================================================================
# G1-info (判定のみ — 実験実行は run_experiment.py)
# ======================================================================

def run_g1_judgment(results_path: str, thresholds: dict | None = None) -> dict:
    """G1 judgment from pre-computed experiment results.

    Expects results JSON with fold_results structure per §5.3.
    """
    if thresholds is None:
        thresholds = load_gate_thresholds().get("g1_info", {})

    with open(results_path, "r", encoding="utf-8") as f:
        exp_results = json.load(f)

    # Import gate_checks (will be created in P1-3/4)
    from ztb.metrics.gate_checks import g1_judgment

    judgment = g1_judgment(
        fold_results=exp_results.get("fold_results", {}),
        alpha=thresholds.get("p_alpha", 0.05),
        min_effect=thresholds.get("min_cliff_d", 0.33),
    )

    return {
        "gate": "G1-info",
        "gate_result": "PASS" if judgment["g1_pass"] else "FAIL",
        "details": judgment,
    }


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="v460 Gate Check")
    parser.add_argument("--gate", required=True, choices=["G0", "G1"],
                        help="Gate to check")
    parser.add_argument("--data-path", default=None,
                        help="Path to data file (G0)")
    parser.add_argument("--expected-hash", default=None,
                        help="Expected SHA-256 hash (G0)")
    parser.add_argument("--results-path", default=None,
                        help="Path to G1 results JSON")
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    if args.gate == "G0":
        if not args.data_path:
            parser.error("--data-path required for G0")
        result = run_g0(args.data_path, args.expected_hash)
    elif args.gate == "G1":
        if not args.results_path:
            parser.error("--results-path required for G1")
        result = run_g1_judgment(args.results_path)
    else:
        parser.error(f"Unknown gate: {args.gate}")
        return

    # Output
    out_str = json.dumps(result, indent=2, ensure_ascii=False)
    print(out_str)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_str)
        logger.info(f"Saved: {args.output}")

    # Exit code
    sys.exit(0 if result["gate_result"] == "PASS" else 1)


if __name__ == "__main__":
    main()
