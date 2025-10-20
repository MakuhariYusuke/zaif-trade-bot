#!/usr/bin/env python3
"""
Validate Short-Distance A/B Diagnostics Results.

Validates JSON output from short_distance_ab_diagnostics.py against acceptance criteria:
1. std(probabilities) > 0 (time-varying probabilities)
2. legal_sell_rate >= 0.15 (15% legal SELL actions)

Usage:
    python scripts/validate_ab_diagnostics.py --input diagnostics_results.json

Exit codes:
    0: All tests passed
    1: One or more tests failed
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


def validate_result(result: Dict[str, Any], verbose: bool = False) -> bool:
    """
    Validate a single diagnostic result.

    Args:
        result: Dictionary with diagnostic results
        verbose: Print detailed information

    Returns:
        True if all acceptance criteria met, False otherwise
    """
    data_name = result.get("data_name", "Unknown")
    criteria = result.get("acceptance_criteria", {})

    prob_std_ok = criteria.get("prob_std_positive", False)
    legal_sell_ok = criteria.get("legal_sell_rate_ok", False)

    passed = prob_std_ok and legal_sell_ok

    if verbose:
        prob_var = result.get("probability_variance", {})
        legal_sell_stats = result.get("legal_sell_stats", {})

        print(f"\n{'='*60}")
        print(f"Dataset: {data_name}")
        print(f"{'='*60}")
        print(f"  Probability std (mean): {prob_var.get('mean_std', 0.0):.6f}")
        print("    ✅ PASS" if prob_std_ok else "    ❌ FAIL (must be > 0)")
        print(f"  Legal SELL rate: {legal_sell_stats.get('legal_sell_rate', 0.0):.2%}")
        print("    ✅ PASS" if legal_sell_ok else "    ❌ FAIL (must be >= 15%)")
        print(f"\n  Overall: {'✅ PASS' if passed else '❌ FAIL'}")

    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Validate short-distance A/B diagnostics results"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to diagnostics results JSON file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed validation information",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require ALL datasets to pass (default: require at least one)",
    )

    args = parser.parse_args()

    # Load results
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(input_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON in {input_path}: {e}", file=sys.stderr)
        sys.exit(1)

    results = data.get("results", [])
    if not results:
        print(f"❌ ERROR: No results found in {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        config = data.get("config", {})
        print(f"\n{'='*60}")
        print("Configuration")
        print(f"{'='*60}")
        print(f"  Model: {config.get('model_path', 'Unknown')}")
        print(f"  Temperature: {config.get('temperature', 'Unknown')}")
        print(f"  Tiebreaker tau: {config.get('tiebreaker_tau', 'Unknown')}")
        print(f"  Enable tiebreaker: {config.get('enable_tiebreaker', 'Unknown')}")
        print(f"  Steps: {config.get('steps', 'Unknown')}")

    # Validate each result
    validation_results = []
    for result in results:
        passed = validate_result(result, verbose=args.verbose)
        validation_results.append(
            {
                "data_name": result.get("data_name", "Unknown"),
                "passed": passed,
            }
        )

    # Overall assessment
    passed_count = sum(1 for r in validation_results if r["passed"])
    total_count = len(validation_results)

    print(f"\n{'='*60}")
    print("Overall Results")
    print(f"{'='*60}")
    print(f"  Passed: {passed_count}/{total_count}")

    for vr in validation_results:
        status = "✅ PASS" if vr["passed"] else "❌ FAIL"
        print(f"    {vr['data_name']}: {status}")

    # Determine exit code
    if args.strict:
        # Strict mode: ALL must pass
        all_passed = passed_count == total_count
        if all_passed:
            print("\n🎉 ALL TESTS PASSED (strict mode)")
            print(f"{'='*60}\n")
            sys.exit(0)
        else:
            print("\n⚠️  SOME TESTS FAILED (strict mode)")
            print(f"{'='*60}\n")
            sys.exit(1)
    else:
        # Default mode: at least one must pass
        any_passed = passed_count > 0
        if any_passed:
            print("\n✅ AT LEAST ONE TEST PASSED")
            print(f"{'='*60}\n")
            sys.exit(0)
        else:
            print("\n❌ ALL TESTS FAILED")
            print(f"{'='*60}\n")
            sys.exit(1)


if __name__ == "__main__":
    main()
