#!/usr/bin/env python3
"""Inspect guidance signal samples from a backtest result file."""

from ztb.io.json_io import read_json_object
from ztb.utils.safety import ensure_dict, safe_to_int, safe_to_float


def main() -> int:
    payload = read_json_object("signal_guidance_backtest_results_20251112_135639.json")
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        print("No results found")
        return 1

    first_result = ensure_dict(results[0])
    signals_obj = first_result.get("guidance_signals")
    signals = signals_obj if isinstance(signals_obj, list) else []

    print("Total signals:", len(signals))
    print("Sample signals:")
    for i in range(min(20, len(signals))):
        signal = ensure_dict(signals[i])
        print(
            "  step",
            safe_to_int(signal.get("step"), -1),
            ": score",
            f"{safe_to_float(signal.get('guidance_score'), 0.0):.6f}",
            "-> guidance_action",
            signal.get("guidance_action", "N/A"),
            "(orig",
            signal.get("original_action", "N/A"),
            ")",
        )

    actions = [
        str(ensure_dict(signal).get("guidance_action", "UNKNOWN")) for signal in signals
    ]
    unique_actions = sorted(set(actions))
    print("Unique guidance_actions:", unique_actions)
    for action in unique_actions:
        count = actions.count(action)
        print(f"  Action {action}: {count} times")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
