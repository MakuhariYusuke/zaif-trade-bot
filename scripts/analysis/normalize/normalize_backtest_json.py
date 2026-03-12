#!/usr/bin/env python3
"""Normalize numeric fields in a backtest results file."""
import csv
import json
from pathlib import Path

RESULTS_PATH = Path("backtest_results_sac_v446.json")
NUMERIC_FIELDS = ["portfolio_history", "btc_holdings", "price_history"]
DATA_PATH = Path("data/btc_jpy_real_dataset.csv")


def load_timestamps_from_dataset(length: int) -> list[str]:
    if DATA_PATH.exists():
        with DATA_PATH.open() as fh:
            reader = csv.DictReader(fh)
            return [row["timestamp"] for _, row in zip(range(length), reader)]
    return [str(i) for i in range(length)]


def normalize_values(data: dict) -> dict:
    for field in NUMERIC_FIELDS:
        values = data.get(field)
        if values is None:
            continue
        data[field] = [float(v) for v in values]

    if "actions" in data:
        data["actions"] = [int(v) for v in data["actions"]]

    if "timestamps" in data:
        current_length = len(data.get("timestamps", []))
        data["timestamps"] = load_timestamps_from_dataset(current_length)

    return data


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"{RESULTS_PATH} not found")

    data = json.loads(RESULTS_PATH.read_text())
    normalized = normalize_values(data)
    RESULTS_PATH.write_text(json.dumps(normalized, indent=2))
    print(f"Normalized {RESULTS_PATH}")


if __name__ == "__main__":
    main()
