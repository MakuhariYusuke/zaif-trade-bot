"""Utility to adapt short-term backtest episodes into analyzer-friendly data."""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Any, Dict


def build_aggregated_result(data: list[Dict[str, Any]], start_ts: datetime) -> Dict[str, Any]:
    if not data:
        raise ValueError("Source dataset must contain at least one episode")

    episode = data[0]
    history = episode.get("portfolio_values") or episode.get("portfolio_history")
    if not history:
        raise ValueError("Episode must provide a portfolio history list")

    timestamps = [
        (start_ts + timedelta(minutes=i)).isoformat() for i in range(len(history))
    ]

    trade_pnls = []
    for prev, curr in zip(history, history[1:]):
        if prev:
            trade_pnls.append((curr - prev) / prev)
        else:
            trade_pnls.append(0.0)

    return {
        "total_steps": episode.get("steps", len(history) - 1),
        "initial_portfolio": float(history[0]),
        "final_portfolio": float(history[-1]),
        "portfolio_history": history,
        "timestamps": timestamps,
        "price_history": history,
        "trade_pnls": trade_pnls,
        "total_return_pct": episode.get("total_return_pct", 0.0),
        "win_rate": episode.get("win_rate", 0.0),
        "initial_btc": 0.0,
        "final_btc": 0.0,
        "btc_holdings": [],
        "actions": [],
    }


def main() -> None:
    parser = ArgumentParser(
        description="Prepare minute-resolution analyzer inputs from a short-term episode file."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("short_term_backtest_results_20251111_130200.json"),
        help="Path to the source short-term episode JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_results/minute_backtest_input.json"),
        help="Target JSON file for analysis",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2025-11-11T00:00:00",
        help="ISO timestamp for the first historical point",
    )

    args = parser.parse_args()
    data = json.loads(args.source.read_text())
    start_ts = datetime.fromisoformat(args.start)
    aggregated = build_aggregated_result(data, start_ts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregated, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
