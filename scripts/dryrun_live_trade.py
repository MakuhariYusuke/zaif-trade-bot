#!/usr/bin/env python3
"""
Dry-run script for modularized live trader.
"""

import argparse
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ztb.trading.live_trader.live_trader import LiveTrader


def main():
    parser = argparse.ArgumentParser(description="Dry-run modularized live trader")
    parser.add_argument("--model-path", required=True, help="Path to the model file")
    parser.add_argument("--duration", type=float, default=0.05, help="Duration in hours (default: 3 minutes)")
    parser.add_argument("--config", default="trade-config.json", help="Config file path")

    args = parser.parse_args()

    # Create live trader instance
    print("Creating LiveTrader instance...")
    trader = LiveTrader(
        model_path=args.model_path,
        dry_run=True
    )
    print("LiveTrader instance created successfully")

    # Run trading loop
    trader.run_trading_loop(duration_hours=args.duration)


if __name__ == "__main__":
    main()