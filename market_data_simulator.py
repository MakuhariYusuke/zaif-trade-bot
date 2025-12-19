"""Minimal stub for tests that import MarketDataSimulator directly.

This provides a tiny in-memory simulator sufficient for import-time and
lightweight integration checks during test collection.
"""
from typing import List, Dict, Any


class MarketDataSimulator:
    def __init__(self, symbols: List[str] = None, start_ts=None, end_ts=None):
        self.symbols = symbols or ["BTCJPY"]
        self.start_ts = start_ts
        self.end_ts = end_ts

    def generate(self, n: int = 100) -> List[Dict[str, Any]]:
        # Return simple sequential records to satisfy tests that only require
        # the class to exist and produce iterable-like output.
        return [{"timestamp": i, "price": 1.0 + i * 0.01} for i in range(n)]
