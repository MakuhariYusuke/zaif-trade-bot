from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def make_sample_trade_records(*, extended: bool = False) -> list[dict[str, float]]:
    """Build deterministic trade records for optimizer-related tests."""
    trades = [
        {"pnl": 100.0, "confidence": 0.80, "entry_price": 100.0},
        {"pnl": -50.0, "confidence": 0.60, "entry_price": 105.0},
        {"pnl": 150.0, "confidence": 0.90, "entry_price": 102.0},
        {"pnl": -30.0, "confidence": 0.70, "entry_price": 108.0},
        {"pnl": 200.0, "confidence": 0.85, "entry_price": 110.0},
    ]
    if not extended:
        return trades
    return trades + [
        {"pnl": 80.0, "confidence": 0.75, "entry_price": 115.0},
        {"pnl": -70.0, "confidence": 0.65, "entry_price": 118.0},
        {"pnl": 120.0, "confidence": 0.82, "entry_price": 120.0},
        {"pnl": -40.0, "confidence": 0.68, "entry_price": 122.0},
        {"pnl": 180.0, "confidence": 0.88, "entry_price": 125.0},
    ]


def make_lr_batch_search_space() -> dict[str, dict[str, Any]]:
    return {
        "learning_rate": {"type": "float", "low": 0.0001, "high": 0.1},
        "batch_size": {"type": "int", "low": 16, "high": 128},
    }


def make_lr_batch_objective(
    *, noise_scale: float = 0.0, seed: int = 42
) -> Callable[[dict[str, Any]], float]:
    rng = np.random.default_rng(seed)

    def objective(params: dict[str, Any]) -> float:
        learning_rate = float(params.get("learning_rate", 0.001))
        batch_size = float(params.get("batch_size", 32))
        score = -((learning_rate - 0.01) ** 2) - ((batch_size - 64.0) ** 2) / 10000.0
        if noise_scale > 0:
            score += float(rng.normal(0.0, noise_scale))
        return score

    return objective


def make_momentum_search_spaces() -> dict[str, dict[str, dict[str, float | str]]]:
    shared_space = {"momentum": {"type": "float", "low": 0.1, "high": 0.9}}
    return {"1m": dict(shared_space), "5m": dict(shared_space)}
