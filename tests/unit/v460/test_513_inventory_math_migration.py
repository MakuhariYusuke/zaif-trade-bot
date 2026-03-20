from __future__ import annotations

import math
from collections import deque

import pytest

from ztb.trading.pricing.inventory_math import (
    decayed_inventory_imbalance,
    update_inventory_counters,
)


class TestInventoryMathMigration:
    def test_update_inventory_counters_balanced_and_eviction(self) -> None:
        history: deque[str] = deque(maxlen=3)
        buy_count = 0

        buy_count, imbalance = update_inventory_counters(history, buy_count, "buy")
        assert list(history) == ["buy"]
        assert buy_count == 1
        assert imbalance == pytest.approx(1.0)

        buy_count, imbalance = update_inventory_counters(history, buy_count, "sell")
        assert list(history) == ["buy", "sell"]
        assert buy_count == 1
        assert imbalance == pytest.approx(0.0)

        buy_count, imbalance = update_inventory_counters(history, buy_count, "buy")
        assert list(history) == ["buy", "sell", "buy"]
        assert buy_count == 2
        assert imbalance == pytest.approx((2 * 2 - 3) / 3)

        buy_count, imbalance = update_inventory_counters(history, buy_count, "sell")
        assert list(history) == ["sell", "buy", "sell"]
        assert buy_count == 1
        assert imbalance == pytest.approx((2 * 1 - 3) / 3)

    def test_decayed_inventory_imbalance_matches_exp_decay(self) -> None:
        result = decayed_inventory_imbalance(
            1.0,
            last_update_time=100.0,
            tau_sec=60.0,
            now=160.0,
        )
        assert result == pytest.approx(math.exp(-1.0), rel=1e-6)

    def test_decayed_inventory_imbalance_returns_raw_for_disabled_tau(self) -> None:
        assert decayed_inventory_imbalance(
            0.4,
            last_update_time=100.0,
            tau_sec=0.0,
            now=160.0,
        ) == pytest.approx(0.4)
