#!/usr/bin/env python3
"""Unit tests for regime evaluation utilities."""

import unittest
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from regime_evaluation import RegimeEvaluator


class TestRegimeEvaluatorActionDistribution(unittest.TestCase):
    """Tests for action distribution calculations in RegimeEvaluator."""

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.evaluator: Optional[RegimeEvaluator] = None
        self.returns: Optional[pd.Series] = None
        self.regime_labels: Optional[NDArray[np.str_]] = None

    def setUp(self) -> None:
        self.evaluator = RegimeEvaluator()
        index = pd.date_range("2024-01-01", periods=3, freq="1h")
        self.returns = pd.Series([0.01, -0.02, 0.005], index=index)
        self.regime_labels = np.array(["range", "range", "range"], dtype=np.str_)

    def test_action_distribution_respects_action_mapping(self) -> None:
        """BUY=1, SELL=2, HOLD=0 mapping must be preserved when aggregating distributions."""
        actions = np.array([0, 1, 2], dtype=int)
        assert self.evaluator is not None
        assert self.returns is not None
        assert self.regime_labels is not None

        metrics = self.evaluator.calculate_regime_metrics(
            self.returns,
            self.regime_labels,
            regime="range",
            actions=actions,
        )

        self.assertAlmostEqual(metrics.action_distribution["HOLD"], 100 / 3, places=6)
        self.assertAlmostEqual(metrics.action_distribution["BUY"], 100 / 3, places=6)
        self.assertAlmostEqual(metrics.action_distribution["SELL"], 100 / 3, places=6)

    def test_action_distribution_handles_length_mismatch(self) -> None:
        """If actions length mismatches labels, distribution should remain zeroed."""
        actions = np.array([1, 2, 0], dtype=int)
        assert self.evaluator is not None
        assert self.returns is not None
        assert self.regime_labels is not None

        metrics = self.evaluator.calculate_regime_metrics(
            self.returns,
            self.regime_labels,
            regime="trend",
            actions=actions,
        )

        self.assertEqual(metrics.action_distribution["HOLD"], 0.0)
        self.assertEqual(metrics.action_distribution["BUY"], 0.0)
        self.assertEqual(metrics.action_distribution["SELL"], 0.0)


if __name__ == "__main__":
    unittest.main()
