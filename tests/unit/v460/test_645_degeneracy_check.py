"""645# Skip Gate model degeneracy detection tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ztb.ml.skip_gate import SkipGate, SkipGateConfig


class _ConstantPipeline:
    """常に同じ値を返す疑似 Pipeline."""

    def __init__(self, value: float) -> None:
        self._value = value

    def predict(self, x: object) -> np.ndarray:
        return np.array([self._value])

    def set_output(self, **kwargs: object) -> None:
        pass


class _VariablePipeline:
    """offset_ratio に基づく可変出力 Pipeline."""

    def predict(self, x: object) -> np.ndarray:
        import pandas as pd

        if isinstance(x, pd.DataFrame):
            offset = float(x.iloc[0].get("offset_ratio", 0.5))
        else:
            offset = 0.5
        return np.array([offset * 3.0 - 1.0])

    def set_output(self, **kwargs: object) -> None:
        pass


def _make_gate(pipeline: object, *, use_ob: bool = False) -> SkipGate:
    """テスト用 SkipGate を構築."""
    features = ["side_buy", "spread_jpy", "offset_ratio"]
    gate = SkipGate.__new__(SkipGate)
    gate.model = None
    gate.scaler = None
    gate.feature_cols = features
    gate.config = SkipGateConfig(mode="pnl", use_ob_features=use_ob)
    gate.metadata = {}
    gate._pipeline = pipeline
    gate._feature_index = {c: i for i, c in enumerate(features)}
    gate._recent_skips_buy = []
    gate._recent_skips_sell = []
    gate._pas_history_buy = []
    gate._pas_history_sell = []
    gate._score_calibrator = None
    return gate


class TestModelDegeneracyCheck:
    """645# _check_model_degeneracy のテスト."""

    @staticmethod
    def _get_checker():
        from scripts.v460.lib.skip_gate_model_loader import SkipGateModelLoaderMixin
        return SkipGateModelLoaderMixin._check_model_degeneracy

    def test_constant_output_is_degenerate(self) -> None:
        """定数出力モデルは退化として検出される."""
        gate = _make_gate(_ConstantPipeline(-0.3))
        check = self._get_checker()
        assert check(gate, "sell") is True

    def test_variable_output_is_not_degenerate(self) -> None:
        """多様な出力モデルは正常として通過する."""
        gate = _make_gate(_VariablePipeline())
        check = self._get_checker()
        assert check(gate, "sell") is False

    def test_no_pipeline_skips_check(self) -> None:
        """Pipeline なしの場合はチェックをスキップ (False)."""
        gate = _make_gate(None)
        check = self._get_checker()
        assert check(gate, "sell") is False

    def test_threshold_boundary(self) -> None:
        """dominant_ratio ちょうど 40% は通過 (> が条件)."""
        # 12テストケース中、4.8件以下なら40%以下 → 実質5件が限界 (41.7%)
        # 定数=すべて同じなので100%→退化
        gate = _make_gate(_ConstantPipeline(1.5))
        check = self._get_checker()
        # 100% dominant → degenerate
        assert check(gate, "buy") is True
        # max_dominant_ratio=1.0 なら通過
        assert check(gate, "buy", max_dominant_ratio=1.0) is False

    def test_prediction_error_is_tolerant(self) -> None:
        """予測例外が発生してもクラッシュしない."""

        class _BrokenPipeline:
            def predict(self, x: object) -> np.ndarray:
                raise RuntimeError("broken")

            def set_output(self, **kwargs: object) -> None:
                pass

        gate = _make_gate(_BrokenPipeline())
        check = self._get_checker()
        # 全て例外 → preds < 6 → False (判定不能)
        assert check(gate, "sell") is False
