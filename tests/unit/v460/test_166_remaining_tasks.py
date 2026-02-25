"""166# §7/§8 remaining tasks: unit tests.

Tests for:
1. C.6: SkipGate Pipeline set_output (sklearn warning fix)
2. C.7: cancel_failed_likely_filled KPI field
3. Deadlock fixes: side alternation on skip paths
"""
from __future__ import annotations

import asyncio
import pickle
import tempfile
import warnings
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── C.6: SkipGate set_output ──────────────────────────────

class TestSklearnWarningFix:
    """§7.3 C.6: sklearn feature-name warning elimination."""

    def test_pipeline_set_output_called(self):
        """Pipeline ロード時に set_output(transform='pandas') が呼ばれる."""
        from scripts.v460.ml.skip_gate import SkipGate, SkipGateConfig

        mock_pipeline = MagicMock()
        mock_pipeline.set_output = MagicMock()

        gate = SkipGate(
            model=MagicMock(),
            scaler=MagicMock(),
            feature_cols=["f1", "f2", "f3"],
            config=SkipGateConfig(),
            pipeline=mock_pipeline,
        )
        mock_pipeline.set_output.assert_called_once_with(transform="pandas")

    def test_scaler_set_output_called(self):
        """スタンドアロン scaler にも set_output が呼ばれる."""
        from scripts.v460.ml.skip_gate import SkipGate, SkipGateConfig

        mock_scaler = MagicMock()
        mock_scaler.set_output = MagicMock()

        gate = SkipGate(
            model=MagicMock(),
            scaler=mock_scaler,
            feature_cols=["f1", "f2", "f3"],
            config=SkipGateConfig(),
        )
        mock_scaler.set_output.assert_called_once_with(transform="pandas")

    def test_no_set_output_when_pipeline_none(self):
        """Pipeline=None の場合は set_output を呼ばない."""
        from scripts.v460.ml.skip_gate import SkipGate, SkipGateConfig

        gate = SkipGate(
            model=MagicMock(),
            scaler=object(),  # no set_output attribute
            feature_cols=["f1"],
            config=SkipGateConfig(),
            pipeline=None,
        )
        # No error → pass

    def test_production_model_no_warnings(self):
        """本番モデルで sklearn 警告が出ないことを確認."""
        model_path = Path("models/v460/skip_gate_lgbm_pnl120.pkl")
        if not model_path.exists():
            pytest.skip("Production model not available")

        from scripts.v460.ml.skip_gate import SkipGate

        gate = SkipGate.load(model_path)
        x = np.zeros(len(gate.feature_cols))
        x_df = pd.DataFrame([x], columns=gate.feature_cols)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            if gate._pipeline is not None:
                gate._pipeline.predict(x_df)
            sklearn_warns = [
                ww for ww in w if "feature names" in str(ww.message)
            ]
            assert len(sklearn_warns) == 0, (
                f"sklearn warnings still present: {sklearn_warns}"
            )


# ── C.7: cancel_failed_likely_filled KPI ──────────────────

class TestCancelFailedKPI:
    """§7.3 C.7: cancel_failed_likely_filled field."""

    def test_fill_monitor_result_has_field(self):
        """FillMonitorResult に cancel_failed_likely_filled がある."""
        from scripts.v460.lib.fill_config import FillMonitorResult

        r = FillMonitorResult()
        assert hasattr(r, "cancel_failed_likely_filled")
        assert r.cancel_failed_likely_filled is False

    def test_fill_monitor_result_field_set(self):
        """FillMonitorResult の cancel_failed_likely_filled を True に設定可能."""
        from scripts.v460.lib.fill_config import FillMonitorResult

        r = FillMonitorResult(
            filled=True,
            cancel_failed_likely_filled=True,
        )
        assert r.cancel_failed_likely_filled is True

    def test_fill_record_has_field(self):
        """FillRecord に cancel_failed_likely_filled がある."""
        from ztb.metrics.fill_quality import FillRecord

        r = FillRecord(
            cycle_id="test",
            timestamp=0.0,
            side="buy",
            order_price=100.0,
            order_quantity=0.001,
        )
        assert hasattr(r, "cancel_failed_likely_filled")
        assert r.cancel_failed_likely_filled is None

    def test_fill_record_roundtrip(self):
        """FillRecord の cancel_failed_likely_filled が JSON roundtrip で保持される."""
        from ztb.metrics.fill_quality import FillRecord

        r = FillRecord(
            cycle_id="test",
            timestamp=0.0,
            side="buy",
            order_price=100.0,
            order_quantity=0.001,
            cancel_failed_likely_filled=True,
        )
        d = r.to_dict()
        assert d["cancel_failed_likely_filled"] is True
        r2 = FillRecord.from_dict(d)
        assert r2.cancel_failed_likely_filled is True


# ── Deadlock fixes ─────────────────────────────────────────

class TestDeadlockSideAlternation:
    """166# deadlock fix: skip paths must update _last_side for alternation."""

    def test_orchestrator_code_has_deadlock_fix_unknown_regime(self):
        """unknown_regime_buy_skip パスに _last_side 更新がある."""
        code = Path("scripts/v460/lib/fill_loop_orchestrator.py").read_text(
            encoding="utf-8"
        )
        # Find the unknown_buy_skip section with deadlock fix
        assert '166# deadlock fix' in code
        idx = code.index('"unknown_buy_skip"')
        # The deadlock fix should be nearby (within 200 chars)
        nearby = code[idx:idx + 300]
        assert 'self._last_side = "buy"' in nearby, (
            "unknown_regime_buy_skip missing _last_side update"
        )

    def test_orchestrator_code_has_deadlock_fix_buy_kill(self):
        """buy_dynamic_kill パスに _last_side 更新がある."""
        code = Path("scripts/v460/lib/fill_loop_orchestrator.py").read_text(
            encoding="utf-8"
        )
        idx = code.index('"buy_dynamic_kill"')
        nearby = code[idx:idx + 300]
        assert 'self._last_side = "buy"' in nearby, (
            "buy_dynamic_kill missing _last_side update"
        )

    def test_orchestrator_code_has_deadlock_fix_sell_kill(self):
        """sell_dynamic_kill パスに _last_side 更新がある."""
        code = Path("scripts/v460/lib/fill_loop_orchestrator.py").read_text(
            encoding="utf-8"
        )
        idx = code.index('"sell_dynamic_kill"')
        nearby = code[idx:idx + 300]
        assert 'self._last_side = "sell"' in nearby, (
            "sell_dynamic_kill missing _last_side update"
        )

    def test_side_selector_alternation_after_buy_skip(self):
        """_last_side='buy' 設定後に _next_side() が 'sell' を返す."""
        from scripts.v460.lib.fill_config import FillTestConfig
        from scripts.v460.lib.side_selector import SideSelector

        config = FillTestConfig()
        ss = SideSelector(config)
        ss.last_side = "buy"
        result = ss.next()
        assert result == "sell", f"Expected 'sell' after _last_side='buy', got '{result}'"

    def test_side_selector_alternation_after_sell_skip(self):
        """_last_side='sell' 設定後に _next_side() が 'buy' を返す."""
        from scripts.v460.lib.fill_config import FillTestConfig
        from scripts.v460.lib.side_selector import SideSelector

        config = FillTestConfig()
        ss = SideSelector(config)
        ss.last_side = "sell"
        result = ss.next()
        assert result == "buy", f"Expected 'buy' after _last_side='sell', got '{result}'"
