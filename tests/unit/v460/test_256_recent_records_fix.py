"""256# _recent_records 累積バグ修正 + 冗長 getattr 排除テスト.

対象:
- fill_loop_orchestrator: _recent_records deque 化 + batch.append 後に累積
- skip_gate_evaluator: ob.bids/ob.asks 冗長 getattr → 直接参照
"""

from __future__ import annotations

import inspect
from collections import deque

import pytest
from tests.unit.v460._fill_test_source import ORCHESTRATOR_POST_CYCLE, read_source_text

from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator

_SKIP_GATE_EVALUATE_SOURCE = inspect.getsource(SkipGateEvaluator.evaluate)


class TestRecentRecordsDeque:
    """256# _recent_records deque 化の確認."""

    def test_recent_records_is_deque(self) -> None:
        """_recent_records がクラスレベルで deque."""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        assert isinstance(FillLoopOrchestratorMixin._recent_records, deque)

    def test_recent_records_maxlen(self) -> None:
        """_recent_records の maxlen が設定されている."""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        assert FillLoopOrchestratorMixin._recent_records.maxlen is not None
        assert FillLoopOrchestratorMixin._recent_records.maxlen >= 100

    def test_batch_append_also_appends_recent_records(self) -> None:
        """batch.append(record) 直後に self._recent_records.append(record) がある."""
        src = read_source_text(ORCHESTRATOR_POST_CYCLE)
        # batch.append(record) の直後数行に _recent_records.append(record) があること
        lines = src.splitlines()
        found = False
        for i, ln in enumerate(lines):
            if "batch.append(record)" in ln and "skip_record" not in ln:
                # 直後 5 行以内に _recent_records.append(record) があること
                window = "\n".join(lines[i : i + 5])
                if "_recent_records.append(record)" in window:
                    found = True
                    break
        assert found, "batch.append(record) 直後に _recent_records.append(record) が見つからない"


class TestRedundantGetAttrRemoval:
    """256# skip_gate_evaluator 冗長 getattr 排除."""

    def test_ob_bids_asks_no_redundant_getattr(self) -> None:
        """ob.bids/ob.asks の取得が getattr ではなく直接参照."""
        lines = [
            ln for ln in _SKIP_GATE_EVALUATE_SOURCE.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        # 存在チェック用の getattr(ob, "bids", None) は OK (duck-typing)
        # 実際の値取得の getattr(ob, "bids") (デフォルトなし) は NG
        for ln in lines:
            if "getattr(ob, " in ln and "None)" not in ln:
                pytest.fail(f"Redundant getattr found: {ln.strip()}")
