"""254# frozen_side 永続化, orchestrator getattr 排除, bare except 改善.

テスト対象:
  P1-1: _one_sided_frozen_side の FillTestState 永続化
  P1-2: saved_state getattr → 直接参照
  P1-3: _recent_records クラスレベルデフォルト
  P1-4: _heartbeat_task クラスレベルデフォルト
  P1-5: heartbeat bare except → logger.debug
"""

from __future__ import annotations

import inspect
import re

import pytest


class TestFrozenSidePersistence:
    """254# P1-1: _one_sided_frozen_side が FillTestState に永続化される."""

    def test_fill_test_state_has_frozen_side_field(self) -> None:
        """FillTestState に one_sided_frozen_side フィールドが存在."""
        from scripts.v460.lib.resilience import FillTestState
        state = FillTestState()
        assert hasattr(state, "one_sided_frozen_side")
        assert state.one_sided_frozen_side is None

    def test_frozen_side_round_trip(self) -> None:
        """frozen_side の serialize/deserialize round-trip."""
        from scripts.v460.lib.resilience import FillTestState, FillTestStatePersistence
        import dataclasses, json, tempfile
        from pathlib import Path

        state = FillTestState(
            one_sided_frozen_side="sell",
            one_sided_freeze_remaining=5,
        )
        d = dataclasses.asdict(state)
        restored = json.loads(json.dumps(d))
        assert restored["one_sided_frozen_side"] == "sell"
        assert restored["one_sided_freeze_remaining"] == 5

    def test_snapshot_includes_frozen_side(self) -> None:
        """_build_state_snapshot が frozen_side を含むことをソースで確認."""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        src = inspect.getsource(FillLoopOrchestratorMixin._build_state_snapshot)
        assert "one_sided_frozen_side" in src

    def test_restore_includes_frozen_side(self) -> None:
        """_restore_common_state が frozen_side を復元することをソースで確認."""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        src = inspect.getsource(FillLoopOrchestratorMixin._restore_common_state)
        assert "_one_sided_frozen_side" in src


class TestSavedStateGetAttrRemoval:
    """254# P1-2: _restore_common_state の getattr 排除."""

    def test_no_getattr_saved_state_in_restore(self) -> None:
        """_restore_common_state に getattr(saved_state, ...) が残存しないこと."""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        src = inspect.getsource(FillLoopOrchestratorMixin._restore_common_state)
        # saved_state に対する getattr は全て除去済
        matches = re.findall(r'getattr\(saved_state', src)
        assert len(matches) == 0, f"getattr(saved_state, ...) found {len(matches)} times"


class TestOrchestratorClassLevelDefaults:
    """254# P1-3/P1-4: クラスレベルデフォルト宣言."""

    def test_recent_records_class_default(self) -> None:
        from collections import deque
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        assert hasattr(FillLoopOrchestratorMixin, "_recent_records")
        assert isinstance(FillLoopOrchestratorMixin._recent_records, deque)
        assert len(FillLoopOrchestratorMixin._recent_records) == 0

    def test_heartbeat_task_class_default(self) -> None:
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        assert hasattr(FillLoopOrchestratorMixin, "_heartbeat_task")
        assert FillLoopOrchestratorMixin._heartbeat_task is None

    def test_check_stop_conditions_no_getattr_self(self) -> None:
        """_check_regime_stop_conditions に getattr(self, ...) が残存しないこと."""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        src = inspect.getsource(FillLoopOrchestratorMixin._check_regime_stop_conditions)
        matches = re.findall(r'getattr\(self', src)
        assert len(matches) == 0

    def test_cleanup_heartbeat_no_getattr(self) -> None:
        """cleanup_heartbeat に getattr 呼び出しが残存しないこと."""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        src = inspect.getsource(FillLoopOrchestratorMixin.cleanup_heartbeat)
        # コードから getattr( 呼び出しを検索（コメントは除外）
        code_lines = [
            line for line in src.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        code_only = "\n".join(code_lines)
        assert "getattr(" not in code_only


class TestBareExceptImproved:
    """254# P1-5: heartbeat bare except → logger.debug."""

    def test_heartbeat_psutil_except_has_logging(self) -> None:
        """psutil except ブロックに logger.debug が含まれること."""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        # run_continuous は巨大なので、全ソースから該当箇所を検索
        src_path = inspect.getfile(FillLoopOrchestratorMixin)
        from pathlib import Path
        full_src = Path(src_path).read_text(encoding="utf-8")
        # "psutil" の except ブロック付近に logger.debug があること
        idx = full_src.find("psutil memory check unavailable")
        assert idx > 0, "logger.debug message not found near psutil except"
