"""345# プロアクティブ修正テスト.

対象:
  A: warmup forced_fill_pnl_downweight 整合性 (343# downweight を warmup にも適用)
  B: CircuitBreaker sync メソッド Py3.12+ 互換 (asyncio.new_event_loop 排除)
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ztb.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)


# ============================================================
# A: warmup forced_fill_pnl_downweight 整合性
# ============================================================


def _make_fill_record(
    *,
    side: str = "sell",
    filled: bool = True,
    pnl: float | None = -1.0,
    timestamp: float | None = None,
    forced: bool = False,
) -> SimpleNamespace:
    """テスト用 FillRecord 互換オブジェクト."""
    import time as _time
    return SimpleNamespace(
        side=side,
        filled=filled,
        post_fill_30s_pnl=pnl,
        timestamp=timestamp or _time.time(),
        balance_forced_switch=forced,
    )


class _StubKillMgr:
    """track() 呼び出しを記録するスタブ."""

    def __init__(self) -> None:
        self.tracked: list[float] = []

    def track(self, pnl: float) -> None:
        self.tracked.append(pnl)


class TestWarmupDownweight:
    """warmup で forced fill に downweight が適用されること (343# 整合性)."""

    def _make_mixin(self, *, downweight: float = 0.5):
        """OrchestratorLifecycleMixin の必要最低限スタブ."""
        from scripts.v460.lib.orchestrator_lifecycle import OrchestratorLifecycleMixin

        mixin = object.__new__(OrchestratorLifecycleMixin)
        mixin._sell_kill_mgr = _StubKillMgr()
        mixin._buy_kill_mgr = _StubKillMgr()
        mixin.config = SimpleNamespace(forced_fill_pnl_downweight=downweight)
        return mixin

    def test_normal_fill_tracked_at_full_pnl(self):
        """通常の fill は PnL がそのまま track される."""
        mixin = self._make_mixin()
        r = _make_fill_record(side="sell", pnl=-2.0, forced=False)
        mixin._warmup_kill_managers_from_records([r])
        assert mixin._sell_kill_mgr.tracked == [-2.0]

    def test_forced_fill_tracked_at_half_pnl(self):
        """forced fill は PnL × downweight (0.5) で track される."""
        mixin = self._make_mixin(downweight=0.5)
        r = _make_fill_record(side="sell", pnl=-4.0, forced=True)
        mixin._warmup_kill_managers_from_records([r])
        assert mixin._sell_kill_mgr.tracked == [-2.0]  # -4.0 * 0.5

    def test_forced_fill_skipped_when_downweight_zero(self):
        """downweight=0.0 (旧挙動) で forced fill は完全除外される."""
        mixin = self._make_mixin(downweight=0.0)
        r = _make_fill_record(side="buy", pnl=-3.0, forced=True)
        mixin._warmup_kill_managers_from_records([r])
        assert mixin._buy_kill_mgr.tracked == []

    def test_forced_fill_buy_side_downweighted(self):
        """buy 側の forced fill も downweight 適用."""
        mixin = self._make_mixin(downweight=0.3)
        r = _make_fill_record(side="buy", pnl=-10.0, forced=True)
        mixin._warmup_kill_managers_from_records([r])
        assert mixin._buy_kill_mgr.tracked == [pytest.approx(-3.0)]

    def test_mixed_normal_and_forced(self):
        """通常 fill と forced fill の混在で正しく重み付け."""
        mixin = self._make_mixin(downweight=0.5)
        records = [
            _make_fill_record(side="sell", pnl=-2.0, forced=False),
            _make_fill_record(side="sell", pnl=-6.0, forced=True),
            _make_fill_record(side="sell", pnl=1.0, forced=False),
        ]
        mixin._warmup_kill_managers_from_records(records)
        assert mixin._sell_kill_mgr.tracked == [-2.0, -3.0, 1.0]

    def test_old_record_without_forced_attr(self):
        """古い FillRecord (balance_forced_switch 属性なし) は通常扱い."""
        mixin = self._make_mixin(downweight=0.5)
        r = _make_fill_record(side="sell", pnl=-5.0)
        del r.balance_forced_switch  # 古い record
        mixin._warmup_kill_managers_from_records([r])
        assert mixin._sell_kill_mgr.tracked == [-5.0]

    def test_consistency_with_live_track(self):
        """warmup と live (_track_fill_pnl) で forced fill の扱いが一致すること.

        live 側: orchestrator_guards._track_fill_pnl() と同じ downweight ロジック。
        """
        from scripts.v460.lib.orchestrator_lifecycle import OrchestratorLifecycleMixin

        downweight = 0.5
        pnl = -8.0

        # warmup 経由
        mixin = self._make_mixin(downweight=downweight)
        r = _make_fill_record(side="sell", pnl=pnl, forced=True)
        mixin._warmup_kill_managers_from_records([r])
        warmup_result = mixin._sell_kill_mgr.tracked[0]

        # 直接計算 (live と同じロジック)
        live_result = pnl * downweight

        assert warmup_result == pytest.approx(live_result)


# ============================================================
# B: CircuitBreaker sync メソッド Py3.12+ 互換
# ============================================================


class TestCircuitBreakerSyncMethods:
    """sync メソッドが asyncio.new_event_loop() を使わないこと."""

    def test_on_success_sync_resets_failure_count(self):
        """_on_success_sync が failure_count をリセットすること."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())
        cb.failure_count = 3
        cb._on_success_sync()
        assert cb.failure_count == 0

    def test_on_success_sync_closes_half_open(self):
        """HALF_OPEN 状態で success_threshold 到達 → CLOSED."""
        cfg = CircuitBreakerConfig(success_threshold=2)
        cb = CircuitBreaker("test", cfg)
        cb.state = CircuitState.HALF_OPEN
        cb.success_count = 1  # あと 1 回で閾値到達
        cb._on_success_sync()
        assert cb.state == CircuitState.CLOSED
        assert cb.success_count == 2

    def test_on_success_sync_half_open_below_threshold(self):
        """HALF_OPEN で success_threshold 未到達 → HALF_OPEN のまま."""
        cfg = CircuitBreakerConfig(success_threshold=3)
        cb = CircuitBreaker("test", cfg)
        cb.state = CircuitState.HALF_OPEN
        cb.success_count = 0
        cb._on_success_sync()
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.success_count == 1

    def test_on_failure_sync_increments_count(self):
        """_on_failure_sync が failure_count をインクリメント."""
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=5))
        cb._on_failure_sync()
        assert cb.failure_count == 1
        assert cb.last_failure_time is not None

    def test_on_failure_sync_opens_on_threshold(self):
        """failure_threshold 到達で CLOSED → OPEN."""
        cfg = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", cfg)
        cb.failure_count = 2
        cb._on_failure_sync()
        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 3

    def test_on_failure_sync_reopens_half_open(self):
        """HALF_OPEN で failure → OPEN."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())
        cb.state = CircuitState.HALF_OPEN
        cb._on_failure_sync()
        assert cb.state == CircuitState.OPEN

    def test_on_failure_sync_stays_closed_below_threshold(self):
        """failure_threshold 未到達で CLOSED のまま."""
        cfg = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker("test", cfg)
        cb._on_failure_sync()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 1

    def test_record_success_no_running_loop(self):
        """record_success() がイベントループなしでも正常動作."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())
        cb.failure_count = 2
        # record_success は create_task を試み、失敗時に _on_success_sync を呼ぶ
        cb.record_success()
        assert cb.failure_count == 0

    def test_record_failure_no_running_loop(self):
        """record_failure() がイベントループなしでも正常動作."""
        cfg = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test", cfg)
        cb.record_failure()
        assert cb.failure_count == 1
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    @patch("ztb.utils.circuit_breaker.asyncio.new_event_loop")
    def test_sync_methods_do_not_create_event_loop(self, mock_new_loop):
        """345# sync メソッドが asyncio.new_event_loop() を呼ばないこと."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())
        cb._on_success_sync()
        cb._on_failure_sync()
        mock_new_loop.assert_not_called()
