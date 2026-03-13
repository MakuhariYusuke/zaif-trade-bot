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
# A: 348# balance_forced 撤廃: TestWarmupDownweight 削除
# ============================================================


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
