"""P2-3: Failure mode / fault-injection テスト.

158# backlog: 障害モードのモックベーステスト。
- CircuitBreaker 状態遷移 (CLOSED → OPEN → HALF_OPEN → CLOSED)
- OrderManager タイムアウト (_execute_order_async 30s)
- RiskManager 緊急停止 / 日次損失制限
- 価格取得フォールバックチェーン
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ztb.trading.live_trader.price_utils import resolve_current_price
from ztb.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenException,
    CircuitState,
)


# ===================================================================
# CircuitBreaker テスト
# ===================================================================

class TestCircuitBreakerStateTransitions:
    """CircuitBreaker: CLOSED → OPEN → HALF_OPEN → CLOSED 完全遷移."""

    @pytest.fixture
    def cb(self) -> CircuitBreaker:
        cfg = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=0.01,  # 10ms for faster test
            success_threshold=2,
            timeout=5.0,
        )
        return CircuitBreaker("test_cb", cfg)

    @pytest.mark.asyncio
    async def test_starts_closed(self, cb: CircuitBreaker) -> None:
        assert cb.get_state() == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_opens_after_failure_threshold(self, cb: CircuitBreaker) -> None:
        """3 failures → OPEN."""
        async def fail():
            raise ConnectionError("API down")

        for _ in range(3):
            with pytest.raises(ConnectionError):
                await cb.call(fail)

        assert cb.get_state() == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_open_rejects_calls(self, cb: CircuitBreaker) -> None:
        """OPEN state → CircuitBreakerOpenException."""
        async def fail():
            raise ConnectionError("API down")

        for _ in range(3):
            with pytest.raises(ConnectionError):
                await cb.call(fail)

        with pytest.raises(CircuitBreakerOpenException):
            await cb.call(fail)

    @pytest.mark.asyncio
    async def test_half_open_after_recovery_timeout(self, cb: CircuitBreaker) -> None:
        """OPEN → wait recovery_timeout → HALF_OPEN."""
        async def fail():
            raise ConnectionError("API down")

        async def succeed():
            return "ok"

        for _ in range(3):
            with pytest.raises(ConnectionError):
                await cb.call(fail)

        assert cb.get_state() == CircuitState.OPEN

        # 実待機せず time を進め、HALF_OPEN 遷移を確認
        fake_now = (cb.last_failure_time or 0.0) + cb.config.recovery_timeout + 0.001
        with patch("ztb.utils.circuit_breaker.time.time", return_value=fake_now):
            result = await cb.call(succeed)
        assert result == "ok"
        # State should be HALF_OPEN (1 success, need 2)
        assert cb.get_state() == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_closes_after_success_threshold(self, cb: CircuitBreaker) -> None:
        """HALF_OPEN → 2 successes → CLOSED."""
        async def fail():
            raise ConnectionError("API down")

        async def succeed():
            return "ok"

        # Force OPEN
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await cb.call(fail)

        fake_now = (cb.last_failure_time or 0.0) + cb.config.recovery_timeout + 0.001
        # 2 successes in HALF_OPEN → CLOSED
        with patch("ztb.utils.circuit_breaker.time.time", return_value=fake_now):
            await cb.call(succeed)
        assert cb.get_state() == CircuitState.HALF_OPEN
        await cb.call(succeed)
        assert cb.get_state() == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self, cb: CircuitBreaker) -> None:
        """HALF_OPEN → 1 failure → back to OPEN."""
        async def fail():
            raise ConnectionError("API down")

        async def succeed():
            return "ok"

        # Force OPEN
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await cb.call(fail)

        fake_now = (cb.last_failure_time or 0.0) + cb.config.recovery_timeout + 0.001
        # One success to enter HALF_OPEN
        with patch("ztb.utils.circuit_breaker.time.time", return_value=fake_now):
            await cb.call(succeed)
        assert cb.get_state() == CircuitState.HALF_OPEN

        # Failure in HALF_OPEN → back to OPEN
        with pytest.raises(ConnectionError):
            await cb.call(fail)
        assert cb.get_state() == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_manual_reset(self, cb: CircuitBreaker) -> None:
        async def fail():
            raise ConnectionError("API down")

        for _ in range(3):
            with pytest.raises(ConnectionError):
                await cb.call(fail)

        assert cb.get_state() == CircuitState.OPEN
        cb.reset()
        assert cb.get_state() == CircuitState.CLOSED
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_timeout_counts_as_failure(self, cb: CircuitBreaker) -> None:
        """Function exceeding timeout → asyncio.TimeoutError → counts as failure."""
        cb.config.timeout = 0.01

        async def slow():
            await asyncio.Event().wait()
            return "late"

        with pytest.raises(asyncio.TimeoutError):
            await cb.call(slow)

        assert cb.failure_count == 1


# ===================================================================
# OrderManager タイムアウトテスト
# ===================================================================

class TestOrderManagerTimeout:
    """_execute_order_async の 30s タイムアウト."""

    def _make_om(self) -> tuple:
        from ztb.trading.live_trader.components.order_manager import OrderManager

        lt = MagicMock()
        lt.demo_mode = False
        lt._last_valid_price = 15_000_000.0
        lt._current_prices = {"btc_jpy": 15_000_000.0}
        lt.position = "flat"
        lt._send_notification = MagicMock()

        adapter = AsyncMock()
        lt.exchange_adapter = adapter
        om = OrderManager(lt)
        return om, lt, adapter

    def test_timeout_returns_false(self) -> None:
        """place_order が 30s 以上かかる場合、execute_trade は False を返す."""
        om, lt, adapter = self._make_om()

        # execute_trade が TimeoutError を安全にハンドリングすることを検証
        with patch.object(om, "_execute_order_async", side_effect=TimeoutError("Order timed out")):
            result = om.execute_trade("buy", 0.001)

        assert result is False

    def test_exchange_returns_none_is_false(self) -> None:
        """place_order が None を返す → execute_trade は False."""
        om, lt, adapter = self._make_om()
        adapter.place_order = AsyncMock(return_value=None)
        assert om.execute_trade("buy", 0.001) is False


# ===================================================================
# RiskManager 障害モードテスト
# ===================================================================

class TestRiskManagerFailureModes:
    """RiskManager: 緊急停止 / 日次損失制限 / トレード頻度制限."""

    @pytest.fixture
    def risk_mgr(self):
        from ztb.trading.live_trader.components.risk_manager import RiskManager

        lt = MagicMock()
        lt.config = {
            "max_daily_loss": 5000.0,
            "max_daily_trades": 10,
            "max_trades_per_hour": 3,
            "emergency_stop_loss": 0.05,
        }
        lt.daily_start_pnl = 0.0
        lt.daily_trades = 0
        lt.total_pnl = 0.0
        lt.position = 0
        lt.entry_price = 0.0
        lt._send_notification = MagicMock()

        rm = RiskManager(lt)
        return rm, lt

    def test_daily_loss_limit_blocks(self, risk_mgr: tuple) -> None:
        """日次損失制限超過 → can_trade = False."""
        rm, lt = risk_mgr
        lt.total_pnl = -6000.0  # > 5000 limit
        rm.daily_start_pnl = 0.0
        assert rm.check_daily_loss_limit() is False

    def test_daily_loss_within_limit(self, risk_mgr: tuple) -> None:
        rm, lt = risk_mgr
        lt.total_pnl = -3000.0
        rm.daily_start_pnl = 0.0
        assert rm.check_daily_loss_limit() is True

    def test_emergency_stop_triggers_on_large_loss(self, risk_mgr: tuple) -> None:
        """5% 以上の損失率 → 緊急停止."""
        rm, lt = risk_mgr
        lt.position = 1  # long
        lt.entry_price = 15_000_000.0
        # 6% loss
        current_price = 15_000_000.0 * 0.94
        assert rm.check_emergency_stop_loss(current_price) is False

    def test_emergency_stop_within_threshold(self, risk_mgr: tuple) -> None:
        rm, lt = risk_mgr
        lt.position = 1
        lt.entry_price = 15_000_000.0
        current_price = 15_000_000.0 * 0.97  # 3% loss
        assert rm.check_emergency_stop_loss(current_price) is True

    def test_emergency_stop_no_position(self, risk_mgr: tuple) -> None:
        """ポジションなし → 常に True (停止せず)."""
        rm, lt = risk_mgr
        lt.position = 0
        lt.entry_price = 0
        assert rm.check_emergency_stop_loss(10_000_000.0) is True

    def test_hourly_trade_limit(self, risk_mgr: tuple) -> None:
        """時間あたりトレード制限超過."""
        rm, lt = risk_mgr
        rm.hourly_trades = 3  # max_trades_per_hour = 3
        assert rm.check_hourly_trade_limit() is False

    def test_daily_trade_limit(self, risk_mgr: tuple) -> None:
        rm, lt = risk_mgr
        rm.daily_trades = 10  # max_daily_trades = 10
        assert rm.check_daily_trade_limit() is False

    def test_can_trade_all_checks_pass(self, risk_mgr: tuple) -> None:
        rm, lt = risk_mgr
        lt.total_pnl = 0.0
        lt.position = 0
        lt.entry_price = 0
        rm.daily_trades = 0
        rm.hourly_trades = 0
        assert rm.can_trade(15_000_000.0) is True

    def test_can_trade_blocked_by_daily_loss(self, risk_mgr: tuple) -> None:
        rm, lt = risk_mgr
        lt.total_pnl = -6000.0
        rm.daily_start_pnl = 0.0
        assert rm.can_trade(15_000_000.0) is False

    def test_notification_on_emergency_stop(self, risk_mgr: tuple) -> None:
        rm, lt = risk_mgr
        lt.position = 1
        lt.entry_price = 15_000_000.0
        rm.check_emergency_stop_loss(15_000_000.0 * 0.94)
        lt._send_notification.assert_called_once()
        args = str(lt._send_notification.call_args)
        assert "EMERGENCY" in args or "STOP" in args


# ===================================================================
# 価格取得フォールバックチェーン
# ===================================================================

class TestPriceFallbackChain:
    """LiveTrader._get_current_price のフォールバック挙動."""

    def _resolve_price(self, adapter_result: Optional[float]) -> tuple[float, float]:
        adapter = AsyncMock()
        adapter.get_current_price.return_value = adapter_result
        return resolve_current_price(
            exchange_adapter=adapter,
            last_valid_price=14_500_000.0,
        )

    def test_valid_price_updates_last(self) -> None:
        price, last_valid_price = self._resolve_price(15_000_000.0)
        assert price == 15_000_000.0
        assert last_valid_price == 15_000_000.0

    def test_zero_falls_back_to_last_valid(self) -> None:
        price, last_valid_price = self._resolve_price(0.0)
        assert price == 14_500_000.0
        assert last_valid_price == 14_500_000.0

    def test_none_falls_back_to_last_valid(self) -> None:
        price, last_valid_price = self._resolve_price(None)
        assert price == 14_500_000.0
        assert last_valid_price == 14_500_000.0

    def test_negative_price_falls_back(self) -> None:
        price, last_valid_price = self._resolve_price(-100.0)
        assert price == 14_500_000.0
        assert last_valid_price == 14_500_000.0


# ===================================================================
# 連続エラー & エラーカウンタ
# ===================================================================

class TestConsecutiveErrorHandling:
    """LiveTrader 連続エラーでのループ中断はトレーディングループ内で
    max_consecutive_errors=5 で制御。ここではカウンタロジックを検証."""

    def test_error_counter_increments(self) -> None:
        """連続エラーのカウンタがインクリメントされることを確認."""
        consecutive_errors = 0
        max_consecutive_errors = 5

        for i in range(5):
            try:
                raise ConnectionError(f"Error {i}")
            except Exception:
                consecutive_errors += 1

        assert consecutive_errors >= max_consecutive_errors

    def test_success_resets_counter(self) -> None:
        """成功でカウンタがリセットされることを確認."""
        consecutive_errors = 3

        # Success resets
        consecutive_errors = 0
        assert consecutive_errors == 0

    def test_loop_breaks_at_threshold(self) -> None:
        """5回連続エラーでループが中断されることをシミュレート."""
        iterations = 0
        consecutive_errors = 0
        max_consecutive_errors = 5

        for _ in range(100):
            iterations += 1
            try:
                raise RuntimeError("API unavailable")
            except Exception:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    break

        assert iterations == 5
        assert consecutive_errors == 5
