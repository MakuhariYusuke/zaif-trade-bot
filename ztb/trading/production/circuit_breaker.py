"""
V433 Phase 5: Emergency Control Layer - Circuit Breaker

システムの異常を検知し、自動的に保護回路を動作させる。
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Callable, Any, Awaitable
from enum import Enum
import json
import os
import threading
import time


class CircuitState(Enum):
    """回路状態"""
    CLOSED = "closed"      # 閉回路（正常動作）
    OPEN = "open"         # 開回路（保護動作）
    HALF_OPEN = "half_open"  # 半開回路（テスト中）


class FailureType(Enum):
    """障害タイプ"""
    TIMEOUT = "timeout"              # タイムアウト
    EXCEPTION = "exception"          # 例外
    ERROR_RATE = "error_rate"        # エラーレート
    PERFORMANCE = "performance"      # パフォーマンス低下
    RESOURCE = "resource"            # リソース不足
    EXTERNAL = "external"           # 外部サービス障害


@dataclass
class CircuitBreakerConfig:
    """回路ブレーカー設定"""
    failure_threshold: int = 5       # 失敗閾値
    recovery_timeout_seconds: int = 60  # 回復タイムアウト
    success_threshold: int = 3       # 成功閾値（半開状態での）
    timeout_seconds: int = 30        # タイムアウト時間
    monitoring_window_seconds: int = 300  # 監視ウィンドウ
    name: str = "default"


@dataclass
class CircuitBreakerMetrics:
    """回路ブレーカーメトリクス"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0


@dataclass
class CircuitBreakerEvent:
    """回路ブレーカーイベント"""
    event_id: str
    timestamp: datetime
    event_type: str  # 'state_change', 'failure', 'recovery_attempt', 'recovery_success'
    from_state: Optional[CircuitState] = None
    to_state: Optional[CircuitState] = None
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class CircuitBreaker:
    """
    回路ブレーカー

    システムの異常を検知し、自動的に保護回路を動作させる。
    マイクロサービスアーキテクチャにおける障害伝播防止に使用。
    """

    def __init__(self, config: CircuitBreakerConfig):
        """
        初期化

        Args:
            config: 回路ブレーカー設定
        """
        self.config = config
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.last_state_change = datetime.now()

        # イベント履歴
        self.events: List[CircuitBreakerEvent] = []

        # コールバック
        self.state_change_callbacks: List[Callable[[CircuitState, CircuitState, str], Awaitable[None]]] = []

        # ロギング
        self.logger = logging.getLogger(f"{__name__}.{config.name}")

        self.logger.info(f"Circuit Breaker '{config.name}' initialized in {self.state.value} state")

    async def call(self, func: Callable[[], Awaitable[Any]], *args, **kwargs) -> Any:
        """
        保護された関数呼び出し

        Args:
            func: 呼び出す関数
            *args: 関数引数
            **kwargs: 関数キーワード引数

        Returns:
            Any: 関数戻り値

        Raises:
            CircuitBreakerOpenException: 回路が開いている場合
            Exception: 元の関数が投げる例外
        """
        if self.state == CircuitState.OPEN:
            if not self._should_attempt_reset():
                raise CircuitBreakerOpenException(f"Circuit breaker '{self.config.name}' is OPEN")

            # 半開状態に移行してテスト
            await self._change_state(CircuitState.HALF_OPEN, "Attempting reset")

        if self.state == CircuitState.HALF_OPEN:
            try:
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.config.timeout_seconds)
                await self._record_success()
                return result
            except Exception as e:
                await self._record_failure(FailureType.EXCEPTION, str(e))
                raise

        # CLOSED状態またはHALF_OPENでの成功
        try:
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.config.timeout_seconds)
            await self._record_success()
            return result
        except asyncio.TimeoutError:
            await self._record_failure(FailureType.TIMEOUT, "Operation timed out")
            raise
        except Exception as e:
            await self._record_failure(FailureType.EXCEPTION, str(e))
            raise

    def call_sync(self, func: Callable[[], Any], *args, **kwargs) -> Any:
        """
        同期関数用の保護された呼び出し

        Args:
            func: 呼び出す関数
            *args: 関数引数
            **kwargs: 関数キーワード引数

        Returns:
            Any: 関数戻り値

        Raises:
            CircuitBreakerOpenException: 回路が開いている場合
            Exception: 元の関数が投げる例外
        """
        if self.state == CircuitState.OPEN:
            if not self._should_attempt_reset():
                raise CircuitBreakerOpenException(f"Circuit breaker '{self.config.name}' is OPEN")

            self._change_state_sync(CircuitState.HALF_OPEN, "Attempting reset")

        if self.state == CircuitState.HALF_OPEN:
            try:
                result = func(*args, **kwargs)
                self._record_success_sync()
                return result
            except Exception as e:
                self._record_failure_sync(FailureType.EXCEPTION, str(e))
                raise

        # CLOSED状態またはHALF_OPENでの成功
        try:
            result = func(*args, **kwargs)
            self._record_success_sync()
            return result
        except Exception as e:
            self._record_failure_sync(FailureType.EXCEPTION, str(e))
            raise

    async def _record_success(self) -> None:
        """成功記録"""
        self.metrics.total_requests += 1
        self.metrics.successful_requests += 1
        self.metrics.consecutive_successes += 1
        self.metrics.consecutive_failures = 0
        self.metrics.last_success_time = datetime.now()

        # 半開状態での成功閾値チェック
        if self.state == CircuitState.HALF_OPEN and self.metrics.consecutive_successes >= self.config.success_threshold:
            await self._change_state(CircuitState.CLOSED, "Recovery successful")

        self._add_event("success", details={"consecutive_successes": self.metrics.consecutive_successes})

    async def _record_failure(self, failure_type: FailureType, reason: str) -> None:
        """失敗記録"""
        self.metrics.total_requests += 1
        self.metrics.failed_requests += 1
        self.metrics.consecutive_failures += 1
        self.metrics.consecutive_successes = 0
        self.metrics.last_failure_time = datetime.now()

        if failure_type == FailureType.TIMEOUT:
            self.metrics.timeout_requests += 1

        # 失敗閾値チェック
        if self.state == CircuitState.CLOSED and self.metrics.consecutive_failures >= self.config.failure_threshold:
            await self._change_state(CircuitState.OPEN, f"Failure threshold exceeded: {self.metrics.consecutive_failures} consecutive failures")

        # 半開状態での失敗
        elif self.state == CircuitState.HALF_OPEN:
            await self._change_state(CircuitState.OPEN, f"Failed during recovery attempt: {reason}")

        self._add_event("failure", details={
            "failure_type": failure_type.value,
            "reason": reason,
            "consecutive_failures": self.metrics.consecutive_failures
        })

    def _record_success_sync(self) -> None:
        """同期版成功記録"""
        asyncio.run(self._record_success())

    def _record_failure_sync(self, failure_type: FailureType, reason: str) -> None:
        """同期版失敗記録"""
        asyncio.run(self._record_failure(failure_type, reason))

    def _should_attempt_reset(self) -> bool:
        """
        リセット試行判定

        Returns:
            bool: リセット試行フラグ
        """
        if self.state != CircuitState.OPEN:
            return False

        time_since_open = datetime.now() - self.last_state_change
        return time_since_open.total_seconds() >= self.config.recovery_timeout_seconds

    async def _change_state(self, new_state: CircuitState, reason: str) -> None:
        """
        状態変更

        Args:
            new_state: 新しい状態
            reason: 変更理由
        """
        old_state = self.state
        self.state = new_state
        self.last_state_change = datetime.now()

        # 状態変更イベント
        self._add_event("state_change", from_state=old_state, to_state=new_state, reason=reason)

        # コールバック実行
        for callback in self.state_change_callbacks:
            try:
                asyncio.create_task(callback(old_state, new_state, reason))
            except Exception as e:
                self.logger.error(f"State change callback error: {e}")

        self.logger.warning(f"Circuit breaker '{self.config.name}' state changed: {old_state.value} -> {new_state.value} ({reason})")

    def _change_state_sync(self, new_state: CircuitState, reason: str) -> None:
        """同期版状態変更"""
        asyncio.run(self._change_state(new_state, reason))

    def _add_event(self, event_type: str, from_state: Optional[CircuitState] = None,
                  to_state: Optional[CircuitState] = None, reason: str = "",
                  details: Optional[Dict[str, Any]] = None) -> None:
        """
        イベント追加

        Args:
            event_type: イベントタイプ
            from_state: 変更元状態
            to_state: 変更先状態
            reason: 理由
            details: 詳細
        """
        event = CircuitBreakerEvent(
            event_id=f"EVENT_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            timestamp=datetime.now(),
            event_type=event_type,
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            details=details or {}
        )

        self.events.append(event)

        # イベント履歴制限（最新1000件）
        if len(self.events) > 1000:
            self.events = self.events[-1000:]

    def get_metrics(self) -> Dict[str, Any]:
        """
        メトリクス取得

        Returns:
            Dict[str, Any]: メトリクス
        """
        error_rate = 0.0
        if self.metrics.total_requests > 0:
            error_rate = (self.metrics.failed_requests / self.metrics.total_requests) * 100

        return {
            'name': self.config.name,
            'state': self.state.value,
            'total_requests': self.metrics.total_requests,
            'successful_requests': self.metrics.successful_requests,
            'failed_requests': self.metrics.failed_requests,
            'timeout_requests': self.metrics.timeout_requests,
            'error_rate_percent': error_rate,
            'consecutive_failures': self.metrics.consecutive_failures,
            'consecutive_successes': self.metrics.consecutive_successes,
            'last_failure_time': self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
            'last_success_time': self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None,
            'last_state_change': self.last_state_change.isoformat()
        }

    def get_events(self, limit: Optional[int] = None) -> List[CircuitBreakerEvent]:
        """
        イベント取得

        Args:
            limit: 取得件数制限

        Returns:
            List[CircuitBreakerEvent]: イベントリスト
        """
        events = self.events
        if limit:
            events = events[-limit:]
        return events

    def reset(self) -> None:
        """手動リセット"""
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.last_state_change = datetime.now()
        self.events.clear()

        self.logger.info(f"Circuit breaker '{self.config.name}' manually reset")

    def force_open(self) -> None:
        """強制開回路"""
        asyncio.run(self._change_state(CircuitState.OPEN, "Manually forced open"))

    def force_close(self) -> None:
        """強制閉回路"""
        asyncio.run(self._change_state(CircuitState.CLOSED, "Manually forced closed"))

    def add_state_change_callback(self, callback: Callable[[CircuitState, CircuitState, str], Awaitable[None]]) -> None:
        """
        状態変更コールバック追加

        Args:
            callback: コールバック関数
        """
        self.state_change_callbacks.append(callback)


class CircuitBreakerOpenException(Exception):
    """回路ブレーカー開回路例外"""
    pass


class CircuitBreakerRegistry:
    """
    回路ブレーカーレジストリ

    複数の回路ブレーカーを管理する。
    """

    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.logger = logging.getLogger(__name__)

    def create_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """
        回路ブレーカー作成

        Args:
            name: ブレーカー名
            config: 設定

        Returns:
            CircuitBreaker: 作成された回路ブレーカー
        """
        if name in self.breakers:
            raise ValueError(f"Circuit breaker '{name}' already exists")

        breaker = CircuitBreaker(config)
        self.breakers[name] = breaker

        self.logger.info(f"Circuit breaker '{name}' registered")
        return breaker

    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """
        回路ブレーカー取得

        Args:
            name: ブレーカー名

        Returns:
            Optional[CircuitBreaker]: 回路ブレーカー
        """
        return self.breakers.get(name)

    def remove_breaker(self, name: str) -> bool:
        """
        回路ブレーカー削除

        Args:
            name: ブレーカー名

        Returns:
            bool: 削除成功フラグ
        """
        if name in self.breakers:
            del self.breakers[name]
            self.logger.info(f"Circuit breaker '{name}' removed")
            return True
        return False

    def get_all_breakers(self) -> Dict[str, CircuitBreaker]:
        """
        全回路ブレーカー取得

        Returns:
            Dict[str, CircuitBreaker]: 回路ブレーカー辞書
        """
        return self.breakers.copy()

    def get_registry_metrics(self) -> Dict[str, Any]:
        """
        レジストリメトリクス取得

        Returns:
            Dict[str, Any]: メトリクス
        """
        total_breakers = len(self.breakers)
        open_breakers = len([b for b in self.breakers.values() if b.state == CircuitState.OPEN])
        closed_breakers = len([b for b in self.breakers.values() if b.state == CircuitState.CLOSED])
        half_open_breakers = len([b for b in self.breakers.values() if b.state == CircuitState.HALF_OPEN])

        return {
            'total_breakers': total_breakers,
            'open_breakers': open_breakers,
            'closed_breakers': closed_breakers,
            'half_open_breakers': half_open_breakers,
            'open_percentage': (open_breakers / total_breakers * 100) if total_breakers > 0 else 0
        }

    def save_state(self, filepath: str) -> None:
        """
        状態保存

        Args:
            filepath: 保存ファイルパス
        """
        state = {
            'breakers': {}
        }

        for name, breaker in self.breakers.items():
            state['breakers'][name] = {
                'config': {
                    'failure_threshold': breaker.config.failure_threshold,
                    'recovery_timeout_seconds': breaker.config.recovery_timeout_seconds,
                    'success_threshold': breaker.config.success_threshold,
                    'timeout_seconds': breaker.config.timeout_seconds,
                    'monitoring_window_seconds': breaker.config.monitoring_window_seconds,
                    'name': breaker.config.name
                },
                'state': breaker.state.value,
                'last_state_change': breaker.last_state_change.isoformat(),
                'metrics': {
                    'total_requests': breaker.metrics.total_requests,
                    'successful_requests': breaker.metrics.successful_requests,
                    'failed_requests': breaker.metrics.failed_requests,
                    'timeout_requests': breaker.metrics.timeout_requests,
                    'last_failure_time': breaker.metrics.last_failure_time.isoformat() if breaker.metrics.last_failure_time else None,
                    'last_success_time': breaker.metrics.last_success_time.isoformat() if breaker.metrics.last_success_time else None,
                    'consecutive_failures': breaker.metrics.consecutive_failures,
                    'consecutive_successes': breaker.metrics.consecutive_successes
                },
                'events': [
                    {
                        'event_id': e.event_id,
                        'timestamp': e.timestamp.isoformat(),
                        'event_type': e.event_type,
                        'from_state': e.from_state.value if e.from_state else None,
                        'to_state': e.to_state.value if e.to_state else None,
                        'reason': e.reason,
                        'details': e.details
                    }
                    for e in breaker.events[-100:]  # 最新100件
                ]
            }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Circuit breaker registry state saved to {filepath}")

    def load_state(self, filepath: str) -> bool:
        """
        状態読み込み

        Args:
            filepath: 読み込みファイルパス

        Returns:
            bool: 読み込み成功フラグ
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)

            for name, breaker_data in state.get('breakers', {}).items():
                # 設定復元
                config_data = breaker_data['config']
                config = CircuitBreakerConfig(
                    failure_threshold=config_data['failure_threshold'],
                    recovery_timeout_seconds=config_data['recovery_timeout_seconds'],
                    success_threshold=config_data['success_threshold'],
                    timeout_seconds=config_data['timeout_seconds'],
                    monitoring_window_seconds=config_data['monitoring_window_seconds'],
                    name=config_data['name']
                )

                # ブレーカー作成
                breaker = CircuitBreaker(config)

                # 状態復元
                breaker.state = CircuitState(breaker_data['state'])
                breaker.last_state_change = datetime.fromisoformat(breaker_data['last_state_change'])

                # メトリクス復元
                metrics_data = breaker_data['metrics']
                breaker.metrics = CircuitBreakerMetrics(
                    total_requests=metrics_data['total_requests'],
                    successful_requests=metrics_data['successful_requests'],
                    failed_requests=metrics_data['failed_requests'],
                    timeout_requests=metrics_data['timeout_requests'],
                    last_failure_time=datetime.fromisoformat(metrics_data['last_failure_time']) if metrics_data['last_failure_time'] else None,
                    last_success_time=datetime.fromisoformat(metrics_data['last_success_time']) if metrics_data['last_success_time'] else None,
                    consecutive_failures=metrics_data['consecutive_failures'],
                    consecutive_successes=metrics_data['consecutive_successes']
                )

                # イベント復元
                breaker.events = []
                for e_data in breaker_data.get('events', []):
                    event = CircuitBreakerEvent(
                        event_id=e_data['event_id'],
                        timestamp=datetime.fromisoformat(e_data['timestamp']),
                        event_type=e_data['event_type'],
                        from_state=CircuitState(e_data['from_state']) if e_data['from_state'] else None,
                        to_state=CircuitState(e_data['to_state']) if e_data['to_state'] else None,
                        reason=e_data['reason'],
                        details=e_data['details']
                    )
                    breaker.events.append(event)

                self.breakers[name] = breaker

            self.logger.info(f"Circuit breaker registry state loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load circuit breaker registry state: {e}")
            return False