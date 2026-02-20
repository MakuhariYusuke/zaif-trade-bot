"""121# 時間帯フィルターモジュール.

FillTestRunner から時間帯フィルター判定を分離。
041# 時間帯フィルター / 073# side 別 / 100# P1-3 union を統合。

責務:
  - 高 AS 時間帯の判定 (グローバル + side別)
  - フィルタ突入/離脱状態の管理
"""

from __future__ import annotations

import logging
import time as _time
from datetime import datetime, timezone

from scripts.v460.lib.fill_config import FillTestConfig

logger = logging.getLogger(__name__)


class TimeFilter:
    """041# 時間帯フィルター: 高 AS 時間帯の判定."""

    def __init__(self, config: FillTestConfig) -> None:
        self._config = config
        self._in_filter: bool = False
        self._last_heartbeat_time: float = 0.0
        # 110# 086# デッドロック修正: 連続 both-filtered カウンタ
        self._consecutive_086_wait: int = 0

    @property
    def in_filter(self) -> bool:
        return self._in_filter

    @in_filter.setter
    def in_filter(self, value: bool) -> None:
        self._in_filter = value

    @property
    def last_heartbeat_time(self) -> float:
        return self._last_heartbeat_time

    @last_heartbeat_time.setter
    def last_heartbeat_time(self, value: float) -> None:
        self._last_heartbeat_time = value

    @property
    def consecutive_086_wait(self) -> int:
        return self._consecutive_086_wait

    @consecutive_086_wait.setter
    def consecutive_086_wait(self, value: int) -> None:
        self._consecutive_086_wait = value

    def is_filtered(self, side: str | None = None) -> bool:
        """時間帯フィルター判定.

        100# fix: グローバル + side 別リストの union で判定。
        side=None の場合はグローバルリストのみで判定。
        Returns True → 呼び出し元はスリープすべき。
        """
        if not self._config.enable_time_filter:
            return False
        current_utc_hour = datetime.now(timezone.utc).hour

        global_hours = set(self._config.skip_utc_hours or [])

        if side == "buy" and self._config.skip_utc_hours_buy is not None:
            side_hours = set(self._config.skip_utc_hours_buy)
            return current_utc_hour in (global_hours | side_hours)
        if side == "sell" and self._config.skip_utc_hours_sell is not None:
            side_hours = set(self._config.skip_utc_hours_sell)
            return current_utc_hour in (global_hours | side_hours)

        return current_utc_hour in global_hours

    def on_enter(self) -> None:
        """フィルタ突入."""
        if not self._in_filter:
            self._in_filter = True
            self._last_heartbeat_time = _time.time()
            utc_h = datetime.now(timezone.utc).hour
            logger.info(
                f"[time_filter] Entering High-AS zone (UTC {utc_h}h) "
                f"— both sides filtered, suppressing cycles"
            )

    def on_exit(self) -> None:
        """フィルタ離脱."""
        if self._in_filter:
            self._in_filter = False
            self._consecutive_086_wait = 0
            logger.info("[time_filter] Exiting High-AS zone — resuming cycles")
