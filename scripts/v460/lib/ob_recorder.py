"""129# OB snapshot recorder — fill_test サイクル内で板データを raw 保存.

run_observation.py (別プロセス) 依存を排除し、fill_test 自体から
retrain_scheduler が参照する OB JSONL.gz を直接蓄積する。

書き込み先: data/v460/raw/orderbook/YYYYMMDD.jsonl.gz
フォーマット: MarketDataCollector と同一 (ts, bids, asks, exchange)
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from scripts.v460.lib.ob_utils import OrderBookLevelLike
from ztb.data.raw_paths import resolve_raw_dir
from ztb.io.jsonl_gz import append_jsonl_gz

logger = logging.getLogger(__name__)

_FLUSH_INTERVAL_SEC = 60  # 60 秒ごとにバッファ flush
_BUFFER_CAP = 10_000  # 146# §13: メモリ保護上限 (TradesRecorder と対称)


class _OBSnapshot(TypedDict):
    ts: float
    bids: list[list[float]]
    asks: list[list[float]]
    exchange: str


def _to_finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = float(stripped)
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _normalize_levels(levels: object) -> list[list[float]]:
    if not isinstance(levels, Sequence) or isinstance(levels, (str, bytes, bytearray)):
        return []

    normalized: list[list[float]] = []
    append_level = normalized.append
    for level in levels:
        if isinstance(level, (list, tuple)):
            if len(level) < 2:
                continue
            price = _to_finite_float(level[0])
            size = _to_finite_float(level[1])
        elif isinstance(level, OrderBookLevelLike):
            # Protocol 準拠: 直接属性アクセス → _to_finite_float で型検証
            price = _to_finite_float(level.price)
            size = _to_finite_float(level.quantity)
        else:
            continue
        if price is None or size is None:
            continue
        append_level([price, size])
    return normalized


class OBRecorder:
    """fill_test サイクルごとの板スナップショットを JSONL.gz で蓄積.

    バッファリング + 定期 flush で I/O 負荷を軽減。
    flush_interval 秒ごと、または明示的 flush() 呼び出しで書き出し。
    146# §13: BUFFER_CAP 超過時は強制 flush してメモリ保護。
    """

    __slots__ = (
        "_raw_dir",
        "_flush_interval",
        "_enabled",
        "_buffer",
        "_last_flush",
        "_total_written",
        "_flush_fail_count",
    )

    def __init__(
        self,
        raw_dir: Path | None = None,
        flush_interval: int = _FLUSH_INTERVAL_SEC,
        *,
        enabled: bool = True,
    ) -> None:
        self._raw_dir = resolve_raw_dir(raw_dir)
        self._flush_interval = flush_interval
        self._enabled = enabled
        self._buffer: list[_OBSnapshot] = []
        self._last_flush: float = time.time()
        self._total_written: int = 0
        self._flush_fail_count: int = 0  # 146# §13: consecutive-fail counter

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def total_written(self) -> int:
        return self._total_written

    def record(
        self,
        bids: object,
        asks: object,
        timestamp: object | None = None,
        exchange: object = "coincheck",
    ) -> None:
        """1 件の板スナップショットをバッファに追加.

        flush_interval 経過時に自動 flush。
        """
        if not self._enabled:
            return
        ts = _to_finite_float(timestamp)
        now_ts = time.time()
        if ts is None:
            ts = now_ts
        norm_bids = _normalize_levels(bids)
        norm_asks = _normalize_levels(asks)
        # 不完全なスナップショットは捨てる (flush 失敗の連鎖を防ぐ)
        if not norm_bids or not norm_asks:
            logger.debug("OB recorder: skipped malformed snapshot")
            return
        self._buffer.append(
            {
                "ts": ts,
                "bids": norm_bids,
                "asks": norm_asks,
                "exchange": str(exchange),
            }
        )
        # 146# §13: バッファ上限 or flush_interval のいずれかで flush
        if len(self._buffer) >= _BUFFER_CAP:
            logger.warning(
                f"OB recorder: buffer cap reached ({_BUFFER_CAP}), forcing flush"
            )
            self.flush()
        elif now_ts - self._last_flush >= self._flush_interval:
            self.flush()

    def flush(self) -> int:
        """バッファを JSONL.gz にフラッシュ.

        Returns:
            フラッシュしたレコード数.
        """
        if not self._buffer:
            return 0
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        ob_dir = self._raw_dir / "orderbook"
        ob_dir.mkdir(parents=True, exist_ok=True)
        path = ob_dir / f"{day}.jsonl.gz"
        n = len(self._buffer)
        try:
            append_jsonl_gz(path, self._buffer)
            self._total_written += n
            self._flush_fail_count = 0
            logger.debug(f"OB recorder: flushed {n} snapshots \u2192 {day}")
        except (OSError, TypeError, ValueError) as e:
            self._flush_fail_count += 1
            logger.error(
                f"OB recorder flush failed ({self._flush_fail_count}/3): {e}"
            )
            if self._flush_fail_count >= 3:
                logger.error(
                    f"OB recorder: 3 consecutive flush failures \u2014 "
                    f"dropping {n} buffered snapshots"
                )
                self._buffer.clear()
                self._flush_fail_count = 0
            return 0
        self._buffer.clear()
        self._last_flush = time.time()
        return n
