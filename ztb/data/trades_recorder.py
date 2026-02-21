"""135# P0-04: Trades snapshot recorder — fill_test サイクル内で約定データを raw 保存.

OBRecorder (129#) と対称的な設計。fill_test サイクルごとに adapter 経由で
取得した recent trades を data/v460/raw/trades/YYYYMMDD.jsonl.gz に蓄積する。

- バッファリング + 定期 flush で I/O 負荷を軽減
- 重複排除 (ts+price+amount+side の composite key)
- メモリ保護 (バッファ上限)
- feature_enricher / retrain_scheduler が参照する trades JSONL.gz と同一フォーマット

書き込み先: data/v460/raw/trades/YYYYMMDD.jsonl.gz
フォーマット: MarketDataCollector と同一 (ts, price, amount, side)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from ztb.io.jsonl_gz import append_jsonl_gz

logger = logging.getLogger(__name__)

_DEFAULT_RAW_DIR = Path("data/v460/raw")
_FLUSH_INTERVAL_SEC = 60
_BUFFER_CAP = 10_000  # メモリ保護: 上限到達で強制 flush


class TradeEntry(NamedTuple):
    """重複排除用の composite key."""

    ts: float
    price: float
    amount: float
    side: str


class TradesRecorder:
    """fill_test サイクルごとの約定データを JSONL.gz で蓄積.

    OBRecorder と同一のバッファリング + 定期 flush パターン。
    重複排除により、同一 trades が複数回 record() されても1回だけ書き込む。
    """

    __slots__ = (
        "_raw_dir",
        "_flush_interval",
        "_enabled",
        "_buffer",
        "_last_flush",
        "_total_written",
        "_seen_keys",
        "_last_trade_key",
    )

    def __init__(
        self,
        raw_dir: Path | None = None,
        flush_interval: int = _FLUSH_INTERVAL_SEC,
        *,
        enabled: bool = True,
    ) -> None:
        self._raw_dir = raw_dir or _DEFAULT_RAW_DIR
        self._flush_interval = flush_interval
        self._enabled = enabled
        self._buffer: list[dict[str, object]] = []
        self._last_flush: float = time.time()
        self._total_written: int = 0
        # 重複排除: 直近の trade key で前方比較
        self._last_trade_key: TradeEntry | None = None
        # 同一 flush 内の重複防止用 set (flush ごとにリセット)
        self._seen_keys: set[TradeEntry] = set()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def total_written(self) -> int:
        return self._total_written

    def record_trades(
        self,
        trades: list[dict[str, object]],
    ) -> int:
        """約定データのリストをバッファに追加 (重複排除付き).

        Args:
            trades: [{"ts": float, "price": float, "amount": float, "side": str}, ...]
                MarketDataCollector / adapter.get_recent_trades と同一形式.

        Returns:
            新規追加されたレコード数.
        """
        if not self._enabled or not trades:
            return 0

        added = 0
        for t in trades:
            key = TradeEntry(
                ts=float(t.get("ts", 0)),
                price=float(t.get("price", 0)),
                amount=float(t.get("amount", 0)),
                side=str(t.get("side", "")),
            )
            # 時系列順で古いものはスキップ
            if self._last_trade_key is not None and key[:3] <= self._last_trade_key[:3]:
                continue
            # 同一 flush バッチ内の重複
            if key in self._seen_keys:
                continue
            self._seen_keys.add(key)
            self._buffer.append({
                "ts": key.ts,
                "price": key.price,
                "amount": key.amount,
                "side": key.side,
            })
            added += 1

        # メモリ保護: バッファ上限
        if len(self._buffer) >= _BUFFER_CAP:
            self.flush()
        elif time.time() - self._last_flush >= self._flush_interval:
            self.flush()

        return added

    def record_from_adapter(
        self,
        trade_records: object,
    ) -> int:
        """adapter.get_recent_trades() の TradeRecord リストから記録.

        Args:
            trade_records: list[TradeRecord] (broker_interfaces.TradeRecord 互換).
                各要素は .timestamp, .price, .amount, .side 属性を持つ.

        Returns:
            新規追加されたレコード数.
        """
        if not self._enabled:
            return 0
        # TradeRecord → dict 変換 (型に依存しない duck-typing)
        dicts: list[dict[str, object]] = []
        for tr in trade_records:  # type: ignore[union-attr]
            dicts.append({
                "ts": getattr(tr, "timestamp", 0.0),
                "price": getattr(tr, "price", 0.0),
                "amount": getattr(tr, "amount", 0.0),
                "side": getattr(tr, "side", ""),
            })
        return self.record_trades(dicts)

    def flush(self) -> int:
        """バッファを JSONL.gz にフラッシュ.

        Returns:
            フラッシュしたレコード数.
        """
        if not self._buffer:
            return 0
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        tr_dir = self._raw_dir / "trades"
        tr_dir.mkdir(parents=True, exist_ok=True)
        path = tr_dir / f"{day}.jsonl.gz"
        n = len(self._buffer)
        try:
            append_jsonl_gz(path, self._buffer)
            self._total_written += n
            # 最新 key を更新
            last = self._buffer[-1]
            self._last_trade_key = TradeEntry(
                ts=float(last["ts"]),
                price=float(last["price"]),
                amount=float(last["amount"]),
                side=str(last["side"]),
            )
            logger.debug(f"Trades recorder: flushed {n} trades → {day}")
        except (OSError, TypeError, ValueError) as e:
            logger.error(f"Trades recorder flush failed: {e}")
            self._buffer.clear()
            self._seen_keys.clear()
            return 0
        self._buffer.clear()
        self._seen_keys.clear()
        self._last_flush = time.time()
        return n
