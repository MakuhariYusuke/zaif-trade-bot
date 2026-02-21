"""129# OB snapshot recorder — fill_test サイクル内で板データを raw 保存.

run_observation.py (別プロセス) 依存を排除し、fill_test 自体から
retrain_scheduler が参照する OB JSONL.gz を直接蓄積する。

書き込み先: data/v460/raw/orderbook/YYYYMMDD.jsonl.gz
フォーマット: MarketDataCollector と同一 (ts, bids, asks, exchange)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from ztb.io.jsonl_gz import append_jsonl_gz

logger = logging.getLogger(__name__)

_DEFAULT_RAW_DIR = Path("data/v460/raw")
_FLUSH_INTERVAL_SEC = 60  # 60 秒ごとにバッファ flush


class OBRecorder:
    """fill_test サイクルごとの板スナップショットを JSONL.gz で蓄積.

    バッファリング + 定期 flush で I/O 負荷を軽減。
    flush_interval 秒ごと、または明示的 flush() 呼び出しで書き出し。
    """

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
        self._buffer: list[dict] = []
        self._last_flush: float = time.time()
        self._total_written: int = 0

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
        bids: list[list[float]] | list[tuple[float, float]],
        asks: list[list[float]] | list[tuple[float, float]],
        timestamp: float | None = None,
        exchange: str = "coincheck",
    ) -> None:
        """1 件の板スナップショットをバッファに追加.

        flush_interval 経過時に自動 flush。
        """
        if not self._enabled:
            return
        ts = timestamp or time.time()
        # bids/asks を list[list] に正規化 (tuple → list for JSON serialization)
        self._buffer.append({
            "ts": ts,
            "bids": [list(b) for b in bids],
            "asks": [list(a) for a in asks],
            "exchange": exchange,
        })
        if time.time() - self._last_flush >= self._flush_interval:
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
            logger.debug(f"OB recorder: flushed {n} snapshots → {day}")
        except (OSError, TypeError, ValueError) as e:
            logger.error(f"OB recorder flush failed: {e}")
            self._buffer.clear()
            return 0
        self._buffer.clear()
        self._last_flush = time.time()
        return n
