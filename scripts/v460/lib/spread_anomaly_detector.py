"""211# P1-C: Spread Anomaly Detector — 流動性枯渇の自動検知・防御.

bid-ask spread の急拡大を検知し、流動性枯渇時の約定リスクを自動回避する。
価格が動く前に spread が拡大するケースが多く、最速の自動検知手段。

市場理論的根拠:
  **Bid-Ask Spread 理論** — Roll (1984) "A Simple Implicit Measure of the
  Effective Bid-Ask Spread in an Efficient Market".
  実効スプレッドは取引コストの最も原始的な指標。
  その急拡大は流動性供給者の撤退を示す。

  **Information-Based Spread** — Copeland & Galai (1983) "Information Effects
  on the Bid-Ask Spread".
  スプレッド拡大は情報優位トレーダーの存在確率上昇を反映。
  FROZEN レベルでの自動 halt は、逆選択コストが約定利益を
  上回る状態での取引回避に対応。

  **Amihud Illiquidity** — Amihud (2002) "Illiquidity and Stock Returns".
  流動性の低下は価格インパクトを増大させ、スプレッド異常検知は
  このリスクの早期警報として機能する。

段階的アクション:
  NORMAL — 通常運転
  WIDE   — offset を spread_ratio に連動して拡大
  DRY    — offset ×3.0 + interval ×2.0 + lot ×0.5
  FROZEN — 自動 halt (30 秒ごとに再評価)

Usage:
    from scripts.v460.lib.spread_anomaly_detector import (
        SpreadAnomalyDetector, SADConfig, SADLevel,
    )
    sad = SpreadAnomalyDetector(SADConfig())
    sad.update(spread_jpy)
    result = sad.check()
    if result.level == SADLevel.FROZEN:
        ...  # skip cycle
"""""

from __future__ import annotations

import logging
import math
import time as _time
from collections import deque
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SADLevel(Enum):
    """Spread Anomaly Detector の段階."""

    NORMAL = "normal"
    WIDE = "wide"
    DRY = "dry"
    FROZEN = "frozen"


@dataclass
class SADConfig:
    """Spread Anomaly Detector 設定."""

    enabled: bool = True
    # spread_ratio 閾値
    wide_ratio: float = 2.0
    dry_ratio: float = 4.0
    frozen_ratio: float = 8.0
    # baseline (中央値) 計算窓 (秒)
    baseline_window_sec: float = 3600.0
    # FROZEN 時の再評価間隔 (秒)
    frozen_reevaluate_sec: float = 30.0
    # WIDE 時: offset_mult = spread_ratio * wide_offset_scale
    wide_offset_scale: float = 0.5
    # DRY 時のオーバーライド
    dry_offset_mult: float = 3.0
    dry_interval_mult: float = 2.0
    dry_lot_mult: float = 0.5
    # サンプリング密度制限 (秒)
    sample_interval_sec: float = 5.0


@dataclass
class SADResult:
    """check() の結果."""

    level: SADLevel = SADLevel.NORMAL
    spread_ratio: float = 1.0
    baseline_spread: float | None = None
    current_spread: float | None = None
    offset_mult: float = 1.0
    interval_mult: float = 1.0
    lot_mult: float = 1.0
    frozen_remaining_sec: float = 0.0


@dataclass
class _SpreadSample:
    """内部用の spread サンプル."""

    timestamp: float
    spread: float


class SpreadAnomalyDetector:
    """211# P1-C: 流動性枯渇 (spread 急拡大) の自動検知・防御.

    直近 1h の spread 中央値を基準に、現在の spread の倍率 (spread_ratio) で
    段階的な防御アクションを発動する。
    """

    __slots__ = (
        "_config",
        "_spread_buffer",
        "_last_sample_ts",
        "_frozen_until",
        "_total_wides",
        "_total_drys",
        "_total_frozens",
        "_last_spread",
    )

    def __init__(self, config: SADConfig | None = None) -> None:
        self._config = config or SADConfig()
        max_samples = int(self._config.baseline_window_sec / max(self._config.sample_interval_sec, 1.0)) + 50
        self._spread_buffer: deque[_SpreadSample] = deque(maxlen=max_samples)
        self._last_sample_ts: float = 0.0
        self._frozen_until: float = 0.0
        self._total_wides: int = 0
        self._total_drys: int = 0
        self._total_frozens: int = 0
        self._last_spread: float = 0.0

    @property
    def config(self) -> SADConfig:
        return self._config

    def update(self, spread_jpy: float, timestamp: float | None = None) -> None:
        """spread 観測を追加.

        Args:
            spread_jpy: bid-ask spread (JPY)
            timestamp: epoch。None なら現在時刻。
        """
        if spread_jpy < 0 or not math.isfinite(spread_jpy):
            return
        ts = timestamp if timestamp is not None else _time.time()
        if ts - self._last_sample_ts < self._config.sample_interval_sec:
            # 密度制限: 前回から interval 未満ならスキップ (last_spread だけ更新)
            self._last_spread = spread_jpy
            return
        self._spread_buffer.append(_SpreadSample(timestamp=ts, spread=spread_jpy))
        self._last_sample_ts = ts
        self._last_spread = spread_jpy

    def check(self, timestamp: float | None = None) -> SADResult:
        """現在の spread 状態を評価.

        Returns:
            SADResult: level, offset_mult, interval_mult, lot_mult を含む結果。
        """
        if not self._config.enabled:
            return SADResult()

        now = timestamp if timestamp is not None else _time.time()

        # FROZEN クールダウン中
        if now < self._frozen_until:
            remaining = self._frozen_until - now
            return SADResult(
                level=SADLevel.FROZEN,
                spread_ratio=0.0,
                current_spread=self._last_spread,
                frozen_remaining_sec=remaining,
            )

        if len(self._spread_buffer) < 3:
            logger.debug(
                "[607#] SAD warmup: buffer=%d/3, spread protection inactive",
                len(self._spread_buffer),
            )
            return SADResult()

        # baseline: 窓内の spread 中央値
        window_start = now - self._config.baseline_window_sec
        window_spreads = [
            s.spread for s in self._spread_buffer if s.timestamp >= window_start
        ]
        if not window_spreads:
            return SADResult()

        baseline = self._median(window_spreads)
        if baseline <= 0:
            return SADResult()

        current = self._last_spread if self._last_spread > 0 else self._spread_buffer[-1].spread
        ratio = current / baseline

        result = SADResult(
            spread_ratio=ratio,
            baseline_spread=baseline,
            current_spread=current,
        )

        # 段階判定
        if ratio >= self._config.frozen_ratio:
            self._frozen_until = now + self._config.frozen_reevaluate_sec
            self._total_frozens += 1
            result.level = SADLevel.FROZEN
            result.frozen_remaining_sec = self._config.frozen_reevaluate_sec
            logger.warning(
                "[211# P1-C] SAD FROZEN: spread_ratio=%.2f (current=%.1f, "
                "baseline=%.1f), halt for %.0fs",
                ratio, current, baseline, self._config.frozen_reevaluate_sec,
            )
        elif ratio >= self._config.dry_ratio:
            self._total_drys += 1
            result.level = SADLevel.DRY
            result.offset_mult = self._config.dry_offset_mult
            result.interval_mult = self._config.dry_interval_mult
            result.lot_mult = self._config.dry_lot_mult
            logger.warning(
                "[211# P1-C] SAD DRY: spread_ratio=%.2f, "
                "offset_mult=%.1f, interval_mult=%.1f, lot_mult=%.1f",
                ratio, result.offset_mult, result.interval_mult, result.lot_mult,
            )
        elif ratio >= self._config.wide_ratio:
            self._total_wides += 1
            result.level = SADLevel.WIDE
            result.offset_mult = ratio * self._config.wide_offset_scale
            logger.info(
                "[211# P1-C] SAD WIDE: spread_ratio=%.2f, offset_mult=%.2f",
                ratio, result.offset_mult,
            )

        return result

    @staticmethod
    def _median(values: list[float]) -> float:
        """中央値を計算."""
        s = sorted(values)
        n = len(s)
        if n % 2 == 1:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2.0

    def export_state(self) -> dict[str, object]:
        """状態のエクスポート (永続化用)."""
        return {
            "frozen_until": self._frozen_until,
            "total_wides": self._total_wides,
            "total_drys": self._total_drys,
            "total_frozens": self._total_frozens,
            "spread_buffer": [
                {"ts": s.timestamp, "spread": s.spread}
                for s in self._spread_buffer
            ],
        }

    def import_state(self, state: dict[str, object]) -> None:
        """状態のインポート."""
        self._frozen_until = float(state.get("frozen_until", 0.0))
        self._total_wides = int(state.get("total_wides", 0))
        self._total_drys = int(state.get("total_drys", 0))
        self._total_frozens = int(state.get("total_frozens", 0))
        raw_buffer = state.get("spread_buffer", [])
        if isinstance(raw_buffer, list):
            self._spread_buffer.clear()
            for item in raw_buffer:
                if isinstance(item, dict) and "ts" in item and "spread" in item:
                    self._spread_buffer.append(
                        _SpreadSample(timestamp=float(item["ts"]), spread=float(item["spread"]))
                    )
            if self._spread_buffer:
                self._last_sample_ts = self._spread_buffer[-1].timestamp
                self._last_spread = self._spread_buffer[-1].spread

    @property
    def stats(self) -> dict[str, int]:
        """累積統計."""
        return {
            "total_wides": self._total_wides,
            "total_drys": self._total_drys,
            "total_frozens": self._total_frozens,
        }
