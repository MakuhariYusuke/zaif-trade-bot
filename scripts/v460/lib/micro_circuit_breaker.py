"""211# P1-B: Micro Circuit Breaker — 短期価格急変の自動検知・防御.

regime_detector (40分 hysteresis) と Volatility Guard (4h ATR) の間を埋める
5分〜1時間の急変検知レイヤー。

段階的アクション:
  CAUTION  — ログ警告のみ
  WARNING  — offset_mult / interval_mult を拡大
  HALT     — 自動 halt (cooldown 後に再評価)

Usage:
    from scripts.v460.lib.micro_circuit_breaker import (
        MicroCircuitBreaker, MCBConfig, MCBLevel,
    )
    mcb = MicroCircuitBreaker(MCBConfig())
    mcb.update(mid_price, timestamp)
    result = mcb.check()
    if result.level == MCBLevel.HALT:
        ...  # skip cycle
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MCBLevel(Enum):
    """Micro Circuit Breaker の段階."""

    NORMAL = "normal"
    CAUTION = "caution"
    WARNING = "warning"
    HALT = "halt"


@dataclass
class MCBConfig:
    """Micro Circuit Breaker 設定."""

    enabled: bool = True
    # 各窓のサイズ (秒)
    window_5m_sec: int = 300
    window_15m_sec: int = 900
    window_1h_sec: int = 3600
    # σ 閾値
    caution_sigma: float = 1.0
    warning_sigma: float = 1.5
    halt_sigma: float = 2.0
    # WARNING/HALT 時の窓数条件 (2窓以上で昇格)
    escalation_window_count: int = 2
    # 単一窓の高σ閾値 (1窓でもこれ超えたら昇格)
    single_window_warning_sigma: float = 2.0
    single_window_halt_sigma: float = 3.0
    # HALT 時のクールダウン (秒)
    halt_cooldown_sec: float = 300.0
    # WARNING 時のオーバーライド
    warning_offset_mult: float = 1.5
    warning_interval_mult: float = 2.0
    # warmup 不足時のデフォルト閾値 (%)
    default_threshold_5m_pct: float = 0.5
    default_threshold_15m_pct: float = 1.0
    default_threshold_1h_pct: float = 2.0
    # 24h baseline 用のサンプリング間隔 (秒) — 記録密度制限
    baseline_sample_interval_sec: float = 10.0


@dataclass
class MCBResult:
    """check() の結果."""

    level: MCBLevel = MCBLevel.NORMAL
    offset_mult: float = 1.0
    interval_mult: float = 1.0
    # 各窓の変動率 (%)
    change_5m_pct: float | None = None
    change_15m_pct: float | None = None
    change_1h_pct: float | None = None
    # 各窓のσ値
    sigma_5m: float | None = None
    sigma_15m: float | None = None
    sigma_1h: float | None = None
    # クールダウン残り (秒)
    cooldown_remaining_sec: float = 0.0


@dataclass
class _PriceSample:
    """内部用の価格サンプル."""

    timestamp: float
    price: float


class MicroCircuitBreaker:
    """211# P1-B: 短期価格急変の自動検知・防御.

    複数時間軸 (5分/15分/1時間) の価格変動率を監視し、
    24時間の rolling σ 基準で異常急変を検知する。
    """

    __slots__ = (
        "_config",
        "_price_buffer",
        "_change_history_5m",
        "_change_history_15m",
        "_change_history_1h",
        "_last_sample_ts",
        "_halt_until",
        "_total_cautions",
        "_total_warnings",
        "_total_halts",
    )

    def __init__(self, config: MCBConfig | None = None) -> None:
        self._config = config or MCBConfig()
        # 直近 1h+α の価格を保持 (最大 3600/sample_interval + バッファ)
        max_samples = int(3600 / max(self._config.baseline_sample_interval_sec, 1.0)) + 100
        self._price_buffer: deque[_PriceSample] = deque(maxlen=max_samples)
        # 24h分の変動率履歴 (σ計算用)
        # 5m窓: 24h/5m = 288 サンプル
        self._change_history_5m: deque[float] = deque(maxlen=300)
        # 15m窓: 24h/15m = 96 サンプル
        self._change_history_15m: deque[float] = deque(maxlen=100)
        # 1h窓: 24h/1h = 24 サンプル
        self._change_history_1h: deque[float] = deque(maxlen=30)
        self._last_sample_ts: float = 0.0
        self._halt_until: float = 0.0
        self._total_cautions: int = 0
        self._total_warnings: int = 0
        self._total_halts: int = 0

    @property
    def config(self) -> MCBConfig:
        return self._config

    def update(self, mid_price: float, timestamp: float) -> None:
        """価格観測を追加.

        baseline_sample_interval_sec 未満の間隔ではスキップ (記録密度制限)。
        """
        if mid_price <= 0 or not math.isfinite(mid_price):
            return
        interval = self._config.baseline_sample_interval_sec
        if timestamp - self._last_sample_ts < interval:
            return
        self._price_buffer.append(_PriceSample(timestamp=timestamp, price=mid_price))
        self._last_sample_ts = timestamp

    def check(self, timestamp: float | None = None) -> MCBResult:
        """現在の価格変動状態を評価.

        Args:
            timestamp: 現在時刻 (epoch)。None の場合は最新サンプルの ts を使用。

        Returns:
            MCBResult: level, offset_mult, interval_mult を含む結果。
        """
        if not self._config.enabled:
            return MCBResult()

        if not self._price_buffer:
            return MCBResult()

        now = timestamp if timestamp is not None else self._price_buffer[-1].timestamp

        # HALT クールダウン中
        if now < self._halt_until:
            remaining = self._halt_until - now
            return MCBResult(
                level=MCBLevel.HALT,
                offset_mult=self._config.warning_offset_mult,
                interval_mult=self._config.warning_interval_mult,
                cooldown_remaining_sec=remaining,
            )

        # 各窓の変動率を計算
        change_5m = self._calc_change_pct(now, self._config.window_5m_sec)
        change_15m = self._calc_change_pct(now, self._config.window_15m_sec)
        change_1h = self._calc_change_pct(now, self._config.window_1h_sec)

        # 変動率履歴に追加 (σ計算用)
        if change_5m is not None:
            self._change_history_5m.append(change_5m)
        if change_15m is not None:
            self._change_history_15m.append(change_15m)
        if change_1h is not None:
            self._change_history_1h.append(change_1h)

        # σ計算 or デフォルト閾値
        threshold_5m = self._calc_threshold(
            self._change_history_5m, self._config.default_threshold_5m_pct
        )
        threshold_15m = self._calc_threshold(
            self._change_history_15m, self._config.default_threshold_15m_pct
        )
        threshold_1h = self._calc_threshold(
            self._change_history_1h, self._config.default_threshold_1h_pct
        )

        # 各窓のσ値
        sigma_5m = abs(change_5m) / threshold_5m if change_5m is not None and threshold_5m > 0 else None
        sigma_15m = abs(change_15m) / threshold_15m if change_15m is not None and threshold_15m > 0 else None
        sigma_1h = abs(change_1h) / threshold_1h if change_1h is not None and threshold_1h > 0 else None

        result = MCBResult(
            change_5m_pct=change_5m,
            change_15m_pct=change_15m,
            change_1h_pct=change_1h,
            sigma_5m=sigma_5m,
            sigma_15m=sigma_15m,
            sigma_1h=sigma_1h,
        )

        # 段階判定
        sigmas = [s for s in (sigma_5m, sigma_15m, sigma_1h) if s is not None]
        if not sigmas:
            return result

        level = self._determine_level(sigmas)
        result.level = level

        if level == MCBLevel.HALT:
            self._halt_until = now + self._config.halt_cooldown_sec
            self._total_halts += 1
            result.offset_mult = self._config.warning_offset_mult
            result.interval_mult = self._config.warning_interval_mult
            result.cooldown_remaining_sec = self._config.halt_cooldown_sec
            logger.warning(
                "[211# P1-B] MCB HALT: "
                "5m=%s%% 15m=%s%% 1h=%s%%, "
                "σ=(%s, %s, %s), "
                "cooldown=%ss",
                f"{change_5m:.3f}" if change_5m is not None else "N/A",
                f"{change_15m:.3f}" if change_15m is not None else "N/A",
                f"{change_1h:.3f}" if change_1h is not None else "N/A",
                f"{sigma_5m:.2f}" if sigma_5m is not None else "N/A",
                f"{sigma_15m:.2f}" if sigma_15m is not None else "N/A",
                f"{sigma_1h:.2f}" if sigma_1h is not None else "N/A",
                self._config.halt_cooldown_sec,
            )
        elif level == MCBLevel.WARNING:
            self._total_warnings += 1
            result.offset_mult = self._config.warning_offset_mult
            result.interval_mult = self._config.warning_interval_mult
            logger.warning(
                f"[211# P1-B] MCB WARNING: "
                f"5m={change_5m} 15m={change_15m} 1h={change_1h}, "
                f"offset_mult={result.offset_mult}, interval_mult={result.interval_mult}"
            )
        elif level == MCBLevel.CAUTION:
            self._total_cautions += 1
            logger.info(
                f"[211# P1-B] MCB CAUTION: "
                f"5m={change_5m} 15m={change_15m} 1h={change_1h}"
            )

        return result

    def _determine_level(self, sigmas: list[float]) -> MCBLevel:
        """σ値リストから MCBLevel を決定."""
        cfg = self._config

        # HALT 条件: 2窓以上で > halt_sigma, or 1窓で > single_window_halt_sigma
        count_halt = sum(1 for s in sigmas if s > cfg.halt_sigma)
        max_sigma = max(sigmas)
        if count_halt >= cfg.escalation_window_count or max_sigma > cfg.single_window_halt_sigma:
            return MCBLevel.HALT

        # WARNING 条件: 2窓以上で > warning_sigma, or 1窓で > single_window_warning_sigma
        count_warning = sum(1 for s in sigmas if s > cfg.warning_sigma)
        if count_warning >= cfg.escalation_window_count or max_sigma > cfg.single_window_warning_sigma:
            return MCBLevel.WARNING

        # CAUTION 条件: いずれか1窓で > caution_sigma
        count_caution = sum(1 for s in sigmas if s > cfg.caution_sigma)
        if count_caution > 0:
            return MCBLevel.CAUTION

        return MCBLevel.NORMAL

    def _calc_change_pct(self, now: float, window_sec: int) -> float | None:
        """指定窓での価格変動率 (%) を計算.

        窓の開始付近の価格と最新価格を比較。
        """
        if not self._price_buffer:
            return None
        latest = self._price_buffer[-1]
        target_ts = now - window_sec
        # target_ts に最も近いサンプルを探す (buffer は時系列順)
        best: _PriceSample | None = None
        for sample in self._price_buffer:
            if sample.timestamp <= target_ts:
                best = sample
            elif best is not None:
                break  # target_ts を超えたら停止 (それ以降はより新しい)
        if best is None:
            return None
        if best.price <= 0:
            return None
        return ((latest.price - best.price) / best.price) * 100.0

    @staticmethod
    def _calc_threshold(history: deque[float], default_pct: float) -> float:
        """変動率履歴からσ (標準偏差) を計算. サンプル不足時はデフォルト値."""
        min_samples = 10
        if len(history) < min_samples:
            return default_pct
        # σ = std of absolute changes
        abs_changes = [abs(c) for c in history]
        n = len(abs_changes)
        mean = sum(abs_changes) / n
        variance = sum((x - mean) ** 2 for x in abs_changes) / n
        sigma = math.sqrt(variance) if variance > 0 else default_pct
        # σ が極端に小さい場合 (flat market) はデフォルト閾値を下限に
        return max(sigma, default_pct * 0.1)

    def export_state(self) -> dict[str, object]:
        """状態のエクスポート (永続化用)."""
        return {
            "halt_until": self._halt_until,
            "total_cautions": self._total_cautions,
            "total_warnings": self._total_warnings,
            "total_halts": self._total_halts,
            "price_buffer": [
                {"ts": s.timestamp, "price": s.price}
                for s in self._price_buffer
            ],
        }

    def import_state(self, state: dict[str, object]) -> None:
        """状態のインポート (永続化からの復元用)."""
        self._halt_until = float(state.get("halt_until", 0.0))
        self._total_cautions = int(state.get("total_cautions", 0))
        self._total_warnings = int(state.get("total_warnings", 0))
        self._total_halts = int(state.get("total_halts", 0))
        raw_buffer = state.get("price_buffer", [])
        if isinstance(raw_buffer, list):
            self._price_buffer.clear()
            for item in raw_buffer:
                if isinstance(item, dict) and "ts" in item and "price" in item:
                    self._price_buffer.append(
                        _PriceSample(timestamp=float(item["ts"]), price=float(item["price"]))
                    )
            if self._price_buffer:
                self._last_sample_ts = self._price_buffer[-1].timestamp

    @property
    def is_halted(self) -> bool:
        """現在 HALT 中かどうか."""
        import time as _time
        return _time.time() < self._halt_until

    @property
    def stats(self) -> dict[str, int]:
        """累積統計."""
        return {
            "total_cautions": self._total_cautions,
            "total_warnings": self._total_warnings,
            "total_halts": self._total_halts,
        }
