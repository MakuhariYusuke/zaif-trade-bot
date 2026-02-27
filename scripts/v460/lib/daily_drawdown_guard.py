"""168# §4.1 #3: 日次ドローダウンガード.

日次累計 PnL (bps) を追跡し、閾値を超過した場合にサイクル実行を
一時停止する。UTC 日境界で自動リセット。

既存の DrawdownController (ztb/risk/) はポートフォリオ価値ベースで
RL 訓練環境向け。本クラスは fill test / ライブ取引向けに
bps ベースの軽量ガードを提供する。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TypedDict

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DrawdownAction(TypedDict):
    """173# update_pnl() の型安全な戻り値."""

    halted: bool
    soft_triggered: bool
    daily_pnl_bps: float


@dataclass
class DailyDrawdownState:
    """日次ドローダウンの内部状態."""

    current_day: str = ""  # "YYYYMMDD" format (UTC)
    daily_pnl_bps: float = 0.0
    daily_fill_count: int = 0
    halted: bool = False
    halt_triggered_at: float | None = None  # timestamp
    total_halt_days: int = 0  # 累計 halt 発生日数
    halt_blocked_cycles: int = 0  # 173# 機会損失: halt 中にブロックされたサイクル数


class DailyDrawdownGuard:
    """日次ドローダウンガード.

    - UTC 日ごとに累計 PnL (bps) を追跡
    - 閾値 (例: -50 bps) 以下になったら halt → 日替わりまでスキップ
    - ソフト/ハード二段:
      - soft: lot 半減 (既存 soft_loss_cap と同様の思想)
      - hard: halt (サイクル実行停止)
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        hard_limit_bps: float = -50.0,
        soft_limit_bps: float = -30.0,
    ) -> None:
        self._enabled = enabled
        self._hard_limit_bps = hard_limit_bps
        self._soft_limit_bps = soft_limit_bps
        self._state = DailyDrawdownState()
        self._soft_triggered_today = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def state(self) -> DailyDrawdownState:
        return self._state

    def maybe_reset_day(self) -> bool:
        """UTC 日が変わっていればリセット. 変わった場合 True を返す."""
        if not self._enabled:
            return False
        today = self._utc_today()
        if today != self._state.current_day:
            prev_day = self._state.current_day
            prev_pnl = self._state.daily_pnl_bps
            was_halted = self._state.halted
            self._state = DailyDrawdownState(
                current_day=today,
                total_halt_days=(
                    self._state.total_halt_days + (1 if was_halted else 0)
                ),
            )
            self._soft_triggered_today = False
            if prev_day:
                logger.info(
                    f"[daily_drawdown] Day reset: {prev_day} → {today} "
                    f"(prev_pnl={prev_pnl:+.2f}bps, was_halted={was_halted})"
                )
            return True
        return False

    def update_pnl(self, pnl_bps: float) -> DrawdownAction:
        """約定後に PnL (bps) を加算し、制御アクションを返す.

        Returns:
            DrawdownAction(halted, soft_triggered, daily_pnl_bps).
        """
        if not self._enabled:
            return DrawdownAction(halted=False, soft_triggered=False, daily_pnl_bps=0.0)

        self.maybe_reset_day()
        self._state.daily_pnl_bps += pnl_bps
        self._state.daily_fill_count += 1

        result = DrawdownAction(
            halted=False,
            soft_triggered=False,
            daily_pnl_bps=self._state.daily_pnl_bps,
        )

        # Hard limit check
        if self._state.daily_pnl_bps <= self._hard_limit_bps:
            if not self._state.halted:
                self._state.halted = True
                self._state.halt_triggered_at = time.time()
                logger.warning(
                    f"[daily_drawdown] HALT: daily PnL {self._state.daily_pnl_bps:+.2f}bps "
                    f"<= hard limit {self._hard_limit_bps}bps after {self._state.daily_fill_count} fills"
                )
            result["halted"] = True

        # Soft limit check
        elif (
            self._state.daily_pnl_bps <= self._soft_limit_bps
            and not self._soft_triggered_today
        ):
            self._soft_triggered_today = True
            logger.warning(
                f"[daily_drawdown] SOFT: daily PnL {self._state.daily_pnl_bps:+.2f}bps "
                f"<= soft limit {self._soft_limit_bps}bps — requesting lot reduction"
            )
            result["soft_triggered"] = True

        return result

    def is_halted(self) -> bool:
        """現在 halt 中かどうかを返す. 日替わりリセットを先にチェック."""
        if not self._enabled:
            return False
        self.maybe_reset_day()
        if self._state.halted:
            self._state.halt_blocked_cycles += 1  # 173# 機会損失カウント
        return self._state.halted

    def get_metrics(self) -> dict[str, object]:
        """監視/レポート用メトリクス."""
        return {
            "enabled": self._enabled,
            "current_day": self._state.current_day,
            "daily_pnl_bps": round(self._state.daily_pnl_bps, 4),
            "daily_fill_count": self._state.daily_fill_count,
            "halted": self._state.halted,
            "soft_triggered": self._soft_triggered_today,
            "hard_limit_bps": self._hard_limit_bps,
            "soft_limit_bps": self._soft_limit_bps,
            "total_halt_days": self._state.total_halt_days,
            "halt_blocked_cycles": self._state.halt_blocked_cycles,  # 173#
        }

    def export_state(self) -> dict[str, object]:
        """永続化用に状態を dict でエクスポート."""
        return {
            "current_day": self._state.current_day,
            "daily_pnl_bps": self._state.daily_pnl_bps,
            "daily_fill_count": self._state.daily_fill_count,
            "halted": self._state.halted,
            "halt_triggered_at": self._state.halt_triggered_at,
            "total_halt_days": self._state.total_halt_days,
            "soft_triggered_today": self._soft_triggered_today,
            "halt_blocked_cycles": self._state.halt_blocked_cycles,  # 173#
        }

    def import_state(self, data: dict[str, object]) -> None:
        """永続化された状態を復元. 日が変わっていれば無視."""
        if not self._enabled or not data:
            return
        saved_day = str(data.get("current_day", ""))
        today = self._utc_today()
        if saved_day != today:
            logger.info(
                f"[daily_drawdown] State stale (saved={saved_day}, today={today}), skip import"
            )
            return
        self._state.current_day = saved_day
        self._state.daily_pnl_bps = float(data.get("daily_pnl_bps", 0.0))
        self._state.daily_fill_count = int(data.get("daily_fill_count", 0))
        self._state.halted = bool(data.get("halted", False))
        _raw_halt_at = data.get("halt_triggered_at")
        self._state.halt_triggered_at = float(_raw_halt_at) if _raw_halt_at is not None else None
        self._state.total_halt_days = int(data.get("total_halt_days", 0))
        self._state.halt_blocked_cycles = int(data.get("halt_blocked_cycles", 0))  # 173#
        self._soft_triggered_today = bool(data.get("soft_triggered_today", False))
        logger.info(
            f"[daily_drawdown] State restored: day={saved_day}, "
            f"pnl={self._state.daily_pnl_bps:+.2f}bps, halted={self._state.halted}"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _utc_today() -> str:
        """Current UTC date as YYYYMMDD string."""
        return datetime.now(timezone.utc).strftime("%Y%m%d")
