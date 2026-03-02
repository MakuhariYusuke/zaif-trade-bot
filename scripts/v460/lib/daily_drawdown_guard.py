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
    side_halted: str  # 205# §9.5: "" = 無し, "buy"/"sell" = 片側封鎖


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
    # 205# §9.5: 片側累積 PnL 追跡
    daily_pnl_bps_buy: float = 0.0
    daily_pnl_bps_sell: float = 0.0
    side_halted_buy: bool = False
    side_halted_sell: bool = False
    side_halt_remaining_buy: int = 0  # 残存封鎖サイクル数 (0=永続)
    side_halt_remaining_sell: int = 0
    # 224# B1: halt解除後ソフトリカバリ — 残存リカバリサイクル
    side_recovery_remaining_buy: int = 0
    side_recovery_remaining_sell: int = 0


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
        per_side_enabled: bool = False,
        per_side_hard_limit_bps: float = -30.0,
        per_side_halt_cycles: int = 0,
        per_side_recovery_cycles: int = 5,
        per_side_recovery_lot_scale: float = 0.5,
    ) -> None:
        self._enabled = enabled
        self._hard_limit_bps = hard_limit_bps
        self._soft_limit_bps = soft_limit_bps
        self._per_side_enabled = per_side_enabled
        self._per_side_hard_limit_bps = per_side_hard_limit_bps
        self._per_side_halt_cycles = per_side_halt_cycles
        # 224# B1: halt解除後ソフトリカバリ
        self._per_side_recovery_cycles = per_side_recovery_cycles
        self._per_side_recovery_lot_scale = per_side_recovery_lot_scale
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
            # 225# 日替わり clarification:
            # DailyDrawdownState() の初期化で side_recovery_remaining_buy/sell も 0 に
            # リセットされる。これは意図的な設計 — 日替わりで前日の halt/recovery は
            # 全て無効化し、新たな daily PnL 計上からやり直す。
            self._soft_triggered_today = False
            if prev_day:
                logger.info(
                    f"[daily_drawdown] Day reset: {prev_day} → {today} "
                    f"(prev_pnl={prev_pnl:+.2f}bps, was_halted={was_halted})"
                )
            return True
        return False

    def update_pnl(self, pnl_bps: float, *, side: str = "") -> DrawdownAction:
        """約定後に PnL (bps) を加算し、制御アクションを返す.

        Args:
            pnl_bps: 約定の PnL (bps).
            side: "buy" or "sell" (片側 DD 追跡用. 空文字列の場合は集約のみ).

        Returns:
            DrawdownAction(halted, soft_triggered, daily_pnl_bps, side_halted).
        """
        if not self._enabled:
            return DrawdownAction(halted=False, soft_triggered=False, daily_pnl_bps=0.0, side_halted="")

        self.maybe_reset_day()
        self._state.daily_pnl_bps += pnl_bps
        self._state.daily_fill_count += 1

        result = DrawdownAction(
            halted=False,
            soft_triggered=False,
            daily_pnl_bps=self._state.daily_pnl_bps,
            side_halted="",
        )

        # 205# §9.5: 片側 PnL 追跡
        if side == "buy":
            self._state.daily_pnl_bps_buy += pnl_bps
        elif side == "sell":
            self._state.daily_pnl_bps_sell += pnl_bps

        # 215# fix: soft trigger は hard halt と独立に評価する。
        # 旧: if/elif で hard halt 時に soft が skip される → import_state 時に
        # soft_triggered_today=false のまま復元され、halt 解除後に soft 防御が未適用。
        # Soft limit check (aggregate) — hard よりも先に評価
        if (
            self._state.daily_pnl_bps <= self._soft_limit_bps
            and not self._soft_triggered_today
        ):
            self._soft_triggered_today = True
            logger.warning(
                f"[daily_drawdown] SOFT: daily PnL {self._state.daily_pnl_bps:+.2f}bps "
                f"<= soft limit {self._soft_limit_bps}bps — requesting lot reduction"
            )
            result["soft_triggered"] = True

        # Hard limit check (aggregate)
        if self._state.daily_pnl_bps <= self._hard_limit_bps:
            if not self._state.halted:
                self._state.halted = True
                self._state.halt_triggered_at = time.time()
                logger.warning(
                    f"[daily_drawdown] HALT: daily PnL {self._state.daily_pnl_bps:+.2f}bps "
                    f"<= hard limit {self._hard_limit_bps}bps after {self._state.daily_fill_count} fills"
                )
            result["halted"] = True

        # 205# §9.5: 片側 DD Halt チェック
        if self._per_side_enabled and side in ("buy", "sell"):
            side_pnl = (
                self._state.daily_pnl_bps_buy if side == "buy"
                else self._state.daily_pnl_bps_sell
            )
            is_halted = (
                self._state.side_halted_buy if side == "buy"
                else self._state.side_halted_sell
            )
            if side_pnl <= self._per_side_hard_limit_bps and not is_halted:
                if side == "buy":
                    self._state.side_halted_buy = True
                    self._state.side_halt_remaining_buy = self._per_side_halt_cycles
                else:
                    self._state.side_halted_sell = True
                    self._state.side_halt_remaining_sell = self._per_side_halt_cycles
                logger.warning(
                    f"[daily_drawdown] PER-SIDE HALT: {side} PnL {side_pnl:+.2f}bps "
                    f"<= {self._per_side_hard_limit_bps}bps — {side} 封鎖 "
                    f"(cycles={self._per_side_halt_cycles or 'until_day_reset'})"
                )
                result["side_halted"] = side

        return result

    def is_halted(self) -> bool:
        """現在 halt 中かどうかを返す. 日替わりリセットを先にチェック."""
        if not self._enabled:
            return False
        self.maybe_reset_day()
        if self._state.halted:
            self._state.halt_blocked_cycles += 1  # 173# 機会損失カウント
        return self._state.halted

    def is_side_halted(self, side: str) -> bool:
        """205# §9.5: 指定サイドが片側封鎖中かどうかを返す."""
        if not self._per_side_enabled:
            return False
        if side == "buy":
            return self._state.side_halted_buy
        elif side == "sell":
            return self._state.side_halted_sell
        return False

    def tick_side_halt(self) -> None:
        """205# §9.5: 片側封鎖のサイクルカウンタをデクリメント (毎サイクル呼出し).

        halt_cycles > 0 の場合、残存サイクルが 0 になったら自動解除。
        halt_cycles == 0 の場合は日替わりまで永続封鎖 (maybe_reset_day で解除)。
        """
        if not self._per_side_enabled:
            return
        if self._state.side_halted_buy and self._per_side_halt_cycles > 0:
            self._state.side_halt_remaining_buy = max(0, self._state.side_halt_remaining_buy - 1)
            if self._state.side_halt_remaining_buy == 0:
                self._state.side_halted_buy = False
                # 224# B1: halt解除 → リカバリ期間開始
                self._state.side_recovery_remaining_buy = self._per_side_recovery_cycles
                logger.info(
                    f"[daily_drawdown] Per-side halt released: buy (cycles exhausted), "
                    f"recovery={self._per_side_recovery_cycles} cycles"
                )
        if self._state.side_halted_sell and self._per_side_halt_cycles > 0:
            self._state.side_halt_remaining_sell = max(0, self._state.side_halt_remaining_sell - 1)
            if self._state.side_halt_remaining_sell == 0:
                self._state.side_halted_sell = False
                # 224# B1: halt解除 → リカバリ期間開始
                self._state.side_recovery_remaining_sell = self._per_side_recovery_cycles
                logger.info(
                    f"[daily_drawdown] Per-side halt released: sell (cycles exhausted), "
                    f"recovery={self._per_side_recovery_cycles} cycles"
                )

    def get_recovery_lot_scale(self, side: str) -> float:
        """224# B1: halt解除後のリカバリ期間中の lot 縮小倍率を返す.

        リカバリ残サイクル > 0 の場合、デクリメントして縮小倍率を返す。
        リカバリ期間外は 1.0 を返す。毎サイクル1回だけ呼ぶこと。
        """
        if not self._per_side_enabled or self._per_side_recovery_cycles <= 0:
            return 1.0
        if side == "buy" and self._state.side_recovery_remaining_buy > 0:
            self._state.side_recovery_remaining_buy -= 1
            logger.info(
                f"[224# B1] Recovery active: buy lot_scale={self._per_side_recovery_lot_scale}, "
                f"remaining={self._state.side_recovery_remaining_buy}"
            )
            return self._per_side_recovery_lot_scale
        if side == "sell" and self._state.side_recovery_remaining_sell > 0:
            self._state.side_recovery_remaining_sell -= 1
            logger.info(
                f"[224# B1] Recovery active: sell lot_scale={self._per_side_recovery_lot_scale}, "
                f"remaining={self._state.side_recovery_remaining_sell}"
            )
            return self._per_side_recovery_lot_scale
        return 1.0

    def restore_recovery_counter(self, side: str) -> None:
        """225# 6.1: 例外でサイクルが中断された場合にリカバリカウンタを復元.

        get_recovery_lot_scale() でデクリメント済みのカウンタを +1 戻す。
        """
        if side == "buy":
            self._state.side_recovery_remaining_buy += 1
        elif side == "sell":
            self._state.side_recovery_remaining_sell += 1

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
            # 205# §9.5: 片側 DD
            "per_side_enabled": self._per_side_enabled,
            "daily_pnl_bps_buy": round(self._state.daily_pnl_bps_buy, 4),
            "daily_pnl_bps_sell": round(self._state.daily_pnl_bps_sell, 4),
            "side_halted_buy": self._state.side_halted_buy,
            "side_halted_sell": self._state.side_halted_sell,
            # 224# B1: リカバリ状態
            "side_recovery_remaining_buy": self._state.side_recovery_remaining_buy,
            "side_recovery_remaining_sell": self._state.side_recovery_remaining_sell,
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
            # 205# §9.5: 片側 DD 永続化
            "daily_pnl_bps_buy": self._state.daily_pnl_bps_buy,
            "daily_pnl_bps_sell": self._state.daily_pnl_bps_sell,
            "side_halted_buy": self._state.side_halted_buy,
            "side_halted_sell": self._state.side_halted_sell,
            "side_halt_remaining_buy": self._state.side_halt_remaining_buy,
            "side_halt_remaining_sell": self._state.side_halt_remaining_sell,
            # 224# B1: リカバリ状態永続化
            "side_recovery_remaining_buy": self._state.side_recovery_remaining_buy,
            "side_recovery_remaining_sell": self._state.side_recovery_remaining_sell,
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
        # 205# §9.5: 片側 DD 状態復元
        self._state.daily_pnl_bps_buy = float(data.get("daily_pnl_bps_buy", 0.0))
        self._state.daily_pnl_bps_sell = float(data.get("daily_pnl_bps_sell", 0.0))
        self._state.side_halted_buy = bool(data.get("side_halted_buy", False))
        self._state.side_halted_sell = bool(data.get("side_halted_sell", False))
        self._state.side_halt_remaining_buy = int(data.get("side_halt_remaining_buy", 0))
        self._state.side_halt_remaining_sell = int(data.get("side_halt_remaining_sell", 0))
        # 224# B1: リカバリ状態復元
        self._state.side_recovery_remaining_buy = int(data.get("side_recovery_remaining_buy", 0))
        self._state.side_recovery_remaining_sell = int(data.get("side_recovery_remaining_sell", 0))
        logger.info(
            f"[daily_drawdown] State restored: day={saved_day}, "
            f"pnl={self._state.daily_pnl_bps:+.2f}bps, halted={self._state.halted}, "
            f"buy_pnl={self._state.daily_pnl_bps_buy:+.2f}bps, "
            f"sell_pnl={self._state.daily_pnl_bps_sell:+.2f}bps"
        )

    def needs_warmup_repair(self) -> bool:
        """215# P0-A: import_state 後の整合性検証.

        per-side PnL が 0.0 のまま合計 PnL が有意な値を持つ場合、
        または soft_triggered_today が整合しない場合に True を返す。
        呼び出し元で _warmup_daily_drawdown_from_records() を発動する。
        """
        if not self._enabled:
            return False
        s = self._state
        if s.daily_fill_count == 0:
            return False  # fill なしなら warmup 不要
        # Case 1: per-side PnL が両方 0.0 だが合計が有意
        _per_side_sum = abs(s.daily_pnl_bps_buy) + abs(s.daily_pnl_bps_sell)
        if _per_side_sum < 0.01 and abs(s.daily_pnl_bps) >= 1.0:
            logger.warning(
                f"[215# P0-A] DD state inconsistency detected: "
                f"daily_pnl={s.daily_pnl_bps:+.2f}bps but "
                f"buy={s.daily_pnl_bps_buy:+.2f}/sell={s.daily_pnl_bps_sell:+.2f}"
            )
            return True
        # Case 2: soft_triggered_today=false だが daily_pnl が soft limit 以下
        if (
            not self._soft_triggered_today
            and s.daily_pnl_bps <= self._soft_limit_bps
            and s.daily_fill_count > 0
        ):
            logger.warning(
                f"[215# P0-A] DD soft_triggered inconsistency: "
                f"soft_triggered=false but pnl={s.daily_pnl_bps:+.2f}bps "
                f"<= soft_limit={self._soft_limit_bps}bps"
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _utc_today() -> str:
        """Current UTC date as YYYYMMDD string."""
        return datetime.now(timezone.utc).strftime("%Y%m%d")
