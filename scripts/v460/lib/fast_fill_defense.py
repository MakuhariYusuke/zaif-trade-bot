"""100# FastFillDefense — side-aware 即約定防御モジュール.

run_fill_test.py からの God Object 分割:
- per-side boost 状態管理 (sell fast-fill が次 buy に伝播する問題を解消)
- two-layer 負エッジ検出 (即時 proxy + post-fill PnL)
- side-specific boost cap (sell offset 0.12 に対し common 0.05 で計算していた問題を修正)

098# §3.1, 099# §2 で特定されたP0問題への対応。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FastFillDefenseConfig:
    """即約定防御の設定 (fill_test.yaml fast_fill_defense セクション対応)."""

    enabled: bool = False
    threshold_sec: float = 5.0
    threshold_sec_buy: Optional[float] = None
    threshold_sec_sell: Optional[float] = None
    offset_boost: float = 2.0
    offset_boost_buy: Optional[float] = None
    offset_boost_sell: Optional[float] = None
    # 102# YAML化: offset 上限・下限
    max_offset_ratio: float = 0.30
    min_offset_ratio: float = 0.01
    # 175# boost TTL (seconds): time_filter中の古いboostが残る問題を防止
    boost_ttl_sec: float = 600.0
    # 230# H-1: Layer 2 deadzone — normal spread cost で誤発火しない閾値
    # AS理論: maker のスプレッドコスト (~2-3 bps) は正常な損失。
    # pnl < -deadzone のときのみ adverseSelection と判定する。
    l2_deadzone_bps: float = 3.0
    # 230# H-2: boost 解除に必要な連続正常 fill 数 (Kyle 1985)
    # 情報トレーダーは複数 fill にわたり取引するため、1回の正常 fill で
    # 安全宣言するのは尚早。N 回連続で正常なら threat 終了と判断。
    boost_release_streak: int = 3


@dataclass
class _SideState:
    """side 別の boost 状態."""

    boost_active: bool = False
    boost_multiplier: float = 1.0
    boost_activated_at: float = 0.0  # 175# TTL decay 用 timestamp
    normal_fill_streak: int = 0  # 230# H-2: 連続正常 fill カウンタ


class FastFillDefense:
    """Side-aware 即約定防御.

    098# P0-3: has_negative_edge が fill_price vs mid_at_fill のみでは
    sell fast-fill AS の 50% を見逃す問題を two-layer で解消。

    Layer 1 (即時): fill_price vs mid_at_fill (従来の proxy)
    Layer 2 (post-fill): post_fill_30s_pnl < 0 なら次サイクルを遅延防御

    099# 追加: sell skip は保持すべき (逆選択)、buy-first 戦略。
    """

    def __init__(
        self,
        config: FastFillDefenseConfig,
        base_offset_ratio: float,
        base_offset_ratio_buy: Optional[float] = None,
        base_offset_ratio_sell: Optional[float] = None,
    ) -> None:
        self._config = config
        self._base_offset_ratio = base_offset_ratio
        self._base_offset_ratio_buy = base_offset_ratio_buy
        self._base_offset_ratio_sell = base_offset_ratio_sell
        # Per-side state (P0-5: sell boost が buy に伝播しないように分離)
        self._state_buy = _SideState()
        self._state_sell = _SideState()

    def _get_state(self, side: str) -> _SideState:
        return self._state_buy if side == "buy" else self._state_sell

    def _resolve_side_value(
        self,
        side: str,
        common_value: float,
        *,
        buy_value: Optional[float] = None,
        sell_value: Optional[float] = None,
    ) -> float:
        """side 別値を解決し、未設定時は共通値へフォールバックする."""
        if side == "buy":
            return common_value if buy_value is None else buy_value
        if side == "sell":
            return common_value if sell_value is None else sell_value
        return common_value

    def get_boost_multiplier(self, side: str) -> float:
        """現サイクルの offset に適用する boost 乗数を返す."""
        state = self._get_state(side)
        # 175# TTL decay: boost_ttl_sec を超過したら自動解除
        if state.boost_active and self._config.boost_ttl_sec > 0:
            if (time.time() - state.boost_activated_at) > self._config.boost_ttl_sec:
                old_mult = state.boost_multiplier
                state.boost_active = False
                state.boost_multiplier = 1.0
                state.normal_fill_streak = 0  # 231# R1: TTL期限切れで streak もクリア
                logger.info(
                    f"[fast_fill_defense] TTL expired ({side}): "
                    f"multiplier {old_mult:.2f}→1.00 "
                    f"(ttl={self._config.boost_ttl_sec:.0f}s)"
                )
        return state.boost_multiplier

    def is_boost_active(self, side: str) -> bool:
        """指定 side で boost が有効かどうか."""
        return self._get_state(side).boost_active

    def _resolve_threshold_sec(self, side: str) -> float:
        """side 別の fast_fill 判定閾値秒数を解決."""
        return self._resolve_side_value(
            side,
            self._config.threshold_sec,
            buy_value=self._config.threshold_sec_buy,
            sell_value=self._config.threshold_sec_sell,
        )

    def _resolve_boost(self, side: str) -> float:
        """side 別の offset boost 倍率を解決."""
        return self._resolve_side_value(
            side,
            self._config.offset_boost,
            buy_value=self._config.offset_boost_buy,
            sell_value=self._config.offset_boost_sell,
        )

    def _resolve_base_offset(self, side: str) -> float:
        """side 別の base offset ratio を解決.

        P0-5 fix: boost cap 計算に common 0.05 ではなく sell=0.12 等の
        side-specific 値を使用する。
        """
        return self._resolve_side_value(
            side,
            self._base_offset_ratio,
            buy_value=self._base_offset_ratio_buy,
            sell_value=self._base_offset_ratio_sell,
        )

    def _compute_capped_multiplier(self, side: str, raw_boost: float) -> float:
        """boost 乗数を side 別 base_offset_ratio で cap する.

        098# P1-2: cap = max_offset_ratio / base_offset を side 別に計算。
        sell (base=0.12) → cap=2.5, buy (base=0.05) → cap=6.0
        """
        if raw_boost <= 1.0:
            return 1.0

        safe_floor = max(self._config.min_offset_ratio, 1e-12)
        base = max(self._resolve_base_offset(side), safe_floor)
        cap = max(1.0, self._config.max_offset_ratio / base)
        return min(raw_boost, cap)

    def evaluate_fill(
        self,
        side: str,
        queue_wait_sec: float,
        fill_price: Optional[float],
        mid_at_fill: Optional[float],
        *,
        post_fill_pnl_bps: Optional[float] = None,
    ) -> None:
        """約定結果を評価し、boost 状態を更新する.

        Two-layer detection:
        - Layer 1: fill_price vs mid_at_fill (即時判定)
        - Layer 2: post_fill_pnl_bps < 0 (30s 後の確定 PnL)

        Args:
            side: "buy" or "sell"
            queue_wait_sec: 約定待ち時間 (秒)
            fill_price: 約定価格
            mid_at_fill: 約定時の mid price
            post_fill_pnl_bps: 30s 後の PnL (bps), 利用可能であれば
        """
        if not self._config.enabled:
            return

        state = self._get_state(side)
        ff_threshold = self._resolve_threshold_sec(side)
        is_fast = queue_wait_sec <= ff_threshold

        # Layer 1: 即時 proxy (fill_price vs mid)
        has_negative_edge_l1 = (
            fill_price is not None
            and mid_at_fill is not None
            and (
                (side == "buy" and fill_price > mid_at_fill)
                or (side == "sell" and fill_price < mid_at_fill)
            )
        )

        # Layer 2: post-fill PnL check (098# §3.1: 50%見逃し問題の対策)
        # 230# H-1: deadzone — 通常スプレッドコスト範囲の軽微な負PnLは無視
        _dz = self._config.l2_deadzone_bps
        has_negative_edge_l2 = (
            post_fill_pnl_bps is not None and post_fill_pnl_bps < -_dz
        )

        has_negative_edge = has_negative_edge_l1 or has_negative_edge_l2

        if is_fast and has_negative_edge:
            if not state.boost_active:
                state.boost_active = True
                raw_boost = self._resolve_boost(side)
                state.boost_multiplier = self._compute_capped_multiplier(side, raw_boost)
                _layer = (
                    "L1+L2" if (has_negative_edge_l1 and has_negative_edge_l2)
                    else ("L1" if has_negative_edge_l1 else "L2(pnl)")
                )
                logger.info(
                    f"[fast_fill_defense] Activated ({side}): "
                    f"wait={queue_wait_sec:.1f}s (< {ff_threshold}s), "
                    f"negative edge detected ({_layer}). "
                    f"multiplier→{state.boost_multiplier:.2f}"
                )
            # 231# R3: 新規/継続に関わらず TTL リフレッシュ + streak リセット
            state.boost_activated_at = time.time()
            state.normal_fill_streak = 0
        elif state.boost_active:
            # 231# R2: slow fill でも negative edge があれば streak リセット
            if has_negative_edge:
                state.normal_fill_streak = 0
            else:
                state.normal_fill_streak += 1
            _required = max(1, self._config.boost_release_streak)
            if state.normal_fill_streak >= _required:
                old_mult = state.boost_multiplier
                state.boost_multiplier = 1.0
                state.boost_active = False
                state.normal_fill_streak = 0
                logger.info(
                    f"[fast_fill_defense] Deactivated ({side}): "
                    f"{_required} consecutive normal fills, "
                    f"multiplier {old_mult:.2f}→1.00"
                )
            else:
                logger.debug(
                    f"[fast_fill_defense] Normal fill streak ({side}): "
                    f"{state.normal_fill_streak}/{_required}"
                )

    def reset_on_unfilled(self, side: str) -> None:
        """未約定時のブースト永続化防止."""
        if not self._config.enabled:
            return
        state = self._get_state(side)
        if state.boost_active:
            old_mult = state.boost_multiplier
            state.boost_multiplier = 1.0
            state.boost_active = False
            state.normal_fill_streak = 0  # 230# H-2
            logger.info(
                f"[fast_fill_defense] Reset on unfilled ({side}): "
                f"multiplier {old_mult:.2f}→1.00"
            )

    def update_base_offsets(
        self,
        base: float,
        buy: Optional[float] = None,
        sell: Optional[float] = None,
    ) -> None:
        """param_adapter 等による base offset 更新を反映."""
        self._base_offset_ratio = base
        self._base_offset_ratio_buy = buy
        self._base_offset_ratio_sell = sell

    def export_state(self) -> dict[str, object]:
        """226# hot-reload 時の boost 状態保存."""
        return {
            "buy_boost_active": self._state_buy.boost_active,
            "buy_boost_multiplier": self._state_buy.boost_multiplier,
            "buy_boost_activated_at": self._state_buy.boost_activated_at,
            "buy_normal_fill_streak": self._state_buy.normal_fill_streak,
            "sell_boost_active": self._state_sell.boost_active,
            "sell_boost_multiplier": self._state_sell.boost_multiplier,
            "sell_boost_activated_at": self._state_sell.boost_activated_at,
            "sell_normal_fill_streak": self._state_sell.normal_fill_streak,
        }

    def import_state(self, state: dict[str, object]) -> None:
        """226# hot-reload 後の boost 状態復元."""
        self._state_buy.boost_active = bool(state.get("buy_boost_active") or False)
        self._state_buy.boost_multiplier = float(state.get("buy_boost_multiplier") or 1.0)
        self._state_buy.boost_activated_at = float(state.get("buy_boost_activated_at") or 0.0)
        self._state_buy.normal_fill_streak = int(state.get("buy_normal_fill_streak") or 0)
        self._state_sell.boost_active = bool(state.get("sell_boost_active") or False)
        self._state_sell.boost_multiplier = float(state.get("sell_boost_multiplier") or 1.0)
        self._state_sell.boost_activated_at = float(state.get("sell_boost_activated_at") or 0.0)
        self._state_sell.normal_fill_streak = int(state.get("sell_normal_fill_streak") or 0)
