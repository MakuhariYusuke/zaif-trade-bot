"""121# Side 選択モジュール.

FillTestRunner から buy/sell 交互ロジック + Smart Side を分離。
009# §4.2 / 054# S2 / 055# Fix を統合。

責務:
  - buy/sell 交互切替 (片側蓄積防止)
  - Smart Side: imbalance ベースの side 抑制/追従 (054# S2)
  - Rapid exit side 強制 (055# Fix)
"""

from __future__ import annotations

import logging

from scripts.v460.lib.fill_config import FillTestConfig

logger = logging.getLogger(__name__)


class SideSelector:
    """009# §4.2: buy/sell 交互 + 054# S2 Smart Side.

    120# A5: inventory-aware side hint — 残高不足 side を一時的に回避。
    """

    def __init__(self, config: FillTestConfig) -> None:
        self._config = config
        # start_side に応じて _last_side を設定 (交互ロジック用)
        if config.start_side == "sell":
            self._last_side: str | None = "buy"  # → next() が "sell" を返す
        else:
            self._last_side = None  # → next() が "buy" を返す
        # 054# S2: 同一 side 連続カウンタ
        self._consecutive_same_side: int = 0
        # 054# S3: Early Exit — rapid exit
        self._rapid_exit_side: str | None = None
        # 120# A5: 残高不足 side の一時凍結 (inventory-aware)
        self._frozen_side: str | None = None
        self._frozen_remaining: int = 0

    @property
    def last_side(self) -> str | None:
        return self._last_side

    @last_side.setter
    def last_side(self, value: str | None) -> None:
        self._last_side = value

    @property
    def consecutive_same_side(self) -> int:
        return self._consecutive_same_side

    @property
    def rapid_exit_side(self) -> str | None:
        return self._rapid_exit_side

    @rapid_exit_side.setter
    def rapid_exit_side(self, value: str | None) -> None:
        self._rapid_exit_side = value

    def next(self, imbalance: float = 0.0) -> str:
        """次の side を決定: 交互 or Smart Side.

        120# A5: frozen_side に該当する side は自動的に反対に迂回。
        Args:
            imbalance: 現在の板不均衡 (Smart Side 用)
        """
        # 055# Fix #1: S3 rapid exit で決定された side を最優先で返却
        if self._rapid_exit_side is not None:
            forced_side = self._rapid_exit_side
            self._rapid_exit_side = None
            logger.info(f"[early_exit] Rapid exit forcing side={forced_side}")
            return forced_side

        base_side = "buy" if (self._last_side is None or self._last_side == "sell") else "sell"

        # 120# A5: inventory-aware — frozen side は反対に迂回
        if self._frozen_side is not None and self._frozen_remaining > 0:
            if base_side == self._frozen_side:
                alt = "sell" if base_side == "buy" else "buy"
                self._frozen_remaining -= 1
                logger.debug(
                    f"[side_selector] Frozen {self._frozen_side} → "
                    f"diverting to {alt} (remaining={self._frozen_remaining})"
                )
                if self._frozen_remaining <= 0:
                    self._frozen_side = None
                return alt

        if not self._config.smart_side_enabled:
            return base_side

        if self._config.smart_side_mode == "suppress":
            should_suppress = False
            if base_side == "buy" and imbalance < -self._config.imbalance_threshold:
                should_suppress = True
            elif base_side == "sell" and imbalance > self._config.imbalance_threshold:
                should_suppress = True

            if should_suppress:
                if self._consecutive_same_side >= self._config.smart_side_max_consecutive:
                    logger.debug(
                        f"[smart_side] Max consecutive ({self._consecutive_same_side}) reached, "
                        f"forcing {base_side}"
                    )
                    return base_side
                alt_side = self._last_side or ("sell" if base_side == "buy" else "buy")
                logger.info(
                    f"[smart_side] Suppressing {base_side} (imb={imbalance:+.3f}), "
                    f"continuing {alt_side}"
                )
                return alt_side

        elif self._config.smart_side_mode == "follow":
            if abs(imbalance) > self._config.imbalance_threshold:
                follow_side = "buy" if imbalance > 0 else "sell"
                if (
                    follow_side == self._last_side
                    and self._consecutive_same_side >= self._config.smart_side_max_consecutive
                ):
                    return base_side
                return follow_side

        return base_side

    def update_after_decision(self, side: str) -> None:
        """side 決定後のカウンタ更新."""
        if side == self._last_side:
            self._consecutive_same_side += 1
        else:
            self._consecutive_same_side = 0
        self._last_side = side

    def set_rapid_exit(self, current_side: str) -> None:
        """054# S3: early exit → 反対 side を設定."""
        self._rapid_exit_side = "sell" if current_side == "buy" else "buy"

    def freeze_side(self, side: str, cycles: int = 3) -> None:
        """120# A5: 残高不足 side を一時凍結 (inventory-aware).

        Args:
            side: 凍結する side ("buy" or "sell")
            cycles: 凍結サイクル数 (デフォルト 3)
        """
        self._frozen_side = side
        self._frozen_remaining = cycles
        logger.info(
            f"[side_selector] Freezing {side} for {cycles} cycles "
            f"(120# inventory-aware)"
        )

    def unfreeze_side(self) -> None:
        """120# A5: side 凍結を解除."""
        if self._frozen_side is not None:
            logger.debug(f"[side_selector] Unfreezing {self._frozen_side}")
            self._frozen_side = None
            self._frozen_remaining = 0
