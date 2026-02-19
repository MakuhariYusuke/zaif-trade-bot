"""121# 残高チェックモジュール.

FillTestRunner から残高 pre-flight チェック + ロット自動縮小を分離。
041# 残高チェック / 052# ロット縮小 / 101# 残高回復ロジックを統合。

責務:
  - buy/sell 残高の事前検証
  - 残高不足時のロット自動縮小 (0.001 BTC 単位)
  - 残高回復時のロット復元
"""

from __future__ import annotations

import logging
from typing import Optional

from scripts.v460.lib.fill_config import FillTestConfig

logger = logging.getLogger(__name__)

# Coincheck 板取引 BTC 最小注文数量
MIN_ORDER_BTC: float = 0.001


class BalanceChecker:
    """041# 残高 pre-flight check + 052# ロット自動縮小."""

    def __init__(self, config: FillTestConfig) -> None:
        self._config = config
        self._min_order_btc = config.min_order_btc
        self._current_lot: float = config.order_quantity
        self._pre_shrink_lot: float = config.order_quantity
        self._balance_shrink_active: bool = False

    @property
    def current_lot(self) -> float:
        return self._current_lot

    @current_lot.setter
    def current_lot(self, value: float) -> None:
        self._current_lot = value

    @property
    def balance_shrink_active(self) -> bool:
        return self._balance_shrink_active

    @balance_shrink_active.setter
    def balance_shrink_active(self, value: bool) -> None:
        self._balance_shrink_active = value

    @property
    def pre_shrink_lot(self) -> float:
        return self._pre_shrink_lot

    @pre_shrink_lot.setter
    def pre_shrink_lot(self, value: float) -> None:
        self._pre_shrink_lot = value

    async def check(self, side: str, adapter: object, symbol: str) -> bool:
        """残高 pre-flight check.

        不足時は True を返す (スキップすべき)。
        052#: 残高に基づくロット自動縮小。
        """
        try:
            if side == "sell":
                return await self._check_sell(adapter, symbol)
            else:
                return await self._check_buy(adapter, symbol)
        except Exception as e:
            logger.debug(f"[balance] Pre-flight check failed (non-fatal): {e}")
        return False

    async def _check_sell(self, adapter: object, symbol: str) -> bool:
        """sell 残高チェック (BTC)."""
        btc_balances = await adapter.get_balance("BTC")  # type: ignore[union-attr]
        btc_free = sum(b.free for b in btc_balances) if btc_balances else 0.0

        if btc_free < self._current_lot:
            # 052#: 最小ロット以上の残高があれば縮小して継続
            if btc_free >= self._min_order_btc:
                new_lot = int(btc_free / self._min_order_btc) * self._min_order_btc
                if new_lot >= self._min_order_btc:
                    old_lot = self._current_lot
                    self._current_lot = new_lot
                    if not self._balance_shrink_active:
                        self._pre_shrink_lot = old_lot
                    logger.info(
                        f"[balance] BTC {btc_free:.6f} < {old_lot:.4f}. "
                        f"ロット自動縮小: {old_lot:.4f} → {new_lot:.4f} BTC"
                    )
                    return False
            logger.warning(
                f"[balance] Insufficient BTC for sell: "
                f"{btc_free:.6f} < {self._min_order_btc:.4f}. "
                f"Skipping sell → will retry buy next."
            )
            return True

        # 101# §6: 残高が十分な場合、以前の縮小から復元
        if (
            not self._balance_shrink_active
            and self._current_lot < self._pre_shrink_lot
            and btc_free >= self._pre_shrink_lot
        ):
            old_lot = self._current_lot
            self._current_lot = self._pre_shrink_lot
            logger.info(
                f"[balance] BTC 残高回復: ロット復元 "
                f"{old_lot:.4f} → {self._current_lot:.4f} BTC"
            )
        return False

    async def _check_buy(self, adapter: object, symbol: str) -> bool:
        """buy 残高チェック (JPY)."""
        price = await adapter.get_current_price(symbol)  # type: ignore[union-attr]
        if not price:
            return False

        jpy_needed = self._current_lot * price * self._config.balance_margin_ratio
        jpy_balances = await adapter.get_balance("JPY")  # type: ignore[union-attr]
        jpy_free = sum(b.free for b in jpy_balances) if jpy_balances else 0.0

        if jpy_free < jpy_needed:
            # 052#: JPY 残高から発注可能なロットを逆算
            affordable_lot = jpy_free / (price * self._config.balance_margin_ratio)
            affordable_lot = int(affordable_lot / self._min_order_btc) * self._min_order_btc
            if affordable_lot >= self._min_order_btc:
                old_lot = self._current_lot
                self._current_lot = affordable_lot
                if not self._balance_shrink_active:
                    self._pre_shrink_lot = old_lot
                logger.info(
                    f"[balance] JPY {jpy_free:.0f} < {jpy_needed:.0f}. "
                    f"ロット自動縮小: {old_lot:.4f} → {affordable_lot:.4f} BTC"
                )
                return False
            logger.warning(
                f"[balance] Insufficient JPY for buy: "
                f"{jpy_free:.0f} < min {self._min_order_btc * price * self._config.balance_margin_ratio:.0f}. "
                f"Skipping buy → will retry sell next."
            )
            return True

        # 101# §6: 残高が十分な場合、以前の縮小から復元 (buy 側)
        if (
            not self._balance_shrink_active
            and self._current_lot < self._pre_shrink_lot
        ):
            pre_lot_needed = self._pre_shrink_lot * price * self._config.balance_margin_ratio
            if jpy_free >= pre_lot_needed:
                old_lot = self._current_lot
                self._current_lot = self._pre_shrink_lot
                logger.info(
                    f"[balance] JPY 残高回復: ロット復元 "
                    f"{old_lot:.4f} → {self._current_lot:.4f} BTC"
                )
        return False

    def apply_lot_floor(self) -> None:
        """105#: lot floor guard — 浮動小数点丸め誤差による API 400 防止."""
        self._current_lot = max(
            self._min_order_btc,
            int(self._current_lot / self._min_order_btc) * self._min_order_btc,
        )

    def restore_lot_on_success(self) -> None:
        """051# P2-3: 成功時に balance_shrink を解除し、ロットを原値に復元."""
        if self._balance_shrink_active:
            old_lot = self._current_lot
            self._current_lot = self._pre_shrink_lot
            self._balance_shrink_active = False
            logger.info(
                f"[balance_shrink] 解除: ロット復元 {old_lot:.4f} → {self._current_lot:.4f} BTC"
            )
