"""121# 残高チェックモジュール.

FillTestRunner から残高 pre-flight チェック + ロット自動縮小を分離。
041# 残高チェック / 052# ロット縮小 / 101# 残高回復ロジックを統合。
128# dust_sweep: 端数 BTC 一掃売却 (sell 時に全BTC売却で最小取引金額未満の残留を解消)。

責務:
  - buy/sell 残高の事前検証
  - 残高不足時のロット自動縮小 (0.001 BTC 単位)
  - 残高回復時のロット復元
  - 128# sell 時 dust 込み全額売却
"""

from __future__ import annotations

import logging
import time as _time
from collections.abc import Sequence
from typing import Protocol

from scripts.v460.lib.fill_config import FillTestConfig

logger = logging.getLogger(__name__)


# 261# P2-7: BalanceAdapterProtocol — adapter: object の型安全化
class _BalanceLike(Protocol):
    """残高エントリ — .free で利用可能残高を取得."""

    @property
    def free(self) -> float: ...


class BalanceAdapterProtocol(Protocol):
    """BalanceChecker が adapter に要求する最小インタフェース."""

    async def get_balance(self, currency: str) -> Sequence[_BalanceLike] | None: ...
    async def get_current_price(self, symbol: str) -> float | None: ...

# Coincheck 板取引 BTC 最小注文数量 (348# satoshi化: config.min_order_btc 優先)
MIN_ORDER_BTC: float = 0.001  # フォールバック定数 (config 優先)


class BalanceChecker:
    """041# 残高 pre-flight check + 052# ロット自動縮小."""

    def __init__(self, config: FillTestConfig) -> None:
        self._config = config
        self._min_order_btc = config.min_order_btc
        self._current_lot: float = config.order_quantity
        self._pre_shrink_lot: float = config.order_quantity
        self._balance_shrink_active: bool = False
        # 128# dust sweep 状態
        self._dust_sweep_active: bool = False
        self._pre_dust_lot: float = config.order_quantity
        # 166# HF3: Insufficient 警告クールダウン (side別)
        self._insufficient_cooldown_sec: float = 120.0  # 同一 side 2分間抑制
        self._last_insufficient_log: dict[str, float] = {}  # side -> timestamp
        # 238# C-2: 直前クエリの BTC/JPY 残高キャッシュ (phantom guard snapshot 用)
        self._last_btc_free: float | None = None
        self._last_jpy_free: float | None = None

    @property
    def last_btc_free(self) -> float | None:
        """238# 直前の BTC 残高クエリ結果 (phantom guard snapshot 用)."""
        return self._last_btc_free

    @property
    def last_jpy_free(self) -> float | None:
        """251# 直前の JPY 残高クエリ結果 (phantom guard buy 側照合用)."""
        return self._last_jpy_free

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
    def dust_sweep_active(self) -> bool:
        """128# dust sweep がアクティブか."""
        return self._dust_sweep_active

    @property
    def pre_shrink_lot(self) -> float:
        return self._pre_shrink_lot

    @pre_shrink_lot.setter
    def pre_shrink_lot(self, value: float) -> None:
        self._pre_shrink_lot = value

    async def check(
        self,
        side: str,
        adapter: BalanceAdapterProtocol,
        symbol: str,
        *,
        regime_mult: float = 1.0,
    ) -> bool:
        """残高 pre-flight check.

        不足時は True を返す (スキップすべき)。
        052#: 残高に基づくロット自動縮小。
        145# §8-#1: regime_mult を加味して実際の注文ロットで判定。
        """
        try:
            if side == "sell":
                return await self._check_sell(adapter, symbol, regime_mult=regime_mult)
            else:
                return await self._check_buy(adapter, symbol, regime_mult=regime_mult)
        except Exception as e:
            logger.warning(f"[balance] Pre-flight check failed — proceeding: {e}")
        return False

    async def _check_sell(
        self, adapter: BalanceAdapterProtocol, symbol: str, *, regime_mult: float = 1.0,
    ) -> bool:
        """sell 残高チェック (BTC).

        145# §8-#1: regime_mult 対応 — 実効ロット (base × mult) で比較し、
        自動縮小時もレジーム倍率を考慮して base ロットを調整する。
        """
        btc_balances = await adapter.get_balance("BTC")
        btc_free = sum(b.free for b in btc_balances) if btc_balances else 0.0
        self._last_btc_free = btc_free  # 238# C-2: phantom guard snapshot 用キャッシュ

        effective_lot = self._current_lot * regime_mult
        if btc_free < effective_lot:
            # 052#: 最小ロット以上の残高があれば縮小して継続
            # 145# §8-#1: base ロット = btc_free / regime_mult として算出
            max_base = btc_free / regime_mult if regime_mult > 0 else btc_free
            new_lot = int(max_base / self._min_order_btc) * self._min_order_btc
            if new_lot >= self._min_order_btc:
                old_lot = self._current_lot
                self._current_lot = new_lot
                if not self._balance_shrink_active:
                    self._pre_shrink_lot = old_lot
                logger.info(
                    f"[balance] BTC {btc_free:.6f} < {effective_lot:.4f} "
                    f"(base {self._current_lot:.4f}×{regime_mult:.2f}). "
                    f"ロット自動縮小: {old_lot:.4f} → {new_lot:.4f} BTC"
                )
                # 128# 縮小後も dust sweep 判定を通過させる
                return self._maybe_dust_sweep(btc_free)
            self._log_insufficient(
                "sell",
                f"[balance] Insufficient BTC for sell: "
                f"{btc_free:.6f} < {self._min_order_btc:.4f} "
                f"(regime_mult={regime_mult:.2f}). "
                f"Skipping sell → will retry buy next.",
            )
            return True

        # 101# §6: 残高が十分な場合、以前の縮小から復元
        # 145# §8-#1: 復元時もレジーム倍率を考慮
        if (
            not self._balance_shrink_active
            and self._current_lot < self._pre_shrink_lot
            and btc_free >= self._pre_shrink_lot * regime_mult
        ):
            old_lot = self._current_lot
            self._current_lot = self._pre_shrink_lot
            logger.info(
                f"[balance] BTC 残高回復: ロット復元 "
                f"{old_lot:.4f} → {self._current_lot:.4f} BTC"
            )

        # 128# dust sweep: 残高十分でも dust があれば全額売却
        return self._maybe_dust_sweep(btc_free)

    async def _check_buy(
        self, adapter: BalanceAdapterProtocol, symbol: str, *, regime_mult: float = 1.0,
    ) -> bool:
        """buy 残高チェック (JPY).

        145# §8-#1: regime_mult 対応 — 実効ロット (base × mult) で判定。
        """
        price = await adapter.get_current_price(symbol)
        if not price:
            return False

        effective_lot = self._current_lot * regime_mult
        jpy_needed = effective_lot * price * self._config.balance_margin_ratio
        jpy_balances = await adapter.get_balance("JPY")
        jpy_free = sum(b.free for b in jpy_balances) if jpy_balances else 0.0
        self._last_jpy_free = jpy_free  # 238# C-2: phantom guard snapshot 用キャッシュ

        if jpy_free < jpy_needed:
            # 052#: JPY 残高から発注可能なロットを逆算
            # 145# §8-#1: レジーム倍率込みで逆算: base = affordable / regime_mult
            affordable_effective = jpy_free / (price * self._config.balance_margin_ratio)
            affordable_base = affordable_effective / regime_mult if regime_mult > 0 else affordable_effective
            affordable_lot = int(affordable_base / self._min_order_btc) * self._min_order_btc
            if affordable_lot >= self._min_order_btc:
                old_lot = self._current_lot
                self._current_lot = affordable_lot
                if not self._balance_shrink_active:
                    self._pre_shrink_lot = old_lot
                logger.info(
                    f"[balance] JPY {jpy_free:.0f} < {jpy_needed:.0f} "
                    f"(base {old_lot:.4f}×{regime_mult:.2f}). "
                    f"ロット自動縮小: {old_lot:.4f} → {affordable_lot:.4f} BTC"
                )
                return False
            self._log_insufficient(
                "buy",
                f"[balance] Insufficient JPY for buy: "
                f"{jpy_free:.0f} < min {self._min_order_btc * regime_mult * price * self._config.balance_margin_ratio:.0f} "
                f"(regime_mult={regime_mult:.2f}). "
                f"Skipping buy → will retry sell next.",
            )
            return True

        # 101# §6: 残高が十分な場合、以前の縮小から復元 (buy 側)
        # 145# §8-#1: 復元時もレジーム倍率を考慮
        if (
            not self._balance_shrink_active
            and self._current_lot < self._pre_shrink_lot
        ):
            pre_lot_needed = self._pre_shrink_lot * regime_mult * price * self._config.balance_margin_ratio
            if jpy_free >= pre_lot_needed:
                old_lot = self._current_lot
                self._current_lot = self._pre_shrink_lot
                logger.info(
                    f"[balance] JPY 残高回復: ロット復元 "
                    f"{old_lot:.4f} → {self._current_lot:.4f} BTC"
                )
        return False

    def _maybe_dust_sweep(self, btc_free: float) -> bool:
        """128# sell 時に dust があれば全BTC残高を売却して一掃.

        dust = btc_free のうち min_order_btc 単位に切り捨てた端数部分。
        dust > 0 の場合、_current_lot を btc_free 全額に拡張して
        sell 後に残留ゼロにする。

        Returns:
            False (= sell 続行) を返す。
        """
        if not self._config.dust_sweep_enabled:
            return False
        dust = btc_free - int(btc_free / self._min_order_btc) * self._min_order_btc
        if dust > 1e-9:
            self._pre_dust_lot = self._current_lot
            self._current_lot = round(btc_free, 8)
            self._dust_sweep_active = True
            logger.info(
                f"[dust_sweep] BTC {btc_free:.8f} has dust {dust:.8f}. "
                f"Selling full balance: {self._current_lot:.8f} BTC"
            )
        return False

    def _log_insufficient(self, side: str, message: str) -> None:
        """166# HF3: side別クールダウン付き Insufficient 警告.

        同一 side の Insufficient 警告を _insufficient_cooldown_sec 間隔で抑制し、
        ログノイズを削減する。
        """
        import time
        now = time.time()
        last = self._last_insufficient_log.get(side, 0.0)
        if now - last >= self._insufficient_cooldown_sec:
            logger.warning(message)
            self._last_insufficient_log[side] = now
        else:
            logger.debug(
                f"[balance] {side} insufficient (suppressed, "
                f"cooldown {self._insufficient_cooldown_sec:.0f}s)"
            )

    def apply_lot_floor(self) -> None:
        """105#: lot floor guard — 浮動小数点丸め誤差による API 400 防止.

        128#: dust sweep アクティブ時はフロア処理をスキップ
        (端数込みの正確な数量を保持する必要があるため)。
        """
        if self._dust_sweep_active:
            return
        self._current_lot = max(
            self._min_order_btc,
            int(self._current_lot / self._min_order_btc) * self._min_order_btc,
        )

    def restore_lot_after_dust_sweep(self) -> None:
        """128# dust sweep 後のロット復元.

        sell サイクル完了後に呼び出し、通常ロットに戻す。
        dust sweep が非アクティブなら no-op。
        """
        if self._dust_sweep_active:
            old_lot = self._current_lot
            self._current_lot = self._pre_dust_lot
            self._dust_sweep_active = False
            logger.info(
                f"[dust_sweep] Lot restored: {old_lot:.8f} → {self._current_lot:.4f} BTC"
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
