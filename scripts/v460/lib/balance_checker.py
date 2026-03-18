"""121# 残高チェックモジュール.

FillTestRunner から残高 pre-flight チェック + ロット自動縮小を分離。
041# 残高チェック / 052# ロット縮小 / 101# 残高回復ロジックを統合。
128# dust_sweep: 端数 BTC 一掃売却 (sell 時に全 BTC 売却で残留を解消)。
372# dust buy-to-clear: btc_free < min_order_btc の micro-dust を
     buy min_order → sell 全額の自動2サイクルで解消。
476#: Coincheck は satoshi 精度 (1e-8) を許容 — 0.001 単位切り捨て廃止。
     sell/buy ともに残高に応じた動的ロットサイジング。

責務:
  - buy/sell 残高の事前検証
  - 残高不足時のロット自動縮小 (satoshi 精度)
  - 残高回復時のロット復元
  - 476# 残高連動ロット拡大 (buy: JPY→max_lot, sell: dust_sweep 全額売却)
  - 128# sell 時 dust 込み全額売却
  - 372# micro-dust buy-to-clear (sell 不能端数の buy 経由解消)
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
        # 372# dust buy-to-clear 状態
        self._dust_buy_pending: bool = False
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
    def dust_buy_pending(self) -> bool:
        """372# dust buy-to-clear が保留中か."""
        return self._dust_buy_pending

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
            # 373# CRITICAL: 例外時は注文スキップ (True) に変更。
            # 旧実装は return False (=注文続行) だったが、残高未検証で
            # 取引を続行すると insufficient funds 連打やロット計算不整合が発生する。
            logger.error(
                f"[balance] Pre-flight check FAILED — skipping order: {e}",
                exc_info=True,
            )
        return True

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
            # 052# + 476#: satoshi 精度で base ロットを算出
            max_base = btc_free / regime_mult if regime_mult > 0 else btc_free
            new_lot = round(max_base, 8)
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
                # 372# sell 可能になったら dust_buy_pending 解除
                self._dust_buy_pending = False
                return self._maybe_dust_sweep(btc_free, regime_mult)
            self._log_insufficient(
                "sell",
                f"[balance] Insufficient BTC for sell: "
                f"{btc_free:.6f} < {self._min_order_btc:.4f} "
                f"(regime_mult={regime_mult:.2f}). "
                f"Skipping sell → will retry buy next.",
            )
            # 372# micro-dust buy-to-clear: min_order 未満だが残高 > 0 → buy 経由で解消
            if btc_free > 1e-9 and self._config.dust_sweep_enabled:
                if not self._dust_buy_pending:
                    logger.info(
                        f"[dust_sweep] Micro-dust {btc_free:.8f} BTC detected "
                        f"(< min_order {self._min_order_btc}). "
                        f"Scheduling buy-to-clear."
                    )
                self._dust_buy_pending = True
            return True

        # 372# sell 可能 → dust_buy_pending 解除
        self._dust_buy_pending = False

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

        # 128# dust sweep: 残高 > current_lot なら全額売却
        return self._maybe_dust_sweep(btc_free, regime_mult)

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
            # 052# + 476#: satoshi 精度で JPY から base ロットを逆算
            affordable_effective = jpy_free / (price * self._config.balance_margin_ratio)
            affordable_base = affordable_effective / regime_mult if regime_mult > 0 else affordable_effective
            affordable_lot = round(affordable_base, 8)
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

        # 476#: 残高連動ロット拡大 — JPY 残高で買える最大ロットに拡大
        _max_lot = self._config.max_lot
        if _max_lot > 0 and self._current_lot < _max_lot and price > 0:
            max_affordable = jpy_free / (price * self._config.balance_margin_ratio)
            max_base = max_affordable / regime_mult if regime_mult > 0 else max_affordable
            target_lot = round(min(max_base, _max_lot), 8)
            if target_lot > self._current_lot and target_lot >= self._min_order_btc:
                old_lot = self._current_lot
                self._current_lot = target_lot
                logger.info(
                    f"[476# balance_lot] buy: JPY {jpy_free:.0f} → "
                    f"lot {old_lot:.6f} → {target_lot:.6f} BTC "
                    f"(max_lot={_max_lot})"
                )

        return False

    def _maybe_dust_sweep(self, btc_free: float, regime_mult: float = 1.0) -> bool:
        """128# sell 時に dust があれば全BTC残高を売却して一掃.

        476#: Coincheck は satoshi 精度許容 — 実効ロット (base × regime_mult) と
        btc_free を比較し、余剰があれば全額売却。regime_mult を考慮して
        base lot を算出する。

        Returns:
            False (= sell 続行) を返す。
        """
        if not self._config.dust_sweep_enabled:
            return False
        effective_lot = self._current_lot * regime_mult
        if btc_free > effective_lot + 1e-9:
            # 実効ロットを超える BTC → 全額売却に拡張
            self._pre_dust_lot = self._current_lot
            sell_base = btc_free / regime_mult if regime_mult > 0 else btc_free
            self._current_lot = round(sell_base, 8)
            self._dust_sweep_active = True
            logger.info(
                f"[dust_sweep] BTC {btc_free:.8f} > effective "
                f"{effective_lot:.8f}. Selling full balance: "
                f"base {self._current_lot:.8f} (×{regime_mult:.2f}) BTC"
            )
        elif (
            abs(effective_lot - btc_free) < 1e-9
            and abs(self._current_lot - self._config.order_quantity) > 1e-9
        ):
            # 476#: shrink 後に lot ≈ btc_free だが order_quantity と乖離
            # → FCE の lot_scale チェーンで縮小されないよう保護フラグを立てる
            self._pre_dust_lot = self._config.order_quantity
            self._dust_sweep_active = True
            logger.info(
                f"[dust_sweep] lot {self._current_lot:.8f} ≈ BTC {btc_free:.8f} "
                f"(≠ order_qty {self._config.order_quantity:.4f}). "
                f"Activating lot_scale protection."
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
        476#: Coincheck は satoshi 精度を許容 — 0.001 単位切り捨て廃止。
        """
        if self._dust_sweep_active:
            return
        self._current_lot = max(
            self._min_order_btc,
            round(self._current_lot, 8),
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

    def prepare_dust_buy(self) -> None:
        """372# dust buy-to-clear: buy ロットを min_order_btc に設定.

        micro-dust 解消のため最小ロットで buy 注文を出す。
        buy 完了後に restore_lot_after_dust_sweep() でロット復元される。
        """
        if not self._dust_sweep_active:
            self._pre_dust_lot = self._current_lot
        self._current_lot = self._min_order_btc
        self._dust_sweep_active = True  # apply_lot_floor スキップ
        logger.info(
            f"[dust_sweep] Buy-to-clear: lot={self._min_order_btc} BTC "
            f"(original={self._pre_dust_lot:.4f})"
        )

    def clear_dust_buy_pending(self) -> None:
        """372# dust buy-to-clear 完了: pending フラグをクリア."""
        if self._dust_buy_pending:
            self._dust_buy_pending = False
            logger.info("[dust_sweep] Buy-to-clear pending cleared.")

    def restore_lot_on_success(self) -> None:
        """051# P2-3: 成功時に balance_shrink を解除し、ロットを原値に復元."""
        if self._balance_shrink_active:
            old_lot = self._current_lot
            self._current_lot = self._pre_shrink_lot
            self._balance_shrink_active = False
            logger.info(
                f"[balance_shrink] 解除: ロット復元 {old_lot:.4f} → {self._current_lot:.4f} BTC"
            )
