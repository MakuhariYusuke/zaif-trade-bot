"""120# MakerPriceCalculator — maker limit 価格算出モジュール.

run_fill_test.py FillTestRunner からの God Object 分割:
- _compute_maker_price (214L) → compute()
- _compute_orderbook_imbalance (17L) → compute_imbalance()
- _get_mid_price (8L) → get_mid_price()
- mid price trend 追跡 (54# S4)
- volatility guard (107#)
- spread adaptive offset (054# S4)
- imbalance 補正 (054# S1)
- regime trending boost (052#)
- sell guard (088#)
- FastFillDefense boost 連携

型安全: Optional チェーン明示化、Final 定数。
"""

from __future__ import annotations

import logging
import time
from typing import Final, NamedTuple, Optional, Protocol

from scripts.v460.lib.fast_fill_defense import FastFillDefense
from scripts.v460.lib.fill_config import FillTestConfig

logger = logging.getLogger(__name__)


class OrderbookProvider(Protocol):
    """板情報を提供するアダプタのプロトコル (型安全)."""

    async def get_orderbook(self, symbol: str, depth: int = 1) -> object: ...


class MakerPriceResult(NamedTuple):
    """compute() の戻り値 — 従来の tuple[float, float, float] を型安全化."""

    price: float
    spread: float
    effective_offset_ratio: float


class ImbalanceResult(NamedTuple):
    """板不均衡計算結果."""

    imbalance: float
    bid_total: float
    ask_total: float


# 定数
_BPS_FACTOR: Final[int] = 10_000
_MIN_ORDER_BTC: Final[float] = 0.001


class MakerPriceCalculator:
    """maker limit 価格計算 — FillTestRunner から分割.

    __slots__ でメモリフットプリントを制御。
    """

    __slots__ = (
        "_config",
        "_fast_fill_defense",
        "_regime_detector",
        "_base_offset_ratio",
        "_base_offset_ratio_buy",
        "_base_offset_ratio_sell",
        "_prev_mid_price",
        "_prev_mid_time",
        "_last_mid_trend_bps",
        "_last_imbalance",
        "_last_bid_depth",
        "_last_ask_depth",
        "_last_vpin",
        "_last_vg_triggered",
        "_last_ob_snapshot",
    )

    def __init__(
        self,
        config: FillTestConfig,
        fast_fill_defense: FastFillDefense,
        regime_detector: object | None,
        *,
        base_offset_ratio: float,
        base_offset_ratio_buy: float | None = None,
        base_offset_ratio_sell: float | None = None,
    ) -> None:
        self._config = config
        self._fast_fill_defense = fast_fill_defense
        self._regime_detector = regime_detector
        self._base_offset_ratio = base_offset_ratio
        self._base_offset_ratio_buy = base_offset_ratio_buy
        self._base_offset_ratio_sell = base_offset_ratio_sell
        # mid price trend 追跡
        self._prev_mid_price: float | None = None
        self._prev_mid_time: float | None = None
        self._last_mid_trend_bps: float | None = None
        # imbalance キャッシュ
        self._last_imbalance: float = 0.0
        self._last_bid_depth: float = 0.0
        self._last_ask_depth: float = 0.0
        # 107# Volatility Guard: VPIN キャッシュ
        self._last_vpin: float | None = None
        # 120# P2-1: VG 発動状態追跡 (寄与分解基盤)
        self._last_vg_triggered: bool = False
        # 129# OB recorder: 生スナップショットキャッシュ
        self._last_ob_snapshot: object | None = None

    def get_fallback_price(self) -> tuple[float | None, float | None]:
        """156# §16: OB エラー時のフォールバック価格と記録時刻を返す.

        Returns:
            (prev_mid_price, prev_mid_time) — 未設定時は (None, None).
        """
        return self._prev_mid_price, self._prev_mid_time

    # ------------------------------------------------------------------
    # offset 同期 (adaptation 後に呼ばれる)
    # ------------------------------------------------------------------
    def update_base_offsets(
        self,
        base: float,
        buy: float | None = None,
        sell: float | None = None,
    ) -> None:
        self._base_offset_ratio = base
        self._base_offset_ratio_buy = buy
        self._base_offset_ratio_sell = sell

    @property
    def base_offset_ratio(self) -> float:
        return self._base_offset_ratio

    @property
    def base_offset_ratio_buy(self) -> float | None:
        return self._base_offset_ratio_buy

    @base_offset_ratio_buy.setter
    def base_offset_ratio_buy(self, value: float | None) -> None:
        self._base_offset_ratio_buy = value

    @property
    def base_offset_ratio_sell(self) -> float | None:
        return self._base_offset_ratio_sell

    @base_offset_ratio_sell.setter
    def base_offset_ratio_sell(self, value: float | None) -> None:
        self._base_offset_ratio_sell = value

    @base_offset_ratio.setter
    def base_offset_ratio(self, value: float) -> None:
        self._base_offset_ratio = value

    @property
    def last_vg_triggered(self) -> bool:
        """120# P2-1: 直近の compute() で VG が発動したか."""
        return self._last_vg_triggered

    # ------------------------------------------------------------------
    # 板不均衡 (054# S1)
    # ------------------------------------------------------------------
    async def compute_imbalance(
        self,
        adapter: object,
        symbol: str,
        depth: int = 5,
    ) -> ImbalanceResult:
        """054# S1: 板不均衡を計算.

        Returns:
            ImbalanceResult(imbalance, bid_total, ask_total).
            imbalance ∈ [-1, +1].
        """
        ob = await adapter.get_orderbook(symbol, depth=depth)  # type: ignore[attr-defined]
        # 129# OB recorder: 生スナップショットをキャッシュ
        self._last_ob_snapshot = ob
        bid_volume = sum(qty for _, qty in ob.bids[:depth]) if ob.bids else 0.0
        ask_volume = sum(qty for _, qty in ob.asks[:depth]) if ob.asks else 0.0
        total = bid_volume + ask_volume
        if total == 0:
            return ImbalanceResult(0.0, 0.0, 0.0)
        imbalance = (bid_volume - ask_volume) / total
        # キャッシュ更新
        self._last_imbalance = imbalance
        self._last_bid_depth = bid_volume
        self._last_ask_depth = ask_volume
        return ImbalanceResult(imbalance, bid_volume, ask_volume)

    # ------------------------------------------------------------------
    # mid price (簡易)
    # ------------------------------------------------------------------
    async def get_mid_price(self, adapter: object, symbol: str) -> float:
        """板の best bid/ask から mid price を算出."""
        ob = await adapter.get_orderbook(symbol, depth=1)  # type: ignore[attr-defined]
        if not ob.bids or not ob.asks:
            raise ValueError("Empty orderbook — cannot compute mid price")
        best_bid = ob.bids[0][0]
        best_ask = ob.asks[0][0]
        return (best_bid + best_ask) / 2.0

    # ------------------------------------------------------------------
    # メイン: maker limit 価格算出
    # ------------------------------------------------------------------
    async def compute(
        self,
        side: str,
        adapter: object,
        symbol: str,
    ) -> MakerPriceResult:
        """maker limit 価格を算出: スプレッド比例オフセット + post_only 安全策.

        009# §4.2: スプレッド内側に配置して maker 約定を狙う.
        CM-1: 固定 1 JPY → スプレッド比例 + post_only リジェクト防止.
        054# S1: Imbalance ベース AS リスク補正.
        054# S4: Spread 適応型 offset.

        Returns:
            MakerPriceResult(price, spread, effective_offset_ratio).
        """
        cfg = self._config

        # 054# S1: imbalance 計算
        # 122# §7.3: pre-fetch で常時計算済みのため、キャッシュ値を使用
        if cfg.imbalance_enabled:
            # imbalance_enabled 時は offset 補正に使う (キャッシュ値は pre-fetch 済み)
            imb = self._last_imbalance
        else:
            imb = 0.0

        ob = await adapter.get_orderbook(symbol, depth=1)  # type: ignore[attr-defined]
        if not ob.bids or not ob.asks:
            raise ValueError("Empty orderbook")
        best_bid = ob.bids[0][0]
        best_ask = ob.asks[0][0]
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2.0

        # 054# mid price trend 追跡
        mid_trend_bps: float | None = None
        now = time.time()
        if self._prev_mid_price is not None and self._prev_mid_time is not None:
            dt = now - self._prev_mid_time
            if 0 < dt < cfg.mid_trend_validity_sec:
                mid_trend_bps = (mid_price - self._prev_mid_price) / self._prev_mid_price * _BPS_FACTOR
        self._prev_mid_price = mid_price
        self._prev_mid_time = now
        self._last_mid_trend_bps = mid_trend_bps

        # 031# スプレッドフィルター
        if spread < cfg.min_spread_jpy:
            raise ValueError(
                f"Spread too narrow: {spread:.0f} JPY < min {cfg.min_spread_jpy:.0f}"
            )

        # === offset 決定ロジック ===
        # 096# 状態分離: _base_offset_ratio* を参照
        effective_offset_ratio = self._base_offset_ratio
        if side == "buy" and self._base_offset_ratio_buy is not None:
            effective_offset_ratio = self._base_offset_ratio_buy
        elif side == "sell" and self._base_offset_ratio_sell is not None:
            effective_offset_ratio = self._base_offset_ratio_sell

        # 088# sell 専用ハードガード: offset floor
        if side == "sell" and cfg.sell_offset_floor > 0:
            effective_offset_ratio = max(effective_offset_ratio, cfg.sell_offset_floor)

        # 088# sell 専用: max_spread 超過で sell スキップ
        if (
            side == "sell"
            and cfg.sell_max_spread_jpy > 0
            and spread > cfg.sell_max_spread_jpy
        ):
            logger.info(
                f"[sell_guard] Spread {spread:.0f} JPY > max {cfg.sell_max_spread_jpy:.0f} "
                f"— skipping sell order (088#)"
            )
            raise ValueError(
                f"sell_guard: spread {spread:.0f} > max {cfg.sell_max_spread_jpy:.0f}"
            )

        # 052#: トレンディング時にオフセットをブースト
        if (
            self._regime_detector is not None
            and hasattr(self._regime_detector, "current_regime")
            and self._regime_detector.current_regime.value == "trending"
            and cfg.regime_trending_offset_boost > 1.0
        ):
            effective_offset_ratio *= cfg.regime_trending_offset_boost
            logger.debug(
                f"[regime] trending → offset boosted: "
                f"{effective_offset_ratio / cfg.regime_trending_offset_boost:.4f} "
                f"→ {effective_offset_ratio:.4f}"
            )

        # 143# R-1a: high_vol 時にオフセットをブースト (AS リスク上昇に対応)
        if (
            self._regime_detector is not None
            and hasattr(self._regime_detector, "current_regime")
            and self._regime_detector.current_regime.value == "high_vol"
            and cfg.regime_high_vol_offset_boost > 1.0
        ):
            pre_offset = effective_offset_ratio
            effective_offset_ratio = min(
                effective_offset_ratio * cfg.regime_high_vol_offset_boost,
                cfg.max_offset_ratio,
            )
            logger.debug(
                f"[regime] high_vol → offset boosted: "
                f"{pre_offset:.4f} → {effective_offset_ratio:.4f} "
                f"(boost={cfg.regime_high_vol_offset_boost:.2f})"
            )

        # 143# R-1a: ranging 時にオフセットを縮小 (安定市場で利幅確保)
        if (
            self._regime_detector is not None
            and hasattr(self._regime_detector, "current_regime")
            and self._regime_detector.current_regime.value == "ranging"
            and cfg.regime_ranging_offset_discount < 1.0
        ):
            pre_offset = effective_offset_ratio
            effective_offset_ratio = max(
                effective_offset_ratio * cfg.regime_ranging_offset_discount,
                cfg.min_offset_ratio,
            )
            logger.debug(
                f"[regime] ranging → offset discounted: "
                f"{pre_offset:.4f} → {effective_offset_ratio:.4f} "
                f"(discount={cfg.regime_ranging_offset_discount:.2f})"
            )

        # 130# unknown regime buy guard: offset boost で AS 回避
        if (
            cfg.unknown_buy_offset_boost > 1.0
            and side == "buy"
            and self._regime_detector is not None
            and hasattr(self._regime_detector, "current_regime")
            and (
                self._regime_detector.current_regime is None
                or self._regime_detector.current_regime.value == "unknown"
            )
        ):
            pre_offset = effective_offset_ratio
            effective_offset_ratio = min(
                effective_offset_ratio * cfg.unknown_buy_offset_boost,
                cfg.max_offset_ratio,
            )
            logger.info(
                f"[unknown_buy_guard] 130# buy offset boosted: "
                f"{pre_offset:.4f}→{effective_offset_ratio:.4f} "
                f"(regime=unknown, boost={cfg.unknown_buy_offset_boost:.2f})"
            )

        # 054# S4: Spread 適応型 offset
        if cfg.spread_adaptive_enabled:
            spread_bps = spread / mid_price * _BPS_FACTOR
            if spread_bps < cfg.narrow_spread_bps:
                sa_boost = cfg.narrow_spread_boost
                if side == "buy" and cfg.narrow_spread_boost_buy is not None:
                    sa_boost = cfg.narrow_spread_boost_buy
                elif side == "sell" and cfg.narrow_spread_boost_sell is not None:
                    sa_boost = cfg.narrow_spread_boost_sell
                effective_offset_ratio = min(
                    effective_offset_ratio * sa_boost, cfg.max_offset_ratio,
                )
                logger.debug(
                    f"[spread_adaptive] Narrow spread {spread_bps:.1f}bps "
                    f"({side} boost={sa_boost:.2f}) "
                    f"→ offset boosted to {effective_offset_ratio:.4f}"
                )
            elif spread_bps > cfg.wide_spread_bps:
                effective_offset_ratio = max(
                    effective_offset_ratio * cfg.wide_spread_ratio, cfg.min_offset_ratio,
                )
                logger.debug(
                    f"[spread_adaptive] Wide spread {spread_bps:.1f}bps "
                    f"→ offset reduced to {effective_offset_ratio:.4f}"
                )

        # 091# sell offset floor 事後再適用
        if side == "sell" and cfg.sell_offset_floor > 0:
            if effective_offset_ratio < cfg.sell_offset_floor:
                logger.debug(
                    f"[sell_guard] Post-adaptive floor re-applied: "
                    f"{effective_offset_ratio:.4f} → {cfg.sell_offset_floor:.4f}"
                )
                effective_offset_ratio = cfg.sell_offset_floor

        # 107# Volatility Guard: リアルタイム急変検知 → offset boost
        if cfg.volatility_guard_enabled:
            vg_triggered = False
            vg_reason = ""
            if (
                mid_trend_bps is not None
                and abs(mid_trend_bps) > cfg.volatility_guard_velocity_threshold_bps
            ):
                vg_triggered = True
                vg_reason = f"velocity={mid_trend_bps:.1f}bps"
            if self._last_vpin is not None:
                if self._last_vpin > cfg.volatility_guard_vpin_threshold:
                    vg_triggered = True
                    vg_reason += (f"{'+' if vg_reason else ''}vpin="
                                  f"{self._last_vpin:.2f}")
            if vg_triggered:
                pre_offset = effective_offset_ratio
                effective_offset_ratio = min(
                    effective_offset_ratio * cfg.volatility_guard_offset_boost_factor,
                    cfg.max_offset_ratio,
                )
                logger.info(
                    f"[volatility_guard] 107# {side} offset boosted: "
                    f"{pre_offset:.4f}→{effective_offset_ratio:.4f} "
                    f"({vg_reason})"
                )
            # 120# P2-1: VG 発動状態を追跡
            self._last_vg_triggered = vg_triggered
        else:
            self._last_vg_triggered = False

        # 054# S1: Imbalance ベース AS リスク補正
        if cfg.imbalance_enabled and abs(imb) > cfg.imbalance_threshold:
            as_risk = (
                (side == "buy" and imb < -cfg.imbalance_threshold)
                or (side == "sell" and imb > cfg.imbalance_threshold)
            )
            if as_risk:
                if abs(imb) >= cfg.imbalance_skip_threshold:
                    logger.info(
                        f"[imbalance] Extreme AS risk: {side} imb={imb:+.3f} "
                        f">= skip_threshold {cfg.imbalance_skip_threshold}. "
                        f"Skipping order."
                    )
                    raise ValueError(
                        f"Imbalance skip: {side} order suppressed (imb={imb:+.3f})"
                    )
                else:
                    effective_offset_ratio *= cfg.imbalance_offset_boost
                    effective_offset_ratio = min(effective_offset_ratio, cfg.max_offset_ratio)
                    logger.info(
                        f"[imbalance] {side} AS risk: imb={imb:+.3f}, "
                        f"offset boosted to {effective_offset_ratio:.4f}"
                    )

        offset = max(cfg.min_offset_jpy, spread * effective_offset_ratio)

        # 100# FastFillDefense: per-side boost 乗数を適用
        boost_mult = self._fast_fill_defense.get_boost_multiplier(side)
        if boost_mult != 1.0:
            offset *= boost_mult
            effective_offset_ratio *= boost_mult

        if side == "buy":
            price = best_bid + offset
            if price >= best_ask:
                price = best_bid
                effective_offset_ratio = 0.0
                logger.info(
                    f"Spread guard: buy price {best_bid + offset:.0f} >= ask {best_ask:.0f}, "
                    f"fallback to best_bid {best_bid:.0f} (spread={spread:.0f})"
                )
            return MakerPriceResult(price, spread, effective_offset_ratio)
        else:
            price = best_ask - offset
            if price <= best_bid:
                price = best_ask
                effective_offset_ratio = 0.0
                logger.info(
                    f"Spread guard: sell price {best_ask - offset:.0f} <= bid {best_bid:.0f}, "
                    f"fallback to best_ask {best_ask:.0f} (spread={spread:.0f})"
                )
            return MakerPriceResult(price, spread, effective_offset_ratio)
