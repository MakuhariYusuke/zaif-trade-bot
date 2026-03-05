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

import collections
import logging
import math
import time
from typing import Final, NamedTuple, Protocol, Sequence

from scripts.v460.lib.fast_fill_defense import FastFillDefense
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.ob_utils import OrderBookSnapshot
from scripts.v460.lib.regime_detector import FillTestRegime, RegimeDetectorLike
from scripts.v460.lib.velocity_math import compute_instant_velocity_bps

logger = logging.getLogger(__name__)


class InfeasibleQuoteError(ValueError):
    """239# 232# §1.5: 制約集合崩壊で quote 不可能時の専用例外.

    ValueError のサブクラスなので既存 except ValueError / Exception は互換維持。
    ``reason`` 属性で文字列パース不要の型安全な分類を提供。

    Reasons:
        - ``"spread_too_narrow"`` — spread < min_spread_jpy
        - ``"sell_guard_reject"`` — sell 時 spread > sell_max_spread_jpy
    """

    __slots__ = ("reason",)

    def __init__(self, reason: str, msg: str) -> None:
        super().__init__(msg)
        self.reason = reason


# 266# OrderBookSnapshot は ob_utils.py に移管済み (from scripts.v460.lib.ob_utils import OrderBookSnapshot)


class OrderbookProvider(Protocol):
    """板情報を提供するアダプタのプロトコル (型安全)."""

    async def get_orderbook(self, symbol: str, depth: int = 1) -> OrderBookSnapshot: ...


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

    ╔══════════════════════════════════════════════════════════════╗
    ║  ⚠ GOD OBJECT 化 禁止 — AI コーディングエージェント向け警告  ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  compute() は 163# で 306→143 行に分割済み。               ║
    ║  ステージパイプライン構造:                                  ║
    ║    compute() → _apply_as_reservation_shift()  (258#/266#)   ║
    ║              → _apply_regime_boosts()                      ║
    ║              → _apply_spread_adaptive()                    ║
    ║              → _apply_kyle_lambda()           (266#)       ║
    ║              → _apply_amihud_illiq()           (266#)       ║
    ║              → _apply_volatility_guard()                   ║
    ║              → _apply_imbalance_risk()                     ║
    ║              → _apply_loss_boost()            (260#)       ║
    ║              → _apply_ffd_boost()             (260#)       ║
    ║  共有ヘルパー: _estimate_sigma(), _dynamic_tau()  (266#)   ║
    ║  新ロジック追加時は新しい _apply_*() private メソッドとして ║
    ║  パイプラインに挿入すること。compute() に直接書かない。     ║
    ║  compute() 行数上限: 150 行。                              ║
    ║  クラス全体の行数上限: 850 行。超過時はモジュール分割。     ║
    ╚══════════════════════════════════════════════════════════════╝
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
        "_last_vg_velocity_bps",
        "_last_vg_vpin",
        "_last_vg_boost_factor",
        "_inv_fill_history",        # 162# inventory skewing fill deque
        "_inv_net_imbalance",        # 162# normalized net imbalance [-1,1]
        "_inv_buy_count",            # 226# P5: O(1) incremental buy counter
        "_inv_last_update_time",     # 228# C2: last fill timestamp for time-decay
        "_last_inv_skew_factor",     # 168# last applied inv_skew factor
        "_last_ob_snapshot",
        "_last_spread",              # 197# cached spread for Gate pre-check
        "_last_spread_time",         # 210# M5: staleness tracking
        "_loss_boost_mult",          # 211# 204# I: per-fill loss offset boost
        "_loss_boost_set_time",      # 226# T1: boost 設定時刻 (指数減衰用)
        "_smoothed_velocity_bps",    # 227# C3: EMA-smoothed velocity (bid-ask bounce noise filter)
        "_last_amihud_illiq",        # 266# Amihud ILLIQ キャッシュ
    )

    def __init__(
        self,
        config: FillTestConfig,
        fast_fill_defense: FastFillDefense,
        regime_detector: RegimeDetectorLike | None,
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
        # 158# P2-6: VG 詳細ログ
        self._last_vg_velocity_bps: float | None = None
        self._last_vg_vpin: float | None = None
        self._last_vg_boost_factor: float | None = None
        # 129# OB recorder: 生スナップショットキャッシュ
        # 261# P2-5: object → OrderBookSnapshot (型安全化)
        self._last_ob_snapshot: OrderBookSnapshot | None = None
        # 162# Inventory Skewing: fill 履歴追跡
        _w = config.inventory_skewing_window if config.inventory_skewing_window > 0 else 100
        self._inv_fill_history: collections.deque[str] = collections.deque(maxlen=_w)
        self._inv_net_imbalance: float = 0.0
        self._inv_buy_count: int = 0  # 226# P5: O(1) buy count tracker
        self._inv_last_update_time: float = 0.0  # 228# C2: time-decay 基準時刻
        # 168# InvSkew/VG 競合解消: 直近の InvSkew 補正係数 (負=sell緩和)
        self._last_inv_skew_factor: float = 0.0
        # 197# cached spread for CycleGateAggregator pre-check
        self._last_spread: float | None = None
        self._last_spread_time: float | None = None  # 210# M5: staleness tracking
        # 211# 204# I: per-fill loss offset boost
        # 226# T1: 1-shot → 指数減衰 (Avellaneda-Stoikov AS理論)
        self._loss_boost_mult: float = 1.0
        self._loss_boost_set_time: float = 0.0
        # 227# C3: EMA-smoothed velocity (bid-ask bounce noise filter)
        self._smoothed_velocity_bps: float | None = None
        # 266# Amihud ILLIQ キャッシュ
        self._last_amihud_illiq: float = 0.0

    def get_fallback_price(self) -> tuple[float | None, float | None]:
        """156# §16: OB エラー時のフォールバック価格と記録時刻を返す.

        Returns:
            (prev_mid_price, prev_mid_time) — 未設定時は (None, None).
        """
        return self._prev_mid_price, self._prev_mid_time

    @property
    def last_spread(self) -> float | None:
        """197# Gate 8-9 用: 直近の compute() で算出された spread (JPY).

        210# M5: staleness guard — 60秒以上更新されていない場合は
        None を返し、Gate 8 がstale値でブロックするフィードバックループを防止。
        """
        if (
            self._last_spread_time is not None
            and time.time() - self._last_spread_time > 60.0
        ):
            return None
        return self._last_spread

    @property
    def last_spread_raw(self) -> float | None:
        """217# SAD 用: staleness guard なしの直近 spread (JPY).

        last_spread は 210# M5 の 60s staleness guard 付きだが、
        SAD は cycle 間隔 (120s) で常に前回値を取得する必要がある。
        Gate 8-9 は staleness guard 版を、SAD はこちらを使用する。
        """
        return self._last_spread

    @property
    def last_mid_price(self) -> float | None:
        """197# Gate 8-9 用: 直近の mid price (JPY)."""
        return self._prev_mid_price

    @property
    def last_mid_trend_bps(self) -> float | None:
        """210# H3: 直近の compute() で算出された mid velocity (bps/s)."""
        return self._last_mid_trend_bps

    def update_inventory(self, side: str) -> None:
        """162# Inventory Skewing: fill 後に在庫偏重を更新.

        226# P5: O(n) scan → O(1) incremental counter に改善。
        228# C2: last update timestamp を記録 (time-decay 用)。
        deque maxlen 溢れ時の eviction も追跡。

        Args:
            side: 'buy' or 'sell'  約定した side.
        """
        # maxlen 到達時の eviction 追跡
        dq = self._inv_fill_history
        if len(dq) == dq.maxlen:
            evicted = dq[0]  # 左端が溢れる
            if evicted == "buy":
                self._inv_buy_count -= 1
        dq.append(side)
        if side == "buy":
            self._inv_buy_count += 1
        n = len(dq)
        # imbalance: +1 = all buys (long偏重), -1 = all sells (short偏重)
        self._inv_net_imbalance = (2 * self._inv_buy_count - n) / n
        # 228# C2: fill 時刻を記録 → compute() で time-decay に使用
        self._inv_last_update_time = time.time()

    def _decayed_imbalance(self, now: float) -> float:
        """228# C2: time-decay 適用後の在庫偏重値を返す.

        最終 fill から elapsed 秒が経過した場合、imbalance を
        exp(-elapsed / τ) で減衰させる。τ=0 で無効 (raw 値を返す)。

        理論根拠: Guéant-Lehalle-Fernandez-Tapia (2013) —
        在庫リスクは最終約定からの経過時間とともに情報価値が減衰する。
        古い fill 履歴に基づくポジション偏重の信頼性低下を反映。
        """
        raw = self._inv_net_imbalance
        tau = self._config.inv_decay_tau_sec
        if not isinstance(tau, (int, float)) or tau <= 0 or self._inv_last_update_time <= 0:
            return raw
        elapsed = now - self._inv_last_update_time
        if elapsed <= 0:
            return raw
        return raw * math.exp(-elapsed / tau)

    @property
    def inv_net_imbalance(self) -> float:
        """172# 在庫偏重指標 (public accessor).

        228# C2: inv_decay_tau_sec > 0 の場合、time-decay を適用して返す。

        Returns:
            float ∈ [-1, 1]: +1=全buy(long偏重), -1=全sell(short偏重), 0=均衡.
        """
        return self._decayed_imbalance(time.time())

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

    def update_fast_fill_defense(self, ffd: "FastFillDefense") -> None:
        """210# F: hot-reload 後の FastFillDefense 参照更新 (カプセル化維持)."""
        self._fast_fill_defense = ffd

    def set_loss_boost(self, mult: float) -> None:
        """211# 204# I: 大損直後に offset を一時的に拡大.

        226# T1: 1-shot → 指数減衰に変更。
        AS 理論 (Avellaneda-Stoikov 2008): 大損後の情報非対称性リスクは
        指数的に減衰する。タイムスタンプを記録し、
        compute() 内で exp(-t/τ) で減衰させる。
        """
        self._loss_boost_mult = mult
        self._loss_boost_set_time = time.time()

    @property
    def last_vg_velocity_bps(self) -> float | None:
        """158# P2-6: 直近 VG 評価時の velocity (bps)."""
        return self._last_vg_velocity_bps

    @property
    def last_vg_vpin(self) -> float | None:
        """158# P2-6: 直近 VG 評価時の VPIN."""
        return self._last_vg_vpin

    @property
    def last_vg_boost_factor(self) -> float | None:
        """158# P2-6: 直近 VG 適用 boost 倍率 (1.0=未発動)."""
        return self._last_vg_boost_factor

    def _effective_sell_offset_floor(self) -> float:
        """173# 動的 sell_offset_floor — 在庫 buy 偏重時にフロアを割引.

        InvSkew が sell offset を下げようとする局面でフロアがそれを阻止
        するのを防ぐ。割引率は sell_offset_floor_inv_discount で設定。
        """
        cfg = self._config
        base_floor = cfg.sell_offset_floor
        if base_floor <= 0:
            return 0.0
        bypass_th = cfg.sell_guard_inv_bypass_threshold
        # 228# C2: time-decay を適用した imbalance で判定
        _imb = self._decayed_imbalance(time.time())
        if bypass_th > 0 and _imb >= bypass_th:
            discounted = base_floor * cfg.sell_offset_floor_inv_discount
            if discounted < base_floor:
                logger.debug(
                    f"[sell_guard] Dynamic floor discount: "
                    f"inv_imb={_imb:.3f} >= {bypass_th} "
                    f"→ floor {base_floor:.4f} → {discounted:.4f}"
                )
            return discounted
        return base_floor

    # ------------------------------------------------------------------
    # 板不均衡 (054# S1)
    # ------------------------------------------------------------------
    async def compute_imbalance(
        self,
        adapter: OrderbookProvider,
        symbol: str,
        depth: int = 5,
    ) -> ImbalanceResult:
        """054# S1: 板不均衡を計算.

        Returns:
            ImbalanceResult(imbalance, bid_total, ask_total).
            imbalance ∈ [-1, +1].
        """
        ob = await adapter.get_orderbook(symbol, depth=depth)
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
    async def get_mid_price(self, adapter: OrderbookProvider, symbol: str) -> float:
        """板の best bid/ask から mid price を算出."""
        ob = await adapter.get_orderbook(symbol, depth=1)
        if not ob.bids or not ob.asks:
            raise ValueError("Empty orderbook — cannot compute mid price")
        best_bid = ob.bids[0][0]
        best_ask = ob.asks[0][0]
        return (best_bid + best_ask) / 2.0

    # ------------------------------------------------------------------
    # メイン: maker limit 価格算出
    # ------------------------------------------------------------------

    # ================================================================
    # compute() ステージ抽出メソッド (163# God Object 分割)
    # WARNING: 以下のメソッドは compute() パイプラインの一部。
    #          単独呼出し禁止。新ステージ追加時は compute() 内の呼出し順に注意。
    # ================================================================

    @staticmethod
    def _resolve_trending_boost(
        cfg: "FillTestConfig", regime_val: str, side: str,
    ) -> float:
        """176# B: 方向×サイド別 trending offset boost を解決.

        優先順位:
        1. trending_up_buy_offset_boost 等 (方向×サイド別、最優先)
        2. regime_trending_offset_boost_buy/sell (サイド別)
        3. regime_trending_offset_boost (共通値)
        """
        # --- 1. 方向×サイド別 (176# 新設) ---
        if regime_val == "trending_up":
            if side == "buy" and cfg.trending_up_buy_offset_boost is not None:
                return cfg.trending_up_buy_offset_boost
            if side == "sell" and cfg.trending_up_sell_offset_boost is not None:
                return cfg.trending_up_sell_offset_boost
        elif regime_val == "trending_down":
            if side == "buy" and cfg.trending_down_buy_offset_boost is not None:
                return cfg.trending_down_buy_offset_boost
            if side == "sell" and cfg.trending_down_sell_offset_boost is not None:
                return cfg.trending_down_sell_offset_boost
        # regime_val == "trending" (方向不明) → 方向別なし → 下位にフォールバック

        # --- 2. サイド別 (157# §19 既存) ---
        if side == "buy" and cfg.regime_trending_offset_boost_buy is not None:
            return cfg.regime_trending_offset_boost_buy
        if side == "sell" and cfg.regime_trending_offset_boost_sell is not None:
            return cfg.regime_trending_offset_boost_sell

        # --- 3. 共通値 ---
        return cfg.regime_trending_offset_boost

    @staticmethod
    def _scale_offset_ratio(
        effective_offset_ratio: float,
        multiplier: float,
        *,
        min_ratio: float | None = None,
        max_ratio: float | None = None,
    ) -> tuple[float, float]:
        """offset ratio に倍率を安全適用し、実際の適用倍率も返す.

        誤設定で 0 以下の倍率が来ても ratio を壊さず no-op にする。
        """
        if effective_offset_ratio <= 0 or multiplier <= 0:
            return effective_offset_ratio, 1.0

        updated = effective_offset_ratio * multiplier
        if min_ratio is not None:
            updated = max(updated, min_ratio)
        if max_ratio is not None:
            updated = min(updated, max_ratio)
        return updated, (updated / effective_offset_ratio)

    @staticmethod
    def _finalize_price_with_spread_guard(
        *,
        side: str,
        best_bid: float,
        best_ask: float,
        spread: float,
        offset: float,
        effective_offset_ratio: float,
    ) -> MakerPriceResult:
        """最終価格を組み立て、cross 時は spread guard で安全側へ戻す."""
        if side == "buy":
            price = best_bid + offset
            if price >= best_ask:
                logger.info(
                    f"Spread guard: buy price {price:.0f} >= ask {best_ask:.0f}, "
                    f"fallback to best_bid {best_bid:.0f} (spread={spread:.0f})"
                )
                return MakerPriceResult(best_bid, spread, 0.0)
            return MakerPriceResult(price, spread, effective_offset_ratio)

        price = best_ask - offset
        if price <= best_bid:
            logger.info(
                f"Spread guard: sell price {price:.0f} <= bid {best_bid:.0f}, "
                f"fallback to best_ask {best_ask:.0f} (spread={spread:.0f})"
            )
            return MakerPriceResult(best_ask, spread, 0.0)
        return MakerPriceResult(price, spread, effective_offset_ratio)

    # ------------------------------------------------------------------
    # 266# 共有ヘルパー: σ推定 + τ動的化 (GLFT/AS δ*/Kyle で再利用)
    # ------------------------------------------------------------------
    def _get_depth(self, side: str) -> float:
        """267# DRY: side に応じた板 depth volume を返す.

        Kyle λ / Amihud ILLIQ 等で共通使用するヘルパー。
        """
        return self._last_bid_depth if side == "buy" else self._last_ask_depth

    def _estimate_sigma(self, spread: float, mid_price: float) -> tuple[float, float]:
        """266# σ 推定: Roll (1984) micro-vol proxy × RegimeDetector vol_ratio.

        Returns:
            (sigma, vol_ratio) — σ 推定値と vol_ratio。
            _apply_as_reservation_shift (AS + δ*) で直接使用。
            _apply_kyle_lambda, _apply_amihud_illiq は depth ベースで独自推定。
        """
        sigma = spread / (2.0 * mid_price) if mid_price > 0 else 0.0
        vol_ratio = 1.0
        if self._regime_detector is not None:
            vol_ratio = max(self._regime_detector.last_volatility_ratio, 0.1)
        sigma *= vol_ratio
        return sigma, vol_ratio

    def _dynamic_tau(self, base_tau: float, vol_ratio: float) -> float:
        """266# GLFT τ動的化: Guéant-Lehalle-Fernandez-Tapia (2013).

        τ_eff = τ_base / vol_ratio — 高ボラ時は τ 短縮 (素早い在庫調整)、
        低ボラ時は τ 延長 (緩やかな調整)。有限期間 AS モデルの拡張。

        Args:
            base_tau: as_reservation_tau_sec (ベース τ)
            vol_ratio: RegimeDetector.last_volatility_ratio

        Returns:
            実効 τ (as_tau_dynamic_min_sec ≤ τ_eff ≤ as_tau_dynamic_max_sec)
        """
        cfg = self._config
        if not cfg.as_tau_dynamic_enabled or vol_ratio <= 0:
            return base_tau
        tau_eff = base_tau / vol_ratio
        tau_eff = max(cfg.as_tau_dynamic_min_sec, min(cfg.as_tau_dynamic_max_sec, tau_eff))
        return tau_eff

    def _apply_as_reservation_shift(
        self,
        side: str,
        spread: float,
        mid_price: float,
        effective_offset_ratio: float,
    ) -> float:
        """257# AS Reservation Price: Avellaneda-Stoikov 在庫×ボラ連動 offset.

        Avellaneda-Stoikov (2008) 予約価格理論:
          r = s - q·γ·σ²·τ

        266# 拡張:
          - _estimate_sigma(): Roll proxy × vol_ratio (他ステージと共有)
          - _dynamic_tau(): GLFT τ動的化 (τ_eff = τ_base / vol_ratio)
          - AS δ*: 理論的最適 offset 下限 (δ* = γσ²τ + (2/γ)ln(1 + γ/k))
        """
        cfg = self._config
        if not cfg.as_reservation_enabled:
            return effective_offset_ratio

        now = time.time()
        q = self._decayed_imbalance(now)
        if abs(q) < cfg.inventory_skewing_neutral_band:
            return effective_offset_ratio

        gamma = cfg.as_reservation_gamma
        base_tau = cfg.as_reservation_tau_sec
        if gamma <= 0 or base_tau <= 0 or mid_price <= 0:
            return effective_offset_ratio

        # 266# 共有 σ 推定 (Roll 1984 × RegimeDetector vol_ratio)
        sigma, vol_ratio = self._estimate_sigma(spread, mid_price)
        sigma_sq = sigma * sigma

        # 266# GLFT τ動的化
        tau = self._dynamic_tau(base_tau, vol_ratio)

        # AS reservation shift in offset ratio units:
        # delta = q · γ · σ² · τ
        delta = q * gamma * sigma_sq * tau
        sign = 1.0 if side == "buy" else -1.0
        shift = delta * sign

        if abs(shift) < 1e-8:
            return effective_offset_ratio

        prev = effective_offset_ratio
        effective_offset_ratio = max(
            cfg.min_offset_ratio,
            min(cfg.max_offset_ratio, effective_offset_ratio + shift),
        )

        # 266#/267# AS δ*: 理論的最適スプレッド幅下限
        # δ* = γσ²τ + (2/γ)ln(1 + γ/k) (Avellaneda-Stoikov 2008 §4)
        # 注意: AS 論文の σ は絶対価格 (JPY/√s) ベース。_estimate_sigma は
        # リターンベース (無次元) なので σ_abs = σ_return × mid_price に変換。
        # δ* (JPY) → offset_ratio = δ* / spread で変換。
        if cfg.as_delta_star_enabled and gamma > 0:
            k = cfg.as_delta_star_fill_rate_k
            if k > 0:
                sigma_abs = sigma * mid_price  # リターン → 絶対価格 (JPY)
                sigma_abs_sq = sigma_abs * sigma_abs
                delta_star_jpy = (
                    gamma * sigma_abs_sq * tau
                    + (2.0 / gamma) * math.log(1.0 + gamma / k)
                )
                # δ* (JPY) → offset_ratio (無次元)
                if spread > 0:
                    delta_star_ratio = delta_star_jpy / spread
                    if effective_offset_ratio < delta_star_ratio:
                        logger.debug(
                            f"[as_delta_star] 266# {side} δ*={delta_star_ratio:.4f} "
                            f"> offset={effective_offset_ratio:.4f} → floor applied"
                        )
                        effective_offset_ratio = min(delta_star_ratio, cfg.max_offset_ratio)

        if effective_offset_ratio != prev:
            logger.info(
                f"[as_reservation] 266# {side} q={q:+.3f} γ={gamma:.3f} "
                f"σ²={sigma_sq:.2e} vol_ratio={vol_ratio:.3f} "
                f"τ={tau:.0f}s{'(dyn)' if cfg.as_tau_dynamic_enabled else ''} → "
                f"offset {prev:.4f} → {effective_offset_ratio:.4f} "
                f"(shift={shift:+.2e})"
            )

        return effective_offset_ratio

    def _apply_regime_boosts(
        self, side: str, effective_offset_ratio: float,
    ) -> float:
        """052# 143# 130# regime 別 offset 補正.

        260# P2-3: 5 独立ステージに分割。
        - trending: buy/sell 非対称 boost
        - high_vol: offset 拡大 (AS リスク上昇)
        - ranging: offset 縮小 (安定市場で利幅確保)
        - low_vol: offset 拡大 (過剰アグレッシブ抑制)
        - unknown: buy guard offset boost
        """
        effective_offset_ratio = self._regime_boost_trending(side, effective_offset_ratio)
        effective_offset_ratio = self._regime_boost_high_vol(side, effective_offset_ratio)
        effective_offset_ratio = self._regime_boost_ranging(side, effective_offset_ratio)
        effective_offset_ratio = self._regime_boost_low_vol(side, effective_offset_ratio)
        effective_offset_ratio = self._regime_boost_unknown_buy(side, effective_offset_ratio)
        return effective_offset_ratio

    def _regime_boost_trending(
        self, side: str, effective_offset_ratio: float,
    ) -> float:
        """052# 156# 157# 176# trending regime offset boost.

        trending_up/trending_down × buy/sell 非対称:
        有利方向取引では boost 不要。
        """
        cfg = self._config
        if (
            self._regime_detector is not None
            and self._regime_detector.current_regime.is_trending
        ):
            _regime_val = self._regime_detector.current_regime.value
            _trending_boost = self._resolve_trending_boost(cfg, _regime_val, side)

            if _trending_boost != 1.0:
                pre_offset = effective_offset_ratio
                effective_offset_ratio, _applied_mult = self._scale_offset_ratio(
                    effective_offset_ratio, _trending_boost,
                )
                _direction = "boosted" if _applied_mult > 1.0 else "discounted"
                if _applied_mult == 1.0:
                    _direction = "unchanged"
                logger.debug(
                    f"[regime] {_regime_val} → {side} offset {_direction}: "
                    f"{pre_offset:.4f} → {effective_offset_ratio:.4f} "
                    f"(mult={_applied_mult:.2f})"
                )
        return effective_offset_ratio

    def _regime_boost_high_vol(
        self, side: str, effective_offset_ratio: float,
    ) -> float:
        """143# R-1a: high_vol 時にオフセットをブースト (AS リスク上昇に対応)."""
        cfg = self._config
        if (
            self._regime_detector is not None
            and self._regime_detector.current_regime == FillTestRegime.HIGH_VOL
            and cfg.regime_high_vol_offset_boost > 1.0
        ):
            pre_offset = effective_offset_ratio
            effective_offset_ratio, _applied_mult = self._scale_offset_ratio(
                effective_offset_ratio,
                cfg.regime_high_vol_offset_boost,
                max_ratio=cfg.max_offset_ratio,
            )
            logger.debug(
                f"[regime] high_vol → offset boosted: "
                f"{pre_offset:.4f} → {effective_offset_ratio:.4f} "
                f"(boost={_applied_mult:.2f})"
            )
        return effective_offset_ratio

    def _regime_boost_ranging(
        self, side: str, effective_offset_ratio: float,
    ) -> float:
        """143# R-1a: ranging 時にオフセットを縮小 (安定市場で利幅確保).

        227# C1: OBI (Order Book Imbalance) を活用した方向別非対称 discount.
        AS理論: ranging (mean-reverting) 市場では板不均衡がリバージョン方向を予測。
        """
        cfg = self._config
        if (
            self._regime_detector is not None
            and self._regime_detector.current_regime == FillTestRegime.RANGING
            and cfg.regime_ranging_offset_discount < 1.0
        ):
            _ranging_mult = cfg.regime_ranging_offset_discount
            # 227# C1: OBI 方向別非対称化
            if cfg.ranging_obi_asymmetry_factor > 0.0:
                _imb = self._last_imbalance
                _obi_thresh = cfg.ranging_obi_threshold
                if abs(_imb) > _obi_thresh:
                    _obi_adj = _imb * cfg.ranging_obi_asymmetry_factor
                    if side == "buy":
                        _ranging_mult = _ranging_mult * (1.0 - _obi_adj)
                    else:
                        _ranging_mult = _ranging_mult * (1.0 + _obi_adj)
                    _ranging_mult = max(cfg.min_offset_ratio / max(effective_offset_ratio, 1e-6),
                                        min(_ranging_mult, 1.0))
            pre_offset = effective_offset_ratio
            effective_offset_ratio, _applied_mult = self._scale_offset_ratio(
                effective_offset_ratio,
                _ranging_mult,
                min_ratio=cfg.min_offset_ratio,
            )
            logger.debug(
                f"[regime] ranging → offset discounted: "
                f"{pre_offset:.4f} → {effective_offset_ratio:.4f} "
                f"(discount={_applied_mult:.2f}, obi={self._last_imbalance:+.3f})"
            )
        return effective_offset_ratio

    def _regime_boost_low_vol(
        self, side: str, effective_offset_ratio: float,
    ) -> float:
        """168# 低ボラティリティ offset boost: vol_ratio < threshold で offset 拡大.

        time_filter の根本対策 — 低 vol 環境での過剰アグレッシブ発注を構造的に抑制。
        200# C: 比例モード — vol_ratio に応じた段階的 boost。
        """
        cfg = self._config
        if (
            cfg.low_vol_offset_boost_enabled
            and self._regime_detector is not None
        ):
            vol_ratio = self._regime_detector.last_volatility_ratio
            if vol_ratio < cfg.low_vol_threshold:
                if cfg.low_vol_boost_proportional and cfg.low_vol_threshold > 0:
                    _ratio = 1.0 - vol_ratio / cfg.low_vol_threshold
                    _low_vol_boost = cfg.low_vol_boost_min + (
                        cfg.low_vol_offset_boost - cfg.low_vol_boost_min
                    ) * _ratio
                else:
                    _low_vol_boost = cfg.low_vol_offset_boost
                pre_offset = effective_offset_ratio
                effective_offset_ratio, _applied_mult = self._scale_offset_ratio(
                    effective_offset_ratio,
                    _low_vol_boost,
                    max_ratio=cfg.max_offset_ratio,
                )
                logger.info(
                    f"[low_vol_boost] 168# {side} vol_ratio={vol_ratio:.3f} "
                    f"< {cfg.low_vol_threshold:.2f} → offset boosted: "
                    f"{pre_offset:.4f}→{effective_offset_ratio:.4f} "
                    f"(boost={_applied_mult:.2f})"
                )
        return effective_offset_ratio

    def _regime_boost_unknown_buy(
        self, side: str, effective_offset_ratio: float,
    ) -> float:
        """130# unknown regime buy guard: offset boost で AS 回避."""
        cfg = self._config
        if (
            cfg.unknown_buy_offset_boost > 1.0
            and side == "buy"
            and self._regime_detector is not None
            and (
                self._regime_detector.current_regime is None
                or self._regime_detector.current_regime == FillTestRegime.UNKNOWN
            )
        ):
            pre_offset = effective_offset_ratio
            effective_offset_ratio, _applied_mult = self._scale_offset_ratio(
                effective_offset_ratio,
                cfg.unknown_buy_offset_boost,
                max_ratio=cfg.max_offset_ratio,
            )
            logger.info(
                f"[unknown_buy_guard] 130# buy offset boosted: "
                f"{pre_offset:.4f}→{effective_offset_ratio:.4f} "
                f"(regime=unknown, boost={_applied_mult:.2f})"
            )
        return effective_offset_ratio

        return effective_offset_ratio

    def _apply_spread_adaptive(
        self,
        side: str,
        spread: float,
        mid_price: float,
        effective_offset_ratio: float,
    ) -> float:
        """054# S4: Spread 適応型 offset + 091# sell floor 事後再適用."""
        cfg = self._config

        if cfg.spread_adaptive_enabled:
            spread_bps = spread / mid_price * _BPS_FACTOR
            if spread_bps < cfg.narrow_spread_bps:
                sa_boost = cfg.narrow_spread_boost
                if side == "buy" and cfg.narrow_spread_boost_buy is not None:
                    sa_boost = cfg.narrow_spread_boost_buy
                elif side == "sell" and cfg.narrow_spread_boost_sell is not None:
                    sa_boost = cfg.narrow_spread_boost_sell
                effective_offset_ratio, _applied_mult = self._scale_offset_ratio(
                    effective_offset_ratio,
                    sa_boost,
                    max_ratio=cfg.max_offset_ratio,
                )
                logger.debug(
                    f"[spread_adaptive] Narrow spread {spread_bps:.1f}bps "
                    f"({side} boost={_applied_mult:.2f}) "
                    f"→ offset boosted to {effective_offset_ratio:.4f}"
                )
            elif spread_bps > cfg.wide_spread_bps:
                effective_offset_ratio, _applied_mult = self._scale_offset_ratio(
                    effective_offset_ratio,
                    cfg.wide_spread_ratio,
                    min_ratio=cfg.min_offset_ratio,
                )
                logger.debug(
                    f"[spread_adaptive] Wide spread {spread_bps:.1f}bps "
                    f"(mult={_applied_mult:.2f}) → offset reduced to {effective_offset_ratio:.4f}"
                )

        # 091# sell offset floor 事後再適用 (173# 動的フロア対応)
        if side == "sell":
            _dyn_floor = self._effective_sell_offset_floor()
            if _dyn_floor > 0 and effective_offset_ratio < _dyn_floor:
                logger.debug(
                    f"[sell_guard] Post-adaptive floor re-applied: "
                    f"{effective_offset_ratio:.4f} → {_dyn_floor:.4f}"
                )
                effective_offset_ratio = _dyn_floor

        return effective_offset_ratio

    def _apply_kyle_lambda(
        self,
        side: str,
        spread: float,
        mid_price: float,
        effective_offset_ratio: float,
    ) -> float:
        """266# Kyle λ: 価格インパクト係数 (Kyle 1985).

        λ_est = spread / (2 · depth_volume)
        自己注文の市場インパクトを推定し、offset に安全マージンを加算する。
        BTC/JPY 0.001 BTC の小注文では影響は軽微だが、板厚が薄い時間帯で
        spread 拡大を先行的に行い、不利約定を予防する。

        _estimate_sigma / _last_bid_depth / _last_ask_depth を再利用。
        """
        cfg = self._config
        if not cfg.kyle_lambda_enabled or spread <= 0 or mid_price <= 0:
            return effective_offset_ratio

        # depth_volume は compute_imbalance で更新済み (267# _get_depth DRY)
        depth = self._get_depth(side)
        if depth <= 0:
            return effective_offset_ratio

        # Kyle λ 推定: λ = spread / (2 · depth_volume)
        kyle_lambda = spread / (2.0 * depth)

        # 自己注文サイズ (config.order_quantity)
        lot = cfg.order_quantity
        # impact = λ · lot → offset ratio 単位に変換
        impact_ratio = (kyle_lambda * lot / mid_price) * cfg.kyle_lambda_impact_mult
        impact_ratio = min(impact_ratio, cfg.kyle_lambda_max_add_ratio)

        if impact_ratio < 1e-8:
            return effective_offset_ratio

        prev = effective_offset_ratio
        effective_offset_ratio = min(
            cfg.max_offset_ratio,
            effective_offset_ratio + impact_ratio,
        )

        if effective_offset_ratio != prev:
            logger.debug(
                f"[kyle_lambda] 266# {side} λ={kyle_lambda:.4e} depth={depth:.4f} "
                f"lot={lot:.4f} → offset {prev:.4f}→{effective_offset_ratio:.4f} "
                f"(+{impact_ratio:.2e})"
            )

        return effective_offset_ratio

    def _apply_amihud_illiq(
        self,
        side: str,
        spread: float,
        mid_price: float,
        effective_offset_ratio: float,
    ) -> float:
        """266# Amihud ILLIQ: 非流動性比率 (Amihud 2002).

        ILLIQ = |ΔP/P| / Volume ≈ (spread/mid) / depth_volume
        高 ILLIQ = 低流動性 → offset 拡大で保守的に。
        spread_adaptive の固定閾値を連続的に補完する。

        _estimate_sigma / _last_bid_depth / _last_ask_depth を再利用。
        """
        cfg = self._config
        if not cfg.amihud_illiq_enabled or spread <= 0 or mid_price <= 0:
            return effective_offset_ratio

        # 双方向の depth volume (267# _get_depth DRY)
        total_depth = self._get_depth("buy") + self._get_depth("sell")
        if total_depth <= 0:
            return effective_offset_ratio

        # Amihud ILLIQ 推定: |R| / V ≈ (spread/mid) / depth
        illiq = (spread / mid_price) / total_depth
        self._last_amihud_illiq = illiq

        # baseline 対比の倍率
        baseline = cfg.amihud_illiq_baseline
        if baseline <= 0:
            return effective_offset_ratio

        illiq_ratio = illiq / baseline
        if illiq_ratio <= 1.0:
            # 流動性十分 → 補正なし
            return effective_offset_ratio

        # ILLIQ 由来の offset 倍率 (上限あり)
        mult = min(illiq_ratio, cfg.amihud_illiq_max_mult)
        prev = effective_offset_ratio
        effective_offset_ratio, _applied = self._scale_offset_ratio(
            effective_offset_ratio, mult, max_ratio=cfg.max_offset_ratio,
        )

        if effective_offset_ratio != prev:
            logger.debug(
                f"[amihud_illiq] 266# {side} ILLIQ={illiq:.4e} "
                f"ratio={illiq_ratio:.2f} mult={_applied:.3f} → "
                f"offset {prev:.4f}→{effective_offset_ratio:.4f}"
            )

        return effective_offset_ratio

    def _apply_volatility_guard(
        self,
        side: str,
        mid_trend_bps: float | None,
        effective_offset_ratio: float,
    ) -> float:
        """107# Volatility Guard: リアルタイム急変検知 → offset boost.

        257# MT-3: VPIN 連続スケーリング対応。
        vg_vpin_continuous_enabled=True の場合、VPIN が min から threshold の間で
        二次関数的に boost を段階適用する。バイナリ閾値判定による急激な
        offset ジャンプを回避し、情報非対称性リスクを滑らかに反映する。
        """
        cfg = self._config

        if cfg.volatility_guard_enabled:
            vg_triggered = False
            vg_reason = ""
            _vg_velocity = mid_trend_bps  # 158# P2-6: ログ用
            _vg_vpin = self._last_vpin    # 158# P2-6: ログ用

            # --- Velocity trigger (unchanged) ---
            velocity_boost = 1.0
            if (
                mid_trend_bps is not None
                and abs(mid_trend_bps) > cfg.volatility_guard_velocity_threshold_bps
            ):
                velocity_boost = cfg.volatility_guard_offset_boost_factor
                vg_reason = f"velocity={mid_trend_bps:.1f}bps"

            # --- VPIN trigger: 257# continuous or binary ---
            vpin_boost = 1.0
            if self._last_vpin is not None:
                if cfg.vg_vpin_continuous_enabled:
                    # 257# MT-3: 二次関数ランプ — 緩やかな onset と閾値付近の急峻化
                    _min_vpin = cfg.vg_vpin_continuous_min
                    _thresh = cfg.volatility_guard_vpin_threshold
                    if self._last_vpin > _min_vpin and _thresh > _min_vpin:
                        _norm = min(
                            (self._last_vpin - _min_vpin) / (_thresh - _min_vpin),
                            1.0,
                        )
                        vpin_boost = 1.0 + (
                            cfg.volatility_guard_offset_boost_factor - 1.0
                        ) * _norm * _norm  # quadratic ramp
                        vg_reason += (
                            f"{'+' if vg_reason else ''}vpin={self._last_vpin:.2f}"
                            f"(cont={_norm:.2f})"
                        )
                else:
                    # Legacy binary mode
                    if self._last_vpin > cfg.volatility_guard_vpin_threshold:
                        vpin_boost = cfg.volatility_guard_offset_boost_factor
                        vg_reason += (
                            f"{'+' if vg_reason else ''}vpin="
                            f"{self._last_vpin:.2f}"
                        )

            # --- 最終 boost: velocity と VPIN の max ---
            _raw_boost = max(velocity_boost, vpin_boost)
            vg_triggered = _raw_boost > 1.0

            _vg_boost = 1.0  # 158# P2-6: 実際の boost 倍率
            if vg_triggered:
                pre_offset = effective_offset_ratio
                # 168# InvSkew/VG 競合解消: InvSkew が offset を緩和している場合、
                # VG boost 倍率を抑制して在庫リバランス効果を保全する。
                # damping: InvSkew factor が負(=sell offset 縮小)なら
                #   effective_boost = 1 + (1 - |factor|) * (boost_factor - 1)
                if (
                    cfg.vg_inv_skew_damping_enabled
                    and self._last_inv_skew_factor < 0.0
                ):
                    _damping = 1.0 - min(abs(self._last_inv_skew_factor), 1.0)
                    _damped = 1.0 + _damping * (_raw_boost - 1.0)
                    logger.info(
                        f"[vg_damping] 168# InvSkew factor="
                        f"{self._last_inv_skew_factor:+.4f} → "
                        f"VG boost {_raw_boost:.4f}"
                        f"→{_damped:.4f}"
                    )
                    _raw_boost = _damped
                effective_offset_ratio, _vg_boost = self._scale_offset_ratio(
                    effective_offset_ratio,
                    _raw_boost,
                    max_ratio=cfg.max_offset_ratio,
                )
                logger.info(
                    f"[volatility_guard] 107# {side} offset boosted: "
                    f"{pre_offset:.4f}→{effective_offset_ratio:.4f} "
                    f"({vg_reason})"
                )
            # 120# P2-1: VG 発動状態を追跡
            self._last_vg_triggered = vg_triggered
            # 158# P2-6: VG 詳細ログ (ヒンドサイト分析用)
            self._last_vg_velocity_bps = _vg_velocity
            self._last_vg_vpin = _vg_vpin
            self._last_vg_boost_factor = _vg_boost if vg_triggered else None
        else:
            self._last_vg_triggered = False
            self._last_vg_velocity_bps = None
            self._last_vg_vpin = None
            self._last_vg_boost_factor = None

        return effective_offset_ratio

    def _apply_imbalance_risk(
        self,
        side: str,
        imb: float,
        effective_offset_ratio: float,
    ) -> float:
        """054# S1: Imbalance ベース AS リスク補正.

        Raises:
            ValueError: Extreme AS risk (imb >= skip_threshold) → 注文抑止.
        """
        cfg = self._config

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
                    effective_offset_ratio, _applied_mult = self._scale_offset_ratio(
                        effective_offset_ratio,
                        cfg.imbalance_offset_boost,
                        max_ratio=cfg.max_offset_ratio,
                    )
                    logger.info(
                        f"[imbalance] {side} AS risk: imb={imb:+.3f}, "
                        f"offset boosted to {effective_offset_ratio:.4f} "
                        f"(mult={_applied_mult:.2f})"
                    )

        return effective_offset_ratio

    def _apply_buy_as_guard(
        self,
        side: str,
        mid_trend_bps: float | None,
        effective_offset_ratio: float,
    ) -> float:
        """286# 283# P1-6 / 284# P1: Buy-side AS Guard.

        Glosten-Milgrom (1985): 価格急落時は情報トレーダーが sell 主導で
        参入し、buy maker は逆選択コストを被る。microprice velocity が
        閾値を超えて下落中の場合、buy offset を強制拡大して防御する。

        sell 側は下落時に順方向であり AS リスクが相対的に低いため、
        このガードは buy のみに適用する (理論的非対称性)。
        """
        cfg = self._config
        if not cfg.buy_as_guard_enabled:
            return effective_offset_ratio
        if side != "buy":
            return effective_offset_ratio
        if mid_trend_bps is None:
            return effective_offset_ratio

        # velocity が閾値以下 (急落中) で発動
        if mid_trend_bps <= cfg.buy_as_guard_velocity_threshold_bps:
            _prev = effective_offset_ratio
            effective_offset_ratio, _applied = self._scale_offset_ratio(
                effective_offset_ratio,
                cfg.buy_as_guard_offset_mult,
                max_ratio=cfg.buy_as_guard_max_offset_ratio,
            )
            logger.debug(
                f"[286# buy_as_guard] velocity={mid_trend_bps:.2f}bps "
                f"<= {cfg.buy_as_guard_velocity_threshold_bps:.1f}bps — "
                f"buy offset expanded {_prev:.4f} → {effective_offset_ratio:.4f} "
                f"(mult={_applied:.2f})"
            )
        return effective_offset_ratio

    def _apply_loss_boost(
        self,
        side: str,
        now: float,
        effective_offset_ratio: float,
    ) -> float:
        """260# P2-2: 211# 204# 226# T1 loss offset boost (指数減衰).

        Avellaneda-Stoikov: AS リスクは指数的に減衰
        mult(t) = 1 + (M-1)·exp(-t/τ)
        """
        if self._loss_boost_mult <= 1.0 or self._loss_boost_set_time <= 0.0:
            return effective_offset_ratio

        cfg = self._config
        _elapsed = now - self._loss_boost_set_time
        _tau = cfg.loss_boost_decay_tau_sec
        if _tau > 0 and _elapsed > 0:
            _decay = math.exp(-_elapsed / _tau)
        else:
            _decay = 1.0
        _decayed_mult = 1.0 + (self._loss_boost_mult - 1.0) * _decay

        # 減衰が十分 (mult < 1.01) ならリセット
        if _decayed_mult < 1.01:
            self._loss_boost_mult = 1.0
            self._loss_boost_set_time = 0.0
            return effective_offset_ratio

        pre_offset = effective_offset_ratio
        effective_offset_ratio, _applied_mult = self._scale_offset_ratio(
            effective_offset_ratio,
            _decayed_mult,
            max_ratio=cfg.max_offset_ratio,
        )
        if _applied_mult != 1.0:
            logger.info(
                f"[226# T1] Loss boost (decay): {side} offset "
                f"{pre_offset:.4f} → {effective_offset_ratio:.4f} "
                f"(mult={_decayed_mult:.3f}, elapsed={_elapsed:.0f}s, "
                f"τ={_tau:.0f}s)"
            )
        return effective_offset_ratio

    def _apply_ffd_boost(
        self,
        side: str,
        spread: float,
        effective_offset_ratio: float,
        offset: float,
    ) -> tuple[float, float]:
        """260# P2-2: 100# FastFillDefense per-side boost 乗数.

        236# CQS: TTL decay を getter 前に明示的に実行。
        175# FFD boost 後も max_offset_ratio クランプを適用し、
        実際の価格補正量と返却 ratio の整合を保つ。

        Returns:
            (effective_offset_ratio, offset)
        """
        cfg = self._config
        self._fast_fill_defense.maybe_expire_boost(side)
        boost_mult = self._fast_fill_defense.get_boost_multiplier(side)
        if boost_mult != 1.0:
            effective_offset_ratio, _applied_mult = self._scale_offset_ratio(
                effective_offset_ratio,
                boost_mult,
                max_ratio=cfg.max_offset_ratio,
            )
            if _applied_mult != 1.0:
                offset = max(cfg.min_offset_jpy, spread * effective_offset_ratio)
        return effective_offset_ratio, offset

    async def compute(
        self,
        side: str,
        adapter: OrderbookProvider,
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

        ob = await adapter.get_orderbook(symbol, depth=1)
        if not ob.bids or not ob.asks:
            raise ValueError("Empty orderbook")
        best_bid = ob.bids[0][0]
        best_ask = ob.asks[0][0]
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2.0

        # 054# → 208# SSOT: mid price velocity を velocity_math で算出
        now = time.time()
        mid_trend_bps: float | None = None
        if self._prev_mid_price is not None and self._prev_mid_time is not None:
            mid_trend_bps = compute_instant_velocity_bps(
                current_mid=mid_price,
                prev_mid=self._prev_mid_price,
                dt=now - self._prev_mid_time,
                max_dt=cfg.mid_trend_validity_sec,
            )
            # 227# C3: EMA smoothing — bid-ask bounce noise filter
            # 薄い板の1-tick変動でmidが振れるノイズを抑制。
            # α = velocity_ema_alpha (default 0.3): 低いほど滑らか。
            if mid_trend_bps is not None and cfg.velocity_ema_alpha < 1.0:
                _alpha = cfg.velocity_ema_alpha
                if self._smoothed_velocity_bps is not None:
                    mid_trend_bps = _alpha * mid_trend_bps + (1.0 - _alpha) * self._smoothed_velocity_bps
                self._smoothed_velocity_bps = mid_trend_bps
        self._prev_mid_price = mid_price
        self._prev_mid_time = now
        self._last_mid_trend_bps = mid_trend_bps
        self._last_spread = spread  # 197# Gate pre-check 用キャッシュ
        self._last_spread_time = now  # 210# M5: staleness tracking

        # 031# スプレッドフィルター
        # 239# 232# §1.5: InfeasibleQuoteError で型安全分類
        if spread < cfg.min_spread_jpy:
            raise InfeasibleQuoteError(
                reason="spread_too_narrow",
                msg=f"Spread too narrow: {spread:.0f} JPY < min {cfg.min_spread_jpy:.0f}",
            )

        # 088# sell 専用: max_spread 超過で sell スキップ
        # 239# 232# §1.5: offset 計算前に前方移動 — 構造的に不可能なサイクルの早期離脱
        if (
            side == "sell"
            and cfg.sell_max_spread_jpy > 0
            and spread > cfg.sell_max_spread_jpy
        ):
            logger.info(
                f"[sell_guard] Spread {spread:.0f} JPY > max {cfg.sell_max_spread_jpy:.0f} "
                f"— skipping sell order (088# → 239# early bailout)"
            )
            raise InfeasibleQuoteError(
                reason="sell_guard_reject",
                msg=f"sell_guard: spread {spread:.0f} > max {cfg.sell_max_spread_jpy:.0f}",
            )

        # === offset 決定ロジック ===
        # 096# 状態分離: _base_offset_ratio* を参照
        effective_offset_ratio = self._base_offset_ratio
        if side == "buy" and self._base_offset_ratio_buy is not None:
            effective_offset_ratio = self._base_offset_ratio_buy
        elif side == "sell" and self._base_offset_ratio_sell is not None:
            effective_offset_ratio = self._base_offset_ratio_sell

        # 162# Inventory Skewing: 在庫偏重に応じた非対称 offset 補正
        # 228# C2: time-decay 適用 — 古い fill 履歴の影響を減衰
        # buy 偏重(imbalance>0) -> buy offset拡大(抑制), sell offset縮小(促進)
        # sell 偏重(imbalance<0) -> sell offset拡大(抑制), buy offset縮小(促進)
        _decayed_imb = self._decayed_imbalance(now)
        # 249# Regime-aware inventory skewing: trending 時は inv_skew を無効化
        # — トレンド方向のポジション蓄積を在庫中立ロジックが阻害しないため
        _inv_skew_regime_blocked = False
        if cfg.inv_skew_regime_gate_enabled and self._regime_detector is not None:
            _r = self._regime_detector.current_regime
            if _r.is_trending:
                _inv_skew_regime_blocked = True
                logger.debug(
                    f"[249# inv_skew_gate] regime={_r.value} — "
                    f"inv_skew DISABLED (directional alpha preservation)"
                )
        if (
            cfg.inventory_skewing_enabled
            and abs(_decayed_imb) > cfg.inventory_skewing_neutral_band
            and not _inv_skew_regime_blocked  # 249# regime gate
        ):
            _imb = _decayed_imb
            _sign = 1.0 if side == "buy" else -1.0
            _factor = _imb * _sign * cfg.inventory_skewing_max_factor
            _prev = effective_offset_ratio
            effective_offset_ratio, _applied_mult = self._scale_offset_ratio(
                effective_offset_ratio,
                1.0 + _factor,
                min_ratio=cfg.min_offset_ratio,
            )
            self._last_inv_skew_factor = _factor
            logger.info(
                f"[inv_skew] {side} imbalance={_imb:+.3f} "
                f"factor={_factor:+.4f} mult={_applied_mult:.4f} "
                f"offset {_prev:.4f}->{effective_offset_ratio:.4f}"
            )
        else:
            self._last_inv_skew_factor = 0.0

        # 088# sell 専用ハードガード: offset floor (173# 動的フロア対応)
        if side == "sell":
            _dyn_floor = self._effective_sell_offset_floor()
            if _dyn_floor > 0:
                effective_offset_ratio = max(effective_offset_ratio, _dyn_floor)

        # 257# ステージ: _apply_as_reservation_shift()
        # Avellaneda-Stoikov 在庫×ボラ連動 offset (inv_skew + σ² 補完)
        effective_offset_ratio = self._apply_as_reservation_shift(
            side, spread, mid_price, effective_offset_ratio,
        )

        # 163# ステージ抽出: _apply_regime_boosts()
        effective_offset_ratio = self._apply_regime_boosts(
            side, effective_offset_ratio,
        )

        # 163# ステージ抽出: _apply_spread_adaptive()
        effective_offset_ratio = self._apply_spread_adaptive(
            side, spread, mid_price, effective_offset_ratio,
        )

        # 266# ステージ: _apply_kyle_lambda()
        # Kyle (1985) 価格インパクト係数 → offset 安全マージン
        effective_offset_ratio = self._apply_kyle_lambda(
            side, spread, mid_price, effective_offset_ratio,
        )

        # 266# ステージ: _apply_amihud_illiq()
        # Amihud (2002) 非流動性比率 → 低流動性時の offset 拡大
        effective_offset_ratio = self._apply_amihud_illiq(
            side, spread, mid_price, effective_offset_ratio,
        )

        # 163# ステージ抽出: _apply_volatility_guard()
        effective_offset_ratio = self._apply_volatility_guard(
            side, mid_trend_bps, effective_offset_ratio,
        )

        # 163# ステージ抽出: _apply_imbalance_risk()
        effective_offset_ratio = self._apply_imbalance_risk(
            side, imb, effective_offset_ratio,
        )

        # 286# ステージ: _apply_buy_as_guard()
        # 283# P1-6 / 284# P1: Buy-side AS 防御 — microprice 急落時の offset 拡大
        effective_offset_ratio = self._apply_buy_as_guard(
            side, mid_trend_bps, effective_offset_ratio,
        )

        # 260# P2-2: loss_boost / FFD boost をパイプラインステージとして抽出
        effective_offset_ratio = self._apply_loss_boost(
            side, now, effective_offset_ratio,
        )

        offset = max(cfg.min_offset_jpy, spread * effective_offset_ratio)

        # 260# P2-2: FastFillDefense boost をパイプラインステージとして抽出
        effective_offset_ratio, offset = self._apply_ffd_boost(
            side, spread, effective_offset_ratio, offset,
        )
        return self._finalize_price_with_spread_guard(
            side=side,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            offset=offset,
            effective_offset_ratio=effective_offset_ratio,
        )
