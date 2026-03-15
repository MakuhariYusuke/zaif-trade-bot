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
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol

from scripts.v460.lib import cancel_reasons as CR
from scripts.v460.lib.fast_fill_defense import FastFillDefense
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.maker_microstructure import MicrostructureMixin
from scripts.v460.lib.maker_regime_boost import RegimeBoostMixin
from scripts.v460.lib.maker_risk_guards import RiskGuardsMixin
from scripts.v460.lib.ob_utils import OrderBookSnapshot
from scripts.v460.lib.regime_detector import RegimeDetectorLike
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


from scripts.v460.lib.constants import BPS_FACTOR as _BPS_FACTOR

if TYPE_CHECKING:
    from scripts.v460.lib.cross_venue_lead_lag import CrossVenueLeadLagHint
    from scripts.v460.lib.fill_probability_model import FillProbabilityModel


class MakerPriceCalculator(RiskGuardsMixin, MicrostructureMixin, RegimeBoostMixin):
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
    ║              → _apply_cross_venue_lead_lag_guard()         ║
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
        "_cross_venue_lead_lag_hint",
        "_cross_venue_lead_lag_vetoed",
        "_cross_venue_lead_lag_veto_reason",
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
        "_mid_high",                 # 305# Parkinson σ: rolling window 内 max mid
        "_mid_low",                  # 305# Parkinson σ: rolling window 内 min mid
        "_mid_hl_reset_time",        # 305# Parkinson σ: high/low リセット時刻
        "_last_sigma",               # 306# L1: 最新 σ キャッシュ (dynamic cycle interval)
        "_last_offset_stages",       # 306# E1: offset stage recording (JSON)
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
        self._cross_venue_lead_lag_hint: CrossVenueLeadLagHint | None = None
        self._cross_venue_lead_lag_vetoed: bool = False
        self._cross_venue_lead_lag_veto_reason: str | None = None
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
        # 305# Parkinson σ: rolling high/low tracking for HF volume estimator
        self._mid_high: float = 0.0
        self._mid_low: float = float("inf")
        self._mid_hl_reset_time: float = 0.0
        # 306# L1: 最新 σ キャッシュ (dynamic cycle interval 用)
        self._last_sigma: float = 0.0
        # 306# E1: offset stage recording
        self._last_offset_stages: str | None = None

    def get_fallback_price(self) -> tuple[float | None, float | None]:
        """156# §16: OB エラー時のフォールバック価格と記録時刻を返す.

        Returns:
            (prev_mid_price, prev_mid_time) — 未設定時は (None, None).
        """
        return self._prev_mid_price, self._prev_mid_time

    def set_fill_prob_model(self, model: FillProbabilityModel | None) -> None:
        """366# M4: GLFT fill probability model を注入する."""
        self._fill_prob_model = model

    def set_cross_venue_lead_lag_hint(
        self,
        hint: CrossVenueLeadLagHint | None,
    ) -> None:
        """439# Cross-venue lead-lag hint を注入する."""
        self._cross_venue_lead_lag_hint = hint
        self._cross_venue_lead_lag_vetoed = False
        self._cross_venue_lead_lag_veto_reason = None

    @property
    def last_sigma(self) -> float:
        """306# L1: 最新の σ 推定値 (dynamic cycle interval 用)."""
        return self._last_sigma

    @property
    def last_offset_stages(self) -> str | None:
        """306# E1: 最新の offset pipeline stage 記録 (JSON)."""
        return self._last_offset_stages

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

    # 366# M1: Gatheral (2018) multi-level microprice の指数減衰重み
    # w_k = exp(-0.5 * k), k=0..4  — α=0.5 で深い板ほど寄与減衰
    _MICRO_WEIGHTS: Final = (1.0, 0.6065, 0.3679, 0.2231, 0.1353)

    def compute_microprice_bias_bps(self) -> float:
        """306# L2 → 366# M1: Multi-level Microprice vs Mid の偏向を bps で返す.

        366# M1 拡張: Gatheral (2018) multi-level microprice.
        L1 のみの単純 microprice から L1-L5 の指数加重 microprice に拡張。
        OB snapshot (depth=5) は compute_imbalance() でキャッシュ済みなので
        追加 API 呼出しは不要。

        μ = Σ w_k · (P_k^bid · Q_k^ask + P_k^ask · Q_k^bid)
            / Σ w_k · (Q_k^ask + Q_k^bid)
        bias_bps = (μ - mid) / mid × 10_000

        正 → 買い圧力 (bid 厚い → microprice > mid)
        負 → 売り圧力 (ask 厚い → microprice < mid)
        309# 注: 旧306#では「sell有利」としていたが、308#レビューにより
        maker理論的には buy pressure 時の sell は AS seeker と判定。
        Side 選択では safety モード (同方向) に修正済。

        OB キャッシュ (_last_ob_snapshot) を使用 — 追加 API 呼出しなし。
        """
        ob = self._last_ob_snapshot
        if ob is None or not ob.bids or not ob.asks:
            return 0.0
        depth = min(len(ob.bids), len(ob.asks), self._config.microprice_depth)
        if depth == 0:
            return 0.0
        # 366# M1: multi-level weighted microprice
        num = 0.0
        den = 0.0
        weights = self._MICRO_WEIGHTS
        min_qty = self._config.microprice_min_qty
        for k in range(depth):
            pb, qb = ob.bids[k]
            pa, qa = ob.asks[k]
            # 366# §8.1-1: 薄い板レベルをスキップ (qty < min_qty)
            if qb < min_qty and qa < min_qty:
                continue
            w = weights[k] if k < len(weights) else 0.0
            num += w * (pb * qa + pa * qb)
            den += w * (qa + qb)
        if den <= 0:
            return 0.0
        microprice = num / den
        mid = (ob.bids[0][0] + ob.asks[0][0]) / 2.0
        if mid <= 0:
            return 0.0
        return (microprice - mid) / mid * 10_000.0

    def estimate_queue_depth(self, side: str, order_price: float) -> float:
        """306# O1: 推定キュー深度 (自注文より有利な価格帯の volume 合計).

        buy: order_price 以上の bid volume (= 先に約定される queue)
        sell: order_price 以下の ask volume
        OB キャッシュを使用 — 追加 API 呼出しなし。
        """
        ob = self._last_ob_snapshot
        if ob is None:
            return 0.0
        depth_ahead = 0.0
        if side == "buy":
            for price, qty in ob.bids:
                if price >= order_price:
                    depth_ahead += qty
                else:
                    break  # sorted desc
        else:
            for price, qty in ob.asks:
                if price <= order_price:
                    depth_ahead += qty
                else:
                    break  # sorted asc
        return depth_ahead

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

    def _effective_max_ratio(self, side: str) -> float:
        """405#: Side-aware intermediate max ratio.

        sell_floor (0.30) == max_offset_ratio (0.30) のデッドロック解消。
        sell 側では offset_ceiling_ratio_sell を intermediate cap に使い、
        中間ブーストが 0.30-0.50 の範囲で有効に機能するようにする。
        buy 側は既存動作を維持 (max_offset_ratio = 0.30)。
        最終段の offset ceiling で side 別の最終クランプは維持。

        cf. 403# §3, 404# Action 1
        """
        cfg = self._config
        base = cfg.max_offset_ratio
        if side == "sell" and cfg.offset_ceiling_ratio_sell is not None:
            return max(base, cfg.offset_ceiling_ratio_sell)
        if side == "buy" and cfg.offset_ceiling_ratio_buy is not None:
            return max(base, cfg.offset_ceiling_ratio_buy)
        return base

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
        # 330# B1: ゼロ除算ガード — effective_offset_ratio が極小値で
        # multiplier 適用後の updated が 0 になるケースを安全側に倒す
        applied = updated / effective_offset_ratio if effective_offset_ratio != 0 else 1.0
        return updated, applied

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
    # 266# 共有ヘルパー: _finalize_price_with_spread_guard (compute() 内で使用)
    # ------------------------------------------------------------------

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
                    max_ratio=self._effective_max_ratio(side),
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
            max_ratio=self._effective_max_ratio(side),
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
                max_ratio=self._effective_max_ratio(side),
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

        # 305# S2: OB キャッシュ再利用 — calculate_imbalance() の depth=5 OB を活用。
        # 二重 API 呼出し (100-200ms/cycle) を排除。
        # キャッシュ未取得時 (imbalance_enabled=False 等) のみ fresh fetch。
        if self._last_ob_snapshot is not None and self._last_ob_snapshot.bids and self._last_ob_snapshot.asks:
            ob = self._last_ob_snapshot
        else:
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

        # === 303# C / 318# F5-1: none/unknown レジーム Passive MM バイパス ===
        # regime 未確定時に 13段パイプラインをスキップし固定 offset で配置。
        # AS 43% (301# F1) の根本対策: 情報がない状態での積極約定を抑止。
        # 318# F5-1 修正: "none" だけでなく "unknown" (warmup/低信頼度) も対象。
        # 旧実装は "none" のみチェックしていたが、FillTestRegime enum に "none" は
        # 存在せず、detector 存在時は常に "unknown" を返すため事実上死んでいた。
        if cfg.none_regime_passive_mm_enabled:
            _current_regime = (
                self._regime_detector.current_regime.value
                if self._regime_detector is not None
                else "none"
            )
            if _current_regime in ("none", "unknown"):
                _fixed_ratio = cfg.none_regime_fixed_offset_bps / 10000.0
                _fixed_offset = max(cfg.min_offset_jpy, mid_price * _fixed_ratio)
                logger.info(
                    f"[303# C] Passive MM bypass: regime={_current_regime}, "
                    f"fixed_offset={_fixed_offset:.0f} JPY "
                    f"({cfg.none_regime_fixed_offset_bps:.1f} bps)"
                )
                return self._finalize_price_with_spread_guard(
                    side=side,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    spread=spread,
                    offset=_fixed_offset,
                    effective_offset_ratio=_fixed_ratio,
                )

        # === offset 決定ロジック ===
        # 096# 状態分離: _base_offset_ratio* を参照
        effective_offset_ratio = self._base_offset_ratio
        if side == "buy" and self._base_offset_ratio_buy is not None:
            effective_offset_ratio = self._base_offset_ratio_buy
        elif side == "sell" and self._base_offset_ratio_sell is not None:
            effective_offset_ratio = self._base_offset_ratio_sell

        # 306# E1: offset stage recording — 各ステージの寄与を追跡
        _stage_tracking = cfg.offset_stage_recording_enabled
        _stages: dict[str, float] = {}
        if _stage_tracking:
            _stages["base"] = effective_offset_ratio

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
        if _stage_tracking:
            _stages["as_shift"] = effective_offset_ratio

        # 163# ステージ抽出: _apply_regime_boosts()
        effective_offset_ratio = self._apply_regime_boosts(
            side, effective_offset_ratio,
        )
        if _stage_tracking:
            _stages["regime"] = effective_offset_ratio

        # 163# ステージ抽出: _apply_spread_adaptive()
        effective_offset_ratio = self._apply_spread_adaptive(
            side, spread, mid_price, effective_offset_ratio,
        )
        if _stage_tracking:
            _stages["spread_adapt"] = effective_offset_ratio

        # 266# ステージ: _apply_kyle_lambda()
        # Kyle (1985) 価格インパクト係数 → offset 安全マージン
        effective_offset_ratio = self._apply_kyle_lambda(
            side, spread, mid_price, effective_offset_ratio,
        )
        if _stage_tracking:
            _stages["kyle"] = effective_offset_ratio

        # 266# ステージ: _apply_amihud_illiq()
        # Amihud (2002) 非流動性比率 → 低流動性時の offset 拡大
        effective_offset_ratio = self._apply_amihud_illiq(
            side, spread, mid_price, effective_offset_ratio,
        )
        if _stage_tracking:
            _stages["amihud"] = effective_offset_ratio

        # 163# ステージ抽出: _apply_volatility_guard()
        effective_offset_ratio = self._apply_volatility_guard(
            side, mid_trend_bps, effective_offset_ratio,
        )
        if _stage_tracking:
            _stages["vol_guard"] = effective_offset_ratio

        # 439# cross-venue lead-lag guard: adverse-side retreat / veto
        effective_offset_ratio = self._apply_cross_venue_lead_lag_guard(
            side, effective_offset_ratio,
        )
        if _stage_tracking:
            _stages["cross_venue"] = effective_offset_ratio
        if self._cross_venue_lead_lag_vetoed:
            raise InfeasibleQuoteError(
                reason=CR.CROSS_VENUE_LEAD_LAG_VETO,
                msg=(
                    self._cross_venue_lead_lag_veto_reason
                    or "cross-venue lead-lag veto"
                ),
            )

        # 163# ステージ抽出: _apply_imbalance_risk()
        effective_offset_ratio = self._apply_imbalance_risk(
            side, imb, effective_offset_ratio,
        )
        if _stage_tracking:
            _stages["imb_risk"] = effective_offset_ratio

        # 286# ステージ: _apply_buy_as_guard()
        # 283# P1-6 / 284# P1: Buy-side AS 防御 — microprice 急落時の offset 拡大
        effective_offset_ratio = self._apply_buy_as_guard(
            side, mid_trend_bps, effective_offset_ratio,
        )
        if _stage_tracking:
            _stages["buy_as_guard"] = effective_offset_ratio

        # 310# A: Sell AS Time-of-Day Offset Boost (307# F3, 306# H5)
        # Ho-Stoll (1981): 時間帯別の情報非対称性変動を offset に反映
        effective_offset_ratio = self._apply_sell_hour_boost(
            side, effective_offset_ratio,
        )
        if _stage_tracking:
            _stages["sell_hour"] = effective_offset_ratio

        # 260# P2-2: loss_boost / FFD boost をパイプラインステージとして抽出
        effective_offset_ratio = self._apply_loss_boost(
            side, now, effective_offset_ratio,
        )
        if _stage_tracking:
            _stages["loss_boost"] = effective_offset_ratio

        offset = max(cfg.min_offset_jpy, spread * effective_offset_ratio)

        # 260# P2-2: FastFillDefense boost をパイプラインステージとして抽出
        effective_offset_ratio, offset = self._apply_ffd_boost(
            side, spread, effective_offset_ratio, offset,
        )
        if _stage_tracking:
            _stages["ffd"] = effective_offset_ratio
            _stages["final"] = effective_offset_ratio

        # 306# E1: offset ceiling — 300# T1-3 指摘の上限制御
        # 320# C-1: サイド別 ceiling — sell floor(0.30) > ceiling(0.15) 矛盾解消
        # 421# DRY: resolve_offset_ceiling ヘルパーに統一
        _ceil = cfg.resolve_offset_ceiling(side)
        if _ceil > 0 and effective_offset_ratio > _ceil:
            logger.info(
                f"[306# ceiling] offset {effective_offset_ratio:.4f} "
                f"> ceiling {_ceil:.4f} — clamped"
            )
            effective_offset_ratio = _ceil
            offset = max(cfg.min_offset_jpy, spread * effective_offset_ratio)
            if _stage_tracking:
                _stages["ceiling"] = effective_offset_ratio

        # 306# E1: cache last offset stages for FillRecord
        if _stage_tracking:
            import json as _json
            self._last_offset_stages = _json.dumps(_stages)
        else:
            self._last_offset_stages = None

        return self._finalize_price_with_spread_guard(
            side=side,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            offset=offset,
            effective_offset_ratio=effective_offset_ratio,
        )
