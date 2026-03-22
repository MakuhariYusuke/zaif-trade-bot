"""322# maker_microstructure — 市場ミクロ構造 offset Mixin.

maker_price.py MakerPriceCalculator からの God Object 分割 (321# §4):
- _get_depth (板 depth ヘルパー)
- _estimate_sigma (Parkinson/Roll σ 推定)
- _dynamic_tau (GLFT τ 動的化)
- _apply_as_reservation_shift (Avellaneda-Stoikov 予約価格 offset)
- _apply_kyle_lambda (Kyle 1985 価格インパクト)
- _apply_amihud_illiq (Amihud 2002 非流動性)

再利用可能性:
  σ 推定 (Parkinson): backtest の volatility 計算 (4 実装並存) と共有可能。
  Kyle λ / Amihud ILLIQ: オフライン分析ツールで板流動性評価に単独使用可。
  AS δ*: 理論的最適 offset 下限の算出 — backtest/最適化で参照用。
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.v460.lib.fill_config import FillTestConfig
    from scripts.v460.lib.fill_probability_model import FillProbabilityModel
    from ztb.trading.signal.regime.regime_detector import RegimeDetectorLike

logger = logging.getLogger(__name__)


class MicrostructureMixin:
    """市場ミクロ構造ベースの offset 調整 Mixin.

    MakerPriceCalculator が継承して使用する。
    依存する属性:
      _config: FillTestConfig
      _regime_detector: RegimeDetectorLike | None
      _last_bid_depth, _last_ask_depth: float
      _mid_high, _mid_low, _mid_hl_reset_time: float
      _last_sigma, _last_amihud_illiq: float
      _decayed_imbalance(now): method
      _scale_offset_ratio: staticmethod
    """

    # --- type stubs for mixin dependencies ---
    _config: FillTestConfig
    _regime_detector: RegimeDetectorLike | None
    _fill_prob_model: FillProbabilityModel | None
    _last_bid_depth: float
    _last_ask_depth: float
    _mid_high: float
    _mid_low: float
    _mid_hl_reset_time: float
    _last_sigma: float
    _last_amihud_illiq: float
    _last_as_delta_star_ratio: float  # 543# A-S δ* 参照スプレッド

    def _decayed_imbalance(self, now: float) -> float: ...
    @staticmethod
    def _scale_offset_ratio(
        effective_offset_ratio: float,
        multiplier: float,
        *,
        min_ratio: float | None = None,
        max_ratio: float | None = None,
    ) -> tuple[float, float]: ...

    def _effective_max_ratio(self, side: str) -> float: ...  # 405# Protocol stub

    # ------------------------------------------------------------------
    # 共有ヘルパー: σ推定 + τ動的化 (GLFT/AS δ*/Kyle で再利用)
    # ------------------------------------------------------------------

    def _get_depth(self, side: str) -> float:
        """267# DRY: side に応じた板 depth volume を返す.

        Kyle λ / Amihud ILLIQ 等で共通使用するヘルパー。
        """
        return self._last_bid_depth if side == "buy" else self._last_ask_depth

    def _estimate_sigma(self, spread: float, mid_price: float) -> tuple[float, float]:
        """266# σ 推定: Roll (1984) micro-vol proxy × RegimeDetector vol_ratio.

        305# Parkinson 拡張: sigma_parkinson_enabled=True 時は
        Parkinson (1980) High-Low Volatility Estimator を使用。
        σ_P = ln(H/L) / (2·√(ln2)) — rolling window 内の max/min mid から推定。
        Roll proxy (spread/(2·mid)) は薄い板で極めてノイジーであるのに対し、
        Parkinson は実際の価格変動範囲に基づくため安定性が高い。

        Returns:
            (sigma, vol_ratio) — σ 推定値と vol_ratio。
            _apply_as_reservation_shift (AS + δ*) で直接使用。
            _apply_kyle_lambda, _apply_amihud_illiq は depth ベースで独自推定。
        """
        vol_ratio = 1.0
        cfg = self._config
        if self._regime_detector is not None:
            vol_ratio = max(
                self._regime_detector.last_volatility_ratio,
                cfg.vol_ratio_floor,
            )

        # 305# Parkinson σ: rolling high/low から Parkinson estimator で σ を推定
        if cfg.sigma_parkinson_enabled and mid_price > 0:
            now = time.time()
            window = cfg.sigma_parkinson_window_sec
            # window 経過でリセット
            if now - self._mid_hl_reset_time > window:
                self._mid_high = mid_price
                self._mid_low = mid_price
                self._mid_hl_reset_time = now
            else:
                if mid_price > self._mid_high:
                    self._mid_high = mid_price
                if mid_price < self._mid_low:
                    self._mid_low = mid_price

            if self._mid_high > 0 and self._mid_low > 0 and self._mid_high > self._mid_low:
                # Parkinson (1980): σ_P = ln(H/L) / (2·√(ln2))
                log_hl = math.log(self._mid_high / self._mid_low)
                sigma = log_hl / (2.0 * math.sqrt(math.log(2.0)))
            else:
                # high == low (動きなし) → Roll proxy にフォールバック
                sigma = spread / (2.0 * mid_price) if mid_price > 0 else 0.0
        else:
            sigma = spread / (2.0 * mid_price) if mid_price > 0 else 0.0

        sigma *= vol_ratio
        # 330# σ floor: σ=0 は AS δ* / Kyle λ / Amihud を完全無効化するため、
        # 最小フロアを設ける。spread=0 (tight book) はむしろ AS 上最も脆弱。
        sigma = max(sigma, cfg.sigma_floor)
        # 306# L1: cache for dynamic cycle interval
        self._last_sigma = sigma
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

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

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
            min(self._effective_max_ratio(side), effective_offset_ratio + shift),
        )

        # 266#/267# AS δ*: 理論的最適スプレッド幅下限
        # δ* = γσ²τ + (2/γ)ln(1 + γ/k) (Avellaneda-Stoikov 2008 §4)
        # 注意: AS 論文の σ は絶対価格 (JPY/√s) ベース。_estimate_sigma は
        # リターンベース (無次元) なので σ_abs = σ_return × mid_price に変換。
        # δ* (JPY) → offset_ratio = δ* / spread で変換。
        if cfg.as_delta_star_enabled and gamma > 0:
            # 366# M4: GLFT Fill Probability — 動的 k (有効時)
            k = cfg.as_delta_star_fill_rate_k  # フォールバック
            # 373# F6: type:ignore[union-attr] を narrowing で排除
            _fpm = self._fill_prob_model
            if (
                _fpm is not None
                and cfg.glft_dynamic_k_enabled
                and _fpm.k > 0
            ):
                k = _fpm.k
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
                    # 543# A-S 参照スプレッド: 計測値をキャッシュ
                    self._last_as_delta_star_ratio = delta_star_ratio
                    if effective_offset_ratio < delta_star_ratio:
                        logger.debug(
                            f"[as_delta_star] 266# {side} δ*={delta_star_ratio:.4f} "
                            f"> offset={effective_offset_ratio:.4f} → floor applied"
                        )
                        effective_offset_ratio = min(delta_star_ratio, self._effective_max_ratio(side))

        if effective_offset_ratio != prev:
            logger.info(
                f"[as_reservation] 266# {side} q={q:+.3f} γ={gamma:.3f} "
                f"σ²={sigma_sq:.2e} vol_ratio={vol_ratio:.3f} "
                f"τ={tau:.0f}s{'(dyn)' if cfg.as_tau_dynamic_enabled else ''} → "
                f"offset {prev:.4f} → {effective_offset_ratio:.4f} "
                f"(shift={shift:+.2e})"
            )

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
            self._effective_max_ratio(side),
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
            effective_offset_ratio, mult, max_ratio=self._effective_max_ratio(side),
        )

        if effective_offset_ratio != prev:
            logger.debug(
                f"[amihud_illiq] 266# {side} ILLIQ={illiq:.4e} "
                f"ratio={illiq_ratio:.2f} mult={_applied:.3f} → "
                f"offset {prev:.4f}→{effective_offset_ratio:.4f}"
            )

        return effective_offset_ratio
