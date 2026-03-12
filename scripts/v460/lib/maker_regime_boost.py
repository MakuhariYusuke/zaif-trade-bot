"""322# maker_regime_boost — Regime別 offset boost Mixin.

maker_price.py MakerPriceCalculator からの God Object 分割 (321# §4):
- _apply_regime_boosts (dispatcher)
- _regime_boost_trending
- _regime_boost_high_vol
- _regime_boost_ranging
- _regime_boost_low_vol
- _regime_boost_unknown_buy
- _resolve_trending_boost (staticmethod)

再利用可能性:
  backtest の offset シミュレーション、regime 別パフォーマンス分析で
  regime boost 計算を単体で呼び出し可能。regime 分類器が 4+ 並存する
  現状 (321# §5) で、共通 interface への第一歩。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from scripts.v460.lib.regime_detector import FillTestRegime

if TYPE_CHECKING:
    from scripts.v460.lib.fill_config import FillTestConfig
    from scripts.v460.lib.regime_detector import RegimeDetectorLike

logger = logging.getLogger(__name__)


class RegimeBoostMixin:
    """Regime別 offset boost 計算の Mixin.

    MakerPriceCalculator が継承して使用する。
    依存する属性:
      _config: FillTestConfig
      _regime_detector: RegimeDetectorLike | None
      _last_imbalance: float
      _scale_offset_ratio: staticmethod
    """

    # --- type stubs for mixin dependencies ---
    _config: FillTestConfig
    _regime_detector: RegimeDetectorLike | None
    _last_imbalance: float

    @staticmethod
    def _scale_offset_ratio(
        effective_offset_ratio: float,
        multiplier: float,
        *,
        min_ratio: float | None = None,
        max_ratio: float | None = None,
    ) -> tuple[float, float]: ...

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _apply_regime_boosts(
        self, side: str, effective_offset_ratio: float,
    ) -> float:
        """052# 143# 130# 397# regime 別 offset 補正.

        260# P2-3: 6 独立ステージに分割。
        - trending: buy/sell 非対称 boost
        - high_vol: offset 拡大 (AS リスク上昇)
        - ranging: offset 縮小 (安定市場で利幅確保)
        - low_vol: offset 拡大 (過剰アグレッシブ抑制)
        - unknown: buy guard offset boost
        - mid_confidence: confidence [0.7,0.9) paradox guard (397#)
        """
        effective_offset_ratio = self._regime_boost_trending(side, effective_offset_ratio)
        effective_offset_ratio = self._regime_boost_high_vol(side, effective_offset_ratio)
        effective_offset_ratio = self._regime_boost_ranging(side, effective_offset_ratio)
        effective_offset_ratio = self._regime_boost_low_vol(side, effective_offset_ratio)
        effective_offset_ratio = self._regime_boost_unknown_buy(side, effective_offset_ratio)
        effective_offset_ratio = self._regime_boost_mid_confidence(side, effective_offset_ratio)
        return effective_offset_ratio

    # ------------------------------------------------------------------
    # Sub-stages
    # ------------------------------------------------------------------

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

    def _regime_boost_mid_confidence(
        self, side: str, effective_offset_ratio: float,
    ) -> float:
        """397# mid-confidence paradox guard.

        395# SHA-fenced 実証: confidence [0.7,0.9) は全 SHA で
        paradoxical underperformance (−0.734 bps, WR=46%)。
        レジーム判定が medium-confident だが実際は不正確な帯域で、
        offset を拡大しリスクを低減する。
        """
        cfg = self._config
        if (
            cfg.regime_mid_confidence_offset_boost > 1.0
            and self._regime_detector is not None
        ):
            conf = self._regime_detector.current_confidence
            if cfg.regime_mid_confidence_lo <= conf < cfg.regime_mid_confidence_hi:
                pre_offset = effective_offset_ratio
                effective_offset_ratio, _applied_mult = self._scale_offset_ratio(
                    effective_offset_ratio,
                    cfg.regime_mid_confidence_offset_boost,
                    max_ratio=cfg.max_offset_ratio,
                )
                logger.info(
                    f"[397# mid_conf_guard] {side} confidence={conf:.3f} "
                    f"in [{cfg.regime_mid_confidence_lo},{cfg.regime_mid_confidence_hi}) "
                    f"→ offset boosted: {pre_offset:.4f}→{effective_offset_ratio:.4f} "
                    f"(boost={_applied_mult:.2f})"
                )
        return effective_offset_ratio

    # ------------------------------------------------------------------
    # Static helper
    # ------------------------------------------------------------------

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
