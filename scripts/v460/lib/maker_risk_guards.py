"""322# maker_risk_guards — リスクガード offset Mixin.

maker_price.py MakerPriceCalculator からの God Object 分割 (321# §4):
- _apply_volatility_guard (VG: velocity + VPIN 急変検知)
- _apply_imbalance_risk (板不均衡 AS リスク)
- _apply_buy_as_guard (buy-side 急落防御)
- _apply_sell_hour_boost (sell 時間帯別 offset 拡大)

再利用可能性:
  VG の VPIN 計算パターンは ztb/features/microstructure.py の
  compute_vpin() と同型で、統合・共有可能。
  imbalance_risk は backtest の AS 評価シミュレーションで参照可能。
  sell_hour_boost の時間帯判定は skip_gate の hour_offsets と併せて
  time-of-day 分析モジュールとして共通化の余地あり。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.v460.lib.fill_config import FillTestConfig

logger = logging.getLogger(__name__)


class RiskGuardsMixin:
    """リスクガード offset 調整の Mixin.

    MakerPriceCalculator が継承して使用する。
    依存する属性:
      _config: FillTestConfig
      _last_vpin: float | None
      _last_inv_skew_factor: float
      _last_vg_triggered, _last_vg_velocity_bps, _last_vg_vpin, _last_vg_boost_factor
      _scale_offset_ratio: staticmethod
    """

    # --- type stubs for mixin dependencies ---
    _config: FillTestConfig
    _last_vpin: float | None
    _last_inv_skew_factor: float
    _last_vg_triggered: bool
    _last_vg_velocity_bps: float | None
    _last_vg_vpin: float | None
    _last_vg_boost_factor: float | None

    @staticmethod
    def _scale_offset_ratio(
        effective_offset_ratio: float,
        multiplier: float,
        *,
        min_ratio: float | None = None,
        max_ratio: float | None = None,
    ) -> tuple[float, float]: ...

    # ------------------------------------------------------------------
    # Risk guard pipeline stages
    # ------------------------------------------------------------------

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

    def _apply_sell_hour_boost(
        self,
        side: str,
        effective_offset_ratio: float,
    ) -> float:
        """310# A: Sell AS Time-of-Day Offset Boost (307# F3, 306# H5).

        Ho-Stoll (1981): 情報非対称性は時間帯により変動する。
        306# deep dive 実証データ:
          UTC 08h AS=63%, 13h AS=42%, 14h AS=43%, 16h AS=61% (sell 側)
        これらの高 AS 時間帯で sell offset を拡大し逆選択コストを低減。
        skip_gate_hour_offsets (ML 閾値調整) や hard_skip_utc_hours (全停止)
        とは独立した第三の防御レイヤー (offset 拡大) として機能する。
        """
        cfg = self._config
        if side != "sell" or not cfg.sell_hour_offset_boost:
            return effective_offset_ratio
        utc_h = datetime.now(timezone.utc).hour
        mult = cfg.sell_hour_offset_boost.get(utc_h)
        if mult is not None and mult > 1.0:
            _prev = effective_offset_ratio
            effective_offset_ratio, _applied = self._scale_offset_ratio(
                effective_offset_ratio,
                mult,
                min_ratio=cfg.min_offset_ratio,
            )
            logger.info(
                f"[310# A] sell_hour_boost: UTC {utc_h}h × {mult:.2f}, "
                f"offset {_prev:.4f} → {effective_offset_ratio:.4f}"
            )
        return effective_offset_ratio
