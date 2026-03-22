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
import time as _time
from datetime import datetime as _datetime
from datetime import timezone as _timezone
from pathlib import Path as _Path
from typing import TYPE_CHECKING

from scripts.v460.lib.hour_rules import current_utc_hour, resolve_optional_hour_float
from ztb.io.jsonl import append_jsonl as _append_jsonl

if TYPE_CHECKING:
    from scripts.v460.lib.cross_venue_lead_lag import CrossVenueLeadLagHint
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
    _last_vg_reason: str | None
    _cross_venue_lead_lag_hint: CrossVenueLeadLagHint | None
    _cross_venue_lead_lag_vetoed: bool
    _cross_venue_lead_lag_veto_reason: str | None
    # 448# F2修正: cap_hit (no-op) 可視化
    _cross_venue_lead_lag_pre_offset: float | None
    _cross_venue_lead_lag_post_offset: float | None
    _cross_venue_lead_lag_cap_hit: bool
    # 533# veto deadlock 防止
    _consecutive_veto_count: int
    _veto_btc_balance: float | None

    @staticmethod
    def _scale_offset_ratio(  # type: ignore[empty-body]  # Protocol stub
        effective_offset_ratio: float,
        multiplier: float,
        *,
        min_ratio: float | None = None,
        max_ratio: float | None = None,
    ) -> tuple[float, float]:
        ...

    def _effective_max_ratio(self, side: str) -> float: ...  # 405# Protocol stub

    @staticmethod
    def _current_utc_hour() -> int:
        """現在の UTC hour を返す.

        time-of-day 依存ロジックを 1 箇所に閉じる。
        """
        return current_utc_hour()

    def _resolve_sell_hour_boost_mult(self) -> float | None:
        """sell_hour_offset_boost から現在 UTC hour の倍率を解決する."""
        return resolve_optional_hour_float(
            self._config.sell_hour_offset_boost,
            self._current_utc_hour(),
        )

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

            # 353# VPIN 非対称 buy boost (351# 盲点1: Ranging ≠ 対称)
            # buy 側は informed trading の逆選択リスクが構造的に高い
            # (164# SHAP: buy vpin_60s=0.548 vs sell=0.316)
            if (
                side == "buy"
                and vpin_boost > 1.0
                and cfg.vg_vpin_buy_extra_mult > 1.0
            ):
                vpin_boost = 1.0 + (vpin_boost - 1.0) * cfg.vg_vpin_buy_extra_mult
                vg_reason += f"(buy_extra={cfg.vg_vpin_buy_extra_mult:.1f}x)"

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
                    max_ratio=self._effective_max_ratio(side),
                )
                logger.info(
                    f"[volatility_guard] 107# {side} offset boosted: "
                    f"{pre_offset:.4f}→{effective_offset_ratio:.4f} "
                    f"({vg_reason})"
                )
                # 372# E12: 構造化イベントログ (JSONL)
                _emit_vg_event(
                    side=side,
                    pre_offset=pre_offset,
                    post_offset=effective_offset_ratio,
                    reason=vg_reason,
                    velocity_bps=_vg_velocity,
                    vpin=_vg_vpin,
                    boost_factor=_vg_boost,
                )
            # 120# P2-1: VG 発動状態を追跡
            self._last_vg_triggered = vg_triggered
            # 158# P2-6: VG 詳細ログ (ヒンドサイト分析用)
            self._last_vg_velocity_bps = _vg_velocity
            self._last_vg_vpin = _vg_vpin
            self._last_vg_boost_factor = _vg_boost if vg_triggered else None
            # 510# VG boost 理由の粒度向上
            if vg_triggered:
                _has_vel = velocity_boost > 1.0
                _has_vpin = vpin_boost > 1.0
                if _has_vel and _has_vpin:
                    self._last_vg_reason = "velocity+vpin"
                elif _has_vel:
                    self._last_vg_reason = "velocity"
                else:
                    self._last_vg_reason = "vpin"
            else:
                self._last_vg_reason = None
        else:
            self._last_vg_triggered = False
            self._last_vg_velocity_bps = None
            self._last_vg_vpin = None
            self._last_vg_boost_factor = None
            self._last_vg_reason = None

        return effective_offset_ratio

    def _apply_cross_venue_lead_lag_guard(
        self,
        side: str,
        effective_offset_ratio: float,
    ) -> float:
        """439# Cross-venue lead-lag guard.

        433# §3 の BitFlyer 案を 434# §4.2 に沿って安全側へ補正し、
        adverse-side の retreat / veto のみを行う。
        442# 拡張: depth imbalance が adverse 方向を裏付ける場合に追加ブーストを適用。
        """
        self._cross_venue_lead_lag_vetoed = False
        self._cross_venue_lead_lag_veto_reason = None
        # 448# F2: no-op 可視化
        self._cross_venue_lead_lag_pre_offset = None
        self._cross_venue_lead_lag_post_offset = None
        self._cross_venue_lead_lag_cap_hit = False

        cfg = self._config
        hint = self._cross_venue_lead_lag_hint
        if (
            not cfg.cross_venue_lead_lag_enabled
            or hint is None
            or hint.age_sec > cfg.cross_venue_lead_lag_max_age_sec
        ):
            return effective_offset_ratio

        # 512# favorable-side tightening: adverse 側でなくても favorable 側で offset 縮小
        if hint.adverse_side != side:
            if not cfg.cross_venue_favorable_tighten_enabled:
                return effective_offset_ratio
            # favorable side: offset を confidence 比例で縮小 (fill rate 向上)
            # tighten_factor = 1 - (1 - mult) × confidence
            # conf=1.0, mult=0.90 → factor=0.90 (10% 縮小)
            # conf=0.5, mult=0.90 → factor=0.95 (5% 縮小)
            pre_offset = effective_offset_ratio
            tighten_factor = 1.0 - (1.0 - cfg.cross_venue_favorable_tighten_mult) * hint.confidence
            effective_offset_ratio *= tighten_factor
            self._cross_venue_lead_lag_pre_offset = pre_offset
            self._cross_venue_lead_lag_post_offset = effective_offset_ratio
            self._cross_venue_lead_lag_cap_hit = False
            logger.info(
                "[cross_venue] %s favorable tighten from %s: offset %.4f->%.4f "
                "(factor=%.3f, conf=%.2f, spread=%+.2fbps, direction=%s)",
                side,
                hint.reference_exchange,
                pre_offset,
                effective_offset_ratio,
                tighten_factor,
                hint.confidence,
                hint.spread_bps,
                hint.direction,
            )
            return effective_offset_ratio

        if (
            cfg.cross_venue_lead_lag_veto_enabled
            and abs(hint.spread_bps) >= cfg.cross_venue_lead_lag_veto_threshold_bps
        ):
            # 533# veto deadlock 防止 #1: 在庫ゼロ時は閾値を緩和
            # BTC=0 で sell vetoed → 買い機会を逃すだけなので閾値を上げる
            _effective_threshold = cfg.cross_venue_lead_lag_veto_threshold_bps
            _btc = getattr(self, "_veto_btc_balance", None)
            if (
                _btc is not None
                and _btc < 1e-8
                and side == "buy"
            ):
                _effective_threshold *= cfg.cross_venue_lead_lag_veto_inventory_zero_threshold_mult
                if abs(hint.spread_bps) < _effective_threshold:
                    logger.info(
                        "[cross_venue] veto relaxed: BTC=0 → threshold %.1f→%.1fbps "
                        "(spread=%+.2fbps, consecutive=%d)",
                        cfg.cross_venue_lead_lag_veto_threshold_bps,
                        _effective_threshold,
                        hint.spread_bps,
                        self._consecutive_veto_count,
                    )
                    self._consecutive_veto_count = 0
                    # fall through to adverse retreat
                else:
                    pass  # still above relaxed threshold → veto

            # 533# veto deadlock 防止 #2: 連続 veto 上限超過で強制解除
            _max_consecutive = cfg.cross_venue_lead_lag_veto_max_consecutive
            if (
                abs(hint.spread_bps) >= _effective_threshold
                and self._consecutive_veto_count < _max_consecutive
            ):
                self._consecutive_veto_count += 1
                self._cross_venue_lead_lag_vetoed = True
                self._cross_venue_lead_lag_veto_reason = (
                    f"cross_venue_veto: {side} suppressed by "
                    f"{hint.reference_exchange} {hint.direction} lead "
                    f"(spread={hint.spread_bps:+.2f}bps"
                    f"{f', pt_spread={hint.point_spread_bps:+.2f}bps' if hint.point_spread_bps is not None else ''}"
                    f", velocity={hint.reference_velocity_bps:+.2f}bps/s"
                    f", age={hint.age_sec:.2f}s"
                    f", consecutive={self._consecutive_veto_count}/{_max_consecutive})"
                )
                logger.info("[%s] %s", "cross_venue", self._cross_venue_lead_lag_veto_reason)
                return effective_offset_ratio
            elif abs(hint.spread_bps) >= _effective_threshold:
                # max_consecutive 超過 → veto 強制解除
                logger.warning(
                    "[cross_venue] veto force-released: consecutive %d >= max %d "
                    "(spread=%+.2fbps, side=%s). Falling through to adverse retreat.",
                    self._consecutive_veto_count,
                    _max_consecutive,
                    hint.spread_bps,
                    side,
                )
                self._consecutive_veto_count = 0
                # fall through to adverse retreat

        # veto 非発動時は連続カウンタをリセット
        self._consecutive_veto_count = 0

        # 439# base offset boost (adverse-side retreat)
        # 445# confidence-proportional: boost = 1 + (max_boost - 1) * confidence
        pre_offset = effective_offset_ratio
        actual_boost = 1.0 + (
            cfg.cross_venue_lead_lag_offset_boost - 1.0
        ) * hint.confidence
        effective_offset_ratio, applied_mult = self._scale_offset_ratio(
            effective_offset_ratio,
            actual_boost,
            max_ratio=self._effective_max_ratio(side),
        )

        # 442# depth imbalance boost: reference 板の不均衡が adverse 方向を裏付ける場合
        depth_imb_applied = False
        if (
            cfg.cross_venue_depth_imbalance_enabled
            and hint.depth_imbalance is not None
        ):
            # direction=up (price rising) → bid_depth > ask_depth → imbalance > 0 → sell が adverse
            # direction=down → bid_depth < ask_depth → imbalance < 0 → buy が adverse
            # 449# threshold を config 化 (従来はハードコード 0.1)
            _di_thr = cfg.cross_venue_depth_imbalance_threshold
            adverse_confirmed = (
                (hint.direction == "up" and hint.depth_imbalance > _di_thr)
                or (hint.direction == "down" and hint.depth_imbalance < -_di_thr)
            )
            if adverse_confirmed:
                effective_offset_ratio, di_mult = self._scale_offset_ratio(
                    effective_offset_ratio,
                    cfg.cross_venue_depth_imbalance_boost,
                    max_ratio=self._effective_max_ratio(side),
                )
                depth_imb_applied = True

        # 448# F3修正: point_spread も出力して診断性向上
        # 448# F2: cap_hit (no-op) 検出 — boost したのに変わらなかった場合
        total_mult = effective_offset_ratio / pre_offset if pre_offset > 0 else 1.0
        cap_hit = abs(total_mult - 1.0) < 1e-6 and actual_boost > 1.0 + 1e-6
        self._cross_venue_lead_lag_pre_offset = pre_offset
        self._cross_venue_lead_lag_post_offset = effective_offset_ratio
        self._cross_venue_lead_lag_cap_hit = cap_hit
        ps_info = (
            f", pt_spread={hint.point_spread_bps:+.2f}bps"
            if hint.point_spread_bps is not None
            else ""
        )
        logger.info(
            "[cross_venue] %s adverse hint from %s: offset %.4f->%.4f "
            "(mult=%.2f, conf=%.2f, spread=%+.2fbps%s, velocity=%+.2fbps/s, age=%.2fs"
            "%s%s)",
            side,
            hint.reference_exchange,
            pre_offset,
            effective_offset_ratio,
            total_mult,
            hint.confidence,
            hint.spread_bps,
            ps_info,
            hint.reference_velocity_bps,
            hint.age_sec,
            f", depth_imb={hint.depth_imbalance:+.3f} BOOST" if depth_imb_applied else "",
            ", CAP_HIT=NO-OP" if cap_hit else "",
        )
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
                        max_ratio=self._effective_max_ratio(side),
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
        if side != "sell":
            return effective_offset_ratio
        utc_h = self._current_utc_hour()
        mult = self._resolve_sell_hour_boost_mult()
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


# ── 372# E12: VG イベント構造化ログ ──────────────────────

_VG_EVENT_LOG_PATH = "logs/vg_events.jsonl"


def _emit_vg_event(
    *,
    side: str,
    pre_offset: float,
    post_offset: float,
    reason: str,
    velocity_bps: float | None,
    vpin: float | None,
    boost_factor: float,
) -> None:
    """VG 発動イベントを JSONL ファイルに追記.

    372# E12 解消: プレーンテキストログの regex パースではなく
    構造化データとして保存し、分析ツール (vg_and_trend.py) から
    直接利用可能にする。
    """
    try:
        now = _datetime.now(_timezone.utc)
        event = {
            "event": "vg_activation",
            "timestamp_iso": now.isoformat(),
            "timestamp_epoch": _time.time(),
            "side": side,
            "pre_offset": round(pre_offset, 6),
            "post_offset": round(post_offset, 6),
            "reason": reason,
            "velocity_bps": round(velocity_bps, 4) if velocity_bps is not None else None,
            "vpin": round(vpin, 4) if vpin is not None else None,
            "boost_factor": round(boost_factor, 4),
        }
        _append_jsonl(_Path(_VG_EVENT_LOG_PATH), [event])
    except Exception:
        # イベントログ失敗で取引を止めてはならない
        logger.debug("[vg_event] JSONL write failed", exc_info=True)
