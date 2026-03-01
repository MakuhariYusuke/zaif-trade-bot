"""200# L / 208# SSOT: Velocity 計算ユーティリティ — Single Source of Truth.

全 velocity 関連の **計算ロジック** を一元管理する。
信号の種類は複数あるが、計算・符号規約・上限処理をここに集約することで
「どのモジュールが何をどう計算しているか」の混乱を排除する。

アーキテクチャ:
  本システムには目的の異なる 2 つの velocity 信号が存在する。

  ┌─────────────────┬─────────────────────┬──────────────────────────────────┐
  │ 名前            │ データソース        │ 用途                             │
  ├─────────────────┼─────────────────────┼──────────────────────────────────┤
  │ instant_vel_bps │ orderbook mid-price │ VG offset boost (瞬間急変検知)   │
  │                 │ (point-to-point)    │ maker_price._apply_volatility_   │
  │                 │                     │ guard() で使用                   │
  ├─────────────────┼─────────────────────┼──────────────────────────────────┤
  │ trade_vel_60s   │ 60s 約定履歴        │ SG skip/offset 判定 (中期動向)   │
  │                 │ (first↔last price)  │ skip_gate_evaluator + ML feature │
  └─────────────────┴─────────────────────┴──────────────────────────────────┘

  両者は異なるデータソース・異なるタイムウィンドウから velocity を計測するが、
  以下を共有する:
    - 符号規約: 正=上昇, 負=下降 (bps 単位)
    - offset 乗数計算: compute_velocity_offset_multiplier()
    - bps 変換: _BPS_FACTOR = 10_000

  205# §3.2 / §9.1 への対応:
    - 201# で multiplier 計算のみ共通化していたのを拡張
    - 208# で instant velocity 計算も本モジュールに移動
    - trade_vel_60s は gate_features パイプライン内で計算されるため、
      ここでは利用ガイドのみ提供 (将来的に extract 可能)

Reference:
    - 054# mid price trend 追跡 (initial implementation)
    - 200# L velocity_math.py 新規作成 (multiplier SSOT)
    - 205# §3.2 / §9.1 SSOT 未達指摘
    - 208# instant velocity 計算を本モジュールに移動
"""

from __future__ import annotations

_BPS_FACTOR: float = 10_000.0


def compute_instant_velocity_bps(
    *,
    current_mid: float,
    prev_mid: float,
    dt: float,
    max_dt: float,
) -> float | None:
    """orderbook mid-price から瞬間 velocity (bps) を算出する.

    maker_price.py の inline 計算 (054#) を SSOT として抽出。
    VG (Volatility Guard) の急変検知トリガに使用される。

    符号規約: 正=上昇, 負=下降 (price_velocity_60s と同一)

    Args:
        current_mid: 現在の mid price
        prev_mid: 前回の mid price
        dt: 前回観測からの経過時間 (秒)
        max_dt: この秒数以上の場合は stale と見なし None を返す

    Returns:
        velocity in bps, or None if stale / invalid
    """
    if prev_mid <= 0 or dt <= 0 or dt >= max_dt:
        return None
    return (current_mid - prev_mid) / prev_mid * _BPS_FACTOR


def compute_velocity_offset_multiplier(
    *,
    observed_velocity_bps: float,
    threshold_bps: float,
    base_multiplier: float,
    max_multiplier: float,
    proportional: bool,
) -> tuple[float, bool]:
    """velocity soft mode の offset 乗数を安全に解決する.

    SkipGateEvaluator._compute_velocity_offset_multiplier() から抽出。
    skip_gate_evaluator と maker_price の両方から使用可能。

    - 0 除算回避: threshold=0 の場合は proportional を使わず固定倍率へフォールバック
    - 保守性維持: 1.0 未満の倍率は無効化し、少なくとも現状維持 (1.0) に丸める
    - 上限暴走防止: max_multiplier で頭打ち

    Args:
        observed_velocity_bps: 観測された velocity (bps)
        threshold_bps: 閾値 (bps) — この値を超えた部分で boost が発動
        base_multiplier: 固定モード時の乗数 / 比例モードの基準
        max_multiplier: 乗数上限
        proportional: True=閾値超過量に比例, False=固定値

    Returns:
        (multiplier, was_proportional)
    """
    capped_max = max(1.0, float(max_multiplier))
    bounded_base = min(max(1.0, float(base_multiplier)), capped_max)
    if not proportional:
        return bounded_base, False

    threshold_abs = abs(float(threshold_bps))
    if threshold_abs <= 0.0:
        return bounded_base, False

    excess_ratio = abs(float(observed_velocity_bps)) / threshold_abs
    boost = 1.0 + (bounded_base - 1.0) * excess_ratio
    return min(boost, capped_max), True
