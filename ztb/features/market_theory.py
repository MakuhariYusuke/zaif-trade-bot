# 379# Market Theory Features — 035#-306# Pre-366# Systems SAC Integration
# 035#-306# の 10 市場理論システムを SAC 観測空間に接続するための特徴量群
#
# 既存 12 FeatureRegistry 特徴量 (scalping.py) を補完する 5 新特徴量:
#   1. parkinson_sigma  (305#) — Parkinson intra-bar σ (H/L ベース)
#   2. vpin_proxy       (107#) — VPIN order flow toxicity
#   3. kyle_lambda_proxy(266#) — Kyle 1985 価格インパクト係数
#   4. amihud_illiq      (266#) — Amihud 2002 非流動性比率
#   5. ema_velocity_bps  (227#/200#) — EMA 平滑化 velocity (bps)
#
# 設計根拠:
#   - 035# regime → M2 (BayesianRegime posterior) で既にカバー
#   - 054# imbalance → order_flow_imbalance (scalping.py) でカバー
#   - 162#/228# inventory → env-internal state (Phase 3 future work)
#   - 226# loss_boost → env-internal state (Phase 3 future work)
#   - 258# AS/GLFT → parkinson_sigma + kyle_lambda_proxy で間接的にカバー
#   - 残り 5 システムの本質的情報を新特徴量として抽出
#
# SAC 観測空間: 12 → 17 特徴量 (50K steps で学習可能な上限内)

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ztb.features.core.registry import FeatureRegistry

register = FeatureRegistry.register

# ── Parkinson 定数 ──
_PARKINSON_DENOM: float = 2.0 * math.sqrt(math.log(2.0))


@register("parkinson_sigma")
def parkinson_sigma(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """305# Parkinson (1980) High-Low Volatility Estimator.

    σ_P = ln(H/L) / (2·√(ln2))

    realized_volatility (return-based RV) と補完的:
    - RV: close-to-close の price path 情報
    - Parkinson: intra-bar の range 情報 (bar 内部の変動)
    maker_microstructure.py _estimate_sigma() のバッチ版。

    Args:
        df: OHLCV DataFrame (high, low 必須)
        window: rolling window for smoothing

    Returns:
        Parkinson σ の rolling mean (滑らかさのため)
    """
    high = df["high"].values.astype(np.float64, copy=False)
    low = df["low"].values.astype(np.float64, copy=False)
    n = len(high)
    sigma = np.zeros(n, dtype=np.float64)

    # バーごとの Parkinson σ: ln(H/L) / (2√ln2)
    valid = (high > 0) & (low > 0) & (high > low)
    log_hl = np.zeros(n, dtype=np.float64)
    log_hl[valid] = np.log(high[valid] / low[valid])
    sigma = log_hl / _PARKINSON_DENOM

    # rolling mean で平滑化 (ノイズ低減)
    if window > 1 and n > window:
        sigma_series = pd.Series(sigma)
        sigma = sigma_series.rolling(window, min_periods=1).mean().to_numpy()

    return pd.Series(sigma, index=df.index, name="parkinson_sigma")


@register("vpin_proxy")
def vpin_proxy(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """107# VPIN (Volume-synchronized Probability of Informed trading) Proxy.

    VPIN = rolling_sum(|buy_vol - sell_vol|) / rolling_sum(total_vol)

    order_flow_imbalance (scalping.py) との違い:
    - order_flow_imbalance: candle 形状 (wick/body ratio) ベース
    - vpin_proxy: volume-weighted signed flow の毒性指標
    maker_risk_guards.py の Volatility Guard が使用する VPIN と同ロジック。

    Args:
        df: OHLCV DataFrame (close, high, low, volume 必須)
        window: rolling window

    Returns:
        VPIN proxy [0, 1]: 1 に近いほど informed trading 支配的
    """
    eps = 1e-10
    close = df["close"].values.astype(np.float64, copy=False)
    high = df["high"].values.astype(np.float64, copy=False)
    low = df["low"].values.astype(np.float64, copy=False)
    volume = df["volume"].values.astype(np.float64, copy=False)
    n = len(close)
    vpin = np.zeros(n, dtype=np.float64)

    if n == 0:
        return pd.Series(vpin, index=df.index, name="vpin_proxy")

    # CLV (Close Location Value) で buy/sell volume を推定
    hl_range = high - low
    safe_range = np.where(hl_range > eps, hl_range, 1.0)
    clv = ((close - low) / safe_range) * 2 - 1  # [-1, +1]

    # |signed_volume| = |CLV × volume|
    abs_signed_vol = np.abs(clv * volume)

    if window > 0 and n > 1:
        abs_sv_series = pd.Series(abs_signed_vol)
        vol_series = pd.Series(volume)
        rolling_abs_sv = abs_sv_series.rolling(window, min_periods=1).sum().to_numpy()
        rolling_vol = vol_series.rolling(window, min_periods=1).sum().to_numpy()
        valid = rolling_vol > eps
        vpin[valid] = rolling_abs_sv[valid] / rolling_vol[valid]

    return pd.Series(vpin, index=df.index, name="vpin_proxy")


@register("kyle_lambda_proxy")
def kyle_lambda_proxy(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """266# Kyle (1985) 価格インパクト係数 λ の OHLCV Proxy.

    λ_est ≈ (high - low) / (2 · volume)

    実際の Kyle λ は板 depth から計算 (maker_microstructure.py):
      λ = spread / (2 · depth_volume)
    OHLCV では depth が不明なため、intra-bar range / volume で近似。

    高値 = 薄い板 (高インパクト/高リスク)。
    SAC が高リスク環境を認識するための指標。

    Args:
        df: OHLCV DataFrame (high, low, volume 必須)
        window: rolling window for smoothing

    Returns:
        Kyle λ proxy (rolling mean, 正規化済み)
    """
    eps = 1e-10
    high = df["high"].values.astype(np.float64, copy=False)
    low = df["low"].values.astype(np.float64, copy=False)
    volume = df["volume"].values.astype(np.float64, copy=False)
    n = len(high)
    lam = np.zeros(n, dtype=np.float64)

    # λ ≈ range / (2 · volume)
    bar_range = high - low
    safe_vol = np.clip(volume, eps, None)
    raw_lambda = bar_range / (2.0 * safe_vol)

    # rolling mean + z-score 正規化 (レベル不変にする)
    if window > 1 and n > window:
        lam_series = pd.Series(raw_lambda)
        rolling_mean = lam_series.rolling(window, min_periods=1).mean()
        rolling_std = lam_series.rolling(window, min_periods=1).std().fillna(eps)
        # std が極小 (price stall) のとき z-score 爆発防止: floor を設ける
        safe_std = np.maximum(rolling_std.to_numpy(), rolling_mean.to_numpy().clip(min=eps) * 0.01)
        lam = (raw_lambda - rolling_mean.to_numpy()) / (safe_std + eps)
    else:
        lam = raw_lambda

    # clip: extreme outliers を抑制 (SAC obs 安定化)
    lam = np.clip(lam, -5.0, 5.0)

    return pd.Series(lam, index=df.index, name="kyle_lambda_proxy")


@register("amihud_illiq")
def amihud_illiq(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """266# Amihud (2002) Illiquidity Ratio OHLCV Proxy.

    ILLIQ = |return| / volume

    Kyle λ と補完的:
    - Kyle λ: spread/depth ベースの瞬間的インパクト
    - Amihud ILLIQ: return/volume ベースの統計的非流動性
    maker_microstructure.py _apply_amihud_illiq() のバッチ版。

    Args:
        df: OHLCV DataFrame (close, volume 必須)
        window: rolling window

    Returns:
        Amihud ILLIQ rolling mean (z-score 正規化済み)
    """
    eps = 1e-10
    close = df["close"].values.astype(np.float64, copy=False)
    volume = df["volume"].values.astype(np.float64, copy=False)
    n = len(close)
    illiq = np.zeros(n, dtype=np.float64)

    if n <= 1:
        return pd.Series(illiq, index=df.index, name="amihud_illiq")

    # |return| = |Δclose / close_{t-1}|
    prev_close = np.empty(n, dtype=np.float64)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    valid_prev = prev_close > eps
    abs_return = np.zeros(n, dtype=np.float64)
    abs_return[valid_prev] = np.abs(
        (close[valid_prev] - prev_close[valid_prev]) / prev_close[valid_prev]
    )
    abs_return[0] = 0.0

    # ILLIQ = |return| / volume
    safe_vol = np.clip(volume, eps, None)
    raw_illiq = abs_return / safe_vol

    # rolling mean + z-score 正規化
    if window > 1 and n > window:
        illiq_series = pd.Series(raw_illiq)
        rolling_mean = illiq_series.rolling(window, min_periods=1).mean()
        rolling_std = illiq_series.rolling(window, min_periods=1).std().fillna(eps)
        # std が極小 (price stall) のとき z-score 爆発防止
        safe_std = np.maximum(rolling_std.to_numpy(), rolling_mean.to_numpy().clip(min=eps) * 0.01)
        illiq = (raw_illiq - rolling_mean.to_numpy()) / (safe_std + eps)
    else:
        illiq = raw_illiq

    # clip: extreme outliers を抑制 (SAC obs 安定化)
    illiq = np.clip(illiq, -5.0, 5.0)

    return pd.Series(illiq, index=df.index, name="amihud_illiq")


@register("ema_velocity_bps")
def ema_velocity_bps(df: pd.DataFrame, span: int = 5) -> pd.Series:
    """227#/200# EMA 平滑化 Velocity (bps).

    price_velocity (scalping.py) との違い:
    - price_velocity: 1-bar リターン (ノイジー、bid-ask bounce に敏感)
    - ema_velocity_bps: EMA 平滑化 (noise filtered)、bps 単位
    maker_price.py _smoothed_velocity_bps のバッチ版。

    velocity_math.py compute_instant_velocity_bps() のコンセプトを
    EMA フィルタリングしてノイズ耐性を高めた特徴量。

    Args:
        df: OHLCV DataFrame (close 必須)
        span: EMA span (default 5 — maker_price.py と同等)

    Returns:
        EMA smoothed velocity in bps
    """
    close = df["close"].values.astype(np.float64, copy=False)
    n = len(close)
    velocity = np.zeros(n, dtype=np.float64)

    if n <= 1:
        return pd.Series(velocity, index=df.index, name="ema_velocity_bps")

    # 1-bar velocity in bps: (close[t] - close[t-1]) / close[t-1] * 10000
    prev_close = np.empty(n, dtype=np.float64)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    valid = prev_close > 0
    velocity[valid] = (close[valid] - prev_close[valid]) / prev_close[valid] * 10000.0
    velocity[0] = 0.0

    # EMA smoothing (pandas ewm)
    velocity_series = pd.Series(velocity, index=df.index)
    ema_velocity = velocity_series.ewm(span=span, min_periods=1, adjust=False).mean()

    return ema_velocity.rename("ema_velocity_bps")
