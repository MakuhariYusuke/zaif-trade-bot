from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_base_features(df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """
    Calculate the 30 base features expected by v456 Factory.
    Uses numpy for speed and minimal deps.
    Replaces random ADX/DI with real calculations.
    """
    if copy:
        df = df.copy()

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values

    def _sma(arr: np.ndarray, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros_like(arr, dtype=float)
        ret = np.cumsum(arr, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return np.concatenate((np.zeros(n - 1), ret[n - 1:] / n))

    def _ema(arr: np.ndarray, n: int) -> np.ndarray:
        alpha = 2 / (n + 1)
        res = np.zeros_like(arr)
        if len(arr) > 0:
            res[0] = arr[0]
        for i in range(1, len(arr)):
            res[i] = alpha * arr[i] + (1 - alpha) * res[i - 1]
        return res

    def _rsi(arr: np.ndarray, n: int) -> np.ndarray:
        if len(arr) <= n:
            return np.zeros_like(arr)
        delta = np.diff(arr)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.zeros(len(arr))
        avg_loss = np.zeros(len(arr))

        avg_gain[n] = np.mean(gain[:n])
        avg_loss[n] = np.mean(loss[:n])
        for i in range(n + 1, len(arr)):
            avg_gain[i] = (avg_gain[i - 1] * (n - 1) + gain[i - 1]) / n
            avg_loss[i] = (avg_loss[i - 1] * (n - 1) + loss[i - 1]) / n

        with np.errstate(divide="ignore", invalid="ignore"):
            rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
            rsi = 100 - (100 / (1 + rs))
        return np.nan_to_num(rsi, nan=50)

    def _adx_di(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(close) <= period:
            return np.zeros(len(close)), np.zeros(len(close)), np.zeros(len(close))

        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]

        plus_dm = np.zeros(len(close))
        minus_dm = np.zeros(len(close))

        for i in range(len(up_move)):
            if up_move[i] > down_move[i] and up_move[i] > 0:
                plus_dm[i + 1] = up_move[i]
            if down_move[i] > up_move[i] and down_move[i] > 0:
                minus_dm[i + 1] = down_move[i]

        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))),
        )
        tr[0] = high[0] - low[0]

        atr = _ema(tr, period)
        smoothed_plus_dm = _ema(plus_dm, period)
        smoothed_minus_dm = _ema(minus_dm, period)

        with np.errstate(divide="ignore", invalid="ignore"):
            plus_di = 100 * smoothed_plus_dm / atr
            minus_di = 100 * smoothed_minus_dm / atr

            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            dx = np.nan_to_num(dx, 0)

        adx = _ema(dx, period)

        return np.nan_to_num(adx), np.nan_to_num(plus_di), np.nan_to_num(minus_di)

    df["sma_5"] = _sma(close, 5)
    df["sma_20"] = _sma(close, 20)
    df["sma_50"] = _sma(close, 50)
    df["ema_5"] = _ema(close, 5)
    df["ema_20"] = _ema(close, 20)
    df["ema_50"] = _ema(close, 50)

    df["rsi_14"] = _rsi(close, 14)
    df["rsi_20"] = _rsi(close, 20)

    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))),
    )
    tr[0] = high[0] - low[0]
    df["atr_14"] = _ema(tr, 14)
    df["atr_20"] = _ema(tr, 20)

    df["atr"] = df["atr_14"]

    sma20 = df["sma_20"].values
    std20 = pd.Series(close).rolling(20).std().fillna(0).values
    df["bb_upper_20"] = sma20 + 2 * std20
    df["bb_lower_20"] = sma20 - 2 * std20
    with np.errstate(divide="ignore", invalid="ignore"):
        df["bb_pct_b_20"] = (close - df["bb_lower_20"]) / (
            df["bb_upper_20"] - df["bb_lower_20"]
        )

    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    df["macd_line"] = ema12 - ema26
    df["macd_signal"] = _ema(df["macd_line"].values, 9)

    df["adx_14"], df["plus_di_14"], df["minus_di_14"] = _adx_di(high, low, close, 14)

    df["obv"] = np.cumsum(np.sign(np.diff(close, prepend=close[0])) * volume)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["vpt"] = np.cumsum(volume * np.diff(close, prepend=close[0]) / close)

    with np.errstate(divide="ignore", invalid="ignore"):
        df["sma_5_close_ratio"] = df["sma_5"] / close
        df["atr_pct_close"] = df["atr_14"] / close
        df["hl_ratio"] = high / low
        df["hml_ratio"] = (high - low) / close
    df["trend_direction"] = np.sign(df["sma_5"] - df["sma_20"])

    df = df.fillna(0)
    return df
