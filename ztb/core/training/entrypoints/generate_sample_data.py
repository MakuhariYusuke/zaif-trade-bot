#!/usr/bin/env python3
# サンプルデータ生成スクリプト

import numpy as np
import pandas as pd

from ztb.utils.talib_wrapper import TaLibWrapper


def generate_sample_data(start_date: str, end_date: str, filename: str) -> pd.DataFrame:
    """サンプル取引データを生成"""
    # 日付範囲の作成
    date_range = pd.date_range(start=start_date, end=end_date, freq="1min")

    n_points = len(date_range)

    # 基本価格データ（ランダムウォーク）
    np.random.seed(42)
    price_changes = np.random.normal(0, 0.001, n_points)  # 0.1%の標準偏差
    base_price = 50000  # BTC価格のベース
    prices = base_price * np.exp(np.cumsum(price_changes))

    # テクニカル指標の生成
    # MACDを先に計算
    macd, macd_signal, macd_hist = TaLibWrapper.macd(prices)
    # ボリンジャーバンドを先に計算
    bb_upper, bb_middle, bb_lower = TaLibWrapper.bbands(prices, 20, 2.0, 2.0)

    # 価格変動率とボラティリティを先に計算
    price_changes = np.concatenate([[0], np.diff(prices) / prices[:-1]])
    price_change_1 = np.concatenate([[0], np.diff(prices) / prices[:-1]])
    price_change_5 = np.concatenate([[0, 0, 0, 0, 0], np.diff(prices, 5) / prices[:-5]])
    price_change_10 = np.concatenate([[0] * 10, np.diff(prices, 10) / prices[:-10]])
    volatility_5 = pd.Series(price_changes).rolling(5).std().values
    volatility_10 = pd.Series(price_changes).rolling(10).std().values
    volatility_20 = pd.Series(price_changes).rolling(20).std().values

    data = {
        "ts": date_range,
        "price": prices,
        "close": prices,  # 環境が期待するcloseカラム
        "volume": np.random.exponential(100, n_points),
        # 移動平均
        "sma_5": TaLibWrapper.sma(prices, 5),
        "sma_10": TaLibWrapper.sma(prices, 10),
        "sma_20": TaLibWrapper.sma(prices, 20),
        "sma_50": TaLibWrapper.sma(prices, 50),
        # RSI
        "rsi_14": 50 + 30 * np.sin(np.linspace(0, 4 * np.pi, n_points)),
        # MACD
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        # ボリンジャーバンド
        "bb_upper": bb_upper,
        "bb_middle": bb_middle,
        "bb_lower": bb_lower,
        # ATR (Average True Range) - high/lowをcloseで近似
        "atr_14": TaLibWrapper.atr(prices, prices, prices, 14),
        # ストキャスティクス
        "stoch_k": 50 + 40 * np.sin(np.linspace(0, 6 * np.pi, n_points)),
        "stoch_d": 50 + 35 * np.sin(np.linspace(0, 6 * np.pi, n_points)),
        # Williams %R
        "williams_r": -50 + 40 * np.sin(np.linspace(0, 8 * np.pi, n_points)),
        # CCI (Commodity Channel Index)
        "cci_20": np.random.normal(0, 100, n_points),
        # ADX (Average Directional Index)
        "adx_14": 20 + 30 * np.random.random(n_points),
        # 価格変動率
        "price_change_1": price_change_1,
        "price_change_5": price_change_5,
        "price_change_10": price_change_10,
        # ボラティリティ
        "volatility_5": volatility_5,
        "volatility_10": volatility_10,
        "volatility_20": volatility_20,
        # 取引所情報
        "exchange": "sample_exchange",
        "pair": "BTC/USD",
    }

    df = pd.DataFrame(data)

    # エピソードIDの追加（1000ステップごとに変更）
    episode_length = 1000
    episode_ids = np.repeat(np.arange(n_points // episode_length + 1), episode_length)[
        :n_points
    ]
    df["episode_id"] = episode_ids

    # NaNをゼロで埋める
    df = df.fillna(0)

    # Parquetファイルとして保存
    df.to_parquet(filename, index=False)
    print(f"Generated sample data: {filename} ({len(df)} rows)")

    return df


def main() -> None:
    """メイン関数"""
    # 各月のデータ生成
    months_data = [
        ("2025-04-01", "2025-04-30", "data/features/2025/04/sample_04.parquet"),
        ("2025-05-01", "2025-05-31", "data/features/2025/05/sample_05.parquet"),
        ("2025-06-01", "2025-06-30", "data/features/2025/06/sample_06.parquet"),
        ("2025-07-01", "2025-07-31", "data/features/2025/07/sample_07.parquet"),
        ("2025-08-01", "2025-08-31", "data/features/2025/08/sample_08.parquet"),
        ("2025-09-01", "2025-09-20", "data/features/2025/09/sample_09.parquet"),
    ]

    for start_date, end_date, filename in months_data:
        generate_sample_data(start_date, end_date, filename)

    print("All sample data generated successfully!")


if __name__ == "__main__":
    main()
