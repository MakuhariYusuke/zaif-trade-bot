#!/usr/bin/env python3
"""
Yahoo Financeデータを使用したActionSignalGuide検証スクリプト
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
    ActionSignalGuide,
    ActionSignalGuideConfig,
)


def fetch_yahoo_finance_data(
    symbol: str = "BTC-JPY", period: str = "7d", interval: str = "1m"
) -> pd.DataFrame:
    """
    Yahoo Financeからデータを取得

    Args:
        symbol: 取得するシンボル（デフォルト: BTC-JPY）
        period: 取得期間（デフォルト: 7d）
        interval: データ間隔（デフォルト: 1m）

    Returns:
        OHLCVデータを含むDataFrame
    """
    print(f"Fetching {symbol} data from Yahoo Finance...")
    print(f"Period: {period}, Interval: {interval}")

    try:
        # データ取得
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            raise ValueError(f"No data found for {symbol}")

        # カラム名を標準化
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        # インデックスをリセットしてtimestampカラムを作成
        df = df.reset_index()
        df = df.rename(columns={"Datetime": "timestamp"})

        # 日本時間に変換（Yahoo FinanceはUTC）
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(
                "Asia/Tokyo"
            )

        # 必要なカラムのみ保持
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df = df[required_columns]

        # データ品質チェック
        print(f"Loaded {len(df)} rows of {symbol} data")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Price range: {df['close'].min():.0f} - {df['close'].max():.0f} JPY")

        # データの完全性チェック
        missing_data = df.isnull().sum()
        if missing_data.any():
            print(
                f"Warning: Missing data found: {missing_data[missing_data > 0].to_dict()}"
            )

        # OHLC関係の妥当性チェック
        invalid_ohlc = (
            (df["high"] < df["low"])
            | (df["open"] < df["low"])
            | (df["open"] > df["high"])
            | (df["close"] < df["low"])
            | (df["close"] > df["high"])
        )
        if invalid_ohlc.any():
            print(f"Warning: {invalid_ohlc.sum()} rows have invalid OHLC relationships")

        return df

    except Exception as e:
        print(f"Error fetching data from Yahoo Finance: {e}")
        raise


def prepare_data_for_testing(df: pd.DataFrame, test_points: int = 7) -> pd.DataFrame:
    """
    テスト用のデータを準備

    Args:
        df: 取得したデータ
        test_points: テストポイント数

    Returns:
        テスト用にフィルタリングされたデータ
    """
    # 最新のデータをテスト用に使用（メモリ節約のため）
    recent_data = df.tail(500).copy()

    print(f"Using last {len(recent_data)} minutes for testing")

    return recent_data


def test_action_signal_guide_yahoo_data(df: pd.DataFrame) -> dict:
    """
    ActionSignalGuideをYahoo Financeデータでテスト

    Args:
        df: テスト用データ

    Returns:
        テスト結果
    """
    print("\n2. Initializing ActionSignalGuide...")

    # ActionSignalGuideの初期化
    start_time = datetime.now()
    config = ActionSignalGuideConfig()
    asg = ActionSignalGuide(config)
    end_time = datetime.now()
    initialization_time = (end_time - start_time).total_seconds()

    print(f"   ActionSignalGuide initialized in {initialization_time:.3f}s")
    print("\n3. Testing signal generation with Yahoo Finance data...")

    # テストポイントの選択（等間隔）
    test_indices = np.linspace(0, len(df) - 1, 7, dtype=int)
    test_indices = [min(idx, len(df) - 1) for idx in test_indices]

    signals_generated = []
    processing_times = []

    for i, idx in enumerate(test_indices):
        test_point = df.iloc[idx]
        timestamp = test_point["timestamp"]

        print(f"   Testing at index {idx} ({timestamp}): ", end="")

        # テストポイントまでのデータを準備
        historical_data = df.iloc[: idx + 1].copy()

        try:
            start_time = datetime.now()

            # 信号生成（current_indexを渡す）
            signals = asg.generate_signals(historical_data, idx)

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            processing_times.append(processing_time)

            signal_count = len(signals) if signals else 0
            signals_generated.append(signals)

            print(f"{signal_count} signals")

            # 最初の信号の詳細を表示
            if signals:
                for j, signal in enumerate(signals[:1]):  # 最初の信号のみ表示
                    print(
                        f"     Signal {j+1}: {signal.signal_type}, dir={signal.direction:.3f}, conf={signal.confidence:.3f}"
                    )

        except Exception as e:
            print(f"Error: {e}")
            signals_generated.append([])
            processing_times.append(0)

    # 結果の集計
    total_signals = sum(len(signals) if signals else 0 for signals in signals_generated)
    avg_signals = total_signals / len(test_indices)
    max_signals = max(len(signals) if signals else 0 for signals in signals_generated)
    min_signals = min(len(signals) if signals else 0 for signals in signals_generated)

    print("\n   Signal generation summary:")
    print(f"   - Total test points: {len(test_indices)}")
    print(f"   - Total signals generated: {total_signals}")
    print(f"   - Average signals per test: {avg_signals:.2f}")
    print(f"   - Max signals per test: {max_signals}")
    print(f"   - Min signals per test: {min_signals}")
    print(
        f"   - Average processing time: {np.mean(processing_times) if processing_times else 0:.4f}s"
    )
    print(
        f"   - Max processing time: {max(processing_times) if processing_times else 0:.4f}s"
    )
    # 信号品質分析
    print("\n4. Signal quality analysis:")

    all_signals = [
        signal for signals in signals_generated for signal in signals if signals
    ]

    if all_signals:
        directions = [s.direction for s in all_signals]
        confidences = [s.confidence for s in all_signals]
        strengths = [s.strength for s in all_signals]

        print(f"   - Direction range: {min(directions):.3f} to {max(directions):.3f}")
        print(f"   - Average direction: {np.mean(directions):.3f}")
        print(
            f"   - Confidence range: {min(confidences):.3f} to {max(confidences):.3f}"
        )
        print(f"   - Average confidence: {np.mean(confidences):.3f}")
        print(f"   - Strength range: {min(strengths):.3f} to {max(strengths):.3f}")
        print(f"   - Average strength: {np.mean(strengths):.3f}")

        # 信号タイプ分析
        signal_types = {}
        for signal in all_signals:
            signal_type = signal.signal_type
            signal_types[signal_type] = signal_types.get(signal_type, 0) + 1

        print(f"   - Signal types detected: {len(signal_types)}")
        print("   - Top signal types:")
        for signal_type, count in sorted(
            signal_types.items(), key=lambda x: x[1], reverse=True
        )[:5]:
            print(f"     {signal_type}: {count} signals")
    else:
        print("   - No signals generated for quality analysis")

    # パフォーマンス分析
    print("\n5. Performance analysis...")

    try:
        performance_report = asg.analyze_performance()
        print("   Performance report generated successfully")
        if isinstance(performance_report, dict):
            print(f"   Report keys: {list(performance_report.keys())}")
    except Exception as e:
        print(f"   Performance analysis failed: {e}")

    return {
        "total_test_points": len(test_indices),
        "total_signals": total_signals,
        "avg_signals_per_test": avg_signals,
        "max_signals_per_test": max_signals,
        "min_signals_per_test": min_signals,
        "avg_processing_time": np.mean(processing_times) if processing_times else 0,
        "max_processing_time": max(processing_times) if processing_times else 0,
        "signal_types": signal_types if "signal_types" in locals() else {},
        "all_signals": all_signals,
    }


def main():
    """メイン実行関数"""
    print("=== ActionSignalGuide Yahoo Finance Data Validation ===")

    try:
        # Yahoo Financeからデータを取得
        df = fetch_yahoo_finance_data()

        # テスト用データを準備
        test_df = prepare_data_for_testing(df)

        # ActionSignalGuideでテスト
        results = test_action_signal_guide_yahoo_data(test_df)

        print("\n=== Yahoo Finance Data Validation Complete ===")
        print("ActionSignalGuide successfully processed Yahoo Finance BTC/JPY data!")
        print(
            f"Generated {results['total_signals']} signals across {results['total_test_points']} test points."
        )

        return True

    except Exception as e:
        print("\n=== Validation Failed ===")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
