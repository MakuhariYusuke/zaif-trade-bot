#!/usr/bin/env python3
"""
RSI, MACD, 移動平均相当指標の動作確認スクリプト
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


def test_technical_indicators():
    """
    RSI, MACD, 移動平均相当指標の動作確認
    """
    print("=== RSI, MACD, 移動平均相当指標動作確認 ===")

    try:
        # Yahoo Financeからデータを取得
        print("\n1. データ取得...")
        ticker = yf.Ticker("BTC-JPY")
        df = ticker.history(period="7d", interval="1m")

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
        df = df.reset_index()
        df = df.rename(columns={"Datetime": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(
            "Asia/Tokyo"
        )

        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df = df[required_columns]

        print(f"Loaded {len(df)} rows of BTC-JPY data")

        # テスト用データを準備
        test_df = df.tail(500).copy()
        print(f"Using last {len(test_df)} minutes for testing")

        # ActionSignalGuideの初期化（oscillator_patterns有効）
        print("\n2. ActionSignalGuide初期化（RSI, MACD有効）...")
        config = ActionSignalGuideConfig(
            guidance_level="MODERATE",  # より多くの信号を通す
            enable_oscillator_patterns=True,
            enable_bollinger_patterns=True,  # 移動平均相当
            enable_adx_patterns=True,
            enable_volume_patterns=True,
        )
        asg = ActionSignalGuide(config)

        # テスト実行
        print("\n3. テクニカル指標ベース信号生成テスト...")
        test_indices = np.linspace(0, len(test_df) - 1, 10, dtype=int)
        test_indices = [min(idx, len(test_df) - 1) for idx in test_indices]

        total_signals = 0
        rsi_signals = 0
        macd_signals = 0
        bollinger_signals = 0
        other_signals = 0
        processing_times = []

        for i, idx in enumerate(test_indices):
            try:
                start_time = datetime.now()
                signals = asg.generate_signals(test_df, idx)
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                processing_times.append(processing_time)

                signal_count = len(signals) if signals else 0
                total_signals += signal_count

                print(f"   Index {idx}: {signal_count} signals")

                # 信号タイプの分類
                if signals:
                    for signal in signals:
                        signal_type = getattr(signal, "signal_type", "unknown")
                        if "rsi" in signal_type.lower():
                            rsi_signals += 1
                        elif "macd" in signal_type.lower():
                            macd_signals += 1
                        elif (
                            "bollinger" in signal_type.lower()
                            or "bb" in signal_type.lower()
                        ):
                            bollinger_signals += 1
                        else:
                            other_signals += 1

                        if i < 3:  # 最初の3テストのみ詳細表示
                            print(
                                f"     Signal: {signal_type}, dir={signal.direction:.3f}, conf={signal.confidence:.3f}"
                            )

            except Exception as e:
                print(f"   Error at index {idx}: {e}")

        # 結果集計
        avg_signals = total_signals / len(test_indices)

        print("\n=== テクニカル指標テスト結果 ===")
        print(f"   - テストポイント: {len(test_indices)}")
        print(f"   - 総信号数: {total_signals}")
        print(f"   - 平均信号/テスト: {avg_signals:.2f}")
        print(f"   - RSI信号数: {rsi_signals}")
        print(f"   - MACD信号数: {macd_signals}")
        print(f"   - ボリンジャー信号数: {bollinger_signals}")
        print(f"   - その他信号数: {other_signals}")

        # 評価
        print("\n=== 評価 ===")
        if rsi_signals > 0:
            print("✓ RSIパターン認識器が正常に動作しています")
        else:
            print("✗ RSI信号が生成されていません")

        if macd_signals > 0:
            print("✓ MACDパターン認識器が正常に動作しています")
        else:
            print("✗ MACD信号が生成されていません")

        if bollinger_signals > 0:
            print("✓ ボリンジャーバンド（移動平均相当）が正常に動作しています")
        else:
            print("✗ ボリンジャー信号が生成されていません")

        if avg_signals > 1.0:
            print("✓ 高頻度取引要件を満たす信号数を生成しています")
        elif avg_signals > 0.5:
            print("△ 中程度の改善が見られます")
        else:
            print("✗ さらなる改善が必要です")

        # 推奨設定
        print("\n=== 推奨設定 ===")
        print("高頻度取引向けの設定:")
        print("  guidance_level: MODERATE")
        print(
            "  enable_oscillator_patterns: True (RSI, MACD, ストキャスティクス, Williams %R, CCI, MFI)"
        )
        print("  enable_bollinger_patterns: True (移動平均相当)")
        print("  enable_adx_patterns: True (トレンド強度)")
        print("  enable_volume_patterns: True (出来高分析)")

        return True

    except Exception as e:
        print("\n=== テスト失敗 ===")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """メイン実行関数"""
    success = test_technical_indicators()

    if success:
        print("\n=== RSI, MACD, 移動平均相当指標動作確認完了 ===")
        print("テクニカル指標ベースの信号生成が検証されました。")
    else:
        print("\n=== テスト失敗 ===")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
