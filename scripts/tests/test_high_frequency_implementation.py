#!/usr/bin/env python3
"""
高頻度取引向けActionSignalGuide実装改善スクリプト
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


class HighFrequencyActionSignalGuide(ActionSignalGuide):
    """
    高頻度取引向けに最適化されたActionSignalGuide
    """

    def __init__(self, config: ActionSignalGuideConfig = None):
        if config is None:
            config = self._create_high_frequency_config()

        super().__init__(config)

    def _create_high_frequency_config(self) -> ActionSignalGuideConfig:
        """高頻度取引向けの設定を作成"""
        config = ActionSignalGuideConfig()

        # 高頻度取引向けの設定調整
        config.guidance_level = "MODERATE"  # より多くの信号を通す
        config.max_signals_per_bar = 5  # 1バーあたりの最大信号数を増やす
        config.enable_parallel_processing = True  # 並列処理を有効化

        # 短期指標を優先的に有効化
        config.enable_oscillator_patterns = True  # RSI, ストキャスティクスなど
        config.enable_candlestick_patterns = True  # 短期的なローソク足パターン
        config.enable_bollinger_patterns = True  # ボリンジャーバンド
        config.enable_volume_patterns = True  # 出来高パターン

        # メモリ管理の最適化
        config.max_signal_history = 2000  # より多くの履歴を保持
        config.memory_cleanup_interval = 50  # 頻繁なクリーンアップ

        return config

    def generate_high_frequency_signals(
        self, data: pd.DataFrame, current_index: int, short_timeframes: list = None
    ) -> list:
        """
        高頻度取引向けの信号生成

        Args:
            data: 市場データ
            current_index: 現在のインデックス
            short_timeframes: 短期タイムフレームデータ

        Returns:
            生成された信号リスト
        """
        # 基本信号生成
        signals = self.generate_signals(data, current_index)

        # 短期タイムフレームデータの活用
        if short_timeframes:
            additional_signals = self._generate_short_termframe_signals(
                data, current_index, short_timeframes
            )
            signals.extend(additional_signals)

        # 高頻度向けのフィルタリング
        filtered_signals = self._apply_high_frequency_filtering(
            signals, data, current_index
        )

        return filtered_signals

    def _generate_short_termframe_signals(
        self, data: pd.DataFrame, current_index: int, short_timeframes: list
    ) -> list:
        """短期タイムフレームからの追加信号生成"""
        additional_signals = []

        try:
            # 5分足データの活用
            if "5m" in short_timeframes:
                five_min_signals = self._analyze_5min_patterns(data, current_index)
                additional_signals.extend(five_min_signals)

            # 1分足特有のパターン分析
            minute_signals = self._analyze_minute_patterns(data, current_index)
            additional_signals.extend(minute_signals)

        except Exception as e:
            self.logger.warning(f"短期タイムフレーム信号生成エラー: {e}")

        return additional_signals

    def _analyze_5min_patterns(self, data: pd.DataFrame, current_index: int) -> list:
        """5分足パターン分析"""
        signals = []

        try:
            # 直近5本の5分足データをシミュレート
            recent_data = data.iloc[max(0, current_index - 4) : current_index + 1]

            if len(recent_data) >= 3:
                # 短期トレンド分析
                closes = recent_data["close"].values
                volumes = recent_data["volume"].values

                # 短期上昇トレンド
                if closes[-1] > closes[-2] > closes[-3] and volumes[-1] > volumes[-2]:
                    signals.append(
                        self._create_signal(
                            direction=0.6,
                            confidence=0.7,
                            strength=0.5,
                            signal_type="short_trend_up",
                            source="5min_analysis",
                        )
                    )

                # 短期下降トレンド
                elif closes[-1] < closes[-2] < closes[-3] and volumes[-1] > volumes[-2]:
                    signals.append(
                        self._create_signal(
                            direction=-0.6,
                            confidence=0.7,
                            strength=0.5,
                            signal_type="short_trend_down",
                            source="5min_analysis",
                        )
                    )

        except Exception as e:
            self.logger.debug(f"5分足パターン分析エラー: {e}")

        return signals

    def _analyze_minute_patterns(self, data: pd.DataFrame, current_index: int) -> list:
        """1分足特有のパターン分析"""
        signals = []

        try:
            # 直近数分のデータを分析
            lookback = min(10, current_index + 1)
            recent_data = data.iloc[current_index - lookback + 1 : current_index + 1]

            if len(recent_data) >= 5:
                closes = recent_data["close"].values
                highs = recent_data["high"].values
                lows = recent_data["low"].values

                # 急激な価格変動の検知
                price_change = (closes[-1] - closes[0]) / closes[0]

                if abs(price_change) > 0.002:  # 0.2%以上の変動
                    direction = 1.0 if price_change > 0 else -1.0
                    signals.append(
                        self._create_signal(
                            direction=direction * 0.8,
                            confidence=min(0.9, abs(price_change) * 100),
                            strength=min(0.8, abs(price_change) * 200),
                            signal_type="momentum_burst",
                            source="minute_analysis",
                        )
                    )

                # レンジブレイクの検知
                recent_high = max(highs[-3:])
                recent_low = min(lows[-3:])

                if highs[-1] > recent_high * 1.001:  # 上昇ブレイク
                    signals.append(
                        self._create_signal(
                            direction=0.7,
                            confidence=0.75,
                            strength=0.6,
                            signal_type="breakout_up",
                            source="minute_analysis",
                        )
                    )

                elif lows[-1] < recent_low * 0.999:  # 下降ブレイク
                    signals.append(
                        self._create_signal(
                            direction=-0.7,
                            confidence=0.75,
                            strength=0.6,
                            signal_type="breakout_down",
                            source="minute_analysis",
                        )
                    )

        except Exception as e:
            self.logger.debug(f"1分足パターン分析エラー: {e}")

        return signals

    def _apply_high_frequency_filtering(
        self, signals: list, data: pd.DataFrame, current_index: int
    ) -> list:
        """高頻度取引向けのフィルタリング"""
        if not signals:
            return signals

        filtered_signals = []

        for signal in signals:
            # 高頻度向けの基準を緩和
            min_confidence = 0.2  # 通常の0.3より緩和
            min_strength = 0.2  # 通常の0.3より緩和

            # 信号の品質チェック
            confidence = getattr(signal, "confidence", 0)
            strength = getattr(signal, "strength", 0)

            if confidence >= min_confidence and strength >= min_strength:
                filtered_signals.append(signal)

        # 最大信号数を制限（高頻度でも多すぎないように）
        max_signals = getattr(self.config, "max_signals_per_bar", 5)
        if len(filtered_signals) > max_signals:
            # 品質の高い順にソートして制限
            filtered_signals.sort(
                key=lambda s: (s.confidence + s.strength), reverse=True
            )
            filtered_signals = filtered_signals[:max_signals]

        return filtered_signals

    def _create_signal(
        self,
        direction: float,
        confidence: float,
        strength: float,
        signal_type: str,
        source: str,
    ) -> object:
        """高頻度向けの信号オブジェクト作成"""
        try:
            from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
                ActionSignal,
            )

            signal = ActionSignal(
                direction=direction,
                confidence=confidence,
                strength=strength,
                signal_type=signal_type,
                source=source,
                timestamp=datetime.now(),
            )

            return signal

        except Exception as e:
            self.logger.error(f"信号作成エラー: {e}")
            return None


def test_high_frequency_implementation():
    """高頻度取引実装のテスト"""
    print("=== 高頻度取引実装テスト ===")

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

        # 高頻度ActionSignalGuideの初期化
        print("\n2. 高頻度ActionSignalGuide初期化...")
        hf_guide = HighFrequencyActionSignalGuide()

        # テスト実行
        print("\n3. 高頻度信号生成テスト...")
        test_indices = np.linspace(0, len(test_df) - 1, 10, dtype=int)
        test_indices = [min(idx, len(test_df) - 1) for idx in test_indices]

        total_signals = 0
        processing_times = []

        for i, idx in enumerate(test_indices):
            try:
                start_time = datetime.now()

                # 高頻度信号生成
                signals = hf_guide.generate_high_frequency_signals(test_df, idx)

                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                processing_times.append(processing_time)

                signal_count = len(signals) if signals else 0
                total_signals += signal_count

                print(f"   Index {idx}: {signal_count} signals")

                # 最初の数信号の詳細を表示
                if signals and i < 3:
                    for j, signal in enumerate(signals[:2]):
                        print(
                            f"     Signal {j+1}: {signal.signal_type}, dir={signal.direction:.3f}, conf={signal.confidence:.3f}"
                        )

            except Exception as e:
                print(f"   Error at index {idx}: {e}")

        # 結果集計
        avg_signals = total_signals / len(test_indices)
        avg_time = np.mean(processing_times) if processing_times else 0

        print("\n=== 高頻度実装結果 ===")
        print(f"   - テストポイント: {len(test_indices)}")
        print(f"   - 総信号数: {total_signals}")
        print(f"   - 平均信号/テスト: {avg_signals:.2f}")
        print(f"   - 平均処理時間: {avg_time:.4f}s")

        # 改善効果の評価
        print("\n=== 改善効果評価 ===")
        if avg_signals > 1.0:
            print("✓ 高頻度取引要件を満たしています（平均1信号以上/テスト）")
        elif avg_signals > 0.5:
            print("△ 中程度の改善が見られます")
        else:
            print("✗ さらなる改善が必要です")

        # 推奨事項
        print("\n=== 高頻度取引実現のための推奨事項 ===")
        print("1. より多くの短期指標を追加（RSI, MACD, 移動平均）")
        print("2. 複数タイムフレーム分析の実装")
        print("3. 市場マイクロストラクチャの活用")
        print("4. 機械学習ベースの信号生成")
        print("5. リアルタイムデータ処理の最適化")

        return True

    except Exception as e:
        print("\n=== テスト失敗 ===")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """メイン実行関数"""
    success = test_high_frequency_implementation()

    if success:
        print("\n=== 高頻度取引実装テスト完了 ===")
        print("ActionSignalGuideの高頻度取引対応が実装されました。")
    else:
        print("\n=== テスト失敗 ===")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
