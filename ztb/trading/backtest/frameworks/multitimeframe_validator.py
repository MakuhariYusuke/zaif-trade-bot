"""
Phase 3-1: シグナル品質向上 - マルチタイムフレーム確認

複数時間軸でのシグナル整合性を確認し、信頼性の高いシグナルのみをフィルタリングします。
既存のバックテストフレームワークと連携して品質向上を実現します。
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from ztb.utils.performance_profiler import PerformanceProfiler

class TimeFrame(Enum):
    """時間軸定義"""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"

@dataclass
class MultiTimeFrameSignal:
    """マルチタイムフレームシグナル"""

    primary_timeframe: TimeFrame
    primary_signal: dict[str, Any]
    aligned_timeframes: list[TimeFrame]
    consistency_score: float
    alignment_strength: float
    timestamp: datetime

    @property
    def is_consistent(self) -> bool:
        """整合性があるかを判定"""
        return self.consistency_score > 0.7 and len(self.aligned_timeframes) >= 2

@dataclass
class TimeFrameAlignmentResult:
    """時間軸整合性結果"""

    timeframe: TimeFrame
    signal_type: str
    strength: float
    confidence: float
    is_aligned: bool

class MultiTimeFrameValidator:
    """マルチタイムフレーム検証器"""

    def __init__(self):
        self.profiler = PerformanceProfiler()

        # 時間軸の階層関係を定義
        self.timeframe_hierarchy = {
            TimeFrame.M1: [TimeFrame.M5, TimeFrame.M15],
            TimeFrame.M5: [TimeFrame.M15, TimeFrame.M30, TimeFrame.H1],
            TimeFrame.M15: [TimeFrame.M30, TimeFrame.H1, TimeFrame.H4],
            TimeFrame.M30: [TimeFrame.H1, TimeFrame.H4],
            TimeFrame.H1: [TimeFrame.H4, TimeFrame.D1],
            TimeFrame.H4: [TimeFrame.D1, TimeFrame.W1],
            TimeFrame.D1: [TimeFrame.W1],
            TimeFrame.W1: [],
        }

    def validate_signal_consistency(
        self,
        signal: dict[str, Any],
        market_data_dict: dict[TimeFrame, pd.DataFrame],
        primary_timeframe: TimeFrame = TimeFrame.H1,
    ) -> MultiTimeFrameSignal:
        """
        シグナルのマルチタイムフレーム整合性を検証

        Args:
            signal: 検証対象のシグナル
            market_data_dict: 時間軸ごとの市場データ
            primary_timeframe: プライマリ時間軸

        Returns:
            MultiTimeFrameSignal: マルチタイムフレーム検証結果
        """
        # タイムスタンプの取得（無効な入力に対応）
        try:
            timestamp = pd.to_datetime(signal.get("timestamp", datetime.now()))
        except (ValueError, TypeError):
            timestamp = datetime.now()

        # 各時間軸でのシグナルを評価
        alignment_results = []
        aligned_timeframes = []

        for timeframe in self.timeframe_hierarchy.get(primary_timeframe, []):
            if timeframe in market_data_dict:
                alignment = self._evaluate_timeframe_alignment(
                    signal, market_data_dict[timeframe], timeframe, timestamp
                )
                alignment_results.append(alignment)

                if alignment.is_aligned:
                    aligned_timeframes.append(timeframe)

        # 整合性スコアを計算
        consistency_score = self._calculate_consistency_score(alignment_results)
        alignment_strength = len(aligned_timeframes) / max(
            len(self.timeframe_hierarchy.get(primary_timeframe, [])), 1
        )

        return MultiTimeFrameSignal(
            primary_timeframe=primary_timeframe,
            primary_signal=signal,
            aligned_timeframes=aligned_timeframes,
            consistency_score=consistency_score,
            alignment_strength=alignment_strength,
            timestamp=timestamp,
        )

    def _evaluate_timeframe_alignment(
        self,
        primary_signal: dict[str, Any],
        timeframe_data: pd.DataFrame,
        timeframe: TimeFrame,
        signal_timestamp: datetime,
    ) -> TimeFrameAlignmentResult:
        """
        特定の時間軸でのシグナル整合性を評価
        """
        if timeframe_data.empty:
            return TimeFrameAlignmentResult(
                timeframe=timeframe,
                signal_type="unknown",
                strength=0.0,
                confidence=0.0,
                is_aligned=False,
            )

        # シグナル発生時刻に最も近いデータポイントを見つける
        closest_idx = self._find_closest_timestamp(timeframe_data, signal_timestamp)

        if closest_idx is None:
            return TimeFrameAlignmentResult(
                timeframe=timeframe,
                signal_type="unknown",
                strength=0.0,
                confidence=0.0,
                is_aligned=False,
            )

        # その時間軸でのトレンドとシグナルを比較
        trend_signal = self._extract_trend_signal(
            timeframe_data, closest_idx, timeframe
        )

        primary_signal_type = primary_signal.get("action", "hold")

        # シグナルタイプの整合性を評価
        is_aligned = self._check_signal_alignment(
            primary_signal_type, trend_signal["type"]
        )
        strength = trend_signal["strength"]
        confidence = trend_signal["confidence"]

        return TimeFrameAlignmentResult(
            timeframe=timeframe,
            signal_type=trend_signal["type"],
            strength=strength,
            confidence=confidence,
            is_aligned=is_aligned,
        )

    def _find_closest_timestamp(
        self, data: pd.DataFrame, target_timestamp: datetime
    ) -> int | None:
        """最も近いタイムスタンプのインデックスを見つける"""
        if data.index.empty:
            return None

        # タイムスタンプの差を計算
        time_diffs = np.abs(data.index - target_timestamp)

        # 最小の差を持つインデックスを返す
        min_idx = time_diffs.argmin()

        # 差が大きすぎる場合はNone
        if time_diffs[min_idx] > timedelta(hours=4):  # 4時間以内のデータのみ使用
            return None

        return min_idx

    def _extract_trend_signal(
        self, data: pd.DataFrame, idx: int, timeframe: TimeFrame
    ) -> dict[str, Any]:
        """
        指定された時間軸と位置でのトレンドシグナルを抽出
        """
        if idx < 10:  # 十分な過去データが必要
            return {"type": "hold", "strength": 0.0, "confidence": 0.0}

        # 時間軸に応じたルックバック期間を設定
        lookback_periods = {
            TimeFrame.M1: 20,
            TimeFrame.M5: 12,
            TimeFrame.M15: 8,
            TimeFrame.M30: 6,
            TimeFrame.H1: 5,
            TimeFrame.H4: 4,
            TimeFrame.D1: 3,
            TimeFrame.W1: 2,
        }

        lookback = lookback_periods.get(timeframe, 5)
        start_idx = max(0, idx - lookback)

        trend_data = data.iloc[start_idx : idx + 1]

        # トレンド強度を計算
        trend_strength = self._calculate_trend_strength(trend_data)

        # シグナルタイプを決定
        if trend_strength > 0.002:  # 強気トレンド
            signal_type = "buy"
            confidence = min(abs(trend_strength) * 100, 1.0)
        elif trend_strength < -0.002:  # 弱気トレンド
            signal_type = "sell"
            confidence = min(abs(trend_strength) * 100, 1.0)
        else:  # 横ばい
            signal_type = "hold"
            confidence = 0.5

        return {
            "type": signal_type,
            "strength": abs(trend_strength),
            "confidence": confidence,
        }

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """トレンド強度を計算"""
        if len(data) < 3:
            return 0.0

        # 線形回帰によるトレンド計算
        x = np.arange(len(data))
        y = data["close"].values

        try:
            slope = np.polyfit(x, y, 1)[0]
            # トレンド強度を価格変動の割合として正規化
            avg_price = np.mean(y)
            return slope / avg_price if avg_price != 0 else 0.0
        except np.RankWarning:
            return 0.0

    def _check_signal_alignment(
        self, primary_signal: str, timeframe_signal: str
    ) -> bool:
        """プライマリシグナルと時間軸シグナルの整合性をチェック"""
        # 同じ方向のシグナルは整合性あり
        if primary_signal == timeframe_signal:
            return True

        # ホールドは中立的
        if primary_signal == "hold" or timeframe_signal == "hold":
            return True

        # 逆方向のシグナルは不整合
        return False

    def _calculate_consistency_score(
        self, alignment_results: list[TimeFrameAlignmentResult]
    ) -> float:
        """整合性スコアを計算"""
        if not alignment_results:
            return 0.0

        aligned_count = sum(1 for result in alignment_results if result.is_aligned)
        total_count = len(alignment_results)

        # 整合性のあるシグナルの割合
        alignment_ratio = aligned_count / total_count

        # 平均コンフィデンス
        avg_confidence = np.mean([result.confidence for result in alignment_results])

        # 平均強度
        avg_strength = np.mean([result.strength for result in alignment_results])

        # 総合スコア
        consistency_score = (
            alignment_ratio * 0.5 + avg_confidence * 0.3 + avg_strength * 0.2
        )

        return min(max(consistency_score, 0.0), 1.0)

    def filter_consistent_signals(
        self,
        signals: list[dict[str, Any]],
        market_data_dict: dict[TimeFrame, pd.DataFrame],
        min_consistency_threshold: float = 0.7,
        min_alignment_strength: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        整合性の高いシグナルのみをフィルタリング

        Args:
            signals: フィルタリング対象のシグナルリスト
            market_data_dict: 時間軸ごとの市場データ
            min_consistency_threshold: 最小整合性スコア
            min_alignment_strength: 最小整合強度

        Returns:
            list[dict[str, Any]]: フィルタリングされたシグナルリスト
        """
        filtered_signals = []

        for signal in signals:
            try:
                validation_result = self.validate_signal_consistency(
                    signal, market_data_dict
                )

                if (
                    validation_result.consistency_score >= min_consistency_threshold
                    and validation_result.alignment_strength >= min_alignment_strength
                ):
                    # 整合性情報をシグナルに追加
                    enhanced_signal = signal.copy()
                    enhanced_signal.update(
                        {
                            "multitimeframe_consistency": validation_result.consistency_score,
                            "multitimeframe_alignment": validation_result.alignment_strength,
                            "aligned_timeframes": [
                                tf.value for tf in validation_result.aligned_timeframes
                            ],
                        }
                    )
                    filtered_signals.append(enhanced_signal)

            except Exception as e:
                # エラーが発生したシグナルはスキップ
                print(f"Warning: Failed to validate signal: {e}")
                continue

        return filtered_signals

    def get_consistency_statistics(self) -> dict[str, Any]:
        """整合性統計を取得"""
        # このメソッドは実際の使用履歴に基づいて統計を計算
        # 簡易実装として基本的な情報を返す
        return {
            "description": "マルチタイムフレーム整合性検証システム",
            "supported_timeframes": [tf.value for tf in TimeFrame],
            "default_consistency_threshold": 0.7,
            "default_alignment_threshold": 0.5,
        }
