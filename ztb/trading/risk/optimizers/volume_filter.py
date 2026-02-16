"""
Phase 3-1: シグナル品質向上 - ボリュームフィルタ

出来高情報を活用してシグナルの信頼性を向上させます。
異常な出来高パターンや確認不足のシグナルをフィルタリングします。
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ztb.utils.performance_profiler import PerformanceProfiler


class VolumePattern(Enum):
    """出来高パターン"""

    NORMAL = "normal"
    HIGH_VOLUME = "high_volume"
    LOW_VOLUME = "low_volume"
    SPIKE = "spike"
    DRY_UP = "dry_up"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"


@dataclass
class VolumeAnalysisResult:
    """出来高分析結果"""

    pattern: VolumePattern
    volume_ratio: float  # 平均出来高に対する比率
    trend: str  # 'increasing', 'decreasing', 'stable'
    confirmation_strength: float  # 0-1の確認強度
    risk_level: str  # 'low', 'medium', 'high'

    @property
    def is_confirming(self) -> bool:
        """シグナルを確認しているか"""
        return self.confirmation_strength > 0.6

    @property
    def is_high_risk(self) -> bool:
        """高リスクか"""
        return self.risk_level == "high"


@dataclass
class VolumeFilterCriteria:
    """ボリュームフィルタ基準"""

    min_volume_ratio: float = 1.2  # 最低出来高比率
    max_volume_ratio: float = 5.0  # 最大出来高比率（異常検知）
    require_volume_confirmation: bool = True
    volume_trend_alignment: bool = True  # 出来高トレンドの整合性要求
    spike_detection_threshold: float = 3.0  # スパイク検知閾値

    # 動的調整
    adaptive_filtering: bool = True
    market_regime_adjustment: bool = True


class VolumeFilter:
    """ボリュームフィルタ"""

    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.filter_criteria = VolumeFilterCriteria()
        self.volume_history: List[VolumeAnalysisResult] = []
        self.max_history_size = 1000  # メモリ管理のため履歴サイズを制限

    def analyze_volume_pattern(
        self,
        market_data: pd.DataFrame,
        signal_timestamp: datetime,
        lookback_periods: int = 20,
    ) -> VolumeAnalysisResult:
        """
        指定時刻の出来高パターンを分析

        Args:
            market_data: 市場データ（volumeカラムを含む）
            signal_timestamp: シグナル発生時刻
            lookback_periods: ルックバック期間

        Returns:
            VolumeAnalysisResult: 出来高分析結果
        """
        if "volume" not in market_data.columns or market_data.empty:
            return VolumeAnalysisResult(
                pattern=VolumePattern.NORMAL,
                volume_ratio=1.0,
                trend="stable",
                confirmation_strength=0.5,
                risk_level="medium",
            )

        # 指定時刻の出来高データを取得
        current_volume = self._get_volume_at_timestamp(market_data, signal_timestamp)

        if current_volume is None:
            return VolumeAnalysisResult(
                pattern=VolumePattern.NORMAL,
                volume_ratio=1.0,
                trend="stable",
                confirmation_strength=0.5,
                risk_level="medium",
            )

        # 過去データの出来高平均を計算
        historical_data = self._get_historical_volume_data(
            market_data, signal_timestamp, lookback_periods
        )
        avg_volume = (
            historical_data["volume"].mean()
            if not historical_data.empty
            else current_volume
        )

        if avg_volume == 0:
            volume_ratio = 1.0
        else:
            volume_ratio = current_volume / avg_volume

        # 出来高パターンを判定
        pattern = self._classify_volume_pattern(volume_ratio, historical_data)

        # 出来高トレンドを分析
        trend = self._analyze_volume_trend(historical_data)

        # 確認強度を計算
        confirmation_strength = self._calculate_confirmation_strength(
            pattern, volume_ratio, trend
        )

        # リスクレベルを評価
        risk_level = self._assess_risk_level(pattern, volume_ratio, trend)

        result = VolumeAnalysisResult(
            pattern=pattern,
            volume_ratio=volume_ratio,
            trend=trend,
            confirmation_strength=confirmation_strength,
            risk_level=risk_level,
        )

        # 履歴に保存（メモリ管理）
        self.volume_history.append(result)
        if len(self.volume_history) > self.max_history_size:
            # 古いデータを削除してメモリを節約
            self.volume_history = self.volume_history[-self.max_history_size // 2 :]

        return result

    def _get_volume_at_timestamp(
        self, market_data: pd.DataFrame, timestamp: datetime
    ) -> Optional[float]:
        """指定時刻の出来高を取得"""
        if market_data.index.empty:
            return None

        # 最も近いタイムスタンプを見つける
        time_diffs = market_data.index - timestamp
        closest_idx = np.abs(time_diffs).argmin()

        # 差が大きすぎる場合はNone
        time_diff = abs(time_diffs[closest_idx])
        if time_diff > timedelta(hours=1):  # 1時間以内のデータのみ使用
            return None

        return market_data.iloc[closest_idx]["volume"]

    def _get_historical_volume_data(
        self,
        market_data: pd.DataFrame,
        signal_timestamp: datetime,
        lookback_periods: int,
    ) -> pd.DataFrame:
        """過去の出来高データを取得"""
        if market_data.index.empty:
            return pd.DataFrame()

        # 指定時刻より前のデータを取得
        mask = market_data.index <= signal_timestamp
        historical_data = market_data[mask]

        if len(historical_data) == 0:
            return pd.DataFrame()

        # 指定期間分のデータを返す
        return historical_data.tail(lookback_periods)

    def _classify_volume_pattern(
        self, volume_ratio: float, historical_data: pd.DataFrame
    ) -> VolumePattern:
        """出来高パターンを分類"""
        if volume_ratio >= self.filter_criteria.spike_detection_threshold:
            return VolumePattern.SPIKE
        elif volume_ratio >= 2.0:
            return VolumePattern.HIGH_VOLUME
        elif volume_ratio <= 0.5:
            return VolumePattern.LOW_VOLUME
        elif volume_ratio <= 0.2:
            return VolumePattern.DRY_UP
        else:
            # トレンドパターンの分析
            if not historical_data.empty and len(historical_data) >= 5:
                recent_trend = self._calculate_volume_trend_slope(
                    historical_data.tail(5)
                )

                if recent_trend > 0.1:
                    return VolumePattern.ACCUMULATION
                elif recent_trend < -0.1:
                    return VolumePattern.DISTRIBUTION

            return VolumePattern.NORMAL

    def _calculate_volume_trend_slope(self, data: pd.DataFrame) -> float:
        """出来高のトレンド傾きを計算"""
        if len(data) < 2:
            return 0.0

        x = np.arange(len(data))
        y = data["volume"].values

        try:
            slope = np.polyfit(x, y, 1)[0]
            # 平均出来高に対する相対的な傾き
            avg_volume = np.mean(y)
            return slope / avg_volume if avg_volume != 0 else 0.0
        except Exception:
            return 0.0

    def _analyze_volume_trend(self, historical_data: pd.DataFrame) -> str:
        """出来高トレンドを分析"""
        if historical_data.empty or len(historical_data) < 5:
            return "stable"

        # 最近5期間のトレンドを計算
        recent_data = historical_data.tail(5)
        slope = self._calculate_volume_trend_slope(recent_data)

        if slope > 0.05:
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        else:
            return "stable"

    def _calculate_confirmation_strength(
        self, pattern: VolumePattern, volume_ratio: float, trend: str
    ) -> float:
        """確認強度を計算"""
        base_strength = 0.5

        # パターンによる調整
        pattern_multipliers = {
            VolumePattern.SPIKE: 0.9,
            VolumePattern.HIGH_VOLUME: 0.8,
            VolumePattern.NORMAL: 0.6,
            VolumePattern.LOW_VOLUME: 0.3,
            VolumePattern.DRY_UP: 0.2,
            VolumePattern.ACCUMULATION: 0.7,
            VolumePattern.DISTRIBUTION: 0.7,
        }

        base_strength = pattern_multipliers.get(pattern, 0.5)

        # 出来高比率による調整
        if volume_ratio > 2.0:
            ratio_bonus = min((volume_ratio - 2.0) * 0.1, 0.3)
            base_strength = min(base_strength + ratio_bonus, 1.0)
        elif volume_ratio < 0.5:
            ratio_penalty = (0.5 - volume_ratio) * 0.5
            base_strength = max(base_strength - ratio_penalty, 0.0)

        # トレンドによる調整
        if trend == "increasing":
            base_strength = min(base_strength + 0.1, 1.0)
        elif trend == "decreasing":
            base_strength = max(base_strength - 0.1, 0.0)

        return base_strength

    def _assess_risk_level(
        self, pattern: VolumePattern, volume_ratio: float, trend: str
    ) -> str:
        """リスクレベルを評価"""
        risk_score = 0.0

        # パターンによるリスク評価
        pattern_risks = {
            VolumePattern.SPIKE: 0.8,  # 急激な出来高増加は要注意
            VolumePattern.HIGH_VOLUME: 0.6,
            VolumePattern.NORMAL: 0.3,
            VolumePattern.LOW_VOLUME: 0.4,
            VolumePattern.DRY_UP: 0.7,  # 出来高枯渇はリスク高い
            VolumePattern.ACCUMULATION: 0.2,
            VolumePattern.DISTRIBUTION: 0.5,
        }

        risk_score = pattern_risks.get(pattern, 0.3)

        # 出来高比率による調整
        if volume_ratio > 3.0:
            risk_score = min(risk_score + 0.3, 1.0)  # 極端な出来高は高リスク
        elif volume_ratio < 0.3:
            risk_score = min(risk_score + 0.2, 1.0)  # 極端な出来高減少も要注意

        # リスクレベル判定
        if risk_score >= 0.7:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        else:
            return "low"

    def should_filter_signal(
        self,
        signal: Dict[str, Any],
        market_data: pd.DataFrame,
        custom_criteria: Optional[VolumeFilterCriteria] = None,
    ) -> Tuple[bool, str, VolumeAnalysisResult]:
        """
        シグナルをフィルタリングすべきかを判定

        Returns:
            Tuple[bool, str, VolumeAnalysisResult]: (フィルタリング可否, 理由, 分析結果)
        """
        criteria = custom_criteria or self.filter_criteria

        # 出来高分析
        try:
            signal_timestamp = pd.to_datetime(signal.get("timestamp", datetime.now()))
        except (ValueError, TypeError):
            signal_timestamp = datetime.now()

        analysis_result = self.analyze_volume_pattern(market_data, signal_timestamp)

        # 基準チェック
        if criteria.require_volume_confirmation and not analysis_result.is_confirming:
            return (
                True,
                f"出来高確認が不十分です: {analysis_result.confirmation_strength:.2f}",
                analysis_result,
            )

        if analysis_result.volume_ratio < criteria.min_volume_ratio:
            return (
                True,
                f"出来高が低すぎます: {analysis_result.volume_ratio:.2f}",
                analysis_result,
            )

        if analysis_result.volume_ratio > criteria.max_volume_ratio:
            return (
                True,
                f"出来高が異常です: {analysis_result.volume_ratio:.2f}",
                analysis_result,
            )

        if analysis_result.is_high_risk:
            return (
                True,
                f"高リスクの出来高パターンです: {analysis_result.pattern.value}",
                analysis_result,
            )

        if criteria.volume_trend_alignment:
            signal_type = signal.get("action", "hold")
            if not self._check_volume_trend_alignment(
                signal_type, analysis_result.trend
            ):
                return (
                    True,
                    f"出来高トレンドがシグナルと不整合です: {analysis_result.trend}",
                    analysis_result,
                )

        return False, "シグナルは基準を満たしています", analysis_result

    def _check_volume_trend_alignment(
        self, signal_type: str, volume_trend: str
    ) -> bool:
        """シグナルタイプと出来高トレンドの整合性をチェック"""
        if signal_type == "buy":
            # 買いシグナル時は出来高が増加傾向が好ましい
            return volume_trend in ["increasing", "stable"]
        elif signal_type == "sell":
            # 売りシグナル時は出来高が減少傾向でも可
            return True  # 売りシグナルは比較的寛容
        else:
            return True  # ホールドは常にOK

    def update_criteria_based_on_market(self, market_data: pd.DataFrame):
        """市場状況に応じてフィルタ基準を更新"""
        if not self.filter_criteria.adaptive_filtering:
            return

        # 最近の市場ボラティリティを計算
        if len(market_data) >= 20:
            recent_returns = market_data["close"].pct_change().tail(20)
            volatility = recent_returns.std()

            # 高ボラティリティ時は基準を厳しく
            if volatility > 0.03:
                self.filter_criteria.min_volume_ratio = min(
                    1.5, self.filter_criteria.min_volume_ratio + 0.1
                )
            # 低ボラティリティ時は基準を緩く
            elif volatility < 0.01:
                self.filter_criteria.min_volume_ratio = max(
                    1.0, self.filter_criteria.min_volume_ratio - 0.1
                )

    def get_volume_statistics(self) -> Dict[str, Any]:
        """出来高統計を取得"""
        if not self.volume_history:
            return {}

        patterns = [result.pattern for result in self.volume_history]
        ratios = [result.volume_ratio for result in self.volume_history]
        confirmations = [result.confirmation_strength for result in self.volume_history]

        return {
            "total_analyses": len(self.volume_history),
            "pattern_distribution": {
                pattern.value: patterns.count(pattern) for pattern in set(patterns)
            },
            "average_volume_ratio": np.mean(ratios),
            "median_volume_ratio": np.median(ratios),
            "average_confirmation": np.mean(confirmations),
            "high_risk_signals": sum(
                1 for result in self.volume_history if result.is_high_risk
            ),
            "confirming_signals": sum(
                1 for result in self.volume_history if result.is_confirming
            ),
        }
