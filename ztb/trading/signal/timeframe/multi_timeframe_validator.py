"""
Multi-Timeframe Signal Validator for Phase 4

複数タイムフレームでのシグナル整合性検証
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ztb.trading.signal.timeframe.adaptive_timeframe_manager import (
    AdaptiveTimeframeManager,
)
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MultiTimeframeSignalValidator:
    """
    マルチタイムフレームシグナル検証器

    複数タイムフレームでのシグナル一致性を検証し、
    信頼性の高いシグナルのみを通過させる
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize multi-timeframe signal validator

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.timeframe_manager = AdaptiveTimeframeManager(
            self.config.get("timeframe_config", {})
        )

        # 検証パラメータ
        self.consistency_threshold = self.config.get(
            "consistency_threshold", 0.67
        )  # 67%一致で有効
        self.max_timeframes = self.config.get(
            "max_timeframes", 3
        )  # 最大確認タイムフレーム数
        self.damping_factor = self.config.get(
            "damping_factor", 0.5
        )  # 不一致時の減衰係数

        logger.info("MultiTimeframeSignalValidator initialized")

    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "consistency_threshold": 0.67,
            "max_timeframes": 3,
            "damping_factor": 0.5,
            "timeframe_config": {},
        }

    def validate_signal_consistency(
        self,
        signal: float,
        market_data: Dict[str, pd.DataFrame],
        base_timeframe: str = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        複数タイムフレームでのシグナル一致性を検証

        Args:
            signal: 基準となるシグナルスコア
            market_data: タイムフレーム別市場データ
            base_timeframe: 基準タイムフレーム

        Returns:
            Tuple[float, Dict]: (検証後シグナル, 検証結果詳細)
        """
        if not market_data or len(market_data) < 2:
            # データが不足している場合は元のシグナルを返す
            return signal, {"status": "insufficient_data", "confidence": 0.5}

        try:
            # 基準タイムフレームを決定
            if base_timeframe is None:
                base_timeframe = list(market_data.keys())[0]

            # 確認対象のタイムフレーム階層を取得
            timeframe_hierarchy = self.timeframe_manager.get_timeframe_hierarchy(
                base_timeframe
            )

            # 利用可能なタイムフレームに制限
            available_timeframes = [
                tf for tf in timeframe_hierarchy if tf in market_data
            ]
            validation_timeframes = available_timeframes[: self.max_timeframes]

            if len(validation_timeframes) < 2:
                return signal, {"status": "insufficient_timeframes", "confidence": 0.5}

            # 各タイムフレームでシグナルを計算
            timeframe_signals = {}
            for tf in validation_timeframes:
                tf_signal = self._calculate_timeframe_signal(market_data[tf], signal)
                timeframe_signals[tf] = tf_signal

            # シグナル一致性を評価
            consistency_score, consistency_details = self._evaluate_consistency(
                signal, timeframe_signals
            )

            # 一致度に基づいてシグナルを調整
            validated_signal = self._adjust_signal_by_consistency(
                signal, consistency_score, consistency_details
            )

            # 検証結果をまとめる
            validation_result = {
                "status": "validated",
                "consistency_score": consistency_score,
                "confidence": consistency_score,
                "timeframe_signals": timeframe_signals,
                "validation_timeframes": validation_timeframes,
                "base_timeframe": base_timeframe,
                "details": consistency_details,
            }

            logger.debug(
                f"Signal validation: {signal:.2f} -> {validated_signal:.2f} "
                f"(consistency: {consistency_score:.2f})"
            )

            return validated_signal, validation_result

        except Exception as e:
            logger.warning(f"Error in signal validation: {e}")
            return signal, {"status": "error", "error": str(e), "confidence": 0.0}

    def _calculate_timeframe_signal(
        self, df: pd.DataFrame, reference_signal: float
    ) -> float:
        """
        指定タイムフレームでのシグナルを計算

        Args:
            df: タイムフレームの市場データ
            reference_signal: 基準シグナル（スケーリング参考用）

        Returns:
            float: 計算されたシグナル
        """
        if len(df) < 10:
            return 50.0  # 中立シグナル

        try:
            # 簡易的なシグナル計算（実際の実装ではSignalQualityScorerを使用）
            close_prices = df["close"].values

            # RSI計算
            rsi = self._calculate_rsi(close_prices, period=14)

            # トレンド強度計算
            trend_strength = self._calculate_simple_trend_strength(
                close_prices, window=20
            )

            # ボラティリティ計算
            volatility = np.std(close_prices[-20:]) / np.mean(close_prices[-20:])

            # シグナル合成（簡易版）
            signal = 50.0  # ベースは中立

            # RSIによる調整
            if rsi < 30:
                signal += 20  # 買いシグナル
            elif rsi > 70:
                signal -= 20  # 売りシグナル

            # トレンドによる調整
            signal += trend_strength * 10

            # ボラティリティによる調整
            if volatility > 0.05:  # 高ボラティリティ
                signal *= 0.8  # シグナルを弱める

            return max(0, min(100, signal))

        except Exception as e:
            logger.warning(f"Error calculating timeframe signal: {e}")
            return 50.0

    def _evaluate_consistency(
        self, base_signal: float, timeframe_signals: Dict[str, float]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        シグナルの一致性を評価

        Args:
            base_signal: 基準シグナル
            timeframe_signals: タイムフレーム別シグナル

        Returns:
            Tuple[float, Dict]: (一致度スコア, 詳細情報)
        """
        if not timeframe_signals:
            return 0.0, {}

        try:
            signals = list(timeframe_signals.values())

            # シグナル方向の一致性を評価
            base_direction = 1 if base_signal > 60 else (-1 if base_signal < 40 else 0)
            directions = []

            for signal in signals:
                if signal > 60:
                    directions.append(1)  # 買い
                elif signal < 40:
                    directions.append(-1)  # 売り
                else:
                    directions.append(0)  # 中立

            # 方向一致数をカウント
            consistent_count = sum(1 for d in directions if d == base_direction)
            total_count = len(directions)

            # 一致度スコア計算
            consistency_score = consistent_count / total_count if total_count > 0 else 0

            # シグナル強度の標準偏差も考慮
            signal_std = np.std(signals) if len(signals) > 1 else 0
            strength_consistency = max(
                0, 1 - signal_std / 50
            )  # 標準偏差が大きいほど一致度低下

            # 最終一致度スコア
            final_consistency = (consistency_score + strength_consistency) / 2

            details = {
                "base_direction": base_direction,
                "directions": directions,
                "consistent_count": consistent_count,
                "total_count": total_count,
                "signal_std": signal_std,
                "strength_consistency": strength_consistency,
            }

            return final_consistency, details

        except Exception as e:
            logger.warning(f"Error evaluating consistency: {e}")
            return 0.0, {"error": str(e)}

    def _adjust_signal_by_consistency(
        self,
        original_signal: float,
        consistency_score: float,
        consistency_details: Dict[str, Any],
    ) -> float:
        """
        一致度に基づいてシグナルを調整

        Args:
            original_signal: 元のシグナル
            consistency_score: 一致度スコア
            consistency_details: 一致性の詳細

        Returns:
            float: 調整後のシグナル
        """
        if consistency_score >= self.consistency_threshold:
            # 一致度が高い場合はシグナルを強化
            confidence_boost = (consistency_score - self.consistency_threshold) / (
                1 - self.consistency_threshold
            )
            adjusted_signal = (
                original_signal + (original_signal - 50) * confidence_boost * 0.2
            )
        else:
            # 一致度が低い場合はシグナルを弱める
            damping_factor = self.damping_factor * (
                1 - consistency_score / self.consistency_threshold
            )
            adjusted_signal = 50.0 + (original_signal - 50.0) * (1 - damping_factor)

        return max(0, min(100, adjusted_signal))

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """簡易RSI計算"""
        try:
            from ztb.features.generators.technical.momentum.rsi import compute_rsi

            df = pd.DataFrame({"close": prices})
            rsi_series = compute_rsi(df, period=period)
            last_val = rsi_series.iloc[-1]
            return float(last_val) if not pd.isna(last_val) else 50.0
        except:
            return 50.0

    def _calculate_simple_trend_strength(
        self, prices: np.ndarray, window: int = 20
    ) -> float:
        """簡易トレンド強度計算"""
        if len(prices) < window:
            return 0.0

        try:
            # 単純な傾き計算
            x = np.arange(window)
            y = prices[-window:]
            slope = np.polyfit(x, y, 1)[0]

            # 正規化（-1 to 1）
            avg_price = np.mean(y)
            normalized_slope = slope / (avg_price * 0.001)  # 0.1%変化を基準

            return max(-1, min(1, normalized_slope))
        except:
            return 0.0

    def get_validation_summary(
        self, validation_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        複数検証結果のサマリーを取得

        Args:
            validation_results: 検証結果のリスト

        Returns:
            Dict: サマリー情報
        """
        if not validation_results:
            return {}

        try:
            total_validations = len(validation_results)
            successful_validations = sum(
                1 for r in validation_results if r.get("status") == "validated"
            )

            consistency_scores = [
                r.get("consistency_score", 0)
                for r in validation_results
                if r.get("status") == "validated"
            ]

            avg_consistency = np.mean(consistency_scores) if consistency_scores else 0
            consistency_std = np.std(consistency_scores) if consistency_scores else 0

            # シグナル品質分布
            signal_changes = []
            for result in validation_results:
                if "original_signal" in result and "validated_signal" in result:
                    change = abs(result["validated_signal"] - result["original_signal"])
                    signal_changes.append(change)

            avg_signal_change = np.mean(signal_changes) if signal_changes else 0

            return {
                "total_validations": total_validations,
                "successful_validations": successful_validations,
                "success_rate": successful_validations / total_validations
                if total_validations > 0
                else 0,
                "avg_consistency": avg_consistency,
                "consistency_std": consistency_std,
                "avg_signal_change": avg_signal_change,
                "high_consistency_signals": sum(
                    1 for s in consistency_scores if s >= 0.8
                ),
                "low_consistency_signals": sum(
                    1 for s in consistency_scores if s < 0.5
                ),
            }

        except Exception as e:
            logger.warning(f"Error generating validation summary: {e}")
            return {"error": str(e)}
