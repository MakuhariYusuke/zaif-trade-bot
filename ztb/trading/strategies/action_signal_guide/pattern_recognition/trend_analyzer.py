"""
統合トレンド分析システム

ADX, Wave Counting, Dow Theoryを統合した階層的トレンド分析
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base import PatternRecognizer, SignalResult


class HierarchicalTrendAnalyzer(PatternRecognizer):
    """
    階層的トレンド分析システム

    Phase 1: 基本トレンド方向 (Dow Theory)
    Phase 2: トレンド強度 (ADX)
    Phase 3: 詳細波動分析 (Wave Counting) - 強トレンド時のみ
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.pattern_type = "hierarchical_trend"

        # 各分析器の設定
        self.dow_theory_config = self.config.get("dow_theory_config", {})
        self.adx_config = self.config.get(
            "adx_config", {"period": 14, "strong_trend_threshold": 25}
        )
        self.wave_config = self.config.get("wave_config", {})

        # 階層的分析の閾値
        self.strong_trend_threshold = self.config.get("strong_trend_threshold", 25)
        self.enable_wave_analysis = self.config.get("enable_wave_analysis", True)

    def recognize(
        self, data: pd.DataFrame, index: int = -1, **kwargs
    ) -> Optional[SignalResult]:
        """
        階層的トレンド分析を実行

        Args:
            data: 市場データ
            index: 分析対象のインデックス

        Returns:
            統合されたトレンドシグナル
        """
        if not self.validate_data(data) or index < 50:
            return None

        try:
            # Phase 1: 基本トレンド方向 (Dow Theory)
            primary_trend = self._analyze_primary_trend(data, index)
            if not primary_trend:
                return self._create_neutral_signal("トレンド不明")

            # Phase 2: トレンド強度 (ADX)
            trend_strength = self._analyze_trend_strength(data, index)

            # Phase 3: 詳細波動分析 (強トレンド時のみ)
            wave_analysis = None
            if (
                self.enable_wave_analysis
                and trend_strength > self.strong_trend_threshold
            ):
                wave_analysis = self._analyze_wave_patterns(data, index)

            # 統合シグナル生成
            return self._integrate_trend_signals(
                primary_trend, trend_strength, wave_analysis
            )

        except Exception as e:
            return self._create_error_signal(str(e))

    def _analyze_primary_trend(
        self, data: pd.DataFrame, index: int
    ) -> Optional[Dict[str, Any]]:
        """Phase 1: Dow Theoryによる基本トレンド分析"""
        try:
            # 移動平均によるトレンド判定
            ma_short = data["close"].rolling(20).mean().iloc[index]
            ma_long = data["close"].rolling(50).mean().iloc[index]
            current_price = data["close"].iloc[index]

            if pd.isna(ma_short) or pd.isna(ma_long):
                return None

            # トレンド方向判定
            if ma_short > ma_long and current_price > ma_short:
                direction = 1  # 上昇トレンド
                strength = min((ma_short - ma_long) / ma_long, 0.1) * 10  # 0-1正規化
            elif ma_short < ma_long and current_price < ma_short:
                direction = -1  # 下降トレンド
                strength = min((ma_long - ma_short) / ma_long, 0.1) * 10  # 0-1正規化
            else:
                direction = 0  # 横ばい
                strength = 0.0

            return {
                "direction": direction,
                "strength": strength,
                "ma_short": ma_short,
                "ma_long": ma_long,
                "current_price": current_price,
            }

        except Exception:
            return None

    def _analyze_trend_strength(self, data: pd.DataFrame, index: int) -> float:
        """Phase 2: ADXによるトレンド強度分析"""
        try:
            # ADX計算（簡易版）
            high = data["high"]
            low = data["low"]
            close = data["close"]

            # True Range
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Directional Movement
            dm_plus = np.where(
                (high - high.shift(1)) > (low.shift(1) - low), high - high.shift(1), 0
            )
            dm_minus = np.where(
                (low.shift(1) - low) > (high - high.shift(1)), low.shift(1) - low, 0
            )

            # Smoothed averages
            period = self.adx_config.get("period", 14)
            atr = tr.rolling(period).mean()
            di_plus = (pd.Series(dm_plus).rolling(period).mean() / atr * 100).iloc[
                index
            ]
            di_minus = (pd.Series(dm_minus).rolling(period).mean() / atr * 100).iloc[
                index
            ]

            # ADX
            dx = (
                (abs(di_plus - di_minus) / (di_plus + di_minus) * 100)
                .rolling(period)
                .mean()
            )
            adx = dx.iloc[index] if not pd.isna(dx.iloc[index]) else 0.0

            return adx

        except Exception:
            return 0.0

    def _analyze_wave_patterns(
        self, data: pd.DataFrame, index: int
    ) -> Optional[Dict[str, Any]]:
        """Phase 3: Wave Countingによる詳細波動分析"""
        try:
            # 簡易的な波動パターン検出
            recent_data = data.iloc[max(0, index - 50) : index + 1]

            # ピボットポイント検出
            pivots = self._find_pivot_points(recent_data)

            if len(pivots) < 3:
                return None

            # 波動構造分析
            wave_structure = self._analyze_wave_structure(pivots)

            return {
                "pivots": pivots,
                "wave_structure": wave_structure,
                "confidence": len(pivots) / 10.0,
            }  # ピボット数による信頼度

        except Exception:
            return None

    def _find_pivot_points(self, data: pd.DataFrame) -> list:
        """ピボットポイント検出"""
        pivots = []
        lookback = 5

        for i in range(lookback, len(data) - lookback):
            high = data["high"].iloc[i]
            low = data["low"].iloc[i]

            # 高値ピボット
            if all(
                high >= data["high"].iloc[i - j] for j in range(1, lookback + 1)
            ) and all(high >= data["high"].iloc[i + j] for j in range(1, lookback + 1)):
                pivots.append({"type": "high", "index": i, "price": high})

            # 安値ピボット
            if all(
                low <= data["low"].iloc[i - j] for j in range(1, lookback + 1)
            ) and all(low <= data["low"].iloc[i + j] for j in range(1, lookback + 1)):
                pivots.append({"type": "low", "index": i, "price": low})

        return sorted(pivots, key=lambda x: x["index"])

    def _analyze_wave_structure(self, pivots: list) -> Dict[str, Any]:
        """波動構造分析"""
        if len(pivots) < 3:
            return {"pattern": "insufficient_data"}

        # 簡易的な波動カウント
        highs = [p for p in pivots if p["type"] == "high"]
        lows = [p for p in pivots if p["type"] == "low"]

        # トレンド判定
        if len(highs) >= 2 and len(lows) >= 2:
            recent_high = max(p["price"] for p in highs[-2:])
            recent_low = min(p["price"] for p in lows[-2:])

            if recent_high > highs[0]["price"] and recent_low > lows[0]["price"]:
                return {"pattern": "impulse_wave", "direction": 1}
            elif recent_high < highs[0]["price"] and recent_low < lows[0]["price"]:
                return {"pattern": "impulse_wave", "direction": -1}

        return {"pattern": "complex_correction", "direction": 0}

    def _integrate_trend_signals(
        self, primary_trend: Dict, trend_strength: float, wave_analysis: Optional[Dict]
    ) -> SignalResult:
        """統合シグナル生成"""
        direction = primary_trend["direction"]
        strength = primary_trend["strength"]

        # ADXによる強度調整
        strength *= min(trend_strength / 20.0, 2.0)  # ADX 20以上でブースト

        # Wave分析による調整
        if (
            wave_analysis
            and wave_analysis.get("wave_structure", {}).get("direction") == direction
        ):
            strength *= 1 + wave_analysis.get("confidence", 0) * 0.5

        # シグナルタイプ決定
        if direction == 1:
            signal_type = "trend_bullish"
            description = f"上昇トレンド検出 (強度: {trend_strength:.1f})"
        elif direction == -1:
            signal_type = "trend_bearish"
            description = f"下降トレンド検出 (強度: {trend_strength:.1f})"
        else:
            signal_type = "trend_sideways"
            description = "横ばい相場"
            strength = 0.0

        return SignalResult(
            signal_type=signal_type,
            strength=min(strength, 1.0),
            direction=direction,
            description=description,
            metadata={
                "primary_trend": primary_trend,
                "trend_strength": trend_strength,
                "wave_analysis": wave_analysis,
                "analysis_phase": "hierarchical",
            },
            validity_period=20,  # トレンドシグナルの有効期間
            risk_level="medium",
        )

    def _create_neutral_signal(self, reason: str) -> SignalResult:
        """中立シグナル生成"""
        return SignalResult(
            signal_type="trend_neutral",
            strength=0.0,
            direction=0,
            description=f"トレンド分析: {reason}",
            metadata={"reason": reason},
            validity_period=1,
            risk_level="low",
        )

    def _create_error_signal(self, error: str) -> SignalResult:
        """エラーシグナル生成"""
        return SignalResult(
            signal_type="trend_error",
            strength=0.0,
            direction=0,
            description=f"トレンド分析エラー: {error}",
            metadata={"error": error},
            validity_period=1,
            risk_level="low",
        )
