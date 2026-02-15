"""
統合トレンド分析システム

ADX, Wave Counting, Dow Theoryを統合した階層的トレンド分析
"""

from __future__ import annotations

from typing import Literal, Optional, TypedDict

import numpy as np
import pandas as pd

from .base import SignalResult, TrendPatternRecognizer


class PivotPoint(TypedDict):
    """Pivot point used by wave-structure analysis."""

    type: Literal["high", "low"]
    index: int
    price: float


class WaveStructure(TypedDict):
    """Wave-structure classification result."""

    pattern: str
    direction: int


class PrimaryTrendAnalysis(TypedDict):
    """Primary trend payload (Dow-theory style)."""

    direction: int
    strength: float
    ma_short: float
    ma_long: float
    current_price: float
    normalized_slope: float


class WaveAnalysis(TypedDict):
    """Wave analysis payload used for confidence adjustment."""

    pivots: list[PivotPoint]
    wave_structure: WaveStructure
    confidence: float


class HierarchicalTrendAnalyzer(TrendPatternRecognizer):
    """
    階層的トレンド分析システム

    Phase 1: 基本トレンド方向 (Dow Theory)
    Phase 2: トレンド強度 (ADX)
    Phase 3: 詳細波動分析 (Wave Counting) - 強トレンド時のみ
    """

    def __init__(self, config: Optional[dict[str, object]] = None):
        super().__init__(config)
        self.pattern_type = "hierarchical_trend"

        # 各分析器の設定
        self.dow_theory_config = self._as_config_map(
            self.config.get("dow_theory_config", {})
        )
        self.adx_config = self._as_config_map(
            self.config.get(
                "adx_config",
                {"period": 14, "strong_trend_threshold": 25},
            )
        )
        self.wave_config = self._as_config_map(self.config.get("wave_config", {}))

        # 階層的分析の閾値
        self.adx_period = max(2, self._to_int(self.adx_config.get("period"), 14))
        self.strong_trend_threshold = self._to_float(
            self.config.get(
                "strong_trend_threshold",
                self.adx_config.get("strong_trend_threshold", 25),
            ),
            25.0,
        )
        self.enable_wave_analysis = bool(self.config.get("enable_wave_analysis", True))
        self.analysis_window = max(120, self._to_int(self.config.get("analysis_window"), 320))
        self.pivot_lookback = max(2, self._to_int(self.config.get("pivot_lookback"), 5))
        self.slope_threshold = self._to_float(
            self.config.get("slope_threshold"),
            0.0005,
        )

    @staticmethod
    def _as_config_map(value: object) -> dict[str, object]:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _to_int(value: object, default: int) -> int:
        try:
            return int(value) if value is not None else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_float(value: object, default: float) -> float:
        try:
            return float(value) if value is not None else default
        except (TypeError, ValueError):
            return default

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
        if not self.validate_data(data):
            return None

        min_required_periods = max(50, self.adx_period * 2)
        resolved_index = self.resolve_analysis_index(
            len(data),
            index,
            min_required_index=min_required_periods - 1,
        )
        if resolved_index is None:
            return None

        analysis_data, local_index = self._build_analysis_view(
            data,
            resolved_index,
            min_required_periods=min_required_periods,
        )

        try:
            # Phase 1: 基本トレンド方向 (Dow Theory)
            primary_trend = self._analyze_primary_trend(analysis_data, local_index)
            if not primary_trend:
                return self._create_neutral_signal("トレンド不明")

            # Phase 2: トレンド強度 (ADX)
            trend_strength = self._analyze_trend_strength(analysis_data, local_index)

            # Phase 3: 詳細波動分析 (強トレンド時のみ)
            wave_analysis = None
            if (
                self.enable_wave_analysis
                and trend_strength > self.strong_trend_threshold
            ):
                wave_analysis = self._analyze_wave_patterns(analysis_data, local_index)

            # 統合シグナル生成
            return self._integrate_trend_signals(
                primary_trend, trend_strength, wave_analysis
            )

        except Exception as e:
            return self._create_error_signal(str(e))

    def _build_analysis_view(
        self,
        data: pd.DataFrame,
        resolved_index: int,
        *,
        min_required_periods: int,
    ) -> tuple[pd.DataFrame, int]:
        """Build bounded view to reduce repeated long-series computation cost."""
        window_size = max(min_required_periods, self.analysis_window)
        start_idx = max(0, resolved_index - window_size + 1)
        view = data.iloc[start_idx : resolved_index + 1]
        if len(view) < min_required_periods:
            view = data.iloc[: resolved_index + 1]
        return view, len(view) - 1

    def _analyze_primary_trend(
        self, data: pd.DataFrame, index: int
    ) -> Optional[PrimaryTrendAnalysis]:
        """Phase 1: Dow Theoryによる基本トレンド分析"""
        try:
            # 移動平均によるトレンド判定
            close = data["close"].astype(float)
            ma_short = float(close.rolling(20, min_periods=20).mean().iloc[index])
            ma_long = float(close.rolling(50, min_periods=50).mean().iloc[index])
            current_price = float(close.iloc[index])

            if pd.isna(ma_short) or pd.isna(ma_long) or ma_long == 0.0:
                return None

            # トレンド方向判定
            if ma_short > ma_long and current_price > ma_short:
                direction = 1  # 上昇トレンド
                ma_gap = self.safe_ratio(ma_short - ma_long, ma_long, default=0.0)
            elif ma_short < ma_long and current_price < ma_short:
                direction = -1  # 下降トレンド
                ma_gap = self.safe_ratio(ma_long - ma_short, ma_long, default=0.0)
            else:
                direction = 0  # 横ばい
                ma_gap = 0.0

            strength = self.clamp(abs(ma_gap) * 10.0, 0.0, 1.0)
            slope_window = close.iloc[max(0, index - 20) : index + 1].to_numpy(
                dtype=np.float64
            )
            normalized_slope = self.calculate_normalized_slope(slope_window)
            slope_direction = self.slope_direction(normalized_slope, self.slope_threshold)

            # MA方向と回帰傾きが逆向きなら強度を落として誤検出を抑える
            if direction != 0 and slope_direction != 0 and slope_direction != direction:
                direction = 0
                strength *= 0.5

            return {
                "direction": direction,
                "strength": strength,
                "ma_short": ma_short,
                "ma_long": ma_long,
                "current_price": current_price,
                "normalized_slope": normalized_slope,
            }

        except Exception:
            return None

    def _analyze_trend_strength(self, data: pd.DataFrame, index: int) -> float:
        """Phase 2: ADXによるトレンド強度分析"""
        try:
            # ADX計算（簡易版）
            high = data["high"].astype(float)
            low = data["low"].astype(float)
            close = data["close"].astype(float)
            prev_close = close.shift(1)

            # True Range
            tr = pd.concat(
                [
                    high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)

            # Directional Movement
            up_move = high.diff()
            down_move = -low.diff()
            dm_plus = np.where(
                (up_move > down_move) & (up_move > 0.0),
                up_move,
                0.0,
            )
            dm_minus = np.where(
                (down_move > up_move) & (down_move > 0.0),
                down_move,
                0.0,
            )

            period = max(2, self.adx_period)
            atr = tr.rolling(window=period, min_periods=period).mean()
            dm_plus_series = pd.Series(dm_plus, index=data.index, dtype="float64")
            dm_minus_series = pd.Series(dm_minus, index=data.index, dtype="float64")
            plus_smooth = dm_plus_series.rolling(window=period, min_periods=period).mean()
            minus_smooth = dm_minus_series.rolling(window=period, min_periods=period).mean()

            atr_safe = atr.replace(0.0, np.nan)
            di_plus = (plus_smooth / atr_safe) * 100.0
            di_minus = (minus_smooth / atr_safe) * 100.0
            di_sum = (di_plus + di_minus).replace(0.0, np.nan)
            dx = ((di_plus - di_minus).abs() / di_sum) * 100.0
            adx_series = dx.rolling(window=period, min_periods=period).mean()

            adx_value = float(adx_series.iloc[index]) if pd.notna(adx_series.iloc[index]) else 0.0
            return self.clamp(adx_value, 0.0, 100.0)

        except Exception:
            return 0.0

    def _analyze_wave_patterns(
        self, data: pd.DataFrame, index: int
    ) -> Optional[WaveAnalysis]:
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
                "confidence": self.clamp(len(pivots) / 10.0, 0.0, 1.0),
            }  # ピボット数による信頼度

        except Exception:
            return None

    def _find_pivot_points(self, data: pd.DataFrame) -> list[PivotPoint]:
        """ピボットポイント検出"""
        lookback = self.pivot_lookback
        window = lookback * 2 + 1
        if len(data) < window:
            return []

        highs = data["high"].astype(float)
        lows = data["low"].astype(float)
        high_mask = highs.eq(
            highs.rolling(window=window, center=True, min_periods=window).max()
        ).fillna(False)
        low_mask = lows.eq(
            lows.rolling(window=window, center=True, min_periods=window).min()
        ).fillna(False)

        pivots: list[PivotPoint] = []
        for idx in np.flatnonzero(high_mask.to_numpy()):
            pivots.append({"type": "high", "index": int(idx), "price": float(highs.iloc[idx])})
        for idx in np.flatnonzero(low_mask.to_numpy()):
            pivots.append({"type": "low", "index": int(idx), "price": float(lows.iloc[idx])})

        pivots.sort(key=lambda item: item["index"])
        return pivots

    def _analyze_wave_structure(self, pivots: list[PivotPoint]) -> WaveStructure:
        """波動構造分析"""
        if len(pivots) < 3:
            return {"pattern": "insufficient_data", "direction": 0}

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
        self,
        primary_trend: PrimaryTrendAnalysis,
        trend_strength: float,
        wave_analysis: Optional[WaveAnalysis],
    ) -> SignalResult:
        """統合シグナル生成"""
        direction = int(primary_trend["direction"])
        strength = float(primary_trend["strength"])

        # ADXによる強度調整
        strength *= min(trend_strength / 20.0, 2.0)  # ADX 20以上でブースト

        # Wave分析による調整
        if (
            wave_analysis
            and wave_analysis["wave_structure"].get("direction") == direction
        ):
            strength *= 1 + wave_analysis["confidence"] * 0.5

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
            strength=self.clamp(strength, 0.0, 1.0),
            direction=float(direction),
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
            direction=0.0,
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
            direction=0.0,
            description=f"トレンド分析エラー: {error}",
            metadata={"error": error},
            validity_period=1,
            risk_level="low",
        )
