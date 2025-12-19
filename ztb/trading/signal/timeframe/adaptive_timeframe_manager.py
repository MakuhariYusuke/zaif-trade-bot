"""
Adaptive Timeframe Manager for Phase 4: Minute-level Trading Support

動的タイムフレーム適応による高頻度取引対応
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from enum import Enum

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Timeframe(Enum):
    """サポートされるタイムフレーム"""
    SCALPING = "1m"      # スキャルピング（1分足）
    INTRADAY = "5m"      # 日中取引（5分足）
    SWING = "15m"        # スイング（15分足）
    POSITION = "1h"      # ポジション（1時間足）
    DAILY = "1d"         # 日次（1日足）


class MarketCondition(Enum):
    """市場条件の分類"""
    LOW_VOLATILITY = "low_volatility"
    NORMAL = "normal"
    HIGH_VOLATILITY = "high_volatility"
    TRENDING = "trending"
    RANGING = "ranging"


class AdaptiveTimeframeManager:
    """
    適応型タイムフレームマネージャー

    市場ボラティリティとトレンド強度に基づいて最適なタイムフレームを動的に選択
    Phase 4のコアコンポーネント
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize adaptive timeframe manager

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()

        # タイムフレーム設定
        self.timeframes = {
            'scalping': Timeframe.SCALPING.value,
            'intraday': Timeframe.INTRADAY.value,
            'swing': Timeframe.SWING.value,
            'position': Timeframe.POSITION.value,
            'daily': Timeframe.DAILY.value
        }

        # 市場条件別最適タイムフレームマッピング
        self.regime_timeframes = {
            MarketCondition.LOW_VOLATILITY.value: 'scalping',    # 低ボラティリティ時は高頻度
            MarketCondition.NORMAL.value: 'intraday',            # 通常は5分足
            MarketCondition.HIGH_VOLATILITY.value: 'swing',      # 高ボラティリティ時は15分足
            MarketCondition.TRENDING.value: 'position',          # 明確トレンド時は1時間足
            MarketCondition.RANGING.value: 'intraday'            # レンジ相場は5分足
        }

        # 適応パラメータ
        self.volatility_thresholds = self.config.get('volatility_thresholds', {
            'low': 0.02,      # 2%未満は低ボラティリティ
            'normal': 0.05,   # 5%未満は通常
            'high': 0.05      # 5%以上は高ボラティリティ
        })

        self.trend_thresholds = self.config.get('trend_thresholds', {
            'weak': 0.3,      # 0.3未満は弱いトレンド
            'strong': 0.7     # 0.7以上は強いトレンド
        })

        logger.info("AdaptiveTimeframeManager initialized")

    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'volatility_thresholds': {
                'low': 0.02,
                'normal': 0.05,
                'high': 0.05
            },
            'trend_thresholds': {
                'weak': 0.3,
                'strong': 0.7
            },
            'min_data_points': {
                '1m': 100,    # 1分足は最低100ポイント
                '5m': 50,     # 5分足は最低50ポイント
                '15m': 20,    # 15分足は最低20ポイント
                '1h': 10,     # 1時間足は最低10ポイント
                '1d': 5       # 日足は最低5ポイント
            }
        }

    def analyze_market_condition(self, df: pd.DataFrame) -> MarketCondition:
        """
        市場データを分析して市場条件を判定

        Args:
            df: 市場データ（OHLCV）

        Returns:
            MarketCondition: 判定された市場条件
        """
        if len(df) < 20:
            return MarketCondition.NORMAL

        try:
            # ボラティリティ計算（過去20期間の標準偏差）
            returns = df['close'].pct_change().dropna()
            volatility = returns.std()

            # トレンド強度計算（線形回帰のR²スコア）
            trend_strength = self._calculate_trend_strength(df)

            # 市場条件の判定
            if volatility < self.volatility_thresholds['low']:
                if trend_strength < self.trend_thresholds['weak']:
                    return MarketCondition.LOW_VOLATILITY
                else:
                    return MarketCondition.TRENDING
            elif volatility > self.volatility_thresholds['high']:
                return MarketCondition.HIGH_VOLATILITY
            else:
                if trend_strength > self.trend_thresholds['strong']:
                    return MarketCondition.TRENDING
                elif trend_strength < self.trend_thresholds['weak']:
                    return MarketCondition.RANGING
                else:
                    return MarketCondition.NORMAL

        except Exception as e:
            logger.warning(f"Error analyzing market condition: {e}")
            return MarketCondition.NORMAL

    def select_optimal_timeframe(self, df: pd.DataFrame,
                               current_timeframe: str = None) -> Tuple[str, MarketCondition]:
        """
        市場データに基づいて最適なタイムフレームを選択

        Args:
            df: 市場データ
            current_timeframe: 現在のタイムフレーム（オプション）

        Returns:
            Tuple[str, MarketCondition]: (最適タイムフレーム, 市場条件)
        """
        # 市場条件を分析
        market_condition = self.analyze_market_condition(df)

        # 市場条件に基づいて最適タイムフレームを選択
        optimal_strategy = self.regime_timeframes[market_condition.value]
        optimal_timeframe = self.timeframes[optimal_strategy]

        # データ充足性を確認
        if not self._validate_data_sufficiency(df, optimal_timeframe):
            # データが不足する場合、現在のタイムフレームを維持
            if current_timeframe:
                optimal_timeframe = current_timeframe
            else:
                # デフォルトは5分足
                optimal_timeframe = self.timeframes['intraday']

        logger.debug(f"Selected timeframe: {optimal_timeframe} for condition: {market_condition.value}")

        return optimal_timeframe, market_condition

    def get_timeframe_hierarchy(self, base_timeframe: str) -> List[str]:
        """
        指定タイムフレームの階層構造を取得

        Args:
            base_timeframe: 基準となるタイムフレーム

        Returns:
            List[str]: 確認用のタイムフレームリスト（短い順）
        """
        hierarchy = {
            '1m': ['1m', '5m', '15m'],
            '5m': ['5m', '15m', '1h'],
            '15m': ['15m', '1h', '1d'],
            '1h': ['1h', '1d'],
            '1d': ['1d']
        }

        return hierarchy.get(base_timeframe, [base_timeframe])

    def _calculate_trend_strength(self, df: pd.DataFrame, window: int = 20) -> float:
        """
        トレンド強度を計算（R²スコア）

        Args:
            df: 市場データ
            window: 計算ウィンドウ

        Returns:
            float: トレンド強度（0-1）
        """
        if len(df) < window:
            return 0.0

        try:
            # 最近の価格データを使用
            prices = df['close'].tail(window).values
            x = np.arange(len(prices))

            # 線形回帰
            slope, intercept = np.polyfit(x, prices, 1)

            # R²スコア計算
            y_pred = slope * x + intercept
            ss_res = np.sum((prices - y_pred) ** 2)
            ss_tot = np.sum((prices - np.mean(prices)) ** 2)

            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return max(0, min(1, r_squared))

        except Exception as e:
            logger.warning(f"Error calculating trend strength: {e}")
            return 0.0

    def _validate_data_sufficiency(self, df: pd.DataFrame, timeframe: str) -> bool:
        """
        指定タイムフレームでのデータ充足性を検証

        Args:
            df: 市場データ
            timeframe: 検証対象のタイムフレーム

        Returns:
            bool: データが十分かどうか
        """
        min_points = self.config['min_data_points'].get(timeframe, 50)
        return len(df) >= min_points

    def get_adaptive_parameters(self, market_condition: MarketCondition) -> Dict:
        """
        市場条件に応じた適応パラメータを取得

        Args:
            market_condition: 市場条件

        Returns:
            Dict: 適応パラメータ
        """
        base_params = {
            'signal_sensitivity': 0.5,
            'trend_filter_strength': 0.3,
            'volatility_adjustment': 1.0
        }

        # 市場条件別の調整
        adjustments = {
            MarketCondition.LOW_VOLATILITY.value: {
                'signal_sensitivity': 0.7,      # 高感度
                'trend_filter_strength': 0.2,   # 弱いフィルター
                'volatility_adjustment': 0.8    # 低ボラティリティ調整
            },
            MarketCondition.HIGH_VOLATILITY.value: {
                'signal_sensitivity': 0.3,      # 低感度
                'trend_filter_strength': 0.5,   # 強いフィルター
                'volatility_adjustment': 1.5    # 高ボラティリティ調整
            },
            MarketCondition.TRENDING.value: {
                'signal_sensitivity': 0.4,      # 中感度
                'trend_filter_strength': 0.6,   # 強いトレンドフィルター
                'volatility_adjustment': 1.2    # トレンド調整
            },
            MarketCondition.RANGING.value: {
                'signal_sensitivity': 0.6,      # 高感度
                'trend_filter_strength': 0.1,   # 弱いフィルター
                'volatility_adjustment': 0.9    # レンジ調整
            }
        }

        condition_key = market_condition.value
        if condition_key in adjustments:
            base_params.update(adjustments[condition_key])

        return base_params
