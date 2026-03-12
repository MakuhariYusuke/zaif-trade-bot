"""
Curated Features Module

統合特徴量セット管理モジュール
PPOトレーナーなどで使用される特徴量セットを提供
"""

from enum import Enum

from ztb.features.feature_set_manager import FeatureSetManager
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

class FeatureSet(Enum):
    """特徴量セットの種類"""

    CURATED = "curated"
    FULL = "full"
    MINIMAL = "minimal"

def get_feature_set(feature_set: str = "curated", config_path=None) -> list[str]:
    """
    指定された特徴量セットを取得

    Args:
        feature_set: 特徴量セット名 ("curated", "full", "minimal")

    Returns:
        特徴量名のリスト
    """
    try:
        manager = FeatureSetManager()
        if feature_set in manager.feature_sets:
            config = manager.feature_sets[feature_set]
            if config.enabled:
                return config.features
            else:
                logger.warning(f"Feature set '{feature_set}' is disabled")
                return []
        else:
            logger.warning(f"Feature set '{feature_set}' not found, using curated")
            # デフォルトとしてcuratedセットを返す
            if "curated" in manager.feature_sets:
                return manager.feature_sets["curated"].features
            else:
                # マネージャーにセットがない場合はデフォルト特徴量を返す
                return _get_default_curated_features()
    except Exception as e:
        logger.error(f"Error getting feature set '{feature_set}': {e}")
        return _get_default_curated_features()

def get_features_to_remove() -> list[str]:
    """
    削除すべき特徴量を取得

    Returns:
        削除対象の特徴量名のリスト
    """
    # TODO: 設定ファイルや分析結果に基づいて削除対象特徴量を決定
    # 現時点では空リストを返す
    return []

def _get_default_curated_features() -> list[str]:
    """
    デフォルトのcurated特徴量セットを取得

    Returns:
        デフォルト特徴量リスト
    """
    return [
        "close",
        "open",
        "high",
        "low",
        "volume",
        "ADX_M1",
        "ADX_M5",
        "ADX_M15",
        "ADX_H1",
        "ADX_H4",
        "ADX_D1",
        "PlusDI",
        "PlusDI_M1",
        "PlusDI_M5",
        "PlusDI_M15",
        "PlusDI_H1",
        "PlusDI_H4",
        "PlusDI_D1",
        "MinusDI",
        "MinusDI_M1",
        "MinusDI_M5",
        "MinusDI_M15",
        "MinusDI_H1",
        "MinusDI_H4",
        "MinusDI_D1",
        "MACD",
        "PSAR",
        "PSAR_Trend",
        "EMACross_Diff",
        "EMACross_Diff_M1",
        "EMACross_Diff_M5",
        "EMACross_Diff_M15",
        "EMACross_Diff_H1",
        "EMACross_Diff_H4",
        "EMACross_Diff_D1",
        "EMACross_Signal",
        "EMACross_Signal_M1",
        "EMACross_Signal_M5",
        "EMACross_Signal_M15",
        "EMACross_Signal_H1",
        "EMACross_Signal_H4",
        "EMACross_Signal_D1",
        "TEMA",
        "ema_5",
        "VWAP",
        "rolling_mean_20",
        "HeikinAshi_Color_M1",
        "HeikinAshi_Color_M5",
        "HeikinAshi_Color_M15",
        "HeikinAshi_Color_H1",
        "HeikinAshi_Color_H4",
        "HeikinAshi_Color_D1",
        "RSI",
        "RSI_M1",
        "RSI_M5",
        "RSI_M15",
        "RSI_H1",
        "RSI_H4",
        "RSI_D1",
        "CCI",
        "Stochastic",
        "Stochastic_Trend_Alignment",
        "Stochastic_Signal_Strength",
        "Williams_R",
        "MFI",
        "ATR",
        "ATR_M1",
        "ATR_M5",
        "ATR_M15",
        "ATR_H1",
        "ATR_H4",
        "ATR_D1",
        "ATR_simplified",
        "ATR_simplified_M1",
        "ATR_simplified_M5",
        "ATR_simplified_M15",
        "ATR_simplified_H1",
        "ATR_simplified_H4",
        "ATR_simplified_D1",
        "Normalized_ATR",
        "atr_10",
        "HV",
        "BB_Position",
        "Bollinger_Percent_B",
        "BB_Width",
        "Bollinger_Band_Expansion",
        "Keltner_Position",
        "Keltner_Width",
        "Donchian_Pos_20",
        "Donchian_Squeeze_Ratio",
        "Donchian_Width_Rel_20",
        "Ichimoku_Tenkan",
        "Ichimoku_Kijun",
        "Ichimoku_Senkou_A",
        "Ichimoku_Senkou_B",
        "Ichimoku_Chikou",
        "Ichimoku_Composite_Signal",
        "Ichimoku_Price_Cloud_Distance",
        "Ichimoku_Cloud_Thickness",
        "Ichimoku_Trend",
        "Ichimoku_Time_Theory",
        "Ichimoku_Wave_Theory",
        "Ichimoku_Value_Measurement",
        "Ichimoku_Momentum_Confirmation",
        "Ichimoku_Cloud_Slope",
        "Ichimoku_Sanyaku_Kouten",
        "Ichimoku_Cloud_Expansion",
        "Ichimoku_Composite_Signal_M1",
        "Ichimoku_Composite_Signal_M5",
        "Ichimoku_Composite_Signal_M15",
        "Ichimoku_Composite_Signal_H1",
        "Ichimoku_Composite_Signal_H4",
        "Ichimoku_Composite_Signal_D1",
        "Ichimoku_Trend_M1",
        "Ichimoku_Trend_M5",
        "Ichimoku_Trend_M15",
        "Ichimoku_Trend_H1",
        "Ichimoku_Trend_H4",
        "Ichimoku_Trend_D1",
        "Ichimoku_Cloud_Thickness_M1",
        "Ichimoku_Cloud_Thickness_M5",
        "Ichimoku_Cloud_Thickness_M15",
        "Ichimoku_Cloud_Thickness_H1",
        "Ichimoku_Cloud_Thickness_H4",
        "Ichimoku_Cloud_Thickness_D1",
        "Ichimoku_Price_Cloud_Distance_M1",
        "Ichimoku_Price_Cloud_Distance_M5",
        "Ichimoku_Price_Cloud_Distance_M15",
        "Ichimoku_Price_Cloud_Distance_H1",
        "Ichimoku_Price_Cloud_Distance_H4",
        "Ichimoku_Price_Cloud_Distance_D1",
        "Supertrend",
        "Supertrend_Direction",
        "Supertrend_Strength",
        "Supertrend_Trend_Duration",
        "OBV",
        "CMF",
        "Chaikin_AD",
        "Chaikin_AD_Oscillator",
        "PriceVolumeCorr",
        "Volume_Profile_Distribution",
        "Volume_Profile_Value_Area_High",
        "liquidity_surge",
        "ROC",
        "ZScore",
        "ReturnMA_Short",
        "ReturnStdDev",
        "Kalman_Residual_Norm",
        "Ultimate_Oscillator",
        "TSI",
        "KST",
        "micro_volatility",
        "price_velocity",
        "price_acceleration",
        "Time_Monthly_Cycle",
        "Time_Quarterly_Cycle",
    ]
