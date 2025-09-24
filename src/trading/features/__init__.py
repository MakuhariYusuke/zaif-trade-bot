"""
Trading features package.
特徴量パッケージ
"""

from .registry import FeatureManager, get_feature_manager
from .base import BaseFeature, CommonPreprocessor
from .wave1 import (
    ROC, RollingMean, RollingStd, ZScore, Lags, DOW, HourOfDay
)
# from .wave2 import (
#     EMACross, TEMA, KAMA
# )
from .wave3 import (
    RegimeClustering, KalmanFilter
)
from .trend import (
    HeikinAshi, Supertrend, Ichimoku, Donchian, ADX, EMACross, TEMA, KAMA
)
from .volatility import (
    ATRSimplified, Bollinger, HVFeature, ReturnStdDev
)
from .momentum import (
    RSI, MACD, Stochastic, CCI
)
from .volume import (
    OBV, VWAP, PriceVolumeCorr, ReturnMA, MFI
)

# Wave1特徴量の登録
def register_wave1_features(manager: FeatureManager):
    """Wave1特徴量をマネージャーに登録"""
    manager.register(ROC(period=14))
    manager.register(RSI(period=14))
    manager.register(RollingMean(14))
    manager.register(RollingStd(14))
    manager.register(RollingMean(50))
    manager.register(RollingStd(50))
    manager.register(ZScore(window=20))
    manager.register(ATRSimplified(period=14))
    manager.register(Lags())
    manager.register(DOW())
    manager.register(HourOfDay())
    manager.register(OBV())
    manager.register(VWAP())

# Wave2特徴量の登録
def register_wave2_features(manager: FeatureManager):
    """Wave2特徴量をマネージャーに登録"""
    manager.register(MACD())
    manager.register(Bollinger())
    manager.register(Stochastic())
    manager.register(CCI())
    manager.register(ADX())
    manager.register(EMACross())
    manager.register(TEMA())
    manager.register(KAMA())

# Wave3特徴量の登録
def register_wave3_features(manager: FeatureManager):
    """Wave3特徴量をマネージャーに登録"""
    manager.register(Ichimoku())
    manager.register(Donchian())
    manager.register(RegimeClustering())
    manager.register(KalmanFilter())

# Wave4特徴量の登録
# 自動登録
try:
    manager = get_feature_manager()
    register_wave1_features(manager)
    register_wave2_features(manager)
    register_wave3_features(manager)
except FileNotFoundError:
    # configファイルがない場合はスキップ
    pass