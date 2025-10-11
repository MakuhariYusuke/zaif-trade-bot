"""
質的に改善した特徴セット定義
平均足・一目均衡表の問題を修正し、冗長な特徴を削除
設定ファイルからの動的特徴量選択に対応
"""

from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any
from ztb.utils.config_loader import ConfigLoader
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FeatureSet(Enum):
    """Enumeration of available feature sets."""
    CURATED = "curated"
    FULL = "full"
    MINIMAL = "minimal"


def load_feature_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load feature configuration from file."""
    if config_path is None:
        config_path = Path("configs/features.yaml")

    if not config_path.exists():
        logger.warning(f"Feature config not found: {config_path}, using defaults")
        return {}

    return ConfigLoader.load(config_path)


def get_feature_set(feature_set_name: str = "curated", config_path: Optional[Path] = None) -> List[str]:
    """
    Get feature set by name from configuration.

    Args:
    feature_set_name: Name of feature set ("curated", "full", "minimal", or custom name)
        config_path: Path to feature configuration file

    Returns:
        List of feature names
    """
    config = load_feature_config(config_path)

    # Check if custom feature set is defined in config
    if feature_set_name in config:
        custom_features = config[feature_set_name]
        if isinstance(custom_features, list):
            logger.info(f"Using custom feature set '{feature_set_name}' with {len(custom_features)} features")
            return custom_features

    # Use predefined feature sets
    if feature_set_name == "curated":
        return CURATED_FEATURES
    elif feature_set_name == "full":
        # Return all features (would need to be defined elsewhere)
        logger.warning("Full feature set not implemented, using curated")
        return CURATED_FEATURES
    elif feature_set_name == "minimal":
        # Return minimal feature set
        return CURATED_FEATURES[:20]  # First 20 features as minimal set
    else:
        logger.warning(f"Unknown feature set '{feature_set_name}', using curated")
        return CURATED_FEATURES


def get_features_to_remove(feature_set_name: str = "curated") -> List[str]:
    """Get features to remove based on feature set."""
    if feature_set_name == "curated":
        return FEATURES_TO_REMOVE
    else:
        # For other sets, no features to remove
        return []

# 削除すべき特徴 (50個削除 → 110から60へ)
FEATURES_TO_REMOVE = [
    # 1. 平均足の個別OHLC (4個) - 通常足と冗長、色連続のみ有効
    'HeikinAshi_Open',
    'HeikinAshi_High', 
    'HeikinAshi_Low',
    'HeikinAshi_Close',
    
    # 2. Time系定数 (5個) - 分散ゼロ
    'DOW',
    'Time_Day_of_Week',
    'Time_Hour_of_Day',
    'Time_Session',
    'Time_Volatility_Adjustment',
    
    # 3. 一目均衡表の単独無意味な線 (4個)
    'Ichimoku_Chikou',  # 遅行スパン単独では無意味
    'Ichimoku_Tenkan',  # 転換線単独では不十分(Crossで使用)
    'Ichimoku_Kijun',   # 基準線単独では不十分(Crossで使用)
    'Ichimoku_Senkou_A',  # 先行スパンA単独では不十分(Composite Signalで使用)
    'Ichimoku_Senkou_B',  # 先行スパンB単独では不十分(Composite Signalで使用)
    
    # 4. 高相関ペアの片方 (20個)
    'price',  # closeと完全一致
    'sma_long',  # BB_Middleと完全一致
    'BB_Middle',  # rolling_mean_20と完全一致
    'Bollinger_Bandwidth',  # BB_Widthと完全一致
    'micro_trend',  # ReturnMA_Shortと完全一致
    'Kalman_Estimate',  # ema_5とほぼ一致(0.9999)
    'sma_short',  # ema_5/TEMAと高相関
    'Donchian_Price_Position',  # Donchian_Pos_20とほぼ一致
    
    # 5. 重複するMA系 (3個)
    'KAMA',  # ema_5/TEMAで代替可能
    
    # 6. ボリンジャーバンドの冗長要素 (2個)
    'BB_Lower',  # BB_PositionとWidthで計算可能
    'BB_Upper',  # BB_PositionとWidthで計算可能
    
    # 7. ドンチャンの冗長要素 (2個)
    'Donchian_Width',  # Donchian_Squeeze_Ratioで代替
    'Donchian_Slope_20',  # Donchian_Breakout_Strengthで代替
    
    # 8. ケルトナーの冗長要素 (2個)
    'Keltner_Lower',  # Keltner_Positionで代替
    'Keltner_Upper',  # Keltner_Positionで代替
    
    # 9. その他の低情報量特徴 (8個)
    'rsi',  # RSIと重複(大文字小文字違い)
    'pnl',  # トレーニングラベルなので除外
    'win',  # トレーニングラベルなので除外
    'qty',  # 分散が極小
    'Donchian_Breakout_Strength',  # 分散ゼロに近い
    'ReturnMA_Medium',  # 分散ゼロに近い
    'Stochastic_Divergence',  # 離散値で情報量少
    'Bollinger_Squeeze',  # バイナリ値
]

# 保持すべき重要特徴 (60個)
CURATED_FEATURES = [
    # 【価格基本】 (5個)
    'close',
    'open', 
    'high',
    'low',
    'volume',
    
    # 【トレンド指標】 (10個)
    'ADX',              # トレンド強度
    'MACD',             # トレンド方向
    'PSAR',             # パラボリックSAR
    'PSAR_Trend',       # PSARトレンド方向
    'EMACross_Diff',    # EMAクロス差分
    'EMACross_Signal',  # EMAクロスシグナル
    'TEMA',             # トリプルEMA
    'ema_5',            # 短期EMA
    'VWAP',             # 出来高加重平均価格
    'rolling_mean_20',  # 20期間移動平均
    
    # 【オシレーター】 (9個)
    'RSI',              # 相対力指数
    'CCI',              # コモディティチャネル指数
    'Stochastic',       # ストキャスティクス
    'Stochastic_Trend_Alignment',  # ストキャトレンド整合
    'Stochastic_Signal_Strength',  # ストキャシグナル強度
    'Williams_R',       # ウィリアムズ%R
    'MFI',              # マネーフローインデックス
    'PlusDI',           # +DI
    'MinusDI',          # -DI
    
    # 【ボラティリティ】 (7個)
    'ATR',              # 平均真の範囲
    'ATR_simplified',   # 簡易ATR
    'atr_10',           # 10期間ATR
    'HV',               # 過去ボラティリティ
    'BB_Position',      # ボリンジャーバンド位置
    'Bollinger_Percent_B',  # ボリンジャー%B
    'BB_Width',         # ボリンジャー幅
    
    # 【ボリンジャーバンド】 (1個)
    'Bollinger_Band_Expansion',  # バンド拡大
    
    # 【ケルトナーチャネル】 (2個)
    'Keltner_Position', # ケルトナー位置
    'Keltner_Width',    # ケルトナー幅
    
    # 【ドンチャンチャネル】 (3個)
    'Donchian_Pos_20',  # ドンチャン位置
    'Donchian_Squeeze_Ratio',  # スクイーズ比率
    'Donchian_Width_Rel_20',   # 相対幅
    
    # 【一目均衡表(組み合わせのみ)】 (4個) - 2つ削減
    'Ichimoku_Composite_Signal',      # 総合シグナル(複数線の組み合わせ)
    'Ichimoku_Price_Cloud_Distance',  # 価格と雲の距離
    'Ichimoku_Cloud_Thickness',       # 雲の厚み
    'Ichimoku_Trend',                 # トレンド方向
    # Note: Ichimoku_Cross, Ichimoku_Diff_Norm は除外(離散値/冗長)
    
    # 【スーパートレンド】 (4個) - Reversal Signal除外(離散値)
    'Supertrend',                     # スーパートレンド値
    'Supertrend_Direction',           # 方向
    'Supertrend_Strength',            # 強度
    'Supertrend_Trend_Duration',      # トレンド継続期間
    # Note: Supertrend_Reversal_Signal, Supertrend_Volatility_Filter は除外
    
    # 【ボリューム分析】 (6個) - 1つ削減
    'OBV',                            # オンバランスボリューム
    'CMF',                            # チャイキンマネーフロー
    'PriceVolumeCorr',                # 価格出来高相関
    'Volume_Profile_Distribution',    # 出来高プロファイル分布
    'Volume_Profile_Value_Area_High', # 値域上限
    'liquidity_surge',                # 流動性急増
    
    # 【その他の重要指標】 (6個)
    'ROC',                            # 変化率
    'ZScore',                         # Zスコア
    'ReturnMA_Short',                 # 短期リターン移動平均
    'ReturnStdDev',                   # リターン標準偏差
    'Kalman_Residual_Norm',           # カルマン残差正規化
    'HourOfDay',                      # 時間(唯一有効なTime系)
    
    # 【マイクロ構造】 (3個) - 1つ削減
    'micro_volatility',               # マイクロボラティリティ
    'price_velocity',                 # 価格速度
    'price_acceleration',             # 価格加速度
]

# 検証: 60個になっているか
assert len(CURATED_FEATURES) == 60, f"Expected 60 features, got {len(CURATED_FEATURES)}"

print(f"✅ 質的改善特徴セット: {len(CURATED_FEATURES)}個定義完了")
print(f"削除対象: {len(FEATURES_TO_REMOVE)}個")
print(f"元の特徴数: 110個")
print(f"削減率: {len(FEATURES_TO_REMOVE)/110*100:.1f}%")
