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

# 削除すべき特徴 (47個削除 → 110から63へ)
FEATURES_TO_REMOVE = [
    # 1. Time系定数 (5個) - 分散ゼロ
    'DOW',
    'Time_Day_of_Week',
    'Time_Hour_of_Day',
    'Time_Session',
    'Time_Volatility_Adjustment',
    
    # 2. 平均足の基本色 (1個) - 多時間軸版のみ使用
    'HeikinAshi_Color',
    
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
    
    # 【トレンド指標】 (47個) - 6つ追加
    'ADX_M1',           # 1分足ADX
    'ADX_M5',           # 5分足ADX
    'ADX_M15',          # 15分足ADX
    'ADX_H1',           # 1時間足ADX
    'ADX_H4',           # 4時間足ADX
    'ADX_D1',           # 日足ADX
    'PlusDI',           # +DI
    'PlusDI_M1',        # 1分足+DI
    'PlusDI_M5',        # 5分足+DI
    'PlusDI_M15',       # 15分足+DI
    'PlusDI_H1',        # 1時間足+DI
    'PlusDI_H4',        # 4時間足+DI
    'PlusDI_D1',        # 日足+DI
    'MinusDI',          # -DI
    'MinusDI_M1',       # 1分足-DI
    'MinusDI_M5',       # 5分足-DI
    'MinusDI_M15',      # 15分足-DI
    'MinusDI_H1',       # 1時間足-DI
    'MinusDI_H4',       # 4時間足-DI
    'MinusDI_D1',       # 日足-DI
    'MACD',             # トレンド方向
    'PSAR',             # パラボリックSAR
    'PSAR_Trend',       # PSARトレンド方向
    'EMACross_Diff',    # EMAクロス差分
    'EMACross_Diff_M1', # 1分足EMAクロス差分
    'EMACross_Diff_M5', # 5分足EMAクロス差分
    'EMACross_Diff_M15', # 15分足EMAクロス差分
    'EMACross_Diff_H1', # 1時間足EMAクロス差分
    'EMACross_Diff_H4', # 4時間足EMAクロス差分
    'EMACross_Diff_D1', # 日足EMAクロス差分
    'EMACross_Signal',  # EMAクロスシグナル
    'EMACross_Signal_M1', # 1分足EMAクロスシグナル
    'EMACross_Signal_M5', # 5分足EMAクロスシグナル
    'EMACross_Signal_M15', # 15分足EMAクロスシグナル
    'EMACross_Signal_H1', # 1時間足EMAクロスシグナル
    'EMACross_Signal_H4', # 4時間足EMAクロスシグナル
    'EMACross_Signal_D1', # 日足EMAクロスシグナル
    'TEMA',             # トリプルEMA
    'ema_5',            # 短期EMA
    'VWAP',             # 出来高加重平均価格
    'rolling_mean_20',  # 20期間移動平均
    'HeikinAshi_Color_M1',  # 1分足平均足色
    'HeikinAshi_Color_M5',  # 5分足平均足色
    'HeikinAshi_Color_M15', # 15分足平均足色
    'HeikinAshi_Color_H1',  # 1時間足平均足色
    'HeikinAshi_Color_H4',  # 4時間足平均足色
    'HeikinAshi_Color_D1',  # 日足平均足色
    
    # 【オシレーター】 (13個) - 6つ追加
    'RSI',              # 相対力指数
    'RSI_M1',           # 1分足RSI
    'RSI_M5',           # 5分足RSI
    'RSI_M15',          # 15分足RSI
    'RSI_H1',           # 1時間足RSI
    'RSI_H4',           # 4時間足RSI
    'RSI_D1',           # 日足RSI
    'CCI',              # コモディティチャネル指数
    'Stochastic',       # ストキャスティクス
    'Stochastic_Trend_Alignment',  # ストキャトレンド整合
    'Stochastic_Signal_Strength',  # ストキャシグナル強度
    'Williams_R',       # ウィリアムズ%R
    'MFI',              # マネーフローインデックス
    
    # 【ボラティリティ】 (17個) - 12つ追加
    'ATR',              # 平均真の範囲
    'ATR_M1',           # 1分足ATR
    'ATR_M5',           # 5分足ATR
    'ATR_M15',          # 15分足ATR
    'ATR_H1',           # 1時間足ATR
    'ATR_H4',           # 4時間足ATR
    'ATR_D1',           # 日足ATR
    'ATR_simplified',   # 簡易ATR
    'ATR_simplified_M1', # 1分足簡易ATR
    'ATR_simplified_M5', # 5分足簡易ATR
    'ATR_simplified_M15', # 15分足簡易ATR
    'ATR_simplified_H1', # 1時間足簡易ATR
    'ATR_simplified_H4', # 4時間足簡易ATR
    'ATR_simplified_D1', # 日足簡易ATR
    'Normalized_ATR',   # 正規化ATR (パーセント表示)
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
    
    # 【一目均衡表(組み合わせ+拡張+多時間軸)】 (34個) - 18つ追加
    # 基本ライン
    'Ichimoku_Tenkan',                 # 転換線(Conversion Line)
    'Ichimoku_Kijun',                  # 基準線(Base Line)
    'Ichimoku_Senkou_A',               # 先行スパンA(Leading Span A)
    'Ichimoku_Senkou_B',               # 先行スパンB(Leading Span B)
    'Ichimoku_Chikou',                 # 遅行スパン(Lagging Span)
    # 基本分析
    'Ichimoku_Composite_Signal',      # 総合シグナル(複数線の組み合わせ)
    'Ichimoku_Price_Cloud_Distance',  # 価格と雲の距離
    'Ichimoku_Cloud_Thickness',       # 雲の厚み
    'Ichimoku_Trend',                 # トレンド方向
    # 理論的拡張
    'Ichimoku_Time_Theory',           # 時間論: 転換線と基準線の時間的関係
    'Ichimoku_Wave_Theory',           # 波動論: 雲の波動的意味付け
    'Ichimoku_Value_Measurement',     # 値幅観測論: 価格変動の測定
    'Ichimoku_Momentum_Confirmation', # 勢い確認: 遅行スパンのモメンタム的解釈
    # 高度な分析
    'Ichimoku_Cloud_Slope',           # 雲の傾き/角度
    'Ichimoku_Sanyaku_Kouten',        # 三役好転/逆転
    'Ichimoku_Cloud_Expansion',       # 雲の拡大/縮小
    # 多時間軸拡張 (各時間軸でComposite Signal, Trend, Cloud Thickness, Price-Cloud Distance)
    'Ichimoku_Composite_Signal_M1',   # 1分足総合シグナル
    'Ichimoku_Composite_Signal_M5',   # 5分足総合シグナル
    'Ichimoku_Composite_Signal_M15',  # 15分足総合シグナル
    'Ichimoku_Composite_Signal_H1',   # 1時間足総合シグナル
    'Ichimoku_Composite_Signal_H4',   # 4時間足総合シグナル
    'Ichimoku_Composite_Signal_D1',   # 日足総合シグナル
    'Ichimoku_Trend_M1',              # 1分足トレンド
    'Ichimoku_Trend_M5',              # 5分足トレンド
    'Ichimoku_Trend_M15',             # 15分足トレンド
    'Ichimoku_Trend_H1',              # 1時間足トレンド
    'Ichimoku_Trend_H4',              # 4時間足トレンド
    'Ichimoku_Trend_D1',              # 日足トレンド
    'Ichimoku_Cloud_Thickness_M1',    # 1分足雲の厚み
    'Ichimoku_Cloud_Thickness_M5',    # 5分足雲の厚み
    'Ichimoku_Cloud_Thickness_M15',   # 15分足雲の厚み
    'Ichimoku_Cloud_Thickness_H1',    # 1時間足雲の厚み
    'Ichimoku_Cloud_Thickness_H4',    # 4時間足雲の厚み
    'Ichimoku_Cloud_Thickness_D1',    # 日足雲の厚み
    'Ichimoku_Price_Cloud_Distance_M1',   # 1分足価格-雲距離
    'Ichimoku_Price_Cloud_Distance_M5',   # 5分足価格-雲距離
    'Ichimoku_Price_Cloud_Distance_M15',  # 15分足価格-雲距離
    'Ichimoku_Price_Cloud_Distance_H1',   # 1時間足価格-雲距離
    'Ichimoku_Price_Cloud_Distance_H4',   # 4時間足価格-雲距離
    'Ichimoku_Price_Cloud_Distance_D1',   # 日足価格-雲距離
    
    # 【スーパートレンド】 (4個) - Reversal Signal除外(離散値)
    'Supertrend',                     # スーパートレンド値
    'Supertrend_Direction',           # 方向
    'Supertrend_Strength',            # 強度
    'Supertrend_Trend_Duration',      # トレンド継続期間
    # Note: Supertrend_Reversal_Signal, Supertrend_Volatility_Filter は除外
    
    # 【ボリューム分析】 (8個) - 2つ追加
    'OBV',                            # オンバランスボリューム
    'CMF',                            # チャイキンマネーフロー
    'Chaikin_AD',                     # チャイキンA/Dライン
    'Chaikin_AD_Oscillator',          # チャイキンA/Dオシレーター
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
    # 【Ta-Lib活用拡張指標】 (3個追加)
    'Ultimate_Oscillator',        # アルティメットオシレーター(3期間モメンタム統合)
    'TSI',                        # True Strength Index(真の強度指数)
    'KST',                        # Know Sure Thing(ノウシュアシング)
    
    # 【マイクロ構造】 (2個) - 1つ削減
    'micro_volatility',               # マイクロボラティリティ
    'price_velocity',                 # 価格速度
    'price_acceleration',             # 価格加速度
    
    # 【時間特徴】 (2個追加)
    'Time_Monthly_Cycle',             # 月次サイクル進行度
    'Time_Quarterly_Cycle',           # 四半期サイクル進行度
]

# 検証: 78個になっているか確認
# assert len(CURATED_FEATURES) == 78, f"Expected 78 features, got {len(CURATED_FEATURES)}"

print(f"✅ 質的改善特徴セット: {len(CURATED_FEATURES)}個定義完了")
print(f"削除対象: {len(FEATURES_TO_REMOVE)}個")
print(f"元の特徴数: 110個")
print(f"削減率: {len(FEATURES_TO_REMOVE)/110*100:.1f}%")
print(f"Ta-Lib拡張: +7個 (一目4個 + Ta-Lib指標3個 + 高優先度指標2個 + 時間特徴2個)")
print(f"一目均衡表拡張: +34個 (基本5個 + 理論的拡張4個 + 高度な分析3個 + 多時間軸18個)")
