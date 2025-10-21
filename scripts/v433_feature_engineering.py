#!/usr/bin/env python3
"""
V433 Feature Engineering: 市場レジーム適応型特徴量生成
現実データ中心主義に基づく動的特徴量エンジニアリング
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy import stats
import warnings

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """
    市場レジーム検知器
    強気/弱気/横ばい/高ボラティリティの分類
    """

    def __init__(self):
        self.regime_windows = {
            'short': 20,   # 短期レジーム
            'medium': 50,  # 中期レジーム
            'long': 100    # 長期レジーム
        }

    def detect_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        市場レジームの検知

        Args:
            df: 価格データ

        Returns:
            レジーム情報が追加されたデータ
        """
        if 'close' not in df.columns:
            return df

        df = df.copy()

        # トレンド指標
        for window_name, window in self.regime_windows.items():
            # 移動平均
            df[f'sma_{window_name}'] = df['close'].rolling(window=window).mean()

            # トレンド強度 (価格 - 移動平均) / 移動平均
            df[f'trend_strength_{window_name}'] = (
                (df['close'] - df[f'sma_{window_name}']) / df[f'sma_{window_name}']
            )

            # ボラティリティ
            df[f'volatility_{window_name}'] = df['close'].pct_change().rolling(window=window).std()

        # レジーム分類
        df['regime'] = self._classify_regime(df)

        # レジーム確信度
        df['regime_confidence'] = self._calculate_regime_confidence(df)

        return df

    def _classify_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        レジームの分類
        """
        regimes = []

        for idx in df.index:
            # 短期トレンド
            short_trend = df.loc[idx, 'trend_strength_short'] if not pd.isna(df.loc[idx, 'trend_strength_short']) else 0
            short_vol = df.loc[idx, 'volatility_short'] if not pd.isna(df.loc[idx, 'volatility_short']) else 0

            # 中期トレンド
            medium_trend = df.loc[idx, 'trend_strength_medium'] if not pd.isna(df.loc[idx, 'trend_strength_medium']) else 0
            medium_vol = df.loc[idx, 'volatility_medium'] if not pd.isna(df.loc[idx, 'volatility_medium']) else 0

            # 分類ロジック
            if abs(short_trend) > 0.05 and abs(medium_trend) > 0.03:  # 強いトレンド
                if short_trend > 0 and medium_trend > 0:
                    regime = 'bull'  # 強気
                elif short_trend < 0 and medium_trend < 0:
                    regime = 'bear'  # 弱気
                else:
                    regime = 'mixed'  # 混合
            elif short_vol > medium_vol * 1.5:  # 高ボラティリティ
                regime = 'volatile'
            else:
                regime = 'sideways'  # 横ばい

            regimes.append(regime)

        return pd.Series(regimes, index=df.index)

    def _calculate_regime_confidence(self, df: pd.DataFrame) -> pd.Series:
        """
        レジームの確信度計算
        """
        confidence = []

        for idx in df.index:
            short_trend = abs(df.loc[idx, 'trend_strength_short']) if not pd.isna(df.loc[idx, 'trend_strength_short']) else 0
            medium_trend = abs(df.loc[idx, 'trend_strength_medium']) if not pd.isna(df.loc[idx, 'trend_strength_medium']) else 0
            short_vol = df.loc[idx, 'volatility_short'] if not pd.isna(df.loc[idx, 'volatility_short']) else 0
            medium_vol = df.loc[idx, 'volatility_medium'] if not pd.isna(df.loc[idx, 'volatility_medium']) else 0

            # 確信度の計算 (0-1の範囲)
            trend_conf = min((short_trend + medium_trend) / 0.1, 1.0)
            vol_conf = min(abs(short_vol - medium_vol) / (medium_vol + 0.001), 1.0)

            conf = (trend_conf + vol_conf) / 2
            confidence.append(conf)

        return pd.Series(confidence, index=df.index)

class AdaptiveFeatureEngineer:
    """
    市場レジーム適応型特徴量エンジニア
    """

    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.scalers = {}
        self.feature_importance = {}

    def create_features(self, df: pd.DataFrame,
                       include_regime_features: bool = True) -> pd.DataFrame:
        """
        包括的な特徴量生成

        Args:
            df: 入力データ
            include_regime_features: レジーム適応型特徴量を含める

        Returns:
            特徴量が追加されたデータ
        """
        logger.info("Starting feature engineering...")

        df = df.copy()

        # 基本的な価格特徴量
        df = self._create_price_features(df)

        # ボラティリティ特徴量
        df = self._create_volatility_features(df)

        # モメンタム特徴量
        df = self._create_momentum_features(df)

        # 出来高特徴量
        df = self._create_volume_features(df)

        # 市場構造特徴量
        df = self._create_market_structure_features(df)

        # レジーム適応型特徴量
        if include_regime_features:
            df = self.regime_detector.detect_regime(df)
            df = self._create_regime_adaptive_features(df)

        # 統計的特徴量
        df = self._create_statistical_features(df)

        # 特徴量のクリーニング
        df = self._clean_features(df)

        logger.info(f"Feature engineering completed. Total features: {len(df.columns)}")
        return df

    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        基本的な価格ベースの特徴量
        """
        if 'close' not in df.columns:
            return df

        # 価格変化率
        df['price_change_1d'] = df['close'].pct_change(1)
        df['price_change_3d'] = df['close'].pct_change(3)
        df['price_change_5d'] = df['close'].pct_change(5)
        df['price_change_10d'] = df['close'].pct_change(10)

        # 対数リターン
        df['log_return_1d'] = np.log(df['close'] / df['close'].shift(1))
        df['log_return_3d'] = np.log(df['close'] / df['close'].shift(3))
        df['log_return_5d'] = np.log(df['close'] / df['close'].shift(5))

        # 価格レベル (高値/安値からの距離)
        if 'high' in df.columns and 'low' in df.columns:
            df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            df['high_low_ratio'] = df['high'] / df['low']

        # 移動平均とその関係
        for window in [5, 10, 20, 50]:
            ma_col = f'sma_{window}'
            df[ma_col] = df['close'].rolling(window=window).mean()

            # 価格と移動平均の乖離
            df[f'price_ma_{window}_diff'] = (df['close'] - df[ma_col]) / df[ma_col]

            # 移動平均の傾き
            df[f'ma_{window}_slope'] = df[ma_col].diff(5) / 5

        return df

    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ボラティリティ関連の特徴量
        """
        if 'close' not in df.columns:
            return df

        returns = df['close'].pct_change()

        # 様々な期間のボラティリティ
        for window in [5, 10, 20, 30]:
            df[f'volatility_{window}d'] = returns.rolling(window=window).std()
            df[f'volatility_skew_{window}d'] = returns.rolling(window=window).skew()
            df[f'volatility_kurt_{window}d'] = returns.rolling(window=window).kurt()

        # ボラティリティの変化
        df['volatility_trend'] = df['volatility_20d'].pct_change(5)

        # レンジベースのボラティリティ
        if 'high' in df.columns and 'low' in df.columns:
            df['daily_range'] = (df['high'] - df['low']) / df['close']
            df['range_volatility'] = df['daily_range'].rolling(window=20).std()

        # ギャップの検知
        if 'open' in df.columns:
            df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
            df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)
            df['gap_size'] = abs(df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        return df

    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        モメンタム関連の特徴量
        """
        if 'close' not in df.columns:
            return df

        # RSI (Relative Strength Index)
        for window in [6, 14, 21]:
            df[f'rsi_{window}'] = self._calculate_rsi(df['close'], window)

        # MACD (Moving Average Convergence Divergence)
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Stochastic Oscillator
        if 'high' in df.columns and 'low' in df.columns:
            for window in [14, 21]:
                df[f'stoch_k_{window}'], df[f'stoch_d_{window}'] = self._calculate_stochastic(
                    df['close'], df['high'], df['low'], window
                )

        # Williams %R
        if 'high' in df.columns and 'low' in df.columns:
            for window in [14, 21]:
                df[f'williams_r_{window}'] = self._calculate_williams_r(
                    df['close'], df['high'], df['low'], window
                )

        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)

        return df

    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        出来高関連の特徴量
        """
        if 'volume' not in df.columns:
            return df

        # 出来高の変化率
        df['volume_change_1d'] = df['volume'].pct_change(1)
        df['volume_change_5d'] = df['volume'].pct_change(5)

        # 移動平均出来高
        for window in [5, 10, 20]:
            df[f'volume_ma_{window}'] = df['volume'].rolling(window=window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_ma_{window}']

        # 出来高のボラティリティ
        df['volume_volatility'] = df['volume'].pct_change().rolling(window=20).std()

        # OBV (On Balance Volume)
        df['obv'] = self._calculate_obv(df['close'], df['volume'])

        # Volume Price Trend
        df['vpt'] = self._calculate_vpt(df['close'], df['volume'])

        return df

    def _create_market_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        市場構造関連の特徴量
        """
        if 'close' not in df.columns:
            return df

        # サポート/レジスタンスレベル
        df['support_level'] = df['low'].rolling(window=20).min()
        df['resistance_level'] = df['high'].rolling(window=20).max()

        # 価格の位置づけ
        df['distance_from_support'] = (df['close'] - df['support_level']) / df['support_level']
        df['distance_from_resistance'] = (df['resistance_level'] - df['close']) / df['resistance_level']

        # トレンド継続性
        df['higher_highs'] = (df['high'] > df['high'].shift(1)).rolling(window=5).sum()
        df['higher_lows'] = (df['low'] > df['low'].shift(1)).rolling(window=5).sum()
        df['lower_highs'] = (df['high'] < df['high'].shift(1)).rolling(window=5).sum()
        df['lower_lows'] = (df['low'] < df['low'].shift(1)).rolling(window=5).sum()

        return df

    def _create_regime_adaptive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        レジーム適応型の特徴量
        """
        if 'regime' not in df.columns:
            return df

        # レジームごとの特徴量重み付け
        regime_weights = {
            'bull': {'momentum': 1.2, 'volatility': 0.8, 'volume': 1.0},
            'bear': {'momentum': 1.2, 'volatility': 0.8, 'volume': 1.0},
            'volatile': {'momentum': 0.7, 'volatility': 1.5, 'volume': 1.3},
            'sideways': {'momentum': 0.8, 'volatility': 1.0, 'volume': 0.9},
            'mixed': {'momentum': 1.0, 'volatility': 1.0, 'volume': 1.0}
        }

        # レジーム適応型複合特徴量
        for regime in df['regime'].unique():
            if regime not in regime_weights:
                continue

            mask = df['regime'] == regime
            weights = regime_weights[regime]

            # 適応型モメンタムスコア
            if 'rsi_14' in df.columns and 'macd' in df.columns:
                df.loc[mask, f'momentum_score_{regime}'] = (
                    df.loc[mask, 'rsi_14'] * weights['momentum'] +
                    df.loc[mask, 'macd'] * weights['momentum']
                ) / 2

            # 適応型リスクスコア
            if 'volatility_20d' in df.columns and 'volume_volatility' in df.columns:
                df.loc[mask, f'risk_score_{regime}'] = (
                    df.loc[mask, 'volatility_20d'] * weights['volatility'] +
                    df.loc[mask, 'volume_volatility'] * weights['volume']
                ) / 2

        # レジーム遷移特徴量
        df['regime_change'] = (df['regime'] != df['regime'].shift(1)).astype(int)
        df['regime_persistence'] = df['regime'].groupby((df['regime'] != df['regime'].shift(1)).cumsum()).cumcount()

        return df

    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        統計的特徴量
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Z-score正規化特徴量
        for col in numeric_cols:
            if col in ['close', 'volume'] or 'price' in col.lower():
                df[f'{col}_zscore'] = stats.zscore(df[col].fillna(df[col].mean()), nan_policy='omit')

        # パーセンタイル特徴量
        for col in ['close', 'volume']:
            if col in df.columns:
                df[f'{col}_percentile'] = df[col].rank(pct=True)

        return df

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量のクリーニング
        """
        # 無限値の除去
        df = df.replace([np.inf, -np.inf], np.nan)

        # 数値カラムのみを処理
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # 極端な外れ値のクリッピング (Z-score > 5)
        for col in numeric_cols:
            if df[col].std() > 0:  # 分散が0でない場合
                z_scores = np.abs(stats.zscore(df[col].fillna(df[col].mean())))
                df.loc[z_scores > 5, col] = np.nan

        # 欠損値の線形補完 (数値カラムのみ)
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')

        # まだ欠損している場合は平均値で補完 (数値カラムのみ)
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        return df

    def _calculate_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        """
        RSIの計算
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_stochastic(self, close: pd.Series, high: pd.Series,
                            low: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillatorの計算
        """
        lowest_low = low.rolling(window=window).min()
        highest_high = high.rolling(window=window).max()

        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=3).mean()

        return k, d

    def _calculate_williams_r(self, close: pd.Series, high: pd.Series,
                            low: pd.Series, window: int) -> pd.Series:
        """
        Williams %Rの計算
        """
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()

        return -100 * ((highest_high - close) / (highest_high - lowest_low))

    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On Balance Volumeの計算
        """
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]

        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]

        return obv

    def _calculate_vpt(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Volume Price Trendの計算
        """
        price_change = close.pct_change()
        vpt = (price_change * volume).cumsum()
        return vpt

    def scale_features(self, df: pd.DataFrame, method: str = 'robust') -> pd.DataFrame:
        """
        特徴量のスケーリング

        Args:
            df: 入力データ
            method: スケーリング方法 ('standard', 'robust', 'minmax')

        Returns:
            スケーリングされたデータ
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            return df  # デフォルトではスケーリングしない

        # スケーラーを保存
        self.scalers[method] = scaler

        # スケーリング
        scaled_data = scaler.fit_transform(df[numeric_cols])
        df[numeric_cols] = scaled_data

        logger.info(f"Features scaled using {method} method")
        return df

    def select_features(self, df: pd.DataFrame, method: str = 'correlation',
                       threshold: float = 0.95) -> pd.DataFrame:
        """
        特徴量選択

        Args:
            df: 入力データ
            method: 選択方法 ('correlation', 'variance', 'pca')
            threshold: 相関係数の閾値 (correlationの場合)

        Returns:
            選択された特徴量
        """
        if method == 'correlation':
            # 数値カラムのみで相関係数を計算
            numeric_df = df.select_dtypes(include=[np.number])
            corr_matrix = numeric_df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

            df_selected = df.drop(to_drop, axis=1)
            logger.info(f"Removed {len(to_drop)} highly correlated features")

        elif method == 'variance':
            # 分散が低い特徴量を除去
            variances = df.var()
            to_keep = variances[variances > threshold].index
            df_selected = df[to_keep]
            logger.info(f"Kept {len(to_keep)} features with variance > {threshold}")

        elif method == 'pca':
            # PCAによる次元削減
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            pca = PCA(n_components=threshold if isinstance(threshold, int) else 0.95)
            pca_data = pca.fit_transform(df[numeric_cols])

            # PCA成分をデータフレームに
            pca_cols = [f'pca_{i}' for i in range(pca_data.shape[1])]
            df_pca = pd.DataFrame(pca_data, columns=pca_cols, index=df.index)
            df_selected = pd.concat([df_pca, df.select_dtypes(exclude=[np.number])], axis=1)

            logger.info(f"PCA reduced to {len(pca_cols)} components")

        else:
            df_selected = df

        return df_selected

def main():
    """
    メイン実行関数
    """
    engineer = AdaptiveFeatureEngineer()

    # 最新のデータを読み込み
    data_files = list(Path("data").glob("btc_jpy_yahoo_real_*.csv"))
    if not data_files:
        logger.error("No BTC/JPY data files found")
        return

    latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Processing {latest_file}")

    # データ読み込み
    df = pd.read_csv(latest_file, index_col=0, parse_dates=True)

    # 特徴量生成
    df_featured = engineer.create_features(df)

    # 特徴量選択
    df_selected = engineer.select_features(df_featured, method='correlation', threshold=0.95)

    # スケーリング
    df_scaled = engineer.scale_features(df_selected, method='robust')

    # 保存
    output_file = latest_file.stem + "_featured.csv"
    df_featured.to_csv(f"data/{output_file}")
    logger.info(f"Featured data saved to data/{output_file}")

    output_file_selected = latest_file.stem + "_featured_selected.csv"
    df_selected.to_csv(f"data/{output_file_selected}")
    logger.info(f"Selected features saved to data/{output_file_selected}")

    output_file_scaled = latest_file.stem + "_featured_scaled.csv"
    df_scaled.to_csv(f"data/{output_file_scaled}")
    logger.info(f"Scaled features saved to data/{output_file_scaled}")

    # 特徴量統計の保存
    feature_stats = {
        'total_features': len(df_featured.columns),
        'selected_features': len(df_selected.columns),
        'regime_distribution': df_featured['regime'].value_counts().to_dict() if 'regime' in df_featured.columns else {},
        'feature_types': {
            'numeric': len(df_featured.select_dtypes(include=[np.number]).columns),
            'categorical': len(df_featured.select_dtypes(include=['object']).columns)
        }
    }

    with open(f"data/{latest_file.stem}_feature_stats.json", 'w', encoding='utf-8') as f:
        json.dump(feature_stats, f, indent=2, ensure_ascii=False)

    logger.info("Feature engineering completed successfully!")
    logger.info(f"Generated {len(df_featured.columns)} features from {len(df.columns)} original columns")

if __name__ == "__main__":
    main()