"""
Unified Feature Engineering Interface

統合特徴量エンジニアリングインターフェース
すべての特徴量関連機能を一元管理し、統一されたAPIを提供

Features:
- 特徴量生成の一元化
- モデル固有の特徴量エンジニアリング
- 特徴量セットの管理
- 計算エンジンの統合
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ztb.features.core.engine import compute_features_batch
from ztb.features.core.registry import FeatureRegistry
from ztb.features.feature_set_manager import get_feature_set
from ztb.features.models.sac.sac_v427_feature_engineering import SACv427FeatureEngineer
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class UnifiedFeatureEngineer:
    """
    統合特徴量エンジニアリングクラス

    すべての特徴量関連機能を統一的に管理し、
    モデル固有の特徴量生成から汎用的な特徴量計算までをサポート
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        """
        初期化

        Args:
            config: Training configuration dictionary
            config_path: 設定ファイルのパス (deprecated, use config)
        """
        self.config = config or {}
        self.config_path = config_path or "configs/features.yaml"
        self.registry = FeatureRegistry()
        self.sac_engineer = SACv427FeatureEngineer(config=self.config, config_path=config_path)

        # 初期化
        self.registry.initialize()
        logger.info("UnifiedFeatureEngineer initialized")

    def generate_features(
        self,
        df: pd.DataFrame,
        feature_set: str = "curated",
        model_type: str = "generic",
        **kwargs,
    ) -> pd.DataFrame:
        """
        特徴量を生成

        Args:
            df: 入力データフレーム
            feature_set: 特徴量セット名 ("curated", "full", "minimal")
            model_type: モデルタイプ ("generic", "sac", "v437")
            **kwargs: 追加パラメータ

        Returns:
            特徴量が追加されたデータフレーム
        """
        logger.info(f"Generating features with set: {feature_set}, model: {model_type}")

        if model_type.lower() == "sac":
            # SACモデル固有の特徴量生成
            return self._generate_sac_features(df, **kwargs)
        elif model_type.lower() == "v437":
            # v437モデル固有の特徴量生成
            return self._generate_v437_features(df, **kwargs)
        else:
            # 汎用特徴量生成
            return self._generate_generic_features(df, feature_set, **kwargs)

    def _generate_sac_features(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """SACモデル用の特徴量生成"""
        try:
            # SAC v427特徴量エンジニアを使用
            features_df = self.sac_engineer.generate_v427_features(df, **kwargs)
            logger.info("Generated SAC features")
            return features_df
        except Exception as e:
            logger.error(f"Error generating SAC features: {e}")
            # フォールバックとして汎用特徴量を使用
            return self._generate_generic_features(df, "curated", **kwargs)

    def _generate_v437_features(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """v437モデル用の特徴量生成"""
        # TODO: v437固有の特徴量生成を実装
        logger.warning("v437 features not yet implemented, using generic features")
        return self._generate_generic_features(df, "full", **kwargs)

    def _generate_generic_features(
        self, df: pd.DataFrame, feature_set: str = "curated", **kwargs
    ) -> pd.DataFrame:
        """汎用特徴量生成"""
        # 特徴量セットを取得
        feature_names = get_feature_set(feature_set)

        # 弱気シグナル特徴量を追加
        bearish_features = self._generate_bearish_signal_features(df)
        df = pd.concat([df, bearish_features], axis=1)

        # 特徴量を計算
        result = compute_features_batch(df, feature_names=feature_names, **kwargs)

        # 戻り値がタプルの場合はDataFrameのみを返す
        if isinstance(result, tuple):
            features_df, _ = result
        else:
            features_df = result

        logger.info(
            f"Generated {len(feature_names)} generic features + {len(bearish_features.columns)} bearish features"
        )
        return features_df

    def get_available_features(self, model_type: str = "generic") -> List[str]:
        """
        利用可能な特徴量リストを取得

        Args:
            model_type: モデルタイプ

        Returns:
            特徴量名のリスト
        """
        if model_type.lower() == "sac":
            # SAC固有の特徴量（設定から取得）
            try:
                return get_feature_set("curated")
            except Exception:
                return []
        else:
            # レジストリから全特徴量を取得
            return list(self.registry._registry.keys())

    def get_feature_sets(self) -> Dict[str, List[str]]:
        """
        利用可能な特徴量セットを取得

        Returns:
            特徴量セットの辞書
        """
        sets = {}
        set_names = ["curated", "full", "minimal"]
        for set_name in set_names:
            try:
                sets[set_name] = get_feature_set(set_name)
            except Exception as e:
                logger.warning(f"Could not load feature set {set_name}: {e}")
                sets[set_name] = []

        return sets

    def validate_features(
        self, df: pd.DataFrame, feature_names: List[str]
    ) -> Dict[str, bool]:
        """
        特徴量の妥当性を検証

        Args:
            df: データフレーム
            feature_names: 検証する特徴量名リスト

        Returns:
            特徴量名 -> 妥当性 の辞書
        """
        results = {}
        for feature_name in feature_names:
            try:
                if feature_name in df.columns:
                    # NaNチェック
                    nan_count = df[feature_name].isna().sum()
                    if nan_count > 0:
                        logger.warning(
                            f"Feature {feature_name} has {nan_count} NaN values"
                        )
                        results[feature_name] = False
                    else:
                        results[feature_name] = True
                else:
                    logger.warning(f"Feature {feature_name} not found in dataframe")
                    results[feature_name] = False
            except Exception as e:
                logger.error(f"Error validating feature {feature_name}: {e}")
                results[feature_name] = False

        return results

    def optimize_features(
        self,
        df: pd.DataFrame,
        target_feature: str = "returns",
        method: str = "correlation",
        **kwargs,
    ) -> List[str]:
        """
        特徴量を最適化（選択）

        Args:
            df: データフレーム
            target_feature: 目的特徴量
            method: 最適化手法 ("correlation", "importance", "recursive")
            **kwargs: 追加パラメータ

        Returns:
            最適化された特徴量リスト
        """
        # TODO: 特徴量最適化を実装
        logger.warning("Feature optimization not yet implemented")
        return list(df.select_dtypes(include=[np.number]).columns)

    def _generate_bearish_signal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        弱気シグナル特化特徴量を生成

        SELL bias対策として、弱気シグナルを強化する特徴量を追加
        """
        features = pd.DataFrame(index=df.index)

        # 必要な列が存在するかチェック
        required_cols = ["close", "high", "low", "open", "volume"]
        if not all(col in df.columns for col in required_cols):
            logger.warning("Required columns missing for bearish features, skipping")
            return features

        try:
            # 1. ベアリッシュダイバージェンス特徴量
            features = self._add_bearish_divergence_features(df, features)

            # 2. 弱気ローソク足パターン
            features = self._add_bearish_candlestick_patterns(df, features)

            # 3. 弱気モメンタム指標
            features = self._add_bearish_momentum_features(df, features)

            # 4. 売りシグナルオシレーター
            features = self._add_sell_signal_oscillators(df, features)

            # 5. 弱気ボリューム指標
            features = self._add_bearish_volume_features(df, features)

            logger.info(f"Generated {len(features.columns)} bearish signal features")
            return features

        except Exception as e:
            logger.error(f"Error generating bearish signal features: {e}")
            return pd.DataFrame(index=df.index)  # 空のDataFrameを返す

    def _add_bearish_divergence_features(
        self, df: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """ベアリッシュダイバージェンス特徴量を追加"""
        try:
            # RSIダイバージェンス (価格が上昇するがRSIが下降)
            if "RSI" in df.columns:
                # 価格の短期トレンド (5期)
                price_ma5 = df["close"].rolling(5).mean()
                price_ma10 = df["close"].rolling(10).mean()
                price_trend = (price_ma5 - price_ma10).rolling(3).mean()

                # RSIのトレンド
                rsi_trend = df["RSI"].rolling(5).mean() - df["RSI"].rolling(10).mean()

                # ベアリッシュダイバージェンス: 価格は上昇傾向だがRSIは下降傾向
                features["Bearish_RSI_Divergence"] = np.where(
                    (price_trend > 0) & (rsi_trend < 0), 1, 0
                )

            # MACDダイバージェンス
            if all(col in df.columns for col in ["MACD", "close"]):
                macd_trend = (
                    df["MACD"].rolling(5).mean() - df["MACD"].rolling(10).mean()
                )
                price_trend = (
                    df["close"].rolling(5).mean() - df["close"].rolling(10).mean()
                )

                features["Bearish_MACD_Divergence"] = np.where(
                    (price_trend > 0) & (macd_trend < 0), 1, 0
                )

        except Exception as e:
            logger.warning(f"Error adding bearish divergence features: {e}")

        return features

    def _add_bearish_candlestick_patterns(
        self, df: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """弱気ローソク足パターンを追加"""
        try:
            # 弱気エングルフィングパターン
            body_size = abs(df["close"] - df["open"])
            prev_body_size = body_size.shift(1)

            # 現在の陰線が前の陽線を完全に包む
            bearish_engulfing = (
                (df["open"].shift(1) < df["close"].shift(1))
                & (df["open"] > df["close"])  # 前のローソクは陽線
                & (df["open"] >= df["close"].shift(1))  # 現在のローソクは陰線
                & (df["close"] <= df["open"].shift(1))
                & (body_size > prev_body_size)
            )
            features["Bearish_Engulfing"] = bearish_engulfing.astype(int)

            # シューティングスター (上ひげが長い)
            upper_shadow = df["high"] - np.maximum(df["open"], df["close"])
            lower_shadow = np.minimum(df["open"], df["close"]) - df["low"]
            body_size_real = abs(df["close"] - df["open"])

            shooting_star = (
                (upper_shadow > 2 * body_size_real)
                & (lower_shadow < body_size_real)
                & (df["close"] > df["open"].shift(1))  # 上昇トレンド中
            )
            features["Shooting_Star"] = shooting_star.astype(int)

            # 弱気ハンマー (下降トレンド中の下ひげが長い)
            hammer_bearish = (
                (lower_shadow > 2 * body_size_real)
                & (upper_shadow < body_size_real)
                & (df["close"] < df["open"].shift(1))  # 下降トレンド中
            )
            features["Hammer_Bearish"] = hammer_bearish.astype(int)

        except Exception as e:
            logger.warning(f"Error adding bearish candlestick patterns: {e}")

        return features

    def _add_bearish_momentum_features(
        self, df: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """弱気モメンタム指標を追加"""
        try:
            # 下降加速 (価格が加速して下降)
            returns = df["close"].pct_change()
            momentum_5 = returns.rolling(5).mean()
            momentum_10 = returns.rolling(10).mean()

            features["Bearish_Momentum_Acceleration"] = np.where(
                (momentum_5 < momentum_10) & (momentum_5 < -0.005), 1, 0
            )

            # 弱気トレンド継続
            trend_20 = (df["close"] - df["close"].shift(20)) / df["close"].shift(20)
            features["Bearish_Trend_Continuation"] = np.where(trend_20 < -0.05, 1, 0)

            # 下降中のボラティリティ増加
            if "close" in df.columns:
                volatility = returns.rolling(10).std()
                trend_direction = np.where(trend_20 < 0, 1, 0)
                features["Bearish_Volatility_Surge"] = (
                    volatility * trend_direction
                ).fillna(0)

        except Exception as e:
            logger.warning(f"Error adding bearish momentum features: {e}")

        return features

    def _add_sell_signal_oscillators(
        self, df: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """売りシグナルオシレーターを追加"""
        try:
            # Stochasticオーバーボート (売りシグナル)
            if "Stochastic" in df.columns:
                features["Stochastic_Overbought_Sell"] = np.where(
                    df["Stochastic"] > 80, 1, 0
                )

                # Stochasticベアリッシュダイバージェンス
                stoch_trend = (
                    df["Stochastic"].rolling(5).mean()
                    - df["Stochastic"].rolling(10).mean()
                )
                price_trend = (
                    df["close"].rolling(5).mean() - df["close"].rolling(10).mean()
                )

                features["Stochastic_Bearish_Divergence"] = np.where(
                    (price_trend > 0) & (stoch_trend < 0), 1, 0
                )

            # RSIオーバーボート
            if "RSI" in df.columns:
                features["RSI_Overbought_Sell"] = np.where(df["RSI"] > 70, 1, 0)

                # RSI弱気ダイバージェンス
                rsi_trend = df["RSI"].rolling(5).mean() - df["RSI"].rolling(10).mean()
                price_trend = (
                    df["close"].rolling(5).mean() - df["close"].rolling(10).mean()
                )

                features["RSI_Bearish_Divergence"] = np.where(
                    (price_trend > 0) & (rsi_trend < 0), 1, 0
                )

            # Williams %R 売りシグナル
            if "Williams_R" in df.columns:
                features["WilliamsR_Oversold_Sell"] = np.where(
                    df["Williams_R"] < -20, 1, 0
                )

        except Exception as e:
            logger.warning(f"Error adding sell signal oscillators: {e}")

        return features

    def _add_bearish_volume_features(
        self, df: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """弱気ボリューム指標を追加"""
        try:
            if "volume" in df.columns:
                # 下降時の出来高増加
                returns = df["close"].pct_change()
                volume_ma10 = df["volume"].rolling(10).mean()
                volume_ratio = df["volume"] / volume_ma10

                features["Bearish_Volume_Surge"] = np.where(
                    (returns < -0.01) & (volume_ratio > 1.5), 1, 0
                )

                # 弱気出来高トレンド
                volume_trend = volume_ma10.pct_change(5)
                features["Bearish_Volume_Trend"] = np.where(volume_trend > 0.2, 1, 0)

        except Exception as e:
            logger.warning(f"Error adding bearish volume features: {e}")

        return features


# グローバルインスタンス
_unified_engineer: Optional[UnifiedFeatureEngineer] = None


def get_unified_feature_engineer(
    config_path: Optional[str] = None,
) -> UnifiedFeatureEngineer:
    """
    統合特徴量エンジニアのグローバルインスタンスを取得

    Args:
        config_path: 設定ファイルのパス

    Returns:
        UnifiedFeatureEngineerインスタンス
    """
    global _unified_engineer
    if _unified_engineer is None:
        _unified_engineer = UnifiedFeatureEngineer(config=None, config_path=config_path)
    return _unified_engineer
    return _unified_engineer
    return _unified_engineer


def generate_features(
    df: pd.DataFrame,
    feature_set: str = "curated",
    model_type: str = "generic",
    **kwargs,
) -> pd.DataFrame:
    """
    便利関数: 特徴量を生成

    Args:
        df: 入力データフレーム
        feature_set: 特徴量セット
        model_type: モデルタイプ
        **kwargs: 追加パラメータ

    Returns:
        特徴量が追加されたデータフレーム
    """
    engineer = get_unified_feature_engineer()
    return engineer.generate_features(df, feature_set, model_type, **kwargs)


def get_available_features(model_type: str = "generic") -> List[str]:
    """
    便利関数: 利用可能な特徴量を取得

    Args:
        model_type: モデルタイプ

    Returns:
        特徴量名のリスト
    """
    engineer = get_unified_feature_engineer()
    return engineer.get_available_features(model_type)
