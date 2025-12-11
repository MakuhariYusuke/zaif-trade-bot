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
from ztb.features.global_market import GlobalMarketFeatureEngineer
from ztb.features.models.sac.sac_v427_feature_engineering import SACv427FeatureEngineer
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class UnifiedFeatureEngineer:
    """
    統合特徴量エンジニアリングクラス

    すべての特徴量関連機能を統一的に管理し、
    モデル固有の特徴量生成から汎用的な特徴量計算までをサポート
    """

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None
    ):
        """
        初期化

        Args:
            config: Training configuration dictionary
            config_path: 設定ファイルのパス (deprecated, use config)
        """
        self.config = config or {}
        self.config_path = config_path or "configs/features.yaml"
        self.registry = FeatureRegistry()
        self.sac_engineer = SACv427FeatureEngineer(
            config=self.config, config_path=config_path
        )
        self.global_engineer = GlobalMarketFeatureEngineer()

        # 初期化
        self.registry.initialize()
        logger.info("UnifiedFeatureEngineer initialized")

    def generate_features(
        self,
        df: pd.DataFrame,
        feature_set: str = "curated",
        model_type: str = "generic",
        external_data: Optional[pd.DataFrame] = None,
        external_suffix: str = "_global",
        **kwargs,
    ) -> pd.DataFrame:
        """
        特徴量を生成

        Args:
            df: 入力データフレーム
            feature_set: 特徴量セット名 ("curated", "full", "minimal")
            model_type: モデルタイプ ("generic", "sac", "v437")
            external_data: 外部市場データ (Optional)
            external_suffix: 外部データのサフィックス
            **kwargs: 追加パラメータ

        Returns:
            特徴量が追加されたデータフレーム
        """
        import time

        start = time.perf_counter()
        logger.info(f"Generating features with set: {feature_set}, model: {model_type}")

        # 外部データが提供された場合、マージしてLead-Lag特徴量を生成
        if external_data is not None:
            logger.info(f"Merging external data with suffix: {external_suffix}")
            df = self.global_engineer.merge_external_data(
                df, external_data, suffix=external_suffix
            )
            lead_lag_features = self.global_engineer.generate_lead_lag_features(
                df, global_col=f"close{external_suffix}"
            )
            df = pd.concat([df, lead_lag_features], axis=1)
            logger.info(
                f"Added {len(lead_lag_features.columns)} global market features"
            )

        if model_type.lower() == "sac":
            # SACモデル固有の特徴量生成
            out = self._generate_sac_features(df, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(
                f"Feature generation (SAC) completed in {elapsed:.3f}s; features={len(out.columns)}"
            )
            return out
        elif model_type.lower() == "v437":
            # v437モデル固有の特徴量生成
            return self._generate_v437_features(df, **kwargs)
        else:
            # 汎用特徴量生成
            out = self._generate_generic_features(df, feature_set, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(
                f"Feature generation (generic) completed in {elapsed:.3f}s; features={len(out.columns)}"
            )
            return out

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


class V4FeatureExtractor:
    """
    V4 Feature Extractor for SAC models

    SACモデル向けのV4特徴量抽出器
    短期間収益性向上のための最適化された特徴量抽出
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化

        Args:
            config: 設定辞書
        """
        self.config = config or {}
        self.unified_engineer = UnifiedFeatureEngineer(config=self.config)

        # FeatureRegistryの初期化
        from ztb.features.core.registry import FeatureRegistry

        FeatureRegistry.initialize()

        # scalping featuresをインポートして登録
        try:
            import ztb.features.scalping  # noqa: F401

            logger.info("Scalping features loaded")
        except ImportError as e:
            logger.warning(f"Failed to load scalping features: {e}")

        logger.info("V4FeatureExtractor initialized")

    def extract_features(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        特徴量抽出

        Args:
            df: 入力データフレーム
            **kwargs: 追加パラメータ

        Returns:
            特徴量が追加されたデータフレーム
        """
        try:
            # Extract short-term feature parameters before passing to unified engineer
            short_term_params = {}
            for param in ["rv_window", "tv_window"]:
                if param in kwargs:
                    short_term_params[param] = kwargs.pop(param)

            # news_data を分離（SAC特徴量生成に渡さない）
            news_data = kwargs.pop("news_data", None)

            # SACモデル向けの特徴量生成
            features_df = self.unified_engineer.generate_features(
                df, feature_set="curated", model_type="sac", **kwargs
            )

            # 短期間収益性向上のための追加特徴量
            enhanced_df = self._add_short_term_features(
                features_df, news_data=news_data, **short_term_params
            )

            logger.info("V4FeatureExtractor: Features extracted successfully")
            return enhanced_df
        except Exception as e:
            logger.error(f"V4FeatureExtractor error: {e}")
            raise

    def get_feature_names(self) -> List[str]:
        """
        特徴量名を取得

        Returns:
            特徴量名のリスト
        """
        base_features = self.unified_engineer.get_available_features("sac")
        # 短期間特徴量を追加
        short_term_features = [
            "realized_volatility",
            "tick_volume_ratio",
            "order_flow_imbalance",
            "news_sentiment_score",
            "news_sentiment_intensity",
        ]
        return base_features + short_term_features

    def _add_short_term_features(
        self, df: pd.DataFrame, news_data=None, **kwargs
    ) -> pd.DataFrame:
        """
        短期間収益性向上のための特徴量を追加

        Args:
            df: ベース特徴量が追加されたデータフレーム
            **kwargs: 追加パラメータ

        Returns:
            短期間特徴量が追加されたデータフレーム
        """
        try:
            from ztb.features.core.registry import FeatureRegistry

            # Extract short-term feature parameters to avoid passing them to lower layers
            rv_window = kwargs.get("rv_window", 10)
            tv_window = kwargs.get("tv_window", 5)

            df = df.copy()

            # Realized Volatility 追加
            if "realized_volatility" in FeatureRegistry._registry:
                rv_func = FeatureRegistry._registry["realized_volatility"]
                rv_series = rv_func(df, window=rv_window)
                df["realized_volatility"] = rv_series

            # Tick Volume Ratio 追加
            if "tick_volume_ratio" in FeatureRegistry._registry:
                tv_func = FeatureRegistry._registry["tick_volume_ratio"]
                tv_series = tv_func(df, window=tv_window)
                df["tick_volume_ratio"] = tv_series

            # Order Flow Imbalance 追加
            if "order_flow_imbalance" in FeatureRegistry._registry:
                of_func = FeatureRegistry._registry["order_flow_imbalance"]
                of_series = of_func(df)
                df["order_flow_imbalance"] = of_series

            # ニュース感情スコア統合 (オプション)
            df = self._add_news_sentiment_features(df, news_data=news_data, **kwargs)

            logger.info("Added short-term enhanced features")
            return df

        except Exception as e:
            logger.warning(f"Failed to add short-term features: {e}")
            return df

    def _add_news_sentiment_features(
        self, df: pd.DataFrame, news_data=None, **kwargs
    ) -> pd.DataFrame:
        """
        ニュース感情スコア特徴量を追加

        Args:
            df: 入力データフレーム
            news_data: ニュースデータ（リストまたはDataFrame）
            **kwargs: 追加パラメータ

        Returns:
            ニュース感情特徴量が追加されたデータフレーム
        """
        try:
            if news_data is None or len(news_data) == 0:
                return df

            from ztb.multimodal.features.news_feature_processor import (
                NewsFeatureProcessor,
            )

            processor = NewsFeatureProcessor()

            if isinstance(news_data, list):
                # ニューステキストのリストの場合
                sentiment_features = processor.extract_sentiment_features(news_data)

                if sentiment_features:
                    df = df.copy()
                    # ニュース感情を全期間に適用（簡易実装）
                    # 本来は時間軸でマップすべき
                    n_periods = len(df)
                    n_news = len(news_data)

                    if n_news > 0:
                        # ニュースを期間に分配
                        scores_per_period = n_periods // n_news
                        remainder = n_periods % n_news

                        sentiment_scores = []
                        sentiment_intensities = []

                        for i in range(n_news):
                            score = (
                                sentiment_features.get("financial_sentiment", [0.0])[i]
                                if i
                                < len(sentiment_features.get("financial_sentiment", []))
                                else 0.0
                            )
                            intensity = (
                                sentiment_features.get("sentiment_intensity", [0.0])[i]
                                if i
                                < len(sentiment_features.get("sentiment_intensity", []))
                                else 0.0
                            )

                            # このニュースの期間数
                            periods_for_news = scores_per_period + (
                                1 if i < remainder else 0
                            )

                            sentiment_scores.extend([score] * periods_for_news)
                            sentiment_intensities.extend([intensity] * periods_for_news)

                        # 長さを調整
                        if len(sentiment_scores) > n_periods:
                            sentiment_scores = sentiment_scores[:n_periods]
                            sentiment_intensities = sentiment_intensities[:n_periods]
                        elif len(sentiment_scores) < n_periods:
                            sentiment_scores.extend(
                                [0.0] * (n_periods - len(sentiment_scores))
                            )
                            sentiment_intensities.extend(
                                [0.0] * (n_periods - len(sentiment_intensities))
                            )

                        df["news_sentiment_score"] = sentiment_scores
                        df["news_sentiment_intensity"] = sentiment_intensities

                        logger.info("Added news sentiment features (distributed)")

            return df

        except Exception as e:
            logger.warning(f"Failed to add news sentiment features: {e}")
            return df
