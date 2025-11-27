#!/usr/bin/env python3
"""
Adaptive Feature Selection for SAC v422
市場状態に応じた動的特徴量選択システム

実装内容:
- 市場状態分類 (トレンド/レンジ/高ボラティリティ/低ボラティリティ)
- Attention-based feature weighting
- 動的特徴量重み付け
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from ztb.features.attention_trainer import AttentionTrainer, FeatureAttentionLayer
except Exception:
    # Allow module import to succeed even if torch isn't available/initializable.
    AttentionTrainer = None  # type: ignore
    FeatureAttentionLayer = None  # type: ignore
from ztb.features.causal_inference import CausalInferenceEngine

logger = logging.getLogger(__name__)


class MarketRegimeClassifier:
    """市場状態分類器"""

    def __init__(
        self, adx_threshold: float = 25.0, volatility_percentile: float = 70.0
    ):
        """
        Args:
            adx_threshold: ADXトレンド判定閾値
            volatility_percentile: ボラティリティ判定パーセンタイル
        """
        self.adx_threshold = adx_threshold
        self.volatility_percentile = volatility_percentile
        self.volatility_history: List[float] = []

    def classify_market_regime(self, df: pd.DataFrame) -> str:
        """
        市場状態を分類

        Returns:
            - "trending": トレンド相場
            - "ranging": レンジ相場
            - "high_volatility": 高ボラティリティ
            - "low_volatility": 低ボラティリティ
        """
        if "ADX" not in df.columns:
            # ADXがない場合はATRベースで判定
            if "ATR" in df.columns:
                current_atr = df["ATR"].iloc[-1]
                self.volatility_history.append(current_atr)

                if len(self.volatility_history) > 100:
                    self.volatility_history = self.volatility_history[-100:]

                if len(self.volatility_history) >= 20:
                    volatility_threshold = np.percentile(
                        self.volatility_history, self.volatility_percentile
                    )
                    if current_atr > volatility_threshold:
                        return "high_volatility"
                    elif current_atr < np.percentile(
                        self.volatility_history, 100 - self.volatility_percentile
                    ):
                        return "low_volatility"

            return "ranging"  # デフォルト

        # ADXベースの判定
        adx = df["ADX"].iloc[-1]

        # ボラティリティ判定
        if "ATR" in df.columns:
            current_atr = df["ATR"].iloc[-1]
            self.volatility_history.append(current_atr)

            if len(self.volatility_history) > 100:
                self.volatility_history = self.volatility_history[-100:]

            if len(self.volatility_history) >= 20:
                volatility_threshold = np.percentile(
                    self.volatility_history, self.volatility_percentile
                )
                if current_atr > volatility_threshold:
                    return "high_volatility"
                elif current_atr < np.percentile(
                    self.volatility_history, 100 - self.volatility_percentile
                ):
                    return "low_volatility"

        # トレンド判定
        if adx > self.adx_threshold:
            return "trending"
        else:
            return "ranging"


class FeaturesAdaptiveFeatureSelector:
    """適応型特徴量選択器"""

    def __init__(self, feature_groups: Optional[Dict[str, List[str]]] = None):
        """
        Args:
            feature_groups: 市場状態ごとの特徴量グループ
        """
        self.classifier = MarketRegimeClassifier()

        # デフォルトの特徴量グループ
        if feature_groups is None:
            self.feature_groups = {
                "trending": [
                    "adx_14",
                    "macd",
                    "macd_signal",
                    "macd_hist",
                    "ema_5",
                    "ema_10",
                    "ema_20",
                    "ema_50",
                    "sma_20",
                    "sma_50",
                    "trend_strength",
                    "momentum",
                ],
                "ranging": [
                    "rsi_6",
                    "rsi_14",
                    "rsi_21",
                    "stoch_k",
                    "stoch_d",
                    "cci_14",
                    "willr_14",
                    "roc_10",
                    "mfi_14",
                    "bb_upper",
                    "bb_lower",
                    "bb_middle",
                    "oscillator_strength",
                ],
                "high_volatility": [
                    "atr_14",
                    "natr_14",
                    "trange",
                    "bb_width",
                    "kc_width",
                    "volatility_ratio",
                    "price_change_1d",
                ],
                "low_volatility": [
                    "price_change_1d",
                    "volume_ratio",
                    "micro_trend",
                    "precision_signals",
                ],
            }
        else:
            self.feature_groups = feature_groups

        # Attention layer and trainer
        self.attention_layer = None
        self.attention_trainer: Optional[AttentionTrainer] = None
        self.causal_engine: Optional[CausalInferenceEngine] = None
        self.feature_scaler = StandardScaler()

    def get_regime_weights(self, regime: str) -> Dict[str, float]:
        """
        市場状態に応じた特徴量重みを取得

        Args:
            regime: 市場状態 ("trending", "ranging", "high_volatility", "low_volatility")

        Returns:
            特徴量名 -> 重みの辞書
        """
        weights = {}

        # 優先グループの特徴量に高い重み
        if regime in self.feature_groups:
            for feature in self.feature_groups[regime]:
                weights[feature] = 1.0

        # 他の特徴量には低い重み
        default_weight = 0.3
        for group_features in self.feature_groups.values():
            for feature in group_features:
                if feature not in weights:
                    weights[feature] = default_weight

        return weights

    def select_features_adaptive(
        self, df: pd.DataFrame, all_features: List[str]
    ) -> Tuple[List[str], np.ndarray]:
        """
        適応型特徴量選択

        Args:
            df: 特徴量を含むDataFrame
            all_features: 利用可能な全特徴量リスト

        Returns:
            (選択された特徴量リスト, 重み配列)
        """
        # 市場状態分類
        regime = self.classifier.classify_market_regime(df)

        # 基本重み取得
        base_weights = self.get_regime_weights(regime)

        # 利用可能な特徴量のみを対象
        available_features = [f for f in all_features if f in df.columns]
        weights = np.array([base_weights.get(f, 0.3) for f in available_features])

        # Attention-based refinement (実装中)
        if self.attention_trainer is not None:
            weights = self._apply_attention_weights(df, available_features, weights)

        # 重みでソートして上位特徴量を選択
        sorted_indices = np.argsort(weights)[::-1]
        selected_features = [available_features[i] for i in sorted_indices]

        # 最低重み以上の特徴量のみ選択
        min_weight = 0.4
        final_features = []
        final_weights = []

        for feature, weight in zip(selected_features, weights[sorted_indices]):
            if weight >= min_weight:
                final_features.append(feature)
                final_weights.append(weight)

        logger.info(f"Market regime: {regime}, Selected {len(final_features)} features")

        return final_features, np.array(final_weights)

    def select_features(
        self,
        df: pd.DataFrame,
        all_features: List[str],
        use_causal: bool = False,
        outcome_feature: str = "reward",
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        統合された特徴量選択（適応型 + 因果推論）

        Args:
            df: 特徴量を含むDataFrame
            all_features: 利用可能な全特徴量リスト
            use_causal: 因果推論を使用するか
            outcome_feature: 結果変数名（因果推論時）

        Returns:
            (選択された特徴量リスト, 選択統計)
        """
        if use_causal and self.causal_engine is not None:
            # 因果推論ベースの選択
            selected_features, stats = self.select_features_causal(
                df, all_features, outcome_feature
            )
            stats["selection_method"] = "causal"
            return selected_features, stats
        else:
            # 適応型選択
            selected_features, weights = self.select_features_adaptive(df, all_features)
            stats = {
                "selection_method": "adaptive",
                "weights": weights.tolist(),
                "n_selected": len(selected_features),
            }
            return selected_features, stats

    def initialize_attention_trainer(
        self,
        n_features: int,
        config: Optional[Dict[str, Any]] = None,
        memory_manager=None,
    ) -> None:
        """
        注意モデルトレーナーを初期化

        Args:
            n_features: 特徴量数
            config: 設定辞書
            memory_manager: メモリマネージャー
        """
        from ztb.features.attention_trainer import create_attention_trainer

        self.attention_trainer = create_attention_trainer(
            n_features=n_features, config=config, memory_manager=memory_manager
        )

        # Attention layerも初期化
        if config and config.get("enabled", False):
            hidden_dim = config.get("hidden_dim", 64)
            self.attention_layer = FeatureAttentionLayer(n_features, hidden_dim)

        # Causal inference engine
        if config and config.get("causal_enabled", False):
            causal_config = config.get("causal_config", {})
            self.causal_engine = CausalInferenceEngine(causal_config, memory_manager)

        logger.info("Initialized attention trainer and layer")

    def add_training_sample(
        self, features: np.ndarray, reward: float, regime: str
    ) -> None:
        """
        トレーニングサンプルを追加

        Args:
            features: 特徴量ベクトル
            reward: 報酬値
            regime: 市場状態
        """
        if self.attention_trainer is not None:
            self.attention_trainer.add_training_sample(features, reward, regime)

    def train_attention_model(self) -> Dict[str, Any]:
        """
        注意モデルをトレーニング

        Returns:
            トレーニング結果
        """
        if self.attention_trainer is None:
            return {"success": False, "error": "Attention trainer not initialized"}

        return self.attention_trainer.train()

    def load_attention_model(self, model_path: str) -> bool:
        """
        注意モデルを読み込み

        Args:
            model_path: モデルファイルパス

        Returns:
            読み込み成功フラグ
        """
        if self.attention_trainer is None:
            return False

        return self.attention_trainer.load_model(model_path)

    def select_features_causal(
        self, df: pd.DataFrame, features: List[str], outcome_feature: str = "reward"
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        因果推論による特徴量選択

        Args:
            df: 特徴量を含むDataFrame
            features: 候補特徴量リスト
            outcome_feature: 結果変数名

        Returns:
            (選択された特徴量リスト, 分析結果)
        """
        if self.causal_engine is None:
            logger.warning(
                "Causal engine not initialized, falling back to regime-based selection"
            )
            return self.select_features_adaptive(df, features)

        try:
            # 因果分析を実行
            analysis_result = self.causal_engine.analyze_causal_relationships(
                df, features, outcome_feature
            )

            selected_features = analysis_result.get("selected_features", [])
            causal_effects = analysis_result.get("causal_effects", {})

            logger.info(
                f"Causal feature selection: {len(selected_features)} features selected"
            )
            return selected_features, analysis_result

        except Exception as e:
            logger.error(
                f"Causal feature selection failed: {e}, falling back to regime-based"
            )
            return self.select_features_adaptive(df, features)

    def update_causal_model(
        self, new_data: pd.DataFrame, outcome_feature: str = "reward"
    ):
        """
        因果モデルを更新

        Args:
            new_data: 新しいデータ
            outcome_feature: 結果変数
        """
        if self.causal_engine is not None:
            self.causal_engine.update_model(new_data, outcome_feature)

    def _apply_attention_weights(
        self, df: pd.DataFrame, features: List[str], base_weights: np.ndarray
    ) -> np.ndarray:
        """
        Attention mechanismによる重み調整

        Args:
            df: 特徴量を含むDataFrame
            features: 特徴量名リスト
            base_weights: 基本重み配列

        Returns:
            調整された重み配列
        """
        if self.attention_trainer is None:
            logger.debug("Attention trainer not available, using base weights")
            return base_weights

        try:
            # 最新の特徴量データを取得
            latest_features = df[features].iloc[-1:].values  # (1, n_features)

            # スケーリング
            scaled_features = self.feature_scaler.fit_transform(latest_features)

            # 注意重みを取得
            attention_weights = self.attention_trainer.get_attention_weights(
                scaled_features[0]
            )

            # 基本重みと注意重みを組み合わせ
            combined_weights = 0.7 * base_weights + 0.3 * attention_weights

            logger.debug(
                f"Applied attention weights, mean: {attention_weights.mean():.3f}"
            )
            return combined_weights

        except Exception as e:
            logger.warning(
                f"Failed to apply attention weights: {e}, using base weights"
            )
            return base_weights

    def update_attention_model(self, feature_data: np.ndarray, rewards: np.ndarray):
        """
        Attention modelの学習更新 (将来実装)

        Args:
            feature_data: 特徴量データ (batch_size, n_features)
            rewards: 報酬データ (batch_size,)
        """
        # TODO: Attention layerの学習
        pass


def create_adaptive_selector() -> "AdaptiveFeatureSelector":
    """適応型特徴量選択器の作成"""
    return AdaptiveFeatureSelector()


# Backwards compatibility: alias to expected name
AdaptiveFeatureSelector = FeaturesAdaptiveFeatureSelector


# テスト用関数
def test_market_regime_classification():
    """市場状態分類のテスト"""
    # サンプルデータ作成
    dates = pd.date_range("2024-01-01", periods=100, freq="1H")
    np.random.seed(42)

    df = pd.DataFrame(
        {
            "ts": dates,
            "close": 100 + np.cumsum(np.random.normal(0, 1, 100)),
            "high": 100 + np.cumsum(np.random.normal(0, 1, 100)) + 0.5,
            "low": 100 + np.cumsum(np.random.normal(0, 1, 100)) - 0.5,
            "volume": np.random.uniform(1000, 10000, 100),
        }
    )

    # ATR計算 (簡易)
    df["TR"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1)),
        ),
    )
    df["ATR"] = df["TR"].rolling(14).mean()

    # ADX計算 (簡易)
    df["DM_plus"] = np.where(
        df["high"] - df["high"].shift(1) > df["low"].shift(1) - df["low"],
        np.maximum(df["high"] - df["high"].shift(1), 0),
        0,
    )
    df["DM_minus"] = np.where(
        df["low"].shift(1) - df["low"] > df["high"] - df["high"].shift(1),
        np.maximum(df["low"].shift(1) - df["low"], 0),
        0,
    )
    df["ADX"] = (
        100
        * (
            abs(df["DM_plus"] - df["DM_minus"])
            / (df["DM_plus"] + df["DM_minus"] + 1e-10)
        )
        .rolling(14)
        .mean()
    )

    classifier = MarketRegimeClassifier()
    regime = classifier.classify_market_regime(df)

    print(f"Detected market regime: {regime}")
    return regime


if __name__ == "__main__":
    # テスト実行
    test_market_regime_classification()
