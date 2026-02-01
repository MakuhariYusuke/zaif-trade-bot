#!/usr/bin/env python3
"""
Causal Inference Feature Selection for SAC v422
因果推論ベースの特徴量選択システム

実装内容:
- 因果効果推定による特徴量重要度評価
- 介入分析による特徴量選択
- メモリ効率的な因果推定
- 混同行列除去
"""

import gc
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Lazy import sklearn to avoid Windows SIGINT issues
_SKIP_SKLEARN = os.getenv("SKIP_HEAVY_IMPORTS") == "1" or os.getenv("ZTB_SKIP_SKLEARN") == "1"
if _SKIP_SKLEARN:
    LinearRegression = None  # type: ignore
    r2_score = None  # type: ignore
    StandardScaler = None  # type: ignore
else:
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        from sklearn.preprocessing import StandardScaler
    except Exception:
        LinearRegression = None  # type: ignore
        r2_score = None  # type: ignore
        StandardScaler = None  # type: ignore

from ztb.utils.logging_utils import get_logger
from ztb.utils.memory.dtypes import optimize_dtypes

logger = get_logger(__name__)


class CausalFeatureSelector:
    """因果推論ベースの特徴量選択器"""

    def __init__(
        self,
        treatment_threshold: float = 0.1,
        min_samples: int = 1000,
        max_features: Optional[int] = None,
        memory_manager=None,
    ):
        """
        Args:
            treatment_threshold: 治療効果の閾値
            min_samples: 最小サンプル数
            max_features: 最大特徴量数
            memory_manager: メモリマネージャー
        """
        self.treatment_threshold = treatment_threshold
        self.min_samples = min_samples
        self.max_features = max_features
        self.memory_manager = memory_manager

        # 因果推定モデル (sklearn available check)
        if LinearRegression is not None and StandardScaler is not None:
            self.causal_model = LinearRegression()
            self.scaler = StandardScaler()
        else:
            self.causal_model = None
            self.scaler = None
            logger.warning("sklearn not available, causal inference disabled")

        # 結果キャッシュ
        self.causal_effects: Dict[str, float] = {}
        self.selected_features: List[str] = []

        logger.info(
            f"Initialized CausalFeatureSelector with treatment_threshold={treatment_threshold}"
        )

    def estimate_causal_effect(
        self,
        df: pd.DataFrame,
        treatment_feature: str,
        outcome_feature: str,
        confounders: List[str],
    ) -> Dict[str, float]:
        """
        因果効果を推定

        Args:
            df: データフレーム
            treatment_feature: 治療変数（介入する特徴量）
            outcome_feature: 結果変数（通常は報酬）
            confounders: 交絡因子

        Returns:
            因果効果推定結果
        """
        try:
            # データ準備
            available_features = [
                f
                for f in [treatment_feature] + confounders + [outcome_feature]
                if f in df.columns
            ]

            if len(available_features) < 2:
                return {"effect": 0.0, "p_value": 1.0, "confidence": 0.0}

            data = df[available_features].dropna()

            if len(data) < self.min_samples:
                return {"effect": 0.0, "p_value": 1.0, "confidence": 0.0}

            # Check if sklearn models are available
            if self.causal_model is None or self.scaler is None:
                logger.warning("sklearn not available, returning zero effect")
                return {"effect": 0.0, "p_value": 1.0, "confidence": 0.0}

            # スケーリング
            scaled_data = self.scaler.fit_transform(data)
            scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

            # 治療変数と結果変数
            X = scaled_df.drop(columns=[outcome_feature])
            y = scaled_df[outcome_feature]

            # 因果効果推定（単純な線形回帰ベース）
            self.causal_model.fit(X, y)
            effect = self.causal_model.coef_[X.columns.get_loc(treatment_feature)]

            # R²スコアをconfidenceとして使用
            r2 = r2_score(y, self.causal_model.predict(X))
            confidence = max(0.0, min(1.0, r2))

            # p値の推定（簡易版）
            # 実際には統計的検定が必要だが、ここでは効果の絶対値で代用
            p_value = 1.0 / (1.0 + abs(effect) * confidence)

            return {
                "effect": float(effect),
                "p_value": float(p_value),
                "confidence": float(confidence),
                "n_samples": len(data),
            }

        except Exception as e:
            logger.warning(
                f"Failed to estimate causal effect for {treatment_feature}: {e}"
            )
            return {"effect": 0.0, "p_value": 1.0, "confidence": 0.0}

    def select_features_causal(
        self,
        df: pd.DataFrame,
        features: List[str],
        outcome_feature: str = "reward",
        confounders: Optional[List[str]] = None,
    ) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
        """
        因果推論による特徴量選択

        Args:
            df: 特徴量を含むDataFrame
            features: 候補特徴量リスト
            outcome_feature: 結果変数名
            confounders: 交絡因子リスト

        Returns:
            (選択された特徴量リスト, 因果効果辞書)
        """
        if confounders is None:
            # デフォルトの交絡因子（価格、出来高関連）
            confounders = []
            for col in df.columns:
                if any(
                    keyword in col.lower()
                    for keyword in ["price", "volume", "close", "high", "low"]
                ):
                    confounders.append(col)
            confounders = confounders[:5]  # 最大5個に制限

        logger.info(
            f"Selecting features using causal inference with {len(features)} candidates"
        )

        # メモリ最適化：大規模データの場合はdtype最適化
        if len(df) > 5000 and self.memory_manager:
            try:
                optimized_df, _ = optimize_dtypes(df)
                work_df = optimized_df
                self.memory_manager.log_memory_usage(
                    "causal_feature_selection_optimization"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to optimize dtypes: {e}, using original dataframe"
                )
                work_df = df
        else:
            work_df = df

        # 各特徴量の因果効果を推定
        causal_results = {}
        for i, feature in enumerate(features):
            if feature in work_df.columns and feature != outcome_feature:
                effect_result = self.estimate_causal_effect(
                    work_df, feature, outcome_feature, confounders
                )
                causal_results[feature] = effect_result

                # 定期的なメモリログとGC
                if self.memory_manager and (i + 1) % 10 == 0:
                    self.memory_manager.log_memory_usage(
                        "causal_feature_selection_progress"
                    )
                    gc.collect()

        # 因果効果に基づいて特徴量をランク付け
        feature_scores = []
        for feature, result in causal_results.items():
            effect = result.get("effect", 0.0)
            confidence = result.get("confidence", 0.0)
            p_value = result.get("p_value", 1.0)

            # スコア計算: 効果 × 信頼性 × (1 - p値)
            score = abs(effect) * confidence * (1.0 - p_value)
            feature_scores.append((feature, score, result))

        # スコアでソート
        feature_scores.sort(key=lambda x: x[1], reverse=True)

        # 閾値以上の特徴量を選択
        selected_features = []
        selected_results = {}

        for feature, score, result in feature_scores:
            if score >= self.treatment_threshold:
                selected_features.append(feature)
                selected_results[feature] = result

                if self.max_features and len(selected_features) >= self.max_features:
                    break

        # 結果をキャッシュ
        self.causal_effects = {k: v for k, v in causal_results.items()}
        self.selected_features = selected_features

        logger.info(f"Selected {len(selected_features)} features via causal inference")
        return selected_features, selected_results

    def get_feature_importance(self) -> Dict[str, float]:
        """
        特徴量の重要度を取得

        Returns:
            特徴量名 -> 重要度スコアの辞書
        """
        importance = {}
        for feature, result in self.causal_effects.items():
            effect = result.get("effect", 0.0)
            confidence = result.get("confidence", 0.0)
            importance[feature] = abs(effect) * confidence

        return importance

    def update_causal_model(
        self, new_data: pd.DataFrame, outcome_feature: str = "reward"
    ):
        """
        因果モデルを新しいデータで更新

        Args:
            new_data: 新しいデータ
            outcome_feature: 結果変数名
        """
        try:
            # メモリ最適化
            if self.memory_manager:
                optimized_data, _ = optimize_dtypes(new_data)
                self.memory_manager.log_memory_usage("causal_model_update")
            else:
                optimized_data = new_data

            # 既存の因果効果を更新
            if self.selected_features:
                _, updated_results = self.select_features_causal(
                    optimized_data, self.selected_features, outcome_feature
                )
                self.causal_effects.update(updated_results)

            # メモリ解放
            del optimized_data
            gc.collect()

            logger.info("Updated causal model with new data")

        except Exception as e:
            logger.error(f"Failed to update causal model: {e}")


class CausalInferenceEngine:
    """因果推論エンジン"""

    def __init__(self, config: Optional[Dict[str, any]] = None, memory_manager=None):
        """
        Args:
            config: 設定辞書
            memory_manager: メモリマネージャー
        """
        if config is None:
            config = {}

        self.selector = CausalFeatureSelector(
            treatment_threshold=config.get("treatment_threshold", 0.1),
            min_samples=config.get("min_samples", 1000),
            max_features=config.get("max_features"),
            memory_manager=memory_manager,
        )

        self.config = config
        self.memory_manager = memory_manager

        logger.info("Initialized CausalInferenceEngine")

    def analyze_causal_relationships(
        self, df: pd.DataFrame, features: List[str], outcome_feature: str = "reward"
    ) -> Dict[str, any]:
        """
        因果関係を分析

        Args:
            df: データフレーム
            features: 分析対象特徴量
            outcome_feature: 結果変数

        Returns:
            分析結果
        """
        try:
            # 特徴量選択
            selected_features, causal_results = self.selector.select_features_causal(
                df, features, outcome_feature
            )

            # 重要度取得
            importance = self.selector.get_feature_importance()

            result = {
                "selected_features": selected_features,
                "causal_effects": causal_results,
                "feature_importance": importance,
                "n_candidates": len(features),
                "n_selected": len(selected_features),
            }

            logger.info(
                f"Causal analysis completed: {len(selected_features)} features selected"
            )
            return result

        except Exception as e:
            logger.error(f"Causal analysis failed: {e}")
            return {
                "selected_features": [],
                "causal_effects": {},
                "feature_importance": {},
                "error": str(e),
            }

    def update_model(self, new_data: pd.DataFrame, outcome_feature: str = "reward"):
        """
        モデルを更新

        Args:
            new_data: 新しいデータ
            outcome_feature: 結果変数
        """
        self.selector.update_causal_model(new_data, outcome_feature)


def create_causal_engine(
    config: Optional[Dict[str, any]] = None, memory_manager=None
) -> CausalInferenceEngine:
    """
    因果推論エンジンを作成

    Args:
        config: 設定辞書
        memory_manager: メモリマネージャー

    Returns:
        CausalInferenceEngineインスタンス
    """
    return CausalInferenceEngine(config, memory_manager)
