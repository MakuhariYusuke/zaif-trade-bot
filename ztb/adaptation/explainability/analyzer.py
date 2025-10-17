"""
Explainability Analyzer Implementation
SHAPベースのモデル解釈性分析
"""

import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

from .config import ExplainabilityConfig, ExplanationMethod, ExplanationScope
from .types import (
    ExplanationResult, FeatureImportance, DecisionExplanation,
    ExplanationType, ExplanationCache
)

logger = logging.getLogger(__name__)


class ExplainabilityAnalyzer:
    """説明可能性アナライザー"""

    def __init__(self, config: ExplainabilityConfig):
        self.config = config
        self.cache: Dict[str, ExplanationCache] = {}
        self.feature_names = config.feature_names
        self.feature_categories = config.feature_categories

        if not SHAP_AVAILABLE:
            logger.warning("SHAP library not available. Install with: pip install shap")
            if config.explanation_method == ExplanationMethod.SHAP:
                logger.warning("Falling back to basic feature importance analysis")

    def explain_prediction(self,
                          model: nn.Module,
                          input_data: Union[np.ndarray, torch.Tensor, pd.DataFrame],
                          prediction: Any = None,
                          background_data: Optional[Union[np.ndarray, torch.Tensor]] = None) -> ExplanationResult:
        """
        予測の説明を生成

        Args:
            model: 説明対象のモデル
            input_data: 入力データ
            prediction: 予測結果（オプション）
            background_data: 背景データ（SHAP用）

        Returns:
            ExplanationResult: 説明結果
        """
        start_time = time.time()
        explanation_id = str(uuid.uuid4())

        try:
            # 入力データの前処理
            input_tensor = self._preprocess_input(input_data)

            # 特徴量重要度の計算
            feature_importance = self._calculate_feature_importance(
                model, input_tensor, background_data
            )

            # 決定説明の生成
            decision_explanation = self._generate_decision_explanation(
                feature_importance, prediction
            )

            # 結果の作成
            result = ExplanationResult(
                explanation_id=explanation_id,
                timestamp=datetime.now(),
                model_version=getattr(model, '_version', 'unknown'),
                explanation_type=ExplanationType.FEATURE_IMPORTANCE,
                target_prediction=prediction,
                feature_importance=feature_importance,
                decision_explanation=decision_explanation,
                processing_time_seconds=time.time() - start_time
            )

            # キャッシュに保存
            if self.config.cache_explanations:
                self._cache_result(result)

            return result

        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            # エラーの場合でも基本的な結果を返す
            return ExplanationResult(
                explanation_id=explanation_id,
                timestamp=datetime.now(),
                model_version=getattr(model, '_version', 'unknown'),
                explanation_type=ExplanationType.FEATURE_IMPORTANCE,
                target_prediction=prediction,
                feature_importance=[],
                processing_time_seconds=time.time() - start_time,
                metadata={"error": str(e)}
            )

    def _preprocess_input(self, input_data: Union[np.ndarray, torch.Tensor, pd.DataFrame]) -> torch.Tensor:
        """入力データの前処理"""
        if isinstance(input_data, pd.DataFrame):
            # DataFrameの場合、数値データのみを使用
            numeric_data = input_data.select_dtypes(include=[np.number]).values
            return torch.tensor(numeric_data, dtype=torch.float32)
        elif isinstance(input_data, np.ndarray):
            return torch.tensor(input_data, dtype=torch.float32)
        elif isinstance(input_data, torch.Tensor):
            return input_data.float()
        else:
            raise ValueError(f"Unsupported input data type: {type(input_data)}")

    def _calculate_feature_importance(self,
                                    model: nn.Module,
                                    input_tensor: torch.Tensor,
                                    background_data: Optional[torch.Tensor] = None) -> List[FeatureImportance]:
        """特徴量重要度の計算"""
        if self.config.explanation_method == ExplanationMethod.SHAP and SHAP_AVAILABLE:
            return self._calculate_shap_importance(model, input_tensor, background_data)
        else:
            return self._calculate_basic_importance(model, input_tensor)

    def _calculate_shap_importance(self,
                                 model: nn.Module,
                                 input_tensor: torch.Tensor,
                                 background_data: Optional[torch.Tensor] = None) -> List[FeatureImportance]:
        """SHAPベースの特徴量重要度計算"""
        try:
            # モデルを評価モードに
            model.eval()

            # 背景データの準備
            if background_data is None:
                # デフォルトの背景データを生成
                background_data = torch.randn(
                    min(self.config.shap_background_samples, input_tensor.shape[0]),
                    input_tensor.shape[1]
                )

            # SHAP Explainerの作成
            def model_predict(x):
                with torch.no_grad():
                    x_tensor = torch.tensor(x, dtype=torch.float32)
                    outputs = model(x_tensor)
                    # 確率分布の場合、最も高い確率のクラスを返す
                    if outputs.dim() > 1 and outputs.shape[1] > 1:
                        return outputs.numpy()
                    else:
                        return outputs.numpy().flatten()

            explainer = shap.Explainer(model_predict, background_data.numpy())

            # SHAP値の計算
            shap_values = explainer(input_tensor.numpy(),
                                  max_evals=self.config.shap_max_evals)

            # 特徴量重要度の抽出
            if hasattr(shap_values, 'values'):
                if len(shap_values.values.shape) > 2:
                    # 多次元出力の場合、最初の次元を使用
                    importance_scores = np.abs(shap_values.values[0]).mean(axis=0)
                else:
                    importance_scores = np.abs(shap_values.values).mean(axis=0)
            else:
                importance_scores = np.abs(shap_values).mean(axis=0)

            # FeatureImportanceオブジェクトの作成
            feature_importance = []
            for i, score in enumerate(importance_scores):
                feature_name = self._get_feature_name(i)
                feature_importance.append(FeatureImportance(
                    feature_name=feature_name,
                    importance_score=float(score),
                    feature_category=self.feature_categories.get(feature_name),
                    description=self._get_feature_description(feature_name),
                    confidence=self._calculate_confidence(score, importance_scores)
                ))

            # 重要度でソート
            feature_importance.sort(key=lambda x: x.importance_score, reverse=True)

            return feature_importance[:self.config.max_features_to_explain]

        except Exception as e:
            logger.error(f"SHAP calculation failed: {e}")
            return self._calculate_basic_importance(model, input_tensor)

    def _calculate_basic_importance(self,
                                  model: nn.Module,
                                  input_tensor: torch.Tensor) -> List[FeatureImportance]:
        """基本的な特徴量重要度計算（勾配ベース）"""
        try:
            model.eval()
            input_tensor.requires_grad_(True)

            # 順伝播
            output = model(input_tensor)
            if output.dim() > 1:
                # 多クラス分類の場合
                target_class = output.argmax(dim=1)
                output = output.gather(1, target_class.unsqueeze(1)).squeeze(1)
            else:
                output = output.squeeze()

            # 逆伝播で勾配を計算
            output.backward(torch.ones_like(output))
            gradients = input_tensor.grad.abs().mean(dim=0)

            # FeatureImportanceオブジェクトの作成
            feature_importance = []
            for i, grad in enumerate(gradients):
                feature_name = self._get_feature_name(i)
                feature_importance.append(FeatureImportance(
                    feature_name=feature_name,
                    importance_score=float(grad),
                    feature_category=self.feature_categories.get(feature_name),
                    description=self._get_feature_description(feature_name)
                ))

            # 重要度でソート
            feature_importance.sort(key=lambda x: x.importance_score, reverse=True)

            return feature_importance[:self.config.max_features_to_explain]

        except Exception as e:
            logger.error(f"Basic importance calculation failed: {e}")
            return []

    def _generate_decision_explanation(self,
                                     feature_importance: List[FeatureImportance],
                                     prediction: Any = None) -> Optional[DecisionExplanation]:
        """決定説明の生成"""
        if not feature_importance or not self.config.generate_natural_language:
            return None

        try:
            # 決定タイプの判定
            decision_type = self._classify_decision(prediction)

            # 主な要因と寄与要因の分類
            primary_factors = feature_importance[:3]  # 上位3つ
            contributing_factors = feature_importance[3:8]  # 4-8位

            # 自然言語説明の生成
            natural_language = self._generate_natural_language_explanation(
                decision_type, primary_factors, contributing_factors
            )

            return DecisionExplanation(
                decision_type=decision_type,
                confidence_score=self._calculate_decision_confidence(feature_importance),
                primary_factors=primary_factors,
                contributing_factors=contributing_factors,
                natural_language_explanation=natural_language
            )

        except Exception as e:
            logger.error(f"Decision explanation generation failed: {e}")
            return None

    def _classify_decision(self, prediction: Any) -> str:
        """決定タイプの分類"""
        if prediction is None:
            return "UNKNOWN"

        if isinstance(prediction, (int, float)):
            if prediction > 0.5:
                return "BUY"
            elif prediction < -0.5:
                return "SELL"
            else:
                return "HOLD"
        elif isinstance(prediction, str):
            return prediction.upper()
        else:
            return "UNKNOWN"

    def _generate_natural_language_explanation(self,
                                            decision_type: str,
                                            primary_factors: List[FeatureImportance],
                                            contributing_factors: List[FeatureImportance]) -> str:
        """自然言語説明の生成"""
        try:
            explanation_parts = []

            # 決定の基本説明
            if decision_type == "BUY":
                explanation_parts.append("買いシグナルが検知されました。")
            elif decision_type == "SELL":
                explanation_parts.append("売りシグナルが検知されました。")
            elif decision_type == "HOLD":
                explanation_parts.append("ポジション保持が推奨されます。")
            else:
                explanation_parts.append("取引シグナルが不明です。")

            # 主な要因の説明
            if primary_factors:
                factor_names = [f.feature_name for f in primary_factors[:2]]
                explanation_parts.append(f"主な要因は{', '.join(factor_names)}です。")

            # 追加の文脈
            if contributing_factors:
                explanation_parts.append("市場の状況を総合的に判断した結果です。")

            return " ".join(explanation_parts)

        except Exception as e:
            logger.error(f"Natural language generation failed: {e}")
            return "説明の生成に失敗しました。"

    def _calculate_decision_confidence(self, feature_importance: List[FeatureImportance]) -> float:
        """決定の信頼度計算"""
        if not feature_importance:
            return 0.0

        # 上位特徴量の重要度の合計を信頼度として使用
        top_importance = sum(fi.importance_score for fi in feature_importance[:3])
        total_importance = sum(fi.importance_score for fi in feature_importance)

        if total_importance == 0:
            return 0.0

        return min(top_importance / total_importance, 1.0)

    def _calculate_confidence(self, score: float, all_scores: np.ndarray) -> float:
        """個別特徴量の信頼度計算"""
        if len(all_scores) <= 1:
            return 1.0

        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)

        if std_score == 0:
            return 1.0

        # Z-scoreに基づく信頼度
        z_score = (score - mean_score) / std_score
        confidence = 1.0 / (1.0 + np.exp(-z_score))  # Sigmoid

        return float(confidence)

    def _get_feature_name(self, index: int) -> str:
        """特徴量名の取得"""
        # インデックスから特徴量名をマッピング
        feature_keys = list(self.feature_names.keys())
        if index < len(feature_keys):
            return feature_keys[index]
        else:
            return f"feature_{index}"

    def _get_feature_description(self, feature_name: str) -> Optional[str]:
        """特徴量の説明を取得"""
        return self.feature_names.get(feature_name)

    def _cache_result(self, result: ExplanationResult) -> None:
        """結果をキャッシュ"""
        cache_entry = ExplanationCache(
            explanation_id=result.explanation_id,
            result=result,
            created_at=datetime.now(),
            ttl_seconds=self.config.cache_ttl_seconds
        )

        self.cache[result.explanation_id] = cache_entry

        # キャッシュサイズの制限
        if len(self.cache) > self.config.max_cached_explanations:
            # 最も古いエントリを削除
            oldest_key = min(self.cache.keys(),
                           key=lambda k: self.cache[k].created_at)
            del self.cache[oldest_key]

    def get_cached_explanation(self, explanation_id: str) -> Optional[ExplanationResult]:
        """キャッシュされた説明を取得"""
        if explanation_id not in self.cache:
            return None

        cache_entry = self.cache[explanation_id]
        if cache_entry.is_expired:
            del self.cache[explanation_id]
            return None

        return cache_entry.result

    def clear_expired_cache(self) -> int:
        """期限切れのキャッシュをクリア"""
        expired_keys = [
            key for key, cache_entry in self.cache.items()
            if cache_entry.is_expired
        ]

        for key in expired_keys:
            del self.cache[key]

        return len(expired_keys)

    def get_feature_importance_summary(self,
                                     explanations: List[ExplanationResult]) -> Dict[str, Any]:
        """特徴量重要度のサマリーを取得"""
        if not explanations:
            return {}

        all_features = {}
        total_explanations = len(explanations)

        for explanation in explanations:
            for fi in explanation.feature_importance:
                if fi.feature_name not in all_features:
                    all_features[fi.feature_name] = {
                        "total_importance": 0.0,
                        "count": 0,
                        "category": fi.feature_category,
                        "description": fi.description
                    }

                all_features[fi.feature_name]["total_importance"] += fi.importance_score
                all_features[fi.feature_name]["count"] += 1

        # 平均重要度を計算
        summary = {}
        for feature_name, data in all_features.items():
            summary[feature_name] = {
                "average_importance": data["total_importance"] / data["count"],
                "frequency": data["count"] / total_explanations,
                "category": data["category"],
                "description": data["description"]
            }

        return summary