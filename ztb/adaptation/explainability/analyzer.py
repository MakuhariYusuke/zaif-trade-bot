"""
Explainability Analyzer Implementation
SHAPベースのモデル解釈性分析
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

try:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None

from .config import ExplainabilityConfig, ExplanationMethod
from .types import (
    DecisionExplanation,
    ExplanationCache,
    ExplanationResult,
    ExplanationType,
    FeatureImportance,
    VisualizationResult,
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

    def explain_prediction(
        self,
        model: nn.Module,
        input_data: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        prediction: Any = None,
        background_data: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> ExplanationResult:
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
                model_version=getattr(model, "_version", "unknown"),
                explanation_type=ExplanationType.FEATURE_IMPORTANCE,
                target_prediction=prediction,
                feature_importance=feature_importance,
                decision_explanation=decision_explanation,
                processing_time_seconds=time.time() - start_time,
            )

            # 可視化の生成（有効な場合）
            if self.config.enable_visualization:
                result.visualization = self._generate_visualizations(
                    feature_importance, decision_explanation, input_data
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
                model_version=getattr(model, "_version", "unknown"),
                explanation_type=ExplanationType.FEATURE_IMPORTANCE,
                target_prediction=prediction,
                feature_importance=[],
                processing_time_seconds=time.time() - start_time,
                metadata={"error": str(e)},
            )

    def _preprocess_input(self, input_data: Union[np.ndarray, "torch.Tensor", pd.DataFrame]):
        """Preprocess input data into a tensor (if torch available) or numpy array"""
        if torch is None:
            if isinstance(input_data, pd.DataFrame):
                return input_data.select_dtypes(include=[np.number]).values
            elif isinstance(input_data, np.ndarray):
                return input_data
            else:
                raise ValueError(f"Unsupported input data type: {type(input_data)}")
        """入力データの前処理"""
        if isinstance(input_data, pd.DataFrame):
            # Use numeric columns only
            numeric_data = input_data.select_dtypes(include=[np.number]).values
            if torch is not None:
                return torch.tensor(numeric_data, dtype=torch.float32)
            return numeric_data
        elif isinstance(input_data, np.ndarray):
            if torch is not None:
                return torch.tensor(input_data, dtype=torch.float32)
            return input_data
        elif torch is not None and isinstance(input_data, torch.Tensor):
            return input_data.float()
        else:
            raise ValueError(f"Unsupported input data type: {type(input_data)}")

    def _calculate_feature_importance(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        background_data: Optional[torch.Tensor] = None,
    ) -> List[FeatureImportance]:
        """特徴量重要度の計算"""
        if self.config.explanation_method == ExplanationMethod.SHAP and SHAP_AVAILABLE:
            return self._calculate_shap_importance(model, input_tensor, background_data)
        else:
            return self._calculate_basic_importance(model, input_tensor)

    def _calculate_shap_importance(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        background_data: Optional[torch.Tensor] = None,
    ) -> List[FeatureImportance]:
        """SHAPベースの特徴量重要度計算"""
        try:
            # モデルを評価モードに
            model.eval()

            # 背景データの準備
            if background_data is None:
                # デフォルトの背景データを生成
                background_data = torch.randn(
                    min(self.config.shap_background_samples, input_tensor.shape[0]),
                    input_tensor.shape[1],
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
            shap_values = explainer(
                input_tensor.numpy(), max_evals=self.config.shap_max_evals
            )

            # 特徴量重要度の抽出
            if hasattr(shap_values, "values"):
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
                feature_importance.append(
                    FeatureImportance(
                        feature_name=feature_name,
                        importance_score=float(score),
                        feature_category=self.feature_categories.get(feature_name),
                        description=self._get_feature_description(feature_name),
                        confidence=self._calculate_confidence(score, importance_scores),
                    )
                )

            # 重要度でソート
            feature_importance.sort(key=lambda x: x.importance_score, reverse=True)

            return feature_importance[: self.config.max_features_to_explain]

        except Exception as e:
            logger.error(f"SHAP calculation failed: {e}")
            return self._calculate_basic_importance(model, input_tensor)

    def _calculate_basic_importance(
        self, model: nn.Module, input_tensor: torch.Tensor
    ) -> List[FeatureImportance]:
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
                feature_importance.append(
                    FeatureImportance(
                        feature_name=feature_name,
                        importance_score=float(grad),
                        feature_category=self.feature_categories.get(feature_name),
                        description=self._get_feature_description(feature_name),
                    )
                )

            # 重要度でソート
            feature_importance.sort(key=lambda x: x.importance_score, reverse=True)

            return feature_importance[: self.config.max_features_to_explain]

        except Exception as e:
            logger.error(f"Basic importance calculation failed: {e}")
            return []

    def _generate_decision_explanation(
        self, feature_importance: List[FeatureImportance], prediction: Any = None
    ) -> Optional[DecisionExplanation]:
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
                confidence_score=self._calculate_decision_confidence(
                    feature_importance
                ),
                primary_factors=primary_factors,
                contributing_factors=contributing_factors,
                natural_language_explanation=natural_language,
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

    def _generate_natural_language_explanation(
        self,
        decision_type: str,
        primary_factors: List[FeatureImportance],
        contributing_factors: List[FeatureImportance],
    ) -> str:
        """自然言語説明の生成（高度化版）"""
        try:
            explanation_parts = []

            # 決定の詳細な説明
            decision_templates = {
                "BUY": [
                    "強い買いシグナルが検知されました。市場は上昇トレンドを示唆しています。",
                    "購入機会が確認されました。現在の市場状況が有利です。",
                    "買いポジションの確立を推奨します。複数の要因が上昇を示しています。",
                ],
                "SELL": [
                    "売りシグナルが検知されました。下落リスクが高まっています。",
                    "ポジションの整理を検討してください。市場の弱さが確認されます。",
                    "売り圧力が強まっています。利益確定または損切りを検討してください。",
                ],
                "HOLD": [
                    "現在のポジションを維持してください。明確なシグナルがありません。",
                    "市場の方向性が不明確です。追加の確認を待ってください。",
                    "観察継続を推奨します。市場の変動が小さい状況です。",
                ],
            }

            templates = decision_templates.get(
                decision_type, ["取引シグナルが不明確です。"]
            )
            explanation_parts.append(np.random.choice(templates))

            # 主な要因の詳細説明
            if primary_factors:
                primary_names = [f.feature_name for f in primary_factors[:3]]

                # 要因のカテゴリ分析
                categories = {}
                for factor in primary_factors[:3]:
                    cat = factor.feature_category or "テクニカル"
                    if cat not in categories:
                        categories[cat] = []
                    categories[cat].append(factor)

                # カテゴリ別の説明生成
                category_explanations = {
                    "trend": "トレンド指標が{}を示しており、市場の方向性を強く示唆しています。",
                    "oscillator": "オシレーター指標が{}を示しており、市場の過熱/冷え込み状態を示しています。",
                    "volatility": "ボラティリティ指標が{}を示しており、市場の変動性が変化しています。",
                    "volume": "出来高指標が{}を示しており、市場参加者の積極性を表しています。",
                    "momentum": "モメンタム指標が{}を示しており、市場の勢いが変化しています。",
                    "テクニカル": "テクニカル指標が{}を示しており、市場の状態変化を表しています。",
                }

                factor_details = []
                for category, factors in categories.items():
                    template = category_explanations.get(
                        category, category_explanations["テクニカル"]
                    )
                    factor_names = [f.feature_name for f in factors]
                    factor_details.append(template.format("、".join(factor_names)))

                explanation_parts.append(" ".join(factor_details))

                # 重要度の分析
                total_importance = sum(f.importance_score for f in primary_factors)
                if total_importance > 0.5:
                    explanation_parts.append(
                        "これらの要因の重要度が高く、信頼性の高いシグナルです。"
                    )
                elif total_importance > 0.3:
                    explanation_parts.append(
                        "これらの要因の重要度は中程度です。追加の確認を推奨します。"
                    )
                else:
                    explanation_parts.append(
                        "これらの要因の重要度は低めです。慎重な判断が必要です。"
                    )

            # 寄与要因の言及
            if contributing_factors:
                contrib_names = [f.feature_name for f in contributing_factors[:3]]
                explanation_parts.append(
                    f"また、{'、'.join(contrib_names)}などの要因も判断に寄与しています。"
                )

            # 市場状況の総合評価
            market_context = self._analyze_market_context(primary_factors)
            if market_context:
                explanation_parts.append(market_context)

            # リスク警告
            risk_warning = self._generate_risk_warning(decision_type, primary_factors)
            if risk_warning:
                explanation_parts.append(risk_warning)

            return " ".join(explanation_parts)

        except Exception as e:
            logger.error(f"Enhanced natural language generation failed: {e}")
            return "説明の生成に失敗しました。"

    def _analyze_market_context(
        self, primary_factors: List[FeatureImportance]
    ) -> Optional[str]:
        """市場状況の分析"""
        try:
            if not primary_factors:
                return None

            # カテゴリ別の要因数をカウント
            category_count = {}
            for factor in primary_factors:
                cat = factor.feature_category or "unknown"
                category_count[cat] = category_count.get(cat, 0) + 1

            # 市場状況の判定
            max_category = max(category_count, key=category_count.get)

            context_templates = {
                "trend": "市場は明確なトレンドを示しており、現在の方向性に沿った取引が適切です。",
                "oscillator": "市場はレンジ相場または転換点にある可能性があります。オシレーターの動きに注意が必要です。",
                "volatility": "市場の変動性が変化しており、リスク管理を強化してください。",
                "volume": "市場参加者の積極性が高まっています。取引量の変化に注意が必要です。",
                "momentum": "市場の勢いが変化しています。モメンタムの方向性を確認してください。",
                "unknown": "市場状況を総合的に分析した結果です。",
            }

            return context_templates.get(max_category, context_templates["unknown"])

        except Exception as e:
            logger.error(f"Market context analysis failed: {e}")
            return None

    def _generate_risk_warning(
        self, decision_type: str, primary_factors: List[FeatureImportance]
    ) -> Optional[str]:
        """リスク警告の生成"""
        try:
            if not primary_factors:
                return None

            # 平均重要度の計算
            avg_importance = np.mean([f.importance_score for f in primary_factors])

            # 決定タイプ別のリスク警告
            warnings = {
                "BUY": {
                    "high": "上昇余地はあるものの、利益確定のタイミングに注意してください。",
                    "medium": "買いシグナルは確認されましたが、市場の変動リスクを考慮してください。",
                    "low": "買いシグナルが弱いため、少量からのポジション構築を推奨します。",
                },
                "SELL": {
                    "high": "下落リスクが高いため、損切りラインの設定を忘れずに。",
                    "medium": "売りシグナルは確認されましたが、反発リスクに注意してください。",
                    "low": "売りシグナルが弱いため、様子見を推奨します。",
                },
                "HOLD": {
                    "high": "ポジション維持が適切ですが、市場の変化に警戒してください。",
                    "medium": "現在のポジションを維持してください。",
                    "low": "明確なシグナルがないため、慎重な姿勢を保ってください。",
                },
            }

            # 重要度に基づくリスクレベル判定
            if avg_importance > 0.7:
                risk_level = "high"
            elif avg_importance > 0.4:
                risk_level = "medium"
            else:
                risk_level = "low"

            decision_warnings = warnings.get(decision_type, warnings["HOLD"])
            return decision_warnings.get(risk_level)

        except Exception as e:
            logger.error(f"Risk warning generation failed: {e}")
            return None

    def _calculate_decision_confidence(
        self, feature_importance: List[FeatureImportance]
    ) -> float:
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

    def _generate_visualizations(
        self,
        feature_importance: List[FeatureImportance],
        decision_explanation: Optional[DecisionExplanation],
        input_data: Any,
    ) -> Optional[VisualizationResult]:
        """可視化の生成"""
        if not self.config.enable_visualization:
            return None

        try:
            visualizations = {}

            # 特徴量重要度の棒グラフ
            if feature_importance:
                visualizations[
                    "feature_importance_plot"
                ] = self._create_feature_importance_plot(feature_importance)

            # 決定プロセスのフローチャート
            if decision_explanation:
                visualizations["decision_flowchart"] = self._create_decision_flowchart(
                    decision_explanation
                )

            # 特徴量分布の可視化
            if hasattr(input_data, "shape"):
                visualizations[
                    "feature_distribution"
                ] = self._create_feature_distribution_plot(
                    input_data, feature_importance
                )

            return VisualizationResult(
                plots=visualizations,
                timestamp=datetime.now(),
                format=self.config.plot_format,
            )

            return VisualizationResult(
                plots=visualizations,
                timestamp=datetime.now(),
                format=self.config.plot_format,
            )

        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            return None

    def _create_feature_importance_plot(
        self, feature_importance: List[FeatureImportance]
    ) -> Dict[str, Any]:
        """特徴量重要度の棒グラフ作成"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                return {
                    "type": "text",
                    "content": "Matplotlib not available for plotting",
                }

            # データの準備
            top_features = feature_importance[:10]  # 上位10個
            feature_names = [fi.feature_name for fi in top_features]
            importance_scores = [fi.importance_score for fi in top_features]
            categories = [fi.feature_category or "Unknown" for fi in top_features]

            # カテゴリ別の色設定
            category_colors = {
                "trend": "#1f77b4",  # 青
                "oscillator": "#ff7f0e",  # オレンジ
                "volume": "#2ca02c",  # 緑
                "volatility": "#d62728",  # 赤
                "momentum": "#9467bd",  # 紫
                "Unknown": "#7f7f7f",  # 灰
            }

            colors = [
                category_colors.get(cat, category_colors["Unknown"])
                for cat in categories
            ]

            # プロットの作成
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.barh(range(len(feature_names)), importance_scores, color=colors)

            # ラベルの設定
            ax.set_yticks(range(len(feature_names)))
            ax.set_yticklabels(feature_names, fontsize=10)
            ax.set_xlabel("Importance Score", fontsize=12)
            ax.set_title("Feature Importance Analysis", fontsize=14, fontweight="bold")

            # 値の表示
            for i, (bar, score) in enumerate(zip(bars, importance_scores)):
                ax.text(
                    score + max(importance_scores) * 0.01,
                    i,
                    ".3f",
                    ha="left",
                    va="center",
                    fontsize=9,
                )

            # 凡例の作成
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor=color, label=cat)
                for cat, color in category_colors.items()
                if cat in categories
            ]
            ax.legend(handles=legend_elements, loc="lower right")

            plt.tight_layout()

            # 画像として保存
            import io

            buf = io.BytesIO()
            fig.savefig(
                buf, format=self.config.plot_format, dpi=150, bbox_inches="tight"
            )
            buf.seek(0)
            image_data = buf.getvalue()
            buf.close()
            plt.close(fig)

            return {
                "type": "image",
                "format": self.config.plot_format,
                "data": image_data,
                "description": "特徴量重要度の棒グラフ",
            }

        except Exception as e:
            logger.error(f"Feature importance plot creation failed: {e}")
            return {"type": "error", "message": str(e)}

    def _create_decision_flowchart(
        self, decision_explanation: DecisionExplanation
    ) -> Dict[str, Any]:
        """決定プロセスのフローチャート作成"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                return {
                    "type": "text",
                    "content": "Matplotlib not available for plotting",
                }

            fig, ax = plt.subplots(figsize=(14, 10))
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis("off")

            # ノードの位置とサイズ
            nodes = [
                {
                    "text": "市場データ\n入力",
                    "pos": (2, 8),
                    "size": (2, 0.8),
                    "color": "#e6f3ff",
                },
                {
                    "text": "特徴量\n抽出",
                    "pos": (2, 6),
                    "size": (2, 0.8),
                    "color": "#fff2e6",
                },
                {
                    "text": "モデル\n予測",
                    "pos": (5, 6),
                    "size": (2, 0.8),
                    "color": "#f0f9ff",
                },
                {
                    "text": f"決定:\n{decision_explanation.decision_type}",
                    "pos": (5, 4),
                    "size": (2, 0.8),
                    "color": "#d4edda"
                    if decision_explanation.decision_type == "BUY"
                    else "#f8d7da"
                    if decision_explanation.decision_type == "SELL"
                    else "#fff3cd",
                },
            ]

            # 主な要因のノード
            primary_factors = decision_explanation.primary_factors[:3]
            for i, factor in enumerate(primary_factors):
                nodes.append(
                    {
                        "text": f"主要要因:\n{factor.feature_name}\n重要度: {factor.importance_score:.3f}",
                        "pos": (7, 6 - i * 1.2),
                        "size": (2.5, 0.8),
                        "color": "#f8f9fa",
                    }
                )

            # ノードの描画
            for node in nodes:
                x, y = node["pos"]
                w, h = node["size"]
                rect = FancyBboxPatch(
                    (x - w / 2, y - h / 2),
                    w,
                    h,
                    boxstyle="round,pad=0.1",
                    facecolor=node["color"],
                    edgecolor="#333333",
                    linewidth=1,
                )
                ax.add_patch(rect)

                # テキストの追加
                ax.text(
                    x,
                    y,
                    node["text"],
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    wrap=True,
                )

            # 矢印の描画
            arrows = [
                ((2, 7.6), (2, 6.4)),  # 入力 → 特徴量抽出
                ((2.8, 6), (4.2, 6)),  # 特徴量抽出 → モデル予測
                ((5, 5.6), (5, 4.4)),  # モデル予測 → 決定
                ((5.8, 4), (6.3, 4)),  # 決定 → 主要要因
            ]

            for start, end in arrows:
                ax.annotate(
                    "",
                    xy=end,
                    xytext=start,
                    arrowprops=dict(arrowstyle="->", color="#666666", linewidth=2),
                )

            # 信頼度の表示
            confidence_text = f"決定信頼度: {decision_explanation.confidence_score:.1%}"
            ax.text(
                5,
                2,
                confidence_text,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#e9ecef"),
            )

            # 画像として保存
            import io

            buf = io.BytesIO()
            fig.savefig(
                buf, format=self.config.plot_format, dpi=150, bbox_inches="tight"
            )
            buf.seek(0)
            image_data = buf.getvalue()
            buf.close()
            plt.close(fig)

            return {
                "type": "image",
                "format": self.config.plot_format,
                "data": image_data,
                "description": "決定プロセスのフローチャート",
            }

        except Exception as e:
            logger.error(f"Decision flowchart creation failed: {e}")
            return {"type": "error", "message": str(e)}

    def _create_feature_distribution_plot(
        self, input_data: Any, feature_importance: List[FeatureImportance]
    ) -> Dict[str, Any]:
        """特徴量分布の可視化"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                return {
                    "type": "text",
                    "content": "Matplotlib not available for plotting",
                }

            # データをDataFrameに変換
            if isinstance(input_data, torch.Tensor):
                data = pd.DataFrame(input_data.numpy())
            elif isinstance(input_data, np.ndarray):
                data = pd.DataFrame(input_data)
            elif isinstance(input_data, pd.DataFrame):
                data = input_data.copy()
            else:
                return {
                    "type": "text",
                    "content": "Unsupported data format for distribution plot",
                }

            # 上位の重要な特徴量を選択
            top_features = feature_importance[:6]  # 上位6個
            feature_indices = []
            feature_names = []

            for fi in top_features:
                # 特徴量名からインデックスを検索
                if fi.feature_name.startswith("feature_"):
                    try:
                        idx = int(fi.feature_name.split("_")[1])
                        if idx < data.shape[1]:
                            feature_indices.append(idx)
                            feature_names.append(fi.feature_name)
                    except (ValueError, IndexError):
                        continue
                else:
                    # 名前ベースの検索（実装依存）
                    continue

            if len(feature_indices) < 2:
                return {
                    "type": "text",
                    "content": "Insufficient feature data for distribution plot",
                }

            # サブプロットの作成
            n_features = len(feature_indices)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()

            for i, (idx, name) in enumerate(zip(feature_indices, feature_names)):
                if i >= len(axes):
                    break

                ax = axes[i]
                values = data.iloc[:, idx].values

                # 分布のプロット
                ax.hist(
                    values,
                    bins=30,
                    alpha=0.7,
                    color="#1f77b4",
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.set_title(f"{name} Distribution", fontsize=10, fontweight="bold")
                ax.set_xlabel("Value", fontsize=8)
                ax.set_ylabel("Frequency", fontsize=8)
                ax.grid(True, alpha=0.3)

                # 統計情報の追加
                mean_val = np.mean(values)
                std_val = np.std(values)
                ax.axvline(
                    mean_val, color="red", linestyle="--", alpha=0.8, label=".2f"
                )
                ax.legend(fontsize=8)

            # 余分なサブプロットを非表示
            for i in range(len(feature_indices), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()

            # 画像として保存
            import io

            buf = io.BytesIO()
            fig.savefig(
                buf, format=self.config.plot_format, dpi=150, bbox_inches="tight"
            )
            buf.seek(0)
            image_data = buf.getvalue()
            buf.close()
            plt.close(fig)

            return {
                "type": "image",
                "format": self.config.plot_format,
                "data": image_data,
                "description": "特徴量分布のヒストグラム",
            }

        except Exception as e:
            logger.error(f"Feature distribution plot creation failed: {e}")
            return {"type": "error", "message": str(e)}

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
            ttl_seconds=self.config.cache_ttl_seconds,
        )

        self.cache[result.explanation_id] = cache_entry

        # キャッシュサイズの制限
        if len(self.cache) > self.config.max_cached_explanations:
            # 最も古いエントリを削除
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
            del self.cache[oldest_key]

    def get_cached_explanation(
        self, explanation_id: str
    ) -> Optional[ExplanationResult]:
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
            key for key, cache_entry in self.cache.items() if cache_entry.is_expired
        ]

        for key in expired_keys:
            del self.cache[key]

        return len(expired_keys)

    def get_feature_importance_summary(
        self, explanations: List[ExplanationResult]
    ) -> Dict[str, Any]:
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
                        "description": fi.description,
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
                "description": data["description"],
            }

        return summary

    def generate_explanation_report(
        self, explanations: List[ExplanationResult], output_path: Optional[str] = None
    ) -> str:
        """説明レポートの生成"""
        try:
            if not explanations:
                return "No explanations to report"

            # レポートデータの集計
            report_data = self._aggregate_explanation_data(explanations)

            # HTMLレポートの生成
            html_content = self._generate_html_report(report_data, explanations)

            # ファイル保存
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                return f"Report saved to: {output_path}"
            else:
                # デフォルトのパスに保存
                default_path = os.path.join(
                    self.config.report_path,
                    f"explanation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                )
                os.makedirs(os.path.dirname(default_path), exist_ok=True)
                with open(default_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                return f"Report saved to: {default_path}"

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return f"Report generation failed: {e}"

    def _aggregate_explanation_data(
        self, explanations: List[ExplanationResult]
    ) -> Dict[str, Any]:
        """説明データの集計"""
        try:
            # 基本統計
            total_explanations = len(explanations)
            avg_processing_time = np.mean(
                [e.processing_time_seconds for e in explanations]
            )

            # 決定タイプ別の集計
            decision_counts = {}
            confidence_scores = []

            for exp in explanations:
                if exp.decision_explanation:
                    decision_type = exp.decision_explanation.decision_type
                    decision_counts[decision_type] = (
                        decision_counts.get(decision_type, 0) + 1
                    )
                    confidence_scores.append(exp.decision_explanation.confidence_score)

            # 特徴量重要度の集計
            feature_summary = self.get_feature_importance_summary(explanations)

            # 時系列分析
            timestamps = [e.timestamp for e in explanations]
            if timestamps:
                time_range = f"{min(timestamps).strftime('%Y-%m-%d %H:%M')} to {max(timestamps).strftime('%Y-%m-%d %H:%M')}"

            return {
                "total_explanations": total_explanations,
                "avg_processing_time": avg_processing_time,
                "decision_counts": decision_counts,
                "avg_confidence": np.mean(confidence_scores)
                if confidence_scores
                else 0,
                "feature_summary": feature_summary,
                "time_range": time_range if timestamps else "N/A",
                "generated_at": datetime.now(),
            }

        except Exception as e:
            logger.error(f"Data aggregation failed: {e}")
            return {}

    def _generate_html_report(
        self, report_data: Dict[str, Any], explanations: List[ExplanationResult]
    ) -> str:
        """HTMLレポートの生成"""
        try:
            html_template = f"""
            <!DOCTYPE html>
            <html lang="ja">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>SAC v421 説明可能性レポート</title>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: #f5f5f5;
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                        background: white;
                        padding: 30px;
                        border-radius: 10px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }}
                    .header {{
                        text-align: center;
                        border-bottom: 2px solid #007acc;
                        padding-bottom: 20px;
                        margin-bottom: 30px;
                    }}
                    .stats-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                        gap: 20px;
                        margin-bottom: 30px;
                    }}
                    .stat-card {{
                        background: #f8f9fa;
                        padding: 20px;
                        border-radius: 8px;
                        border-left: 4px solid #007acc;
                    }}
                    .stat-value {{
                        font-size: 2em;
                        font-weight: bold;
                        color: #007acc;
                    }}
                    .feature-table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 20px;
                    }}
                    .feature-table th, .feature-table td {{
                        padding: 12px;
                        text-align: left;
                        border-bottom: 1px solid #ddd;
                    }}
                    .feature-table th {{
                        background-color: #f8f9fa;
                        font-weight: bold;
                    }}
                    .decision-chart {{
                        margin: 20px 0;
                        padding: 20px;
                        background: #f8f9fa;
                        border-radius: 8px;
                    }}
                    .explanation-sample {{
                        background: #f0f8ff;
                        padding: 15px;
                        border-radius: 8px;
                        margin: 10px 0;
                        border-left: 4px solid #007acc;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>SAC v421 説明可能性レポート</h1>
                        <p>生成日時: {report_data.get('generated_at', datetime.now()).strftime('%Y年%m月%d日 %H:%M:%S')}</p>
                        <p>分析期間: {report_data.get('time_range', 'N/A')}</p>
                    </div>

                    <div class="stats-grid">
                        <div class="stat-card">
                            <h3>総説明数</h3>
                            <div class="stat-value">{report_data.get('total_explanations', 0)}</div>
                        </div>
                        <div class="stat-card">
                            <h3>平均処理時間</h3>
                            <div class="stat-value">{report_data.get('avg_processing_time', 0):.3f}s</div>
                        </div>
                        <div class="stat-card">
                            <h3>平均信頼度</h3>
                            <div class="stat-value">{report_data.get('avg_confidence', 0):.1%}</div>
                        </div>
                    </div>

                    <h2>決定タイプ分布</h2>
                    <div class="decision-chart">
                        {self._generate_decision_chart_html(report_data.get('decision_counts', {}))}
                    </div>

                    <h2>トップ特徴量</h2>
                    <table class="feature-table">
                        <thead>
                            <tr>
                                <th>特徴量名</th>
                                <th>平均重要度</th>
                                <th>出現頻度</th>
                                <th>カテゴリ</th>
                            </tr>
                        </thead>
                        <tbody>
                            {self._generate_feature_table_html(report_data.get('feature_summary', {}))}
                        </tbody>
                    </table>

                    <h2>説明サンプル</h2>
                    {self._generate_explanation_samples_html(explanations[:5])}

                </div>
            </body>
            </html>
            """

            return html_template

        except Exception as e:
            logger.error(f"HTML report generation failed: {e}")
            return (
                f"<html><body><h1>Report Generation Failed</h1><p>{e}</p></body></html>"
            )

    def _generate_decision_chart_html(self, decision_counts: Dict[str, int]) -> str:
        """決定チャートのHTML生成"""
        if not decision_counts:
            return "<p>データなし</p>"

        total = sum(decision_counts.values())
        chart_html = "<div style='display: flex; gap: 20px; flex-wrap: wrap;'>"

        colors = {"BUY": "#28a745", "SELL": "#dc3545", "HOLD": "#ffc107"}

        for decision, count in decision_counts.items():
            percentage = (count / total) * 100
            color = colors.get(decision, "#6c757d")
            chart_html += f"""
                <div style='text-align: center;'>
                    <div style='width: 100px; height: 100px; border-radius: 50%; background: conic-gradient({color} 0% {percentage}%, #e9ecef {percentage}% 100%); display: inline-block; margin-bottom: 10px;'></div>
                    <div><strong>{decision}</strong></div>
                    <div>{count} ({percentage:.1f}%)</div>
                </div>
            """

        chart_html += "</div>"
        return chart_html

    def _generate_feature_table_html(self, feature_summary: Dict[str, Any]) -> str:
        """特徴量テーブルのHTML生成"""
        rows = ""
        for feature_name, data in list(feature_summary.items())[:10]:  # トップ10
            rows += f"""
                <tr>
                    <td>{feature_name}</td>
                    <td>{data.get('average_importance', 0):.4f}</td>
                    <td>{data.get('frequency', 0):.1%}</td>
                    <td>{data.get('category', 'N/A')}</td>
                </tr>
            """
        return rows

    def _generate_explanation_samples_html(
        self, explanations: List[ExplanationResult]
    ) -> str:
        """説明サンプルのHTML生成"""
        html = ""
        for exp in explanations:
            if (
                exp.decision_explanation
                and exp.decision_explanation.natural_language_explanation
            ):
                html += f"""
                    <div class="explanation-sample">
                        <h4>決定: {exp.decision_explanation.decision_type} (信頼度: {exp.decision_explanation.confidence_score:.1%})</h4>
                        <p>{exp.decision_explanation.natural_language_explanation}</p>
                        <small>処理時間: {exp.processing_time_seconds:.3f}秒</small>
                    </div>
                """
        return html if html else "<p>自然言語説明なし</p>"
