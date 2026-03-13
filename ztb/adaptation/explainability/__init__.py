"""
Explainability Module for SAC v421
モデル解釈性と説明可能性の強化

Features:
- SHAP-based feature importance analysis
- Natural language explanations
- Decision process visualization
- Model interpretability metrics
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analyzer import ExplainabilityAnalyzer
else:
    # Lazy import to avoid heavy dependencies (shap, IPython, jedi)
    ExplainabilityAnalyzer = None

from .config import ExplainabilityConfig
from .types import DecisionExplanation, ExplanationResult, FeatureImportance

__all__ = [
    "ExplainabilityAnalyzer",
    "ExplainabilityConfig",
    "ExplanationResult",
    "FeatureImportance",
    "DecisionExplanation",
]
