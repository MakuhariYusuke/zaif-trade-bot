"""
Configuration for Explainability Module
説明可能性モジュールの設定
"""

from dataclasses import dataclass, field
from enum import Enum

class ExplanationMethod(Enum):
    """説明手法"""

    SHAP = "shap"
    LIME = "lime"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    DEEP_LIFT = "deep_lift"

class ExplanationScope(Enum):
    """説明範囲"""

    GLOBAL = "global"  # 全体的な特徴量重要度
    LOCAL = "local"  # 個別予測の説明
    COHORT = "cohort"  # グループベースの説明

@dataclass
class ExplainabilityConfig:
    """説明可能性設定"""

    # 基本設定
    enabled: bool = True
    explanation_method: ExplanationMethod = ExplanationMethod.SHAP
    explanation_scope: ExplanationScope = ExplanationScope.LOCAL

    # SHAP設定
    shap_max_evals: int = 1000
    shap_batch_size: int = 50
    shap_background_samples: int = 100

    # 説明生成設定
    generate_natural_language: bool = True
    max_features_to_explain: int = 10
    explanation_confidence_threshold: float = 0.7

    # 可視化設定
    enable_visualization: bool = True
    plot_save_path: str = "reports/explanations"
    plot_format: str = "png"

    # パフォーマンス設定
    cache_explanations: bool = True
    cache_ttl_seconds: int = 3600  # 1時間
    max_cached_explanations: int = 1000

    # 特徴量マッピング
    feature_names: dict[str, str] = field(default_factory=dict)
    feature_categories: dict[str, str] = field(default_factory=dict)

    # レポート設定
    generate_reports: bool = True
    report_frequency: str = "daily"  # daily, weekly, monthly
    report_path: str = "reports/explainability"

    def __post_init__(self):
        """設定の検証"""
        if self.shap_max_evals <= 0:
            raise ValueError("shap_max_evals must be positive")

        if (
            self.explanation_confidence_threshold < 0
            or self.explanation_confidence_threshold > 1
        ):
            raise ValueError("explanation_confidence_threshold must be between 0 and 1")

        # デフォルトの特徴量名マッピングを設定
        if not self.feature_names:
            self._set_default_feature_names()

    def _set_default_feature_names(self):
        """デフォルトの特徴量名を設定"""
        self.feature_names = {
            "price_change": "価格変化率",
            "volume_ratio": "出来高比率",
            "rsi_14": "RSI(14)",
            "macd_signal": "MACDシグナル",
            "bb_position": "ボリンジャーバンド位置",
            "adx": "ADX",
            "ichimoku_tenkan": "一目均衡表転換線",
            "ichimoku_kijun": "一目均衡表基準線",
            "supertrend": "スーパートレンド",
            "williams_r": "ウィリアムズ%R",
        }

        self.feature_categories = {
            "price_change": "価格指標",
            "volume_ratio": "出来高指標",
            "rsi_14": "オシレーター",
            "macd_signal": "トレンド指標",
            "bb_position": "ボラティリティ指標",
            "adx": "トレンド強度",
            "ichimoku_tenkan": "一目均衡表",
            "ichimoku_kijun": "一目均衡表",
            "supertrend": "トレンド指標",
            "williams_r": "オシレーター",
        }
