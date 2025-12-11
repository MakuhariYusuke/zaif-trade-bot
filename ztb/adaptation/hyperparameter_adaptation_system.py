"""
Dynamic Hyperparameter Adaptation Integration
動的ハイパーパラメータ適応統合
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import pandas as pd

from .dynamic_hyperparameter_adapter import (
    AdaptationStrategy,
    DynamicHyperparameterAdapter,
    HyperparameterConfig,
    HyperparameterType,
)
from .market_aware_hyperparameter_manager import (
    MarketAwareConfig,
    MarketAwareHyperparameterManager,
)
from .monitoring.evaluation_manager import ContinuousEvaluationManager

if TYPE_CHECKING:
    from .online_learning.pipeline import OnlineLearningPipeline

# from .monitoring.market_regime_detector import MarketRegimeDetector  # Optional


logger = logging.getLogger(__name__)


class HyperparameterAdaptationSystem:
    """ハイパーパラメータ適応システム統合"""

    def __init__(
        self,
        online_learning_pipeline: "OnlineLearningPipeline",
        evaluation_manager: ContinuousEvaluationManager,
        market_regime_detector: Optional[Any] = None,
    ):
        self.online_learning = online_learning_pipeline
        self.evaluation_manager = evaluation_manager
        self.market_detector = market_regime_detector

        # 設定
        self.hyperparameter_config = self._create_hyperparameter_config()
        self.market_aware_config = self._create_market_aware_config()

        # コンポーネント
        self.dynamic_adapter: Optional[DynamicHyperparameterAdapter] = None
        self.market_aware_manager: Optional[MarketAwareHyperparameterManager] = None

        # 状態
        self.is_initialized = False
        self.is_active = False

        logger.info("HyperparameterAdaptationSystem initialized")

    def _create_hyperparameter_config(self) -> HyperparameterConfig:
        """ハイパーパラメータ設定を作成"""
        config = HyperparameterConfig()

        # SAC v421向けに最適化された設定
        config.enabled_parameters = [
            HyperparameterType.LEARNING_RATE,
            HyperparameterType.BATCH_SIZE,
            HyperparameterType.REGULARIZATION_STRENGTH,
            HyperparameterType.DROPOUT_RATE,
        ]

        config.enabled_strategies = [
            AdaptationStrategy.PERFORMANCE_BASED,
            AdaptationStrategy.VOLATILITY_BASED,
            AdaptationStrategy.GRADIENT_BASED,
        ]

        # リアルタイム適応向け設定
        config.adaptation_interval_minutes = 15  # 15分ごとに適応
        config.min_adaptation_interval_minutes = 5  # 最小5分
        config.performance_window_size = 50  # パフォーマンス評価ウィンドウ
        config.performance_improvement_threshold = 0.005  # 0.5%改善で適応

        # 市場適応設定
        config.volatility_window_minutes = 30  # 30分ボラティリティ
        config.high_volatility_threshold = 0.025  # 2.5%高ボラティリティ
        config.low_volatility_threshold = 0.008  # 0.8%低ボラティリティ

        # 安全設定
        config.safety_margin = 0.15  # 15%安全マージン
        config.max_parameter_change_rate = 0.25  # 最大25%変更

        return config

    def _create_market_aware_config(self) -> MarketAwareConfig:
        """市場対応設定を作成"""
        config = MarketAwareConfig()

        # 市場適応有効化
        config.market_regime_adaptation = True
        config.volatility_based_scaling = True
        config.trend_adaptive_parameters = True

        # 学習設定
        config.online_learning_enabled = True
        config.meta_learning_enabled = True
        config.transfer_learning_enabled = False  # 現在は無効

        # 予測設定
        config.use_performance_prediction = True
        config.prediction_model_update_interval = 30  # 30分ごとに更新
        config.prediction_history_window = 500  # 500サンプル履歴

        # 適応戦略設定
        config.adaptive_strategy_selection = True
        config.strategy_performance_tracking = True
        config.strategy_switching_threshold = 0.03  # 3%改善で戦略変更

        # リスク管理
        config.risk_aware_adaptation = True
        config.max_risk_adjustment = 0.2  # 最大20%リスク調整

        return config

    def initialize(self) -> bool:
        """システムを初期化"""
        try:
            if self.is_initialized:
                logger.warning("System already initialized")
                return True

            # 動的アダプターを初期化
            self.dynamic_adapter = DynamicHyperparameterAdapter(
                self.online_learning,
                self.evaluation_manager,
                self.hyperparameter_config,
            )

            # 市場対応マネージャーを初期化
            self.market_aware_manager = MarketAwareHyperparameterManager(
                self.online_learning,
                self.evaluation_manager,
                self.market_detector,
                self.market_aware_config,
            )

            self.is_initialized = True
            logger.info("Hyperparameter adaptation system initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize hyperparameter adaptation system: {e}")
            return False

    def start(self) -> bool:
        """適応を開始"""
        try:
            if not self.is_initialized:
                if not self.initialize():
                    return False

            if self.is_active:
                logger.warning("Adaptation system already active")
                return True

            # 市場対応マネージャーを開始
            if not self.market_aware_manager.start_adaptation():
                logger.error("Failed to start market-aware manager")
                return False

            self.is_active = True
            logger.info("Hyperparameter adaptation system started")
            return True

        except Exception as e:
            logger.error(f"Failed to start adaptation system: {e}")
            return False

    def stop(self) -> None:
        """適応を停止"""
        try:
            if self.market_aware_manager:
                self.market_aware_manager.stop_adaptation()

            self.is_active = False
            logger.info("Hyperparameter adaptation system stopped")

        except Exception as e:
            logger.error(f"Error stopping adaptation system: {e}")

    def adapt_hyperparameters(
        self, market_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """ハイパーパラメータを適応"""
        try:
            if not self.is_initialized or not self.market_aware_manager:
                return {
                    "success": False,
                    "error": "System not initialized",
                    "adaptations": [],
                    "performance_improvement": 0.0,
                }

            # 市場対応適応を実行
            result = self.market_aware_manager.adapt_hyperparameters_market_aware(
                market_data
            )

            return {
                "success": True,
                "adaptations": [
                    {
                        "parameter": adaptation.parameter_type.value,
                        "old_value": adaptation.old_value,
                        "new_value": adaptation.new_value,
                        "strategy": adaptation.adaptation_strategy.value,
                        "reason": adaptation.reason,
                    }
                    for adaptation in result.adaptations
                ],
                "performance_improvement": result.overall_performance_improvement,
                "confidence": result.adaptation_confidence,
                "market_conditions": result.market_conditions,
                "recommendations": result.recommendations,
                "timestamp": result.timestamp.isoformat(),
            }

        except Exception as e:
            logger.error(f"Hyperparameter adaptation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "adaptations": [],
                "performance_improvement": 0.0,
            }

    def get_current_hyperparameters(self) -> Dict[str, float]:
        """現在のハイパーパラメータを取得"""
        try:
            if not self.dynamic_adapter:
                return {}

            return self.dynamic_adapter.get_current_hyperparameters()

        except Exception as e:
            logger.error(f"Failed to get current hyperparameters: {e}")
            return {}

    def get_adaptation_status(self) -> Dict[str, Any]:
        """適応状態を取得"""
        try:
            status = {
                "is_initialized": self.is_initialized,
                "is_active": self.is_active,
                "current_hyperparameters": {},
                "adaptation_history": [],
                "performance_predictions": {},
                "adaptation_statistics": {},
                "recommendations": [],
            }

            if self.dynamic_adapter:
                status[
                    "current_hyperparameters"
                ] = self.dynamic_adapter.get_current_hyperparameters()
                status[
                    "adaptation_history"
                ] = self.dynamic_adapter.get_adaptation_history(hours=24)

            if self.market_aware_manager:
                status[
                    "performance_predictions"
                ] = self.market_aware_manager.get_performance_predictions()
                status[
                    "adaptation_statistics"
                ] = self.market_aware_manager.get_adaptation_statistics()
                status[
                    "recommendations"
                ] = self.market_aware_manager.get_adaptation_recommendations()

            return status

        except Exception as e:
            logger.error(f"Failed to get adaptation status: {e}")
            return {"error": str(e)}

    def update_config(self, config_updates: Dict[str, Any]) -> bool:
        """設定を更新"""
        try:
            # ハイパーパラメータ設定の更新
            if "hyperparameter_config" in config_updates:
                hp_updates = config_updates["hyperparameter_config"]
                for key, value in hp_updates.items():
                    if hasattr(self.hyperparameter_config, key):
                        setattr(self.hyperparameter_config, key, value)

            # 市場対応設定の更新
            if "market_aware_config" in config_updates:
                ma_updates = config_updates["market_aware_config"]
                for key, value in ma_updates.items():
                    if hasattr(self.market_aware_config, key):
                        setattr(self.market_aware_config, key, value)

            # アダプターに設定を適用
            if self.dynamic_adapter:
                self.dynamic_adapter.config = self.hyperparameter_config

            if self.market_aware_manager:
                self.market_aware_manager.config = self.market_aware_config

            logger.info("Configuration updated successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False

    def get_performance_metrics(self) -> Dict[str, Any]:
        """パフォーマンスメトリクスを取得"""
        try:
            metrics = {
                "adaptation_effectiveness": 0.0,
                "parameter_stability": 0.0,
                "market_adaptiveness": 0.0,
                "risk_adjustment_score": 0.0,
                "timestamp": datetime.now().isoformat(),
            }

            if self.market_aware_manager:
                stats = self.market_aware_manager.get_adaptation_statistics()

                # 適応有効性を計算
                total_adaptations = stats.get("total_adaptations", 0)
                if total_adaptations > 0:
                    strategy_performances = stats.get("strategy_performance", {})
                    if strategy_performances:
                        avg_performances = [
                            perf_data.get("average_performance", 0.0)
                            for perf_data in strategy_performances.values()
                        ]
                        metrics["adaptation_effectiveness"] = float(
                            np.mean(avg_performances)
                        )

                # パラメータ安定性を計算
                param_counts = stats.get("parameter_adaptation_count", {})
                if param_counts:
                    total_changes = sum(param_counts.values())
                    unique_params = len(param_counts)
                    metrics["parameter_stability"] = float(
                        unique_params / max(total_changes, 1)
                    )

            return metrics

        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {"error": str(e)}

    def reset_adaptation_history(self) -> bool:
        """適応履歴をリセット"""
        try:
            if self.dynamic_adapter:
                self.dynamic_adapter.adaptation_history.clear()
                self.dynamic_adapter.performance_history.clear()
                for param_history in self.dynamic_adapter.parameter_history.values():
                    param_history.clear()

            if self.market_aware_manager:
                self.market_aware_manager.adaptation_history.clear()
                self.market_aware_manager.market_conditions.clear()
                for training_data in self.market_aware_manager.training_data.values():
                    training_data.clear()
                for (
                    performances
                ) in self.market_aware_manager.strategy_performance.values():
                    performances.clear()

            logger.info("Adaptation history reset successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to reset adaptation history: {e}")
            return False
