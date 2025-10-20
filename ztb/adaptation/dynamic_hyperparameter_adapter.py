"""
Dynamic Hyperparameter Adaptation System
動的ハイパーパラメータ適応システム
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

from .monitoring.evaluation_manager import ContinuousEvaluationManager
from .online_learning.pipeline import OnlineLearningPipeline

logger = logging.getLogger(__name__)


class HyperparameterType(Enum):
    """ハイパーパラメータタイプ"""

    LEARNING_RATE = "learning_rate"  # 学習率
    BATCH_SIZE = "batch_size"  # バッチサイズ
    REGULARIZATION_STRENGTH = "regularization_strength"  # 正則化強度
    DROPOUT_RATE = "dropout_rate"  # ドロップアウト率
    GRADIENT_CLIP_VALUE = "gradient_clip_value"  # 勾配クリッピング値
    WEIGHT_DECAY = "weight_decay"  # 重み減衰


class AdaptationStrategy(Enum):
    """適応戦略"""

    PERFORMANCE_BASED = "performance_based"  # パフォーマンスベース
    VOLATILITY_BASED = "volatility_based"  # ボラティリティベース
    GRADIENT_BASED = "gradient_based"  # 勾配ベース
    CURRICULUM_BASED = "curriculum_based"  # カリキュラムベース
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"  # ベイズ最適化


@dataclass
class HyperparameterConfig:
    """ハイパーパラメータ設定"""

    # 適応対象パラメータ
    enabled_parameters: List[HyperparameterType] = field(
        default_factory=lambda: [
            HyperparameterType.LEARNING_RATE,
            HyperparameterType.BATCH_SIZE,
            HyperparameterType.REGULARIZATION_STRENGTH,
        ]
    )

    # 適応戦略
    enabled_strategies: List[AdaptationStrategy] = field(
        default_factory=lambda: [
            AdaptationStrategy.PERFORMANCE_BASED,
            AdaptationStrategy.VOLATILITY_BASED,
        ]
    )

    # パラメータ範囲設定
    parameter_ranges: Dict[HyperparameterType, Tuple[float, float]] = field(
        default_factory=lambda: {
            HyperparameterType.LEARNING_RATE: (1e-6, 1e-2),
            HyperparameterType.BATCH_SIZE: (16, 512),
            HyperparameterType.REGULARIZATION_STRENGTH: (1e-6, 1e-2),
            HyperparameterType.DROPOUT_RATE: (0.0, 0.5),
            HyperparameterType.GRADIENT_CLIP_VALUE: (0.1, 10.0),
            HyperparameterType.WEIGHT_DECAY: (1e-6, 1e-3),
        }
    )

    # 適応間隔設定
    adaptation_interval_minutes: int = 30  # 適応間隔
    min_adaptation_interval_minutes: int = 10  # 最小適応間隔

    # パフォーマンス評価設定
    performance_window_size: int = 100  # パフォーマンス評価ウィンドウ
    performance_improvement_threshold: float = 0.01  # パフォーマンス改善閾値

    # ボラティリティ適応設定
    volatility_window_minutes: int = 60  # ボラティリティ評価ウィンドウ
    high_volatility_threshold: float = 0.03  # 高ボラティリティ閾値
    low_volatility_threshold: float = 0.01  # 低ボラティリティ閾値

    # 安全マージン設定
    safety_margin: float = 0.1  # 安全マージン（パラメータ変更の10%以内に制限）
    max_parameter_change_rate: float = 0.2  # 最大パラメータ変更率

    # 履歴保持設定
    max_history_size: int = 1000  # 最大履歴サイズ
    history_retention_days: int = 7  # 履歴保持期間


@dataclass
class HyperparameterAdaptation:
    """ハイパーパラメータ適応結果"""

    parameter_type: HyperparameterType
    old_value: float
    new_value: float
    adaptation_strategy: AdaptationStrategy
    performance_score: float
    volatility_score: float
    timestamp: datetime
    reason: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptationResult:
    """適応結果"""

    adaptations: List[HyperparameterAdaptation]
    overall_performance_improvement: float
    adaptation_confidence: float
    timestamp: datetime
    market_conditions: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class DynamicHyperparameterAdapter:
    """動的ハイパーパラメータ適応マネージャー"""

    def __init__(
        self,
        online_learning_pipeline: OnlineLearningPipeline,
        evaluation_manager: ContinuousEvaluationManager,
        config: Optional[HyperparameterConfig] = None,
    ):
        self.online_learning = online_learning_pipeline
        self.evaluation_manager = evaluation_manager
        self.config = config or HyperparameterConfig()

        # 現在のハイパーパラメータ値
        self.current_parameters: Dict[HyperparameterType, float] = {}
        self._initialize_current_parameters()

        # パフォーマンス履歴
        self.performance_history: List[Tuple[datetime, float]] = []
        self.parameter_history: Dict[
            HyperparameterType, List[Tuple[datetime, float]]
        ] = {param: [] for param in HyperparameterType}

        # 適応履歴
        self.adaptation_history: List[AdaptationResult] = []

        # 市場状態追跡
        self.market_volatility_history: List[Tuple[datetime, float]] = []

        # 状態管理
        self.is_active = False
        self.last_adaptation_time: Optional[datetime] = None

        # コールバック
        self.adaptation_callbacks: List[Callable[[AdaptationResult], None]] = []

        # スレッド管理
        self.adaptation_thread: Optional[threading.Thread] = None

        # 初期化
        self._initialize_adapter()

        logger.info("DynamicHyperparameterAdapter initialized")

    def _initialize_current_parameters(self) -> None:
        """現在のハイパーパラメータを初期化"""
        try:
            # オンライン学習パイプラインから現在の値を設定
            # （実際の実装では適切なメソッドを呼び出す）
            self.current_parameters = {
                HyperparameterType.LEARNING_RATE: 1e-4,
                HyperparameterType.BATCH_SIZE: 64,
                HyperparameterType.REGULARIZATION_STRENGTH: 1e-5,
                HyperparameterType.DROPOUT_RATE: 0.1,
                HyperparameterType.GRADIENT_CLIP_VALUE: 1.0,
                HyperparameterType.WEIGHT_DECAY: 1e-4,
            }

            logger.info(f"Initialized current parameters: {self.current_parameters}")

        except Exception as e:
            logger.error(f"Failed to initialize current parameters: {e}")

    def _initialize_adapter(self) -> None:
        """アダプターを初期化"""
        try:
            # 初期パフォーマンスを記録
            initial_performance = self._evaluate_current_performance()
            if initial_performance is not None:
                self.performance_history.append((datetime.now(), initial_performance))

            # 初期パラメータを履歴に記録
            current_time = datetime.now()
            for param_type, value in self.current_parameters.items():
                self.parameter_history[param_type].append((current_time, value))

        except Exception as e:
            logger.error(f"Failed to initialize adapter: {e}")

    def start_adaptation(self) -> bool:
        """適応を開始"""
        try:
            if self.is_active:
                logger.warning("Adaptation already active")
                return True

            self.is_active = True
            self.adaptation_thread = threading.Thread(
                target=self._adaptation_worker, daemon=True
            )
            self.adaptation_thread.start()

            logger.info("Dynamic hyperparameter adaptation started")
            return True

        except Exception as e:
            logger.error(f"Failed to start adaptation: {e}")
            return False

    def stop_adaptation(self) -> None:
        """適応を停止"""
        self.is_active = False
        if self.adaptation_thread and self.adaptation_thread.is_alive():
            self.adaptation_thread.join(timeout=5.0)
        logger.info("Dynamic hyperparameter adaptation stopped")

    def adapt_hyperparameters(
        self, market_data: Optional[pd.DataFrame] = None
    ) -> AdaptationResult:
        """ハイパーパラメータを適応"""
        try:
            current_time = datetime.now()

            # 市場状態を評価
            market_conditions = self._evaluate_market_conditions(market_data)

            # 各戦略で適応を計算
            all_adaptations = []
            for strategy in self.config.enabled_strategies:
                try:
                    adaptations = self._adapt_with_strategy(strategy, market_conditions)
                    all_adaptations.extend(adaptations)
                except Exception as e:
                    logger.error(f"Error in {strategy.value} adaptation: {e}")

            # 適応を統合
            final_adaptations = self._combine_adaptations(all_adaptations)

            # パラメータを適用
            applied_adaptations = self._apply_adaptations(final_adaptations)

            # パフォーマンスを評価
            performance_improvement = self._evaluate_adaptation_performance(
                applied_adaptations
            )

            # 結果を作成
            result = AdaptationResult(
                adaptations=applied_adaptations,
                overall_performance_improvement=performance_improvement,
                adaptation_confidence=self._calculate_adaptation_confidence(
                    applied_adaptations
                ),
                timestamp=current_time,
                market_conditions=market_conditions,
            )

            # 履歴に追加
            self.adaptation_history.append(result)
            if len(self.adaptation_history) > 100:
                self.adaptation_history = self.adaptation_history[-100:]

            # コールバックを実行
            self._trigger_adaptation_callbacks(result)

            logger.info(
                f"Adapted {len(applied_adaptations)} hyperparameters with {performance_improvement:.4f} performance improvement"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to adapt hyperparameters: {e}")
            return AdaptationResult(
                adaptations=[],
                overall_performance_improvement=0.0,
                adaptation_confidence=0.0,
                timestamp=datetime.now(),
                reason="Adaptation failed due to error",
            )

    def _evaluate_market_conditions(
        self, market_data: Optional[pd.DataFrame]
    ) -> Dict[str, float]:
        """市場条件を評価"""
        try:
            conditions = {}

            if market_data is not None and len(market_data) > 10:
                # 価格データからボラティリティを計算
                if "close" in market_data.columns:
                    prices = market_data["close"].values
                    returns = np.diff(prices) / prices[:-1]
                    volatility = np.std(returns)
                    conditions["volatility"] = float(volatility)
                else:
                    conditions["volatility"] = 0.02  # デフォルト値

                # トレンド強度を計算
                if len(prices) > 20:
                    sma_short = pd.Series(prices).rolling(10).mean()
                    sma_long = pd.Series(prices).rolling(20).mean()
                    trend_strength = (
                        abs(sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
                    )
                    conditions["trend_strength"] = float(trend_strength)
                else:
                    conditions["trend_strength"] = 0.0

                # 市場状態を分類
                volatility = conditions["volatility"]
                if volatility > self.config.high_volatility_threshold:
                    conditions["market_state"] = 2.0  # 高ボラティリティ
                elif volatility < self.config.low_volatility_threshold:
                    conditions["market_state"] = 0.0  # 低ボラティリティ
                else:
                    conditions["market_state"] = 1.0  # 中ボラティリティ

            else:
                # デフォルト値
                conditions["volatility"] = 0.02
                conditions["trend_strength"] = 0.0
                conditions["market_state"] = 1.0

            # 履歴に追加
            self.market_volatility_history.append(
                (datetime.now(), conditions["volatility"])
            )
            if len(self.market_volatility_history) > 1000:
                self.market_volatility_history = self.market_volatility_history[-1000:]

            return conditions

        except Exception as e:
            logger.error(f"Failed to evaluate market conditions: {e}")
            return {"volatility": 0.02, "trend_strength": 0.0, "market_state": 1.0}

    def _adapt_with_strategy(
        self, strategy: AdaptationStrategy, market_conditions: Dict[str, float]
    ) -> List[HyperparameterAdaptation]:
        """指定された戦略で適応"""
        try:
            if strategy == AdaptationStrategy.PERFORMANCE_BASED:
                return self._performance_based_adaptation(market_conditions)
            elif strategy == AdaptationStrategy.VOLATILITY_BASED:
                return self._volatility_based_adaptation(market_conditions)
            elif strategy == AdaptationStrategy.GRADIENT_BASED:
                return self._gradient_based_adaptation(market_conditions)
            elif strategy == AdaptationStrategy.CURRICULUM_BASED:
                return self._curriculum_based_adaptation(market_conditions)
            else:
                return []

        except Exception as e:
            logger.error(f"Error in {strategy.value} adaptation: {e}")
            return []

    def _performance_based_adaptation(
        self, market_conditions: Dict[str, float]
    ) -> List[HyperparameterAdaptation]:
        """パフォーマンスベース適応"""
        adaptations = []

        try:
            # 最近のパフォーマンスを取得
            recent_performance = self._get_recent_performance(
                window_size=self.config.performance_window_size
            )

            if len(recent_performance) < 10:
                return adaptations

            # パフォーマンストレンドを分析
            performance_trend = self._calculate_performance_trend(recent_performance)

            for param_type in self.config.enabled_parameters:
                try:
                    # パフォーマンスベースの最適化
                    optimal_value = self._optimize_parameter_for_performance(
                        param_type, performance_trend, market_conditions
                    )

                    if optimal_value is not None:
                        current_value = self.current_parameters[param_type]
                        if (
                            abs(optimal_value - current_value) / current_value
                            > self.config.performance_improvement_threshold
                        ):
                            adaptation = HyperparameterAdaptation(
                                parameter_type=param_type,
                                old_value=current_value,
                                new_value=optimal_value,
                                adaptation_strategy=AdaptationStrategy.PERFORMANCE_BASED,
                                performance_score=performance_trend,
                                volatility_score=market_conditions.get(
                                    "volatility", 0.0
                                ),
                                timestamp=datetime.now(),
                                reason=f"Performance-based optimization: trend={performance_trend:.4f}",
                            )
                            adaptations.append(adaptation)

                except Exception as e:
                    logger.error(f"Error adapting {param_type.value}: {e}")

            return adaptations

        except Exception as e:
            logger.error(f"Performance-based adaptation failed: {e}")
            return []

    def _volatility_based_adaptation(
        self, market_conditions: Dict[str, float]
    ) -> List[HyperparameterAdaptation]:
        """ボラティリティベース適応"""
        adaptations = []

        try:
            volatility = market_conditions.get("volatility", 0.02)
            market_state = market_conditions.get("market_state", 1.0)

            for param_type in self.config.enabled_parameters:
                try:
                    # ボラティリティに応じたパラメータ調整
                    new_value = self._adjust_parameter_for_volatility(
                        param_type, volatility, market_state
                    )

                    if new_value is not None:
                        current_value = self.current_parameters[param_type]
                        change_rate = abs(new_value - current_value) / max(
                            current_value, 1e-10
                        )

                        if change_rate > self.config.performance_improvement_threshold:
                            adaptation = HyperparameterAdaptation(
                                parameter_type=param_type,
                                old_value=current_value,
                                new_value=new_value,
                                adaptation_strategy=AdaptationStrategy.VOLATILITY_BASED,
                                performance_score=0.0,  # パフォーマンスベースではない
                                volatility_score=volatility,
                                timestamp=datetime.now(),
                                reason=f"Volatility-based adaptation: volatility={volatility:.4f}",
                            )
                            adaptations.append(adaptation)

                except Exception as e:
                    logger.error(
                        f"Error adapting {param_type.value} for volatility: {e}"
                    )

            return adaptations

        except Exception as e:
            logger.error(f"Volatility-based adaptation failed: {e}")
            return []

    def _gradient_based_adaptation(
        self, market_conditions: Dict[str, float]
    ) -> List[HyperparameterAdaptation]:
        """勾配ベース適応"""
        adaptations = []

        try:
            # 勾配情報を取得（実際の実装ではモデルから取得）
            gradient_stats = self._get_gradient_statistics()

            for param_type in [
                HyperparameterType.LEARNING_RATE,
                HyperparameterType.GRADIENT_CLIP_VALUE,
            ]:
                try:
                    if param_type in self.config.enabled_parameters:
                        new_value = self._adjust_parameter_for_gradients(
                            param_type, gradient_stats
                        )

                        if new_value is not None:
                            current_value = self.current_parameters[param_type]
                            if (
                                abs(new_value - current_value) / current_value > 0.05
                            ):  # 5%以上の変更
                                adaptation = HyperparameterAdaptation(
                                    parameter_type=param_type,
                                    old_value=current_value,
                                    new_value=new_value,
                                    adaptation_strategy=AdaptationStrategy.GRADIENT_BASED,
                                    performance_score=0.0,
                                    volatility_score=market_conditions.get(
                                        "volatility", 0.0
                                    ),
                                    timestamp=datetime.now(),
                                    reason="Gradient-based adaptation",
                                )
                                adaptations.append(adaptation)

                except Exception as e:
                    logger.error(
                        f"Error adapting {param_type.value} for gradients: {e}"
                    )

            return adaptations

        except Exception as e:
            logger.error(f"Gradient-based adaptation failed: {e}")
            return []

    def _curriculum_based_adaptation(
        self, market_conditions: Dict[str, float]
    ) -> List[HyperparameterAdaptation]:
        """カリキュラムベース適応"""
        adaptations = []

        try:
            # 学習進捗を取得
            training_progress = self._get_training_progress()

            for param_type in self.config.enabled_parameters:
                try:
                    new_value = self._adjust_parameter_for_curriculum(
                        param_type, training_progress
                    )

                    if new_value is not None:
                        current_value = self.current_parameters[param_type]
                        if (
                            abs(new_value - current_value) / current_value > 0.1
                        ):  # 10%以上の変更
                            adaptation = HyperparameterAdaptation(
                                parameter_type=param_type,
                                old_value=current_value,
                                new_value=new_value,
                                adaptation_strategy=AdaptationStrategy.CURRICULUM_BASED,
                                performance_score=0.0,
                                volatility_score=market_conditions.get(
                                    "volatility", 0.0
                                ),
                                timestamp=datetime.now(),
                                reason=f"Curriculum-based adaptation: progress={training_progress:.2f}",
                            )
                            adaptations.append(adaptation)

                except Exception as e:
                    logger.error(
                        f"Error adapting {param_type.value} for curriculum: {e}"
                    )

            return adaptations

        except Exception as e:
            logger.error(f"Curriculum-based adaptation failed: {e}")
            return []

    def _optimize_parameter_for_performance(
        self,
        param_type: HyperparameterType,
        performance_trend: float,
        market_conditions: Dict[str, float],
    ) -> Optional[float]:
        """パフォーマンスのためのパラメータ最適化"""
        try:
            current_value = self.current_parameters[param_type]
            param_range = self.config.parameter_ranges[param_type]

            # 最適化関数を定義
            def objective_function(param_value):
                # パラメータ変更によるパフォーマンス予測
                # （実際の実装ではより洗練されたモデルを使用）
                change_ratio = abs(param_value - current_value) / current_value
                if performance_trend > 0:
                    # パフォーマンスが向上傾向の場合、保守的に
                    return change_ratio * 0.1
                else:
                    # パフォーマンスが低下傾向の場合、積極的に
                    return -change_ratio * 0.2

            # 最適化を実行
            result = minimize_scalar(
                objective_function, bounds=param_range, method="bounded"
            )

            if result.success:
                optimal_value = float(result.x)

                # 安全マージンを適用
                max_change = current_value * self.config.max_parameter_change_rate
                optimal_value = np.clip(
                    optimal_value,
                    current_value - max_change,
                    current_value + max_change,
                )

                return optimal_value

            return None

        except Exception as e:
            logger.error(f"Parameter optimization failed for {param_type.value}: {e}")
            return None

    def _adjust_parameter_for_volatility(
        self, param_type: HyperparameterType, volatility: float, market_state: float
    ) -> Optional[float]:
        """ボラティリティに応じたパラメータ調整"""
        try:
            current_value = self.current_parameters[param_type]

            if param_type == HyperparameterType.LEARNING_RATE:
                # 高ボラティリティ時は学習率を下げる
                if market_state == 2.0:  # 高ボラティリティ
                    adjustment_factor = 0.7
                elif market_state == 0.0:  # 低ボラティリティ
                    adjustment_factor = 1.2
                else:
                    adjustment_factor = 1.0

            elif param_type == HyperparameterType.BATCH_SIZE:
                # 高ボラティリティ時はバッチサイズを小さく
                if market_state == 2.0:
                    adjustment_factor = 0.8
                elif market_state == 0.0:
                    adjustment_factor = 1.1
                else:
                    adjustment_factor = 1.0

            elif param_type == HyperparameterType.REGULARIZATION_STRENGTH:
                # 高ボラティリティ時は正則化を強める
                if market_state == 2.0:
                    adjustment_factor = 1.5
                elif market_state == 0.0:
                    adjustment_factor = 0.8
                else:
                    adjustment_factor = 1.0

            else:
                return None

            new_value = current_value * adjustment_factor

            # 範囲内に収める
            param_range = self.config.parameter_ranges[param_type]
            new_value = np.clip(new_value, param_range[0], param_range[1])

            return float(new_value)

        except Exception as e:
            logger.error(f"Volatility adjustment failed for {param_type.value}: {e}")
            return None

    def _combine_adaptations(
        self, adaptations: List[HyperparameterAdaptation]
    ) -> List[HyperparameterAdaptation]:
        """適応を統合"""
        try:
            # パラメータタイプごとに統合
            combined_adaptations = {}
            strategy_weights = {
                AdaptationStrategy.PERFORMANCE_BASED: 0.5,
                AdaptationStrategy.VOLATILITY_BASED: 0.3,
                AdaptationStrategy.GRADIENT_BASED: 0.1,
                AdaptationStrategy.CURRICULUM_BASED: 0.1,
            }

            for adaptation in adaptations:
                param_type = adaptation.parameter_type

                if param_type not in combined_adaptations:
                    combined_adaptations[param_type] = {
                        "values": [],
                        "weights": [],
                        "strategies": [],
                    }

                combined_adaptations[param_type]["values"].append(adaptation.new_value)
                combined_adaptations[param_type]["weights"].append(
                    strategy_weights.get(adaptation.adaptation_strategy, 0.1)
                )
                combined_adaptations[param_type]["strategies"].append(
                    adaptation.adaptation_strategy
                )

            # 重み付き平均を計算
            final_adaptations = []
            for param_type, data in combined_adaptations.items():
                if data["values"]:
                    weights = np.array(data["weights"])
                    weights = weights / np.sum(weights)  # 正規化

                    combined_value = np.average(data["values"], weights=weights)

                    # 主要な戦略を選択
                    main_strategy = data["strategies"][np.argmax(weights)]

                    final_adaptation = HyperparameterAdaptation(
                        parameter_type=param_type,
                        old_value=self.current_parameters[param_type],
                        new_value=float(combined_value),
                        adaptation_strategy=main_strategy,
                        performance_score=0.0,
                        volatility_score=0.0,
                        timestamp=datetime.now(),
                        reason=f"Combined adaptation from {len(data['strategies'])} strategies",
                    )
                    final_adaptations.append(final_adaptation)

            return final_adaptations

        except Exception as e:
            logger.error(f"Failed to combine adaptations: {e}")
            return adaptations

    def _apply_adaptations(
        self, adaptations: List[HyperparameterAdaptation]
    ) -> List[HyperparameterAdaptation]:
        """適応を適用"""
        applied_adaptations = []

        try:
            for adaptation in adaptations:
                # 安全チェック
                if self._is_adaptation_safe(adaptation):
                    # パラメータを更新
                    self.current_parameters[
                        adaptation.parameter_type
                    ] = adaptation.new_value

                    # 履歴に記録
                    current_time = datetime.now()
                    self.parameter_history[adaptation.parameter_type].append(
                        (current_time, adaptation.new_value)
                    )

                    applied_adaptations.append(adaptation)

                    # オンライン学習パイプラインに適用
                    self._apply_to_pipeline(adaptation)

                    logger.info(
                        f"Applied adaptation: {adaptation.parameter_type.value} = {adaptation.new_value}"
                    )

        except Exception as e:
            logger.error(f"Failed to apply adaptations: {e}")

        return applied_adaptations

    def _is_adaptation_safe(self, adaptation: HyperparameterAdaptation) -> bool:
        """適応が安全かチェック"""
        try:
            old_value = adaptation.old_value
            new_value = adaptation.new_value

            # 範囲チェック
            param_range = self.config.parameter_ranges[adaptation.parameter_type]
            if not (param_range[0] <= new_value <= param_range[1]):
                return False

            # 変更率チェック
            change_rate = abs(new_value - old_value) / max(old_value, 1e-10)
            if change_rate > self.config.max_parameter_change_rate:
                return False

            # 安全マージンチェック
            safety_change = old_value * self.config.safety_margin
            if abs(new_value - old_value) > safety_change:
                return False

            return True

        except Exception:
            return False

    def _apply_to_pipeline(self, adaptation: HyperparameterAdaptation) -> None:
        """パイプラインに適応を適用"""
        try:
            # オンライン学習パイプラインにパラメータを適用
            # （実際の実装では適切なメソッドを呼び出す）
            param_name = adaptation.parameter_type.value
            param_value = adaptation.new_value

            # パイプラインの設定を更新
            if hasattr(self.online_learning, "update_hyperparameter"):
                self.online_learning.update_hyperparameter(param_name, param_value)

        except Exception as e:
            logger.error(f"Failed to apply adaptation to pipeline: {e}")

    def _evaluate_adaptation_performance(
        self, adaptations: List[HyperparameterAdaptation]
    ) -> float:
        """適応のパフォーマンスを評価"""
        try:
            if not adaptations:
                return 0.0

            # 適応前のパフォーマンスを取得
            before_performance = self._get_recent_performance(window_size=50)

            # 短時間待機して適応効果を評価
            time.sleep(5)  # 5秒待機

            # 適応後のパフォーマンスを取得
            after_performance = self._evaluate_current_performance()

            if before_performance and after_performance:
                # パフォーマンス改善を計算
                avg_before = np.mean([p for _, p in before_performance])
                improvement = after_performance - avg_before
                return float(improvement)

            return 0.0

        except Exception as e:
            logger.error(f"Failed to evaluate adaptation performance: {e}")
            return 0.0

    def _calculate_adaptation_confidence(
        self, adaptations: List[HyperparameterAdaptation]
    ) -> float:
        """適応の確信度を計算"""
        try:
            if not adaptations:
                return 0.0

            # 適応の数と変更率に基づいて確信度を計算
            num_adaptations = len(adaptations)
            avg_change_rate = np.mean(
                [
                    abs(a.new_value - a.old_value) / max(a.old_value, 1e-10)
                    for a in adaptations
                ]
            )

            # 適応数が多いほど、変更率が小さいほど確信度が高い
            confidence = min(num_adaptations / 5.0, 1.0) * (
                1.0 - min(avg_change_rate / 0.5, 1.0)
            )

            return float(confidence)

        except Exception:
            return 0.5

    def _get_recent_performance(
        self, window_size: int = 100
    ) -> List[Tuple[datetime, float]]:
        """最近のパフォーマンスを取得"""
        try:
            if len(self.performance_history) < window_size:
                return self.performance_history[-window_size:]

            return self.performance_history[-window_size:]

        except Exception:
            return []

    def _evaluate_current_performance(self) -> Optional[float]:
        """現在のパフォーマンスを評価"""
        try:
            # 評価マネージャーからパフォーマンスを取得
            if hasattr(self.evaluation_manager, "get_current_performance"):
                return self.evaluation_manager.get_current_performance()

            # デフォルト値
            return 0.5

        except Exception as e:
            logger.error(f"Failed to evaluate current performance: {e}")
            return None

    def _calculate_performance_trend(
        self, performance_data: List[Tuple[datetime, float]]
    ) -> float:
        """パフォーマンストレンドを計算"""
        try:
            if len(performance_data) < 5:
                return 0.0

            # 線形回帰でトレンドを計算
            times = np.array(
                [
                    (t - performance_data[0][0]).total_seconds()
                    for t, _ in performance_data
                ]
            )
            performances = np.array([p for _, p in performance_data])

            slope, _, _, _, _ = stats.linregress(times, performances)
            return float(slope)

        except Exception:
            return 0.0

    def _get_gradient_statistics(self) -> Dict[str, float]:
        """勾配統計を取得"""
        try:
            # 実際の実装ではモデルから勾配情報を取得
            return {
                "gradient_norm": 1.0,
                "gradient_variance": 0.1,
                "gradient_exploding": False,
            }

        except Exception:
            return {
                "gradient_norm": 1.0,
                "gradient_variance": 0.1,
                "gradient_exploding": False,
            }

    def _get_training_progress(self) -> float:
        """学習進捗を取得"""
        try:
            # 実際の実装では学習マネージャーから進捗を取得
            return 0.5  # 50%完了

        except Exception:
            return 0.5

    def _adjust_parameter_for_gradients(
        self, param_type: HyperparameterType, gradient_stats: Dict[str, float]
    ) -> Optional[float]:
        """勾配に応じたパラメータ調整"""
        try:
            current_value = self.current_parameters[param_type]

            if param_type == HyperparameterType.LEARNING_RATE:
                gradient_norm = gradient_stats.get("gradient_norm", 1.0)
                # 勾配が大きい場合は学習率を下げる
                if gradient_norm > 2.0:
                    return current_value * 0.9
                elif gradient_norm < 0.5:
                    return current_value * 1.1

            elif param_type == HyperparameterType.GRADIENT_CLIP_VALUE:
                gradient_norm = gradient_stats.get("gradient_norm", 1.0)
                # 勾配クリッピング値を適応
                return min(max(gradient_norm * 0.5, 0.1), 10.0)

            return None

        except Exception:
            return None

    def _adjust_parameter_for_curriculum(
        self, param_type: HyperparameterType, progress: float
    ) -> Optional[float]:
        """カリキュラムに応じたパラメータ調整"""
        try:
            current_value = self.current_parameters[param_type]

            if param_type == HyperparameterType.LEARNING_RATE:
                # 学習初期は高く、後期は低く
                if progress < 0.3:
                    return current_value * 1.2  # 初期は高い学習率
                elif progress > 0.7:
                    return current_value * 0.8  # 後期は低い学習率

            elif param_type == HyperparameterType.REGULARIZATION_STRENGTH:
                # 学習後期は正則化を強める
                if progress > 0.6:
                    return current_value * 1.5

            return None

        except Exception:
            return None

    def _adaptation_worker(self) -> None:
        """適応ワーカー"""
        while self.is_active:
            try:
                current_time = datetime.now()

                # 適応間隔をチェック
                if (
                    self.last_adaptation_time is None
                    or (current_time - self.last_adaptation_time).total_seconds()
                    >= self.config.adaptation_interval_minutes * 60
                ):
                    # 適応を実行
                    self.adapt_hyperparameters()
                    self.last_adaptation_time = current_time

                time.sleep(60)  # 1分ごとにチェック

            except Exception as e:
                logger.error(f"Adaptation worker error: {e}")
                time.sleep(300)  # エラー時は5分待機

    def add_adaptation_callback(
        self, callback: Callable[[AdaptationResult], None]
    ) -> None:
        """適応コールバックを追加"""
        self.adaptation_callbacks.append(callback)

    def _trigger_adaptation_callbacks(self, result: AdaptationResult) -> None:
        """適応コールバックを実行"""
        for callback in self.adaptation_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Adaptation callback failed: {e}")

    def get_adaptation_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """適応履歴を取得"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_adaptations = [
                a for a in self.adaptation_history if a.timestamp > cutoff_time
            ]

            return [
                {
                    "timestamp": a.timestamp.isoformat(),
                    "num_adaptations": len(a.adaptations),
                    "performance_improvement": a.overall_performance_improvement,
                    "confidence": a.adaptation_confidence,
                    "market_conditions": a.market_conditions,
                }
                for a in recent_adaptations
            ]

        except Exception as e:
            logger.error(f"Failed to get adaptation history: {e}")
            return []

    def get_parameter_history(
        self, param_type: HyperparameterType, hours: int = 24
    ) -> List[Dict[str, Any]]:
        """パラメータ履歴を取得"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_history = [
                (t, v) for t, v in self.parameter_history[param_type] if t > cutoff_time
            ]

            return [{"timestamp": t.isoformat(), "value": v} for t, v in recent_history]

        except Exception as e:
            logger.error(f"Failed to get parameter history: {e}")
            return []

    def get_current_hyperparameters(self) -> Dict[str, float]:
        """現在のハイパーパラメータを取得"""
        try:
            return {
                param_type.value: value
                for param_type, value in self.current_parameters.items()
            }

        except Exception as e:
            logger.error(f"Failed to get current hyperparameters: {e}")
            return {}
