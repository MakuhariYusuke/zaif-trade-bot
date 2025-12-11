"""
Market-Aware Hyperparameter Adaptation Manager
市場対応ハイパーパラメータ適応マネージャー
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from .dynamic_hyperparameter_adapter import (
    AdaptationResult,
    AdaptationStrategy,
    DynamicHyperparameterAdapter,
    HyperparameterConfig,
    HyperparameterType,
)
from .monitoring.evaluation_manager import ContinuousEvaluationManager

if TYPE_CHECKING:
    from .online_learning.pipeline import OnlineLearningPipeline

# from .monitoring.market_regime_detector import MarketRegimeDetector  # Optional component


logger = logging.getLogger(__name__)


@dataclass
class MarketAwareConfig:
    """市場対応設定"""

    # 市場適応設定
    market_regime_adaptation: bool = True
    volatility_based_scaling: bool = True
    trend_adaptive_parameters: bool = True

    # 学習設定
    online_learning_enabled: bool = True
    meta_learning_enabled: bool = True
    transfer_learning_enabled: bool = True

    # 予測モデル設定
    use_performance_prediction: bool = True
    prediction_model_update_interval: int = 60  # 分単位
    prediction_history_window: int = 1000

    # 適応戦略設定
    adaptive_strategy_selection: bool = True
    strategy_performance_tracking: bool = True
    strategy_switching_threshold: float = 0.05

    # リスク管理設定
    risk_aware_adaptation: bool = True
    max_risk_adjustment: float = 0.3
    risk_sensitivity: float = 0.8

    # 履歴と学習設定
    history_retention_days: int = 30
    model_retraining_interval: int = 1440  # 分単位（24時間）
    min_training_samples: int = 100


@dataclass
class MarketCondition:
    """市場条件"""

    timestamp: datetime
    volatility: float
    trend_strength: float
    market_regime: str
    volume_profile: float
    liquidity_score: float
    sentiment_score: float = 0.0

    def to_features(self) -> np.ndarray:
        """特徴量ベクトルに変換"""
        return np.array(
            [
                self.volatility,
                self.trend_strength,
                self.volume_profile,
                self.liquidity_score,
                self.sentiment_score,
                # 市場レジームをワンホットエンコーディング
                1.0 if self.market_regime == "bull" else 0.0,
                1.0 if self.market_regime == "bear" else 0.0,
                1.0 if self.market_regime == "sideways" else 0.0,
                1.0 if self.market_regime == "volatile" else 0.0,
            ]
        )


@dataclass
class PerformancePrediction:
    """パフォーマンス予測"""

    parameter_type: HyperparameterType
    predicted_performance: float
    confidence: float
    optimal_value: float
    feature_importance: Dict[str, float] = field(default_factory=dict)


class MarketAwareHyperparameterManager:
    """市場対応ハイパーパラメータマネージャー"""

    def __init__(
        self,
        online_learning_pipeline: "OnlineLearningPipeline",
        evaluation_manager: ContinuousEvaluationManager,
        market_regime_detector: Optional[Any] = None,  # Optional component
        config: Optional[MarketAwareConfig] = None,
    ):
        self.online_learning = online_learning_pipeline
        self.evaluation_manager = evaluation_manager
        self.market_detector = market_regime_detector
        self.config = config or MarketAwareConfig()

        # 動的適応アダプター
        self.hyperparameter_adapter = DynamicHyperparameterAdapter(
            online_learning_pipeline, evaluation_manager
        )

        # 市場条件履歴
        self.market_conditions: List[MarketCondition] = []

        # パフォーマンス予測モデル
        self.performance_predictors: Dict[
            HyperparameterType, RandomForestRegressor
        ] = {}
        self.scalers: Dict[HyperparameterType, StandardScaler] = {}

        # 適応履歴と学習データ
        self.adaptation_history: List[Tuple[MarketCondition, AdaptationResult]] = []
        self.training_data: Dict[HyperparameterType, List[Tuple[np.ndarray, float]]] = {
            param: [] for param in HyperparameterType
        }

        # 戦略パフォーマンス追跡
        self.strategy_performance: Dict[AdaptationStrategy, List[float]] = {
            strategy: [] for strategy in AdaptationStrategy
        }

        # 状態管理
        self.is_active = False
        self.last_prediction_update = datetime.now()
        self.last_model_training = datetime.now()

        # スレッド管理
        self.worker_thread: Optional[threading.Thread] = None

        # 初期化
        self._initialize_predictors()
        self._initialize_manager()

        logger.info("MarketAwareHyperparameterManager initialized")

    def _initialize_predictors(self) -> None:
        """パフォーマンス予測モデルを初期化"""
        try:
            for param_type in HyperparameterType:
                # ランダムフォレスト回帰モデル
                self.performance_predictors[param_type] = RandomForestRegressor(
                    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
                )

                # 特徴量スケーラー
                self.scalers[param_type] = StandardScaler()

            logger.info("Performance predictors initialized")

        except Exception as e:
            logger.error(f"Failed to initialize predictors: {e}")

    def _initialize_manager(self) -> None:
        """マネージャーを初期化"""
        try:
            # 初期市場条件を取得
            initial_condition = self._get_current_market_condition()
            if initial_condition:
                self.market_conditions.append(initial_condition)

        except Exception as e:
            logger.error(f"Failed to initialize manager: {e}")

    def start_adaptation(self) -> bool:
        """適応を開始"""
        try:
            if self.is_active:
                logger.warning("Market-aware adaptation already active")
                return True

            # 動的適応を開始
            if not self.hyperparameter_adapter.start_adaptation():
                logger.error("Failed to start hyperparameter adapter")
                return False

            self.is_active = True
            self.worker_thread = threading.Thread(
                target=self._adaptation_worker, daemon=True
            )
            self.worker_thread.start()

            logger.info("Market-aware hyperparameter adaptation started")
            return True

        except Exception as e:
            logger.error(f"Failed to start adaptation: {e}")
            return False

    def stop_adaptation(self) -> None:
        """適応を停止"""
        self.is_active = False
        self.hyperparameter_adapter.stop_adaptation()

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)

        logger.info("Market-aware hyperparameter adaptation stopped")

    def adapt_hyperparameters_market_aware(
        self, market_data: Optional[pd.DataFrame] = None, force_adaptation: bool = False
    ) -> AdaptationResult:
        """市場対応ハイパーパラメータ適応"""
        try:
            current_time = datetime.now()

            # 市場条件を評価
            market_condition = self._get_current_market_condition(market_data)
            if market_condition:
                self.market_conditions.append(market_condition)
                if len(self.market_conditions) > 1000:
                    self.market_conditions = self.market_conditions[-1000:]

            # 適応が必要かチェック
            if not force_adaptation and not self._should_adapt_now(market_condition):
                return AdaptationResult(
                    adaptations=[],
                    overall_performance_improvement=0.0,
                    adaptation_confidence=0.0,
                    timestamp=current_time,
                    recommendations=["No adaptation needed at this time"],
                )

            # パフォーマンス予測を取得
            performance_predictions = self._predict_optimal_parameters(market_condition)

            # 適応戦略を選択
            selected_strategies = self._select_adaptation_strategies(
                market_condition, performance_predictions
            )

            # ハイパーパラメータ設定を更新
            updated_config = self._create_adaptive_config(
                market_condition, selected_strategies
            )

            # 動的適応を実行
            adaptation_result = self.hyperparameter_adapter.adapt_hyperparameters(
                market_data
            )

            # 結果を学習データに追加
            if market_condition:
                self.adaptation_history.append((market_condition, adaptation_result))
                self._update_training_data(market_condition, adaptation_result)

            # 戦略パフォーマンスを更新
            self._update_strategy_performance(adaptation_result)

            # モデルを更新（定期的に）
            if (
                current_time - self.last_model_training
            ).total_seconds() >= self.config.model_retraining_interval * 60:
                self._retrain_prediction_models()
                self.last_model_training = current_time

            logger.info(
                f"Market-aware adaptation completed with {len(adaptation_result.adaptations)} parameter changes"
            )
            return adaptation_result

        except Exception as e:
            logger.error(f"Market-aware adaptation failed: {e}")
            return AdaptationResult(
                adaptations=[],
                overall_performance_improvement=0.0,
                adaptation_confidence=0.0,
                timestamp=datetime.now(),
                recommendations=["Adaptation failed due to error"],
            )

    def _get_current_market_condition(
        self, market_data: Optional[pd.DataFrame] = None
    ) -> Optional[MarketCondition]:
        """現在の市場条件を取得"""
        try:
            # 市場データから条件を抽出
            if market_data is not None and len(market_data) > 10:
                # ボラティリティ計算
                if "close" in market_data.columns:
                    prices = market_data["close"].values
                    returns = np.diff(prices) / prices[:-1]
                    volatility = np.std(returns)
                else:
                    volatility = 0.02

                # トレンド強度
                if len(prices) > 20:
                    sma_short = pd.Series(prices).rolling(10).mean()
                    sma_long = pd.Series(prices).rolling(20).mean()
                    trend_strength = (
                        abs(sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
                    )
                else:
                    trend_strength = 0.0

                # 出来高プロファイル
                if "volume" in market_data.columns:
                    volume_profile = market_data["volume"].mean()
                else:
                    volume_profile = 1000.0

                # 流動性スコア（仮定）
                liquidity_score = 0.8

            else:
                # デフォルト値
                volatility = 0.02
                trend_strength = 0.0
                volume_profile = 1000.0
                liquidity_score = 0.8

            # 市場レジームを取得
            market_regime = "neutral"
            if self.market_detector:
                try:
                    regime_info = self.market_detector.detect_regime(market_data)
                    market_regime = regime_info.get("regime", "neutral")
                except Exception:
                    pass

            return MarketCondition(
                timestamp=datetime.now(),
                volatility=float(volatility),
                trend_strength=float(trend_strength),
                market_regime=market_regime,
                volume_profile=float(volume_profile),
                liquidity_score=float(liquidity_score),
            )

        except Exception as e:
            logger.error(f"Failed to get market condition: {e}")
            return None

    def _should_adapt_now(self, market_condition: Optional[MarketCondition]) -> bool:
        """適応が必要かチェック"""
        try:
            if not market_condition:
                return False

            # 最終適応からの時間をチェック
            last_adaptation = self.hyperparameter_adapter.last_adaptation_time
            if last_adaptation:
                time_since_adaptation = (
                    datetime.now() - last_adaptation
                ).total_seconds() / 60
                if (
                    time_since_adaptation
                    < self.hyperparameter_adapter.config.min_adaptation_interval_minutes
                ):
                    return False

            # 市場条件の大きな変化をチェック
            if len(self.market_conditions) >= 2:
                prev_condition = self.market_conditions[-2]

                volatility_change = abs(
                    market_condition.volatility - prev_condition.volatility
                )
                trend_change = abs(
                    market_condition.trend_strength - prev_condition.trend_strength
                )

                # 閾値を超える変化があれば適応
                if volatility_change > 0.01 or trend_change > 0.05:
                    return True

            # 定期適応
            return True

        except Exception:
            return False

    def _predict_optimal_parameters(
        self, market_condition: Optional[MarketCondition]
    ) -> Dict[HyperparameterType, PerformancePrediction]:
        """最適パラメータを予測"""
        predictions = {}

        try:
            if not market_condition or not self.config.use_performance_prediction:
                return predictions

            features = market_condition.to_features().reshape(1, -1)

            for param_type in HyperparameterType:
                try:
                    predictor = self.performance_predictors[param_type]
                    scaler = self.scalers[param_type]

                    # 特徴量をスケーリング
                    scaled_features = scaler.transform(features)

                    # パフォーマンスを予測
                    predicted_performance = predictor.predict(scaled_features)[0]

                    # 最適値を推定（予測モデルに基づく）
                    optimal_value = self._estimate_optimal_value(
                        param_type, predicted_performance, market_condition
                    )

                    # 特徴量重要度を取得
                    feature_importance = {}
                    if hasattr(predictor, "feature_importances_"):
                        feature_names = [
                            "volatility",
                            "trend_strength",
                            "volume_profile",
                            "liquidity_score",
                            "sentiment_score",
                            "regime_bull",
                            "regime_bear",
                            "regime_sideways",
                            "regime_volatile",
                        ]
                        for name, importance in zip(
                            feature_names, predictor.feature_importances_
                        ):
                            feature_importance[name] = float(importance)

                    predictions[param_type] = PerformancePrediction(
                        parameter_type=param_type,
                        predicted_performance=float(predicted_performance),
                        confidence=0.8,  # 仮定値
                        optimal_value=float(optimal_value),
                        feature_importance=feature_importance,
                    )

                except Exception as e:
                    logger.error(f"Failed to predict for {param_type.value}: {e}")

            return predictions

        except Exception as e:
            logger.error(f"Parameter prediction failed: {e}")
            return predictions

    def _estimate_optimal_value(
        self,
        param_type: HyperparameterType,
        predicted_performance: float,
        market_condition: MarketCondition,
    ) -> float:
        """最適値を推定"""
        try:
            current_value = self.hyperparameter_adapter.current_parameters[param_type]
            param_range = self.hyperparameter_adapter.config.parameter_ranges[
                param_type
            ]

            # 市場条件に基づいて調整
            if market_condition.volatility > 0.05:  # 高ボラティリティ
                if param_type == HyperparameterType.LEARNING_RATE:
                    adjustment = 0.8  # 学習率を下げる
                elif param_type == HyperparameterType.BATCH_SIZE:
                    adjustment = 0.9  # バッチサイズを小さく
                elif param_type == HyperparameterType.REGULARIZATION_STRENGTH:
                    adjustment = 1.2  # 正則化を強める
                else:
                    adjustment = 1.0
            elif market_condition.volatility < 0.01:  # 低ボラティリティ
                if param_type == HyperparameterType.LEARNING_RATE:
                    adjustment = 1.1  # 学習率を上げる
                elif param_type == HyperparameterType.BATCH_SIZE:
                    adjustment = 1.1  # バッチサイズを大きく
                elif param_type == HyperparameterType.REGULARIZATION_STRENGTH:
                    adjustment = 0.9  # 正則化を弱める
                else:
                    adjustment = 1.0
            else:
                adjustment = 1.0

            optimal_value = current_value * adjustment

            # 範囲内に収める
            optimal_value = np.clip(optimal_value, param_range[0], param_range[1])

            return float(optimal_value)

        except Exception:
            return float(current_value)

    def _select_adaptation_strategies(
        self,
        market_condition: Optional[MarketCondition],
        predictions: Dict[HyperparameterType, PerformancePrediction],
    ) -> List[AdaptationStrategy]:
        """適応戦略を選択"""
        try:
            if not self.config.adaptive_strategy_selection:
                return list(AdaptationStrategy)

            strategies = []

            if market_condition:
                # 市場条件に基づいて戦略を選択
                if market_condition.volatility > 0.03:
                    strategies.append(AdaptationStrategy.VOLATILITY_BASED)
                if market_condition.trend_strength > 0.1:
                    strategies.append(AdaptationStrategy.PERFORMANCE_BASED)

            # パフォーマンス予測がある場合は追加
            if predictions:
                strategies.append(AdaptationStrategy.PERFORMANCE_BASED)

            # デフォルト戦略
            if not strategies:
                strategies = [
                    AdaptationStrategy.PERFORMANCE_BASED,
                    AdaptationStrategy.VOLATILITY_BASED,
                ]

            return strategies

        except Exception as e:
            logger.error(f"Strategy selection failed: {e}")
            return [AdaptationStrategy.PERFORMANCE_BASED]

    def _create_adaptive_config(
        self,
        market_condition: Optional[MarketCondition],
        strategies: List[AdaptationStrategy],
    ) -> HyperparameterConfig:
        """適応設定を作成"""
        try:
            config = HyperparameterConfig()
            config.enabled_strategies = strategies

            if market_condition:
                # 市場条件に基づいて設定を調整
                if market_condition.volatility > 0.05:
                    # 高ボラティリティ時はより頻繁に適応
                    config.adaptation_interval_minutes = max(
                        10, config.adaptation_interval_minutes // 2
                    )
                    config.max_parameter_change_rate = 0.15  # 変更率を制限
                elif market_condition.volatility < 0.01:
                    # 低ボラティリティ時は安定した適応
                    config.adaptation_interval_minutes = (
                        config.adaptation_interval_minutes * 2
                    )
                    config.max_parameter_change_rate = 0.25  # 変更率を緩和

            return config

        except Exception:
            return HyperparameterConfig()

    def _update_training_data(
        self, market_condition: MarketCondition, adaptation_result: AdaptationResult
    ) -> None:
        """学習データを更新"""
        try:
            features = market_condition.to_features()

            for adaptation in adaptation_result.adaptations:
                param_type = adaptation.parameter_type
                performance_score = adaptation_result.overall_performance_improvement

                self.training_data[param_type].append(
                    (features.copy(), performance_score)
                )

                # データサイズを制限
                if (
                    len(self.training_data[param_type])
                    > self.config.prediction_history_window
                ):
                    self.training_data[param_type] = self.training_data[param_type][
                        -self.config.prediction_history_window :
                    ]

        except Exception as e:
            logger.error(f"Failed to update training data: {e}")

    def _update_strategy_performance(self, adaptation_result: AdaptationResult) -> None:
        """戦略パフォーマンスを更新"""
        try:
            if not self.config.strategy_performance_tracking:
                return

            performance_score = adaptation_result.overall_performance_improvement

            # 各適応に使用された戦略のパフォーマンスを更新
            for adaptation in adaptation_result.adaptations:
                strategy = adaptation.adaptation_strategy
                self.strategy_performance[strategy].append(performance_score)

                # 履歴サイズを制限
                if len(self.strategy_performance[strategy]) > 100:
                    self.strategy_performance[strategy] = self.strategy_performance[
                        strategy
                    ][-100:]

        except Exception as e:
            logger.error(f"Failed to update strategy performance: {e}")

    def _retrain_prediction_models(self) -> None:
        """予測モデルを再学習"""
        try:
            for param_type in HyperparameterType:
                training_samples = self.training_data[param_type]

                if len(training_samples) < self.config.min_training_samples:
                    continue

                # データを準備
                X = np.array([features for features, _ in training_samples])
                y = np.array([performance for _, performance in training_samples])

                # スケーラーをフィット
                self.scalers[param_type].fit(X)

                # 特徴量をスケーリング
                X_scaled = self.scalers[param_type].transform(X)

                # モデルを学習
                self.performance_predictors[param_type].fit(X_scaled, y)

                logger.info(
                    f"Retrained prediction model for {param_type.value} with {len(training_samples)} samples"
                )

            self.last_model_training = datetime.now()

        except Exception as e:
            logger.error(f"Model retraining failed: {e}")

    def _adaptation_worker(self) -> None:
        """適応ワーカー"""
        while self.is_active:
            try:
                # 市場対応適応を実行
                self.adapt_hyperparameters_market_aware()

                # 予測モデル更新をチェック
                current_time = datetime.now()
                if (
                    current_time - self.last_prediction_update
                ).total_seconds() >= self.config.prediction_model_update_interval * 60:
                    self._retrain_prediction_models()
                    self.last_prediction_update = current_time

                time.sleep(300)  # 5分ごとに実行

            except Exception as e:
                logger.error(f"Adaptation worker error: {e}")
                time.sleep(600)  # エラー時は10分待機

    def get_adaptation_recommendations(
        self, market_condition: Optional[MarketCondition] = None
    ) -> List[str]:
        """適応推奨を取得"""
        recommendations = []

        try:
            if not market_condition:
                market_condition = self._get_current_market_condition()

            if market_condition:
                # ボラティリティに基づく推奨
                if market_condition.volatility > 0.05:
                    recommendations.append(
                        "High volatility detected - consider reducing learning rate and increasing regularization"
                    )
                elif market_condition.volatility < 0.01:
                    recommendations.append(
                        "Low volatility detected - consider increasing learning rate for faster adaptation"
                    )

                # トレンドに基づく推奨
                if market_condition.trend_strength > 0.1:
                    recommendations.append(
                        "Strong trend detected - performance-based adaptation recommended"
                    )

                # 市場レジームに基づく推奨
                if market_condition.market_regime == "volatile":
                    recommendations.append(
                        "Volatile market regime - focus on stability and risk management"
                    )
                elif market_condition.market_regime == "bull":
                    recommendations.append(
                        "Bull market regime - consider more aggressive parameter updates"
                    )

            # 戦略パフォーマンスに基づく推奨
            best_strategy = self._get_best_performing_strategy()
            if best_strategy:
                recommendations.append(
                    f"Best performing strategy: {best_strategy.value}"
                )

            if not recommendations:
                recommendations.append(
                    "Market conditions stable - no specific recommendations"
                )

            return recommendations

        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            return ["Unable to generate recommendations due to error"]

    def _get_best_performing_strategy(self) -> Optional[AdaptationStrategy]:
        """最高パフォーマンスの戦略を取得"""
        try:
            strategy_scores = {}
            for strategy, performances in self.strategy_performance.items():
                if performances:
                    avg_performance = np.mean(performances)
                    strategy_scores[strategy] = avg_performance

            if strategy_scores:
                return max(strategy_scores, key=strategy_scores.get)

            return None

        except Exception:
            return None

    def get_performance_predictions(self) -> Dict[str, Any]:
        """パフォーマンス予測を取得"""
        try:
            market_condition = self._get_current_market_condition()
            if not market_condition:
                return {}

            predictions = self._predict_optimal_parameters(market_condition)

            return {
                "market_condition": {
                    "volatility": market_condition.volatility,
                    "trend_strength": market_condition.trend_strength,
                    "market_regime": market_condition.market_regime,
                    "timestamp": market_condition.timestamp.isoformat(),
                },
                "predictions": {
                    param.value: {
                        "predicted_performance": pred.predicted_performance,
                        "optimal_value": pred.optimal_value,
                        "confidence": pred.confidence,
                        "feature_importance": pred.feature_importance,
                    }
                    for param, pred in predictions.items()
                },
            }

        except Exception as e:
            logger.error(f"Failed to get performance predictions: {e}")
            return {}

    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """適応統計を取得"""
        try:
            stats = {
                "total_adaptations": len(self.adaptation_history),
                "market_conditions_count": len(self.market_conditions),
                "strategy_performance": {},
                "parameter_adaptation_count": {},
            }

            # 戦略パフォーマンス
            for strategy, performances in self.strategy_performance.items():
                if performances:
                    stats["strategy_performance"][strategy.value] = {
                        "average_performance": float(np.mean(performances)),
                        "total_adaptations": len(performances),
                        "best_performance": float(np.max(performances)),
                        "worst_performance": float(np.min(performances)),
                    }

            # パラメータ適応数
            param_counts = {}
            for _, adaptation_result in self.adaptation_history:
                for adaptation in adaptation_result.adaptations:
                    param_type = adaptation.parameter_type.value
                    param_counts[param_type] = param_counts.get(param_type, 0) + 1

            stats["parameter_adaptation_count"] = param_counts

            return stats

        except Exception as e:
            logger.error(f"Failed to get adaptation statistics: {e}")
            return {}
