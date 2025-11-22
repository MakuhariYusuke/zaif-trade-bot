"""
Adaptive Feature Selection Manager
適応型特徴量選択マネージャー
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

from .monitoring.evaluation_manager import ContinuousEvaluationManager
from .online_learning.pipeline import OnlineLearningPipeline

logger = logging.getLogger(__name__)


class FeatureSelectionMethod(Enum):
    """特徴量選択手法"""

    CORRELATION_BASED = "correlation_based"  # 相関ベース
    IMPORTANCE_BASED = "importance_based"  # 重要度ベース
    MUTUAL_INFO = "mutual_info"  # 相互情報量
    VARIANCE_BASED = "variance_based"  # 分散ベース
    STABILITY_BASED = "stability_based"  # 安定性ベース
    MARKET_CONDITION_BASED = "market_condition_based"  # 市場条件ベース


class MarketCondition(Enum):
    """市場条件"""

    TRENDING = "trending"  # トレンド相場
    RANGING = "ranging"  # レンジ相場
    VOLATILE = "volatile"  # 高ボラティリティ
    CALM = "calm"  # 低ボラティリティ
    NEWS_DRIVEN = "news_driven"  # ニュース主導
    TECHNICAL = "technical"  # テクニカル主導


@dataclass
class AdaptiveFeatureConfig:
    """適応型特徴量選択設定"""

    # 選択手法設定
    enabled_methods: List[FeatureSelectionMethod] = field(
        default_factory=lambda: [
            FeatureSelectionMethod.IMPORTANCE_BASED,
            FeatureSelectionMethod.CORRELATION_BASED,
            FeatureSelectionMethod.MARKET_CONDITION_BASED,
        ]
    )

    # 特徴量数設定
    min_features: int = 20  # 最小特徴量数
    max_features: int = 100  # 最大特徴量数
    target_features: int = 60  # 目標特徴量数

    # 適応間隔設定
    adaptation_interval_minutes: int = 60  # 適応間隔
    min_adaptation_interval_minutes: int = 15  # 最小適応間隔

    # 重要度計算設定
    importance_calculation_window: int = 1000  # 重要度計算ウィンドウ
    importance_update_threshold: float = 0.1  # 重要度更新閾値

    # 相関設定
    max_correlation_threshold: float = 0.85  # 最大相関係数閾値
    correlation_calculation_window: int = 500  # 相関計算ウィンドウ

    # 市場条件適応設定
    market_condition_window_minutes: int = 60  # 市場条件評価ウィンドウ
    volatility_threshold_high: float = 0.02  # 高ボラティリティ閾値
    volatility_threshold_low: float = 0.005  # 低ボラティリティ閾値
    trend_strength_threshold: float = 0.7  # トレンド強度閾値

    # 安定性設定
    stability_window_days: int = 7  # 安定性評価期間
    stability_threshold: float = 0.8  # 安定性閾値

    # 重み付け設定
    feature_weights: Dict[str, float] = field(default_factory=dict)  # 特徴量別重み
    method_weights: Dict[FeatureSelectionMethod, float] = field(
        default_factory=lambda: {
            FeatureSelectionMethod.IMPORTANCE_BASED: 0.4,
            FeatureSelectionMethod.CORRELATION_BASED: 0.3,
            FeatureSelectionMethod.MARKET_CONDITION_BASED: 0.3,
        }
    )


@dataclass
class FeatureImportance:
    """特徴量重要度"""

    feature_name: str
    importance_score: float
    method: FeatureSelectionMethod
    timestamp: datetime
    market_condition: Optional[MarketCondition] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureSelectionResult:
    """特徴量選択結果"""

    selected_features: List[str]
    feature_weights: Dict[str, float]
    selection_method: FeatureSelectionMethod
    market_condition: MarketCondition
    timestamp: datetime
    performance_score: float = 0.0
    stability_score: float = 1.0
    reason: str = ""


class AdaptiveFeatureSelector:
    """適応型特徴量選択マネージャー"""

    def __init__(
        self,
        online_learning_pipeline: OnlineLearningPipeline,
        evaluation_manager: ContinuousEvaluationManager,
        config: Optional[AdaptiveFeatureConfig] = None,
    ):
        self.online_learning = online_learning_pipeline
        self.evaluation_manager = evaluation_manager
        self.config = config or AdaptiveFeatureConfig()

        # 特徴量管理
        self.all_features: List[str] = []
        self.selected_features: List[str] = []
        self.feature_weights: Dict[str, float] = {}
        self.feature_importance_history: Dict[str, List[FeatureImportance]] = {}

        # 市場条件追跡
        self.current_market_condition: MarketCondition = MarketCondition.CALM
        self.market_condition_history: List[Tuple[datetime, MarketCondition]] = []

        # 選択結果履歴
        self.selection_history: List[FeatureSelectionResult] = []

        # モデルとスケーラー
        self.importance_models: Dict[FeatureSelectionMethod, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}

        # 状態管理
        self.is_active = False
        self.last_adaptation_time: Optional[datetime] = None

        # コールバック
        self.feature_selection_callbacks: List[
            Callable[[FeatureSelectionResult], None]
        ] = []

        # スレッド管理
        self.adaptation_thread: Optional[threading.Thread] = None

        # 初期化
        self._initialize_selector()

        logger.info("AdaptiveFeatureSelector initialized")

    def _initialize_selector(self) -> None:
        """セレクターを初期化"""
        try:
            # 利用可能な特徴量を取得
            self.all_features = self._get_available_features()

            # 初期特徴量選択
            initial_selection = self._perform_initial_selection()
            self.selected_features = initial_selection.selected_features
            self.feature_weights = initial_selection.feature_weights

            # 重要度モデルを初期化
            self._initialize_importance_models()

            logger.info(
                f"Initialized with {len(self.all_features)} features, selected {len(self.selected_features)}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize selector: {e}")

    def start_adaptive_selection(self) -> bool:
        """適応型選択を開始"""
        try:
            if self.is_active:
                logger.warning("Adaptive selection already active")
                return True

            self.is_active = True
            self.adaptation_thread = threading.Thread(
                target=self._adaptation_worker, daemon=True
            )
            self.adaptation_thread.start()

            logger.info("Adaptive feature selection started")
            return True

        except Exception as e:
            logger.error(f"Failed to start adaptive selection: {e}")
            return False

    def stop_adaptive_selection(self) -> None:
        """適応型選択を停止"""
        self.is_active = False
        if self.adaptation_thread and self.adaptation_thread.is_alive():
            self.adaptation_thread.join(timeout=5.0)
        logger.info("Adaptive feature selection stopped")

    def adapt_features(
        self, market_data: pd.DataFrame, target: pd.Series
    ) -> FeatureSelectionResult:
        """特徴量を適応"""
        try:
            current_time = datetime.now()

            # 市場条件を評価
            market_condition = self._evaluate_market_condition(market_data)

            # 各手法で特徴量を選択
            selection_results = {}
            for method in self.config.enabled_methods:
                try:
                    result = self._select_features_with_method(
                        method, market_data, target, market_condition
                    )
                    selection_results[method] = result
                except Exception as e:
                    logger.error(f"Error in {method.value} selection: {e}")

            # 結果を統合
            final_selection = self._combine_selections(
                selection_results, market_condition, current_time
            )

            # 選択結果を更新
            self.selected_features = final_selection.selected_features
            self.feature_weights = final_selection.feature_weights
            self.current_market_condition = market_condition

            # 履歴に追加
            self.selection_history.append(final_selection)
            if len(self.selection_history) > 100:
                self.selection_history = self.selection_history[-100:]

            # コールバックを実行
            self._trigger_selection_callbacks(final_selection)

            logger.info(
                f"Adapted features: selected {len(final_selection.selected_features)} "
                f"features for {market_condition.value}"
            )
            return final_selection

        except Exception as e:
            logger.error(f"Failed to adapt features: {e}")
            # フォールバックとして現在の選択を返す
            return FeatureSelectionResult(
                selected_features=self.selected_features,
                feature_weights=self.feature_weights,
                selection_method=FeatureSelectionMethod.IMPORTANCE_BASED,
                market_condition=self.current_market_condition,
                timestamp=datetime.now(),
                reason="Fallback due to adaptation error",
            )

    def _evaluate_market_condition(self, market_data: pd.DataFrame) -> MarketCondition:
        """市場条件を評価"""
        try:
            if len(market_data) < 10:
                return MarketCondition.CALM

            # 価格データを取得
            if "close" in market_data.columns:
                prices = market_data["close"].values
            else:
                # 最初の数値列を使用
                numeric_cols = market_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    return MarketCondition.CALM
                prices = market_data[numeric_cols[0]].values

            # ボラティリティを計算
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns)

            # トレンド強度を計算（ADXのようなもの）
            if len(prices) > 14:
                # 簡易的なトレンド強度計算
                sma_short = pd.Series(prices).rolling(5).mean()
                sma_long = pd.Series(prices).rolling(20).mean()
                trend_strength = (
                    abs(sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
                )
            else:
                trend_strength = 0.0

            # 市場条件を決定
            if volatility > self.config.volatility_threshold_high:
                if trend_strength > self.config.trend_strength_threshold:
                    return MarketCondition.VOLATILE
                else:
                    return MarketCondition.NEWS_DRIVEN
            elif volatility < self.config.volatility_threshold_low:
                return MarketCondition.CALM
            else:
                if trend_strength > self.config.trend_strength_threshold:
                    return MarketCondition.TRENDING
                else:
                    return MarketCondition.RANGING

        except Exception as e:
            logger.error(f"Failed to evaluate market condition: {e}")
            return MarketCondition.CALM

    def _select_features_with_method(
        self,
        method: FeatureSelectionMethod,
        data: pd.DataFrame,
        target: pd.Series,
        market_condition: MarketCondition,
    ) -> FeatureSelectionResult:
        """指定された手法で特徴量を選択"""
        try:
            if method == FeatureSelectionMethod.IMPORTANCE_BASED:
                return self._importance_based_selection(data, target, market_condition)
            elif method == FeatureSelectionMethod.CORRELATION_BASED:
                return self._correlation_based_selection(data, target, market_condition)
            elif method == FeatureSelectionMethod.MUTUAL_INFO:
                return self._mutual_info_selection(data, target, market_condition)
            elif method == FeatureSelectionMethod.MARKET_CONDITION_BASED:
                return self._market_condition_based_selection(
                    data, target, market_condition
                )
            else:
                return self._default_selection(data, target, market_condition)

        except Exception as e:
            logger.error(f"Error in {method.value} selection: {e}")
            return self._default_selection(data, target, market_condition)

    def _importance_based_selection(
        self, data: pd.DataFrame, target: pd.Series, market_condition: MarketCondition
    ) -> FeatureSelectionResult:
        """重要度ベースの特徴量選択"""
        try:
            # モデルベースの特徴量重要度を計算
            model = self.importance_models.get(FeatureSelectionMethod.IMPORTANCE_BASED)
            if model is None:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                self.importance_models[FeatureSelectionMethod.IMPORTANCE_BASED] = model

            # データを準備
            X = data.values
            y = target.values

            # スケーリング
            if "importance_scaler" not in self.scalers:
                self.scalers["importance_scaler"] = StandardScaler()
                X_scaled = self.scalers["importance_scaler"].fit_transform(X)
            else:
                X_scaled = self.scalers["importance_scaler"].transform(X)

            # モデルを訓練
            model.fit(X_scaled, y)

            # 特徴量重要度を取得
            importances = model.feature_importances_
            feature_names = list(data.columns)

            # 重要度でソート
            feature_importance_pairs = list(zip(feature_names, importances))
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

            # 上位特徴量を選択
            num_features = min(
                self.config.target_features, len(feature_importance_pairs)
            )
            selected_pairs = feature_importance_pairs[:num_features]

            selected_features = [name for name, _ in selected_pairs]
            feature_weights = {name: float(imp) for name, imp in selected_pairs}

            return FeatureSelectionResult(
                selected_features=selected_features,
                feature_weights=feature_weights,
                selection_method=FeatureSelectionMethod.IMPORTANCE_BASED,
                market_condition=market_condition,
                timestamp=datetime.now(),
                reason="Importance-based selection using Random Forest",
            )

        except Exception as e:
            logger.error(f"Importance-based selection failed: {e}")
            return self._default_selection(data, target, market_condition)

    def _correlation_based_selection(
        self, data: pd.DataFrame, target: pd.Series, market_condition: MarketCondition
    ) -> FeatureSelectionResult:
        """相関ベースの特徴量選択"""
        try:
            # ターゲットとの相関係数を計算
            correlations = {}
            for col in data.columns:
                if data[col].dtype in ["int64", "float64"]:
                    try:
                        corr = abs(data[col].corr(target))
                        if not np.isnan(corr):
                            correlations[col] = corr
                    except Exception:
                        continue

            # 相関の高い特徴量を選択
            sorted_features = sorted(
                correlations.items(), key=lambda x: x[1], reverse=True
            )

            # 相関が閾値以下の特徴量を除外
            filtered_features = [
                (name, corr) for name, corr in sorted_features if corr >= 0.1
            ]

            # 特徴量間の相関をチェック（多重共線性を避ける）
            selected_features = []
            selected_weights = {}

            for name, corr in filtered_features:
                # 既に選択された特徴量との相関をチェック
                should_include = True
                for selected_name in selected_features:
                    feature_corr = abs(data[name].corr(data[selected_name]))
                    if feature_corr > self.config.max_correlation_threshold:
                        should_include = False
                        break

                if should_include:
                    selected_features.append(name)
                    selected_weights[name] = float(corr)

                if len(selected_features) >= self.config.target_features:
                    break

            return FeatureSelectionResult(
                selected_features=selected_features,
                feature_weights=selected_weights,
                selection_method=FeatureSelectionMethod.CORRELATION_BASED,
                market_condition=market_condition,
                timestamp=datetime.now(),
                reason="Correlation-based selection with multicollinearity check",
            )

        except Exception as e:
            logger.error(f"Correlation-based selection failed: {e}")
            return self._default_selection(data, target, market_condition)

    def _mutual_info_selection(
        self, data: pd.DataFrame, target: pd.Series, market_condition: MarketCondition
    ) -> FeatureSelectionResult:
        """相互情報量ベースの特徴量選択"""
        try:
            # 数値特徴量のみを使用
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                return self._default_selection(data, target, market_condition)

            X = numeric_data.values
            y = target.values

            # 相互情報量を計算
            mi_scores = mutual_info_regression(X, y, random_state=42)

            # 特徴量とスコアをペアリング
            feature_mi_pairs = list(zip(numeric_data.columns, mi_scores))
            feature_mi_pairs.sort(key=lambda x: x[1], reverse=True)

            # 上位特徴量を選択
            num_features = min(self.config.target_features, len(feature_mi_pairs))
            selected_pairs = feature_mi_pairs[:num_features]

            selected_features = [name for name, _ in selected_pairs]
            feature_weights = {name: float(score) for name, score in selected_pairs}

            return FeatureSelectionResult(
                selected_features=selected_features,
                feature_weights=feature_weights,
                selection_method=FeatureSelectionMethod.MUTUAL_INFO,
                market_condition=market_condition,
                timestamp=datetime.now(),
                reason="Mutual information-based selection",
            )

        except Exception as e:
            logger.error(f"Mutual info selection failed: {e}")
            return self._default_selection(data, target, market_condition)

    def _market_condition_based_selection(
        self, data: pd.DataFrame, target: pd.Series, market_condition: MarketCondition
    ) -> FeatureSelectionResult:
        """市場条件ベースの特徴量選択"""
        try:
            # 市場条件に応じた特徴量重み付け
            condition_weights = self._get_market_condition_weights(market_condition)

            # 各特徴量の適合性を評価
            feature_scores = {}
            for feature_name in data.columns:
                base_score = 1.0

                # 特徴量タイプに応じた重み付け
                if any(
                    keyword in feature_name.lower()
                    for keyword in ["trend", "adx", "slope"]
                ):
                    # トレンド系指標
                    base_score *= condition_weights.get("trend", 1.0)
                elif any(
                    keyword in feature_name.lower()
                    for keyword in ["rsi", "stoch", "williams"]
                ):
                    # オシレーター系指標
                    base_score *= condition_weights.get("oscillator", 1.0)
                elif any(
                    keyword in feature_name.lower()
                    for keyword in ["bb", "band", "atr", "volatility"]
                ):
                    # ボラティリティ系指標
                    base_score *= condition_weights.get("volatility", 1.0)
                elif any(
                    keyword in feature_name.lower() for keyword in ["volume", "flow"]
                ):
                    # 出来高系指標
                    base_score *= condition_weights.get("volume", 1.0)

                feature_scores[feature_name] = base_score

            # スコアでソート
            sorted_features = sorted(
                feature_scores.items(), key=lambda x: x[1], reverse=True
            )

            # 上位特徴量を選択
            num_features = min(self.config.target_features, len(sorted_features))
            selected_pairs = sorted_features[:num_features]

            selected_features = [name for name, _ in selected_pairs]
            feature_weights = {name: float(score) for name, score in selected_pairs}

            return FeatureSelectionResult(
                selected_features=selected_features,
                feature_weights=feature_weights,
                selection_method=FeatureSelectionMethod.MARKET_CONDITION_BASED,
                market_condition=market_condition,
                timestamp=datetime.now(),
                reason=f"Market condition-based selection for {market_condition.value}",
            )

        except Exception as e:
            logger.error(f"Market condition-based selection failed: {e}")
            return self._default_selection(data, target, market_condition)

    def _get_market_condition_weights(
        self, market_condition: MarketCondition
    ) -> Dict[str, float]:
        """市場条件に応じた重みを取得"""
        try:
            if market_condition == MarketCondition.TRENDING:
                return {
                    "trend": 1.5,
                    "oscillator": 0.8,
                    "volatility": 1.0,
                    "volume": 1.2,
                }  # トレンド指標を重視  # オシレーターを軽視
            elif market_condition == MarketCondition.RANGING:
                return {
                    "trend": 0.7,
                    "oscillator": 1.4,
                    "volatility": 0.9,
                    "volume": 1.0,
                }  # トレンド指標を軽視  # オシレーターを重視
            elif market_condition == MarketCondition.VOLATILE:
                return {
                    "trend": 1.0,
                    "oscillator": 1.0,
                    "volatility": 1.6,
                    "volume": 1.3,
                }  # ボラティリティ指標を重視
            elif market_condition == MarketCondition.CALM:
                return {
                    "trend": 1.1,
                    "oscillator": 1.1,
                    "volatility": 0.6,
                    "volume": 0.8,
                }  # ボラティリティ指標を軽視
            else:  # NEWS_DRIVEN or TECHNICAL
                return {
                    "trend": 1.0,
                    "oscillator": 1.0,
                    "volatility": 1.2,
                    "volume": 1.4,
                }  # 出来高を重視

        except Exception:
            return {"trend": 1.0, "oscillator": 1.0, "volatility": 1.0, "volume": 1.0}

    def _combine_selections(
        self,
        selection_results: Dict[FeatureSelectionMethod, FeatureSelectionResult],
        market_condition: MarketCondition,
        timestamp: datetime,
    ) -> FeatureSelectionResult:
        """複数の選択結果を統合"""
        try:
            if not selection_results:
                return self._default_selection(
                    pd.DataFrame(), pd.Series(), market_condition
                )

            # 全特徴量のスコアを集計
            all_features = set()
            feature_total_scores = {}
            method_contributions = {}

            for method, result in selection_results.items():
                method_weight = self.config.method_weights.get(method, 1.0)
                method_contributions[method] = method_weight

                for feature_name, weight in result.feature_weights.items():
                    all_features.add(feature_name)
                    if feature_name not in feature_total_scores:
                        feature_total_scores[feature_name] = 0.0
                    feature_total_scores[feature_name] += weight * method_weight

            # スコアでソート
            sorted_features = sorted(
                feature_total_scores.items(), key=lambda x: x[1], reverse=True
            )

            # 上位特徴量を選択
            num_features = min(self.config.target_features, len(sorted_features))
            selected_pairs = sorted_features[:num_features]

            selected_features = [name for name, _ in selected_pairs]
            feature_weights = {name: float(score) for name, score in selected_pairs}

            # 正規化された重みを計算
            total_weight = sum(feature_weights.values())
            if total_weight > 0:
                feature_weights = {
                    name: weight / total_weight
                    for name, weight in feature_weights.items()
                }

            return FeatureSelectionResult(
                selected_features=selected_features,
                feature_weights=feature_weights,
                selection_method=FeatureSelectionMethod.IMPORTANCE_BASED,  # 統合結果
                market_condition=market_condition,
                timestamp=timestamp,
                reason=f"Combined selection from {len(selection_results)} methods",
            )

        except Exception as e:
            logger.error(f"Failed to combine selections: {e}")
            return self._default_selection(
                pd.DataFrame(), pd.Series(), market_condition
            )

    def _default_selection(
        self, data: pd.DataFrame, target: pd.Series, market_condition: MarketCondition
    ) -> FeatureSelectionResult:
        """デフォルトの特徴量選択"""
        try:
            # 利用可能な特徴量からランダムに選択
            available_features = (
                list(data.columns)
                if not data.empty
                else self.all_features[: self.config.target_features]
            )

            num_features = min(self.config.target_features, len(available_features))
            selected_features = available_features[:num_features]

            # 等しい重み付け
            feature_weights = {
                name: 1.0 / len(selected_features) for name in selected_features
            }

            return FeatureSelectionResult(
                selected_features=selected_features,
                feature_weights=feature_weights,
                selection_method=FeatureSelectionMethod.IMPORTANCE_BASED,
                market_condition=market_condition,
                timestamp=datetime.now(),
                reason="Default selection (fallback)",
            )

        except Exception as e:
            logger.error(f"Default selection failed: {e}")
            return FeatureSelectionResult(
                selected_features=self.selected_features[:10]
                if self.selected_features
                else [],
                feature_weights={
                    name: 1.0
                    for name in (
                        self.selected_features[:10] if self.selected_features else []
                    )
                },
                selection_method=FeatureSelectionMethod.IMPORTANCE_BASED,
                market_condition=market_condition,
                timestamp=datetime.now(),
                reason="Emergency fallback selection",
            )

    def _perform_initial_selection(self) -> FeatureSelectionResult:
        """初期特徴量選択を実行"""
        try:
            # ダミーデータで初期選択
            dummy_data = pd.DataFrame(
                {
                    f"feature_{i}": np.random.randn(100)
                    for i in range(min(50, len(self.all_features)))
                }
            )
            dummy_target = pd.Series(np.random.randn(100))

            return self._default_selection(
                dummy_data, dummy_target, MarketCondition.CALM
            )

        except Exception as e:
            logger.error(f"Initial selection failed: {e}")
            return FeatureSelectionResult(
                selected_features=self.all_features[: self.config.target_features]
                if self.all_features
                else [],
                feature_weights={
                    name: 1.0
                    for name in (
                        self.all_features[: self.config.target_features]
                        if self.all_features
                        else []
                    )
                },
                selection_method=FeatureSelectionMethod.IMPORTANCE_BASED,
                market_condition=MarketCondition.CALM,
                timestamp=datetime.now(),
                reason="Initial selection",
            )

    def _initialize_importance_models(self) -> None:
        """重要度モデルを初期化"""
        try:
            for method in self.config.enabled_methods:
                if method == FeatureSelectionMethod.IMPORTANCE_BASED:
                    self.importance_models[method] = RandomForestRegressor(
                        n_estimators=100, random_state=42, n_jobs=-1
                    )
                elif method == FeatureSelectionMethod.MUTUAL_INFO:
                    # 相互情報量はモデル不要
                    pass

            logger.info(f"Initialized {len(self.importance_models)} importance models")

        except Exception as e:
            logger.error(f"Failed to initialize importance models: {e}")

    def _get_available_features(self) -> List[str]:
        """利用可能な特徴量を取得"""
        try:
            # オンライン学習パイプラインから特徴量を取得
            # （実際の実装では適切なメソッドを呼び出す）
            return [f"feature_{i}" for i in range(156)]  # ダミー特徴量

        except Exception as e:
            logger.error(f"Failed to get available features: {e}")
            return []

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
                    # 最新データを取得
                    market_data, target = self._get_latest_data()

                    if market_data is not None and target is not None:
                        # 特徴量適応を実行
                        self.adapt_features(market_data, target)
                        self.last_adaptation_time = current_time

                time.sleep(60)  # 1分ごとにチェック

            except Exception as e:
                logger.error(f"Adaptation worker error: {e}")
                time.sleep(300)  # エラー時は5分待機

    def _get_latest_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """最新データを取得"""
        try:
            # オンライン学習パイプラインからデータを取得
            # （実際の実装では適切なメソッドを呼び出す）
            return None, None  # ダミー

        except Exception as e:
            logger.error(f"Failed to get latest data: {e}")
            return None, None

    def add_selection_callback(
        self, callback: Callable[[FeatureSelectionResult], None]
    ) -> None:
        """特徴量選択コールバックを追加"""
        self.feature_selection_callbacks.append(callback)

    def _trigger_selection_callbacks(self, result: FeatureSelectionResult) -> None:
        """特徴量選択コールバックを実行"""
        for callback in self.feature_selection_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Feature selection callback failed: {e}")

    def get_selection_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """選択履歴を取得"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_selections = [
                s for s in self.selection_history if s.timestamp > cutoff_time
            ]

            return [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "method": s.selection_method.value,
                    "market_condition": s.market_condition.value,
                    "num_features": len(s.selected_features),
                    "performance_score": s.performance_score,
                    "stability_score": s.stability_score,
                    "reason": s.reason,
                }
                for s in recent_selections
            ]

        except Exception as e:
            logger.error(f"Failed to get selection history: {e}")
            return []

    def get_feature_importance_stats(
        self, feature_name: str, hours: int = 24
    ) -> Dict[str, Any]:
        """特徴量重要度統計を取得"""
        try:
            if feature_name not in self.feature_importance_history:
                return {"message": f"No history for feature {feature_name}"}

            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_importance = [
                imp
                for imp in self.feature_importance_history[feature_name]
                if imp.timestamp > cutoff_time
            ]

            if not recent_importance:
                return {"message": "No recent importance data"}

            scores = [imp.importance_score for imp in recent_importance]

            return {
                "feature_name": feature_name,
                "period_hours": hours,
                "num_measurements": len(scores),
                "mean_importance": float(np.mean(scores)),
                "std_importance": float(np.std(scores)),
                "min_importance": float(np.min(scores)),
                "max_importance": float(np.max(scores)),
                "latest_importance": scores[-1] if scores else 0.0,
            }

        except Exception as e:
            logger.error(f"Failed to get feature importance stats: {e}")
            return {"error": str(e)}
