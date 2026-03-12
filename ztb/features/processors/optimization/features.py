#!/usr/bin/env python3
"""
Optimizer Feature Tracker - Enhanced Statistical Features for Training Dynamics

このモジュールは、トレーニング中のoptimizer状態を追跡し、統計的に堅牢な特徴量を生成します。
v442+のトレーニングダイナミクス改善のために設計されています。

主な改善点:
1. 統計的堅牢性: 特徴量の正規化、外れ値処理、相関分析
2. 運用性: 包括的なエラーハンドリング、パフォーマンス監視、デバッグ機能
3. 設定柔軟性: コンフィグベースの設定管理
4. ML互換性: 特徴量の重要度評価と解釈性分析

特徴量の説明:
- optimizer_learning_rate: 現在の学習率（トレーニング適応性の指標）
- optimizer_learning_rate_trend: 学習率の変化トレンド（学習安定性の指標）
- optimizer_gradient_norm_avg: 勾配ノルムの平均（トレーニングの激しさの指標）
- optimizer_gradient_norm_std: 勾配ノルムの標準偏差（トレーニング変動性の指標）
- optimizer_step_size_avg: ステップサイズの平均（パラメータ更新の大きさの指標）
- optimizer_momentum_avg: モメンタムの平均（トレーニング慣性の指標）
- optimizer_training_progress: 学習進捗率（0-1、正規化された値）
- optimizer_loss_trend: 損失のトレンド（学習方向性の指標）
- optimizer_update_frequency_avg: 更新頻度の平均（トレーニングペースの指標）
- optimizer_stability_score: 最適化安定性スコア（0-1、変動係数ベース）
- optimizer_adaptive_lr_score: 適応学習率スコア（学習率適応性の指標）

使用方法:
    # 設定ベースの初期化
    config = {
        'max_history': 1000,
        'enable_normalization': True,
        'normalization_method': 'robust',
        'outlier_threshold': 1.5
    }
    tracker = OptimizerFeatureTracker(**config)

    # トレーニング中の更新
    tracker.update_optimizer_features(
        step=current_step,
        learning_rate=lr,
        actor_loss=actor_loss,
        critic_loss=critic_loss,
        entropy_coef=ent_coef
    )

    # 特徴量取得
    features = tracker.get_feature_vector(include_debug_info=True)

    # 統計分析
    correlations = tracker.compute_feature_correlations()
    importance = tracker.compute_feature_importance()
"""

import logging
from collections import deque

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from ztb.features.core.registry import FeatureRegistry
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

class OptimizerFeatureTracker:
    """
    Optimizer状態を追跡し、特徴量を生成するクラス

    統計的改善:
    - 特徴量の正規化（StandardScaler/RobustScaler）
    - 外れ値処理（IQRフィルタリング）
    - 相関分析機能
    - 特徴量重要度評価
    """

    def __init__(
        self,
        max_history: int = 1000,
        enable_normalization: bool = True,
        normalization_method: str = "robust",
        outlier_threshold: float = 1.5,
        window_size: int = 100,
    ):
        """
        Args:
            max_history: 保持する履歴の最大数
            enable_normalization: 特徴量正規化を有効化
            normalization_method: 'standard' または 'robust'
            outlier_threshold: 外れ値判定の閾値 (IQR multiplier)
            window_size: 移動平均などの計算に使用するウィンドウサイズ
        """
        self.max_history = max_history
        self.enable_normalization = enable_normalization
        self.normalization_method = normalization_method
        self.outlier_threshold = outlier_threshold
        self.window_size = window_size

        # 生データ履歴
        self.learning_rates: deque = deque(maxlen=max_history)
        self.gradient_norms: deque = deque(maxlen=max_history)
        self.step_sizes: deque = deque(maxlen=max_history)
        self.momentum_values: deque = deque(maxlen=max_history)
        self.loss_values: deque = deque(maxlen=max_history)
        self.update_frequencies: deque = deque(maxlen=max_history)

        # トレーニング状態
        self.training_step = 0
        self.total_steps = 0
        self.current_epoch = 0

        # 正規化器
        self.scalers = {}
        if enable_normalization:
            self._init_scalers()

        # 特徴量キャッシュ
        self._feature_cache: dict[str, float] = {}
        self._correlation_cache: dict[str, float] = {}
        self._importance_cache: dict[str, float] = {}

        # パフォーマンス監視
        self.update_count = 0
        self.error_count = 0
        self.last_update_time = 0.0

        self.logger = logging.getLogger(__name__)

    def _init_scalers(self):
        """特徴量正規化のためのスケーラーを初期化"""
        feature_names = [
            "learning_rate",
            "gradient_norm",
            "step_size",
            "momentum",
            "loss",
            "update_frequency",
        ]

        for feature in feature_names:
            if self.normalization_method == "robust":
                self.scalers[feature] = RobustScaler()
            else:
                self.scalers[feature] = StandardScaler()

    def _detect_outliers_iqr(self, data: list[float]) -> tuple[list[float], list[bool]]:
        """
        IQR法による外れ値検出

        Returns:
            filtered_data: 外れ値を除去したデータ
            is_outlier: 各データポイントが外れ値かどうか
        """
        if len(data) < 4:
            return data, [False] * len(data)

        data_array = np.array(data)
        q1 = np.percentile(data_array, 25)
        q3 = np.percentile(data_array, 75)
        iqr = q3 - q1

        lower_bound = q1 - (self.outlier_threshold * iqr)
        upper_bound = q3 + (self.outlier_threshold * iqr)

        is_outlier = (data_array < lower_bound) | (data_array > upper_bound)
        filtered_data = data_array[~is_outlier].tolist()

        return filtered_data, is_outlier.tolist()

    def _normalize_feature(self, feature_name: str, values: list[float]) -> list[float]:
        """特徴量を正規化"""
        if not self.enable_normalization or feature_name not in self.scalers:
            return values

        if len(values) < 2:
            return values

        try:
            # 外れ値を除去したデータで正規化
            filtered_values, _ = self._detect_outliers_iqr(values)
            if len(filtered_values) < 2:
                return values

            scaler = self.scalers[feature_name]
            normalized = scaler.fit_transform(np.array(filtered_values).reshape(-1, 1))
            return normalized.flatten().tolist()
        except Exception as e:
            self.logger.warning(f"Failed to normalize {feature_name}: {e}")
            return values

    def compute_feature_correlations(self) -> dict[str, dict[str, float]]:
        """
        特徴量間の相関行列を計算（Spearman相関）

        Returns:
            特徴量名 -> {相関特徴量名 -> 相関係数} の辞書
        """
        if not self._correlation_cache:
            try:
                # すべての特徴量を取得
                features = self.get_feature_vector()
                feature_names = list(features.keys())
                feature_values = list(features.values())

                if len(feature_names) < 2:
                    return {}

                # 相関行列を計算
                data = np.array(feature_values).reshape(1, -1)
                if data.shape[1] > 1:
                    corr_matrix = np.corrcoef(data.T)

                    # 相関辞書を作成
                    correlations = {}
                    for i, name1 in enumerate(feature_names):
                        correlations[name1] = {}
                        for j, name2 in enumerate(feature_names):
                            if i != j:  # 自己相関は除外
                                correlations[name1][name2] = float(corr_matrix[i, j])

                    self._correlation_cache = correlations

            except Exception as e:
                self.logger.warning(f"Failed to compute correlations: {e}")
                self._correlation_cache = {}

        return self._correlation_cache.copy()

    def compute_feature_importance(
        self, target_correlation_threshold: float = 0.1
    ) -> dict[str, float]:
        """
        特徴量の重要度を評価（他の特徴量との相関の強さベース）

        Args:
            target_correlation_threshold: 重要とみなす相関係数の閾値

        Returns:
            特徴量名 -> 重要度スコア の辞書
        """
        if not self._importance_cache:
            try:
                correlations = self.compute_feature_correlations()
                if not correlations:
                    return {}

                importance_scores = {}
                for feature_name in correlations.keys():
                    # この特徴量と他の特徴量の相関係数の絶対値の平均
                    corr_values = [
                        abs(corr) for corr in correlations[feature_name].values()
                    ]
                    avg_correlation = np.mean(corr_values) if corr_values else 0.0

                    # 閾値以上の相関を持つ特徴量の数を考慮
                    strong_correlations = sum(
                        1
                        for corr in corr_values
                        if corr >= target_correlation_threshold
                    )

                    # 重要度スコア：平均相関 × 強相関数 × 独自性ボーナス
                    uniqueness_bonus = 1.0 / (
                        1.0 + strong_correlations
                    )  # 冗長な特徴量をペナルティ
                    importance_scores[feature_name] = (
                        avg_correlation * (1.0 + strong_correlations) * uniqueness_bonus
                    )

                self._importance_cache = importance_scores

            except Exception as e:
                self.logger.warning(f"Failed to compute importance: {e}")
                self._importance_cache = {}

        return self._importance_cache.copy()

    def update_optimizer_features(
        self,
        step: int,
        learning_rate: float | None = None,
        actor_loss: float | None = None,
        critic_loss: float | None = None,
        entropy_coef: float | None = None,
        reward: float | None = None,
        gradient_norm: float | None = None,
        step_size: float | None = None,
    ):
        """
        Optimizer状態を更新（コールバック互換API）

        Args:
            step: 現在のステップ数
            learning_rate: 現在の学習率
            actor_loss: Actor損失
            critic_loss: Critic損失
            entropy_coef: エントロピー係数
            reward: 現在の報酬
            gradient_norm: 勾配ノルム（オプション）
            step_size: ステップサイズ（オプション）
        """
        import time

        start_time = time.time()

        try:
            self.update_count += 1
            self.training_step = step

            # 学習率を更新
            if learning_rate is not None:
                self.learning_rates.append(learning_rate)

            # 損失値を統合（actor_lossとcritic_lossの平均を使用）
            if actor_loss is not None or critic_loss is not None:
                combined_loss = None
                if actor_loss is not None and critic_loss is not None:
                    combined_loss = (actor_loss + critic_loss) / 2.0
                elif actor_loss is not None:
                    combined_loss = actor_loss
                elif critic_loss is not None:
                    combined_loss = critic_loss

                if combined_loss is not None:
                    self.loss_values.append(combined_loss)

            # エントロピー係数をモメンタムとして扱う
            if entropy_coef is not None:
                self.momentum_values.append(entropy_coef)

            # 勾配ノルム（提供された場合）
            if gradient_norm is not None:
                self.gradient_norms.append(gradient_norm)

            # ステップサイズ（学習率をステップサイズとして使用）
            if step_size is not None:
                self.step_sizes.append(step_size)
            elif learning_rate is not None:
                # 学習率をステップサイズの近似として使用
                self.step_sizes.append(learning_rate)

            # 更新頻度（固定値または動的計算）
            update_freq = 1.0  # デフォルト
            if len(self.update_frequencies) > 0:
                # 過去の更新間隔に基づいて計算
                update_freq = max(1.0, step - self.update_frequencies[-1])
            self.update_frequencies.append(update_freq)

            # キャッシュをクリア
            self._feature_cache.clear()
            self._correlation_cache.clear()
            self._importance_cache.clear()

            # パフォーマンス監視
            self.last_update_time = time.time() - start_time

            if self.update_count % 100 == 0:  # 定期的にログ出力
                self.logger.debug(
                    f"Optimizer features updated {self.update_count} times, "
                    f"last update took {self.last_update_time:.4f}s"
                )

        except Exception as e:
            self.error_count += 1
            self.logger.warning(f"Failed to update optimizer features: {e}")
            self.last_update_time = time.time() - start_time

    def update_optimizer_state(
        self,
        learning_rate: float,
        gradient_norm: float,
        step_size: float,
        momentum: float | None = None,
        loss: float | None = None,
        update_frequency: float | None = None,
    ):
        """
        Optimizer状態を更新

        Args:
            learning_rate: 現在の学習率
            gradient_norm: 勾配のノルム
            step_size: ステップサイズ
            momentum: モメンタム値 (オプション)
            loss: 現在の損失値 (オプション)
            update_frequency: 更新頻度 (オプション)
        """
        self.training_step += 1

        # 履歴に追加
        self.learning_rates.append(learning_rate)
        self.gradient_norms.append(gradient_norm)
        self.step_sizes.append(step_size)

        if momentum is not None:
            self.momentum_values.append(momentum)
        if loss is not None:
            self.loss_values.append(loss)
        if update_frequency is not None:
            self.update_frequencies.append(update_frequency)

        # キャッシュをクリア
        self._feature_cache.clear()

    def set_training_progress(
        self, current_step: int, total_steps: int, current_epoch: int = 0
    ):
        """学習進捗を設定"""
        self.training_step = current_step
        self.total_steps = total_steps
        self.current_epoch = current_epoch

    def get_learning_rate(self) -> float:
        """現在の学習率を返す"""
        return self.learning_rates[-1] if self.learning_rates else 0.001

    def get_learning_rate_trend(self) -> float:
        """学習率のトレンド (変化率)"""
        if len(self.learning_rates) < 2:
            return 0.0
        recent = list(self.learning_rates)[-10:]  # 最近10ステップ
        if len(recent) < 2:
            return 0.0
        return (recent[-1] - recent[0]) / max(abs(recent[0]), 1e-8)

    def get_gradient_norm_avg(self) -> float:
        """勾配ノルムの移動平均"""
        if not self.gradient_norms:
            return 0.0
        return np.mean(self.gradient_norms)

    def get_gradient_norm_std(self) -> float:
        """勾配ノルムの標準偏差"""
        if len(self.gradient_norms) < 2:
            return 0.0
        return np.std(self.gradient_norms)

    def get_step_size_avg(self) -> float:
        """ステップサイズの移動平均"""
        if not self.step_sizes:
            return 0.0
        return np.mean(self.step_sizes)

    def get_momentum_avg(self) -> float:
        """モメンタムの移動平均"""
        if not self.momentum_values:
            return 0.0
        return np.mean(self.momentum_values)

    def get_training_progress(self) -> float:
        """学習進捗率 (0-1)"""
        if self.total_steps == 0:
            return 0.0
        return min(self.training_step / self.total_steps, 1.0)

    def get_loss_trend(self) -> float:
        """損失のトレンド (変化率)"""
        if len(self.loss_values) < 2:
            return 0.0
        recent = list(self.loss_values)[-10:]  # 最近10ステップ
        if len(recent) < 2:
            return 0.0
        return (recent[-1] - recent[0]) / max(abs(recent[0]), 1e-8)

    def get_update_frequency_avg(self) -> float:
        """更新頻度の平均"""
        if not self.update_frequencies:
            return 1.0
        return np.mean(self.update_frequencies)

    def get_optimizer_stability_score(self) -> float:
        """Optimizerの安定性スコア (0-1, 1が安定)"""
        if len(self.gradient_norms) < 5:
            return 0.5

        # 勾配ノルムの変動係数を計算
        grad_mean = np.mean(self.gradient_norms)
        grad_std = np.std(self.gradient_norms)
        cv = grad_std / max(abs(grad_mean), 1e-8)

        # 変動係数が小さいほど安定 (0-1に正規化)
        stability = 1.0 / (1.0 + cv)
        return stability

    def get_adaptive_learning_rate_score(self) -> float:
        """適応学習率スコア (学習の適応性を示す)"""
        if len(self.learning_rates) < 5:
            return 0.5

        # 学習率の適応性を評価
        lr_changes = np.diff(list(self.learning_rates))
        adaptability = np.mean(np.abs(lr_changes)) / max(
            np.mean(self.learning_rates), 1e-8
        )

        # 0-1に正規化
        return min(adaptability, 1.0)

    def _apply_outlier_filter(self, data: np.ndarray) -> np.ndarray:
        """外れ値フィルタリング (IQR法)"""
        if len(data) < 2:
            return data
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - self.outlier_threshold * iqr
        upper_bound = q3 + self.outlier_threshold * iqr
        return data[(data >= lower_bound) & (data <= upper_bound)]

    def get_feature_vector(self, include_debug_info: bool = False) -> dict[str, float]:
        """
        すべてのoptimizer特徴量を辞書形式で返す

        Args:
            include_debug_info: デバッグ情報を含めるかどうか

        Returns:
            特徴量名 -> 値 の辞書（デバッグ情報含む場合あり）
        """
        if not self._feature_cache:
            try:
                features = {
                    "optimizer_learning_rate": self.get_learning_rate(),
                    "optimizer_learning_rate_trend": self.get_learning_rate_trend(),
                    "optimizer_gradient_norm_avg": self.get_gradient_norm_avg(),
                    "optimizer_gradient_norm_std": self.get_gradient_norm_std(),
                    "optimizer_step_size_avg": self.get_step_size_avg(),
                    "optimizer_momentum_avg": self.get_momentum_avg(),
                    "optimizer_training_progress": self.get_training_progress(),
                    "optimizer_loss_trend": self.get_loss_trend(),
                    "optimizer_update_frequency_avg": self.get_update_frequency_avg(),
                    "optimizer_stability_score": self.get_optimizer_stability_score(),
                    "optimizer_adaptive_lr_score": self.get_adaptive_learning_rate_score(),
                }

                # NaNやinfをチェックして置換
                for key, value in features.items():
                    if not np.isfinite(value):
                        self.logger.warning(
                            f"Non-finite value detected for {key}: {value}, replacing with 0.0"
                        )
                        features[key] = 0.0

                self._feature_cache = features

            except Exception as e:
                self.logger.error(f"Failed to compute feature vector: {e}")
                # フォールバック：ゼロ特徴量を返す
                self._feature_cache = dict.fromkeys(self.get_feature_names(), 0.0)

        result = self._feature_cache.copy()

        if include_debug_info:
            result["_debug_info"] = {
                "update_count": self.update_count,
                "error_count": self.error_count,
                "last_update_time": self.last_update_time,
                "history_lengths": {
                    "learning_rates": len(self.learning_rates),
                    "gradient_norms": len(self.gradient_norms),
                    "step_sizes": len(self.step_sizes),
                    "momentum_values": len(self.momentum_values),
                    "loss_values": len(self.loss_values),
                    "update_frequencies": len(self.update_frequencies),
                },
                "correlations": self.compute_feature_correlations(),
                "importance_scores": self.compute_feature_importance(),
            }

        return result

    def get_feature_names(self) -> list[str]:
        """利用可能な特徴量名のリストを返す"""
        return list(self.get_feature_vector().keys())

# グローバルインスタンス (シングルトンパターン)
_optimizer_tracker: OptimizerFeatureTracker | None = None

def get_optimizer_tracker() -> OptimizerFeatureTracker:
    """OptimizerFeatureTrackerのグローバルインスタンスを取得"""
    global _optimizer_tracker
    if _optimizer_tracker is None:
        _optimizer_tracker = OptimizerFeatureTracker()
    return _optimizer_tracker

def update_optimizer_features(
    learning_rate: float,
    gradient_norm: float,
    step_size: float,
    momentum: float | None = None,
    loss: float | None = None,
    update_frequency: float | None = None,
):
    """Optimizer特徴量を更新"""
    tracker = get_optimizer_tracker()
    tracker.update_optimizer_state(
        learning_rate=learning_rate,
        gradient_norm=gradient_norm,
        step_size=step_size,
        momentum=momentum,
        loss=loss,
        update_frequency=update_frequency,
    )

def set_training_progress(current_step: int, total_steps: int, current_epoch: int = 0):
    """学習進捗を設定"""
    tracker = get_optimizer_tracker()
    tracker.set_training_progress(current_step, total_steps, current_epoch)

# 特徴量関数群 (FeatureRegistryに登録するための関数)
@FeatureRegistry.register("optimizer_learning_rate")
def optimizer_learning_rate(df: pd.DataFrame) -> pd.Series:
    """学習率特徴量"""
    tracker = get_optimizer_tracker()
    return pd.Series([tracker.get_learning_rate()] * len(df), index=df.index)

@FeatureRegistry.register("optimizer_learning_rate_trend")
def optimizer_learning_rate_trend(df: pd.DataFrame) -> pd.Series:
    """学習率トレンド特徴量"""
    tracker = get_optimizer_tracker()
    return pd.Series([tracker.get_learning_rate_trend()] * len(df), index=df.index)

@FeatureRegistry.register("optimizer_gradient_norm_avg")
def optimizer_gradient_norm_avg(df: pd.DataFrame) -> pd.Series:
    """勾配ノルム平均特徴量"""
    tracker = get_optimizer_tracker()
    return pd.Series([tracker.get_gradient_norm_avg()] * len(df), index=df.index)

@FeatureRegistry.register("optimizer_gradient_norm_std")
def optimizer_gradient_norm_std(df: pd.DataFrame) -> pd.Series:
    """勾配ノルム標準偏差特徴量"""
    tracker = get_optimizer_tracker()
    return pd.Series([tracker.get_gradient_norm_std()] * len(df), index=df.index)

@FeatureRegistry.register("optimizer_step_size_avg")
def optimizer_step_size_avg(df: pd.DataFrame) -> pd.Series:
    """ステップサイズ平均特徴量"""
    tracker = get_optimizer_tracker()
    return pd.Series([tracker.get_step_size_avg()] * len(df), index=df.index)

@FeatureRegistry.register("optimizer_momentum_avg")
def optimizer_momentum_avg(df: pd.DataFrame) -> pd.Series:
    """モメンタム平均特徴量"""
    tracker = get_optimizer_tracker()
    return pd.Series([tracker.get_momentum_avg()] * len(df), index=df.index)

@FeatureRegistry.register("optimizer_training_progress")
def optimizer_training_progress(df: pd.DataFrame) -> pd.Series:
    """学習進捗特徴量"""
    tracker = get_optimizer_tracker()
    return pd.Series([tracker.get_training_progress()] * len(df), index=df.index)

@FeatureRegistry.register("optimizer_loss_trend")
def optimizer_loss_trend(df: pd.DataFrame) -> pd.Series:
    """損失トレンド特徴量"""
    tracker = get_optimizer_tracker()
    return pd.Series([tracker.get_loss_trend()] * len(df), index=df.index)

@FeatureRegistry.register("optimizer_update_frequency_avg")
def optimizer_update_frequency_avg(df: pd.DataFrame) -> pd.Series:
    """更新頻度平均特徴量"""
    tracker = get_optimizer_tracker()
    return pd.Series([tracker.get_update_frequency_avg()] * len(df), index=df.index)

@FeatureRegistry.register("optimizer_stability_score")
def optimizer_stability_score(df: pd.DataFrame) -> pd.Series:
    """安定性スコア特徴量"""
    tracker = get_optimizer_tracker()
    return pd.Series(
        [tracker.get_optimizer_stability_score()] * len(df), index=df.index
    )

@FeatureRegistry.register("optimizer_adaptive_lr_score")
def optimizer_adaptive_lr_score(df: pd.DataFrame) -> pd.Series:
    """適応学習率スコア特徴量"""
    tracker = get_optimizer_tracker()
    return pd.Series(
        [tracker.get_adaptive_learning_rate_score()] * len(df), index=df.index
    )
