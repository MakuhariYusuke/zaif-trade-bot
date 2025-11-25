#!/usr/bin/env python3
"""
V433 Adaptive SAC Core
市場レジーム適応型SAC実装
"""

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from ztb.utils.logging_utils import get_logger


# Dummy MarketRegimeDetector for testing
class MarketRegimeDetector:
    def __init__(self):
        self.current_regime = "neutral"

    def detect_regime(self, market_data):
        # Dummy implementation - always return neutral regime
        return MarketRegimeState(
            regime="neutral",
            confidence=0.5,
            volatility=0.1,
            trend_strength=0.0,
            volume_profile="normal",
        )

    def get_current_regime(self):
        return MarketRegimeState(
            regime=self.current_regime,
            confidence=0.5,
            volatility=0.1,
            trend_strength=0.0,
            volume_profile="normal",
        )


from ztb.optimization.unified_optimizer import OptimizationConfig, UnifiedOptimizer

logger = get_logger(__name__)


@dataclass
class AdaptiveSACConfig:
    """適応型SAC設定"""

    # 基本SAC設定
    learning_rate: float = 3e-4
    buffer_size: int = 1000000
    learning_starts: int = 100
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1

    # 適応機能設定
    enable_market_regime_adaptation: bool = True
    regime_detection_window: int = 1000
    adaptation_interval_steps: int = 1000
    adaptation_smoothing_factor: float = 0.1

    # オンライン学習設定
    enable_online_learning: bool = True
    online_update_freq: int = 100
    experience_replay_alpha: float = 0.6
    experience_replay_beta: float = 0.4

    # パフォーマンス監視
    performance_window_size: int = 100
    performance_threshold: float = 0.7
    reoptimization_trigger_threshold: float = 0.5

    # 動的パラメータ調整
    dynamic_lr_range: Tuple[float, float] = (1e-5, 1e-2)
    dynamic_tau_range: Tuple[float, float] = (0.001, 0.01)
    dynamic_gamma_range: Tuple[float, float] = (0.95, 0.999)

    # リスク適応
    risk_adaptation_enabled: bool = True
    volatility_scaling_factor: float = 1.0
    drawdown_adaptation_threshold: float = 0.1


@dataclass
class MarketRegimeState:
    """市場レジーム状態"""

    regime: str = "neutral"
    confidence: float = 0.5
    volatility: float = 0.0
    trend_strength: float = 0.0
    volume_profile: str = "normal"
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime,
            "confidence": self.confidence,
            "volatility": self.volatility,
            "trend_strength": self.trend_strength,
            "volume_profile": self.volume_profile,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EpisodePerformanceMetrics:
    """パフォーマンス指標"""

    episode_reward: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 1.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    volatility: float = 0.0
    total_trades: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class AdaptiveSACPolicy(nn.Module):
    """適応型SACポリシー"""

    def __init__(
        self, observation_dim: int, action_dim: int, hidden_dims: List[int] = None
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        # 市場レジーム適応層
        self.regime_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),  # regime features: volatility, trend, volume, confidence
        )

        # アダプティブポリシーネットワーク
        layers = []
        prev_dim = observation_dim + 16  # observation + regime encoding

        for hidden_dim in hidden_dims:
            layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))
        self.policy_net = nn.Sequential(*layers)

        # アダプティブ価値ネットワーク
        value_layers = []
        prev_dim = observation_dim + 16

        for hidden_dim in hidden_dims:
            value_layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)]
            )
            prev_dim = hidden_dim

        value_layers.append(nn.Linear(prev_dim, 1))
        self.value_net = nn.Sequential(*value_layers)

        # Qネットワーク（双子構造）
        self.q1_net = nn.Sequential(
            *value_layers[:-1], nn.Linear(hidden_dims[-1], action_dim)
        )
        self.q2_net = nn.Sequential(
            *value_layers[:-1], nn.Linear(hidden_dims[-1], action_dim)
        )

    def forward(
        self, obs: torch.Tensor, regime_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """順伝播"""
        # レジーム特徴量のエンコーディング
        regime_encoded = self.regime_encoder(regime_features)

        # 観測とレジーム特徴量の結合
        combined_input = torch.cat([obs, regime_encoded], dim=-1)

        # 各ネットワークの出力
        policy_output = self.policy_net(combined_input)
        value_output = self.value_net(combined_input)
        q1_output = self.q1_net(combined_input)
        q2_output = self.q2_net(combined_input)

        return policy_output, value_output, q1_output, q2_output


class AdaptiveSACCore:
    """
    V433適応型SACコア
    市場レジーム検知と動的適応機能を統合
    """

    def __init__(
        self, config: AdaptiveSACConfig, observation_dim: int, action_dim: int
    ):
        self.config = config
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.logger = get_logger(__name__)

        # コンポーネントの初期化
        self.market_regime_detector = MarketRegimeDetector()
        self.unified_optimizer = UnifiedOptimizer(OptimizationConfig())

        # SACモデルの初期化
        self.sac_model = None
        self.adaptive_policy = AdaptiveSACPolicy(observation_dim, action_dim)

        # 状態管理
        self.current_regime_state = MarketRegimeState()
        self.performance_history = deque(maxlen=config.performance_window_size)
        self.regime_history = deque(maxlen=100)

        # 適応パラメータ
        self.adaptation_params = {
            "learning_rate": config.learning_rate,
            "tau": config.tau,
            "gamma": config.gamma,
        }

        # オンライン学習バッファ
        self.online_buffer = deque(maxlen=config.buffer_size)

        # スレッド管理
        self.adaptation_thread = None
        self.monitoring_thread = None
        self.is_running = False

        # パフォーマンス追跡
        self.episode_rewards = []
        self.adaptation_log = []

    def initialize_sac_model(self, env) -> SAC:
        """SACモデルの初期化"""
        self.logger.info("Initializing adaptive SAC model")

        # カスタムポリシーを使用したSACモデルの作成
        policy_kwargs = {
            "features_extractor_class": AdaptiveFeatureExtractor,
            "features_extractor_kwargs": {
                "regime_detector": self.market_regime_detector
            },
            "net_arch": [256, 256],
        }

        self.sac_model = SAC(
            "MlpPolicy",
            env,
            learning_rate=self.config.learning_rate,
            buffer_size=self.config.buffer_size,
            learning_starts=self.config.learning_starts,
            batch_size=self.config.batch_size,
            tau=self.config.tau,
            gamma=self.config.gamma,
            train_freq=self.config.train_freq,
            gradient_steps=self.config.gradient_steps,
            policy_kwargs=policy_kwargs,
            verbose=1,
        )

        return self.sac_model

    def start_adaptive_training(self, env, total_timesteps: int):
        """適応型トレーニングを開始"""
        self.is_running = True

        # モニタリングスレッドを開始
        self.monitoring_thread = threading.Thread(
            target=self._performance_monitoring_loop
        )
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        # 適応スレッドを開始
        self.adaptation_thread = threading.Thread(target=self._adaptation_loop)
        self.adaptation_thread.daemon = True
        self.adaptation_thread.start()

        # カスタムコールバックでトレーニング
        callback = AdaptiveSACCallback(
            regime_detector=self.market_regime_detector,
            adaptation_core=self,
            check_freq=self.config.adaptation_interval_steps,
        )

        self.logger.info("Starting adaptive SAC training")
        self.sac_model.learn(
            total_timesteps=total_timesteps, callback=callback, log_interval=100
        )

        self.is_running = False

    def _performance_monitoring_loop(self):
        """パフォーマンス監視ループ"""
        while self.is_running:
            try:
                # パフォーマンス指標の計算
                if len(self.episode_rewards) > 0:
                    recent_rewards = self.episode_rewards[
                        -self.config.performance_window_size :
                    ]
                    metrics = self._calculate_performance_metrics(recent_rewards)

                    self.performance_history.append(metrics)

                    # パフォーマンス低下検知
                    if self._detect_performance_degradation(metrics):
                        self.logger.warning(
                            "Performance degradation detected, triggering adaptation"
                        )
                        self._trigger_adaptation()

                time.sleep(10)  # 10秒ごとに監視

            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                time.sleep(30)

    def _adaptation_loop(self):
        """適応ループ"""
        while self.is_running:
            try:
                # 市場レジームの検知と更新
                if self.config.enable_market_regime_adaptation:
                    self._update_market_regime()

                # 動的パラメータ調整
                self._adaptive_parameter_tuning()

                time.sleep(self.config.adaptation_interval_steps)

            except Exception as e:
                self.logger.error(f"Adaptation loop error: {e}")
                time.sleep(60)

    def _update_market_regime(self):
        """市場レジームの更新"""
        try:
            # 市場データの取得（実際の実装ではデータソースから取得）
            market_data = self._get_current_market_data()

            if market_data is not None:
                regime_state = self.market_regime_detector.detect_regime(market_data)
                self.current_regime_state = regime_state
                self.regime_history.append(regime_state)

                self.logger.info(
                    f"Market regime updated: {regime_state.regime} "
                    f"(confidence: {regime_state.confidence:.2f})"
                )

        except Exception as e:
            self.logger.error(f"Market regime update failed: {e}")

    def _adaptive_parameter_tuning(self):
        """動的パラメータ調整"""
        if not self.performance_history:
            return

        latest_metrics = self.performance_history[-1]
        regime = self.current_regime_state.regime

        # レジームに応じたパラメータ調整
        regime_adjustments = {
            "bull": {
                "learning_rate_multiplier": 1.2,
                "tau_multiplier": 0.8,
                "gamma_adjustment": 0.005,
            },
            "bear": {
                "learning_rate_multiplier": 0.8,
                "tau_multiplier": 1.3,
                "gamma_adjustment": -0.005,
            },
            "volatile": {
                "learning_rate_multiplier": 0.9,
                "tau_multiplier": 1.5,
                "gamma_adjustment": 0.01,
            },
            "neutral": {
                "learning_rate_multiplier": 1.0,
                "tau_multiplier": 1.0,
                "gamma_adjustment": 0.0,
            },
        }

        adjustments = regime_adjustments.get(regime, regime_adjustments["neutral"])

        # パフォーマンスに基づく追加調整
        if latest_metrics.sharpe_ratio < 0.5:
            adjustments["learning_rate_multiplier"] *= 0.8
        elif latest_metrics.sharpe_ratio > 1.0:
            adjustments["learning_rate_multiplier"] *= 1.1

        # パラメータの更新
        new_lr = np.clip(
            self.adaptation_params["learning_rate"]
            * adjustments["learning_rate_multiplier"],
            self.config.dynamic_lr_range[0],
            self.config.dynamic_lr_range[1],
        )

        new_tau = np.clip(
            self.adaptation_params["tau"] * adjustments["tau_multiplier"],
            self.config.dynamic_tau_range[0],
            self.config.dynamic_tau_range[1],
        )

        new_gamma = np.clip(
            self.adaptation_params["gamma"] + adjustments["gamma_adjustment"],
            self.config.dynamic_gamma_range[0],
            self.config.dynamic_gamma_range[1],
        )

        # SACモデルのパラメータ更新
        if self.sac_model:
            self.sac_model.learning_rate = new_lr
            self.sac_model.tau = new_tau
            self.sac_model.gamma = new_gamma

        self.adaptation_params.update(
            {"learning_rate": new_lr, "tau": new_tau, "gamma": new_gamma}
        )

        self.adaptation_log.append(
            {
                "timestamp": datetime.now(),
                "regime": regime,
                "adjustments": adjustments,
                "new_params": self.adaptation_params.copy(),
            }
        )

        self.logger.info(
            f"Adaptive parameters updated for {regime} regime: "
            f"LR={new_lr:.6f}, Tau={new_tau:.4f}, Gamma={new_gamma:.4f}"
        )

    def _calculate_performance_metrics(
        self, rewards: List[float]
    ) -> EpisodePerformanceMetrics:
        """パフォーマンス指標の計算"""
        if not rewards:
            return EpisodePerformanceMetrics()

        metrics = EpisodePerformanceMetrics()

        # 基本指標
        metrics.episode_reward = np.mean(rewards)
        metrics.total_trades = len(rewards)

        # 勝率とプロフィットファクターの計算（簡易版）
        positive_rewards = [r for r in rewards if r > 0]
        negative_rewards = [r for r in rewards if r < 0]

        if rewards:
            metrics.win_rate = len(positive_rewards) / len(rewards)

        if negative_rewards:
            gross_profit = sum(positive_rewards)
            gross_loss = abs(sum(negative_rewards))
            metrics.profit_factor = (
                gross_profit / gross_loss if gross_loss > 0 else float("inf")
            )

        # ドローダウンの計算
        cumulative = np.cumsum(rewards)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        metrics.max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

        # シャープレシオの計算
        if len(rewards) > 1:
            returns = np.array(rewards)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            metrics.sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
            metrics.volatility = std_return

        return metrics

    def _detect_performance_degradation(
        self, metrics: EpisodePerformanceMetrics
    ) -> bool:
        """パフォーマンス低下の検知"""
        if len(self.performance_history) < 5:
            return False

        # 最近のパフォーマンスと比較
        recent_metrics = list(self.performance_history)[-5:]
        avg_recent_sharpe = np.mean([m.sharpe_ratio for m in recent_metrics])

        degradation_threshold = self.config.reoptimization_trigger_threshold
        return metrics.sharpe_ratio < avg_recent_sharpe * (1 - degradation_threshold)

    def _trigger_adaptation(self):
        """適応トリガーの実行"""
        self.logger.info("Triggering emergency adaptation")

        # 即時パラメータ調整
        emergency_adjustments = {
            "learning_rate": self.adaptation_params["learning_rate"] * 0.5,
            "tau": self.adaptation_params["tau"] * 1.5,
            "gamma": max(0.9, self.adaptation_params["gamma"] - 0.05),
        }

        if self.sac_model:
            self.sac_model.learning_rate = emergency_adjustments["learning_rate"]
            self.sac_model.tau = emergency_adjustments["tau"]
            self.sac_model.gamma = emergency_adjustments["gamma"]

        self.adaptation_params.update(emergency_adjustments)

        # 最適化のトリガー
        try:
            self.unified_optimizer.adaptive_optimize(
                current_performance={"score": 0.5},
                market_regime=self.current_regime_state.regime,  # 現在の推定スコア
            )
        except Exception as e:
            self.logger.error(f"Emergency optimization failed: {e}")

    def _get_current_market_data(self) -> Optional[pd.DataFrame]:
        """現在の市場データを取得（実際の実装ではデータソースから取得）"""
        # ダミー実装 - 実際には市場データフィードから取得
        return None

    def online_learn(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ):
        """オンライン学習"""
        if not self.config.enable_online_learning:
            return

        # エクスペリエンスをバッファに追加
        experience = (observation, action, reward, next_observation, done)
        self.online_buffer.append(experience)

        # 定期的なオンライン更新
        if len(self.online_buffer) >= self.config.online_update_freq:
            self._perform_online_update()

    def _perform_online_update(self):
        """オンライン更新の実行"""
        if not self.sac_model or len(self.online_buffer) < self.config.batch_size:
            return

        try:
            # バッファからバッチをサンプリング
            batch_indices = np.random.choice(
                len(self.online_buffer),
                size=min(self.config.batch_size, len(self.online_buffer)),
                replace=False,
            )

            batch = [self.online_buffer[i] for i in batch_indices]

            # オンライン学習ステップ
            observations = np.array([exp[0] for exp in batch])
            actions = np.array([exp[1] for exp in batch])
            rewards = np.array([exp[2] for exp in batch])
            next_observations = np.array([exp[3] for exp in batch])
            dones = np.array([exp[4] for exp in batch])

            # SACの学習ステップ
            self.sac_model.train(batch_size=len(batch))

            self.logger.debug(
                f"Online learning update completed with {len(batch)} samples"
            )

        except Exception as e:
            self.logger.error(f"Online learning update failed: {e}")

    def get_adaptation_status(self) -> Dict[str, Any]:
        """適応状態を取得"""
        return {
            "current_regime": self.current_regime_state.to_dict(),
            "adaptation_params": self.adaptation_params.copy(),
            "performance_metrics": self.performance_history[-1].__dict__
            if self.performance_history
            else None,
            "online_buffer_size": len(self.online_buffer),
            "adaptation_log_size": len(self.adaptation_log),
            "is_adapting": self.is_running,
        }

    def save_adaptive_state(self, filepath: str):
        """適応状態を保存"""
        state = {
            "config": self.config.__dict__,
            "current_regime_state": self.current_regime_state.to_dict(),
            "adaptation_params": self.adaptation_params,
            "performance_history": [m.__dict__ for m in self.performance_history],
            "adaptation_log": self.adaptation_log,
            "timestamp": datetime.now().isoformat(),
        }

        torch.save(state, filepath)
        self.logger.info(f"Adaptive state saved to {filepath}")

    def load_adaptive_state(self, filepath: str):
        """適応状態を読み込み"""
        state = torch.load(filepath)

        self.current_regime_state = MarketRegimeState(**state["current_regime_state"])
        self.adaptation_params = state["adaptation_params"]
        self.performance_history = deque(
            [EpisodePerformanceMetrics(**m) for m in state["performance_history"]],
            maxlen=self.config.performance_window_size,
        )
        self.adaptation_log = state["adaptation_log"]

        self.logger.info(f"Adaptive state loaded from {filepath}")


class AdaptiveFeatureExtractor(nn.Module):
    """適応型特徴抽出器"""

    def __init__(
        self, observation_space, features_dim: int = 256, regime_detector=None
    ):
        super().__init__()
        self.regime_detector = regime_detector

        # 特徴抽出ネットワーク
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 基本特徴抽出
        features = self.net(observations)

        # レジーム適応特徴（オプション）
        if self.regime_detector:
            # レジーム状態に基づく特徴調整
            regime_features = self._get_regime_features()
            if regime_features is not None:
                # 特徴量にレジーム情報を統合
                regime_encoded = torch.tensor(
                    regime_features, device=observations.device
                )
                features = features * (
                    1 + regime_encoded.unsqueeze(0).expand(features.shape[0], -1)
                )

        return features

    def _get_regime_features(self) -> Optional[np.ndarray]:
        """レジーム特徴量を取得"""
        if not self.regime_detector:
            return None

        try:
            # 現在のレジーム状態を取得
            regime_state = self.regime_detector.get_current_regime()
            if regime_state:
                return np.array(
                    [
                        regime_state.confidence,
                        regime_state.volatility,
                        regime_state.trend_strength,
                        1.0 if regime_state.volume_profile == "high" else 0.0,
                    ]
                )
        except Exception:
            pass

        return None


class AdaptiveSACCallback(BaseCallback):
    """適応型SACトレーニングコールバック"""

    def __init__(
        self, regime_detector, adaptation_core: AdaptiveSACCore, check_freq: int = 1000
    ):
        super().__init__(check_freq)
        self.regime_detector = regime_detector
        self.adaptation_core = adaptation_core
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        # エピソード報酬の追跡
        reward = self.locals.get("reward", 0)
        self.current_episode_reward += reward

        if self.locals.get("done", False):
            self.episode_rewards.append(self.current_episode_reward)
            self.adaptation_core.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0

        return True

    def _on_rollout_end(self) -> None:
        # ロールアウト終了時の適応チェック
        if len(self.episode_rewards) >= 10:  # 十分なエピソードデータがある場合
            self.adaptation_core._adaptive_parameter_tuning()


def create_adaptive_sac_core(
    config: AdaptiveSACConfig = None,
    observation_dim: int = None,
    action_dim: int = None,
) -> AdaptiveSACCore:
    """AdaptiveSACCoreのファクトリ関数"""
    if config is None:
        config = AdaptiveSACConfig()

    if observation_dim is None or action_dim is None:
        raise ValueError("observation_dim and action_dim must be specified")

    return AdaptiveSACCore(config, observation_dim, action_dim)


# 使用例
if __name__ == "__main__":
    # 設定の作成
    config = AdaptiveSACConfig(
        enable_market_regime_adaptation=True,
        enable_online_learning=True,
        adaptation_interval_steps=500,
        performance_window_size=50,
    )

    # 適応型SACコアの作成
    observation_dim = 10  # 観測空間の次元
    action_dim = 3  # 行動空間の次元

    adaptive_sac = create_adaptive_sac_core(config, observation_dim, action_dim)

    print("Adaptive SAC Core created successfully")
    print(f"Configuration: {config}")
    print(f"Adaptation status: {adaptive_sac.get_adaptation_status()}")
