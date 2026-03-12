#!/usr/bin/env python3
"""
Simplified Aggressive Performance Training Script
基本的なSACトレーニングで健全性チェックを行う
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from ztb.utils.training_utils import display_training_complete, save_model

try:
    import gymnasium as gym
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("Attempting to continue with available modules...")


class SimplifiedAggressiveTrainer:
    """簡略化された積極的アプローチトレーナー"""

    def __init__(self, config_path: str, verbose: bool = True):
        self.config_path = config_path
        self.verbose = verbose
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.model = None
        self.env = None
        self.eval_env = None

        # 結果保存用
        self.training_results = {}
        self.performance_metrics = {}


    def _setup_logging(self) -> logging.Logger:
        """ログ設定"""
        logging.basicConfig(
            level=logging.INFO if self.verbose else logging.WARNING,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(
                    f'simplified_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                ),
                logging.StreamHandler(),
            ],
        )
        return logging.getLogger(__name__)

    def _create_environment(self, is_eval: bool = False):
        """シンプルなGym環境を作成"""
        # テスト用のPendulum環境を使用
        env = gym.make("Pendulum-v1")

        if is_eval:
            env = Monitor(env)

        return env

    def _create_model(self):
        """SACモデルの作成"""
        sac_params = self.config["training"]["sac_hyperparameters"]

        # 基本的なパラメータのみを使用
        model = SAC(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=sac_params.get("learning_rate", 0.0003),
            buffer_size=min(
                sac_params.get("buffer_size", 1000000), 100000
            ),  # メモリ制限
            learning_starts=sac_params.get("learning_starts", 1000),
            batch_size=sac_params.get("batch_size", 256),
            tau=sac_params.get("tau", 0.005),
            gamma=sac_params.get("gamma", 0.99),
            train_freq=sac_params.get("train_freq", 1),
            gradient_steps=sac_params.get("gradient_steps", 1),
            ent_coef=sac_params.get("ent_coef", "auto"),
            verbose=1 if self.verbose else 0,
            device="cpu",  # CPUを使用
        )

        self.logger.info("SAC model created with simplified parameters")
        return model

    def _setup_callbacks(self):
        """コールバックの設定"""
        callbacks = []

        # チェックポイントコールバック
        checkpoint_callback = CheckpointCallback(
            save_freq=1000,  # 1kステップごとに保存
            save_path="./checkpoints/",
            name_prefix="simplified_aggressive",
            save_replay_buffer=False,  # メモリ節約
            save_vecnormalize=False,
        )
        callbacks.append(checkpoint_callback)

        # 評価コールバック
        if self.eval_env:
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path="./best_model/",
                log_path="./eval_logs/",
                eval_freq=500,  # 500ステップごとに評価
                deterministic=True,
                render=False,
                n_eval_episodes=3,
            )
            callbacks.append(eval_callback)

        return callbacks

    def train(self) -> Dict[str, Any]:
        """トレーニング実行"""
        try:
            self.logger.info("🚀 Starting Simplified Aggressive Performance Training")
            self.logger.info(f"Config: {self.config_path}")
            self.logger.info(
                f"Total timesteps: {self.config['training']['total_timesteps']}"
            )

            # 環境作成
            self.env = self._create_environment(is_eval=False)
            self.eval_env = self._create_environment(is_eval=True)

            # モデル作成
            self.model = self._create_model()

            # コールバック設定
            callbacks = self._setup_callbacks()

            # トレーニング開始
            start_time = datetime.now()
            self.logger.info(f"Training started at {start_time}")

            self.model.learn(
                total_timesteps=self.config["training"]["total_timesteps"],
                callback=callbacks,
                progress_bar=True,
            )

            end_time = datetime.now()
            training_duration = end_time - start_time

            # 結果保存
            self.training_results = {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": str(training_duration),
                "total_timesteps": self.config["training"]["total_timesteps"],
                "config_used": self.config_path,
                "model_saved": True,
            }

            # パフォーマンス評価
            self._evaluate_performance()

            self.logger.info(
                "✅ Simplified Aggressive Performance Training completed successfully!"
            )
            return self.training_results

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

    def _evaluate_performance(self):
        """パフォーマンス評価"""
        try:
            self.logger.info("📊 Evaluating performance...")

            # 評価実行
            episode_rewards = []
            episode_lengths = []
            entropies = []
            action_means = []
            action_stds = []

            for episode in range(10):  # 10エピソード評価
                obs, _ = self.eval_env.reset()
                episode_reward = 0
                episode_length = 0
                episode_entropies = []
                episode_actions = []

                done = False
                truncated = False

                while not (done or truncated):
                    action, _ = self.model.predict(obs, deterministic=False)
                    episode_actions.append(
                        action[0] if hasattr(action, "__len__") else action
                    )

                    obs, reward, done, truncated, info = self.eval_env.step(action)
                    episode_reward += reward
                    episode_length += 1

                    # エントロピー情報の収集（利用可能な場合）
                    if hasattr(self.model, "policy") and hasattr(
                        self.model.policy, "log_std"
                    ):
                        log_std = self.model.policy.log_std.detach().cpu().numpy()
                        entropy = -0.5 * np.sum(2 * log_std + 1 + 2 * np.log(2 * np.pi))
                        episode_entropies.append(entropy)

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

                if episode_entropies:
                    entropies.extend(episode_entropies)

                if episode_actions:
                    action_means.append(np.mean(episode_actions))
                    action_stds.append(np.std(episode_actions))

            # メトリクス計算
            self.performance_metrics = {
                "mean_reward": float(np.mean(episode_rewards)),
                "std_reward": float(np.std(episode_rewards)),
                "mean_episode_length": float(np.mean(episode_lengths)),
                "total_episodes": len(episode_rewards),
                "evaluation_timestamp": datetime.now().isoformat(),
            }

            # アクション分布の分析
            if action_means:
                self.performance_metrics.update(
                    {
                        "action_mean_avg": float(np.mean(action_means)),
                        "action_mean_std": float(np.std(action_means)),
                        "action_std_avg": float(np.mean(action_stds)),
                        "action_std_std": float(np.std(action_stds)),
                    }
                )

            # エントロピー分析
            if entropies:
                self.performance_metrics.update(
                    {
                        "entropy_mean": float(np.mean(entropies)),
                        "entropy_std": float(np.std(entropies)),
                        "entropy_min": float(np.min(entropies)),
                        "entropy_max": float(np.max(entropies)),
                    }
                )

            self.logger.info(f"Performance Metrics: {self.performance_metrics}")

            # 健全性チェック
            self._check_training_health()

        except Exception as e:
            self.logger.error(f"Performance evaluation failed: {e}")

    def _check_training_health(self):
        """トレーニングの健全性をチェック"""
        metrics = self.performance_metrics

        self.logger.info("🔍 Training Health Check:")

        # エントロピー分析
        if "entropy_mean" in metrics:
            entropy = metrics["entropy_mean"]
            self.logger.info(f"  Entropy: {entropy:.4f}")
            if entropy < -2.0:
                self.logger.warning("  ⚠️  Low entropy - possible value sticking")
            elif entropy > 2.0:
                self.logger.warning("  ⚠️  High entropy - possible instability")
            else:
                self.logger.info("  ✅ Entropy in healthy range")

        # アクション分布分析
        if "action_mean_avg" in metrics and "action_std_avg" in metrics:
            action_mean = metrics["action_mean_avg"]
            action_std = metrics["action_std_avg"]
            self.logger.info(f"  Action Mean: {action_mean:.4f}, Std: {action_std:.4f}")

            # アクションの範囲チェック（Pendulumは-2から2の範囲）
            if abs(action_mean) > 1.5:
                self.logger.warning("  ⚠️  Action mean far from center - possible bias")
            else:
                self.logger.info("  ✅ Action mean in reasonable range")

            if action_std < 0.1:
                self.logger.warning("  ⚠️  Low action variance - possible sticking")
            elif action_std > 1.5:
                self.logger.warning("  ⚠️  High action variance - possible instability")
            else:
                self.logger.info("  ✅ Action variance in healthy range")

        # リワード分析
        reward_mean = metrics.get("mean_reward", 0)
        reward_std = metrics.get("std_reward", 0)
        self.logger.info(f"  Reward: Mean={reward_mean:.2f}, Std={reward_std:.2f}")

        if abs(reward_mean) > 1000:
            self.logger.warning("  ⚠️  Extreme reward values - check reward scaling")
        else:
            self.logger.info("  ✅ Reward values in reasonable range")

    def save_results(self, output_dir: str = "./training_results"):
        """結果保存"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # トレーニング結果
        results_file = output_path / f"simplified_training_results_{timestamp}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "training_results": self.training_results,
                    "performance_metrics": self.performance_metrics,
                    "config": self.config,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        # モデル保存
        if self.model:
            model_file = output_path / f"simplified_aggressive_model_{timestamp}"
            save_model(self.model, str(model_file))
            self.logger.info(f"Model saved to {model_file}")

        self.logger.info(f"Results saved to {output_path}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Simplified Aggressive Performance SAC Training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/v445/sac_v445.2_aggressive_performance_optimized.json",
        help="Path to configuration file",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./training_results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    start_time = time.time()

    trainer = SimplifiedAggressiveTrainer(args.config, args.verbose)

    try:
        results = trainer.train()
        trainer.save_results(args.output_dir)

        training_time = time.time() - start_time
        final_metrics = {
            "training_success": True,
            "results_saved_to": args.output_dir,
        }
        display_training_complete(final_metrics, training_time)

    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
