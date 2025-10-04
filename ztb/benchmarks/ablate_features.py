#!/usr/bin/env python3
"""
Feature ablation script for trading.
特徴量アブレーションスクリプト
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from ztb.utils.data.data_generation import generate_synthetic_data

# ロギング設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.as_posix()
if project_root not in sys.path:
    sys.path.append(project_root)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.features import get_feature_manager
from ztb.trading.environment import HeavyTradingEnv
from ztb.utils.errors import safe_operation


class MetricsCallback(BaseCallback):
    """メトリクス収集コールバック"""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        if len(self.locals["rewards"]) > 0:
            self.current_episode_reward += self.locals["rewards"][0]
        self.current_episode_length += 1
        # If using multiple environments, consider: if all(self.locals['dones'])
        # This code assumes a single environment.
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.current_episode_length = 0

        return True

    def get_metrics(self) -> Dict[str, float]:
        if not self.episode_rewards:
            return {
                "mean_reward": 0.0,
                "win_rate": 0.0,
                "sharpe_like": 0.0,
                "trades_per_1k": 0.0,
                "fps": 0.0,
                "env_step_ms": 0.0,
            }

        rewards = np.array(self.episode_rewards)
        mean_reward = np.mean(rewards)

        # win_rate: 報酬 > 0 の割合
        win_rate = np.mean(rewards > 0)

        # sharpe_like: mean / std (簡易版)
        if len(rewards) > 1:
            sharpe_like = mean_reward / np.std(rewards) if np.std(rewards) > 0 else 0
        else:
            sharpe_like = 0

        # trades_per_1k: 環境から取得（複数環境時は平均を取る）
        trades_per_1k_values = self.training_env.env_method("get_trades_per_1k")
        if trades_per_1k_values and all(
            isinstance(v, (int, float)) for v in trades_per_1k_values
        ):
            trades_per_1k = float(np.mean(trades_per_1k_values))
        else:
            trades_per_1k = 0.0

        # fps: ステップ数 / 時間
        total_steps = sum(self.episode_lengths)
        fps = total_steps / (self.num_timesteps / 1000) if self.num_timesteps > 0 else 0

        # env_step_ms: 環境ステップ時間
        # Robustly handle env_method return value for get_last_step_time
        step_times = self.training_env.env_method("get_last_step_time")
        if step_times and isinstance(step_times[0], (int, float)):
            env_step_ms = step_times[0] * 1000
        else:
            env_step_ms = 0.0

        return {
            "mean_reward": float(mean_reward),
            "win_rate": float(win_rate),
            "sharpe_like": float(sharpe_like),
            "trades_per_1k": float(trades_per_1k),
            "fps": float(fps),
            "env_step_ms": float(env_step_ms),
        }


def create_env_with_features(
    df: pd.DataFrame, enabled_features: List[str]
) -> HeavyTradingEnv:
    """特徴量を有効化した環境を作成"""
    return safe_operation(
        logger=None,  # Use default logger
        operation=lambda: _create_env_with_features_impl(df, enabled_features),
        context="environment_creation_with_features",
        default_result=None,  # Return None on failure
    )


def _create_env_with_features_impl(
    df: pd.DataFrame, enabled_features: List[str]
) -> HeavyTradingEnv:
    """Implementation of environment creation with features."""
    # 特徴量計算
    manager = get_feature_manager()
    original_enabled = manager.get_enabled_features()

    # Wave3特徴量チェック
    wave3_features = {"Ichimoku", "Donchian", "RegimeClustering", "KalmanFilter"}
    is_wave3_only = set(enabled_features).issubset(wave3_features)

    # 一時的に特徴量を変更
    for name in manager.features:
        config = manager.config["features"].get(name, {})
        if is_wave3_only and name in wave3_features:
            # Wave3のみの場合、harmfulを無視
            config["enabled"] = name in enabled_features
        else:
            config["enabled"] = name in enabled_features
        manager.config["features"][name] = config

    # 特徴量計算 (wave=Noneで全て計算)
    df_with_features = manager.compute_features(df, wave=None)

    logging.info(f"Enabled features: {enabled_features}")
    logging.info(f"Features in df: {len(df_with_features.columns)} columns")

    # 環境作成
    env = HeavyTradingEnv(df_with_features)

    # 設定戻す
    for name in manager.features:
        config = manager.config["features"].get(name, {})
        config["enabled"] = name in original_enabled
        manager.config["features"][name] = config

    return env


def run_training(
    seed: int, timesteps: int, enabled_features: List[str], df: pd.DataFrame
) -> Dict[str, float]:
    """学習実行"""
    return safe_operation(
        logger=None,  # Use default logger
        operation=lambda: _run_training_impl(seed, timesteps, enabled_features, df),
        context="training_execution",
        default_result={},  # Return empty dict on failure
    )


def _run_training_impl(
    seed: int, timesteps: int, enabled_features: List[str], df: pd.DataFrame
) -> Dict[str, float]:
    """Implementation of training execution."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 環境作成
    def make_env() -> Any:
        env = create_env_with_features(df, enabled_features)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])

    # PPO設定（テスト用）
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        seed=seed,
    )

    # コールバック
    callback = MetricsCallback()

    # 学習
    model.learn(total_timesteps=timesteps, callback=callback)

    # メトリクス取得
    metrics = callback.get_metrics()

    env.close()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature ablation analysis")
    parser.add_argument(
        "--timesteps", type=int, default=10000, help="Training timesteps"
    )
    parser.add_argument(
        "--seeds", type=str, default="42,123,2025", help="Comma-separated seeds"
    )
    parser.add_argument("--waves", type=str, default="1", help="Comma-separated waves")
    parser.add_argument(
        "--features", type=str, default=None, help="Feature set to ablate (e.g., wave3)"
    )
    parser.add_argument(
        "--mode", type=str, default=None, help="Special mode (e.g., wave3_refined)"
    )
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument(
        "--data-rows", type=int, default=5000, help="Number of data rows"
    )
    parser.add_argument(
        "--float-precision", type=int, default=3, help="Float precision for output"
    )
    parser.add_argument(
        "--dump-metrics",
        action="store_true",
        help="Dump individual run metrics to JSON",
    )

    args = parser.parse_args()

    # モード処理
    if args.mode == "wave3_refined":
        args.features = "wave3"
        args.timesteps = 10000
        args.seeds = "42,123"
        args.float_precision = 6

    # マネージャー初期化
    manager = get_feature_manager()
    # シード
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    df = generate_synthetic_data(args.data_rows)

    # シード
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # 特徴量セット
    if args.features == "wave3":
        # Wave3のみをアブレーション（harmfulフラグを無視）
        all_features = ["Ichimoku", "Donchian", "RegimeClustering", "KalmanFilter"]
    else:
        waves = [int(w.strip()) for w in args.waves.split(",") if w.strip()]
        all_features = []
        for wave in waves:
            all_features.extend(manager.get_enabled_features(wave))

    # 出力ディレクトリ
    output_dir = Path("reports/feature_ranking")
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output is None:
        output_file = output_dir / f"ablation_{date_str}.csv"
    else:
        output_file = Path(args.output)

    results = []

    logging.info(
        f"Running ablation analysis with {len(seeds)} seeds, {args.timesteps} timesteps each..."
    )

    # ベースライン（全特徴量ON）
    logging.info("Baseline (all features)...")
    baseline_metrics = {}
    for seed in seeds:
        metrics = run_training(seed, args.timesteps, all_features, df)
        for key, value in metrics.items():
            if key not in baseline_metrics:
                baseline_metrics[key] = []
            baseline_metrics[key].append(value)

    # 平均
    baseline_avg = {k: np.mean(v) for k, v in baseline_metrics.items()}

    results.append(
        {
            "feature": "baseline",
            "mean_reward": baseline_avg["mean_reward"],
            "win_rate": baseline_avg["win_rate"],
            "sharpe_like": baseline_avg["sharpe_like"],
            "trades_per_1k": baseline_avg["trades_per_1k"],
            "fps": baseline_avg["fps"],
            "env_step_ms": baseline_avg["env_step_ms"],
            "delta_mean_reward": 0,
            "delta_win_rate": 0,
            "delta_sharpe_like": 0,
            "delta_trades_per_1k": 0,
            "delta_fps": 0,
            "delta_env_step_ms": 0,
        }
    )

    # Leave-One-Outアブレーション
    for ablated_feature in all_features:
        logging.info(f"Ablating {ablated_feature}...")
        ablated_features = [f for f in all_features if f != ablated_feature]

        ablation_metrics = {}
        for seed in seeds:
            metrics = run_training(seed, args.timesteps, ablated_features, df)
            for key, value in metrics.items():
                if key not in ablation_metrics:
                    ablation_metrics[key] = []
                ablation_metrics[key].append(value)

        # 平均
        ablation_avg = {k: np.mean(v) for k, v in ablation_metrics.items()}

        results.append(
            {
                "feature": ablated_feature,
                "mean_reward": ablation_avg["mean_reward"],
                "win_rate": ablation_avg["win_rate"],
                "sharpe_like": ablation_avg["sharpe_like"],
                "trades_per_1k": ablation_avg["trades_per_1k"],
                "fps": ablation_avg["fps"],
                "env_step_ms": ablation_avg["env_step_ms"],
                "delta_mean_reward": ablation_avg["mean_reward"]
                - baseline_avg["mean_reward"],
                "delta_win_rate": ablation_avg["win_rate"] - baseline_avg["win_rate"],
                "delta_sharpe_like": ablation_avg["sharpe_like"]
                - baseline_avg["sharpe_like"],
                "delta_trades_per_1k": ablation_avg["trades_per_1k"]
                - baseline_avg["trades_per_1k"],
                "delta_fps": ablation_avg["fps"] - baseline_avg["fps"],
                "delta_env_step_ms": ablation_avg["env_step_ms"]
                - baseline_avg["env_step_ms"],
            }
        )

    # CSV出力
    # float_format の指定例: float_format="%.3f" で小数点以下3桁にフォーマット
    df_results = pd.DataFrame(results)
    df_results.to_csv(
        output_file, index=False, float_format=f"%.{args.float_precision}f"
    )

    if args.dump_metrics:
        # 個別metrics保存
        baseline_json = output_dir / f"baseline_{date_str}.json"
        with open(baseline_json, "w") as f:
            json.dump(baseline_metrics, f, indent=2)
    logging.info(f"Results saved to {output_file}")
    logging.info("\nTop 5 most impactful features (by delta_sharpe_like):")
    ablation_results = df_results[df_results["feature"] != "baseline"].sort_values(
        "delta_sharpe_like", ascending=False
    )
    for _, row in ablation_results.head(5).iterrows():
        logging.info(
            f"{row['feature']}: {row['delta_sharpe_like']:.{args.float_precision}f}"
        )
        logging.info(
            f"{row['feature']}: {row['delta_sharpe_like']:.{args.float_precision}f}"
        )


if __name__ == "__main__":
    main()
