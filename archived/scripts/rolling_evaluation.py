#!/usr/bin/env python3
"""
ローリング評価ツール

学習中のモデルを定期的に評価し、過学習を検出します。
1Mロングラン設計の早期停止条件（Sharpe_proxy）を検出します。

Usage:
    python scripts/rolling_evaluation.py --checkpoint-dir checkpoints/ensemble_B_100k_test --data-path ml-dataset-enhanced.csv
    python scripts/rolling_evaluation.py --checkpoint-dir checkpoints/ensemble_B_100k_test --data-path ml-dataset-enhanced.csv --n-episodes 50
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.policy_utils import predict_with_masks

# Import Long-Run evaluation constants
from ztb.training.ppo_config import (
    ROLLING_OOS_STEPS,
    SHARPE_PATIENCE_EVALS,
    SHARPE_PROXY_THRESHOLD,
)


@dataclass
class EvaluationResult:
    """評価結果"""

    checkpoint_name: str
    step: int
    timestamp: str
    metrics: Dict[str, float] = field(default_factory=dict)
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_checkpoint(
    checkpoint_path: Path,
    data_path: Path,
    n_episodes: int = 50,
    deterministic: bool = True,
) -> EvaluationResult:
    """チェックポイントを評価"""
    # 動的インポート（初回のみ）
    from stable_baselines3 import PPO

    from ztb.trading.env.zaif_env import ZaifEnv

    print(f"  Evaluating {checkpoint_path.name}...")

    # モデル読み込み
    model_path = checkpoint_path / "model.zip"
    if not model_path.exists():
        print(f"    ⚠️  Model not found: {model_path}")
        return None

    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"    ❌ Failed to load model: {e}")
        return None

    # 環境作成
    env = ZaifEnv(
        data_path=str(data_path),
        transaction_cost=0.001,
        max_position_size=1.0,
    )

    # 評価実行
    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0

        while not done:
            # Predict action (using predict_with_masks for MaskablePPO support)
            action, _ = predict_with_masks(model, obs, env, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    # メトリック計算
    metrics = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
    }

    # Sharpe ratio approximation
    if metrics["std_reward"] > 0:
        metrics["sharpe_ratio"] = metrics["mean_reward"] / metrics["std_reward"]
    else:
        metrics["sharpe_ratio"] = 0.0

    step = int(checkpoint_path.name.split("_")[-1])

    result = EvaluationResult(
        checkpoint_name=checkpoint_path.name,
        step=step,
        timestamp=datetime.now().isoformat(),
        metrics=metrics,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
    )

    print(
        f"    ✅ Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}"
    )

    return result


def rolling_evaluation(
    checkpoint_dir: Path,
    data_path: Path,
    n_episodes: int = 50,
    output_dir: Optional[Path] = None,
) -> List[EvaluationResult]:
    """ローリング評価を実行"""
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        sys.exit(1)

    # チェックポイント一覧
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*"))

    if not checkpoints:
        print(f"❌ No checkpoints found in {checkpoint_dir}")
        sys.exit(1)

    print(f"🔍 Found {len(checkpoints)} checkpoints")
    print(f"📊 Evaluating each with {n_episodes} episodes...")
    print()

    # 評価実行
    results = []

    for checkpoint in checkpoints:
        result = evaluate_checkpoint(
            checkpoint_path=checkpoint,
            data_path=data_path,
            n_episodes=n_episodes,
        )

        if result:
            results.append(result)

    # 結果保存
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON形式で保存
        output_file = output_dir / f"rolling_eval_{checkpoint_dir.name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                [r.to_dict() for r in results],
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"\n✅ Results saved to {output_file}")

    return results


def print_summary(results: List[EvaluationResult]):
    """サマリーを表示（1M早期停止条件も表示）"""
    if not results:
        print("❌ No results to display")
        return

    print("\n" + "=" * 100)
    print("📊 Rolling Evaluation Summary (1M Long-Run Design)")
    print("=" * 100)
    print()

    # ヘッダー
    print(
        f"{'Checkpoint':25s} {'Step':10s} {'Mean Reward':15s} {'Std':12s} {'Sharpe':12s}"
    )
    print("-" * 100)

    # 各チェックポイント
    best_sharpe = max(
        results, key=lambda r: r.metrics.get("sharpe_ratio", float("-inf"))
    )
    low_sharpe_streak = 0

    for result in results:
        mean_reward = result.metrics["mean_reward"]
        std_reward = result.metrics["std_reward"]
        sharpe = result.metrics.get("sharpe_ratio", 0.0)

        # ベストモデルにマーク
        marker = "⭐" if result == best_sharpe else "  "

        # 早期停止条件3: Sharpe_proxy ≤ 0 for 2 consecutive evals
        if sharpe <= SHARPE_PROXY_THRESHOLD:
            low_sharpe_streak += 1
            marker = "⚠️ "
        else:
            low_sharpe_streak = 0

        print(
            f"{marker} {result.checkpoint_name:23s} {result.step:10d} "
            f"{mean_reward:15.2f} {std_reward:12.2f} {sharpe:12.4f}"
        )

    print()
    print("=" * 100)
    print(f"\n⭐ Best Sharpe: {best_sharpe.checkpoint_name} (step {best_sharpe.step})")
    print(f"   Sharpe Ratio: {best_sharpe.metrics['sharpe_ratio']:.4f}")
    print(
        f"   Mean Reward: {best_sharpe.metrics['mean_reward']:.2f} ± {best_sharpe.metrics['std_reward']:.2f}"
    )
    print()

    # 早期停止条件チェック
    if low_sharpe_streak >= SHARPE_PATIENCE_EVALS:
        print("🚨 EARLY STOP CONDITION 3 DETECTED:")
        print(
            f"   Sharpe_proxy ≤ {SHARPE_PROXY_THRESHOLD} for {low_sharpe_streak} consecutive evaluations"
        )
        print(f"   (Threshold: {SHARPE_PATIENCE_EVALS} consecutive evals)")
        print()

    # 過学習検出
    if len(results) >= 3:
        # 最後の3つのチェックポイントで性能が悪化していないかチェック
        recent = results[-3:]
        rewards = [r.metrics["mean_reward"] for r in recent]

        if rewards[0] > rewards[-1]:
            degradation = (rewards[0] - rewards[-1]) / abs(rewards[0]) * 100
            print("⚠️  Potential overfitting detected:")
            print(
                f"   Performance degraded by {degradation:.1f}% in last 3 checkpoints"
            )
            print()

    # 設定情報表示
    print("ℹ️  Configuration:")
    print(f"   Rolling OOS steps: {ROLLING_OOS_STEPS}")
    print(f"   Sharpe threshold: {SHARPE_PROXY_THRESHOLD}")
    print(f"   Sharpe patience: {SHARPE_PATIENCE_EVALS} evaluations")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Rolling evaluation for checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all checkpoints
  python scripts/rolling_evaluation.py --checkpoint-dir checkpoints/ensemble_B_100k_test --data-path ml-dataset-enhanced.csv

  # Custom number of episodes
  python scripts/rolling_evaluation.py --checkpoint-dir checkpoints/ensemble_B_100k_test --data-path ml-dataset-enhanced.csv --n-episodes 100

  # Save results
  python scripts/rolling_evaluation.py --checkpoint-dir checkpoints/ensemble_B_100k_test --data-path ml-dataset-enhanced.csv --output-dir eval_results
        """,
    )

    parser.add_argument(
        "--checkpoint-dir", required=True, help="Path to checkpoint directory"
    )
    parser.add_argument("--data-path", required=True, help="Path to evaluation data")
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes per checkpoint (default: 50)",
    )
    parser.add_argument(
        "--output-dir", help="Output directory for results (default: no save)"
    )

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir) if args.output_dir else None

    # 評価実行
    results = rolling_evaluation(
        checkpoint_dir=checkpoint_dir,
        data_path=data_path,
        n_episodes=args.n_episodes,
        output_dir=output_dir,
    )

    # サマリー表示
    print_summary(results)


if __name__ == "__main__":
    main()
