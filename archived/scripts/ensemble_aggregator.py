#!/usr/bin/env python3
"""
アンサンブル集計ツール

複数のモデルの予測を集計し、最終的なアンサンブル予測を生成します。
Confidence-weighted votingをサポートします。

Usage:
    python scripts/ensemble_aggregator.py --model-dirs checkpoints/ensemble_A_100k_test/checkpoint_100000 checkpoints/ensemble_B_100k_test/checkpoint_100000 checkpoints/ensemble_C_100k_test/checkpoint_100000
    python scripts/ensemble_aggregator.py --model-dirs checkpoints/ensemble_*/checkpoint_100000 --method confidence_weighted
    python scripts/ensemble_aggregator.py --model-dirs checkpoints/ensemble_*/checkpoint_100000 --eval-data ml-dataset-enhanced.csv --n-eval 100
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.policy_utils import predict_with_masks


@dataclass
class ModelPrediction:
    """個別モデルの予測"""

    model_name: str
    action: int  # 0=BUY, 1=HOLD, 2=SELL
    action_probs: np.ndarray  # [P(BUY), P(HOLD), P(SELL)]
    confidence: float  # 最大確率
    value: float  # 価値関数の推定値


@dataclass
class EnsemblePrediction:
    """アンサンブル予測"""

    action: int
    action_probs: np.ndarray
    confidence: float
    individual_predictions: List[ModelPrediction] = field(default_factory=list)
    method: str = "majority_vote"

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "action": int(self.action),
            "action_probs": self.action_probs.tolist(),
            "confidence": float(self.confidence),
            "method": self.method,
            "individual_predictions": [
                {
                    "model_name": p.model_name,
                    "action": int(p.action),
                    "action_probs": p.action_probs.tolist(),
                    "confidence": float(p.confidence),
                    "value": float(p.value),
                }
                for p in self.individual_predictions
            ],
        }
        return result


class EnsembleAggregator:
    """アンサンブル集計器"""

    def __init__(
        self,
        model_paths: List[Path],
        method: str = "confidence_weighted",
        disqualification_threshold: float = 0.5,  # all-masked率の閾値
        min_sharpe_threshold: float = -2.0,  # 最低Sharpe ratio閾値
    ):
        """
        Args:
            model_paths: モデルファイル(.zip)のパスリスト
            method: 集計方法 ("majority_vote", "confidence_weighted", "soft_voting")
            disqualification_threshold: all-masked多発の閾値（この割合以上でweight=0）
            min_sharpe_threshold: 最低Sharpe ratio（これ以下でweight=0）
        """
        self.model_paths = model_paths
        self.method = method
        self.models = []
        self.model_weights = []  # confidence-weighted用の重み
        self.model_sharpe_ratios = []  # 各モデルのSharpe ratio
        self.model_masked_rates = []  # 各モデルのall-masked発生率
        self.disqualified_models = []  # 失格モデルのインデックス
        self.disqualification_threshold = disqualification_threshold
        self.min_sharpe_threshold = min_sharpe_threshold

        self._load_models()

    def _load_models(self):
        """モデル読み込み"""
        from stable_baselines3 import PPO

        print(f"📦 Loading {len(self.model_paths)} models...")

        for path in self.model_paths:
            try:
                model = PPO.load(path)
                self.models.append(model)
                self.model_weights.append(1.0)  # 初期重みは均等
                print(f"  ✅ Loaded: {path.parent.name}")
            except Exception as e:
                print(f"  ❌ Failed to load {path}: {e}")

        print(f"✅ Loaded {len(self.models)}/{len(self.model_paths)} models")
        print()

    def calibrate_weights(
        self,
        eval_env,
        n_episodes: int = 50,
        use_confidence_scaling: bool = True,
    ):
        """
        評価データで各モデルの重みを校正

        各モデルのSharpe ratio + 信頼度スケーリング + 失格モデル検出に基づいて重みを設定

        Args:
            eval_env: 評価環境
            n_episodes: 評価エピソード数
            use_confidence_scaling: 信頼度スケーリングを使用するか
        """
        print(f"🔧 Calibrating model weights with {n_episodes} episodes...")
        print(f"   - Confidence scaling: {use_confidence_scaling}")
        print(
            f"   - Disqualification threshold (all-masked): {self.disqualification_threshold:.1%}"
        )
        print(f"   - Min Sharpe threshold: {self.min_sharpe_threshold}")
        print()

        model_performances = []
        model_confidences = []
        model_masked_counts = []

        for i, model in enumerate(self.models):
            episode_rewards = []
            episode_confidences = []
            masked_episode_count = 0

            for _ in range(n_episodes):
                obs, _ = eval_env.reset()
                done = False
                episode_reward = 0.0
                step_confidences = []
                step_count = 0
                masked_step_count = 0

                while not done:
                    # 予測と信頼度取得 (using predict_with_masks for MaskablePPO support)
                    action, _ = predict_with_masks(
                        model, obs, eval_env, deterministic=True
                    )

                    # 行動確率を取得して信頼度計算
                    obs_tensor = model.policy.obs_to_tensor(obs)[0]
                    with model.policy.th.no_grad():
                        distribution = model.policy.get_distribution(obs_tensor)
                        action_probs = distribution.distribution.probs.cpu().numpy()[0]
                        confidence = float(np.max(action_probs))
                        step_confidences.append(confidence)

                    # all-masked検出（全行動が同じ確率 = マスクされている可能性）
                    if np.std(action_probs) < 1e-6:
                        masked_step_count += 1

                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    step_count += 1

                episode_rewards.append(episode_reward)
                episode_confidences.append(np.mean(step_confidences))

                # エピソードの半分以上がmaskedならカウント
                if masked_step_count / max(step_count, 1) > 0.5:
                    masked_episode_count += 1

            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            sharpe = mean_reward / std_reward if std_reward > 0 else 0.0
            mean_confidence = np.mean(episode_confidences)
            masked_rate = masked_episode_count / n_episodes

            model_performances.append(sharpe)
            model_confidences.append(mean_confidence)
            model_masked_counts.append(masked_rate)

            model_name = self.model_paths[i].parent.name
            print(f"  Model {i+1} ({model_name}):")
            print(f"    Sharpe: {sharpe:.4f}")
            print(f"    Confidence: {mean_confidence:.4f}")
            print(f"    All-masked rate: {masked_rate:.2%}")

        # 失格モデル検出
        self.model_sharpe_ratios = model_performances
        self.model_masked_rates = model_masked_counts
        self.disqualified_models = []

        for i, (sharpe, masked_rate) in enumerate(
            zip(model_performances, model_masked_counts)
        ):
            model_name = self.model_paths[i].parent.name

            # 失格条件1: Sharpeが閾値以下
            if sharpe < self.min_sharpe_threshold:
                self.disqualified_models.append(i)
                print(
                    f"  ⚠️  Model {i+1} ({model_name}) DISQUALIFIED: Sharpe {sharpe:.4f} < {self.min_sharpe_threshold}"
                )

            # 失格条件2: all-masked多発
            elif masked_rate >= self.disqualification_threshold:
                self.disqualified_models.append(i)
                print(
                    f"  ⚠️  Model {i+1} ({model_name}) DISQUALIFIED: All-masked {masked_rate:.1%} >= {self.disqualification_threshold:.1%}"
                )

        if self.disqualified_models:
            print(f"\n🚫 {len(self.disqualified_models)} model(s) disqualified")

        # 重み計算
        self.model_weights = []
        for i in range(len(self.models)):
            if i in self.disqualified_models:
                # 失格モデルはweight=0
                weight = 0.0
            else:
                # Sharpe ratioベースの重み
                sharpe = max(model_performances[i], 0.0)

                # 信頼度スケーリング（オプション）
                if use_confidence_scaling:
                    confidence = model_confidences[i]
                    # Sharpe × 信頼度でスケーリング
                    weight = sharpe * confidence
                else:
                    weight = sharpe

            self.model_weights.append(weight)

        # 正規化
        total = sum(self.model_weights)
        if total > 0:
            self.model_weights = [w / total for w in self.model_weights]
        else:
            # 全て失格の場合は均等重み（ただし警告）
            print(
                "⚠️  WARNING: All models disqualified or zero weight, using equal weights"
            )
            self.model_weights = [1.0 / len(self.models)] * len(self.models)

        print("\n📊 Calibrated weights:")
        for i, (path, weight) in enumerate(zip(self.model_paths, self.model_weights)):
            status = "❌ DISQUALIFIED" if i in self.disqualified_models else "✅"
            print(f"  {status} Model {i+1} ({path.parent.name}): {weight:.4f}")
        print()

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> EnsemblePrediction:
        """アンサンブル予測"""
        individual_predictions = []

        # 各モデルの予測を取得
        for i, model in enumerate(self.models):
            action, _ = model.predict(observation, deterministic=deterministic)

            # 行動確率を取得（PPOの場合）
            obs_tensor = model.policy.obs_to_tensor(observation)[0]
            with model.policy.th.no_grad():
                distribution = model.policy.get_distribution(obs_tensor)
                action_probs = distribution.distribution.probs.cpu().numpy()[0]

                # 価値関数
                value = model.policy.predict_values(obs_tensor)[0].cpu().numpy()[0]

            confidence = float(np.max(action_probs))

            model_name = self.model_paths[i].parent.name

            individual_predictions.append(
                ModelPrediction(
                    model_name=model_name,
                    action=int(action),
                    action_probs=action_probs,
                    confidence=confidence,
                    value=float(value),
                )
            )

        # 集計方法に応じて最終予測を計算
        if self.method == "majority_vote":
            final_action, final_probs, final_confidence = self._majority_vote(
                individual_predictions
            )

        elif self.method == "confidence_weighted":
            final_action, final_probs, final_confidence = self._confidence_weighted(
                individual_predictions
            )

        elif self.method == "soft_voting":
            final_action, final_probs, final_confidence = self._soft_voting(
                individual_predictions
            )

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return EnsemblePrediction(
            action=final_action,
            action_probs=final_probs,
            confidence=final_confidence,
            individual_predictions=individual_predictions,
            method=self.method,
        )

    def _majority_vote(
        self, predictions: List[ModelPrediction]
    ) -> Tuple[int, np.ndarray, float]:
        """多数決"""
        actions = [p.action for p in predictions]

        # 最頻値
        unique, counts = np.unique(actions, return_counts=True)
        final_action = int(unique[np.argmax(counts)])

        # 確率は投票比率
        total = len(predictions)
        final_probs = np.zeros(3)
        for action in actions:
            final_probs[action] += 1.0 / total

        final_confidence = float(np.max(final_probs))

        return final_action, final_probs, final_confidence

    def _confidence_weighted(
        self, predictions: List[ModelPrediction]
    ) -> Tuple[int, np.ndarray, float]:
        """信頼度重み付け投票"""
        # 各モデルの確率を重みで重み付けして平均
        final_probs = np.zeros(3)

        for pred, weight in zip(predictions, self.model_weights):
            final_probs += pred.action_probs * weight

        final_action = int(np.argmax(final_probs))
        final_confidence = float(np.max(final_probs))

        return final_action, final_probs, final_confidence

    def _soft_voting(
        self, predictions: List[ModelPrediction]
    ) -> Tuple[int, np.ndarray, float]:
        """ソフト投票（確率の平均）"""
        # 確率を平均
        final_probs = np.mean([p.action_probs for p in predictions], axis=0)
        final_action = int(np.argmax(final_probs))
        final_confidence = float(np.max(final_probs))

        return final_action, final_probs, final_confidence


def evaluate_ensemble(
    aggregator: EnsembleAggregator,
    eval_env,
    n_episodes: int = 100,
) -> Dict[str, Any]:
    """アンサンブルモデルを評価"""
    print(f"📊 Evaluating ensemble with {n_episodes} episodes...")

    episode_rewards = []
    episode_lengths = []
    action_counts = {0: 0, 1: 0, 2: 0}  # BUY, HOLD, SELL

    for episode in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0

        while not done:
            prediction = aggregator.predict(obs, deterministic=True)
            action = prediction.action

            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1
            action_counts[action] += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{n_episodes} completed")

    # メトリック計算
    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    sharpe = mean_reward / std_reward if std_reward > 0 else 0.0

    total_actions = sum(action_counts.values())
    action_distribution = {k: v / total_actions for k, v in action_counts.items()}

    results = {
        "n_episodes": n_episodes,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "sharpe_ratio": sharpe,
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "action_distribution": action_distribution,
    }

    print("\n✅ Evaluation complete:")
    print(f"   Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"   Sharpe ratio: {sharpe:.4f}")
    print(
        f"   Action distribution: BUY={action_distribution[0]:.2%}, HOLD={action_distribution[1]:.2%}, SELL={action_distribution[2]:.2%}"
    )
    print()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble aggregation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic ensemble with majority vote
  python scripts/ensemble_aggregator.py --model-dirs checkpoints/ensemble_A_100k_test/checkpoint_100000 checkpoints/ensemble_B_100k_test/checkpoint_100000 checkpoints/ensemble_C_100k_test/checkpoint_100000

  # Confidence-weighted voting
  python scripts/ensemble_aggregator.py --model-dirs checkpoints/ensemble_*/checkpoint_100000 --method confidence_weighted

  # With evaluation and weight calibration
  python scripts/ensemble_aggregator.py --model-dirs checkpoints/ensemble_*/checkpoint_100000 --eval-data ml-dataset-enhanced.csv --calibrate --n-eval 100
        """,
    )

    parser.add_argument(
        "--model-dirs",
        nargs="+",
        required=True,
        help="Checkpoint directories containing model.zip files",
    )
    parser.add_argument(
        "--method",
        choices=["majority_vote", "confidence_weighted", "soft_voting"],
        default="confidence_weighted",
        help="Aggregation method (default: confidence_weighted)",
    )
    parser.add_argument(
        "--eval-data",
        help="Evaluation data path (required for calibration and evaluation)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Calibrate model weights using evaluation data",
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=100,
        help="Number of evaluation episodes (default: 100)",
    )
    parser.add_argument("--output", help="Output JSON file for results")

    args = parser.parse_args()

    # モデルパスを構築
    model_paths = []
    for dir_path in args.model_dirs:
        dir_path = Path(dir_path)
        model_file = dir_path / "model.zip"
        if model_file.exists():
            model_paths.append(model_file)
        else:
            print(f"⚠️  Model not found: {model_file}")

    if not model_paths:
        print("❌ No valid model files found")
        sys.exit(1)

    # アンサンブル集計器作成
    aggregator = EnsembleAggregator(
        model_paths=model_paths,
        method=args.method,
    )

    results = {
        "method": args.method,
        "n_models": len(model_paths),
        "model_paths": [str(p) for p in model_paths],
    }

    # 評価データがある場合
    if args.eval_data:
        from ztb.trading.env.zaif_env import ZaifEnv

        eval_env = ZaifEnv(
            data_path=args.eval_data,
            transaction_cost=0.001,
            max_position_size=1.0,
        )

        # 重み校正
        if args.calibrate:
            aggregator.calibrate_weights(eval_env, n_episodes=50)
            results["model_weights"] = aggregator.model_weights

        # 評価
        eval_results = evaluate_ensemble(
            aggregator,
            eval_env,
            n_episodes=args.n_eval,
        )

        results["evaluation"] = eval_results

    # 結果保存
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()
