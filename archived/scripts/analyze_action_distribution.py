"""
学習結果分析スクリプト - アクション分布の問題診断

最新の学習セッションからアクション分布とメトリクスを抽出・分析します。
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

# TensorBoard event file reader
try:
    from tensorboard.backend.event_processing import event_accumulator

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️ TensorBoard not available. Install with: pip install tensorboard")


class MetricValue(TypedDict, total=False):
    """Single metric value with step information."""

    step: int
    value: float


class ActionDistributionResults(TypedDict, total=False):
    """Results from extracting action distribution from TensorBoard events."""

    action_distribution: Dict[str, MetricValue]
    diversity_metrics: Dict[str, MetricValue]
    training_metrics: Dict[str, MetricValue]
    reward_metrics: Dict[str, MetricValue]


class ActionDistributionAnalysis(TypedDict, total=False):
    """Analysis results of action distribution problems."""

    issues: List[str]
    recommendations: List[str]
    severity: str
    status: str
    sell_rate: float
    entropy: float
    gini_coefficient: float
    lambda_value: float
    action_balance_score: float


def find_latest_checkpoint_dir(base_dir: str = "checkpoints") -> Optional[Path]:
    """最新のチェックポイントディレクトリを見つける"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return None

    # タイムスタンプでソート
    subdirs = [
        d for d in base_path.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]
    if not subdirs:
        return None

    # 最新のディレクトリを取得（modified time基準）
    latest_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
    print(f"📁 Latest checkpoint directory: {latest_dir}")
    return latest_dir


def find_event_files(checkpoint_dir: Path) -> List[Path]:
    """TensorBoard event fileを見つける"""
    event_files = list(checkpoint_dir.glob("events.out.tfevents.*"))
    print(f"📊 Found {len(event_files)} event files")
    return event_files


def extract_action_distribution_from_events(
    event_file: Path,
) -> ActionDistributionResults:
    """TensorBoardイベントファイルからアクション分布を抽出"""
    if not TENSORBOARD_AVAILABLE:
        return {}

    print(f"\n🔍 Analyzing event file: {event_file.name}")

    ea = event_accumulator.EventAccumulator(str(event_file))
    ea.Reload()

    # 利用可能なタグを確認
    tags = ea.Tags()
    scalar_tags_raw = tags.get("scalars", [])
    scalar_tags: List[str] = (
        scalar_tags_raw if isinstance(scalar_tags_raw, list) else []
    )
    print(f"📈 Available scalar tags: {len(scalar_tags)}")

    results: ActionDistributionResults = {
        "action_distribution": {},
        "diversity_metrics": {},
        "training_metrics": {},
        "reward_metrics": {},
    }

    # アクション分布関連のメトリクスを抽出
    action_tags = [
        "rollout/action_dist_hold",
        "rollout/action_dist_buy",
        "rollout/action_dist_sell",
        "rollout/sell_rate",
        "diversity/action_entropy",
        "diversity/action_gini",
        "lagrange/lambda",
        "lagrange/sell_rate_error",
    ]

    for tag in action_tags:
        if tag in scalar_tags:
            events = ea.Scalars(tag)
            if events:
                # 最後の値を取得
                last_event = events[-1]
                results["action_distribution"][tag] = {
                    "step": last_event.step,
                    "value": last_event.value,
                }
                print(f"  ✓ {tag}: {last_event.value:.4f} (step {last_event.step})")

    # 学習メトリクスを抽出
    training_tags = [
        "train/learning_rate",
        "train/loss",
        "train/policy_loss",
        "train/value_loss",
        "train/entropy_loss",
        "rollout/ep_rew_mean",
        "rollout/ep_len_mean",
    ]

    print("\n📊 Training Metrics:")
    for tag in training_tags:
        if tag in scalar_tags:
            events = ea.Scalars(tag)
            if events:
                last_event = events[-1]
                results["training_metrics"][tag] = {
                    "step": last_event.step,
                    "value": last_event.value,
                }
                print(f"  ✓ {tag}: {last_event.value:.4f}")

    return results


def analyze_action_distribution(
    results: ActionDistributionResults,
) -> ActionDistributionAnalysis:
    """アクション分布を分析して問題を診断"""
    analysis: ActionDistributionAnalysis = {
        "issues": [],
        "recommendations": [],
        "severity": "unknown",
    }

    action_dist = results.get("action_distribution", {})

    # アクション分布の値を取得
    hold_rate = action_dist.get("rollout/action_dist_hold", {}).get("value")
    buy_rate = action_dist.get("rollout/action_dist_buy", {}).get("value")
    sell_rate = action_dist.get("rollout/action_dist_sell", {}).get("value")

    if hold_rate is None or buy_rate is None or sell_rate is None:
        analysis["issues"].append("❌ アクション分布データが見つかりません")
        analysis["status"] = "no_data"
        return analysis

    print("\n📊 Action Distribution Analysis:")
    print(f"  HOLD: {hold_rate*100:.2f}%")
    print(f"  BUY:  {buy_rate*100:.2f}%")
    print(f"  SELL: {sell_rate*100:.2f}%")

    # 理想的な分布（均等）
    ideal_rate = 1.0 / 3.0
    tolerance = 0.10  # ±10%

    # SELL bias問題のチェック
    if sell_rate > (ideal_rate + tolerance):
        analysis["issues"].append(
            f"🔴 SELL bias detected: {sell_rate*100:.2f}% (target: 33.33%)"
        )
        analysis["status"] = "sell_bias"
        analysis["recommendations"].append(
            "Increase Lagrange constraint strength (lagrange_eta)"
        )
        analysis["recommendations"].append("Enable PAN (Per-Action Normalization)")
        analysis["recommendations"].append("Check reward function for SELL favoritism")

    # HOLD偏重問題のチェック
    if hold_rate > (ideal_rate + tolerance * 2):
        analysis["issues"].append(
            f"🟡 HOLD bias detected: {hold_rate*100:.2f}% (target: 33.33%)"
        )
        analysis["status"] = "hold_bias"
        analysis["recommendations"].append("Increase entropy coefficient (ent_coef)")
        analysis["recommendations"].append("Reduce action penalties")
        analysis["recommendations"].append(
            "Check if min_holding_period is too restrictive"
        )

    # BUY偏重問題のチェック
    if buy_rate > (ideal_rate + tolerance):
        analysis["issues"].append(
            f"🟠 BUY bias detected: {buy_rate*100:.2f}% (target: 33.33%)"
        )
        analysis["status"] = "buy_bias"
        analysis["recommendations"].append("Check reward function for BUY favoritism")
        analysis["recommendations"].append(
            "Verify transaction costs are applied correctly"
        )

    # 極端な不均衡のチェック
    max_rate = max(hold_rate, buy_rate, sell_rate)
    min_rate = min(hold_rate, buy_rate, sell_rate)
    imbalance_ratio = max_rate / (min_rate + 1e-6)

    if imbalance_ratio > 5.0:
        analysis["issues"].append(
            f"🔴 Severe action imbalance: {imbalance_ratio:.2f}x ratio"
        )
        analysis["status"] = "severe_imbalance"
        analysis["recommendations"].append("Consider forced curriculum learning stage")
        analysis["recommendations"].append("Enable stratified sampling")

    # エントロピーのチェック
    entropy = action_dist.get("diversity/action_entropy", {}).get("value")
    if entropy is not None:
        print(f"  Entropy: {entropy:.4f}")
        # 3アクションの最大エントロピーは log(3) ≈ 1.099
        max_entropy = 1.099
        if entropy < max_entropy * 0.7:
            analysis["issues"].append(
                f"🟡 Low action entropy: {entropy:.4f} (max: {max_entropy:.4f})"
            )
            analysis["recommendations"].append(
                "Increase entropy coefficient (ent_coef)"
            )

    # Lagrange multiplierのチェック
    lambda_val = action_dist.get("lagrange/lambda", {}).get("value")
    if lambda_val is not None:
        print(f"  Lagrange λ: {lambda_val:.4f}")
        if lambda_val >= 1.9:  # lambda_max = 2.0に近い
            analysis["issues"].append(
                f"⚠️ Lagrange constraint saturated: λ={lambda_val:.4f}"
            )
            analysis["recommendations"].append("Increase lagrange_lambda_max")
            analysis["recommendations"].append(
                "Increase lagrange_eta for faster adaptation"
            )

    # ステータスが未設定の場合
    if not analysis["issues"]:
        analysis["status"] = "balanced"
        print("\n✅ Action distribution is reasonably balanced!")

    return analysis


def print_recommendations(analysis: ActionDistributionAnalysis):
    """推奨事項を表示"""
    if not analysis.get("recommendations"):
        return

    print("\n" + "=" * 60)
    print("🔧 Recommended Actions:")
    print("=" * 60)

    for i, rec in enumerate(analysis.get("recommendations", []), 1):
        print(f"{i}. {rec}")

    print("\n📝 Configuration Changes to Try:")

    status = analysis.get("status", "")
    if "sell_bias" in status:
        print(
            """
{
  "lagrange_eta": 0.1,  // Increase from 0.05
  "lagrange_lambda_max": 3.0,  // Increase from 2.0
  "enable_pan": true,
  "enable_probes": true,
  "ent_coef": 0.15  // Increase from 0.1
}
"""
        )

    if "hold_bias" in status:
        print(
            """
{
  "ent_coef": 0.2,  // Increase from 0.1
  "reward_settings": {
    "action_penalty_scale": 0.005  // Reduce from 0.01
  }
}
"""
        )

    if "severe_imbalance" in status:
        print(
            """
{
  "curriculum_stage": "forced_balance",
  "enable_stratified_sampling": true,
  "enable_forced_diversity": true,
  "ent_coef": 0.2
}
"""
        )


def main() -> None:
    """メイン分析実行"""
    print("=" * 60)
    print("📊 Training Results Analysis - Action Distribution")
    print("=" * 60)

    # 最新のチェックポイントディレクトリを見つける
    checkpoint_dir = find_latest_checkpoint_dir()
    if checkpoint_dir is None:
        print("❌ No checkpoint directories found")
        return

    # イベントファイルを見つける
    event_files = find_event_files(checkpoint_dir)
    if not event_files:
        print("❌ No TensorBoard event files found")
        print("\n💡 Alternative: Check training logs or model metadata")
        return

    # 最新のイベントファイルを分析
    latest_event_file = max(event_files, key=lambda f: f.stat().st_mtime)

    if not TENSORBOARD_AVAILABLE:
        print("\n⚠️ TensorBoard not available for automatic analysis")
        print("Please install with: pip install tensorboard")
        print("\nOr manually check TensorBoard:")
        print(f"  tensorboard --logdir {checkpoint_dir}")
        return

    # イベントファイルから情報を抽出
    results = extract_action_distribution_from_events(latest_event_file)

    # アクション分布を分析
    analysis = analyze_action_distribution(results)

    # 問題点を表示
    if analysis.get("issues"):
        print("\n" + "=" * 60)
        print("⚠️ Issues Detected:")
        print("=" * 60)
        for issue in analysis.get("issues", []):
            print(f"  {issue}")

    # 推奨事項を表示
    print_recommendations(analysis)

    # 結果をJSONで保存
    output_file = checkpoint_dir / "action_distribution_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint_dir": str(checkpoint_dir),
                "results": results,
                "analysis": analysis,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n💾 Analysis saved to: {output_file}")


if __name__ == "__main__":
    main()
