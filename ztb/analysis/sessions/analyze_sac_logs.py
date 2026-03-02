"""
SAC訓練ログ分析スクリプト
TensorBoardログから詳細なメトリクスを抽出・分析
"""
import os
from typing import Any

from tensorboard.backend.event_processing import event_accumulator
from ztb.io.json_io import write_json

def analyze_sac_logs(
    log_path: str | None = None,
    session_id: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any] | None:
    """
    TensorBoardログを分析

    Args:
        log_path: Path to log directory
        session_id: Session ID to analyze
        output_path: Path to save results

    Returns:
        Analysis results dictionary
    """
    if log_path is None:
        log_path = "logs"

    print(f"📊 Analyzing logs in: {log_path}")
    print("=" * 80)

    if not os.path.exists(log_path):
        print(f"❌ Log directory not found: {log_path}")
        return None

    # Find SAC sessions
    sac_sessions = sorted([d for d in os.listdir(log_path) if d.startswith("SAC_")])
    if not sac_sessions:
        print("❌ No SAC sessions found")
        return None

    # Use specified session or latest
    target_session = session_id if session_id else sac_sessions[-1]
    session_path = os.path.join(log_dir, latest_session)

    print(f"📁 Latest session: {latest_session}")
    print(f"📂 Path: {session_path}")
    print()

    # Load TensorBoard events
    ea = event_accumulator.EventAccumulator(session_path)
    ea.Reload()

    print("📈 Available scalar tags:")
    tags = ea.Tags()
    scalar_tags: list[str] = tags.get("scalars", [])
    for i, tag in enumerate(scalar_tags, 1):
        print(f"  {i}. {tag}")
    print()

    # Analyze key metrics
    metrics_to_analyze = [
        "train/ent_coef",
        "train/ent_coef_loss",
        "train/actor_loss",
        "train/critic_loss",
        "train/learning_rate",
        "rollout/ep_len_mean",
        "rollout/ep_rew_mean",
    ]

    results = {}

    for tag in metrics_to_analyze:
        if tag in scalar_tags:
            events = ea.Scalars(tag)
            values = [e.value for e in events]
            steps = [e.step for e in events]

            if values:
                results[tag] = {
                    "initial": values[0],
                    "final": values[-1],
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "trend": "increasing" if values[-1] > values[0] else "decreasing",
                    "change_pct": ((values[-1] - values[0]) / abs(values[0]) * 100)
                    if values[0] != 0
                    else 0,
                    "values": values[:10]
                    if len(values) > 10
                    else values,  # First 10 values
                    "steps": steps[:10] if len(steps) > 10 else steps,
                }

                print(f"📊 {tag}:")
                print(f"   Initial: {values[0]:.4f}")
                print(f"   Final: {values[-1]:.4f}")
                print(f"   Min: {min(values):.4f}")
                print(f"   Max: {max(values):.4f}")
                print(f"   Mean: {results[tag]['mean']:.4f}")
                print(
                    f"   Trend: {results[tag]['trend']} ({results[tag]['change_pct']:.1f}%)"
                )
                print()

    # Specific analysis for entropy coefficient
    if "train/ent_coef" in results:
        print("🔥 Entropy Coefficient Analysis:")
        ent_coef_data = results["train/ent_coef"]
        print(f"   変化: {ent_coef_data['initial']:.2f} → {ent_coef_data['final']:.2f}")
        print(f"   変化率: +{ent_coef_data['change_pct']:.1f}%")

        if ent_coef_data["final"] > 3.0:
            print("   ⚠️  WARNING: エントロピー係数が非常に高い")
            print("   💡 推奨: target_entropyを調整する必要があるかもしれません")
        elif ent_coef_data["final"] < 0.5:
            print("   ⚠️  WARNING: エントロピー係数が低すぎる")
            print("   💡 推奨: 探索不足の可能性があります")
        else:
            print("   ✅ エントロピー係数は適切な範囲です")
        print()

    # Loss analysis
    if "train/critic_loss" in results:
        print("📉 Critic Loss Analysis:")
        critic_loss = results["train/critic_loss"]
        print(f"   初期値: {critic_loss['initial']:.2e}")
        print(f"   最終値: {critic_loss['final']:.2e}")

        if critic_loss["final"] > 1e8:
            print("   ⚠️  WARNING: Critic Lossが非常に大きい")
            print("   💡 推奨:")
            print("      - learning_rateを下げる (0.0003 → 0.0001)")
            print("      - batch_sizeを増やす (64 → 128)")
            print("      - gammaを調整 (0.99 → 0.95)")
        elif critic_loss["trend"] == "decreasing":
            print("   ✅ Critic Lossは減少傾向です")
        print()

    if "train/actor_loss" in results:
        print("🎭 Actor Loss Analysis:")
        actor_loss = results["train/actor_loss"]
        print(f"   初期値: {actor_loss['initial']:.2e}")
        print(f"   最終値: {actor_loss['final']:.2e}")

        if actor_loss["final"] > 1e5:
            print("   ⚠️  Actor Lossが大きい")
            print("   💡 推奨: learning_rateを下げることを検討")
        print()

    # Save results to JSON
    output_file = "sac_log_analysis.json"
    write_json(output_file, results, indent=2, ensure_ascii=False)

    print(f"💾 Analysis saved to: {output_file}")
    print("=" * 80)

    return results

if __name__ == "__main__":
    log_dir = "checkpoints/sac_session"
    analyze_sac_logs(log_dir)
