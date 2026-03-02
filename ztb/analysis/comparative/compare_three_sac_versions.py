"""
SAC 3バージョン比較
v395a (original) vs v395c (conservative) vs v395b (aggressive)
"""
import os

from tensorboard.backend.event_processing import event_accumulator

from ztb.io.json_io import write_json

def compare_three_versions(log_dir: str) -> None:
    """
    3つのSACバージョンを比較
    """
    print("📊 SAC Three-Way Comparison")
    print("=" * 100)

    versions = {
        5: "v395a (Original)",
        6: "v395c (Conservative)",
        7: "v395b (Aggressive)",
    }

    results = {}

    for session_id, name in versions.items():
        session_path = os.path.join(log_dir, f"SAC_{session_id}")

        if not os.path.exists(session_path):
            print(f"⚠️ {name} not found")
            continue

        print(f"\n📁 Loading {name}...")

        ea = event_accumulator.EventAccumulator(session_path)
        ea.Reload()

        scalar_tags = ea.Tags()["scalars"]
        session_results = {}

        metrics = [
            "train/ent_coef",
            "train/critic_loss",
            "train/actor_loss",
            "train/ent_coef_loss",
        ]

        for tag in metrics:
            if tag in scalar_tags:
                events = ea.Scalars(tag)
                values = [e.value for e in events]

                if values:
                    session_results[tag] = {
                        "initial": values[0],
                        "final": values[-1],
                        "max": max(values),
                        "min": min(values),
                        "mean": sum(values) / len(values),
                    }

        results[name] = session_results

    # Detailed comparison
    print("\n" + "=" * 100)
    print("📊 DETAILED COMPARISON")
    print("=" * 100)

    metrics_to_compare = {
        "train/ent_coef": "Entropy Coefficient",
        "train/critic_loss": "Critic Loss",
        "train/actor_loss": "Actor Loss",
        "train/ent_coef_loss": "Entropy Coefficient Loss",
    }

    for metric, name in metrics_to_compare.items():
        print(f"\n🔍 {name} ({metric}):")
        print("-" * 100)
        print(
            f"{'Version':<30} {'Initial':>15} {'Final':>15} {'Min':>15} {'Max':>15} {'Mean':>15}"
        )
        print("-" * 100)

        for version_name in versions.values():
            if version_name in results and metric in results[version_name]:
                data = results[version_name][metric]
                print(
                    f"{version_name:<30} {data['initial']:>15.4e} {data['final']:>15.4e} "
                    f"{data['min']:>15.4e} {data['max']:>15.4e} {data['mean']:>15.4e}"
                )

    # Winner analysis
    print("\n" + "=" * 100)
    print("🏆 WINNER ANALYSIS")
    print("=" * 100)

    # Best ent_coef (closest to 1.0-1.5 range)
    print("\n📈 Best Entropy Coefficient (target: 1.0-1.5):")
    print("  → エントロピー係数。1.0-1.5が理想範囲（探索と活用のバランス）")
    for version_name in versions.values():
        if version_name in results and "train/ent_coef" in results[version_name]:
            final_ent = results[version_name]["train/ent_coef"]["final"]
            distance_from_ideal = abs(final_ent - 1.25)
            print(
                f"  {version_name}: {final_ent:.2f} (distance from 1.25: {distance_from_ideal:.2f})"
            )

    # Best critic_loss (lowest final value)
    print("\n📉 Lowest Critic Loss (final value):")
    print("  → Critic損失。低いほど価値関数の予測精度が高い")
    critic_losses = []
    for version_name in versions.values():
        if version_name in results and "train/critic_loss" in results[version_name]:
            final_loss = results[version_name]["train/critic_loss"]["final"]
            critic_losses.append((version_name, final_loss))

    critic_losses.sort(key=lambda x: x[1])
    for i, (version_name, loss) in enumerate(critic_losses, 1):
        symbol = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        print(f"  {symbol} {version_name}: {loss:.4e}")

    # Best actor_loss
    print("\n🎭 Lowest Actor Loss (final value):")
    print("  → Actor損失。低いほど行動選択の最適化が進んでいる")
    actor_losses = []
    for version_name in versions.values():
        if version_name in results and "train/actor_loss" in results[version_name]:
            final_loss = results[version_name]["train/actor_loss"]["final"]
            actor_losses.append((version_name, final_loss))

    actor_losses.sort(key=lambda x: x[1])
    for i, (version_name, loss) in enumerate(actor_losses, 1):
        symbol = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        print(f"  {symbol} {version_name}: {loss:.4e}")

    # Overall recommendation
    print("\n" + "=" * 100)
    print("💡 RECOMMENDATIONS")
    print("=" * 100)

    # Calculate scores
    scores = dict.fromkeys(versions.values(), 0)

    # Score for ent_coef (closer to 1.25 is better)
    for version_name in versions.values():
        if version_name in results and "train/ent_coef" in results[version_name]:
            final_ent = results[version_name]["train/ent_coef"]["final"]
            if 1.0 <= final_ent <= 2.0:
                scores[version_name] += 2
            elif 0.5 <= final_ent <= 3.0:
                scores[version_name] += 1

    # Score for critic_loss ranking
    for i, (version_name, _) in enumerate(critic_losses):
        scores[version_name] += 3 - i  # 3 points for 1st, 2 for 2nd, 1 for 3rd

    # Score for actor_loss ranking
    for i, (version_name, _) in enumerate(actor_losses):
        scores[version_name] += 3 - i

    print("\n🏅 Overall Scores:")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for i, (version_name, score) in enumerate(sorted_scores, 1):
        symbol = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        print(f"  {symbol} {version_name}: {score} points")

    winner = sorted_scores[0][0]
    print(f"\n🎯 RECOMMENDED VERSION: {winner}")

    # Save results
    output_file = "sac_three_way_comparison.json"
    write_json(output_file, results, indent=2, ensure_ascii=False)

    print(f"\n💾 Detailed comparison saved to: {output_file}")
    print("=" * 100)

if __name__ == "__main__":
    log_dir = "checkpoints/sac_session"
    compare_three_versions(log_dir)
