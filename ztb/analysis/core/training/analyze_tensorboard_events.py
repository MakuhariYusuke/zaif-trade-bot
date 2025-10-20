"""
TensorBoardイベントファイル解析スクリプト
v395g（失敗）とv395i（成功）の違いを徹底的に分析
"""
import json
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing import event_accumulator


def analyze_tensorboard_events(
    event_file_path: str, session_name: str
) -> dict[str, Any]:
    """TensorBoardイベントを解析"""
    print(f"\n{'='*80}")
    print(f"分析中: {session_name}")
    print(f"ファイル: {event_file_path}")
    print(f"{'='*80}\n")

    # EventAccumulatorを作成
    ea = event_accumulator.EventAccumulator(str(event_file_path))
    ea.Reload()

    # 利用可能なタグを取得
    print("【利用可能なスカラー】")
    scalar_tags = ea.Tags()["scalars"]
    for tag in scalar_tags:
        print(f"  - {tag}")

    # 主要メトリクスを抽出
    metrics = {}
    key_metrics = [
        "train/critic_loss",
        "train/actor_loss",
        "train/ent_coef",
        "train/ent_coef_loss",
        "train/learning_rate",
        "train/n_updates",
        "rollout/ep_rew_mean",
        "rollout/ep_len_mean",
    ]

    for metric in key_metrics:
        if metric in scalar_tags:
            events = ea.Scalars(metric)
            values = [(e.step, e.value) for e in events]
            metrics[metric] = values

            if values:
                print(f"\n【{metric}】")
                print(f"  初期値: {values[0][1]:.6f} (step {values[0][0]})")
                print(f"  最終値: {values[-1][1]:.6f} (step {values[-1][0]})")

                # 統計情報
                vals = [v[1] for v in values]
                print(f"  最小値: {min(vals):.6f}")
                print(f"  最大値: {max(vals):.6f}")
                print(f"  平均値: {sum(vals)/len(vals):.6f}")

                # メトリクスごとの説明
                if "critic_loss" in metric:
                    print("  → Critic損失。低いほど価値関数の予測精度が高い")
                elif "actor_loss" in metric:
                    print("  → Actor損失。低いほど行動選択の最適化が進んでいる")
                elif "ent_coef" in metric:
                    print("  → エントロピー係数。探索と活用のバランスを制御")
                elif "ep_rew_mean" in metric:
                    print("  → エピソード平均報酬。高いほどパフォーマンスが良い")
                elif "ep_len_mean" in metric:
                    print("  → エピソード平均長。安定した取引を示す")

                # 詳細データ（最初の5つと最後の5つ）
                print(f"  データ点数: {len(values)}")
                print("  最初の5点:")
                for step, val in values[:5]:
                    print(f"    Step {step}: {val:.6f}")
                if len(values) > 5:
                    print("  最後の5点:")
                    for step, val in values[-5:]:
                        print(f"    Step {step}: {val:.6f}")

    return metrics


def compare_sessions(
    metrics1: dict[str, Any], metrics2: dict[str, Any], name1: str, name2: str
) -> None:
    """2つのセッションを比較"""
    print(f"\n{'='*80}")
    print(f"比較: {name1} vs {name2}")
    print(f"{'='*80}\n")

    common_metrics = set(metrics1.keys()) & set(metrics2.keys())

    for metric in sorted(common_metrics):
        vals1 = [v[1] for v in metrics1[metric]]
        vals2 = [v[1] for v in metrics2[metric]]

        if vals1 and vals2:
            avg1 = sum(vals1) / len(vals1)
            avg2 = sum(vals2) / len(vals2)

            print(f"\n【{metric}】")
            print(
                f"  {name1}: 平均 {avg1:.6f}, 範囲 [{min(vals1):.6f}, {max(vals1):.6f}]"
            )
            print(
                f"  {name2}: 平均 {avg2:.6f}, 範囲 [{min(vals2):.6f}, {max(vals2):.6f}]"
            )

            if avg1 != 0:
                improvement = ((avg2 - avg1) / abs(avg1)) * 100
                print(f"  改善率: {improvement:+.2f}%")

                if "loss" in metric.lower():
                    if avg2 < avg1:
                        print(f"  ✅ {name2}の方が良い（損失が低い）")
                    else:
                        print(f"  ❌ {name1}の方が良い（損失が低い）")


def main() -> None:
    # SAC_11 (v395g) - 失敗版
    sac11_path = Path("checkpoints/sac_session/SAC_11")
    sac11_event = list(sac11_path.glob("events.out.tfevents.*"))[0]

    # SAC_13 (v395i) - 成功版
    sac13_path = Path("checkpoints/sac_session/SAC_13")
    sac13_event = list(sac13_path.glob("events.out.tfevents.*"))[0]

    # 各セッションを解析
    print("\n" + "=" * 80)
    print("TensorBoard イベント解析")
    print("=" * 80)

    metrics_v395g = analyze_tensorboard_events(
        str(sac11_event), "v395g (SAC_11) - 観測値正規化なし"
    )
    metrics_v395i = analyze_tensorboard_events(
        str(sac13_event), "v395i (SAC_13) - 観測値正規化あり"
    )

    # 比較
    compare_sessions(metrics_v395g, metrics_v395i, "v395g", "v395i")

    # 結果をJSONに保存
    result = {
        "v395g": {metric: values for metric, values in metrics_v395g.items()},
        "v395i": {metric: values for metric, values in metrics_v395i.items()},
    }

    with open("sac_session_comparison.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print("結果を sac_session_comparison.json に保存しました")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
