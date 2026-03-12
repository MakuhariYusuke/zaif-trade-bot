"""
SAC v396 50k訓練の統計的収束検証

TensorBoardログから学習曲線を抽出し、以下の統計的検定を実施:
1. KPSS定常性検定 (Stationarity Test)
2. Mann-Kendall傾向検定 (Trend Test)
3. 変動係数 (Coefficient of Variation)
4. 改善率の計算
5. 収束ポイントの特定
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ztb.metrics.metrics import coefficient_of_variation
from ztb.utils.system_utils import check_library_availability, safe_import

# TensorBoard の利用可能性チェックとインポート
TENSORBOARD_AVAILABLE = check_library_availability(
    "tensorboard.backend.event_processing", "TensorBoard event processing"
)
event_accumulator = (
    safe_import(
        "tensorboard.backend.event_processing.event_accumulator",
        "TensorBoard event accumulator",
    )
    if TENSORBOARD_AVAILABLE
    else None
)


def extract_scalar_from_tensorboard(log_dir: Path, tag: str) -> List[Tuple[int, float]]:
    """TensorBoardログからスカラー値を抽出"""
    if not TENSORBOARD_AVAILABLE or event_accumulator is None:
        return []

    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()

    if tag not in ea.Tags()["scalars"]:
        print(f"❌ Tag '{tag}' not found in TensorBoard logs")
        return []

    events = ea.Scalars(tag)
    return [(e.step, e.value) for e in events]


def kpss_test(data: np.ndarray) -> Tuple[float, float, bool]:
    """KPSS定常性検定"""
    try:
        from statsmodels.tsa.stattools import kpss

        result = kpss(data, regression="c", nlags="auto")
        statistic = result[0]
        p_value = result[1]
        is_stationary = p_value > 0.05

        return statistic, p_value, is_stationary
    except Exception as e:
        print(f"⚠️ KPSS test failed: {e}")
        return 0.0, 0.0, False


def mann_kendall_test(data: np.ndarray) -> Tuple[float, float, bool]:
    """Mann-Kendall傾向検定"""
    try:
        from scipy.stats import kendalltau

        n = len(data)
        steps = np.arange(n)
        tau, p_value = kendalltau(steps, data)
        no_trend = p_value > 0.05

        return tau, p_value, no_trend
    except Exception as e:
        print(f"⚠️ Mann-Kendall test failed: {e}")
        return 0.0, 0.0, False


def find_convergence_point(
    losses: List[float], window_size: int = 1000, improvement_threshold: float = 0.01
) -> int:
    """収束ポイントを特定 (改善率が閾値以下になる点)"""
    losses_array = np.array(losses)

    for i in range(window_size, len(losses_array)):
        window_start = i - window_size
        recent_min = np.min(losses_array[window_start:i])
        previous_min = np.min(losses_array[:window_start])

        if previous_min == 0:
            continue

        improvement = (previous_min - recent_min) / previous_min

        if improvement < improvement_threshold:
            return i

    return len(losses_array)


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """移動平均計算"""
    return np.convolve(data, np.ones(window) / window, mode="valid")


def analyze_convergence(log_dir: Path, output_path: Path):
    """収束分析の実行"""

    print("=" * 80)
    print("  SAC v396 50k訓練 - 統計的収束検証")
    print("=" * 80)
    print()

    # Critic Loss抽出
    print("📊 TensorBoardログからCritic Lossを抽出中...")
    critic_loss_data = extract_scalar_from_tensorboard(log_dir, "train/critic_loss")

    if not critic_loss_data:
        print("❌ Critic Lossデータが見つかりません")
        return

    steps = np.array([s for s, _ in critic_loss_data])
    critic_losses = np.array([v for _, v in critic_loss_data])

    print(f"✅ {len(critic_losses)}個のデータポイント抽出完了")
    print()

    # 基本統計
    print("=" * 80)
    print("  基本統計")
    print("=" * 80)
    print(f"データポイント数: {len(critic_losses):,}")
    print(f"最小値: {np.min(critic_losses):.6f}")
    print(f"最大値: {np.max(critic_losses):.6f}")
    print(f"平均: {np.mean(critic_losses):.6f}")
    print(f"標準偏差: {np.std(critic_losses):.6f}")
    print(f"中央値: {np.median(critic_losses):.6f}")
    print()

    # 各区間の分析
    print("=" * 80)
    print("  区間別分析")
    print("=" * 80)

    intervals = [
        ("0-10k", 0, 10000),
        ("10k-25k", 10000, 25000),
        ("25k-40k", 25000, 40000),
        ("40k-50k", 40000, 50000),
        ("最後の5k", 45000, 50000),
        ("最後の2k", 48000, 50000),
    ]

    for name, start, end in intervals:
        mask = (steps >= start) & (steps < end)
        interval_losses = critic_losses[mask]

        if len(interval_losses) > 0:
            print(f"\n{name} steps:")
            print(f"  平均: {np.mean(interval_losses):.6f}")
            print(f"  標準偏差: {np.std(interval_losses):.6f}")
            print(f"  最小値: {np.min(interval_losses):.6f}")
            print(f"  CV: {coefficient_of_variation(interval_losses):.4f}")

    print()

    # 収束ポイント特定
    print("=" * 80)
    print("  収束ポイント特定")
    print("=" * 80)

    convergence_point = find_convergence_point(
        critic_losses.tolist(), window_size=1000, improvement_threshold=0.01
    )

    print(f"収束ポイント (改善率<1%): {convergence_point:,} steps")
    print(
        f"収束時のCritic Loss: {critic_losses[min(convergence_point, len(critic_losses)-1)]:.6f}"
    )
    print()

    # 統計的検定 (最後の10,000ステップ)
    print("=" * 80)
    print("  統計的検定 (最後の10,000ステップ)")
    print("=" * 80)

    last_10k_mask = steps >= 40000
    last_10k_losses = critic_losses[last_10k_mask]

    print(f"\nデータポイント数: {len(last_10k_losses):,}")
    print()

    # 1. KPSS定常性検定
    print("1. KPSS定常性検定:")
    kpss_stat, kpss_pval, is_stationary = kpss_test(last_10k_losses)
    print(f"   統計量: {kpss_stat:.4f}")
    print(f"   p値: {kpss_pval:.4f}")
    print(
        f"   結果: {'✅ 定常性あり (収束)' if is_stationary else '⚠️ 定常性なし (未収束)'}"
    )
    print()

    # 2. Mann-Kendall傾向検定 (最後の5,000ステップ)
    print("2. Mann-Kendall傾向検定 (最後の5,000ステップ):")
    last_5k_mask = steps >= 45000
    last_5k_losses = critic_losses[last_5k_mask]

    mk_tau, mk_pval, no_trend = mann_kendall_test(last_5k_losses)
    print(f"   Kendall's tau: {mk_tau:.4f}")
    print(f"   p値: {mk_pval:.4f}")
    print(f"   結果: {'✅ 傾向なし (収束)' if no_trend else '⚠️ 傾向あり (未収束)'}")
    print()

    # 3. 変動係数 (最後の1,000ステップ)
    print("3. 変動係数 (最後の1,000ステップ):")
    last_1k_mask = steps >= 49000
    last_1k_losses = critic_losses[last_1k_mask]

    cv = coefficient_of_variation(last_1k_losses)
    print(f"   CV: {cv:.4f}")

    if cv < 0.05:
        cv_status = "✅ 優秀 (CV < 0.05)"
    elif cv < 0.10:
        cv_status = "✅ 良好 (CV < 0.10)"
    elif cv < 0.20:
        cv_status = "⚠️ 許容範囲 (CV < 0.20)"
    else:
        cv_status = "❌ 変動大 (CV > 0.20)"

    print(f"   結果: {cv_status}")
    print()

    # 4. 改善率計算
    print("4. 改善率分析:")

    # 各区間の最小値
    early_min = np.min(critic_losses[steps < 10000])
    mid_min = np.min(critic_losses[(steps >= 10000) & (steps < 30000)])
    late_min = np.min(critic_losses[steps >= 30000])
    final_min = np.min(last_1k_losses)

    print(f"   0-10k最小値: {early_min:.6f}")
    print(f"   10k-30k最小値: {mid_min:.6f}")
    print(f"   30k-50k最小値: {late_min:.6f}")
    print(f"   最後の1k最小値: {final_min:.6f}")
    print()

    improvement_early_to_mid = (early_min - mid_min) / early_min * 100
    improvement_mid_to_late = (mid_min - late_min) / mid_min * 100

    print(f"   0-10k → 10k-30k改善率: {improvement_early_to_mid:.2f}%")
    print(f"   10k-30k → 30k-50k改善率: {improvement_mid_to_late:.2f}%")
    print()

    # 総合判定
    print("=" * 80)
    print("  総合判定")
    print("=" * 80)
    print()

    convergence_score = 0
    max_score = 4

    if is_stationary:
        convergence_score += 1
        print("✅ 定常性: 合格")
    else:
        print("⚠️ 定常性: 不合格")

    if no_trend:
        convergence_score += 1
        print("✅ 傾向なし: 合格")
    else:
        print("⚠️ 傾向あり: 不合格")

    if cv < 0.10:
        convergence_score += 1
        print("✅ 変動小: 合格 (CV < 0.10)")
    else:
        print("⚠️ 変動大: 不合格 (CV >= 0.10)")

    if final_min < 0.001:
        convergence_score += 1
        print("✅ 絶対値: 合格 (< 0.001)")
    else:
        print("⚠️ 絶対値: 不合格 (>= 0.001)")

    print()
    print(f"収束スコア: {convergence_score}/{max_score}")
    print()

    if convergence_score >= 3:
        print("🎉 結論: 統計的に収束していると判定されます!")
        print(f"   収束ポイント: {convergence_point:,} steps付近")
        print(f"   最終Critic Loss: {final_min:.6f}")
    else:
        print("⚠️ 結論: 完全な収束には至っていない可能性があります")
        print("   さらなる訓練を推奨します")

    print()

    # 結果をJSON保存
    result = {
        "total_steps": int(len(critic_losses)),
        "final_critic_loss": float(critic_losses[-1]),
        "min_critic_loss": float(np.min(critic_losses)),
        "mean_critic_loss": float(np.mean(critic_losses)),
        "std_critic_loss": float(np.std(critic_losses)),
        "convergence_point": int(convergence_point),
        "statistical_tests": {
            "kpss": {
                "statistic": float(kpss_stat),
                "p_value": float(kpss_pval),
                "is_stationary": bool(is_stationary),
            },
            "mann_kendall": {
                "tau": float(mk_tau),
                "p_value": float(mk_pval),
                "no_trend": bool(no_trend),
            },
            "cv_last_1k": float(cv),
        },
        "interval_analysis": {
            "0_10k": {
                "mean": float(np.mean(critic_losses[steps < 10000])),
                "min": float(early_min),
            },
            "10k_30k": {
                "mean": float(
                    np.mean(critic_losses[(steps >= 10000) & (steps < 30000)])
                ),
                "min": float(mid_min),
            },
            "30k_50k": {
                "mean": float(np.mean(critic_losses[steps >= 30000])),
                "min": float(late_min),
            },
        },
        "convergence_score": convergence_score,
        "convergence_achieved": convergence_score >= 3,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"📝 結果を保存: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    log_dir = project_root / "checkpoints" / "sac_session" / "SAC_28"
    output_path = (
        project_root / "checkpoints" / "sac_session" / "convergence_analysis.json"
    )

    if not log_dir.exists():
        print(f"❌ ログディレクトリが見つかりません: {log_dir}")
        sys.exit(1)

    analyze_convergence(log_dir, output_path)
