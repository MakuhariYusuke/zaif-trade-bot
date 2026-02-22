#!/usr/bin/env python3
"""
Phase 3 Day 4-5: Single Experiment Analysis
最新のトレーニングレポートを分析して、検証可能なメトリクスを抽出
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import TypedDict

import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.v459.json_compat import write_json_compatible
from ztb.reporting.services.catalog import (
    extract_action_distribution_from_payload,
    get_recent_training_reports,
    load_training_report,
)
from ztb.utils.safety import ensure_dict, safe_to_bool, safe_to_float, safe_to_int


class ActionAnalysis(TypedDict):
    distribution_pct: dict[str, float]
    balance_score: float
    is_balanced: bool
    dominant_action: str
    dominant_pct: float


class PerformanceAnalysis(TypedDict):
    total_time_sec: float
    total_time_min: float
    timesteps: int
    steps_per_sec: float
    feature_time_sec: float
    training_time_sec: float
    feature_time_ratio_pct: float
    final_reward: float
    action_diversity: float


class StabilityAnalysis(TypedDict):
    entropy: float
    max_entropy: float
    normalized_entropy: float
    diversity_score_pct: float
    is_diverse: bool


def _sanitize_timestamp(raw_timestamp: object) -> str:
    if isinstance(raw_timestamp, str) and raw_timestamp:
        return raw_timestamp.replace(":", "").replace("-", "").replace(" ", "_")
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_action_distribution(report: dict[str, object]) -> dict[str, float]:
    dist = extract_action_distribution_from_payload(report)
    return {
        "HOLD": safe_to_float(dist.get("HOLD"), 0.0),
        "BUY": safe_to_float(dist.get("BUY"), 0.0),
        "SELL": safe_to_float(dist.get("SELL"), 0.0),
    }


def load_latest_report() -> dict[str, object]:
    """最新のトレーニングレポートをロード"""
    reports_dir = project_root / "reports"
    report_files = get_recent_training_reports(limit=1, reports_dir=reports_dir)
    
    if not report_files:
        raise FileNotFoundError("No training reports found")
    
    latest = report_files[0]
    print(f"📁 Loading: {latest.name}")

    report = load_training_report(latest)
    if report is None:
        raise ValueError(f"Invalid JSON report: {latest}")
    return report


def analyze_action_distribution(report: dict[str, object]) -> ActionAnalysis:
    """アクション分布の分析"""
    action_dist = _load_action_distribution(report)
    if not action_dist:
        return {
            "distribution_pct": {"HOLD": 0.0, "BUY": 0.0, "SELL": 0.0},
            "balance_score": 0.0,
            "is_balanced": False,
            "dominant_action": "N/A",
            "dominant_pct": 0.0,
        }
    
    # Convert to percentages
    action_pct = {k: safe_to_float(v, 0.0) * 100.0 for k, v in action_dist.items()}
    
    # Calculate balance (how evenly distributed)
    ideal = 100 / 3  # 33.33%
    balance_score = 100 - sum(abs(v - ideal) for v in action_pct.values()) / 2
    
    # Check if any action is overly dominant
    max_action = max(action_pct.items(), key=lambda x: x[1])
    is_balanced = max_action[1] < 50  # No action should be > 50%
    
    return {
        "distribution_pct": action_pct,
        "balance_score": balance_score,
        "is_balanced": is_balanced,
        "dominant_action": max_action[0],
        "dominant_pct": safe_to_float(max_action[1], 0.0),
    }


def analyze_performance(report: dict[str, object]) -> PerformanceAnalysis:
    """パフォーマンスメトリクスの分析"""
    stats = ensure_dict(report.get("training_stats"))
    perf = ensure_dict(report.get("performance_metrics"))
    
    # Training efficiency
    total_time = safe_to_float(stats.get("training_time"), 0.0)
    timesteps = safe_to_int(stats.get("total_timesteps"), 0)
    steps_per_sec = safe_to_float(stats.get("steps_per_second"), 0.0)
    
    # Time analysis
    feature_time = safe_to_float(stats.get("feature_generation_time_s"), 0.0)
    training_time = total_time - feature_time
    feature_ratio = feature_time / total_time * 100.0 if total_time > 0 else 0.0
    
    # Reward analysis
    final_reward = safe_to_float(stats.get("final_reward"), 0.0)
    
    return {
        "total_time_sec": total_time,
        "total_time_min": total_time / 60.0,
        "timesteps": timesteps,
        "steps_per_sec": steps_per_sec,
        "feature_time_sec": feature_time,
        "training_time_sec": training_time,
        "feature_time_ratio_pct": feature_ratio,
        "final_reward": final_reward,
        "action_diversity": safe_to_float(perf.get("action_diversity"), 0.0),
    }


def analyze_stability(report: dict[str, object]) -> StabilityAnalysis:
    """安定性の評価（単一実験からの推定）"""
    action_dist = _load_action_distribution(report)
    values = [v for v in action_dist.values() if v > 0]
    if not values:
        return {
            "entropy": 0.0,
            "max_entropy": 0.0,
            "normalized_entropy": 0.0,
            "diversity_score_pct": 0.0,
            "is_diverse": False,
        }
    
    # Entropy calculation (Shannon entropy)
    entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in values)
    max_entropy = np.log2(len(action_dist)) if len(action_dist) > 0 else 0.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    # Stability score (高いエントロピー = 多様性)
    diversity_score = normalized_entropy * 100
    
    return {
        "entropy": entropy,
        "max_entropy": max_entropy,
        "normalized_entropy": normalized_entropy,
        "diversity_score_pct": diversity_score,
        "is_diverse": normalized_entropy > 0.9  # > 90% of max entropy
    }


def generate_insights(report: dict[str, object]) -> list[str]:
    """洞察とレコメンデーションを生成"""
    insights = []
    
    action_analysis = analyze_action_distribution(report)
    perf_analysis = analyze_performance(report)
    stability_analysis = analyze_stability(report)
    
    # Action distribution insights
    if action_analysis["is_balanced"]:
        insights.append(
            f"✅ アクション分布は良好にバランスされています "
            f"(max: {action_analysis['dominant_action']} "
            f"{action_analysis['dominant_pct']:.1f}%)"
        )
    else:
        insights.append(
            f"⚠️ アクション分布に偏りがあります: "
            f"{action_analysis['dominant_action']} が "
            f"{action_analysis['dominant_pct']:.1f}% を占めています"
        )
    
    # Performance insights
    if perf_analysis["steps_per_sec"] >= 7.0:
        insights.append(
            f"✅ トレーニング速度は良好です "
            f"({perf_analysis['steps_per_sec']:.2f} steps/sec)"
        )
    else:
        insights.append(
            f"⚠️ トレーニング速度が遅いです "
            f"({perf_analysis['steps_per_sec']:.2f} steps/sec)"
        )
    
    # Feature generation time
    if perf_analysis["feature_time_ratio_pct"] > 60:
        insights.append(
            f"⚠️ 特徴生成に時間がかかりすぎています "
            f"({perf_analysis['feature_time_ratio_pct']:.1f}% of total time)"
        )
    
    # Diversity insights
    if stability_analysis["is_diverse"]:
        insights.append(
            f"✅ 高い多様性を示しています "
            f"(entropy: {stability_analysis['normalized_entropy']:.3f})"
        )
    else:
        insights.append(
            f"⚠️ 多様性が不足しています "
            f"(entropy: {stability_analysis['normalized_entropy']:.3f})"
        )
    
    # Reward insights
    if perf_analysis["final_reward"] > 0.15:
        insights.append(
            f"✅ Final reward が良好です "
            f"({perf_analysis['final_reward']:.4f})"
        )
    elif perf_analysis["final_reward"] > 0:
        insights.append(
            f"🔶 Final reward は正の値ですが改善の余地があります "
            f"({perf_analysis['final_reward']:.4f})"
        )
    else:
        insights.append(
            f"❌ Final reward が負です "
            f"({perf_analysis['final_reward']:.4f})"
        )
    
    return insights


def print_report(report: dict[str, object]) -> None:
    """分析レポートを出力"""
    print("\n" + "=" * 80)
    print("📊 SINGLE EXPERIMENT ANALYSIS")
    print("=" * 80)
    
    # Metadata
    metadata = ensure_dict(report.get("metadata"))
    print(f"\n🔖 Metadata:")
    print(f"   Timestamp: {metadata.get('timestamp', 'N/A')}")
    print(f"   Algorithm: {metadata.get('algorithm', 'N/A')}")
    print(f"   Success: {safe_to_bool(metadata.get('success'), False)}")
    
    # Action distribution
    action_analysis = analyze_action_distribution(report)
    print(f"\n🎯 Action Distribution:")
    for action, pct in action_analysis["distribution_pct"].items():
        print(f"   {action:5s}: {pct:5.2f}%")
    print(f"   Balance Score: {action_analysis['balance_score']:.2f}/100")
    print(f"   Is Balanced: {'✅ Yes' if action_analysis['is_balanced'] else '❌ No'}")
    
    # Performance
    perf_analysis = analyze_performance(report)
    print(f"\n⚡ Performance:")
    print(f"   Total Time: {perf_analysis['total_time_min']:.2f} min")
    print(f"   Training Speed: {perf_analysis['steps_per_sec']:.2f} steps/sec")
    print(f"   Feature Gen Time: {perf_analysis['feature_time_sec']:.1f}s "
          f"({perf_analysis['feature_time_ratio_pct']:.1f}%)")
    print(f"   Final Reward: {perf_analysis['final_reward']:.4f}")
    print(f"   Action Diversity: {perf_analysis['action_diversity']:.4f}")
    
    # Stability
    stability_analysis = analyze_stability(report)
    print(f"\n🔄 Stability:")
    print(f"   Entropy: {stability_analysis['entropy']:.4f} "
          f"(max: {stability_analysis['max_entropy']:.4f})")
    print(f"   Normalized: {stability_analysis['normalized_entropy']:.4f}")
    print(f"   Diversity Score: {stability_analysis['diversity_score_pct']:.2f}%")
    print(f"   Is Diverse: {'✅ Yes' if stability_analysis['is_diverse'] else '❌ No'}")
    
    # Insights
    insights = generate_insights(report)
    print(f"\n💡 Insights & Recommendations:")
    for i, insight in enumerate(insights, 1):
        print(f"   {i}. {insight}")
    
    # System info
    sys_info = ensure_dict(report.get("system_info"))
    print(f"\n💻 System Info:")
    print(f"   Platform: {sys_info.get('platform', 'N/A')}")
    print(f"   Python: {sys_info.get('python_version', 'N/A')}")
    print(f"   CPUs: {safe_to_int(sys_info.get('cpu_count'), 0)}")
    memory_total = safe_to_float(sys_info.get("memory_total"), 0.0)
    memory_available = safe_to_float(sys_info.get("memory_available"), 0.0)
    print(
        f"   Memory: {memory_total / (1024**3):.1f} GB total, "
        f"{memory_available / (1024**3):.1f} GB available"
    )
    
    print("\n" + "=" * 80)
    print("✅ Analysis Complete")
    print("=" * 80)


def save_summary(report: dict[str, object], output_path: Path) -> None:
    """サマリーをJSONとして保存"""

    summary = {
        "metadata": ensure_dict(report.get("metadata")),
        "action_analysis": analyze_action_distribution(report),
        "performance_analysis": analyze_performance(report),
        "stability_analysis": analyze_stability(report),
        "insights": generate_insights(report)
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_compatible(output_path, summary)
    
    print(f"\n📁 Summary saved to: {output_path}")


def main():
    try:
        # Load latest report
        report = load_latest_report()
        
        # Print analysis
        print_report(report)
        
        # Save summary
        output_dir = project_root / "analysis_results" / "v459"
        metadata = ensure_dict(report.get("metadata"))
        timestamp = _sanitize_timestamp(metadata.get("timestamp"))[:15]
        output_path = output_dir / f"single_experiment_analysis_{timestamp}.json"
        save_summary(report, output_path)
        
        print(f"\n🎯 Next Steps:")
        print(f"   1. このパラメータで複数実験を実行し、再現性を確認")
        print(f"   2. 異なるreward configで比較実験を実行")
        print(f"   3. 統計的検定で有意差を確認")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
