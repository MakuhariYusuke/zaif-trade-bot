#!/usr/bin/env python3
"""
Phase 3 Day 4-5: Single Experiment Analysis
最新のトレーニングレポートを分析して、検証可能なメトリクスを抽出
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def load_latest_report() -> Dict[str, Any]:
    """最新のトレーニングレポートをロード"""
    reports_dir = project_root / "reports"
    report_files = sorted(reports_dir.glob("training_report_*.json"), reverse=True)
    
    if not report_files:
        raise FileNotFoundError("No training reports found")
    
    latest = report_files[0]
    print(f"📁 Loading: {latest.name}")
    
    with open(latest, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_action_distribution(report: Dict[str, Any]) -> Dict[str, Any]:
    """アクション分布の分析"""
    action_dist = report["training_stats"]["action_distribution"]
    
    # Convert to percentages
    action_pct = {k: v * 100 for k, v in action_dist.items()}
    
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
        "dominant_pct": max_action[1]
    }


def analyze_performance(report: Dict[str, Any]) -> Dict[str, Any]:
    """パフォーマンスメトリクスの分析"""
    stats = report["training_stats"]
    perf = report["performance_metrics"]
    
    # Training efficiency
    total_time = stats["training_time"]
    timesteps = stats["total_timesteps"]
    steps_per_sec = stats["steps_per_second"]
    
    # Time analysis
    feature_time = stats.get("feature_generation_time_s", 0)
    training_time = total_time - feature_time
    feature_ratio = feature_time / total_time * 100
    
    # Reward analysis
    final_reward = float(stats["final_reward"])
    
    return {
        "total_time_sec": total_time,
        "total_time_min": total_time / 60,
        "timesteps": timesteps,
        "steps_per_sec": steps_per_sec,
        "feature_time_sec": feature_time,
        "training_time_sec": training_time,
        "feature_time_ratio_pct": feature_ratio,
        "final_reward": final_reward,
        "action_diversity": perf["action_diversity"]
    }


def analyze_stability(report: Dict[str, Any]) -> Dict[str, Any]:
    """安定性の評価（単一実験からの推定）"""
    action_dist = report["training_stats"]["action_distribution"]
    
    # Entropy calculation (Shannon entropy)
    values = list(action_dist.values())
    entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in values)
    max_entropy = np.log2(len(values))
    normalized_entropy = entropy / max_entropy
    
    # Stability score (高いエントロピー = 多様性)
    diversity_score = normalized_entropy * 100
    
    return {
        "entropy": entropy,
        "max_entropy": max_entropy,
        "normalized_entropy": normalized_entropy,
        "diversity_score_pct": diversity_score,
        "is_diverse": normalized_entropy > 0.9  # > 90% of max entropy
    }


def generate_insights(report: Dict[str, Any]) -> list[str]:
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


def print_report(report: Dict[str, Any]):
    """分析レポートを出力"""
    print("\n" + "=" * 80)
    print("📊 SINGLE EXPERIMENT ANALYSIS")
    print("=" * 80)
    
    # Metadata
    metadata = report["metadata"]
    print(f"\n🔖 Metadata:")
    print(f"   Timestamp: {metadata['timestamp']}")
    print(f"   Algorithm: {metadata['algorithm']}")
    print(f"   Success: {metadata['success']}")
    
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
    sys_info = report["system_info"]
    print(f"\n💻 System Info:")
    print(f"   Platform: {sys_info['platform']}")
    print(f"   Python: {sys_info['python_version']}")
    print(f"   CPUs: {sys_info['cpu_count']}")
    print(f"   Memory: {sys_info['memory_total'] / (1024**3):.1f} GB total, "
          f"{sys_info['memory_available'] / (1024**3):.1f} GB available")
    
    print("\n" + "=" * 80)
    print("✅ Analysis Complete")
    print("=" * 80)


def save_summary(report: Dict[str, Any], output_path: Path):
    """サマリーをJSONとして保存"""
    
    # Convert numpy types to native Python types
    def convert_to_native(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        else:
            return obj
    
    summary = {
        "metadata": report["metadata"],
        "action_analysis": analyze_action_distribution(report),
        "performance_analysis": analyze_performance(report),
        "stability_analysis": analyze_stability(report),
        "insights": generate_insights(report)
    }
    
    summary = convert_to_native(summary)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 Summary saved to: {output_path}")


def main():
    try:
        # Load latest report
        report = load_latest_report()
        
        # Print analysis
        print_report(report)
        
        # Save summary
        output_dir = project_root / "analysis_results" / "v459"
        timestamp = report["metadata"]["timestamp"].replace(":", "").replace("-", "")[:15]
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
