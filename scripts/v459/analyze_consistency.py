#!/usr/bin/env python3
"""
Phase 3 Day 4-5: Multi-Experiment Consistency Analysis
複数の実験レポートを比較して再現性を検証
"""

import sys
from pathlib import Path
from typing import TypedDict
import pandas as pd
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.v459.json_compat import write_json_compatible
from ztb.reporting.services.catalog import (
    extract_action_distribution_from_payload,
    get_recent_training_reports,
    load_training_report,
)
from ztb.utils.safety import ensure_dict, safe_to_bool, safe_to_float


class ConsistencyMetric(TypedDict):
    mean: float
    std: float
    min: float
    max: float
    range: float
    cv_pct: float


class EvaluationResult(TypedDict):
    cv_pct: float
    level: str
    emoji: str


def load_recent_reports(n: int = 5) -> list[dict[str, object]]:
    """最近のN個のトレーニングレポートをロード"""
    reports_dir = project_root / "reports"
    report_files = get_recent_training_reports(limit=n, reports_dir=reports_dir)
    
    reports: list[dict[str, object]] = []
    for report_file in report_files:
        report = load_training_report(report_file)
        if report is None:
            continue
        report["_filename"] = report_file.name
        reports.append(report)
    
    print(f"📁 Loaded {len(reports)} reports")
    return reports


def extract_metrics(reports: list[dict[str, object]]) -> pd.DataFrame:
    """全レポートからメトリクスを抽出"""
    data: list[dict[str, object]] = []
    
    for report in reports:
        metadata = ensure_dict(report.get("metadata"))
        stats = ensure_dict(report.get("training_stats"))
        action_dist = extract_action_distribution_from_payload(report)
        perf = ensure_dict(report.get("performance_metrics"))
        
        data.append({
            "timestamp": str(metadata.get("timestamp", "")),
            "filename": str(report.get("_filename", "")),
            "success": safe_to_bool(metadata.get("success"), False),
            "total_time_min": safe_to_float(stats.get("training_time"), 0.0) / 60.0,
            "steps_per_sec": safe_to_float(stats.get("steps_per_second"), 0.0),
            "final_reward": safe_to_float(stats.get("final_reward"), 0.0),
            "hold_pct": safe_to_float(action_dist.get("HOLD"), 0.0) * 100.0,
            "buy_pct": safe_to_float(action_dist.get("BUY"), 0.0) * 100.0,
            "sell_pct": safe_to_float(action_dist.get("SELL"), 0.0) * 100.0,
            "action_diversity": safe_to_float(perf.get("action_diversity"), 0.0),
            "feature_time_sec": safe_to_float(stats.get("feature_generation_time_s"), 0.0),
        })
    
    return pd.DataFrame(data)


def calculate_consistency(df: pd.DataFrame) -> dict[str, ConsistencyMetric]:
    """再現性メトリクスを計算"""
    
    # Key metrics for consistency
    metrics = [
        "final_reward", "hold_pct", "buy_pct", "sell_pct", 
        "action_diversity", "steps_per_sec"
    ]
    
    consistency: dict[str, ConsistencyMetric] = {}
    for metric in metrics:
        values = df[metric]
        consistency[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "range": float(values.max() - values.min()),
            "cv_pct": float(values.std() / values.mean() * 100) if values.mean() != 0 else 0.0
        }
    
    return consistency


def evaluate_reproducibility(
    consistency: dict[str, ConsistencyMetric]
) -> dict[str, EvaluationResult]:
    """再現性を評価"""
    
    # Coefficient of Variation (CV) thresholds for "good" reproducibility
    cv_thresholds = {
        "excellent": 2.0,  # CV < 2%
        "good": 5.0,       # CV < 5%
        "acceptable": 10.0 # CV < 10%
    }
    
    evaluations: dict[str, EvaluationResult] = {}
    for metric, stats in consistency.items():
        cv = stats["cv_pct"]
        
        if cv < cv_thresholds["excellent"]:
            level = "excellent"
            emoji = "🟢"
        elif cv < cv_thresholds["good"]:
            level = "good"
            emoji = "🟡"
        elif cv < cv_thresholds["acceptable"]:
            level = "acceptable"
            emoji = "🟠"
        else:
            level = "poor"
            emoji = "🔴"
        
        evaluations[metric] = {
            "cv_pct": cv,
            "level": level,
            "emoji": emoji
        }
    
    return evaluations


def print_analysis(
    df: pd.DataFrame,
    consistency: dict[str, ConsistencyMetric],
    evaluations: dict[str, EvaluationResult],
) -> None:
    """分析結果を出力"""
    print("\n" + "=" * 80)
    print("📊 MULTI-EXPERIMENT CONSISTENCY ANALYSIS")
    print("=" * 80)
    
    print(f"\n📋 Dataset:")
    print(f"   Number of experiments: {len(df)}")
    print(f"   Success rate: {df['success'].sum()}/{len(df)} "
          f"({df['success'].mean() * 100:.1f}%)")
    
    # Time range
    timestamps = pd.to_datetime(df["timestamp"])
    print(f"   Time range: {timestamps.min()} to {timestamps.max()}")
    print(f"   Duration: {(timestamps.max() - timestamps.min()).total_seconds() / 60:.1f} minutes")
    
    print(f"\n🎯 Action Distribution Consistency:")
    for action in ["hold_pct", "buy_pct", "sell_pct"]:
        stats = consistency[action]
        eval_info = evaluations[action]
        action_name = action.replace("_pct", "").upper()
        
        print(f"   {action_name:4s}: {stats['mean']:5.2f}% ± {stats['std']:4.2f}% "
              f"(range: {stats['min']:5.2f}%-{stats['max']:5.2f}%, "
              f"CV: {stats['cv_pct']:.2f}%) {eval_info['emoji']} {eval_info['level']}")
    
    print(f"\n⚡ Performance Consistency:")
    perf_metrics = {
        "final_reward": ("Final Reward", ""),
        "steps_per_sec": ("Steps/Sec", ""),
        "action_diversity": ("Diversity", ""),
        "total_time_min": ("Time (min)", "")
    }
    
    for metric, (label, unit) in perf_metrics.items():
        if metric in consistency:
            stats = consistency[metric]
            eval_info = evaluations[metric]
            
            print(f"   {label:13s}: {stats['mean']:7.3f} ± {stats['std']:6.3f} "
                  f"(CV: {stats['cv_pct']:5.2f}%) "
                  f"{eval_info['emoji']} {eval_info['level']}")
    
    print(f"\n📈 Overall Reproducibility Assessment:")
    
    # Count by level
    level_counts = {}
    for eval_info in evaluations.values():
        level = eval_info["level"]
        level_counts[level] = level_counts.get(level, 0) + 1
    
    total = len(evaluations)
    for level in ["excellent", "good", "acceptable", "poor"]:
        count = level_counts.get(level, 0)
        pct = count / total * 100
        emoji = "🟢" if level == "excellent" else "🟡" if level == "good" else "🟠" if level == "acceptable" else "🔴"
        print(f"   {emoji} {level.capitalize():10s}: {count}/{total} ({pct:.1f}%)")
    
    # Overall score
    excellent_score = level_counts.get("excellent", 0) / total
    good_score = level_counts.get("good", 0) / total
    overall_score = (excellent_score * 1.0 + good_score * 0.8) * 100
    
    print(f"\n   Overall Reproducibility Score: {overall_score:.1f}/100")
    
    if overall_score >= 80:
        print(f"   ✅ Excellent reproducibility - ready for production")
    elif overall_score >= 60:
        print(f"   🟡 Good reproducibility - acceptable for experiments")
    elif overall_score >= 40:
        print(f"   🟠 Moderate reproducibility - needs improvement")
    else:
        print(f"   🔴 Poor reproducibility - investigate causes")
    
    print("\n" + "=" * 80)


def save_analysis(
    df: pd.DataFrame,
    consistency: dict[str, ConsistencyMetric],
    evaluations: dict[str, EvaluationResult],
    output_path: Path,
) -> None:
    """分析結果を保存"""

    result = {
        "timestamp": datetime.now().isoformat(),
        "n_experiments": len(df),
        "experiments": df.to_dict(orient="records"),
        "consistency_metrics": consistency,
        "reproducibility_evaluation": evaluations
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_compatible(output_path, result)
    
    print(f"\n📁 Analysis saved to: {output_path}")


def main():
    try:
        # Load reports
        reports = load_recent_reports(n=5)
        
        if len(reports) < 2:
            print("⚠️ Need at least 2 reports for consistency analysis")
            return
        
        # Extract metrics
        df = extract_metrics(reports)
        
        # Calculate consistency
        consistency = calculate_consistency(df)
        
        # Evaluate reproducibility
        evaluations = evaluate_reproducibility(consistency)
        
        # Print analysis
        print_analysis(df, consistency, evaluations)
        
        # Save results
        output_dir = project_root / "analysis_results" / "v459"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"consistency_analysis_{timestamp}.json"
        save_analysis(df, consistency, evaluations, output_path)
        
        print(f"\n🎯 Conclusions:")
        print(f"   - Experiments show {'high' if evaluations['final_reward']['level'] in ['excellent', 'good'] else 'moderate'} consistency")
        print(f"   - Action distribution is {'stable' if evaluations['buy_pct']['cv_pct'] < 5 else 'variable'}")
        print(f"   - {'✅ Ready' if evaluations['final_reward']['cv_pct'] < 10 else '⚠️ Consider more runs'} for AB testing")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
