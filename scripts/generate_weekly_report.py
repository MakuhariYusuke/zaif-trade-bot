#!/usr/bin/env python3
"""
generate_weekly_report.py
週次特徴量評価レポート生成スクリプト
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any


def load_ablation_results(json_path: Path) -> Dict:
    """アブレーション結果を読み込み"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_experimental_results(json_path: Path) -> Dict:
    """実験的特徴量評価結果を読み込み"""
    if not json_path.exists():
        return {}

    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def should_re_evaluate(feature_result: Dict) -> str:
    """再評価判定"""
    ds = feature_result.get('delta_sharpe', {})
    if not ds:
        return "Insufficient Data"

    mean = ds.get('mean', 0)
    ci_low = ds.get('ci95', [0, 0])[0]

    if mean > 0.05 and ci_low > 0:
        return "**Re-evaluate**"
    elif mean > 0.01:
        return "Monitor"
    else:
        return "Maintain"


def generate_feature_table(ablation_results: Dict) -> str:
    """特徴量評価テーブル生成"""
    lines = []
    lines.append("| Feature | Mean | Std | CI95 Low | CI95 High | Status | Experimental |")
    lines.append("|---------|------|-----|----------|-----------|--------|--------------|")

    for feature_name, result in ablation_results["ablation_results"].items():
        ds = result.get('delta_sharpe', {})
        if not ds:
            mean, std, ci_low, ci_high = "N/A", "N/A", "N/A", "N/A"
        else:
            mean = f"{ds.get('mean', 0):.3f}"
            std = f"{ds.get('std', 0):.3f}"
            ci_low = f"{ds.get('ci95', [0, 0])[0]:.3f}"
            ci_high = f"{ds.get('ci95', [0, 0])[1]:.3f}"

        status = should_re_evaluate(result)
        is_exp = "✓" if result.get('is_experimental', False) else ""

        lines.append(f"| {feature_name} | {mean} | {std} | {ci_low} | {ci_high} | {status} | {is_exp} |")

    return "\n".join(lines)


def generate_experimental_table(experimental_results: Dict) -> str:
    """実験的特徴量テーブル生成"""
    if not experimental_results:
        return "No experimental features evaluated."

    lines = []
    lines.append("| Feature | Duration (ms) | NaN Rate | Columns | Delta Sharpe |")
    lines.append("|---------|---------------|----------|---------|--------------|")

    for feature_name, result in experimental_results.items():
        if "error" in result:
            duration = "Error"
            nan_rate = "Error"
            columns = "Error"
            delta_sharpe = "Error"
        else:
            duration = f"{result.get('duration_ms', 0):.1f}"
            nan_rate = f"{result.get('nan_rate', 0):.4f}"
            columns = len(result.get('columns', []))
            ds = result.get('delta_sharpe', {})
            if ds:
                delta_sharpe = f"{ds.get('mean', 0):.3f}"
            else:
                delta_sharpe = "N/A"

        lines.append(f"| {feature_name} | {duration} | {nan_rate} | {columns} | {delta_sharpe} |")

    return "\n".join(lines)


def generate_summary(ablation_results: Dict) -> str:
    """サマリー生成"""
    total_features = len(ablation_results["ablation_results"])
    valid_results = ablation_results.get("summary", {}).get("valid_results", 0)
    experimental_count = len(ablation_results.get("experimental_features", []))

    re_evaluate_count = sum(
        1 for result in ablation_results["ablation_results"].values()
        if should_re_evaluate(result) == "**Re-evaluate**"
    )

    return f"""## Summary
- Total features evaluated: {total_features}
- Valid results: {valid_results}/{total_features}
- Experimental features: {experimental_count}
- Re-evaluation candidates: {re_evaluate_count}"""


def generate_weekly_report() -> str:
    """週次レポート生成"""
    # 結果ファイル読み込み
    results_dir = Path("results")
    ablation_json = results_dir / "ablation_results.json"
    experimental_json = results_dir / "experimental_eval.json"

    ablation_results = load_ablation_results(ablation_json)
    experimental_results = load_experimental_results(experimental_json)

    # レポート生成
    report = []
    report.append("# Weekly Feature Evaluation Report")
    report.append("")
    report.append(generate_summary(ablation_results))
    report.append("")
    report.append("## Delta Sharpe Results")
    report.append("")
    report.append(generate_feature_table(ablation_results))
    report.append("")
    report.append("## Experimental Features")
    report.append("")
    report.append(generate_experimental_table(experimental_results))
    report.append("")
    report.append("## Notes")
    report.append("- **Re-evaluate**: delta_sharpe.mean > 0.05 and CI95_low > 0")
    report.append("- **Monitor**: delta_sharpe.mean > 0.01")
    report.append("- **Maintain**: Otherwise")
    report.append("- Experimental features are marked with ✓")

    return "\n".join(report)


def main():
    """メイン関数"""
    report = generate_weekly_report()

    # レポート保存
    output_path = Path("weekly_report.md")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Weekly report generated: {output_path}")


if __name__ == "__main__":
    main()