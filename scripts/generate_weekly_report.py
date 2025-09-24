#!/usr/bin/env python3
"""
generate_weekly_report.py
週次特徴量評価レポート生成スクリプト
"""

import json
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import argparse
from notifier import send_summary, create_summary_dict


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


def load_evaluation_config() -> Dict:
    """評価設定を読み込み"""
    config_path = Path("config/evaluation.yaml")
    if not config_path.exists():
        # デフォルト値
        return {
            'thresholds': {
                're_evaluate': 0.05,
                'monitor': 0.01
            },
            'min_samples': 10000
        }

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def should_re_evaluate(feature_result: Dict, config: Dict) -> str:
    """再評価判定"""
    ds = feature_result.get('delta_sharpe', {})
    if not ds:
        return "Insufficient Data"

    mean = ds.get('mean', 0)
    ci_low = ds.get('ci95', [0, 0])[0]

    thresholds = config.get('thresholds', {})
    re_evaluate_threshold = thresholds.get('re_evaluate', 0.05)
    monitor_threshold = thresholds.get('monitor', 0.01)

    if mean > re_evaluate_threshold and ci_low > 0:
        return "**Re-evaluate**"
    elif mean > monitor_threshold:
        return "Monitor"
    else:
        return "Maintain"


def generate_feature_table(ablation_results: Dict, config: Dict) -> str:
    """特徴量評価テーブル生成（カテゴリ別）"""
    # カテゴリ分類
    categories = {
        "trend": [],
        "volatility": [],
        "momentum": [],
        "volume": [],
        "experimental": []
    }

    # features.yamlからカテゴリ情報を取得
    try:
        with open(Path("config/features.yaml"), 'r', encoding='utf-8') as f:
            features_config = yaml.safe_load(f)
    except:
        features_config = {"features": {}}

    for feature_name, result in ablation_results["ablation_results"].items():
        # カテゴリ判定
        category = "experimental" if result.get('is_experimental', False) else "trend"  # デフォルト

        # features.yamlからwave情報を取得
        if feature_name in features_config.get("features", {}):
            wave = features_config["features"][feature_name].get("wave", "trend")
            if wave in ["trend", "volatility", "momentum", "volume"]:
                category = wave

        categories[category].append((feature_name, result))

    # カテゴリ別テーブル生成
    sections = []
    for category, features in categories.items():
        if not features:
            continue

        sections.append(f"### {category.title()} Features")
        sections.append("")

        lines = []
        lines.append("| Feature | Mean | Std | CI95 Low | CI95 High | Status | Runs | NaN Rate |")
        lines.append("|---------|------|-----|----------|-----------|--------|------|----------|")

        for feature_name, result in features:
            ds = result.get('delta_sharpe', {})
            if not ds:
                mean, std, ci_low, ci_high = "N/A", "N/A", "N/A", "N/A"
            else:
                mean = f"{ds.get('mean', 0):.3f}"
                std = f"{ds.get('std', 0):.3f}"
                ci_low = f"{ds.get('ci95', [0, 0])[0]:.3f}"
                ci_high = f"{ds.get('ci95', [0, 0])[1]:.3f}"

            status = should_re_evaluate(result, config)
            runs = result.get('runs', 'N/A')

            # NaN率取得（experimental結果から）
            nan_rate = "N/A"
            exp_results = ablation_results.get("experimental_eval", {})
            if feature_name in exp_results and "nan_rate" in exp_results[feature_name]:
                nan_rate = f"{exp_results[feature_name]['nan_rate']:.4f}"

            lines.append(f"| {feature_name} | {mean} | {std} | {ci_low} | {ci_high} | {status} | {runs} | {nan_rate} |")

        sections.extend(lines)
        sections.append("")

    return "\n".join(sections)


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


def generate_delta_sharpe_plot(ablation_results: Dict, output_dir: Path) -> str:
    """delta_sharpeの分布グラフ生成"""
    features = []
    means = []
    ci_lows = []
    ci_highs = []
    categories = []

    for feature_name, result in ablation_results["ablation_results"].items():
        ds = result.get('delta_sharpe', {})
        if ds:
            features.append(feature_name)
            means.append(ds.get('mean', 0))
            ci_lows.append(ds.get('ci95', [0, 0])[0])
            ci_highs.append(ds.get('ci95', [0, 0])[1])
            categories.append("experimental" if result.get('is_experimental', False) else "stable")

    if not features:
        return ""

    # グラフ生成
    plt.figure(figsize=(12, 6))
    colors = ['red' if cat == 'experimental' else 'blue' for cat in categories]

    # エラーバー付き散布図
    means_arr = np.array(means)
    ci_lows_arr = np.array(ci_lows)
    ci_highs_arr = np.array(ci_highs)
    plt.errorbar(range(len(features)), means_arr,
                yerr=[means_arr - ci_lows_arr, ci_highs_arr - means_arr],
                fmt='o', capsize=5, capthick=2, elinewidth=2,
                color='blue', alpha=0.7)

    # 基準線
    plt.axhline(y=0.05, color='green', linestyle='--', alpha=0.7, label='Re-evaluate threshold')
    plt.axhline(y=0.01, color='orange', linestyle='--', alpha=0.7, label='Monitor threshold')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.xticks(range(len(features)), features, rotation=45, ha='right')
    plt.ylabel('Delta Sharpe')
    plt.title('Delta Sharpe Distribution with 95% CI')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存
    plot_path = output_dir / "delta_sharpe_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return f"![Delta Sharpe Distribution]({plot_path.name})"


def generate_summary(ablation_results: Dict, config: Dict) -> str:
    """サマリー生成"""
    total_features = len(ablation_results["ablation_results"])
    valid_results = ablation_results.get("summary", {}).get("valid_results", 0)
    experimental_count = len(ablation_results.get("experimental_features", []))

    re_evaluate_count = sum(
        1 for result in ablation_results["ablation_results"].values()
        if should_re_evaluate(result, config) == "**Re-evaluate**"
    )

    return f"""## Summary

- **Total Features**: {total_features}
- **Valid Results**: {valid_results}
- **Experimental Features**: {experimental_count}
- **Re-evaluate Candidates**: {re_evaluate_count}
- **Success Rate**: {valid_results/total_features*100:.1f}%"""


def update_re_evaluate_list(ablation_results: Dict, config: Dict) -> None:
    """再評価リスト更新"""
    from datetime import datetime, timedelta

    list_path = Path("re_evaluate_list.yaml")

    # 既存リスト読み込み
    if list_path.exists():
        with open(list_path, 'r', encoding='utf-8') as f:
            re_evaluate_list = yaml.safe_load(f) or {"re_evaluate": []}
    else:
        re_evaluate_list = {"re_evaluate": []}

    today = datetime.now().date()
    next_due = today + timedelta(days=90)  # 3ヶ月後

    # Re-evaluate対象の特徴量を追加/更新
    for feature_name, result in ablation_results["ablation_results"].items():
        if should_re_evaluate(result, config) == "**Re-evaluate**":
            # 既存エントリ確認
            existing = next((item for item in re_evaluate_list["re_evaluate"]
                           if item["feature"] == feature_name), None)

            if existing:
                # 更新
                existing["last_checked"] = today.isoformat()
                existing["next_due"] = next_due.isoformat()
            else:
                # 新規追加
                re_evaluate_list["re_evaluate"].append({
                    "feature": feature_name,
                    "last_checked": today.isoformat(),
                    "next_due": next_due.isoformat()
                })

    # 保存
    with open(list_path, 'w', encoding='utf-8') as f:
        yaml.dump(re_evaluate_list, f, default_flow_style=False, allow_unicode=True)

    print(f"Re-evaluate list updated: {list_path}")


def generate_weekly_report() -> str:
    """週次レポート生成"""
    # 設定読み込み
    config = load_evaluation_config()

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
    report.append(generate_summary(ablation_results, config))
    report.append("")
    report.append("## Delta Sharpe Distribution")
    report.append("")
    report.append(generate_delta_sharpe_plot(ablation_results, results_dir))
    report.append("")
    report.append("## Delta Sharpe Results")
    report.append("")
    report.append(generate_feature_table(ablation_results, config))
    report.append("")
    report.append("## Experimental Features")
    report.append("")
    report.append(generate_experimental_table(experimental_results))
    report.append("")
    report.append("## Notes")
    thresholds = config.get('thresholds', {})
    re_evaluate_threshold = thresholds.get('re_evaluate', 0.05)
    monitor_threshold = thresholds.get('monitor', 0.01)
    report.append(f"- **Re-evaluate**: delta_sharpe.mean > {re_evaluate_threshold} and CI95_low > 0")
    report.append(f"- **Monitor**: delta_sharpe.mean > {monitor_threshold}")
    report.append("- **Maintain**: Otherwise")
    report.append("- Experimental features are marked with ✓")

    # 再評価リスト更新
    update_re_evaluate_list(ablation_results, config)

    return "\n".join(report)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Generate weekly feature evaluation report')
    parser.add_argument('--notify', choices=['slack', 'discord'], help='Send notification to specified platform')
    args = parser.parse_args()

    report = generate_weekly_report()

    # レポート保存
    output_path = Path("weekly_report.md")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Weekly report generated: {output_path}")

    # 通知送信
    if args.notify:
        config = load_evaluation_config()
        ablation_results = load_ablation_results(Path("results/ablation_results.json"))
        summary = create_summary_dict(ablation_results, config)
        success = send_summary(summary, args.notify)
        if success:
            print(f"Notification sent to {args.notify}")
        else:
            print(f"Failed to send notification to {args.notify}")


if __name__ == "__main__":
    main()