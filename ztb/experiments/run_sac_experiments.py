#!/usr/bin/env python3
"""
SAC v434.2 複数実験自動化スクリプト
異なるハイパーパラメータでトレーニング実験を自動実行
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """実験設定ファイルを読み込み"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_single_experiment(
    experiment: Dict[str, Any], common_config: Dict[str, Any]
) -> Dict[str, Any]:
    """単一の実験を実行"""
    version = experiment["version"]
    description = experiment["description"]
    hyperparams = experiment["hyperparams"]

    print(f"\n{'='*60}")
    print(f"実験開始: {version} - {description}")
    print(f"ハイパーパラメータ: {hyperparams}")
    print(f"{'='*60}")

    # 出力ディレクトリ作成
    output_dir = f"{common_config['output_base_dir']}/{version}"
    backtest_output_dir = f"{common_config['backtest_output_base_dir']}/{version}"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(backtest_output_dir, exist_ok=True)

    # トレーニングコマンド作成
    cmd = [
        sys.executable,
        "ztb/training/integrated/train_sac_v434_2_integrated.py",
        "--data",
        common_config["data_path"],
        "--output",
        output_dir,
        "--timesteps",
        str(common_config["timesteps"]),
        "--learning-rate",
        str(hyperparams["learning_rate"]),
        "--batch-size",
        str(hyperparams["batch_size"]),
        "--gamma",
        str(hyperparams["gamma"]),
        "--tau",
        str(hyperparams["tau"]),
        "--experiment-version",
        version,
    ]

    try:
        # PYTHONPATH設定
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path.cwd())

        print(f"コマンド実行: {' '.join(cmd)}")

        # トレーニング実行
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=3600
        )  # 1時間タイムアウト

        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        if result.returncode == 0:
            print(f"✓ 実験 {version} 成功完了")

            # 結果収集
            experiment_result = {
                "version": version,
                "description": description,
                "hyperparams": hyperparams,
                "status": "success",
                "output_dir": output_dir,
                "backtest_output_dir": backtest_output_dir,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

            # バックテスト結果ファイルが存在するか確認
            backtest_results_path = os.path.join(
                backtest_output_dir, "backtest_results.json"
            )
            if os.path.exists(backtest_results_path):
                with open(backtest_results_path, "r", encoding="utf-8") as f:
                    backtest_data = json.load(f)
                experiment_result["backtest_result"] = backtest_data
                print(f"バックテスト結果読み込み完了: {backtest_data}")

            return experiment_result

        else:
            print(f"✗ 実験 {version} 失敗 (終了コード: {result.returncode})")
            return {
                "version": version,
                "description": description,
                "hyperparams": hyperparams,
                "status": "failed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

    except subprocess.TimeoutExpired:
        print(f"✗ 実験 {version} タイムアウト")
        return {
            "version": version,
            "description": description,
            "hyperparams": hyperparams,
            "status": "timeout",
        }
    except Exception as e:
        print(f"✗ 実験 {version} 例外発生: {e}")
        return {
            "version": version,
            "description": description,
            "hyperparams": hyperparams,
            "status": "error",
            "error": str(e),
        }


def save_experiment_results(results: List[Dict[str, Any]], output_path: str):
    """実験結果を保存"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"実験結果を保存: {output_path}")


def generate_summary_report(results: List[Dict[str, Any]], output_path: str):
    """サマリーレポート生成"""
    successful_experiments = [r for r in results if r.get("status") == "success"]
    failed_experiments = [r for r in results if r.get("status") != "success"]

    report = {
        "summary": {
            "total_experiments": len(results),
            "successful": len(successful_experiments),
            "failed": len(failed_experiments),
            "success_rate": len(successful_experiments) / len(results) * 100
            if results
            else 0,
        },
        "successful_experiments": [],
        "failed_experiments": [r["version"] for r in failed_experiments],
        "best_experiment": None,
    }

    # 成功した実験の詳細
    for exp in successful_experiments:
        backtest = exp.get("backtest_result", {})
        summary = {
            "version": exp["version"],
            "description": exp["description"],
            "hyperparams": exp["hyperparams"],
            "backtest_metrics": {
                "total_reward": backtest.get("total_reward", 0),
                "win_rate": backtest.get("win_rate", 0),
                "sharpe_ratio": backtest.get("sharpe_ratio", 0),
                "max_drawdown": backtest.get("max_drawdown", 0),
                "final_portfolio_value": backtest.get("final_portfolio_value", 0),
            },
        }
        report["successful_experiments"].append(summary)

    # 最適な実験を決定（総報酬が最大のもの）
    if successful_experiments:
        best_exp = max(
            successful_experiments,
            key=lambda x: x.get("backtest_result", {}).get("total_reward", 0),
        )
        report["best_experiment"] = {
            "version": best_exp["version"],
            "description": best_exp["description"],
            "hyperparams": best_exp["hyperparams"],
            "backtest_result": best_exp.get("backtest_result", {}),
        }

    # レポート保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"サマリーレポート生成: {output_path}")
    print(
        f"成功率: {report['summary']['success_rate']:.1f}% ({len(successful_experiments)}/{len(results)})"
    )

    if report["best_experiment"]:
        best = report["best_experiment"]
        print(
            f"最適設定: {best['version']} - 総報酬: {best['backtest_result'].get('total_reward', 0):.2f}"
        )


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="SAC v434.2 複数実験自動化")
    parser.add_argument(
        "--config",
        type=str,
        default="sac_v434_2_experiments_config.json",
        help="実験設定ファイル",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="実行する最大実験数（デバッグ用）",
    )

    args = parser.parse_args()

    print("=== SAC v434.2 複数実験自動化開始 ===")

    # 設定読み込み
    if not os.path.exists(args.config):
        print(f"設定ファイルが見つかりません: {args.config}")
        return 1

    config = load_experiment_config(args.config)
    experiments = config["experiments"]
    common_config = config["common_config"]

    if args.max_experiments:
        experiments = experiments[: args.max_experiments]
        print(f"実験数を制限: {len(experiments)} 件")

    print(f"実行予定実験数: {len(experiments)}")

    # 実験実行
    results = []
    for i, experiment in enumerate(experiments, 1):
        print(f"\n--- 実験 {i}/{len(experiments)} ---")
        result = run_single_experiment(experiment, common_config)
        results.append(result)

    # 結果保存
    results_path = "sac_v434_2_experiment_results.json"
    save_experiment_results(results, results_path)

    # サマリーレポート生成
    report_path = "sac_v434_2_experiment_report.json"
    generate_summary_report(results, report_path)

    print("\n=== 実験完了 ===")
    return 0


if __name__ == "__main__":
    exit(main())
