#!/usr/bin/env python3
"""
SAC v434.2 複数実験自動化スクリプト
異なるハイパーパラメータでトレーニング実験を自動実行
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import NotRequired, TypedDict

from ztb.io.common import PathLike
from ztb.io.json_io import read_json_object, write_json


class HyperParams(TypedDict):
    learning_rate: float
    batch_size: int
    gamma: float
    tau: float


class ExperimentConfig(TypedDict):
    version: str
    description: str
    hyperparams: HyperParams


class CommonConfig(TypedDict):
    output_base_dir: str
    backtest_output_base_dir: str
    data_path: str
    timesteps: int


class ExperimentSuiteConfig(TypedDict):
    experiments: list[ExperimentConfig]
    common_config: CommonConfig


class ExperimentResult(TypedDict):
    version: str
    description: str
    hyperparams: HyperParams
    status: str
    output_dir: NotRequired[str]
    backtest_output_dir: NotRequired[str]
    stdout: NotRequired[str]
    stderr: NotRequired[str]
    returncode: NotRequired[int]
    error: NotRequired[str]
    backtest_result: NotRequired[dict[str, object]]


def _as_object_map(value: object) -> dict[str, object]:
    """Safely coerce object into dict."""
    return value if isinstance(value, dict) else {}


def _as_float(value: object, default: float = 0.0) -> float:
    """Convert object to float with fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int = 0) -> int:
    """Convert object to int with fallback."""
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_hyperparams(value: object) -> HyperParams | None:
    """Parse hyper parameter section."""
    payload = _as_object_map(value)
    learning_rate = _as_float(payload.get("learning_rate"), -1.0)
    batch_size = _as_int(payload.get("batch_size"), -1)
    gamma = _as_float(payload.get("gamma"), -1.0)
    tau = _as_float(payload.get("tau"), -1.0)

    if learning_rate <= 0 or batch_size <= 0 or gamma <= 0 or tau <= 0:
        return None

    return {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gamma": gamma,
        "tau": tau,
    }


def _parse_experiment(value: object) -> ExperimentConfig | None:
    """Parse single experiment payload."""
    payload = _as_object_map(value)
    version = payload.get("version")
    description = payload.get("description")
    hyperparams = _parse_hyperparams(payload.get("hyperparams"))
    if not isinstance(version, str) or not isinstance(description, str):
        return None
    if hyperparams is None:
        return None
    return {
        "version": version,
        "description": description,
        "hyperparams": hyperparams,
    }


def _parse_common_config(value: object) -> CommonConfig | None:
    """Parse common config payload."""
    payload = _as_object_map(value)
    output_base_dir = payload.get("output_base_dir")
    backtest_output_base_dir = payload.get("backtest_output_base_dir")
    data_path = payload.get("data_path")
    timesteps = _as_int(payload.get("timesteps"), -1)
    if (
        not isinstance(output_base_dir, str)
        or not isinstance(backtest_output_base_dir, str)
        or not isinstance(data_path, str)
        or timesteps <= 0
    ):
        return None
    return {
        "output_base_dir": output_base_dir,
        "backtest_output_base_dir": backtest_output_base_dir,
        "data_path": data_path,
        "timesteps": timesteps,
    }


def load_experiment_config(config_path: PathLike) -> ExperimentSuiteConfig:
    """実験設定ファイルを読み込み"""
    payload = read_json_object(config_path)
    raw_experiments = payload.get("experiments")
    experiments: list[ExperimentConfig] = []
    if isinstance(raw_experiments, list):
        for item in raw_experiments:
            experiment = _parse_experiment(item)
            if experiment is not None:
                experiments.append(experiment)

    common_config = _parse_common_config(payload.get("common_config"))
    if common_config is None:
        raise ValueError("Invalid or missing common_config in experiment config")
    if not experiments:
        raise ValueError("No valid experiments found in experiment config")

    return {
        "experiments": experiments,
        "common_config": common_config,
    }


def run_single_experiment(
    experiment: ExperimentConfig, common_config: CommonConfig
) -> ExperimentResult:
    """単一の実験を実行"""
    version = experiment["version"]
    description = experiment["description"]
    hyperparams = experiment["hyperparams"]

    print(f"\n{'='*60}")
    print(f"実験開始: {version} - {description}")
    print(f"ハイパーパラメータ: {hyperparams}")
    print(f"{'='*60}")

    # 出力ディレクトリ作成
    output_dir = Path(common_config["output_base_dir"]) / version
    backtest_output_dir = Path(common_config["backtest_output_base_dir"]) / version
    output_dir.mkdir(parents=True, exist_ok=True)
    backtest_output_dir.mkdir(parents=True, exist_ok=True)

    # トレーニングコマンド作成
    cmd = [
        sys.executable,
        "ztb/training/integrated/train_sac_v434_2_integrated.py",
        "--data",
        common_config["data_path"],
        "--output",
        str(output_dir),
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
                "output_dir": str(output_dir),
                "backtest_output_dir": str(backtest_output_dir),
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

            # バックテスト結果ファイルが存在するか確認
            backtest_results_path = backtest_output_dir / "backtest_results.json"
            if backtest_results_path.exists():
                backtest_data = read_json_object(backtest_results_path)
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


def save_experiment_results(
    results: list[ExperimentResult], output_path: PathLike
) -> None:
    """実験結果を保存"""
    write_json(output_path, results, indent=2, ensure_ascii=False)
    print(f"実験結果を保存: {output_path}")


def _extract_backtest_result(result: ExperimentResult) -> dict[str, object]:
    """Get backtest payload if present."""
    return _as_object_map(result.get("backtest_result", {}))


def generate_summary_report(
    results: list[ExperimentResult], output_path: PathLike
) -> None:
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
        backtest = _extract_backtest_result(exp)
        summary = {
            "version": exp["version"],
            "description": exp["description"],
            "hyperparams": exp["hyperparams"],
            "backtest_metrics": {
                "total_reward": _as_float(backtest.get("total_reward", 0.0), 0.0),
                "win_rate": _as_float(backtest.get("win_rate", 0.0), 0.0),
                "sharpe_ratio": _as_float(backtest.get("sharpe_ratio", 0.0), 0.0),
                "max_drawdown": _as_float(backtest.get("max_drawdown", 0.0), 0.0),
                "final_portfolio_value": _as_float(
                    backtest.get("final_portfolio_value", 0.0), 0.0
                ),
            },
        }
        report["successful_experiments"].append(summary)

    # 最適な実験を決定（総報酬が最大のもの）
    if successful_experiments:
        best_exp = max(
            successful_experiments,
            key=lambda x: _as_float(
                _extract_backtest_result(x).get("total_reward", 0.0), 0.0
            ),
        )
        report["best_experiment"] = {
            "version": best_exp["version"],
            "description": best_exp["description"],
            "hyperparams": best_exp["hyperparams"],
            "backtest_result": _extract_backtest_result(best_exp),
        }

    # レポート保存
    write_json(output_path, report, indent=2, ensure_ascii=False)

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
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"設定ファイルが見つかりません: {args.config}")
        return 1

    config = load_experiment_config(config_path)
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
