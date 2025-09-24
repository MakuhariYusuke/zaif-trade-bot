#!/usr/bin/env python3
"""
ablation_runner.py
特徴量アブレーション分析実行スクリプト
experimental特徴量の評価も統合
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trading.features.experimental_evaluator import evaluate_experimental_features
from src.trading.metrics import sharpe_ratio, sharpe_with_stats, calculate_delta_sharpe, validate_ablation_results


def load_config(config_path: Path) -> dict:
    """設定ファイルを読み込み"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_evaluation_config() -> dict:
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


def load_feature_sets(sets_path: Path) -> dict:
    """特徴量セットを読み込み"""
    with open(sets_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_ablation_analysis(
    base_features: List[str],
    experimental_features: Optional[List[str]] = None,
    data_path: Optional[Path] = None,
    num_runs: int = 5,
    evaluation_config: Optional[dict] = None
) -> Dict:
    """
    アブレーション分析を実行

    Args:
        base_features: ベースとなる特徴量リスト
        experimental_features: 実験的特徴量リスト（オプション）
        data_path: データファイルパス
        num_runs: 実行回数

    Returns:
        分析結果
    """
    results = {
        "base_features": base_features,
        "experimental_features": experimental_features or [],
        "ablation_results": {},
        "experimental_eval": {},
        "summary": {
            "total_runs": num_runs,
            "valid_results": 0
        }
    }

    # データ読み込み（仮）
    if data_path and data_path.exists():
        df = pd.read_csv(data_path)
    else:
        # サンプルデータ生成
        np.random.seed(42)  # 再現性のためにシード固定
        n_samples = 1000
        df = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.normal(0.001, 0.02, n_samples)),
            'price': 100 + np.cumsum(np.random.normal(0.001, 0.02, n_samples)),
            'volume': np.random.normal(1000, 100, n_samples)
        })

    # 評価設定からパラメータ取得
    min_samples = evaluation_config.get('min_samples', 10000) if evaluation_config else 10000

    # ベース特徴量でのSharpe ratio計算（シミュレーション）
    base_sharpes = []
    for run in range(num_runs):
        # ランダムなリターンを生成（実際のトレーディング結果をシミュレート）
        returns = np.random.normal(0.001, 0.02, 252)  # 1年分の日次リターン
        sharpe = sharpe_ratio(returns)
        base_sharpes.append(sharpe)

    base_stats = sharpe_with_stats(base_sharpes)

    # 各特徴量のアブレーション分析
    for feature_name in base_features + (experimental_features or []):
        # 特徴量追加時のSharpe ratio計算（シミュレーション）
        with_feature_sharpes = []
        for run in range(num_runs):
            # 特徴量の影響をシミュレート（ランダムノイズ）
            impact = np.random.normal(0, 0.01, 1)[0]  # 特徴量の影響
            returns = np.random.normal(0.001 + impact, 0.02, 252)
            sharpe = sharpe_ratio(returns)
            with_feature_sharpes.append(sharpe)

        # delta_sharpeの計算
        delta_sharpe = calculate_delta_sharpe(base_sharpes, with_feature_sharpes)

        results["ablation_results"][feature_name] = {
            "success": delta_sharpe is not None,
            "sharpe_stats": sharpe_with_stats(with_feature_sharpes),
            "delta_sharpe": delta_sharpe,
            "runs": num_runs,
            "is_experimental": feature_name in (experimental_features or [])
        }

        if validate_ablation_results(results["ablation_results"][feature_name]):
            results["summary"]["valid_results"] += 1

    base_stats = sharpe_with_stats(base_sharpes)

    # experimental特徴量の評価
    if experimental_features:
        exp_results = evaluate_experimental_features(df, experimental_features, baseline_sharpe=base_stats["mean"])
        results["experimental_eval"] = exp_results

    return results


def save_results(results: Dict, output_path: Path):
    """結果をJSONファイルとCSVファイルに保存"""
    # JSON保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # CSV保存
    csv_path = output_path.with_suffix('.csv')
    csv_data = []

    for feature_name, feature_result in results["ablation_results"].items():
        row = {
            "feature": feature_name,
            "is_experimental": feature_result.get("is_experimental", False),
            "success": feature_result.get("success", False),
            "runs": feature_result.get("runs", 0)
        }

        # Sharpe統計
        if "sharpe_stats" in feature_result:
            stats = feature_result["sharpe_stats"]
            row.update({
                "sharpe_mean": stats.get("mean", 0.0),
                "sharpe_std": stats.get("std", 0.0),
                "sharpe_ci_low": stats.get("ci95", [0.0, 0.0])[0],
                "sharpe_ci_high": stats.get("ci95", [0.0, 0.0])[1]
            })

        # delta_sharpe統計
        if "delta_sharpe" in feature_result and feature_result["delta_sharpe"] is not None:
            ds = feature_result["delta_sharpe"]
            row.update({
                "delta_sharpe_mean": ds.get("mean", 0.0),
                "delta_sharpe_std": ds.get("std", 0.0),
                "delta_sharpe_ci_low": ds.get("ci95", [0.0, 0.0])[0],
                "delta_sharpe_ci_high": ds.get("ci95", [0.0, 0.0])[1]
            })
        else:
            row.update({
                "delta_sharpe_mean": None,
                "delta_sharpe_std": None,
                "delta_sharpe_ci_low": None,
                "delta_sharpe_ci_high": None
            })

        csv_data.append(row)

    # CSV出力
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False, encoding='utf-8')

    logging.info(f"Results saved to {output_path} and {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='特徴量アブレーション分析')
    parser.add_argument('--config', type=Path, default=Path('config/features.yaml'),
                       help='特徴量設定ファイル')
    parser.add_argument('--feature-sets', type=Path, default=Path('config/feature_sets.yaml'),
                       help='特徴量セット設定ファイル')
    parser.add_argument('--set', type=str, default='balanced',
                       help='使用する特徴量セット (minimal/balanced/extended)')
    parser.add_argument('--include-experimental', action='store_true',
                       help='experimental特徴量を含めて評価')
    parser.add_argument('--data', type=Path,
                       help='評価用データファイル')
    parser.add_argument('--num-runs', type=int, default=5,
                       help='アブレーション分析の実行回数')
    parser.add_argument('--output', type=Path, default=Path('results/ablation_results.json'),
                       help='出力ファイルパス')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # 設定読み込み
        config = load_config(args.config)
        evaluation_config = load_evaluation_config()
        feature_sets = load_feature_sets(args.feature_sets)

        # 特徴量セット取得
        if args.set not in feature_sets['sets']:
            raise ValueError(f"Unknown feature set: {args.set}")

        base_features = feature_sets['sets'][args.set]
        experimental_features = None

        if args.include_experimental:
            # experimental特徴量を取得（experimental.yamlから）
            exp_config_path = Path('config/experimental.yaml')
            if exp_config_path.exists():
                exp_config = load_config(exp_config_path)
                experimental_features = list(exp_config.get('features', {}).keys())
            else:
                logging.warning("experimental.yaml not found, skipping experimental features")

        # アブレーション分析実行
        logging.info(f"Running ablation analysis for set: {args.set}")
        if experimental_features:
            logging.info(f"Including experimental features: {experimental_features}")

        results = run_ablation_analysis(
            base_features=base_features,
            experimental_features=experimental_features,
            data_path=args.data,
            num_runs=args.num_runs,
            evaluation_config=evaluation_config
        )

        # 結果保存
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_results(results, args.output)

        logging.info(f"Results saved to {args.output}")

        # experimental評価結果を別ファイルにも保存
        if results["experimental_eval"]:
            exp_output = args.output.parent / "experimental_eval.json"
            with open(exp_output, 'w', encoding='utf-8') as f:
                json.dump(results["experimental_eval"], f, indent=2, ensure_ascii=False)
            logging.info(f"Experimental evaluation saved to {exp_output}")

    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()