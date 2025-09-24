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

import pandas as pd
import yaml

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trading.features.experimental_evaluator import evaluate_experimental_features


def load_config(config_path: Path) -> dict:
    """設定ファイルを読み込み"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_feature_sets(sets_path: Path) -> dict:
    """特徴量セットを読み込み"""
    with open(sets_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_ablation_analysis(
    base_features: List[str],
    experimental_features: Optional[List[str]] = None,
    data_path: Optional[Path] = None
) -> Dict:
    """
    アブレーション分析を実行

    Args:
        base_features: ベースとなる特徴量リスト
        experimental_features: 実験的特徴量リスト（オプション）
        data_path: データファイルパス

    Returns:
        分析結果
    """
    results = {
        "base_features": base_features,
        "experimental_features": experimental_features or [],
        "ablation_results": {},
        "experimental_eval": {}
    }

    # データ読み込み（仮）
    if data_path and data_path.exists():
        df = pd.read_csv(data_path)
    else:
        # サンプルデータ生成
        df = pd.DataFrame({
            'close': [100 + i * 0.1 for i in range(100)],
            'price': [100 + i * 0.1 for i in range(100)],
            'volume': [1000 + i * 10 for i in range(100)]
        })

    # ベース特徴量のアブレーション分析（仮実装）
    for feature_name in base_features:
        results["ablation_results"][feature_name] = {
            "success": True,
            "sharpe_impact": 0.0,  # 仮の値
            "is_experimental": False
        }

    # experimental特徴量の評価
    if experimental_features:
        exp_results = evaluate_experimental_features(df, experimental_features)
        results["experimental_eval"] = exp_results

        # experimental特徴量もアブレーション分析に含める
        for feature_name, eval_result in exp_results.items():
            if "error" not in eval_result:
                results["ablation_results"][feature_name] = {
                    "success": True,
                    "columns": eval_result["columns"],
                    "duration_ms": eval_result["duration_ms"],
                    "nan_rate": eval_result["nan_rate"],
                    "sharpe_impact": 0.0,  # 仮の値
                    "is_experimental": True
                }

    return results


def save_results(results: Dict, output_path: Path):
    """結果をJSONファイルに保存"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


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
    parser.add_argument('--output', type=Path, default=Path('results/ablation_results.json'),
                       help='出力ファイルパス')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # 設定読み込み
        config = load_config(args.config)
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
            data_path=args.data
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