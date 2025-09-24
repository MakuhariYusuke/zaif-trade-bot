"""
experimental_evaluator.py
experimental 特徴量の評価用モジュール
"""

import json
import time
import logging
import pandas as pd
from pathlib import Path
from .experimental import MovingAverages, GradientSign

EXPERIMENTAL_FEATURES = {
    "MovingAverages": MovingAverages,
    "GradientSign": GradientSign,
}

def evaluate_experimental_features(df: pd.DataFrame, feature_names=None, output_json=True) -> dict:
    """
    experimental 特徴量を評価

    Args:
        df: 評価用データフレーム
        feature_names: 評価対象の特徴量名リスト（Noneで全評価）
        output_json: JSONファイル出力フラグ

    Returns:
        各特徴量の評価結果を含む辞書
    """
    if feature_names is None:
        feature_names = list(EXPERIMENTAL_FEATURES.keys())

    results = {}
    for name in feature_names:
        try:
            feature_cls = EXPERIMENTAL_FEATURES[name]
            feature = feature_cls()
            start = time.time()
            output = feature.compute(df)
            duration = (time.time() - start) * 1000
            nan_rate = output.isna().mean().mean()
            results[name] = {
                "duration_ms": round(duration, 2),
                "columns": list(output.columns),
                "rows": len(output),
                "nan_rate": round(nan_rate, 4),
                "delta_sharpe": 0.0,  # ランキング結果との連携（仮）
            }
        except Exception as e:
            logging.error(f"Failed to evaluate {name}: {e}")
            results[name] = {"error": str(e)}

    # JSON出力
    if output_json:
        output_path = Path("results/experimental_eval.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logging.info(f"Experimental evaluation results saved to {output_path}")

    return results