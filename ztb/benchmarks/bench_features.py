#!/usr/bin/env python3
"""
Feature benchmarking script.
特徴量ベンチマークスクリプト
"""

import argparse

# プロジェクトルートをパスに追加
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ztb.features.registry import FeatureRegistry
from ztb.features import get_feature_manager
from ztb.utils.data.data_generation import generate_synthetic_data
from ztb.utils.errors import safe_operation

project_root = Path(__file__).resolve().parent.parent
if project_root.exists():
    sys.path.append(str(project_root))


def load_real_data(sample_path: Path, n_rows: Optional[int] = None) -> pd.DataFrame:
    """実データを読み込み"""
    return safe_operation(
        logger=None,  # Use default logger
        operation=lambda: _load_real_data_impl(sample_path, n_rows),
        context="real_data_loading",
        default_result=generate_synthetic_data(
            n_rows or 1000, freq="1min", episode_length=None, volume_range=(100, 1000)
        ),  # Fallback to synthetic data
    )


def _load_real_data_impl(sample_path: Path, n_rows: Optional[int] = None) -> pd.DataFrame:
    """Implementation of real data loading."""
    if sample_path.exists():
        df = pd.read_parquet(sample_path)
        # Check if required columns exist
        required = ["close", "high", "low", "volume"]
        if all(col in df.columns for col in required):
            if n_rows:
                df = df.head(n_rows)
            return df
        else:
            print(
                f"Warning: {sample_path} missing required columns, using synthetic data"
            )
    else:
        print(f"Warning: {sample_path} not found, using synthetic data")
    return generate_synthetic_data(
        n_rows or 1000, freq="1min", episode_length=None, volume_range=(100, 1000)
    )


def benchmark_feature(
    feature_name: str, manager: FeatureRegistry, df: pd.DataFrame, n_runs: int = 5
) -> Dict[str, float]:
    """単一特徴量のベンチマーク"""
    times = []
    memories = []

    # 特徴量オブジェクトを取得
    if feature_name not in manager.features:
        print(f"Feature {feature_name} not found in manager")
        return {"ms_real": 0.0, "peak_MB": 0.0}
    feature = manager.features[feature_name]
    params = manager.get_feature_info(feature_name).get("params", {})

    for _ in range(n_runs):
        # メモリ追跡開始
        tracemalloc.start()
        start_time = time.perf_counter()

        # 特徴量計算
        try:
            feature.compute(df, **params)
        except Exception as e:
            print(f"Error computing {feature_name}: {e}")
            tracemalloc.stop()
            return {"ms_real": 0.0, "peak_MB": 0.0}

        end_time = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append((end_time - start_time) * 1000)  # ms
        memories.append(peak / 1024 / 1024)  # MB

    return {"ms_real": float(np.median(times)), "peak_MB": float(np.max(memories))}


def benchmark_bundle(
    manager: FeatureRegistry, df: pd.DataFrame, waves: Any, n_runs: int = 5
) -> Dict[str, float]:
    """バンドルのベンチマーク（指定されたwavesで実行）"""
    times = []
    memories = []

    for _ in range(n_runs):
        tracemalloc.start()
        start_time = time.perf_counter()

        for wave in waves:
            manager.compute_features(df, wave=wave)

        end_time = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append((end_time - start_time) * 1000)
        memories.append(peak / 1024 / 1024)

    return {"ms_real": float(np.median(times)), "peak_MB": float(np.max(memories))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark trading features")
    parser.add_argument(
        "--sample",
        type=str,
        default="data/features/2025/09/sample_09.parquet",
        help="Path to sample data",
    )
    parser.add_argument(
        "--waves", type=str, default="1", help="Comma-separated waves to benchmark"
    )
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument(
        "--n-runs", type=int, default=5, help="Number of benchmark runs"
    )
    parser.add_argument(
        "--n-rows", type=int, default=1000, help="Number of rows to use from sample"
    )
    parser.add_argument(
        "--n-synth-rows",
        type=int,
        default=10000,
        help="Number of rows to use for synthetic data",
    )

    args = parser.parse_args()

    # マネージャー初期化
    manager = get_feature_manager()

    # データ読み込み
    sample_path = Path(args.sample)
    real_df = load_real_data(sample_path, args.n_rows)
    synth_df = generate_synthetic_data(args.n_synth_rows)

    # 出力ディレクトリ作成
    output_dir = Path("reports/feature_ranking")
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output is None:
        output_file = output_dir / f"bench_{date_str}.csv"
    else:
        output_file = Path(args.output)
    # ベンチマーク対象の特徴量
    waves = [int(w) for w in args.waves.split(",") if w.strip() != ""]
    enabled_features = []
    for wave in waves:
        enabled_features.extend(manager.get_enabled_features(wave))

    results = []

    print("Benchmarking individual features...")

    # 実データで個別ベンチ
    for feature in enabled_features:
        print(f"  {feature}...")
        real_result = benchmark_feature(feature, manager, real_df, args.n_runs)
        synth_result = benchmark_feature(feature, manager, synth_df, args.n_runs)

        results.append(
            {
                "feature": feature,
                "ms_real": real_result["ms_real"],
                "ms_synth": synth_result["ms_synth"],
                "peak_MB_real": real_result["peak_MB"],
                "peak_MB_synth": synth_result["peak_MB"],
            }
        )

    # バンドルベンチ
    print("Benchmarking bundle...")
    real_bundle = benchmark_bundle(manager, real_df, waves, args.n_runs)
    synth_bundle = benchmark_bundle(manager, synth_df, waves, args.n_runs)

    results.append(
        {
            "feature": f"Wave{'_'.join(map(str, waves))}_bundle",
            "ms_real": real_bundle["ms_real"],
            "ms_synth": synth_bundle["ms_synth"],
            "peak_MB_real": real_bundle["peak_MB"],
            "peak_MB_synth": synth_bundle["peak_MB"],
        }
    )
    # 出力されるCSVのカラム説明
    # feature: 特徴量名またはバンドル名
    # ms_real: 実データでの特徴量計算の中央値実行時間（ミリ秒）
    # ms_synth: 合成データでの特徴量計算の中央値実行時間（ミリ秒）
    # peak_MB_real: 実データでの特徴量計算時の最大メモリ使用量（MB）
    # peak_MB_synth: 合成データでの特徴量計算時の最大メモリ使用量（MB）

    # CSV出力
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    df_results.to_csv(output_file, index=False)

    print(f"Results saved to {output_file}")
    print("\nTop 5 slowest features (real data):")
    sorted_results = df_results[df_results["feature"] != "Wave1_bundle"].sort_values(
        "ms_real", ascending=False
    )
    for _, row in sorted_results.head(5).iterrows():
        print(f"  - {row['feature']}: {row['ms_real']:.2f} ms")


if __name__ == "__main__":
    main()
