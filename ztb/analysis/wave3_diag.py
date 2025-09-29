#!/usr/bin/env python3
"""
Wave3 diagnostic script - correlation, VIF, MI, leak check
Wave3診断スクリプト - 相関、VIF、相互情報量、リークチェック
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.feature_selection import (
    mutual_info_regression,  # type: ignore[import-untyped]
)
from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]

# プロジェクトルートをパスに追加
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from ztb.features import get_feature_manager


def generate_synthetic_data(n_rows: int = 10000) -> pd.DataFrame:
    """合成データを生成"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="1H")

    returns = np.random.normal(0, 0.02, n_rows)
    price = 100 * np.exp(np.cumsum(returns))

    high = price * (1 + np.random.uniform(0, 0.03, n_rows))
    low = price * (1 - np.random.uniform(0, 0.03, n_rows))
    close = price
    volume = np.random.uniform(1000, 10000, n_rows)

    episode_length = 1000
    episode_ids = np.repeat(np.arange(n_rows // episode_length + 1), episode_length)[
        :n_rows
    ]

    df = pd.DataFrame(
        {
            "ts": dates.view("int64") // 10**9,
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "exchange": "synthetic",
            "pair": "BTC/USD",
            "episode_id": episode_ids,
        }
    )

    return df


def calculate_correlations(
    df: pd.DataFrame, target_col: str = "return"
) -> Dict[str, pd.DataFrame]:
    """相関計算"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [
        c for c in numeric_cols if c not in ["ts", "episode_id", target_col]
    ]

    corr_pearson = df[feature_cols + [target_col]].corr(method="pearson")
    corr_spearman = df[feature_cols + [target_col]].corr(method="spearman")

    return {"pearson": corr_pearson, "spearman": corr_spearman}


def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    """VIF計算"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in ["ts", "episode_id", "return"]]

    vif_data = []
    for col in feature_cols:
        try:
            X = df[feature_cols].drop(columns=[col]).fillna(0)
            y = df[col].fillna(0)
            if X.shape[1] > 0:
                lr = LinearRegression()
                lr.fit(X, y)
                r_squared = lr.score(X, y)
                vif = 1 / (1 - r_squared) if r_squared < 1 else np.inf
            else:
                vif = 1.0
        except Exception:
            vif = np.nan

        vif_data.append({"feature": col, "vif": vif, "high_vif": vif > 10})

    return pd.DataFrame(vif_data)


def calculate_mutual_info(
    df: pd.DataFrame, horizons: List[int]
) -> Dict[str, pd.DataFrame]:
    """相互情報量計算"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in ["ts", "episode_id", "return"]]

    mi_results = {}
    for h in horizons:
        target = f"return_h{h}"
        if target not in df.columns:
            df[target] = df["close"].pct_change(h).shift(-h).fillna(0)

        mi_scores = []
        for col in feature_cols:
            try:
                X = df[col].fillna(0).to_numpy().reshape(-1, 1)
                y = df[target].fillna(0).to_numpy()
                mi = mutual_info_regression(X, y, random_state=42)[0]
            except Exception:
                mi = np.nan
            mi_scores.append({"feature": col, "mi": mi})

        mi_results[f"h{h}"] = pd.DataFrame(mi_scores)

    return mi_results


def check_leaks(df: pd.DataFrame) -> pd.DataFrame:
    """リークチェック"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in ["ts", "episode_id", "return"]]

    leak_checks = []
    for col in feature_cols:
        # 現在の価格との相関
        try:
            corr_current, _ = pearsonr(df[col].fillna(0), df["close"].fillna(0))
        except Exception:
            corr_current = np.nan

        # 未来のリターンとの相関 (1ステップ先)
        try:
            future_return = df["close"].pct_change(1).shift(-1).fillna(0)
            corr_future, _ = pearsonr(df[col].fillna(0), future_return)
        except Exception:
            corr_future = np.nan

        # 警告判定
        warning = (
            isinstance(corr_future, float)
            and not np.isnan(corr_future)
            and isinstance(corr_current, float)
            and not np.isnan(corr_current)
            and abs(corr_future) > abs(corr_current) * 1.5
        )
        reason = (
            "Future correlation significantly higher than current" if warning else ""
        )

        leak_checks.append(
            {
                "feature": col,
                "corr_current": corr_current,
                "corr_future": corr_future,
                "warning": warning,
                "reason": reason,
            }
        )

    return pd.DataFrame(leak_checks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wave3 diagnostic analysis")
    parser.add_argument(
        "--waves", type=str, default="1,2,3", help="Comma-separated waves"
    )
    parser.add_argument(
        "--horizons", type=str, default="1,3,6", help="Comma-separated horizons"
    )
    parser.add_argument("--n-rows", type=int, default=10000, help="Number of data rows")

    args = parser.parse_args()

    # マネージャー初期化
    manager = get_feature_manager()

    # データ生成
    df = generate_synthetic_data(args.n_rows)

    # Wave指定
    waves = [int(w) for w in args.waves.split(",")]
    all_features = []
    for wave in waves:
        all_features.extend(manager.get_enabled_features(wave))

    # 特徴量計算
    df_with_features = manager.compute_features(df, wave=None)

    # リターン追加
    df_with_features["return"] = df_with_features["close"].pct_change().fillna(0)

    # 出力ディレクトリ
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path("reports") / f"wave3_diag-{date_str}"
    output_dir.mkdir(parents=True, exist_ok=True)

    horizons = [int(h) for h in args.horizons.split(",")]

    print("Calculating correlations...")
    correlations = calculate_correlations(df_with_features)
    correlations["pearson"].to_csv(output_dir / "corr_pearson.csv")
    correlations["spearman"].to_csv(output_dir / "corr_spearman.csv")

    # Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlations["pearson"], annot=False, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap (Pearson)")
    plt.tight_layout()
    plt.savefig(output_dir / "corr_heatmap.png", dpi=150)
    plt.close()

    print("Calculating VIF...")
    vif_df = calculate_vif(df_with_features)
    vif_df.to_csv(output_dir / "vif.csv", index=False)

    print("Calculating mutual information...")
    mi_results = calculate_mutual_info(df_with_features, horizons)
    for h, mi_df in mi_results.items():
        mi_df.to_csv(output_dir / f"mi_h{h}.csv", index=False)

    print("Checking for leaks...")
    leak_df = check_leaks(df_with_features)
    leak_df.to_csv(output_dir / "leak_check.csv", index=False)

    # High VIF
    high_vif = vif_df[vif_df["high_vif"]]
    high_vif.to_csv(output_dir / "vif_high.csv", index=False)

    # Mapping table
    mapping = {
        "Ichimoku": {
            "old": [
                "ichimoku_tenkan",
                "ichimoku_kijun",
                "ichimoku_senkou_a",
                "ichimoku_senkou_b",
                "ichimoku_chikou",
            ],
            "new": ["ichimoku_diff_norm", "ichimoku_cross"],
        },
        "Donchian": {
            "old": ["donchian_position", "donchian_width_rel"],
            "new": [
                "donchian_pos_20",
                "donchian_slope_20",
                "donchian_pos_55",
                "donchian_slope_55",
            ],
        },
        "KalmanFilter": {
            "old": ["kalman_filtered"],
            "new": ["kalman_residual", "kalman_residual_diff", "kalman_zscore"],
        },
        "RegimeClustering": {
            "old": ["regime_cluster"],
            "new": ["regime_cluster_0", "regime_cluster_1"],
        },
    }
    mapping_df = pd.DataFrame(
        [
            {
                "feature": k,
                "old_columns": ", ".join(v["old"]),
                "new_columns": ", ".join(v["new"]),
            }
            for k, v in mapping.items()
        ]
    )
    mapping_df.to_csv(output_dir / "column_mapping.csv", index=False)

    print(f"Results saved to {output_dir}")

    # サマリー
    high_vif = vif_df[vif_df["high_vif"]]
    warnings = leak_df[leak_df["warning"]]

    print(f"\nSummary:")
    print(f"High VIF features (>10): {len(high_vif)}")
    print(f"Leak warnings: {len(warnings)}")
    print(f"Correlation heatmap saved to corr_heatmap.png")
    print(f"Column mapping saved to column_mapping.csv")

    if len(high_vif) > 0:
        print("High VIF features:")
        for _, row in high_vif.iterrows():
            print(f"  {row['feature']}: {row['vif']:.2f}")

    if len(warnings) > 0:
        print("Leak warnings:")
        for _, row in warnings.iterrows():
            print(f"  {row['feature']}: {row['reason']}")


if __name__ == "__main__":
    main()
