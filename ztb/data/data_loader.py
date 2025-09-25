# データ品質詳細チェックスクリプト
# 外れ値検出と詳細な分布分析

import sys
import os
import pandas as pd
import numpy as np
import numpy.ma as ma
from pathlib import Path
from scipy import stats
import json
from typing import Union

# プロジェクトルートをパスに追加
# sys.path.append(str(Path(__file__).parent.parent.parent))

def detect_outliers_iqr(data, column):
    """IQR法による外れ値検出"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(data, column, threshold=3):
    """Z-score法による外れ値検出"""
    series = data[column]
    # stats.zscoreはMaskedArrayを返すことがあるため、通常のnumpy配列に変換
    z_scores_raw = stats.zscore(series, nan_policy='omit')
    
    # MaskedArrayをnumpy配列に変換し、マスクされた値をNaNで埋める
    # np.ma.filledはMaskedArrayと通常のndarrayの両方を処理できる
    z_scores_unmasked = ma.filled(z_scores_raw, np.nan)
    
    # NaNをそのまま除外（0に変換しない）
    z_scores_series = pd.Series(np.abs(z_scores_unmasked), index=series.index)
    outliers = data[z_scores_series > threshold]
    return outliers

def analyze_feature_distributions(multiplier: float = 1.0, config_path: Union[str, Path, None] = None):
    """特徴量分布の詳細分析"""
    print("=== 詳細データ品質チェック ===")

    # 設定ファイル読み込み
    env_config_path = os.environ.get("RL_CONFIG_PATH")
    if config_path is None:
        config_path = env_config_path or Path(__file__).parent.parent.parent / "config/rl_config.json"
    
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        base_threshold = config.get('data_quality', {}).get('outlier_threshold_percent', 10.0) / 100.0
    else:
        base_threshold = 0.10  # デフォルト10%

    outlier_threshold = base_threshold * multiplier
    print(f"外れ値閾値: ±{outlier_threshold * 100}% (multiplier: {multiplier})")
    data_paths = [
        Path(__file__).parent.parent.parent / "data/features/2025/04/sample_04.parquet",
        Path(__file__).parent.parent.parent / "data/features/2025/05/sample_05.parquet",
        Path(__file__).parent.parent.parent / "data/features/2025/06/sample_06.parquet",
    ]

    all_data = []
    for path in data_paths:
        if path.exists():
            df = pd.read_parquet(path)
            all_data.append(df)

    if not all_data:
        print("❌ データファイルが見つかりません")
        return

    # データを結合
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"結合データ形状: {combined_df.shape}")
    print(f"データ期間: {combined_df['ts'].min()} から {combined_df['ts'].max()}")

    # 欠損値の詳細チェック
    print("\n=== 欠損値分析 ===")
    null_counts = combined_df.isnull().sum()
    total_nulls = null_counts.sum()
    print(f"総欠損値: {total_nulls}")

    if total_nulls > 0:
        print("欠損値のある列:")
        non_zero_nulls = null_counts[null_counts > 0]
        for col, count in non_zero_nulls.items():
            null_ratio = count / len(combined_df) * 100
            print(f"{col}: {count}件 ({null_ratio:.2f}%)")
    else:
        print("✅ 欠損値なし - データ品質良好")

    # 外れ値検出
    print("\n=== 外れ値検出 ===")
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
    key_features = ['price', 'volume', 'sma_5', 'sma_10', 'rsi_14', 'macd']

    outlier_summary = []

    # 価格変動の極端値チェック（フラッシュクラッシュ対応）
    if 'price' in combined_df.columns:
        price_changes = combined_df['price'].pct_change()
        extreme_changes = combined_df[price_changes.abs() > outlier_threshold]
        print(f"価格変動 ±{outlier_threshold * 100}% 超: {len(extreme_changes)} 件")
        if len(extreme_changes) > 0:
            print("極端変動サンプル:")
            for idx, _ in extreme_changes.head(3).iterrows():
                change_pct = price_changes.at[idx] * 100
                print(f"  Index {idx}: 変動率 {change_pct:.2f}%")

    for col in key_features:
        if col in numeric_cols:
            # IQR法
            outliers_iqr, lower, upper = detect_outliers_iqr(combined_df, col)
            # Z-score法
            outliers_z = detect_outliers_zscore(combined_df, col)

            outlier_info = {
                'feature': col,
                'iqr_outliers': len(outliers_iqr),
                'zscore_outliers': len(outliers_z),
                'iqr_percentage': len(outliers_iqr) / len(combined_df) * 100,
                'zscore_percentage': len(outliers_z) / len(combined_df) * 100,
                'lower_bound': lower,
                'upper_bound': upper
            }
            outlier_summary.append(outlier_info)

    # 外れ値サマリーを表示
    print("特徴量別外れ値検出結果:")
    print("-" * 80)
    print(f"{'Feature':<12} {'IQR Outliers':<20} {'Z-score Outliers':<22} {'IQR Range'}")
    print("-" * 80)

    for info in outlier_summary:
        print(f"{info['feature']:<12} {info['iqr_outliers']} ({info['iqr_percentage']:.2f}%) "
              f"Z-score: {info['zscore_outliers']} ({info['zscore_percentage']:.2f}%) "
              f"範囲: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")

    # 分布の詳細分析
    print("\n=== 分布詳細分析 ===")
    for col in key_features[:3]:  # 最初の3つの特徴量のみ詳細分析
        if col in numeric_cols:
            data = combined_df[col].dropna()
            print(f"\n{col} の詳細統計:")

            # 基本統計
            desc = data.describe()
            print(f"  平均: {desc['mean']:.2f}")
            print(f"  標準偏差: {desc['std']:.2f}")
            print(f"  最小値: {desc['min']:.2f}")
            print(f"  第1四分位数: {desc['25%']:.2f}")
            print(f"  中央値: {desc['50%']:.2f}")
            print(f"  第3四分位数: {desc['75%']:.2f}")
            print(f"  最大値: {desc['max']:.2f}")

            # 歪度と尖度
            skewness = data.skew()
            kurtosis = data.kurtosis()
            print(f"  歪度: {skewness:.4f}")
            print(f"  尖度: {kurtosis:.4f}")

            # 正規性検定
            if len(data) > 5000:  # サンプルサイズが十分な場合のみ
                sample_size = min(len(data), 5000)
                _, p_value = stats.shapiro(data.sample(sample_size, random_state=42))
                print(f"  正規性検定 p値: {p_value:.6f} ({'正規分布' if p_value > 0.05 else '非正規分布'})")

    print("\n=== データ連続性チェック ===")
    # ts列がdatetime型でない場合のみ変換（unit指定はint型のときのみ）
    if not pd.api.types.is_datetime64_any_dtype(combined_df['ts']):
        try:
            # int型ならunit='s'で変換、そうでなければunitなし
            if pd.api.types.is_integer_dtype(combined_df['ts'].dtype):
                combined_df['ts'] = pd.to_datetime(combined_df['ts'], unit='s')
            else:
                combined_df['ts'] = pd.to_datetime(combined_df['ts'])
        except Exception as e:
            print(f"❌ ts列のdatetime変換エラー: {e}")
            return # 変換に失敗した場合は以降の処理をスキップ

    if pd.api.types.is_datetime64_any_dtype(combined_df['ts']):
        time_diffs = combined_df['ts'].sort_values().diff()
        print("タイムスタンプ間隔の統計:")
        print(time_diffs.describe())
        print(f"  最頻間隔: {time_diffs.mode().iloc[0] if not time_diffs.mode().empty else 'N/A'}")
        print(f"  最大間隔: {time_diffs.max()}")
        print(f"  欠損間隔数 (5分以上): {(time_diffs > pd.Timedelta('5min')).sum()}")
    else:
        print("ts列がdatetime型でないため、連続性チェックをスキップします。")

    # 特徴量の相関チェック
    print("\n=== 特徴量相関チェック ===")
    # 存在するカラムのみで相関を計算
    available_features = [col for col in key_features if col in combined_df.columns]
    if available_features:
        correlation_matrix = combined_df[available_features].corr()
        print("主要特徴量の相関係数:")
        print(correlation_matrix.round(3))

        # 高相関のペアを特定
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if isinstance(corr, (int, float)) and abs(corr) > 0.8:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr
                    ))

        if high_corr_pairs:
            print("\n高相関の特徴量ペア (相関係数 > 0.8):")
            for feat1, feat2, corr in high_corr_pairs:
                print(f"{feat1} - {feat2}: 相関係数 {corr:.3f}")
        else:
            print("\n高相関の特徴量ペアなし")
    else:
        print("主要特徴量がデータに存在しません。")

def main():
    """メイン実行関数"""
    print("🔍 詳細データ品質チェック開始")
    
    # コマンドライン引数でmultiplierとconfig_pathを取得
    import argparse
    parser = argparse.ArgumentParser(description="詳細データ品質チェックスクリプト")
    parser.add_argument(
        '--multiplier',
        type=float,
        default=1.0,
        help='外れ値検出の閾値倍率（例: 1.5にすると閾値が1.5倍になります）。データの分布や異常値の許容度に応じて調整してください。'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help='設定ファイル（JSON形式）のパスを指定します。未指定の場合は環境変数RL_CONFIG_PATHまたはデフォルトパスが使用されます。'
    )
    args = parser.parse_args()

    try:
        analyze_feature_distributions(multiplier=args.multiplier, config_path=args.config_path)
        print("\n" + "=" * 60)
        print("✅ 詳細データ品質チェック完了")

    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()