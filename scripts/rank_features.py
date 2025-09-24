#!/usr/bin/env python3
"""
Feature ranking script.
特徴量ランキングスクリプト
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    variance_inflation_factor = None  # 未定義エラーを回避
    print("Warning: statsmodels not available, VIF calculation skipped")

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.trading.features import get_feature_manager


def calculate_correlations(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """特徴量間の相関係数を計算"""
    available_features = [f for f in features if f in df.columns]
    if not available_features:
        return pd.DataFrame()

    corr_matrix = df[available_features].corr()
    return corr_matrix


def calculate_vif(df: pd.DataFrame, features: List[str]) -> Dict[str, float]:
    """VIF（分散拡大要因）を計算"""
    available_features = [f for f in features if f in df.columns]
    if not HAS_STATSMODELS or not available_features:
        return {f: 1.0 for f in features}

    vif_data = df[available_features].dropna()
    if vif_data.empty or len(vif_data.columns) < 2:
        return {f: 1.0 for f in features}

    # HAS_STATSMODELSがTrueの場合、variance_inflation_factorはNoneではない
    assert variance_inflation_factor is not None

    vif_dict = {}
    for i, col in enumerate(vif_data.columns):
        try:
            vif = variance_inflation_factor(vif_data.values, i)
            vif_dict[col] = vif
        except:
            vif_dict[col] = 1.0

    # すべてのfeaturesに値を設定
    result = {f: vif_dict.get(f, 1.0) for f in features}
    return result


def calculate_mutual_info(df: pd.DataFrame, features: List[str], target: pd.Series) -> Dict[str, float]:
    """相互情報量を計算"""
    mi_dict = {}
    for feature in features:
        if feature not in df.columns:
            mi_dict[feature] = 0.0
            continue
        try:
            # NaN除去
            valid_data = df[[feature]].join(target).dropna()
            if len(valid_data) < 10:
                mi_dict[feature] = 0.0
                continue

            mi = mutual_info_regression(
                valid_data[[feature]],
                valid_data[target.name],
                random_state=42
            )[0]
            mi_dict[feature] = mi
        except Exception as e:
            print(f"Error calculating MI for {feature}: {e}")
            mi_dict[feature] = 0.0

    return mi_dict


def calculate_auc(df: pd.DataFrame, features: List[str], target: pd.Series) -> Dict[str, float]:
    """AUCを計算（ロジスティック回帰、3-fold CV）"""
    auc_dict = {}
    for feature in features:
        if feature not in df.columns:
            auc_dict[feature] = 0.5
            continue
        try:
            # NaN除去
            valid_data = df[[feature]].join(target).dropna()
            if len(valid_data) < 30:  # CVに必要な最小サンプル
                auc_dict[feature] = 0.5
                continue

            X = valid_data[[feature]]
            y = valid_data[target.name]

            # スケーリング
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # ロジスティック回帰
            model = LogisticRegression(random_state=42, max_iter=1000)

            # 3-fold CV
            auc_scores = cross_val_score(
                model, X_scaled, y,
                cv=3, scoring='roc_auc'
            )

            auc_dict[feature] = np.mean(auc_scores)

        except Exception as e:
            print(f"Error calculating AUC for {feature}: {e}")
            auc_dict[feature] = 0.5

    return auc_dict


def load_ablation_results(ablation_csv: Path) -> Dict[str, Dict[str, float]]:
    """アブレーション結果を読み込み"""
    if not ablation_csv.exists():
        print(f"Warning: {ablation_csv} not found, using default values")
        return {}

    df = pd.read_csv(ablation_csv)
    results = {}

    for _, row in df.iterrows():
        feature = row['feature']
        if feature == 'baseline':
            continue

        results[feature] = {
            'delta_sharpe_like': row.get('delta_sharpe_like', 0),
            'delta_fps': row.get('delta_fps', 0),
            'delta_env_step_ms': row.get('delta_env_step_ms', 0)
        }

    return results


def calculate_priority_score(
    mi: float, auc: float, delta_sharpe: float,
    speed_gain: float, stability: float
) -> float:
    """優先度スコアを計算"""
    # 正規化（0-1スケール、仮定）
    mi_norm = min(mi / 1.0, 1.0)  # MIの上限を1.0と仮定
    auc_norm = auc  # AUCはすでに0-1
    sharpe_norm = max(0, (delta_sharpe + 1) / 2)  # -1 to 1 -> 0 to 1

    # speed_gain: 速くなるほど正（負のdelta_env_step_ms）
    speed_norm = max(0, min(1, (speed_gain + 50) / 100))  # -50ms to +50ms -> 0 to 1

    stability_norm = min(stability, 1.0)  # 0-1

    # 重み付け
    score = (
        0.45 * sharpe_norm +
        0.25 * mi_norm +
        0.15 * auc_norm +
        0.10 * speed_norm +
        0.05 * stability_norm
    )

    return score


def determine_decision(score: float) -> str:
    """決定を判定"""
    if score >= 0.65:
        return 'Keep'
    elif score >= 0.45:
        return 'Candidate'
    else:
        return 'Drop'


def main():
    parser = argparse.ArgumentParser(description='Feature ranking analysis')
    parser.add_argument('--waves', type=str, default='1', help='Comma-separated waves')
    parser.add_argument('--label', type=str, default='next_return', help='Target label column')
    parser.add_argument('--ablation-csv', type=str, default=None, help='Ablation results CSV')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--data-rows', type=int, default=10000, help='Number of data rows for analysis')

    args = parser.parse_args()

    # マネージャー初期化
    manager = get_feature_manager()

    # Wave特徴量取得
    waves = [int(w) for w in args.waves.split(',')]
    features = []
    for wave in waves:
        features.extend(manager.get_enabled_features(wave))

    if not features:
        print("No features enabled")
        return

    # 合成データ生成
    np.random.seed(42)
    n_rows = args.data_rows
    dates = pd.date_range('2024-01-01', periods=n_rows, freq='1H')

    returns = np.random.normal(0, 0.02, n_rows)
    price = 100 * np.exp(np.cumsum(returns))

    high = price * (1 + np.random.uniform(0, 0.03, n_rows))
    low = price * (1 - np.random.uniform(0, 0.03, n_rows))
    close = price
    volume = np.random.uniform(1000, 10000, n_rows)

    df = pd.DataFrame({
        'ts': dates.astype(np.int64) // 10**9,
        'close': close,
        'high': high,
        'low': low,
        'volume': volume,
        'exchange': 'synthetic',
        'pair': 'BTC/USD',
        'episode_id': 0
    })

    # 特徴量計算
    df_with_features = manager.compute_features(df, wave=max(waves))

    print(f"df_with_features shape: {df_with_features.shape}")
    print(f"df_with_features columns: {list(df_with_features.columns)[:10]}")
    print(f"'close' in columns: {df_with_features.columns.tolist().count('close')}")
    print(f"type of df_with_features['close']: {type(df_with_features['close'])}")
    print(f"close columns: {[col for col in df_with_features.columns if 'close' in col]}")

    # ラベル作成
    if args.label == 'next_return':
        if 'next_return' not in df_with_features.columns:
            df_with_features['next_return'] = df_with_features['close'].shift(-1) - df_with_features['close']
        df_with_features['label'] = (df_with_features['next_return'] > 0).astype(int)
    else:
        df_with_features['label'] = df_with_features[args.label] if args.label in df_with_features.columns else 0

    # アブレーション結果読み込み
    ablation_csv = None
    if args.ablation_csv:
        ablation_csv = Path(args.ablation_csv)
    else:
        # 最新のアブレーション結果を探す
        ranking_dir = Path('reports/feature_ranking')
        if ranking_dir.exists():
            ablation_files = list(ranking_dir.glob('ablation_*.csv'))
            if ablation_files:
                ablation_csv = max(ablation_files, key=lambda x: x.stat().st_mtime)

    ablation_results = load_ablation_results(ablation_csv) if ablation_csv else {}

    # 出力ディレクトリ
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('reports/feature_ranking') / date_str

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Calculating correlations...")
    corr_df = calculate_correlations(df_with_features, features)

    print("Calculating VIF...")
    vif_dict = calculate_vif(df_with_features, features)

    print("Calculating mutual information...")
    mi_dict = calculate_mutual_info(df_with_features, features, df_with_features['label'])

    print("Calculating AUC...")
    auc_dict = calculate_auc(df_with_features, features, df_with_features['label'])

    # 結果集計
    results = []
    for feature in features:
        # 相関クラスタ（簡易：相関係数0.9以上の特徴量数）
        if feature in corr_df.columns:
            high_corr = (corr_df[feature].abs() > 0.9).sum() - 1  # 自分除く
        else:
            high_corr = 0

        vif = vif_dict.get(feature, 1.0)
        mi = mi_dict.get(feature, 0.0)
        auc = auc_dict.get(feature, 0.5)

        # アブレーション結果
        ablation = ablation_results.get(feature, {})
        delta_sharpe = ablation.get('delta_sharpe_like', 0)
        delta_fps = ablation.get('delta_fps', 0)
        delta_env_ms = ablation.get('delta_env_step_ms', 0)

        # speed_gain: ベースラインより速いほど正
        speed_gain = -delta_env_ms  # 負のdelta_env_ms = 速くなる

        # stability: シード間分散の小ささ（仮定: 固定値）
        stability = 0.8  # TODO: 実際の計算が必要

        # スコア計算
        score = calculate_priority_score(mi, auc, delta_sharpe, speed_gain, stability)
        decision = determine_decision(score)

        results.append({
            'feature': feature,
            'MI': mi,
            'AUC': auc,
            'corr_cluster': high_corr,
            'VIF': vif,
            'delta_sharpe_like': delta_sharpe,
            'speed_gain': speed_gain,
            'env_step_ms': delta_env_ms,
            'score': score,
            'decision': decision
        })

    # CSV出力
    df_results = pd.DataFrame(results)
    info_csv = output_dir / 'info.csv'
    df_results.to_csv(info_csv, index=False)

    # ランキングCSV
    ranking_csv = output_dir / 'ranking.csv'
    ranking_df = df_results.sort_values('score', ascending=False)
    ranking_df.to_csv(ranking_csv, index=False)

    # レポート生成
    report_md = output_dir / 'report.md'
    with open(report_md, 'w', encoding='utf-8') as f:
        f.write("# Feature Ranking Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Summary\n\n")
        decisions = df_results['decision'].value_counts()
        f.write(f"- Keep: {decisions.get('Keep', 0)}\n")
        f.write(f"- Candidate: {decisions.get('Candidate', 0)}\n")
        f.write(f"- Drop: {decisions.get('Drop', 0)}\n\n")

        f.write("## Top 5 Features\n\n")
        top5 = ranking_df.head(5)
        f.write("| Feature | Score | ΔSharpe | MI | AUC | Decision |\n")
        f.write("|---------|-------|---------|----|-----|----------|\n")
        for _, row in top5.iterrows():
            feature = row['feature'] if 'feature' in row else "NaN"
            score = f"{row['score']:.3f}" if pd.notnull(row['score']) else "NaN"
            delta_sharpe = f"{row['delta_sharpe_like']:.3f}" if pd.notnull(row['delta_sharpe_like']) else "NaN"
            mi = f"{row['MI']:.3f}" if pd.notnull(row['MI']) else "NaN"
            auc = f"{row['AUC']:.3f}" if pd.notnull(row['AUC']) else "NaN"
            decision = row['decision'] if pd.notnull(row['decision']) else "NaN"
            f.write(f"| {feature} | {score} | {delta_sharpe} | {mi} | {auc} | {decision} |\n")
        f.write("\n## Bottom 5 Features\n\n")
        bottom5 = ranking_df.tail(5)
        f.write("| Feature | Score | ΔSharpe | MI | AUC | Decision |\n")
        f.write("|---------|-------|---------|----|-----|----------|\n")
        for _, row in bottom5.iterrows():
            feature = row['feature'] if 'feature' in row else "NaN"
            score = f"{row['score']:.3f}" if pd.notnull(row['score']) else "NaN"
            delta_sharpe = f"{row['delta_sharpe_like']:.3f}" if pd.notnull(row['delta_sharpe_like']) else "NaN"
            mi = f"{row['MI']:.3f}" if pd.notnull(row['MI']) else "NaN"
            auc = f"{row['AUC']:.3f}" if pd.notnull(row['AUC']) else "NaN"
            decision = row['decision'] if pd.notnull(row['decision']) else "NaN"
            f.write(f"| {feature} | {score} | {delta_sharpe} | {mi} | {auc} | {decision} |\n")

        # Balanced Set
        keep_features = ranking_df[ranking_df['decision'] == 'Keep']['feature'].tolist()
        balanced = keep_features + ranking_df[ranking_df['decision'] == 'Candidate']['feature'].tolist()[:5]
        f.write(f"\n**Balanced Set** ({len(balanced)} features): {', '.join(balanced[:15])}\n\n")

if __name__ == '__main__':
    main()