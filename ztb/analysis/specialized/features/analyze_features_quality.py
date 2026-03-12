"""
特徴量の詳細分析スクリプト
全特徴量をカテゴリ別に分類し、冗長・単独無意味な特徴を特定
"""

import pandas as pd

from ztb.io.data_loader import DataLoader

def main() -> None:
    # Load dataset
    df = DataLoader.load_csv_optimized("ml-dataset-enhanced-balanced.csv")

    print("=" * 100)
    print("特徴量分析レポート")
    print("=" * 100)
    print(f"総カラム数: {len(df.columns)}")
    print(f"データ行数: {len(df)}")

    # Exclude metadata columns
    exclude_cols = [
        "ts",
        "timestamp",
        "exchange",
        "pair",
        "episode_id",
        "side",
        "source",
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"\n分析対象特徴量: {len(feature_cols)}個")
    print("=" * 100)

    # Categorize features
    categories: dict[str, list[str]] = {
        "価格基本": [],
        "平均足(HeikinAshi)": [],
        "一目均衡表(Ichimoku)": [],
        "ボリンジャーバンド": [],
        "ドンチャン": [],
        "ケルトナー": [],
        "スーパートレンド": [],
        "トレンド系": [],
        "オシレーター系": [],
        "ボリューム系": [],
        "その他": [],
    }

    for col in feature_cols:
        col_lower = col.lower()

        if col in ["open", "high", "low", "close", "price", "volume"]:
            categories["価格基本"].append(col)
        elif col.startswith("HeikinAshi"):
            categories["平均足(HeikinAshi)"].append(col)
        elif col.startswith("Ichimoku"):
            categories["一目均衡表(Ichimoku)"].append(col)
        elif col.startswith("BB_") or col.startswith("Bollinger"):
            categories["ボリンジャーバンド"].append(col)
        elif col.startswith("Donchian"):
            categories["ドンチャン"].append(col)
        elif col.startswith("Keltner"):
            categories["ケルトナー"].append(col)
        elif col.startswith("Supertrend"):
            categories["スーパートレンド"].append(col)
        elif any(
            x in col
            for x in [
                "MACD",
                "EMA",
                "SMA",
                "TEMA",
                "KAMA",
                "VWAP",
                "PSAR",
                "ADX",
                "trend",
                "Trend",
                "Slope",
                "slope",
            ]
        ):
            categories["トレンド系"].append(col)
        elif any(
            x in col
            for x in ["RSI", "Stochastic", "CCI", "Williams", "MFI", "DI", "ROC"]
        ):
            categories["オシレーター系"].append(col)
        elif any(
            x in col for x in ["Volume", "OBV", "CMF", "volume", "liquidity", "tick"]
        ):
            categories["ボリューム系"].append(col)
        else:
            categories["その他"].append(col)

    # Print categorized features
    print("\n【カテゴリ別特徴量】\n")
    for cat_name, features in categories.items():
        if features:
            print(f"\n■ {cat_name} ({len(features)}個)")
            print("-" * 100)
            for i, feat in enumerate(features, 1):
                # Show basic stats
                non_null = df[feat].notna().sum()
                unique = df[feat].nunique()
                variance = (
                    df[feat].var() if pd.api.types.is_numeric_dtype(df[feat]) else None
                )

                var_str = f"分散={variance:.6f}" if variance is not None else "非数値"
                print(
                    f"  {i:2d}. {feat:50s} | 有効値={non_null:4d}/{len(df)} | ユニーク={unique:4d} | {var_str}"
                )

    print("\n" + "=" * 100)
    print("\n【問題のある特徴量の分析】\n")

    # 1. 平均足の分析
    print("■ 平均足(HeikinAshi)の問題")
    print("-" * 100)
    ha_features = categories["平均足(HeikinAshi)"]
    if ha_features:
        print("平均足の個別OHLC値:")
        for feat in ha_features:
            print(f"  - {feat}")
        print("\n⚠️  問題点:")
        print("  - 平均足の個別OHLC値は、色の連続性を見る以外は通常足と重複")
        print("  - Open/High/Low/Closeすべてを使うと情報が冗長")
        print("\n✅ 推奨:")
        print("  - 平均足色の連続カウント(陽線何本連続など)のみを使用")
        print("  - 個別OHLC値は削除し、通常足(close, open, high, low)を使用")

    # 2. 一目均衡表の分析
    print("\n■ 一目均衡表(Ichimoku)の問題")
    print("-" * 100)
    ichimoku_features = categories["一目均衡表(Ichimoku)"]
    if ichimoku_features:
        print("一目均衡表の各要素:")
        for feat in ichimoku_features:
            print(f"  - {feat}")
        print("\n⚠️  問題点:")
        print("  - Ichimoku_Chikou(遅行スパン)単独では意味がない")
        print("  - 個別の線(転換線、基準線など)だけでは不十分")
        print("\n✅ 推奨:")
        print("  - 雲との位置関係(価格 vs 先行スパン)")
        print("  - 転換線と基準線のクロス")
        print("  - 雲の厚み(先行スパンAとBの差)")
        print("  - 価格と雲の距離")
        print(
            "  → これらは既に Ichimoku_Composite_Signal, Ichimoku_Price_Cloud_Distance などで実装済み"
        )

    # 3. 分散が極端に低い特徴
    print("\n■ 分散が極端に低い特徴(ほぼ定数)")
    print("-" * 100)
    low_variance_features = []
    for feat in feature_cols:
        if pd.api.types.is_numeric_dtype(df[feat]):
            var = df[feat].var()
            if var < 1e-10:
                low_variance_features.append((feat, var))

    if low_variance_features:
        for feat, var in sorted(low_variance_features, key=lambda x: x[1]):
            print(f"  - {feat:50s} | 分散={var:.2e}")
        print(f"\n⚠️  {len(low_variance_features)}個の特徴がほぼ定数 → 削除推奨")
    else:
        print("  ✅ 極端に低い分散の特徴はなし")

    # 4. NaNが多すぎる特徴
    print("\n■ 欠損値(NaN)が多い特徴(>10%)")
    print("-" * 100)
    high_nan_features = []
    for feat in feature_cols:
        nan_ratio = df[feat].isna().sum() / len(df)
        if nan_ratio > 0.1:
            high_nan_features.append((feat, nan_ratio))

    if high_nan_features:
        for feat, ratio in sorted(high_nan_features, key=lambda x: x[1], reverse=True):
            print(f"  - {feat:50s} | 欠損率={ratio*100:.1f}%")
        print(f"\n⚠️  {len(high_nan_features)}個の特徴で欠損率>10% → 要確認")
    else:
        print("  ✅ 欠損率が高い特徴はなし")

    # 5. 相関が非常に高いペア
    print("\n■ 高相関特徴ペア(相関係数>0.95)")
    print("-" * 100)
    numeric_features = [f for f in feature_cols if pd.api.types.is_numeric_dtype(df[f])]
    corr_matrix = df[numeric_features].corr().abs()

    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.95:
                high_corr_pairs.append(
                    (
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j],
                    )
                )

    if high_corr_pairs:
        print(f"  発見: {len(high_corr_pairs)}ペア")
        for feat1, feat2, corr in sorted(
            high_corr_pairs, key=lambda x: x[2], reverse=True
        )[:20]:
            print(f"  - {feat1:40s} <-> {feat2:40s} | 相関={corr:.4f}")
        if len(high_corr_pairs) > 20:
            print(f"  ... 他{len(high_corr_pairs)-20}ペア")
    else:
        print("  ✅ 高相関ペアなし")

    print("\n" + "=" * 100)
    print("\n【推奨される改善策】\n")

    print("1. 削除推奨特徴:")
    delete_candidates = []

    # 平均足のOHLC(色連続以外)
    for feat in ha_features:
        if feat in [
            "HeikinAshi_Open",
            "HeikinAshi_High",
            "HeikinAshi_Low",
            "HeikinAshi_Close",
        ]:
            delete_candidates.append(f"  - {feat} (通常足と冗長、色連続のみ有効)")

    # 低分散
    for feat, var in low_variance_features:
        delete_candidates.append(f"  - {feat} (分散={var:.2e}, ほぼ定数)")

    # 一目の遅行スパン単独
    if "Ichimoku_Chikou" in ichimoku_features:
        delete_candidates.append(
            "  - Ichimoku_Chikou (単独では無意味、他の線との組み合わせが必要)"
        )

    if delete_candidates:
        for item in delete_candidates[:15]:
            print(item)
        if len(delete_candidates) > 15:
            print(f"  ... 他{len(delete_candidates)-15}個")

    print("\n2. 追加推奨特徴(組み合わせ):")
    print("  - HeikinAshi_Consecutive_Color (平均足の色連続カウント)")
    print("  - Ichimoku_Tenkan_Kijun_Cross (転換線と基準線のクロス)")
    print("  - Ichimoku_Price_Above_Cloud (価格が雲より上かどうか)")
    print("  - Composite indicators already implemented (継続使用)")

    print("\n3. 保持すべき重要特徴:")
    important = [
        "close",
        "open",
        "high",
        "low",
        "volume",  # 基本
        "RSI",
        "MACD",
        "ADX",
        "ATR",
        "CCI",  # 主要指標
        "BB_Position",
        "Bollinger_Percent_B",  # ボリンジャー
        "Ichimoku_Composite_Signal",
        "Ichimoku_Price_Cloud_Distance",  # 一目(組み合わせ)
        "Supertrend_Direction",
        "Supertrend_Strength",  # スーパートレンド
    ]
    print("  " + ", ".join(important[:10]))
    print("  " + ", ".join(important[10:]))

    print("\n" + "=" * 100)
    print("分析完了")
    print("=" * 100)

if __name__ == "__main__":
    main()
