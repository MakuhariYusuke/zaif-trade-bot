# 拡張特徴量分析ツール使用ガイド

## 📋 概要

このツールは、既存のEnhancedFeatureAnalyzerを拡張し、SAC v427特徴量の高度な品質分析とharmful判定を行うシステムです。

## 🚀 クイックスタート

### 基本的な実行方法

```bash
# サンプルデータでのテスト
python docs/features/analysis/run_enhanced_analysis.py

# 実際のデータでの分析
python docs/features/analysis/run_enhanced_analysis.py \
  --data data/your_featured_data.csv \
  --target-column future_return \
  --output-dir docs/features/analysis/results/your_analysis
```

### コマンドラインオプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--data` | 分析対象のCSVファイルパス | サンプルデータ生成 |
| `--target-column` | ターゲット列名 | future_return |
| `--output-dir` | 出力ディレクトリ | docs/features/analysis/results |

## 📊 出力ファイル

各分析実行で以下の3つのファイルが生成されます：

### 1. 総合レポート (JSON)
**ファイル名**: `sac_v427_feature_analysis_YYYYMMDD_HHMMSS.json`

完全な分析結果を含む構造化データ：
- 品質スコア詳細
- 有害特徴量の完全リスト
- SAC v427カテゴリ別分析
- 実行可能な推奨事項

### 2. テキストサマリー
**ファイル名**: `analysis_summary_YYYYMMDD_HHMMSS.txt`

人間可読な概要レポート：
- 主要統計情報
- トップの問題特徴量
- 推奨アクション

### 3. 有害特徴量リスト
**ファイル名**: `harmful_features_YYYYMMDD_HHMMSS.txt`

削除対象特徴量の詳細リスト：
- 特徴量名
- 深刻度レベル
- 問題の詳細
- 削除推奨理由

## 🔍 判定基準の詳細

### Harmful判定カテゴリ

#### 1. Critical (即時削除)
- **NaN率 > 10%**
- **分散 = 0** (定数特徴量)
- **ゼロ値率 > 80%**

#### 2. Moderate (検討推奨)
- **外れ値率 > 30%**
- **極端外れ値率 > 5%** (Z-score > 3)

#### 3. Minor (注意)
- **過度相関 > 95%** (7個以上の特徴量と相関)
- **SAC v427特化問題** (レジーム/技術指標の異常)

### SAC v427特化判定

#### 市場レジーム特徴量
- 値が0-1の範囲外
- 異常な分散パターン

#### 相関特徴量
- 相関係数が-1から1の範囲外
- 計算の不安定性

#### 技術指標
- 標準偏差が平均の10倍以上
- 連続値の変化が激しすぎる

## 📈 品質スコア計算

各特徴量に0-1の品質スコアを付与：

```
品質スコア = 1.0 - (NaNペナルティ + 分散ペナルティ + 外れ値ペナルティ + ゼロ値ペナルティ)
```

### 品質カテゴリ
- **Excellent**: スコア ≥ 0.8
- **Good**: スコア ≥ 0.6
- **Fair**: スコア ≥ 0.4
- **Poor**: スコア < 0.4

## 🎯 活用シナリオ

### 1. 特徴量エンジニアリングパイプライン
```python
from docs.features.analysis.run_enhanced_analysis import run_enhanced_feature_analysis

# 新しい特徴量セットの品質チェック
results = run_enhanced_feature_analysis("data/new_features.csv")

# 有害特徴量の自動除去
harmful_features = list(results['harmful_features'].keys())
df_cleaned = df.drop(columns=harmful_features)
```

### 2. 定期的な品質監視
```bash
# 毎日のデータ更新時の自動チェック
python docs/features/analysis/run_enhanced_analysis.py \
  --data data/daily_featured_data.csv \
  --output-dir reports/daily_quality_checks/
```

### 3. モデル開発前の特徴量選択
```python
# 特徴量選択前の品質フィルタリング
quality_report = run_enhanced_feature_analysis(featured_data_path)
excellent_features = quality_report['categories']['excellent']
selected_features = correlation_filter.select_from(excellent_features)
```

## 🔧 拡張方法

### 新しい判定基準の追加

`ztb/analysis/specialized/features/analyze_feature_selection.py` の
`_check_sac_v427_specific_issues` メソッドに判定ロジックを追加：

```python
def _check_sac_v427_specific_issues(self, feature: str, data: pd.Series) -> List[str]:
    issues = []

    # 既存の判定...

    # 新しい判定の追加例
    if 'new_category' in feature.lower():
        if data.skew() > 2.0:  # 歪度が大きい場合
            issues.append("high_skewness")

    return issues
```

### カスタム品質スコア

`_calculate_quality_scores` メソッドを拡張して、
ドメイン特化の品質基準を追加。

## 📊 分析結果の解釈

### 一般的なパターン

1. **OHLCVデータの相関過多**
   - 金融データでは一般的
   - 特徴量選択アルゴリズムで解決

2. **定数特徴量の出現**
   - データ収集の問題を示唆
   - 即時削除推奨

3. **外れ値の多さ**
   - データ品質の問題
   - ロバストな特徴量設計が必要

### 意思決定の指針

- **Critical**: 無条件で削除
- **Moderate**: 重要度分析と比較検討
- **Minor**: 相関パターンに基づく選択的削除

## 🚨 注意事項

1. **統計的判定の限界**
   - 相関 ≠ 因果関係
   - ドメイン知識との組み合わせを推奨

2. **データ依存性**
   - 判定結果はデータセットに依存
   - 異なるデータセットでの再検証を推奨

3. **計算コスト**
   - 大規模データセットではメモリ使用に注意
   - 必要に応じてサンプリングを検討

## 📚 関連ドキュメント

- [SAC v427 Feature Analysis Report](./SAC_v427_Feature_Analysis_Report_20251025.md)
- [Feature Expansion Analysis](../feature_expansion_analysis.md)
- [Enhanced Feature Analyzer API](../api/enhanced_feature_analyzer.md)

---

*最終更新: 2025年10月25日*
