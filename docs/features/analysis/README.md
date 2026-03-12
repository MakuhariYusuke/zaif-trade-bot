# Features Analysis Documentation

このディレクトリには、SAC v427特徴量分析関連のドキュメントとツールが含まれています。

## 📁 ディレクトリ構造

```
docs/features/analysis/
├── README.md                           # このファイル
├── Enhanced_Feature_Analysis_Guide.md  # 拡張分析ツール使用ガイド
├── SAC_v427_Feature_Analysis_Report_20251025.md  # 実際の分析レポート
├── run_enhanced_analysis.py            # 拡張分析実行スクリプト
├── feature_importance_analysis.py      # 特徴量重要度分析スクリプト
├── test_efficiency_improvements.py     # 効率改善テストスクリプト
├── results/                            # 分析結果出力ディレクトリ
│   ├── real_data/                      # 実データ分析結果
│   │   ├── sac_v427_feature_analysis_20251025_004717.json
│   │   ├── analysis_summary_20251025_004717.txt
│   │   └── harmful_features_20251025_004717.txt
│   └── ...
└── __pycache__/
```

## 🚀 クイックスタート

### 1. 拡張特徴量分析の実行

```bash
# 実データでの分析
python run_enhanced_analysis.py --data ../data/btc_jpy_featured_dataset.csv

# カスタム出力ディレクトリ
python run_enhanced_analysis.py \
  --data your_data.csv \
  --target-column future_return \
  --output-dir results/custom_analysis
```

### 2. 特徴量重要度分析

```python
from feature_importance_analysis import analyze_current_features
results = analyze_current_features(feature_engineer)
```

### 3. 効率改善テスト

```bash
python test_efficiency_improvements.py
```

## 📊 利用可能なツール

### Enhanced Feature Analyzer (拡張版)
- **ファイル**: `run_enhanced_analysis.py`
- **機能**:
  - Harmful特徴量の高度な自動判定
  - SAC v427特化の品質分析
  - 実行可能な改善推奨の生成
- **出力**: JSONレポート、テキストサマリー、有害特徴量リスト

### Feature Importance Analyzer
- **ファイル**: `feature_importance_analysis.py`
- **機能**:
  - 特徴量の重要度ランキング
  - 相関分析と冗長性検出
  - 削除推奨の生成

### Efficiency Test Suite
- **ファイル**: `test_efficiency_improvements.py`
- **機能**:
  - 特徴量生成の性能テスト
  - メモリ使用量分析
  - 品質メトリクス評価

## 📋 分析結果レポート

### 最新の分析結果 (2025年10月25日)

**データセット**: BTC/JPY リアルデータ (352行 × 117特徴量)

#### 主要発見
- **有害特徴量**: 67個検出 (57.3%)
- **優良品質特徴量**: 50個 (42.7%)
- **クリティカル問題**: 2個 (dividends, stock splits - 完全に定数)

#### 問題カテゴリ
1. **過度な相関**: 65個 (OHLCVデータ間の相関過多)
2. **定数特徴量**: 2個 (dividends, stock splits)
3. **その他**: 0個

#### 推奨アクション
1. **即時実行**: Critical特徴量の削除
2. **検討事項**: 相関ベースの特徴量選択の実装
3. **改善策**: 特徴量エンジニアリングの最適化

詳細は [`SAC_v427_Feature_Analysis_Report_20251025.md`](SAC_v427_Feature_Analysis_Report_20251025.md) を参照してください。

## 🎯 使用ガイドライン

### 分析の実行タイミング
- 新しい特徴量セットの作成時
- モデル性能の低下時
- データ分布の変化時
- 定期的な品質監視（週次/月次）

### 結果の解釈
1. **Critical特徴量**: 無条件で削除
2. **Moderate特徴量**: 重要度分析と比較検討
3. **Minor特徴量**: 相関パターンに基づく選択

### 品質基準
- **Excellent**: 品質スコア ≥ 0.8
- **Good**: 品質スコア ≥ 0.6
- **Fair**: 品質スコア ≥ 0.4
- **Poor**: 品質スコア < 0.4

## 🔧 拡張とカスタマイズ

### 新しい判定基準の追加
`ztb/analysis/specialized/features/analyze_feature_selection.py` を編集：

```python
def _check_sac_v427_specific_issues(self, feature: str, data: pd.Series) -> List[str]:
    issues = []
    # カスタム判定ロジックを追加
    return issues
```

### 品質スコアのカスタマイズ
`_calculate_quality_scores` メソッドをドメイン特化の基準で拡張。

## 📈 分析履歴

| 日付 | データセット | 特徴量数 | 有害特徴量 | レポート |
|------|-------------|----------|------------|----------|
| 2025-10-25 | BTC/JPY リアル | 117 | 67 (57%) | [レポート](./SAC_v427_Feature_Analysis_Report_20251025.md) |

## 📚 関連リンク

- [特徴量拡張分析](../feature_expansion_analysis.md)
- [SAC v427 特徴量エンジニアリング](../../ztb/features/sac_v427_feature_engineering.py)
- [Enhanced Feature Analyzer API](../../ztb/analysis/specialized/features/analyze_feature_selection.py)

## 🤝 貢献

新しい分析手法や判定基準を追加する場合は：

1. `run_enhanced_analysis.py` に実行ロジックを追加
2. `Enhanced_Feature_Analysis_Guide.md` を更新
3. 新しい分析結果を `results/` に保存
4. このREADMEを更新

---

*最終更新: 2025年10月25日*
