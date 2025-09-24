# 特徴量評価ドキュメント

## 概要

このドキュメントでは、特徴量の評価方法、特にSharpe ratioベースの定量評価について説明します。

## Sharpe Ratioの定義

### 基本計算式

```
python
sharpe_ratio = (mean_return / std_return) * sqrt(252)
```

### 無リスク金利対応

超過リターンを計算する際に無リスク金利を考慮：

```python
excess_returns = returns - risk_free_rate / 252  # 日次無リスク金利
sharpe_ratio = (mean(excess_returns) / std(excess_returns)) * sqrt(252)
```

### ゼロ除算防止

標準偏差が0の場合の処理：

```python
if std_return == 0:
    return 0.0
```

## delta_sharpeの統計化

### 定義

delta_sharpe = 特徴量追加後のSharpe - ベースラインSharpe

### 安定化処理

- **最低試行数チェック**: `seeds × timesteps > 10,000` を満たさない場合はNaN
- **統計情報**: 平均/標準偏差/95%信頼区間を計算

### 計算例

```python
# ベースラインSharpe: [0.1, 0.15, 0.12, 0.18, 0.09]
# 特徴量追加後: [0.13, 0.17, 0.15, 0.21, 0.11]

delta_sharpe = {
    "mean": 0.032,      # 平均改善
    "std": 0.015,       # 標準偏差
    "ci95": [0.012, 0.052]  # 95%信頼区間
}
```

## harmful.mdとの接続

### 再評価条件

harmful判定された特徴量の再評価基準：

- **Wave間の補完効果**: Wave1/2特徴量との組み合わせ効果
- **特徴変換の導入**: 差分化・正規化・次元削減
- **Ablationの定量基準**: `delta_sharpe.mean > +0.05`
- **局面依存性の確認**: 特定Regimeでの有効性
- **軽量化後の見直し**: 計算効率改善

### 自動判定

```python
def should_re_evaluate(feature_result):
    ds = feature_result.get('delta_sharpe', {})
    if not ds:
        return False

    # 基準: mean > +0.05 かつ ci95下限 > 0
    return ds.get('mean', 0) > 0.05 and ds.get('ci95', [0, 0])[0] > 0
```

## 出力フォーマット

### JSONフォーマット

```json
{
  "feature": "Ichimoku",
  "success": true,
  "sharpe_stats": {
    "mean": 0.234,
    "std": 0.089,
    "ci95": [0.123, 0.345]
  },
  "delta_sharpe": {
    "mean": -0.037,
    "std": 0.012,
    "ci95": [-0.061, -0.013]
  },
  "runs": 5,
  "is_experimental": false
}
```

### CSVフォーマット

| feature | is_experimental | success | runs | sharpe_mean | sharpe_std | sharpe_ci_low | sharpe_ci_high | delta_sharpe_mean | delta_sharpe_std | delta_sharpe_ci_low | delta_sharpe_ci_high |
|---------|----------------|---------|------|-------------|------------|---------------|----------------|------------------|----------------|-------------------|-------------------|
| Ichimoku | false | true | 5 | 0.234 | 0.089 | 0.123 | 0.345 | -0.037 | 0.012 | -0.061 | -0.013 |

## CI/CD統合

### Weeklyレポート生成

毎週のablation実行結果から以下の情報を集計：

- 各特徴量のdelta_sharpe統計
- harmful再評価候補の自動判定
- 実験的特徴量の評価結果

### レポート例

```markdown
# Weekly Feature Evaluation Report

## Summary
- Total features evaluated: 15
- Re-evaluation candidates: 2
- Valid results: 12/15

## Delta Sharpe Results

| Feature | Mean | Std | CI95 Low | CI95 High | Status |
|---------|------|-----|----------|-----------|--------|
| Ichimoku | -0.037 | 0.012 | -0.061 | -0.013 | Maintain |
| RegimeClustering | 0.089 | 0.034 | 0.045 | 0.133 | **Re-evaluate** |
| Donchian | 0.023 | 0.018 | 0.002 | 0.044 | Monitor |

## Experimental Features

| Feature | Duration (ms) | NaN Rate | Delta Sharpe |
|---------|---------------|----------|--------------|
| MovingAverages | 5.2 | 0.012 | 0.034 |
| GradientSign | 1.1 | 0.000 | -0.008 |
```

## 使用方法

### ローカル実行

```bash
# 基本実行
python benchmarks/ablation_runner.py --set balanced --include-experimental

# 詳細設定
python benchmarks/ablation_runner.py \
  --set extended \
  --include-experimental \
  --num-runs 10 \
  --output results/custom_eval.json
```

### CI実行

GitHub Actionsで毎週自動実行：

```yaml
- name: Run ablation analysis
  run: |
    python benchmarks/ablation_runner.py \
      --set extended \
      --include-experimental \
      --num-runs 5

- name: Generate weekly report
  run: python scripts/generate_weekly_report.py
```

## 注意事項

- delta_sharpeの計算には十分な試行回数（>10,000サンプル）が必要
- 統計的有意性を確保するため、95%信頼区間を必ず確認
- experimental特徴量は本番環境での安定性を別途検証すること
