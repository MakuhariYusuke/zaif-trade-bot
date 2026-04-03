# 704# Task 3: Sell 全レジーム損失の構造分析 + spread_capture 改善

## 背景
3日間のライブデータ分析結果:

| Side×Regime | Count | Avg PnL | Total |
|-------------|-------|---------|-------|
| sell+trending_down | 64 | -1.17 | -74.84 |
| sell+ranging | 113 | -0.55 | -61.65 |
| sell+trending_up | 36 | -0.87 | -31.42 |
| buy+trending_up | 41 | -0.56 | -22.96 |
| buy+trending_down | 65 | +0.36 | +23.58 |
| buy+ranging | 110 | +0.53 | +58.80 |

**sell が全レジームでマイナス** — これは offset (spread capture) が不十分であることを示唆。
- spread_capture_bps avg: -0.52bps (約定価格がスプレッドの半分すら回収できていない)

## 診断タスク

### 1. sell offset pipeline ステージ分析
`scripts/v460/analysis/` に新規分析スクリプトを作成:

**ファイル**: `scripts/v460/analysis/analyze_704_sell_offset_pipeline.py`

```python
"""704# sell offset pipeline 分析.

fill_records から sell 約定の offset pipeline stages を分解し、
各ステージの寄与と最終 effective_offset_ratio を分析する。
"""
```

分析内容:
1. **effective_offset_ratio 分布** (sell vs buy): ヒストグラム出力
2. **offset_stages 別寄与**: sell fill の last_offset_stages JSON から各ステージの寄与を集計
3. **spread_capture_bps vs effective_offset_ratio 相関**: offset を上げれば capture が改善するか
4. **sell_hour_offset_boost 効果検証**: boost 適用時間帯 vs 非適用時間帯の capture 比較
5. **結論**: offset をどの程度上げれば capture ≥ 0 になるかの推定

### 2. sell_offset_base の調整提案
分析結果に基づいて、以下のいずれかを提案する形式でスクリプトの最後に出力:
- A: `min_offset_ratio` 引き上げ (グローバル)
- B: sell-only offset floor の追加 (新パラメータ)
- C: sell_hour_offset_boost の全時間帯基準値引き上げ

### 出力
- `analysis_results/704_sell_offset_pipeline.json`: 構造化結果
- stdout: テキストサマリー

## テスト
- `tests/unit/v460/test_704_sell_offset_analysis.py`
  - test_analysis_script_imports: モジュール import が壊れない
  - test_offset_stage_parsing: JSON offset_stages パース関数の正確性
  - test_spread_capture_correlation: 正の相関が検出されること (synthetic data)

## 制約
- `results/v460/fill_test/fill_records_*.jsonl` がない環境でもエラーなく終了 (空結果)
- 既存の `scripts/v460/analysis/` パターンに準拠
- pandas 不使用 (標準ライブラリ + statistics のみ)
