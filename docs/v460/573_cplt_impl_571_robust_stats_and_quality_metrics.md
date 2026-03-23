# 573# CPLT: 571# robust_stats + 執行品質比較セクション実装

## 概要

Gemini ドラフト 571# に基づき、以下を実装:

### Task B: `ztb/utils/robust_stats.py`

eDRC 入力保護用ロバスト統計ユーティリティ。`RobustStats` クラスに 4 つの静的メソッド:

| メソッド | 用途 |
|---------|------|
| `clip_outliers_mad(data, threshold)` | MAD ベース外れ値クリッピング |
| `robust_ema(current, prev, alpha, sigma_limit)` | スパイク保護付き EMA |
| `asymmetric_ema(current, prev, alpha_up, alpha_down)` | 非対称 EMA (逆行感度向上) |
| `median_filter_fast(buffer)` | 中央値ノイズフィルタ |

設計方針:
- **ステートレス**: 全 `@staticmethod`、インスタンス化不要 (`__slots__ = ()`)
- **メモリ安全**: 内部バッファなし
- **numpy 依存のみ**

### Task A: `section_execution_quality_comparison()` in analyze_fill_logs.py

Kissell & Glantz 執行品質指標による加法 vs 乗法パイプライン比較:
- `execution_additive_enabled` フラグで fill を分類
- Spread Capture (bps), Adverse Selection Cost (bps), Net Spread PnL, ICR を算出
- $ICR = \frac{\text{Spread Capture}}{|AS\ Cost|}$

## テスト

`tests/unit/v460/test_571_robust_stats.py` — 21 テスト全合格:
- `TestClipOutliersMad` (5): 空配列、均一データ、正常データ、外れ値クリップ、閾値感度
- `TestRobustEma` (5): 通常更新、上方スパイク、下方スパイク、制限なし、制限内
- `TestAsymmetricEma` (3): 上昇、下降、等値
- `TestMedianFilterFast` (4): 奇数長、偶数長、単一要素、型チェック
- `TestSectionExecutionQualityComparison` (4): 無 fill、乗法のみ、加法グループ、ICR 計算

## 変更ファイル

| ファイル | 変更 |
|---------|------|
| `ztb/utils/robust_stats.py` | 新規作成 |
| `scripts/v460/analysis/analyze_fill_logs.py` | `section_execution_quality_comparison()` 追加 + main() に組込 |
| `tests/unit/v460/test_571_robust_stats.py` | 新規作成 (21 テスト) |
| `docs/v460/573_cplt_impl_571_robust_stats_and_quality_metrics.md` | 本ドキュメント |
