# 349# 分析ツール整理 + 重複コード削減

## 概要

分析ツールの乱雑化に対処し、散在していたスクリプトを一元化。
併せて DRY 原則に基づき重複 JSONL 読み込み・deprecated モジュールを整理。
再起動後ログの直接分析も実施。

## 1. 分析ツール整理 (25463555d)

### ディレクトリ統合

| 操作 | 対象 | 内容 |
|------|------|------|
| 移動 | `tools/analysis/analyze_fill_logs.py` | → `scripts/v460/analysis/` |
| 移動 | `tools/analysis/print_ab_summary.py` | → `scripts/v460/analysis/` |
| アーカイブ | `tools/analysis/v443_2_analysis/` | → `archived/` |
| 保持 | `scripts/v460/analysis/oracle_test.py` 他3件 | テスト依存のため維持 |

整理後: `scripts/v460/analysis/` に全分析ツールを一元化。

### PnL 分析 (7日間)

| 指標 | 値 |
|------|-----|
| Total fills | 700 |
| Fill rate | 21.1% |
| Win rate | 49% |
| Total PnL | -94.6 bps (≈ ¥-70k) |
| Buy PnL | -97.3 bps |
| Sell PnL | +2.7 bps |
| Worst segment | buy_ranging: -102.9 bps |

## 2. 重複コード削減 (a051b6a8c)

### P0: analyze_fill_logs.py — ztb 共有 API 移行 (-85行)

`load_records()` (42行) と `apply_filters()` (43行) が手動 `json.loads` ループ +
日付プリフィルタ + run_id/git_sha/date フィルタを独自実装していた。

**置換先:**
- `ztb.metrics.fill_quality.load_fill_record_objects_glob()` — glob + 日付プリフィルタ + 重複排除
- `ztb.metrics.fill_quality.apply_fill_record_filters()` — run_id/git_sha/date フィルタ

side/regime フィルタのみローカルに残留（共有 API 側に存在しないため）。

### P2: iter_fill_records() — iter_jsonl_objects 活用 (-15行)

`fill_quality.py` の `iter_fill_records()` が独自に `open()` → `json.loads(line)` →
BOM 処理・malformed スキップを実装していたが、同モジュールが既にインポート済みの
`ztb.io.jsonl.iter_jsonl_objects()` に全く同じ機能がある。

`FillRecord.from_dict()` 変換 + cycle_id 重複排除のみ維持し、パースを委譲。

### P3: deprecated regime_evaluation.py 削除 (-341行)

`ztb.analysis.regime.regime_evaluation` は自身が冒頭で `DeprecationWarning` を発行しており、
プロジェクト全体でインポート実績ゼロ（grep 確認済み）。

後継: `ztb.analysis.regime.regime_eval` / `UnifiedEvaluator(EvaluationType.REGIME)`

### P1: PnL 集計パターン共通化 (見送り)

`analyze_fill_logs.py` の各 `section_*` 関数が独自に numpy で
avg/p10/p90/勝率を算出（15箇所以上反復）。`PnlAccumulator` への統合は
CLI レポート出力形式との差異が大きく、p10/p90 メソッド追加 +
グルーピングユーティリティ新設が前提。コスト対効果で今回は見送り。

## 3. 再起動後ログ分析 (05:13 JST～)

### Fill 結果 (5件)

| 時刻 | Side | PnL (bps) | Regime | EV |
|------|------|-----------|--------|-----|
| 05:13 | buy | -1.92 | ranging | 0.07 |
| 05:17 | sell | -10.71 | ranging | -1.08 |
| 05:46 | buy | +6.59 | ranging | -0.22 |
| 07:32 | sell | -5.84 | trending_down | 3.17 |
| 08:06 | buy | -6.67 | ranging | 0.83 |

**合計: -18.54 bps, 勝率 20% (1/5)**

### 動作パターン

- 93 skips: `buy_dynamic_kill` / `sell_dynamic_kill` が支配的
- ~30分 kill → TIME LIMIT 解除 → 1 trade → 即 kill の繰り返し
- JPY 残高 ¥1,342 < 最低必要額 ~¥10,500 → buy 永久ブロック

## 4. 付随修正

- `tests/test_analyze_fill_logs.py`: import パスを `tools.analysis.` → `scripts.v460.analysis.` に修正
- `docs/evaluation/extended_evaluation.md`: regime_evaluation セクションを後継モジュールへの案内に更新

## 変更サマリ

| ファイル | 行数変動 |
|----------|---------|
| `scripts/v460/analysis/analyze_fill_logs.py` | -84 → +43 (net -41) |
| `ztb/metrics/fill_quality.py` | -34 → +19 (net -15) |
| `ztb/analysis/regime/regime_evaluation.py` | **削除** (-341) |
| `tests/test_analyze_fill_logs.py` | import パス修正 |
| `docs/evaluation/extended_evaluation.md` | deprecated 案内 |

**合計: -445行 / +43行 = net -402行**
