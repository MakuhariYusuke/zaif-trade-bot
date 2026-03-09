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

## 4. dynamic_kill EWMA 深堀り分析

### 発見した問題

再起動後ログで `sell_dynamic_kill` の rolling mean が **-10.710bps**（05:17 の sell fill PnL そのまま）に
固定され、30分ごとの TIME LIMIT → 即再 kill を 24 回繰り返す異常パターンを確認。

根本原因を 3 つ特定:

#### P0 (Critical Bug): EWMA 状態が永続化されていない

`export_state()` / `import_state()` に `_ewma_value` が含まれていなかった。

**影響チェーン:**
1. 再起動 → `import_state()` で状態復元 → `_ewma_value = None`
2. `_get_rolling_mean()` が EWMA モードで `None` を返す → kill 判定スキップ
3. ガード無しで 1 trade 許可 → fill の PnL がそのまま EWMA seed に（α=1.0 相当）
4. 05:17 sell -10.71bps が seed → EWMA = -10.710 固定
5. threshold -0.5bps を大幅に下回る → 即 kill → TIME LIMIT 30分 → 解除 → 再 kill の無限ループ

**修正:** `export_state()` に `ewma_value` を追加、`import_state()` で復元。
欠落時は `_rebuild_ewma_from_history()` で pnl_history から再構築。

#### P1: EWMA シードが単一観測値で脆弱

EWMA の初回シードが `pnl_bps`（単一値）で行われていた。
α=0.05 の場合、この単一値が EWMA に残留する影響は 0.95^n で指数減衰するが、
-10.71bps のような外れ値では ~45 fills で threshold に収束 — 事実上、回復不能。

**修正:** 初回シードを `pnl_history` の算術平均に変更。
複数データがあれば外れ値の影響が希釈され、安定起動が可能に。

#### P2: TIME LIMIT 解除が EWMA をリセットしない

273# の TIME LIMIT は cooldown と kill_activated_at をリセットするが、
EWMA 値はそのまま維持されるため、次の `check_kill()` で即再 kill される。

**実際のログパターン:**
```
05:52 sell kill TIME LIMIT expired → auto-releasing
05:54 sell dynamic kill activated: rolling50 mean=-10.710bps < -0.6bps
```
解除後わずか 2 分で再 kill。EWMA が -10.710 のまま変わらないため。

**修正:** TIME LIMIT 解除時に EWMA を `threshold * 0.8` にリセット。
kill 閾値のすぐ上に置くことで、次の悪い fill では再 kill されるが、
良い fill なら回復の余地がある。

### ログ実証データ

```
00:10 warmup: sell=64 records → rolling50 mean=-0.553bps
00:10 sell kill activated (mean=-0.553 < -0.5)  ← 正常な kill

05:13 RESTART → import_state でEWMAがNullに!
05:17 sell FILL -10.71bps → EWMA seed = -10.710 (単一値)
05:22 sell kill activated: mean=-10.710 < -0.5  ← 修復不能
05:52 TIME LIMIT → release → 05:54 即再kill (EWMA 変わらず)
07:01 TIME LIMIT → release → 07:02 即再kill
...以降 24 回繰り返し...
```

### 修正内容

| 修正 | ファイル | 内容 |
|------|---------|------|
| P0 | `sell_dynamic_kill.py` | `export_state()`/`import_state()` に `ewma_value` 追加 + `_rebuild_ewma_from_history()` |
| P1 | `sell_dynamic_kill.py` | `track()` 初回シードを history 平均に変更 |
| P2 | `sell_dynamic_kill.py` | TIME LIMIT 解除時に EWMA を `threshold * 0.8` にリセット |
| テスト | `test_349_ewma_fixes.py` | 13 test cases (永続化・シード・decay・reset) |

## 5. 付随修正

- `tests/test_analyze_fill_logs.py`: import パスを `tools.analysis.` → `scripts.v460.analysis.` に修正
- `docs/evaluation/extended_evaluation.md`: regime_evaluation セクションを後継モジュールへの案内に更新

## 変更サマリ

| ファイル | 行数変動 |
|----------|---------|
| `scripts/v460/analysis/analyze_fill_logs.py` | -84 → +43 (net -41) |
| `ztb/metrics/fill_quality.py` | -34 → +19 (net -15) |
| `ztb/analysis/regime/regime_evaluation.py` | **削除** (-341) |
| `ztb/risk/sell_dynamic_kill.py` | P0/P1/P2 EWMA 修正 (+50) |
| `tests/unit/v460/test_349_ewma_fixes.py` | **新規** (13 tests) |
| `tests/test_analyze_fill_logs.py` | import パス修正 |
| `docs/evaluation/extended_evaluation.md` | deprecated 案内 |
