# 140# Phase 2 レビュー修正 — CRITICAL: FillRecord 統一 + run_id 比較

**日付**: 2026-02-22  
**コミット**: `83962ec88`  
**テスト**: 1161 → **1171 passed** (+10 テスト追加)  
**前提**: 139# (`42bbc21d4`) での §8 外部再レビュー指摘項目の即時修正

---

## §1 背景

139# ドキュメント §8 で外部再レビューを実施した結果、**CRITICAL 1 件 + HIGH 1 件 + MEDIUM 2 件 + LOW 1 件** の是正が必要と判明。  
特に `_append_fill_record()` 未定義メソッド呼び出しは、skip 分岐で `AttributeError` を発生させる致命バグであり、即時修正が必要であった。

---

## §2 修正一覧 (重大度順)

| # | 重大度 | 対象ファイル | 問題 | 修正内容 |
|---|---|---|---|---|
| 1 | **CRITICAL** | `run_fill_test.py` | `_append_fill_record()` が `FillTestRunner` に未定義 → `AttributeError` | `batch.append()` + `maybe_flush()` に統一。`_append_fill_record` 呼び出し全廃 |
| 2 | **HIGH** | `run_fill_test.py` | `time_filter` / `preflight` skip 系で FillRecord が生成されない → 可観測性欠損 | `time_filter_both_sides`, `time_filter_086_deadlock`, `preflight_insufficient` の 3 分岐に FillRecord 生成追加。全 7 skip パスに `cancel_reason` 付き FillRecord を統一 |
| 3 | **MEDIUM** | `retrain_scheduler.py` | `new_samples` 負値でヒューリスティック比較のみ → 精度不足 | metadata に `source_run_id` 保存。run_id 直接比較を優先、旧モデル (source_run_id 未保存) は負値フォールバックで後方互換維持 |
| 4 | **MEDIUM** | テスト | ソース文字列検査中心でランタイム不整合を検出不能 | `TestRunContinuousBranchExecution` (5 テスト) + `TestRetrainRunIdComparison` (5 テスト) を追加。`hasattr` ランタイム検証含む |
| 5 | **LOW** | ドキュメント | 「17件対応完了」結論が過大 | 結論を「22件対応完了」に更新し正確な記述に修正 |

---

## §3 変更ファイル詳細

### §3.1 `run_fill_test.py` (+41 行)

**問題**: `FillTestRunner` クラスに `_append_fill_record()` メソッドが存在しないにも関わらず、  
複数の skip 分岐で `self._append_fill_record(...)` を呼び出していた。  
これにより、`time_filter_both_sides` 等の skip パスで `AttributeError` が発生し、  
サイクル全体が例外終了する致命バグであった。

**修正**:
- `_append_fill_record()` 呼び出しを全て `batch.append(FillRecord(...))` + `maybe_flush()` パターンに統一
- 以下 4 分岐の FillRecord 化:
  - `time_filter_both_sides` (cancel_reason 付き)
  - `time_filter_086_deadlock` (cancel_reason 付き)
  - `preflight_insufficient` (cancel_reason 付き)
  - `preflight_pause` (cancel_reason 付き)
- 結果: 全 7 skip パスが `cancel_reason` 付き FillRecord を一貫して生成

### §3.2 `retrain_scheduler.py` (+30 行)

**問題**: retrain 後の `new_samples` 算出が「現行 total — 前回 total」で負値になりうる。  
run_id が変わると total がリセットされ、`new_samples < 0` → 永久 retrain 停止のリスク。

**修正**:
- retrain 完了時のメタデータに `source_run_id` を保存
- `new_samples` 判定で run_id 直接比較を優先:
  - `source_run_id != current_run_id` → 新 run 検知 → retrain 許可
  - `source_run_id` 未保存 (旧モデル) → 従来の負値フォールバック

### §3.3 `test_139_review_fixes.py` (+173 行, 10 テスト追加)

| テストクラス | テスト数 | 内容 |
|---|---|---|
| `TestRunContinuousBranchExecution` | 5 | `_append_fill_record` 非存在ランタイム検証、`batch.append` パターン確認、`maybe_flush` 呼び出し確認、全 7 skip パス cancel_reason 確認 |
| `TestRetrainRunIdComparison` | 5 | `source_run_id` メタデータ保存確認、run_id 一致/不一致判定、旧モデル後方互換、負値フォールバック |

### §3.4 `test_091_fixes.py` (+3 行)

- 086 deadlock 関連テストの行数断言を修正 (FillRecord 追加に伴うソース行数変更)

---

## §4 整合性点検

| 起点 | 論点 | 140# 整合 |
|---|---|---|
| 132# F2 | retrain `new_samples` 停滞 | ✅ run_id 直接比較で根本解決 |
| 132# F4 | skip 系可観測性欠損 | ✅ 全 7 skip パスに cancel_reason 付き FillRecord |
| 133# P0-03 | trades 供給復旧 | ✅ 影響なし |
| 134# Phase C | 24h 再計測 | ⬜ 実装基盤完了、運用はこれから |

---

## §5 検証ログ

```
tests/unit/v460/test_139_review_fixes.py: 37 passed, 1 warning (27→37, +10)
tests/unit/v460: 1171 passed, 91 warnings (1161→1171, +10)
ランタイム検証: hasattr(FillTestRunner, "_append_fill_record") → False ✅
self._append_fill_record 呼び出し → ソース内に存在しない ✅
```

---

## §6 次ステップ

→ **141# へ**: P1-01/02 (side 別モデル分離) + P1-04 (regime 別閾値) + P1-12 (online monitor) を実装

---

## §7 実装レビュー追記 (2026-02-22)

### §7.1 重大度付きレビュー結果

| # | 重大度 | 対象 | 指摘 | 推奨対応 |
|---|---|---|---|---|
| 1 | MEDIUM | `scripts/v460/run_fill_test.py`, `ztb/metrics/fill_quality.py` | 140# で追加した skip FillRecord (`time_filter_*`, `preflight_*`) は `order_price=0` / `order_quantity=0`（`preflight_pause` は `side=\"none\"`）のため、`filter_clean_records()` で quarantine される。結果として Gate 集計上の可観測性には未反映。 | 非発注イベント用のレコード種別を分離するか、`cancel_reason` が監査系の場合は clean 判定ルールを拡張して別集計に載せる。 |
| 2 | LOW | `scripts/v460/run_fill_test.py` | `preflight_pause` の `cycle_id` が `preflight_pause_{count}` 固定で run 内再利用される。長期解析で ID 一意性を前提にすると衝突し得る。 | `uuid` を suffix 付与して一意化する。 |
| 3 | LOW | `docs/v460/140_ph2_fix_critical_fillrecord.md` | テスト件数の記述は 140# 時点としては妥当だが、現時点の実測は `tests/unit/v460 = 1218 passed` まで進んでおり、参照時に誤解を生みやすい。 | 「140# 時点値」を明示し、最新値は別章で追記する。 |

### §7.2 範囲外を含む改善提案

1. `FillRecord` とは別に `RunEventRecord`（time_filter/preflight/pause など）を新設し、発注イベントと監査イベントを分離する。  
2. `gate_judgment` 側で `run_event` 集計を併記し、`latest-run` 判定時に「未発注による見かけ改善/悪化」を分離表示する。  
3. `cancel_reason` を enum 化し、分析スクリプト側の表記ゆれ再発を防止する。  
