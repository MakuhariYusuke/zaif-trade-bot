# 173# Code Review Sweep — 包括的コードレビュー & 修正実装

> **日付**: 2026-02-28  
> **ベース**: `f77d82ce4` (172# self-review)  
> **テスト**: 2138 passed (30 new), 0 failed

---

## 1. 概要

172# Guard Paradox 根本対策の完了後、コードベース全体の包括的レビューを実施。
20 件の問題を発見し、CRITICAL 1 / HIGH 5 / MED 6 / 機能改善 1 の計 13 項目を修正。
型安全性・観測性・堅牢性の全面的な底上げを行った。

---

## 2. 修正一覧

### CRITICAL

| # | 対象ファイル | 問題 | 修正内容 |
|---|---|---|---|
| C1 | `ztb/risk/sell_dynamic_kill.py` | `DynamicKillConfig(window=0)` で `sum([]) / len([])` → ZeroDivisionError | `__post_init__` で `window >= 1`, `resume_window >= 0` を検証 |

### HIGH

| # | 対象ファイル | 問題 | 修正内容 |
|---|---|---|---|
| H1 | `scripts/v460/lib/fill_loop_orchestrator.py` | Rescue log で `consecutive_skip` が常に 0 表示 | reset 前に `_prev_skip_count` を保存してログに使用 |
| H2 | `scripts/v460/lib/maker_price.py` | `adapter: object` → `type: ignore[attr-defined]` 3箇所 | `OrderbookProvider` Protocol 型に変更、`type: ignore` 全削除 |
| H3 | `scripts/v460/lib/daily_drawdown_guard.py` | `import_state` の `halt_triggered_at` 型不整合 | `float()` 明示変換で `type: ignore[assignment]` 削除 |
| H4 | `scripts/v460/lib/config_hot_reload.py` | 6 フィールドが Hot-Reload 対象外 | `_HOT_RELOADABLE_FIELDS` に追加: `post_fill_wait_sec_sell`, `sell_offset_floor`, `sell_offset_floor_inv_discount`, `sell_max_spread_jpy`, `unknown_buy_offset_boost`, `fallback_stale_sec` |
| H5 | `scripts/v460/analysis/hindsight_filter.py` | 4 cancel_reason が H7_other に誤分類 | `ranging_low_vol_skip`, `daily_drawdown_halt`, velocity skip, `unknown_regime_sell_skip` を正しいカテゴリへ追加 + 文字列リテラル → CR 定数化 |

### MEDIUM

| # | 対象ファイル | 問題 | 修正内容 |
|---|---|---|---|
| M1 | `scripts/v460/lib/cancel_reasons.py` | cancel_reason が素の `str` — typo 検出不能 | `CancelReason = Literal[...]` 型エイリアス追加 |
| M2 | `scripts/v460/lib/fill_record_helpers.py` | `_make_skip_record(cancel_reason: str)` | `cancel_reason: CancelReason` に型制約 |
| M3 | `scripts/v460/lib/fill_config.py` | `sell_guard_inv_bypass_threshold` 範囲未検証 | `__post_init__` で `[0.0, 1.0]` 範囲チェック |
| M4 | `scripts/v460/lib/daily_drawdown_guard.py` | Halt 中の機会損失が定量化不能 | `halt_blocked_cycles` を state/metrics/export に追加 |
| M5 | `scripts/v460/lib/fill_loop_orchestrator.py` | `progress_log` で `fill_rate_pct = filled / total` → 0 除算 | `total_count > 0` ガード追加 |
| M6 | `scripts/v460/lib/daily_drawdown_guard.py` | `update_pnl` が `dict[str, object]` 返却 — 型不明 | `DrawdownAction(TypedDict)` を定義し返却型を明確化 |

### 機能改善

| # | 対象ファイル | 改善内容 |
|---|---|---|
| F1 | `scripts/v460/lib/maker_price.py` | `_effective_sell_offset_floor()` 導入 — InvSkew 活性時に `sell_offset_floor` を `sell_offset_floor_inv_discount` (default 0.5) で割引。Guard Paradox 対策の補強 |
| F1 | `scripts/v460/lib/fill_config.py` | `sell_offset_floor_inv_discount: float = 0.5` フィールド追加 |

---

## 3. 変更ファイル一覧

| ファイル | 変更種別 |
|---|---|
| `ztb/risk/sell_dynamic_kill.py` | 修正 |
| `scripts/v460/lib/fill_loop_orchestrator.py` | 修正 (2箇所) |
| `scripts/v460/lib/maker_price.py` | 修正 (5箇所) |
| `scripts/v460/lib/daily_drawdown_guard.py` | 修正 (TypedDict + metrics + float conversion) |
| `scripts/v460/lib/cancel_reasons.py` | 修正 (Literal type) |
| `scripts/v460/lib/fill_record_helpers.py` | 修正 (型引数) |
| `scripts/v460/lib/fill_config.py` | 修正 (新フィールド + validation) |
| `scripts/v460/lib/config_hot_reload.py` | 修正 (6 fields) |
| `scripts/v460/analysis/hindsight_filter.py` | 修正 (4 reasons + CR 定数化) |
| `tests/unit/v460/test_155_hindsight_review.py` | テスト修正 |
| `tests/unit/v460/test_166_hotfixes.py` | テスト修正 |
| `tests/unit/v460/test_168_daily_drawdown_guard.py` | テスト修正 |
| `tests/unit/v460/test_173_code_review_fixes.py` | **新規** (30 tests) |

---

## 4. テスト

### 新規テスト (test_173_code_review_fixes.py — 30 tests)

| クラス | テスト数 | 対象 |
|---|---|---|
| `TestDynamicKillConfigValidation` | 4 | C1: window/resume_window バリデーション |
| `TestSellGuardInvBypassValidation` | 3 | M3: bypass threshold [0,1] 範囲 |
| `TestCancelReasonLiteralType` | 2 | M1: Literal 型存在 + 全定数包含 |
| `TestDailyDrawdownHaltBlockedCycles` | 3 | M4: halt_blocked_cycles インクリメント/metrics/export |
| `TestDrawdownActionTypedDict` | 1 | M6: update_pnl 返却型 |
| `TestOrderbookProviderType` | 3 | H2: 型アノテーション検証 |
| `TestHotReloadFieldsAdded` | 6 | H4: 6 フィールドの Hot-Reload 対象確認 |
| `TestHindsightFilterReasons` | 4 | H5: 4 reason の正しいカテゴリ分類 |
| `TestDynamicSellOffsetFloor` | 4 | F1: 動的フロア計算 (InvSkew on/off/disabled/default) |

### 既存テスト修正

| ファイル | 修正理由 |
|---|---|
| `test_155_hindsight_review.py` | 171# InvSkew コード拡張により block size 400→800 |
| `test_166_hotfixes.py` | 171# YAML 変更: `rescue_offset_mult` 2.0→1.3, `max_consec` 20→10 |
| `test_168_daily_drawdown_guard.py` | M4 で `halt_blocked_cycles` 追加によるメトリクスキー変更 |

### 回帰テスト結果

```
2138 passed, 0 failed, 13 warnings in 224.51s
```

---

## 5. 残課題 (今後対応)

| 優先度 | 課題 | 出典 |
|---|---|---|
| P2 | CircuitBreaker 階層統合 | 170# §10.5 |
| P2 | skip_gate counterfactual 追加 | 171# §8.4 #4 |
| P2 | 2/23 型崩壊の予防 | 171# §8.4 #5 |
| P2 | 日次レポート 3 系列固定出力 | 170# §10.5 |
| P3 | StatisticalValidator A/B テスト統合 | 170# §10.5 |
| D4 | InvSkew bypass 閾値 0.3 チューニング | 172# §10.2 (実運用データ待ち) |
| D5 | guard_value EV_blocked 拡張 | 172# §10.2 |
| LOW | EV_per_cycle guard_value unit mismatch 精査 | レビュー指摘 #7 |
| LOW | config_hot_reload inventory_skewing_window 扱い確認 | レビュー指摘 #13 |
| LOW | YAML vs code default 不整合精査 | レビュー指摘 #14-15 |
