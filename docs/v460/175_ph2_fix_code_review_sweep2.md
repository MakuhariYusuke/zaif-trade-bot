# 175# Code Review Sweep #2 — 深層コードレビュー & 修正実装

> **日付**: 2026-02-27  
> **ベース**: `ea20c41a4` (174# fresh code review)  
> **テスト**: 2161 passed (23 new), 0 failed

---

## 1. 概要

173#/174# に続く第3回コードレビュー。
前回未修正の5件を含む11件の課題を修正。
FFD boost の max_offset_ratio 超過バグ (HIGH)、buy_dynamic_kill_window 未検証 (HIGH)、
order_monitor の `object` 型排除、FFD boost TTL decay 機能追加など。

---

## 2. 修正一覧

### HIGH

| # | 対象ファイル | 問題 | 修正内容 |
|---|---|---|---|
| H1 | `maker_price.py` L705-709 | FFD boost が `max_offset_ratio` クランプを bypass → 過剰保守的 offset | `min(effective_offset_ratio * boost_mult, cfg.max_offset_ratio)` |
| H2 | `fill_config.py` | `buy_dynamic_kill_window` バリデーション欠落 (sell は検証済み) | `__post_init__` に `buy_dynamic_kill_window >= 1` 検証追加 |

### MED

| # | 対象ファイル | 問題 | 修正内容 |
|---|---|---|---|
| M1 | `skip_gate_evaluator.py` L337 | `_inject_calibrator` 二重呼び出し (hot-reload 時) | `_load_gate_from_path` 内で実行済みのため削除 |
| M2 | `fill_loop_orchestrator.py` | `heartbeat_task` 未処理例外時のリーク | `cleanup_heartbeat()` async method 追加 + instance var 化 |
| M3 | `fill_config.py` | `sell_offset_floor_inv_discount` YAML バインド欠落 | `_parse_stale_vg_section` に `offset_floor_inv_discount` マッピング追加 |
| M4 | `config_hot_reload.py` | stale side 別 6 フィールドが hot-reload 対象外 | `_HOT_RELOADABLE_FIELDS` に追加 |
| M5 | `order_monitor.py` | 6 引数が `object` 型 → 11 箇所 `type: ignore` | Protocol 型定義 (`_KillSwitchLike`, `_SkipGateLike`) + `Callable` 型化、7 箇所 `type: ignore` 削除 |

### LOW

| # | 対象ファイル | 問題 | 修正内容 |
|---|---|---|---|
| L1 | `maker_price.py` L161 | `if n == 0` dead code (append 後で n>=1 確定) | dead code 削除 |
| L2 | `fill_record_helpers.py` | skip counter conflation: tss/bfs 交互出現時に過大カウント | 各カウンタを独立ループで計算 |
| L3 | `fast_fill_defense.py` | boost に TTL なし → time_filter 後も古い boost が残存 | `boost_ttl_sec=600.0` 設定追加、`boost_activated_at` 記録、TTL 超過で自動 decay |

### 既存テスト修正

| # | 対象ファイル | 修正内容 |
|---|---|---|
| T1 | `test_145_structural_fixes.py` | AUDIT_CANCEL_REASONS にSKIP_GATE 系3定数追加 (174# 連動) |

---

## 3. 変更ファイル一覧

| ファイル | 変更種別 |
|---|---|
| `scripts/v460/lib/maker_price.py` | 修正 (FFD clamp + dead code) |
| `scripts/v460/lib/fill_config.py` | 修正 (buy_kill validation + YAML binding) |
| `scripts/v460/lib/skip_gate_evaluator.py` | 修正 (calibrator 二重呼び出し) |
| `scripts/v460/lib/fill_loop_orchestrator.py` | 修正 (heartbeat cleanup) |
| `scripts/v460/lib/config_hot_reload.py` | 修正 (stale side fields) |
| `scripts/v460/lib/order_monitor.py` | 修正 (Protocol 型化, 7x type: ignore 削除) |
| `scripts/v460/lib/fill_record_helpers.py` | 修正 (skip counter 独立化) |
| `scripts/v460/lib/fast_fill_defense.py` | 修正 (boost TTL decay) |
| `tests/unit/v460/test_145_structural_fixes.py` | テスト修正 |
| `tests/unit/v460/test_175_code_review_sweep2.py` | **新規** (23 tests) |

---

## 4. テスト

### 新規テスト (test_175_code_review_sweep2.py — 23 tests)

| クラス | テスト数 | 対象 |
|---|---|---|
| `TestFFDBoostClamp` | 1 | H1: FFD boost max_offset clamp |
| `TestBuyDynamicKillWindowValidation` | 3 | H2: buy kill window 検証 |
| `TestCalibratorNotDoubled` | 1 | M1: 二重呼び出し確認 |
| `TestYAMLInvDiscountBinding` | 2 | M3: YAML パース |
| `TestStaleSideHotReload` | 6 | M4: stale side ×6 フィールド |
| `TestDeadCodeRemoved` | 1 | L1: dead code 除去確認 |
| `TestSkipCounterSeparation` | 2 | L2: 交互/連続カウント |
| `TestFFDBoostTTL` | 3 | L3: TTL 設定/追跡 |
| `TestOrderMonitorProtocols` | 3 | M5: Protocol 型検証 |
| `TestHeartbeatCleanup` | 1 | M2: cleanup メソッド存在 |

### 回帰テスト結果

```
2161 passed, 0 failed, 13 warnings in 238.23s
```

---

## 5. 残課題

| 優先度 | 課題 | 出典 |
|---|---|---|
| P2 | CircuitBreaker 階層統合 | 170# §10.5 |
| P2 | skip_gate counterfactual 追加 | 171# §8.4 #4 |
| P2 | 日次レポート 3 系列固定出力 | 170# §10.5 |
| D4 | InvSkew bypass 閾値 0.3 チューニング | 172# §10.2 (実運用データ待ち) |
| LOW | `regime_detector: object | None` → Protocol (order_monitor) | type: ignore 残4箇所 |
| LOW | coincheck_adapter 冗長 exception catch | `(NetworkError, Exception)` → `Exception` |
| LOW | FFD boost TTL の YAML 化 | `boost_ttl_sec` の YAML バインド追加 |
