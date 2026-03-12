# 331# Self-Review: 329#/330# 自己監査 + 328# 残タスク確認

> **種別**: rev (自己レビュー)  
> **対象コミット**: `894c1bf8a` (329#), `14dfa5e2c` (330#)  
> **日付**: 2026-03-15  

---

## 1. レビュー対象

### 329# (`894c1bf8a`) — fill_config.py God Object 分割

- `fill_config.py` 2,046→724 行 (▲65%)
- 3 ファイルに分離:
  - `fill_config_parser.py` (YAML パーサー)
  - `fill_config_validation.py` (バリデーション)
  - `fill_config_results.py` (Result/Summary dataclass)
- 遅延 import + `__init__` 再 export で後方互換維持

### 330# (`14dfa5e2c`) — run_continuous pre-cycle 抽出 + bugfix

- `orchestrator_pre_cycle.py` 新設 (565 行)
- `CycleContext` dataclass 導入
- MCBLevel/SADLevel Enum 移管
- σ floor + vol_ratio_floor ゼロ除算ガード
- `_feed_mcb_sad()` による MCB/SAD データ投入一元化

---

## 2. レビュー結果サマリ

| 分類 | 件数 | 修正済み |
|---|---|---|
| **BUG** | 2 | ✅ 2/2 |
| **DESIGN** | 5 | ✅ 2/5 (残 3 は DEFERRED) |
| **MISSING** | 3 | ✅ 1/3 (残 2 は DEFERRED) |
| **STYLE** | 2 | ✅ 2/2 |

---

## 3. BUG (2 件) — 全修正済み

### BUG-1: MCB HALT 中に SAD ベースラインがフィードされない

- **場所**: `orchestrator_pre_cycle.py` — `_check_circuit_breakers()`
- **問題**: MCB HALT で `return True` (早期リターン) すると、その後の `_feed_mcb_sad()` 呼出しがスキップされ、SAD ベースラインウィンドウにギャップが発生する
- **影響**: MCB HALT → 解除後に SAD ベースラインが劣化し、不正確な判定を招く可能性
- **修正**: `_check_circuit_breakers()` 冒頭に `self._feed_mcb_sad()` を追加。早期リターン前に必ずデータ投入される

### BUG-2: `warnings.warn` の stacklevel が浅い

- **場所**: `fill_config_validation.py` — kyle_lambda/amihud 警告
- **問題**: `stacklevel=2` → `validate_fill_config()` 内を指す (ユーザーに不明)
- **修正**: `stacklevel=3` に変更 — ユーザーの `FillTestConfig()` 呼出し元を正しく指示

---

## 4. DESIGN (5 件)

### D-1: CycleContext の 5/6 フィールドが死んでいる ✅ 修正済み

- **問題**: `balance_forced`, `is_rescue`, `one_sided_balance`, `inventory_escape`, `regime_mult` が `fill_loop_orchestrator.py` で参照されていない (インラインローカル変数が使われている)
- **修正**: 5 フィールド削除、`next_side` のみ残す。Phase 4 抽出時に必要なら再追加
- **注記**: `run_continuous` 内のインラインロジック (~800 行) が Phase 4 で抽出される際にフィールドが復活する可能性あり

### D-2/S-2: dead import — MCBLevel/SADLevel ✅ 修正済み

- **問題**: `fill_loop_orchestrator.py` に MCBLevel/SADLevel の import が残存 (330# で orchestrator_pre_cycle に移管済み)
- **修正**: import 行をコメントに置換 (移管経緯を記録)
- **テスト修正**: `test_227` の MCBLevel/SADLevel テストを `orchestrator_pre_cycle` モジュール参照に変更

### D-3: MCB/SAD フィードロジック重複 — DEFERRED

- **問題**: BUG-1 修正により `_check_circuit_breakers` 冒頭と `run_continuous` 本体で `_feed_mcb_sad()` が二重呼出しになる可能性
- **影響**: `update()` は冪等なため実害なし
- **方針**: Phase 4 で `run_continuous` を分割する際に整理

### D-4: sigma_floor/vol_ratio_floor の YAML 配置 — DEFERRED

- **問題**: `sigma_parkinson` YAML セクション配下に配置されているが、概念的には独立したリスクパラメータ
- **方針**: M-2 と合わせて YAML スキーマ見直し時に対応

### D-5: pre-cycle 抽出の不完全性 — DEFERRED

- **問題**: `alert_mode`, `tick_side_halt`, regime カウントなどが `run_continuous` にインラインで残存
- **方針**: Phase 4 (328# P2-10) で対応

---

## 5. MISSING (3 件)

### M-1: sigma_floor/vol_ratio_floor のバリデーション不在 ✅ 修正済み

- **問題**: 負値が設定されてもエラーにならない
- **修正**: `validate_fill_config()` 末尾に追加:
  - `sigma_floor >= 0` (0 は許容 = 無効化)
  - `vol_ratio_floor > 0` (0 はゼロ除算を招くため不可)

### M-2: sigma_floor/vol_ratio_floor のトップレベル YAML キー — DEFERRED

- **問題**: 現在 `sigma_parkinson.sigma_floor` だが `risk.sigma_floor` が自然
- **方針**: D-4 と合わせて対応

### M-3: fill_config.py に `__all__` がない — DEFERRED

- **問題**: 再 export スコープが暗黙的
- **方針**: 他の God Object 分割完了後にまとめて対応

---

## 6. STYLE (2 件) — 全修正済み

### S-1: `import warnings as _w` の配置

- **問題**: 関数末尾にインラインで配置、可読性低下
- **修正**: 条件分岐内で `import warnings` に統一

### S-2: → D-2 に統合

---

## 7. 328# 残タスク確認

### 完了済み

| ID | 内容 | 完了セッション |
|---|---|---|
| Phase 1 | fill_config Result 分離 | 329# |
| Phase 2 | fill_config Validation 分離 | 329# |
| Phase 3 | fill_config YAML Parser 分離 | 329# |
| Phase 3 | CycleContext + orchestrator_pre_cycle | 330# |
| P1-4 | fill_config God Object 分割 | 329# (2,046→724 行) |
| P1-5 | offset_ceiling_ratio YAML 修正 | 321# |

### 進行中

| ID | 内容 | 状態 |
|---|---|---|
| G1.2 | 168h 連続クリーンデータ蓄積 | ⏳ bot `dcc3064a8` 稼働中 |

### 未着手 (優先度順)

| 優先度 | ID | 内容 | ブロッカー |
|---|---|---|---|
| **P1** | P1-1 | SkipGate 再訓練 (n≥500) | G1.2 データ蓄積 |
| **P1** | P1-2 | spread_adaptive AB テスト | 未着手 |
| **P1** | P1-3 | Volatility Guard 動的ゲーティング | 設計済み |
| **P2** | P2-10 | run_continuous 更なる分割 (~800行→<500) | Phase 4 |
| **P2** | P2-1~9 | コード品質改善 21 件 | — |
| **P3** | P3-1~14 | 低優先 14 件 | v461+ |

---

## 8. テスト結果

```
4105 passed, 14 warnings in 24.03s — ALL GREEN ✅
```

---

## 9. 修正ファイル一覧

| ファイル | 修正内容 |
|---|---|
| `scripts/v460/lib/orchestrator_pre_cycle.py` | BUG-1 fix + D-1 CycleContext 死フィールド削除 |
| `scripts/v460/lib/fill_loop_orchestrator.py` | D-2 dead import 削除 |
| `scripts/v460/lib/fill_config_validation.py` | BUG-2 stacklevel + S-1 import + M-1 validation |
| `tests/unit/v460/test_227_*.py` | MCBLevel/SADLevel テスト先変更 |
