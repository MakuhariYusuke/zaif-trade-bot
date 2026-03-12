# 225# warmup日付フィルタ + state save強化 + recovery復元 + 市場理論補強

## 変更の目的
224# で発見した残課題 (F1/F2/6.1/5.1-5.2) の修正 + 市場理論に基づく補強 + 盲点対応。

## 変更内容

### F1 (MEDIUM): Kill manager warmup 日付フィルタ — DD warmup との矛盾修正
- **ファイル**: `fill_loop_orchestrator.py` → `_warmup_kill_managers_from_records()`
- **問題**: 再起動時に全 fill records （前日含む）を kill manager に replay していた。B2 の日替わり `reset()` の効果が無効化される矛盾。
- **修正**: DD warmup と同じ `utc_today` フィルタを適用。前日以前の records は skip。
- **影響**: kill manager の rolling PnL が当日分のみに正確化。

### F2 (MEDIUM): 通常サイクルパスの time-based state save
- **ファイル**: `fill_loop_orchestrator.py` → state 永続化ロジック
- **問題**: 通常パスでは `progress_log_interval` (50 cycles × ~120s = ~100 分) 間隔でしか state 保存されず、クラッシュ時に最大 100 分のデータ喪失。
- **修正**: `_STATE_SAVE_INTERVAL_SEC` (300s) 経過でも保存するよう統合。skip-time path の 223# パターンを normal path にも適用。

### 6.1 (HIGH): recovery counter 例外時復元
- **ファイル**: `fill_loop_orchestrator.py` + `daily_drawdown_guard.py`
- **問題**: `get_recovery_lot_scale()` でカウンタが消費された後、`run_single_cycle()` が例外で中断すると、lot 縮小が適用されないままカウンタだけ減る。
- **修正**: 例外ハンドラで `restore_recovery_counter()` を呼び、次サイクルで再適用。

### 5.1/5.2 (MEDIUM): Guard fire count 漏れ追加
- **ファイル**: `fill_loop_orchestrator.py`
- **問題**: `time_filter_both_sides` と `preflight_insufficient` のパスで `_inc_guard_fire()` が呼ばれていない。
- **修正**: 両パスに fire count記録を追加。

### 市場理論補強: Regime-aware recovery scaling
- **ファイル**: `fill_loop_orchestrator.py` + `fill_config.py` + `fill_test.yaml` + `regime_detector.py`
- **理論的根拠**: Avellaneda-Stoikov の在庫リスク管理原理。ボラティリティ/トレンドが持続する環境ではリスクが比例的に増大する。
- **実装**: halt 解除後のリカバリ期間中、regime に応じて lot 縮小倍率を追加乗算:
  - `trending` → `×0.7` (adverse selection リスク残存)
  - `high_vol` → `×0.8` (在庫リスク増大)
  - `ranging/unknown` → 追加ペナルティなし (mean reversion 期待)
- **新パラメータ**: `recovery_trending_penalty`, `recovery_high_vol_penalty`
- **新 property**: `FillTestRegime.is_high_vol`

### 盲点修正: MCB/SAD 状態永続化
- **ファイル**: `resilience.py` + `fill_loop_orchestrator.py`
- **問題**: `MicroCircuitBreaker` と `SpreadAnomalyDetector` の状態 (`price_buffer`, `halt_until`, `spread_buffer`, `frozen_until`) が `FillTestState` に含まれておらず、再起動時に失われていた。
- **修正**: `mcb_state`, `sad_state` フィールドを追加し、`_build_state_snapshot()` / `_restore_common_state()` で export/import。

### 盲点修正: DD Guard 日替わり recovery リセットの clarification
- **ファイル**: `daily_drawdown_guard.py`
- **内容**: 日替わり `maybe_reset_day()` で `side_recovery_remaining` もリセットされることについて、意図的設計であるコメントを追加。

## セルフレビュー
| ID | チェック項目 | 結果 |
|----|-------------|------|
| R1 | F1 日付フィルタが DD warmup と同一パターン | ✅ |
| R2 | F2 state save が time/progress の OR 条件で動作 | ✅ |
| R3 | 6.1 recovery 復元が `_recovery_scale < 1.0` ガード付き | ✅ |
| R4 | 5.1/5.2 fire count キー名が既存パターンと整合 | ✅ |
| R5 | regime-aware recovery が null-safe | ✅ |
| R6 | MCB/SAD state が `getattr` fallback で後方互換 | ✅ |
| R7 | 全 3016 テスト通過、0 regression | ✅ |

## 残課題 (今回対象外)
| ID | 重要度 | 内容 |
|----|--------|------|
| #1-1 | HIGH | ファントムポジション検出 (取引所残高 vs 内部追跡の定期照合) |
| #2-1 | MEDIUM | FFD hot-reload 時の適応済み offset 状態保存 |
| #4-2 | MEDIUM | MCB の `_change_history` (σ計算用) が export_state に未含 |
| #3-2 | MEDIUM | MCB/SAD halt/skip パスでの `_alert_*_mult` 前サイクル値残存 |

## テスト
- 新規: `test_225_warmup_recovery_fire_counts.py` (18 テスト)
- 既存: 2998 テスト通過、regression なし
- 合計: **3016 passed, 0 failed**
