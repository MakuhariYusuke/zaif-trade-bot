# 203# DD状態永続化修正 + halt カウンタ修正

> **日付**: 2026-03-01  
> **前提**: 202# コミット後のログ追跡で発見された P0 バグ

---

## 1. 発見経緯

再起動後のログ分析で以下の critical パターンを発見:

```
10:07 HALT: daily PnL -53.20bps (12 fills)
...HALT中 ~8h...
18:55 (restart) State stale (saved=20260228, today=20260301), skip import
...DD guard リセット → 再取引...
20:35 HALT: daily PnL -57.74bps (17 fills) ← 同日2回目のHALT
```

## 2. 根本原因

### 203# E: HALT 中の state 保存バグ
- 旧実装: `self._cycle_count % progress_log_interval == 0` で定期保存
- **バグ**: `_cycle_count` は halt 中に不変 → 50の倍数でなければ **一度も保存されない**
- **結果**: state file が HALT 前の値のまま固定、再起動で DD state を復元不可

### 203# F: state 復元失敗時のフォールバック不在
- `import_state()` が stale 日付で skip → DD guard が 0bps にリセット
- fill records にはデータがあるのに累積 PnL が消失

### 203# G: halt_elapsed カウンタバグ
- 旧実装: `_halt_elapsed = _cycle_count - _halt_start_cycle` → 常に 0 (不変)
- 200# K の halt record 削減が実質機能していない (毎 iteration 記録)

---

## 3. 実装

### 203# E: HALT 開始時 state 強制保存
- `_halt_entering` フラグ: halt 初回は無条件で state 保存
- 以降は `_halt_iter_count` ベースで periodic 保存 (旧 `_cycle_count` 依存を除去)

### 203# F: DD warmup from fill records
- `_warmup_daily_drawdown_from_records()` メソッド追加
- import_state 失敗 (daily_fill_count == 0) 時に自動発動
- 当日 UTC の fill records から daily_pnl を再計算
- hard/soft limit も正しく判定

### 203# G: halt iteration counter 導入
- `_halt_iter_count` で正確にカウント (record 頻度 + state save 頻度の両方に使用)
- halt 終了時に `_halt_iter_count = 0` でリセット
- halt end ログも iterations ベースに変更

---

## 4. テスト

- 新規: `test_203_dd_state_persistence.py` — 11 tests (全 PASS)
- 回帰: 2287 passed, 0 new failures
