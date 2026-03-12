# 218# Anti-Deadlock: DynamicKill probe + per-side halt + detection

| key | value |
|---|---|
| 対象 | DynamicKill デッドロック基本対策 |
| commit | `36177b2ae` |
| テスト | 2933→2940 passed / 0 failed |
| 新規テスト | 7件 |

---

## 背景

24時間ログ分析で Bot 稼働時間の大半が停止状態と判明:

| 時間帯 | 状態 | 詳細 |
|---|---|---|
| 00:00–09:06 | 停止 (9h) | global DD halt |
| 09:06–12:10 | 取引 (3h) | 33 fills, -33.9bps |
| 12:10–EOD | 停止 (2h+) | DynamicKill deadlock |

**根本原因**: DynamicKill で両 side が kill → rolling50 に新データが入らない → 閾値を下回り続ける → 永久停止

---

## 修正内容

### Fix 1: `max_stale_kill_cycles` — Probe サイクル新設

| 項目 | 内容 |
|---|---|
| ファイル | `ztb/risk/sell_dynamic_kill.py` |
| 変更 | kill 後 `max_stale_kill_cycles` サイクル経過で一時的に kill 解除 (probe) |
| デフォルト | 30 サイクル (= 60min @ 2min/cycle) |
| 目的 | 新しい取引データを rolling window に注入し、再判定を可能にする |

### Fix 2: `per_side_halt_cycles` 0→15

| 項目 | 内容 |
|---|---|
| ファイル | `configs/v460/fill_test.yaml` |
| 変更 | 同一 side の連続ブロック上限を設定 |
| 効果 | 片側が無期限にブロックされることを防止 |

### Fix 3: Deadlock Detection Log

| 項目 | 内容 |
|---|---|
| ファイル | `scripts/v460/lib/fill_loop_orchestrator.py` |
| 変更 | `_consecutive_gate_blocks` カウンタ追加。10/20 サイクル連続ブロックで WARNING ログ |
| 目的 | デッドロック状態の可視化・早期検知 |

---

## テスト

7件の新規テスト追加:
- `test_218_anti_deadlock.py`: probe cycle 発火、per_side_halt 動作、detection log
- 既存テスト全パス: 2940 passed / 0 failed
