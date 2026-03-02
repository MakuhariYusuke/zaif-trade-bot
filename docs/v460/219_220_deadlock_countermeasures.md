# 219#-220# DynamicKill デッドロック対策 完全ドキュメント

| key | value |
|---|---|
| 対象 | 218# probe → 219# progressive probe + force release → 220# gate-level deadlock 3fixes |
| commits | `36177b2ae` (218#), `e9a979dbe` (219#), 220# (本コミット) |
| テスト | 2971 passed / 0 failed |
| バグ修正 | 4件 (MEDIUM 2, LOW 2) |
| 新規テスト | 31件 (219# 9件 + 220# 22件) |

---

## 背景: デッドロック問題

24時間ログ分析から、Bot が大半の時間停止していることが判明:

| 時間帯 | 状態 | 詳細 |
|---|---|---|
| 00:00–09:06 | 停止 (9h) | global DD halt |
| 09:06–12:10 | 取引 (3h) | 33 fills, -33.9bps |
| 12:10–EOD | 停止 (2h+) | DynamicKill deadlock |

**根本原因**: DynamicKill で両 side が kill → probe サイクルが長すぎ (60min) → rolling50 が動かない → 再 kill → 無限ループ

---

## 218# Anti-Deadlock (commit `36177b2ae`)

| 項目 | 内容 |
|---|---|
| `max_stale_kill_cycles` | 新設。kill 後一定サイクルでプローブ発火 |
| `per_side_halt_cycles` | 0→15。同一 side 連続ブロック上限 |
| deadlock detection log | WARNING ログ (10/20 cycle 連続ブロック) |

---

## 219# Progressive Probe + Force Release (commit `e9a979dbe`)

### 問題点
218# probe は `max_stale=30` (60min) で1回だけ → 一時的に kill 解除 → rolling50 微動 → 再 kill → 次の probe まで60min待ち

### 3つの修正

#### Fix 1: `max_stale_kill_cycles` 30→10

| 項目 | 内容 |
|---|---|
| ファイル | `ztb/risk/sell_dynamic_kill.py` |
| 変更 | デフォルト 30→10 (60min→20min で初回 probe) |

#### Fix 2: Progressive probe interval

| 項目 | 内容 |
|---|---|
| ファイル | `ztb/risk/sell_dynamic_kill.py` |
| メソッド | `_effective_probe_interval()` |
| ロジック | base=10 → 5 → 3 → 2 → 2 (半減、min=2) |
| 新フィールド | `_consecutive_probes: int`, `min_probe_interval: int = 2` |

#### Fix 3: Force release after N probes

| 項目 | 内容 |
|---|---|
| ファイル | `ztb/risk/sell_dynamic_kill.py` |
| ロジック | `_consecutive_probes >= max_force_release_probes` → kill 永久解除 |
| 新フィールド | `_force_released: bool`, `max_force_release_probes: int = 5` |
| リセット | `track()` で `_consecutive_probes=0`, `_force_released=False` |

### 本番確認
- probe 14:54 に発火 (再起動20min後) ✅
- Buy fill @ 10,508,393 JPY, pnl=-3.03bps
- rolling50 mean: -1.283→-1.278bps (微改善)

---

## 220# Gate-level Deadlock 3fixes (本コミット)

### デッドロック監査結果

26 のブロッキングメカニズムを網羅的に調査:

| リスク | 件数 | 内容 |
|---|---|---|
| MEDIUM | 2 | dual-kill (#7), unknown_regime Gate7 asymmetry (#9) |
| LOW | 24 | 全て自動回復メカニズムあり |

### Fix A: Gate 7 `_check_unknown_regime_sell` — balance_forced bypass 追加

| 項目 | 内容 |
|---|---|
| ファイル | `scripts/v460/lib/cycle_gate_aggregator.py` |
| 原因 | Gate 1 (buy) は `not balance_forced` 条件あり → bypass 可。Gate 7 (sell) には条件なし → 非対称バグ |
| 影響 | `balance_forced=True` 時に buy は unknown regime を通過できるが sell は通過不可 → 在庫偏重時にデッドロック |
| 修正 | `_check_unknown_regime_sell()` に `balance_forced: bool` 引数追加、`not balance_forced` 条件追加 |
| 重大度 | MEDIUM — balance_forced と unknown_regime が同時発生時に sell 側デッドロック |

### Fix B: Dual-kill deadlock breaker

| 項目 | 内容 |
|---|---|
| ファイル | `scripts/v460/lib/cycle_gate_aggregator.py` |
| 原因 | buy_kill + sell_kill 同時発生時、Gate 4/5 が交互にブロック → 両 side 停止 |
| 影響 | 219# force release (5 probes ≈ 100min) まで完全停止 |
| 修正 | `evaluate()` 内で `is_buy_killed and is_sell_killed` を検出 → `_dual_kill_bypass=True` → Gate 4/5 をバイパス |
| ロジック | dual kill 検出時、即座に取引許可 (gate レベル高速化) |
| 重大度 | MEDIUM — 最大100min のデッドロックを0サイクルに短縮 |

### Fix C: Unknown regime 連続ブロック自動バイパス

| 項目 | 内容 |
|---|---|
| ファイル | `scripts/v460/lib/cycle_gate_aggregator.py` |
| 原因 | プロセス再起動 + 状態復元失敗 → regime=unknown → Gate1/7 が両 side 無期限ブロック |
| 対策 | `_consecutive_unknown_blocks` カウンタ → 10サイクル連続 unknown で強制通過 |
| 定数 | `UNKNOWN_REGIME_MAX_CONSECUTIVE = 10` |
| リセット | non-unknown regime 通過時にカウンタ=0 |
| 重大度 | LOW (通常数サイクルで regime 復帰) — だが防御的対策として実装 |

---

## テスト

### 219# テスト (9件) — `test_219_progressive_probe.py`

- progressive interval halves / effective calculation / force release after N
- force release ends on track / zero disables / track resets consecutive
- export/import / BuyManager works / default config values

### 220# テスト (22件) — `test_220_deadlock_fixes.py`

| クラス | テスト数 | カバー範囲 |
|---|---|---|
| `TestGate7BalanceForcedBypass` | 5 | sell/buy unknown × balance_forced 対称性 |
| `TestDualKillBreaker` | 7 | 片方kill, 両方kill, balance_forced組合せ |
| `TestUnknownRegimeConsecutiveBypass` | 8 | カウンタ増減, リセット, MAX到達bypass, 混合side |
| `TestDeadlockIntegration` | 2 | triple deadlock bypass, 9ゲート維持確認 |

### リグレッション結果

```
2971 passed, 0 failed, 20 warnings in 149.09s
```

前回 2949 + 新規 22 = 2971

---

## デッドロック対策の全体像

```
                    ┌─────────────────────────────────┐
                    │    DynamicKill deadlock 対策     │
                    └─────────────────────────────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                      ▼
    ┌───────────────┐    ┌─────────────────┐    ┌──────────────────┐
    │ 218# Layer 1  │    │ 219# Layer 2    │    │ 220# Layer 3     │
    │ 基本 probe    │    │ Progressive     │    │ Gate-level       │
    │               │    │ probe + force   │    │ deadlock fix     │
    ├───────────────┤    ├─────────────────┤    ├──────────────────┤
    │ max_stale     │    │ interval: 半減  │    │ Gate7 balance_   │
    │ per_side_halt │    │ force release   │    │ forced bypass    │
    │ detection log │    │ after 5 probes  │    │ Dual-kill bypass │
    │               │    │ max_stale 30→10 │    │ Unknown N-cycle  │
    └───────┬───────┘    └────────┬────────┘    └────────┬─────────┘
            │                     │                      │
            ▼                     ▼                      ▼
      回復: ~60min          回復: ~20-40min         回復: 即座 (0cycle)
```
