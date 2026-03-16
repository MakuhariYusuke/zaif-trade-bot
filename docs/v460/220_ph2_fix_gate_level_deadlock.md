# 220# Gate-level Deadlock 3fixes: デッドロック根絶

| key | value |
|---|---|
| 対象 | CycleGateAggregator のデッドロック脆弱性 3件修正 |
| commit | `2243c90f4` |
| テスト | 2949→2971 passed / 0 failed |
| バグ修正 | 3件 (MEDIUM 2, LOW 1) |
| 新規テスト | 22件 (`test_220_deadlock_fixes.py`) |

---

## デッドロック監査結果

26 のブロッキングメカニズムを網羅的に調査:

| リスク | 件数 | 内容 |
|---|---|---|
| MEDIUM | 2 | dual-kill (#7), unknown_regime Gate7 非対称バグ (#9) |
| LOW | 24 | 全て自動回復メカニズムあり |

---

## Fix A: Gate 7 `_check_unknown_regime_sell` — `balance_forced` bypass 追加

| 項目 | 内容 |
|---|---|
| ファイル | `scripts/v460/lib/cycle_gate_aggregator.py` |
| 原因 | Gate 1 (buy) は `not balance_forced` 条件あり → bypass 可。Gate 7 (sell) には条件なし → **非対称バグ** |
| 影響 | `balance_forced=True` 時に buy は unknown regime を通過できるが sell は通過不可 → 在庫偏重時にデッドロック |
| 修正 | `_check_unknown_regime_sell()` に `balance_forced: bool` 引数追加、`not balance_forced` 条件追加 |
| 重大度 | MEDIUM |

---

## Fix B: Dual-kill Deadlock Breaker

| 項目 | 内容 |
|---|---|
| ファイル | `scripts/v460/lib/cycle_gate_aggregator.py` |
| 原因 | buy_kill + sell_kill 同時発生時、Gate 4/5 が交互にブロック → 両 side 完全停止 |
| 影響 | 219# force release (5 probes ≈ 40min) まで完全停止 |
| 修正 | `evaluate()` 内で `is_buy_killed and is_sell_killed` を検出 → `_dual_kill_bypass=True` → Gate 4/5 をバイパス |
| 効果 | デッドロック回復: 最大 ~40min → **即座 (0 cycle)** |
| 重大度 | MEDIUM |

---

## Fix C: Unknown Regime 連続ブロック自動バイパス

| 項目 | 内容 |
|---|---|
| ファイル | `scripts/v460/lib/cycle_gate_aggregator.py` |
| 原因 | プロセス再起動 + 状態復元失敗 → regime=unknown → Gate1/7 が両 side 無期限ブロック |
| 対策 | `_consecutive_unknown_blocks` カウンタ → 10 サイクル連続 unknown で強制通過 |
| 定数 | `UNKNOWN_REGIME_MAX_CONSECUTIVE = 10` |
| リセット | non-unknown regime 通過時にカウンタ = 0 |
| 重大度 | LOW (通常数サイクルで regime 復帰) — 防御的対策 |

---

## テスト (22件)

`tests/unit/v460/test_220_deadlock_fixes.py`:

| クラス | テスト数 | カバー範囲 |
|---|---|---|
| `TestGate7BalanceForcedBypass` | 5 | sell/buy unknown × balance_forced 対称性 |
| `TestDualKillBreaker` | 7 | 片方 kill, 両方 kill, balance_forced 組合せ |
| `TestUnknownRegimeConsecutiveBypass` | 8 | カウンタ増減, リセット, MAX 到達 bypass, 混合 side |
| `TestDeadlockIntegration` | 2 | triple deadlock bypass, 9 ゲート維持確認 |

---

## デッドロック対策の全体像 (218#–220#)

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
