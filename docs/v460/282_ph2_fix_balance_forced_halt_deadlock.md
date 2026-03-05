# 282# balance_forced + per-side halt デッドロック修正

| 項目 | 内容 |
|------|------|
| 日付 | 2026-03-05 |
| 起票 | Copilot (担当者レビュー済) |
| 前提 | v460 fill_test on Coincheck, HEAD=91f050a76 (280#) |
| レビュー依頼先 | Codex / Gemini |

---

## 1. インシデント概要

2026-03-05 13:10 JST ～ 21:21 JST (8時間以上)、fill_test プロセスが **完全なデッドロック** に陥り、取引が一切実行されなかった。

### 状態

| 項目 | 値 |
|------|-----|
| `daily_pnl_bps` | +24.9 (全体はプラス) |
| `daily_pnl_bps_buy` | -33.9bps → per-side halt 発動 |
| `daily_pnl_bps_sell` | +58.8bps (sell は好調) |
| `side_halted_buy` | True |
| `side_halt_remaining_buy` | 12 (固定、デクリメントなし) |
| BTC 残高 | 0.000000 |
| JPY 残高 | 23,805.79 |
| スキップ回数 | 236 回 (2分間隔で反復) |

### デッドロックのループ

```
tick_side_halt()          ← halt_remaining: 12→11
  ↓
[205#] buy halted → sell に切替
  ↓
[balance] BTC=0 → sell 不可 → buy に戻す (balance_forced)
  ↓
[222#] balance_forced 後の再チェック → buy is halted
  ↓
[269#] Inventory Escape? → next_side=="sell" 条件不一致 (next_side=="buy") → 不発
  ↓
[223#] refuse bypass (safety > liveness)
  ↓
untick_side_halt()        ← halt_remaining: 11→12 (tick を補償)
  ↓
continue                 → 2分待機 → 最初に戻る
```

**結果**: `halt_remaining` は永久に 12 のまま。halt が解除されることがない。

---

## 2. 根本原因 (Root Cause)

### 原因 A: 273# I3 の `untick_side_halt()` — halt カウントダウン停止

273# I3 で追加された `untick_side_halt()` は、空サイクル（取引試行なし）の halt カウントを除外する目的で tick を補償する。

**設計意図** (268# 分析): デッドロック中に halt_remaining が消費されると、halt 解除後も PnL 改善がないため即再halt → 30分ごとの halt/解除サイクルの無駄を防ぐ。

**副作用**: halt カウントダウンが **完全に停止** し、balance 制約と組み合わさると **永久デッドロック** が発生する。30分サイクルの無駄を防ぐつもりが、8時間以上の完全停止を引き起こした。

### 原因 B: Inventory Escape が sell 方向のみ

269# P0 (Gemini 270# Action A) で導入された Inventory Escape は sell 方向のみ対応:
```python
if _ie_enabled and next_side == "sell":  # ← sell 限定
```

buy halt + BTC=0 のパターンでは `next_side == "buy"` であるため、Inventory Escape が作動しない。

**269# の設計意図**: BTC 過剰在庫の「縮退清算」(= sell) を想定。buy 方向の脱出は考慮外だった。

---

## 3. 修正内容

### Fix A: `untick_side_halt()` 除去 (2箇所)

| パス | 変更前 | 変更後 |
|------|--------|--------|
| `balance_forced_halt_block` (L1855) | `untick_side_halt()` あり | **除去** |
| `per_side_dd_both_halt` (L1604) | `untick_side_halt()` あり | **除去** |

**理由**: halt は `per_side_halt_cycles` (=15) 回のサイクルで自然満了する設計。untick による補償は永久デッドロックの原因となるため除去。halt 解除後は reanchor (269#) が PnL 基準をリセットし、安全に取引再開する。

**halt 所要時間**: 15 cycles × 120s = **30分** で halt 自然解除 (旧: 永久停止)

### Fix B: Inventory Escape 双方向化

```python
# BEFORE: sell のみ
if _ie_enabled and next_side == "sell":

# AFTER: buy/sell 両方向
if _ie_enabled:
```

**理由**: BTC=0 + buy halt のパターンでも degraded params で縮退取得を許可。duty cycle (1-in-5) で頻度制限。

### 修正の効果 (新挙動)

```
tick_side_halt()          ← halt_remaining: 12→11
[269# IE] duty cycle 1/5: Inventory Escape 発動 → degraded buy 実行
          duty cycle 2-5/5: skip, halt カウントダウン継続 (untick なし)
...
12 cycles (24分) 後: halt 自然解除 → 通常取引再開
reanchor (269#) → PnL 基準リセット → 安全な再開
```

---

## 4. 13:08 以前のログ分析

### 取引サマリ (09:00-13:08 JST)

| 項目 | 値 |
|------|-----|
| 総 fill 数 | 25 |
| Buy fill (PnL) | 12 fills, -12.01 bps |
| Sell fill (PnL) | 13 fills, -11.31 bps |
| BF (balance_forced) buy | 6/12 = 50% |
| 最大単一損失 | sell #1: -11.91 bps (09:00 開始直後) |

### スキップ分析 (pre-13:08)

| スキップ理由 | 回数 | 備考 |
|-------------|------|------|
| `buy_dynamic_kill` | 52 | **圧倒的多数** — buy 側の kill manager が長時間作動 |
| `skip_gate` | 8 | 正常なゲート制御 |
| `sell_guard_reject` | 4 | sell 側の保護動作 |
| `spread_too_narrow` | 3 | スプレッド不足 |
| `toxic_fill_side_veto` | 3 | 毒性拒否 |
| `degraded_liquidation_duty_skip` | 2 | 縮退清算のデューティスキップ |

### 重要パターン

1. **buy_dynamic_kill 長時間抑制**: 09:06-09:47 に 23 連続スキップ (41分) 。buy 側が開始直後から長時間 kill されていた。market が変化しても buy 復帰が遅い。

2. **BF buy 比率が高い (50%)**: balance_forced による buy が多発 → 「sell で BTC 消費 → BTC=0 → forced buy」のパターン。forced buy は market タイミングに関係なく発動するため、PnL が悪化しやすい。

3. **午後の buy 損失急激化**: #19 (-6.97), #21 (-4.45), #24 (-5.77) → buy 側 PnL が急速に悪化し per-side halt トリガー。

4. **sell 側の午後回復**: #22 (+2.32), #23 (+4.41) → sell 側は午後に回復基調だったが、halt 後のデッドロックで活用できなかった。

### 改善検討事項

| 項目 | 内容 | 優先度 |
|------|------|--------|
| buy_dynamic_kill 解除速度 | 41分の連続 kill は長い。`max_kill_duration_sec` / `resume_window` の見直し | Medium |
| BF buy の品質向上 | balance_forced buy のオフセット / ゲート条件の再検討 | Medium |
| 片側集中型のリスク | buy kill → sell 集中 → BTC 枯渇 → forced buy → buy halt のフィードバックループ | High |
| 朝の大損失耐性 | 初回取引 (-11.91 bps) で一日の tone が決まる問題 | Low |

---

## 5. レビュー依頼事項

### Q1: 273# I3 `untick_side_halt()` の完全除去は適切か？

**268# の懸念**: halt 満了 → 改善なく即再halt → 30分ごとの無駄なサイクル。

**281# の反論**: 
- reanchor (269#) が解除後の PnL 基準をリセット → 即再halt にはならない (新たな budget が付与)
- 30分サイクルの「無駄」より、8時間の完全停止の方が遥かに有害
- halt 中も market は変動しており、30分後の状況は異なる可能性が高い

**質問**: untick 除去以外に、halt カウントダウンと balance 制約を両立する better approach はあるか？

### Q2: Inventory Escape 双方向化のリスク

buy halt は「buy が損失を出している」という信号。IE で buy halt を貫通することは:
- **メリット**: デッドロック脱出、取引再開
- **リスク**: 損失が拡大する可能性

dutyCycle=5 (1-in-5) + degraded params で十分か？ 追加制約は必要か？

### Q3: buy_dynamic_kill と per-side halt のフィードバックループ

```
buy_kill 長時間作動 → sell 集中 → BTC 枯渇 → forced buy 多発 → buy PnL 悪化 → per-side halt → デッドロック
```

このフィードバックループを構造的に断ち切る方法は？

### Q4: `untick_side_halt()` メソッドの存続

orchestrator からの呼出しは全除去した。メソッド自体を deprecated 化すべきか、将来のユースケースのために残すべきか？

---

## 6. テスト結果

| テスト | 結果 |
|--------|------|
| 281# 新規テスト (15件) | ✅ 全パス |
| v460 全テスト (3874件) | ✅ 全パス |
| 回帰テスト | 273# I3 テスト含め全パス (DDG メソッド自体は未変更) |

---

## 7. 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `scripts/v460/lib/fill_loop_orchestrator.py` | Fix A (untick除去×2) + Fix B (IE双方向化) |
| `tests/unit/v460/test_281_deadlock_fix.py` | 新規テスト 15件 |
| `docs/issues/281_deadlock_fix.md` | 本ドキュメント |

---

## 付録: 修正前後のコード対比

### A. balance_forced_halt_block パス

```python
# BEFORE (273# I3):
self._daily_drawdown_guard.untick_side_halt()
self._tick_toxic_veto("halt_block")

# AFTER (281# fix):
# untick_side_halt() 除去 — halt 自然カウントダウン
self._tick_toxic_veto("halt_block")
```

### B. per_side_dd_both_halt パス

```python
# BEFORE (273# I3):
self._inc_guard_fire("per_side_dd_both_halt")
self._daily_drawdown_guard.untick_side_halt()
await self._execute_skip(...)

# AFTER (281# fix):
self._inc_guard_fire("per_side_dd_both_halt")
# untick_side_halt() 除去
await self._execute_skip(...)
```

### C. Inventory Escape 条件

```python
# BEFORE (269# P0):
if _ie_enabled and next_side == "sell":

# AFTER (281# fix):
if _ie_enabled:
```
