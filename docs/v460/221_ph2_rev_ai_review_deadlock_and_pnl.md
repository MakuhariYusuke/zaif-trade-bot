# 221# AI Review: デッドロック対策 (218#-220#) 総合レビュー + PnL 構造分析

> **目的**: Codex / Gemini による外部 AI レビュー用ドキュメント  
> **対象**: v460 BTC/JPY maker-bot — coincheck 取引所  
> **日付**: 2026-03-02  
> **Git HEAD**: `f4630bd10` (docs) / Code: `2243c90f4` (220#)  
> **レビュー依頼**: 設計妥当性、改善提案、リスク指摘

---

## 1. エグゼクティブサマリー

v460 は BTC/JPY の maker-only limit order bot。  
218#-220# でデッドロック対策を3段階で実装し、220# の DUAL KILL bypass で **即時復帰 (0秒)** を実現。  
しかし **Rolling50 PnL が構造的にマイナス** であり、Kill が恒常的に発動 → bypass でしか取引できない問題が顕在化。

### Key Metrics (2026-03-02)
| 指標 | 値 |
|------|-----|
| 本日 fill 数 | 42 |
| 本日 PnL 合計 | -25.48 bps |
| 平均 PnL | -0.607 bps |
| 勝率 (WR) | 45.2% |
| Buy rolling50 | -1.226 bps (threshold: -0.8) |
| Sell rolling50 | -3.420 bps (threshold: -0.5) |
| DUAL KILL bypass発動 | 6回 |
| Per-side DD halt (sell) | 3回 |
| 大損失 (>10bps) | 2件 (-13.51, -21.35) |

### 週間トレンド (悪化傾向)
| 日付 | fills | sum (bps) | avg (bps) | WR |
|------|-------|-----------|-----------|-----|
| 2/24 | 141 | +47.2 | +0.335 | 49.6% |
| 2/25 | 162 | -34.5 | -0.213 | 48.1% |
| 2/26 | 222 | -208.2 | -0.938 | 44.6% |
| 2/27 | 163 | +40.0 | +0.246 | 44.8% |
| 2/28 | 170 | -126.1 | -0.742 | 42.4% |
| 3/1 | 45 | -119.6 | -2.657 | 37.8% |
| 3/2 | 42 | -25.5 | -0.607 | 45.2% |

**全期間 (2167 fills)**: avg = -0.323 bps, sum = -699.5 bps

---

## 2. アーキテクチャ概要

```
┌─────────────────────────────────────┐
│     FillLoopOrchestratorMixin       │ ← メインループ
│ ┌─────────────────────────────────┐ │
│ │  CycleGateAggregator (9 gates)  │ │ ← per-cycle skip/block 判定
│ │  Gate1: unknown_regime_buy      │ │
│ │  Gate2: ranging_buy_low_vol     │ │
│ │  Gate3: trending_sell           │ │
│ │  Gate4: buy_dynamic_kill ←──────┼─┤── DynamicKillManager (buy)
│ │  Gate5: sell_dynamic_kill ←─────┼─┤── DynamicKillManager (sell)
│ │  Gate6: velocity_skip           │ │
│ │  Gate7: unknown_regime_sell     │ │
│ │  Gate8: narrow_spread_pause     │ │
│ │  Gate9: maker_price_precheck    │ │
│ └─────────────────────────────────┘ │
│ ┌─────────────────┐ ┌────────────┐  │
│ │ FillCycleExecutor│ │ MakerPrice │  │
│ │ (発注/約定/PnL) │ │ Calculator │  │
│ └─────────────────┘ └────────────┘  │
│ ┌──────────────────────────────────┐ │
│ │    DailyDrawdownGuard            │ │
│ │    (hard -50bps / soft -30bps)   │ │
│ │    per-side halt (-30bps)        │ │
│ └──────────────────────────────────┘ │
└─────────────────────────────────────┘
```

---

## 3. デッドロック対策の3層構造 (218#-220#)

### Layer 1: 218# DynamicKill Probe (commit `36177b2ae`)
- **問題**: kill 発動中に fill されない → 新データなし → rolling50 更新されない → kill が永続化
- **対策**: `max_stale_kill_cycles=10` — Kill 状態で N サイクル経過したら 1 cycle だけ probe (発注許可)
- **per_side_halt_cycles**: 0 → 15 (永続 halt → 30分で解除)

### Layer 2: 219# Progressive Probe + Force Release (commit `e9a979dbe`)
- **問題**: probe (10 cycle 間隔) が遅すぎて回復に 20+ 分かかる
- **対策**: 
  - Progressive interval halving: 10 → 5 → 3 → 2 → 2 cycles
  - Force release: 5回連続 probe が空振り → kill 強制解除

### Layer 3: 220# Gate-Level Deadlock Breaker (commit `2243c90f4`)
- **問題**: buy + sell 両方 kill → どちらも発注不可 → 完全停止
- **対策**: 
  - **DUAL KILL bypass**: Gate4/5 で `is_buy_killed and is_sell_killed` 時に即座通過
  - **Gate7 balance_forced 対称性**: Gate1 と対称的に sell 側も balance_forced で bypass
  - **Unknown regime consecutive bypass**: `_consecutive_unknown_blocks >= 10` で強制通過

### 結果
- 220# 以前: デッドロック → 20分以上の停止
- 220# 以後: DUAL KILL bypass で **0秒** で復帰 (4→6回発動、全て正常動作)

---

## 4. 発見された構造的問題

### 4.1 Kill-Bypass-Rekill ループ (致命的)

```
Rolling50 → threshold 下回る → Kill 発動 (cooldown=10)
  ↓
cooldown 消費 (10 cycle ≈ 20分)
  ↓
cooldown=0 → 再判定 → Rolling50 は依然マイナス → 即座に Kill 再発動
  ↓
DUAL KILL (buy+sell 両方 Kill) → bypass で 1 fill 許可
  ↓
その 1 fill の PnL が Rolling50 に入るが、改善不十分
  ↓
→ 最初に戻る (無限ループ)
```

**根本原因分析**:
- buy rolling50 = -1.226 bps vs threshold -0.8 bps → **超過幅: 0.426 bps**
- sell rolling50 = -3.420 bps vs threshold -0.5 bps → **超過幅: 2.920 bps**
- 1 fill では rolling50 の 1/50 しか更新されない
- 平均 PnL = -0.607 bps の状況では、bypass fill が正 PnL でも回復に **25-50 fill** 必要
- しかし DUAL KILL bypass は 1 fill ずつしか許可しない → 構造的に回復不可能

### 4.2 Per-Side DD Halt の連鎖崩壊

```
本日のタイムライン:
00:00-09:06  DD halt (前日の -110.94bps)
09:06-11:13  取引再開 → sell が 32 fills で -32.51bps 蓄積
11:13        PER-SIDE HALT: sell 封鎖 (threshold -30bps)
11:13-12:10  buy only で 1 fill → sell halt 解除 → 即再halt
12:10        PER-SIDE HALT 2回目: sell -30.06bps
12:10-14:55  買いのみ (PnL 改善なし)
14:55-15:35  再起動で 6 fills (-2.8bps)
16:24        220# 再起動 → DUAL KILL bypass
16:40        PER-SIDE HALT 3回目: sell -30.40bps (15 cycle 限定)
```

**sell side の慢性的劣勢**:
- 本日の sell PnL が繰り返し -30bps halt を触発
- Per-side halt → buy only → 不均衡拡大 → balance_forced → 不利な sell 実行

### 4.3 テールリスクの肥大

全期間 PnL パーセンタイル:
| パーセンタイル | PnL (bps) |
|------------|----------|
| P1 (worst 1%) | -16.5 |
| P5 | -8.02 |
| P10 | -5.79 |
| P25 | -2.58 |
| P50 (中央値) | -0.21 |
| P75 | +1.80 |
| P90 | +4.91 |
| P95 | +7.55 |
| P99 (best 1%) | +15.98 |

**問題**: P50 が -0.21 (マイナス中央値) → 半数以上の取引が損失  
**非対称性**: P1 (-16.5) vs P99 (+16.0) → テール損失の方が大きい  
**大損失 (>10bps)**: 全体の 3.3% (71/2167) が損失全体の不釣り合いに大きい割合を占める

### 4.4 sell_guard spread 拒否の損失

`sell_max_spread_jpy=5000` で sell が拒否されるケース (本日3回):
- spread 5080, 5616, 6745 JPY での sell 不可
- 高 spread 時は利益チャンスでもあるため、一律拒否は機会損失の可能性

---

## 5. コード品質 — レビューポイント

### 5.1 DynamicKillManager の設計

**ファイル**: `ztb/risk/sell_dynamic_kill.py` (334行)

```python
# 問題1: stale_counter のインクリメント位置が分散
# check_kill() 内の3箇所で self._stale_counter += 1
# → cooldown 中 (L194), kill 発動時 (L211), 
#    いずれも track() なしで呼ばれた場合のカウント

# 問題2: check_kill() が副作用を持つ (状態変更 + 判定)
# → _stale_counter, _cooldown, _total_kills, _consecutive_probes, 
#    _force_released 全てが check_kill() 内で変更される
# → 呼び出し順序依存が暗黙的

# 問題3: rolling mean は fill 発生時のみ更新される
# → 市場環境が変化しても rolling50 は直近50 fill のみ反映
# → 取引頻度が低下すると rolling mean が時間的に陳腐化
```

**レビュー質問**:
1. `check_kill()` の副作用を Query/Command に分離すべきか？
2. rolling mean に時間減衰 (exponential decay) を適用すべきか？
3. window サイズを市場状態に応じて動的変更すべきか？

### 5.2 CycleGateAggregator の順序依存性

**ファイル**: `scripts/v460/lib/cycle_gate_aggregator.py` (634行)

```python
# Gate4, Gate5 で dual_kill_bypass を判定
# → Gate1, Gate7 で unknown_bypass を判定
# → 各 bypass フラグは evaluate() 冒頭で計算
# → Gate 順序に暗黙の優先順位が存在

# 問題: Gate1 (buy unknown skip) が Gate7 (sell unknown skip) より先に評価される
# → sell+unknown の場合、Gate1 を通過しても Gate7 で止まる可能性
# → しかし Gate7 にも同じ bypass が適用されるため実害なし (現状)

# 潜在リスク: 新 Gate 追加時に順序が意味を持つことを見落とす可能性
```

### 5.3 fill_loop_orchestrator の肥大化

**ファイル**: `scripts/v460/lib/fill_loop_orchestrator.py` (1809行)

- 自己申告の MAX LINES: 1200 — **実際: 1809行 (超過)**
- Mixin パターンで `FillTestRunner` に注入 → テスト時の mock が複雑
- `_consecutive_gate_blocks`, `_balance_forced_skip_count`, `_trending_sell_skip_count` 等のカウンタが分散
- `run_continuous()` メソッド単体で推定 600+ 行

---

## 6. 改善提案 (優先度順)

### P0: Kill Threshold の緊急見直し (収益直結)

**現状のパラメータ問題と提案値**:

| パラメータ | 現行値 | 問題 | 提案値 | 理由 |
|-----------|--------|------|--------|------|
| sell threshold | -0.5 bps | rolling50=-3.4で即発動 | -2.0 bps | 中央値考慮で緩和 |
| buy threshold | -0.8 bps | rolling50=-1.2で即発動 | -1.5 bps | 現rolling50近傍 |
| sell trending_up | -0.3 bps | trending_up でも kill | -1.0 bps | trending_up は sell 有利 |
| resume_window | 10 cycles | 20分停止 | 5 cycles | 高速回復 |

### P1: Rolling Mean の時間減衰導入

```python
# 提案: Exponential Weighted Moving Average (EWMA)
# 現在: simple rolling mean (直近50 fill の等重み平均)
# 問題: 古い fill の影響が50 fill 分持続
# 提案: 半減期ベースの指数減衰

class DynamicKillManager:
    def check_kill(self, regime=None):
        # 現在の単純平均の代わりに:
        #   ewma = sum(pnl[i] * alpha^(n-i)) / sum(alpha^(n-i))
        #   alpha = 0.5^(1/half_life)  # half_life=20 fills
        pass
```

### P2: テールリスク制御の強化

**問題**: >10bps 損失 (3.3%) が P&L を支配

**提案**:
- **即時ストップ**: fill 後 10秒以内に mid price が -8bps 以上動いたら rapid exit
- **Toxic fill veto 拡張**: threshold -5bps → -3bps (現在の P25=-2.58bps 考慮)
- **Loss cooldown mult 強化**: 大損失後の interval 延長を現行以上に

### P3: Sell Guard のインテリジェント化

**現状**: `sell_max_spread_jpy=5000` の一律閾値
**問題**: volatility 高い時は spread 広い方が利益率が高い
**提案**: 
- spread 閾値をボラティリティに連動: `max_spread = base × (1 + vol_ratio)`
- regime=trending 時は max_spread を 1.5x に緩和

### P4: Per-Side DD Guard の改善

**問題**: sell が -30bps で halt → buy only → 不均衡 → 悪化

**提案**:
- Per-side halt 後の最大 buy-only 継続を制限 (例: 10 cycles)
- Per-side halt 解除時に sell threshold を一時緩和
- Per-side halt カウンタを fill 数ベースに変更 (cycle 数は待機込みで不正確)

### P5: DUAL KILL Bypass の段階的強化

**現状**: bypass で 1 fill → 即 rekill → 再 bypass (ループ)
**提案**: 
- Bypass 後の cooldown を kill manager 側でリセット (現在リセットされていない)
- Bypass 成功時に `max_stale_kill_cycles` を一時的に 0 に (probe 無効化)
- Kill threshold を一時的に 2x 緩和する "recovery mode" の導入

### P6: Gate 評価のリファクタリング

- Gate 順序の明示的なドキュメント化 (依存関係図)
- bypass フラグの計算を evaluate() 先頭で一括化 (現状は partially done)
- Gate 評価結果の統計をメトリクスとして定期出力

---

## 7. 質問事項 (AI レビューアへ)

1. **Kill Manager の設計**: check_kill() が Query + Command を兼ねている (CQS 違反)。分離価値は高いか、それとも MM bot の特殊性を考えると現実的か？

2. **Rolling Window の適正サイズ**: window=50 は BTC/JPY の市場特性に対して適切か？ 取引頻度 (~50-200 fills/day) を考慮すると、window=50 は 6-24 時間分のデータに相当。半日前の市場データで現在の判断をすることの妥当性は？

3. **DUAL KILL の設計思想**: Both sides killed → force through の方針は、「損を拡大するリスク」vs「完全停止のリスク」のトレードオフ。どちらが maker-bot として正しいか？

4. **Correction vs Adaptation**: rolling mean がマイナスの時に「取引を止める (kill)」のか「パラメータを調整する (offset 拡大、lot 縮小)」のか、どちらがより適切な対応策か？

5. **テールリスクの構造的原因**: P1=-16.5bps の大損失は逆選択 (adverse selection) か、単にボラティリティの結果か？ 逆選択なら offset で防げるか？

6. **Sell 側の構造的劣勢**: sell avg PnL が buy avg PnL を大幅に下回る傾向がある。これは BTC/JPY の構造的バイアス (上昇トレンド) か、bot のパラメータ設定の問題か？

7. **デッドロック対策の層数**: 3層 (218#/219#/220#) は過剰ではないか？ 220# の DUAL KILL bypass だけで十分なのでは？ 逆に、Layer 1-2 を削除した場合のリスクは？

---

## 8. 補足データ

### PnL 分布 (全 2167 fills)
```
大損失 > 5bps:  279件 (12.9%)
大損失 > 10bps:  71件 (3.3%)
大損失 > 15bps:  26件 (1.2%)
大損失 > 20bps:  14件 (0.6%)
```

### Fill Wait Time
```
Min: 5.9s, Max: 65.9s, Avg: 14.0s
Long waits (>30s): 4件
```

### 本日の時間帯別 PnL
```
Hour 09: 11 fills, -5.8bps, avg -0.53bps
Hour 10: 11 fills, -15.8bps, avg -1.44bps
Hour 11: 10 fills, -11.2bps, avg -1.12bps
Hour 12:  1 fill,  -1.1bps
Hour 14:  1 fill,  -3.0bps
Hour 15:  5 fills, -2.8bps, avg -0.55bps
Hour 16:  5 fills, +12.9bps, avg +2.58bps ← 220# DUAL KILL bypass 期間
```

### Regime 分布 (本日)
```
ranging:       227 cycles (81.9%)
trending_down:  26 cycles (9.4%)
trending_up:    25 cycles (9.0%)
```

### 本日のイベント頻度
```
Toxic fill veto:    4回
Maker price rejected (spread): 3回
Balance forced:     55回
One-sided balance:  58回
post_only_reject:   1回
status_unknown:     2回
```

---

## 9. 関連ファイル一覧

| ファイル | 行数 | 概要 |
|---------|------|------|
| `ztb/risk/sell_dynamic_kill.py` | 334 | DynamicKillManager (buy/sell 共用) |
| `scripts/v460/lib/cycle_gate_aggregator.py` | 634 | 9-Gate per-cycle skip 判定 |
| `scripts/v460/lib/fill_loop_orchestrator.py` | 1809 | メインループ Mixin |
| `scripts/v460/lib/fill_cycle_executor.py` | ~700 | 単一サイクル実行 |
| `scripts/v460/lib/maker_price.py` | 910 | Maker price 算出 |
| `scripts/v460/lib/daily_drawdown_guard.py` | ~300 | DD guard |
| `configs/v460/fill_test.yaml` | 712 | 全パラメータ |

---

## 10. このドキュメントの使い方

1. このファイルを AI (Codex / Gemini) に渡す
2. §7 の質問に回答してもらう
3. §6 の提案に対するフィードバックを得る
4. 追加の改善点を発見してもらう
5. コードの§5 で指摘した箇所の具体的なリファクタリング案を求める

**注意**: このドキュメントは v460 の設計レビュー用であり、コードの全貌は含まれていない。詳細なコードレビューには関連ファイルの全文が必要。
