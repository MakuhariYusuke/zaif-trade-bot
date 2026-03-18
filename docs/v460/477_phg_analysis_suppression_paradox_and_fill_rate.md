# 477# 抑制パラドックス分析：20層の防御が fill rate 17.7% を生む構造

## Executive Summary

3/17-3/18 の 905 レコード（160 約定 / 745 キャンセル）を分析した結果、**20以上の独立した抑制メカニズムが多重に発火し、注文機会の 82.3% を潰している**ことが判明した。

核心的矛盾：**「儲からない注文を出さない」ために設計された各ガードが、相互作用により「そもそも注文を出せない」状態を作り出している。** PnL mean=-0.44 bps は改善余地があるが、fill rate 17.7% では改善の統計的検出力すら確保できない。

---

## §1 定量分析：何が注文を殺しているのか

### §1.1 Cancel Reason 内訳（3/17-3/18, n=745）

| Rank | Cancel Reason | Count | 全体比 | Side | 種別 |
|------|--------------|-------|--------|------|------|
| 1 | **ranging_low_vol_skip** | 217 | 29.1% | Buy 217 / Sell 0 | HARD SKIP |
| 2 | **preflight_insufficient** | 142 | 19.1% | Buy 71 / Sell 71 | HARD SKIP |
| 3 | **timeout** | 88 | 11.8% | Buy 42 / Sell 46 | 約定せず |
| 4 | **spread_too_narrow** | 70 | 9.4% | Buy 34 / Sell 36 | HARD SKIP |
| 5 | **no_feasible_quote** | 69 | 9.3% | Buy 68 / Sell 1 | HARD SKIP |
| 6 | **skip_gate** | 66 | 8.9% | Buy 30 / Sell 36 | HARD SKIP |
| 7 | preflight_pause | 20 | 2.7% | — | HARD SKIP |
| 8 | cross_venue_lead_lag_veto | 12 | 1.6% | — | HARD SKIP |
| 9 | postonly_crossing_skip | 11 | 1.5% | — | HARD SKIP |
| 10 | buy_dynamic_kill | 11 | 1.5% | Buy | HARD SKIP |
| 11 | final_clamp_hard_skip | 11 | 1.5% | — | HARD SKIP |
| 12 | status_unknown_fast | 8 | 1.1% | — | 不明 |
| 13 | route_to_kill_deadlock | 5 | 0.7% | — | BUG? |
| 14 | post_only_reject | 4 | 0.5% | — | 取引所拒否 |
| 15 | daily_drawdown_halt | 3 | 0.4% | — | HARD SKIP |
| 16 | hard_skip_utc_hour | 3 | 0.4% | — | HARD SKIP |
| 17 | stale_adverse_drift | 2 | 0.3% | — | CANCEL |
| 18 | sell_guard_reject | 2 | 0.3% | — | HARD SKIP |
| 19 | toxic_fill_side_veto | 1 | 0.1% | — | HARD SKIP |

### §1.2 さらにソフト抑制（ログベース, 同期間）

| メカニズム | 発火回数 | 効果 |
|-----------|---------|------|
| EV-weighted offset modulation | 300 | オフセット増減 |
| Cross-venue offset boost | 131 | オフセット拡大 |
| Cooldown lot scale (0.3×) | 129 | ロット 70% 削減 |
| Sell dynamic kill | 35 | 売り注文抹消 |
| Buy dynamic kill | 31 | 買い注文抹消 |
| Toxic fill veto block | 33 | 3サイクル片側停止 |
| DD halt | 47 | 全面停止 |

---

## §2 構造的問題：抑制の多重層がパフォーマンスを殺す

### §2.1 Buy 側の「三重苦」

Buy 側はキャンセル 582 件中 **填充 82 件 (14.1%)** と特に深刻：

```
Buy 注文の運命フロー (n=664 試行):
  → ranging_low_vol_skip で 217 消滅 (32.7%)   ← ★最大の殺し屋
  → preflight_insufficient で 71 消滅 (10.7%)
  → no_feasible_quote で 68 消滅 (10.2%)        ← Buy 偏重
  → spread_too_narrow で 34 消滅 (5.1%)
  → skip_gate で 30 消滅 (4.5%)
  → timeout で 42 失敗 (6.3%)                   ← 出したが約定せず
  → 残り 122 発注, うち 82 が約定 (12.3%)
```

### §2.2 Sell 側の「残高不足」

Sell 側は 288 レコード中 78 約定 (27.1%) と Buy よりマシだが：
- `preflight_insufficient` 71 件 = **BTC 残高 0.00097546 < min_order_btc 0.001** が原因
- 476# で dust sweep を修正したが、Bot 再起動前のデータ

### §2.3 「約定したものの質」

| 指標 | Buy | Sell | 全体 |
|------|-----|------|------|
| PnL 30s mean | -0.15 bps | -0.74 bps | -0.44 bps |
| AS rate | 19.5% | 32.1% | 25.6% |
| PnL 120s mean | — | — | -0.66 bps |

PnL 分布：
- P5: -8.96 bps / P25: -2.51 bps / **Median: +0.10 bps** / P75: +2.68 bps / P95: +7.94 bps
- σ = 5.58 bps

**中央値はプラス (0.10 bps) であり、負の平均は少数の大きな AS 損失（tail risk）に引っ張られている。**

---

## §3 抑制メカニズム全棚卸し（20+層）

### §3.1 HARD SKIP（注文を出さない）

| # | 名称 | ファイル | 閾値 | 影響度 |
|---|------|----------|------|--------|
| 1 | SkipGate ML | skip_gate_evaluator.py | pnl_threshold=-0.5 | 8.9% |
| 2 | Ranging Low Vol Skip | orchestrator_guards.py | vol_ratio<0.65 | **29.1%** |
| 3 | Cross-Venue Veto | maker_risk_guards.py | diff>6bps | 1.6% |
| 4 | Spread Too Narrow | fill_cycle_executor.py | <1000JPY | 9.4% |
| 5 | No Feasible Quote | offset_pipeline.py | パイプライン計算不能 | 9.3% |
| 6 | PostOnly Crossing | fill_cycle_executor.py | price≤best_bid/ask | 1.5% |
| 7 | Final Clamp Hard Skip | offset_pipeline.py | clamp 範囲外 | 1.5% |
| 8 | DD Hard Halt | daily_drawdown_guard.py | pnl≤-50bps | 0.4% |
| 9 | Hard Skip UTC Hour | skip_gate_evaluator.py | 危険時間帯 | 0.4% |
| 10 | Toxic Fill Veto | orchestrator_post_cycle.py | pnl≤-3bps | 0.1% |
| 11 | Sell Guard Reject | maker_risk_guards.py | — | 0.3% |
| 12 | Buy Dynamic Kill | orchestrator_guards.py | EWMA≤-1.0bps | 1.5% |
| 13 | Sell Dynamic Kill | sell_dynamic_kill.py | EWMA≤-0.7bps | — |
| 14 | Unknown Regime Skip | skip_gate config | regime=unknown | — |
| 15 | One-Sided Escalation | orchestrator_guards.py | 連続同方向4+ | — |
| 16 | Quiescence (Dual Kill) | cycle_gate_aggregator.py | 両側 kill | — |
| 17 | Preflight Insufficient | balance_checker.py | BTC<0.001 or JPY不足 | **19.1%** |
| 18 | Velocity Skip Rule | skip_gate_evaluator.py | vel>6bps(sell), <-4bps(buy) | — |

### §3.2 SOFT REDUCTION（ロット・オフセット減衰）

| # | 名称 | 効果 | 発火頻度 |
|---|------|------|---------|
| 1 | DD Soft Lot Scale | lot×0.75 | 5回 |
| 2 | Cooldown Lot Scale | lot×0.3 | 129回 |
| 3 | Alert Mode Lot | lot×0.5 | — |
| 4 | VG Offset Boost | offset×2.0 | — |
| 5 | Cross-Venue Offset Boost | offset 拡大 | 131回 |
| 6 | EV-Weighted Offset | offset±50% | 300回 |
| 7 | Regime Lot Multiplier | lot×regime_mult | — |
| 8 | AS Reservation Price | 非対称 offset | 常時 |
| 9 | Imbalance Risk Guard | offset 拡大 | — |

---

## §4 核心問題の診断

### §4.1 Top-3 殺傷力を持つ抑制

#### 問題 1: `ranging_low_vol_skip` — 29.1% を一挙に排除

- **buy 側限定** で 217/745 キャンセルの最大要因
- ranging regime(全体の 96%) × low volatility → ほぼ常時発動
- Coincheck の BTC/JPY は bitFlyer より流動性が低く spread が広い ≒ **ranging + low vol が「通常の市場状態」**
- この抑制が「通常の市場状態を異常と判定して排除する」という自己矛盾

#### 問題 2: `preflight_insufficient` — 19.1% が残高不足

- Buy 71 件 = JPY 不足（残高 ~15,600 JPY × order_quantity 0.001 BTC ≈ 11,700 JPY → ギリギリ）
- Sell 71 件 = BTC 不足（0.00097546 < min_order_btc 0.001）
- 476# の dust sweep 修正で sell 側は改善見込み
- Buy 側は JPY 残高 ~15,600 で 1 注文ようやく出せる水準 → **資金制約が構造的ボトルネック**

#### 問題 3: `no_feasible_quote` — 9.3% でオフセットパイプラインが解を出せない

- buy 68 / sell 1 と buy 偏重
- offset パイプラインの多段ブーストが累積し、best_bid/ask を超える不可能な価格に到達
- 「防御を重ねるほど発注不可能になる」という 470# で指摘されたセマンティック反転の残影

### §4.2 パラドックスの構造

```
[低 fill rate]
    ↓ 統計的検出力不足
[改善効果の判定不能]
    ↓ 安全側倒し
[抑制閾値を保守的に維持]
    ↓ 
[さらに低い fill rate]
    ↓ 悪循環
```

追加して：
```
[少ない約定] → [サンプル不足] → [SkipGate 精度低下] → [過剰 skip] → [さらに少ない約定]
```

---

## §5 改善提案（優先順位付き）

### P0: `ranging_low_vol_skip` の緩和（最大インパクト）

**現状**: vol_ratio < 0.65 で buy を全 skip
**問題**: Coincheck BTC/JPY の「通常状態」を排除している
**提案**: 
- A) 閾値を 0.65 → 0.40 に引き下げ（真に低 vol のみ skip）
- B) hard skip → offset boost に変更（468# soft mode の全面化）
- C) 完全無効化して SkipGate ML に統合
- **推奨**: B（offset boost 化）— SG が既に EV ベースで同等の判定を行える

**期待効果**: 217 件中 ~150 件が注文化 → fill rate +5-10% ポイント

### P1: `no_feasible_quote` の根本原因修正

**現状**: Buy 68 / Sell 1 と著しく偏向
**問題**: 多段 offset boost の累積で実現不可能な価格に到達
**提案**:
- パイプライン内で「best_bid/ask からの最大乖離」を設定（例: spread の 2倍まで）
- 現行の `final_clamp_hard_skip` は最後段 — 中間段での異常検知を追加
- offset boost の累積に上限をかける（product cap、例: 合計 offset_mult ≤ 3.0）

**期待効果**: 69 件中 ~50 件が注文化 → fill rate +3-5%

### P2: `spread_too_narrow` 閾値の見直し

**現状**: min_spread_jpy = 1000 JPY
**問題**: 1000 JPY は BTC 1170万 JPY の 0.85 bps — 他取引所の taker fee 以下の spread で maker 注文は利益が出にくいのは事実だが、常に skip する必要はない
**提案**: 
- 500 JPY に引き下げ（0.4 bps — maker fee=0 なので微益でも黒字）
- または SkipGate の EV スコアで判定に統合

**期待効果**: 70 件中 ~30 件が注文化 → fill rate +2%

### P3: Cooldown lot scale の影響緩和

**現状**: lot×0.3 が 129 回発動（ほぼ全サイクル）
**問題**: base lot=0.001 × 0.3 = 0.0003 < min_order_btc=0.001 → 事実上 skip と同等
- 476# で floor を min_order_btc に修正済みだが、lot=min_order_btc は dust 化リスク

### P4: 資金増強

**現状**: JPY ~15,600 / BTC ~0.00097546
**問題**: 1サイクル1注文がギリギリの資金量。片側約定すると反対側が残高不足で skip
**提案**: JPY 残高を最低 50,000 以上に増資（5注文分を常時確保）

---

## §6 感度分析との照合

G3 Monte Carlo (476#) の感度分析結果：

| fill_rate | pnl_adj | E[PnL] | 判定 |
|-----------|---------|--------|------|
| 50% | +0.0 bps | -3,048 JPY/mo | FAIL |
| 50% | **+1.0 bps** | **+8,618 JPY/mo** | **PASS** |
| 70% | +1.0 bps | +12,083 JPY/mo | PASS |

**要点**: fill rate を上げるだけでは黒字にならない。PnL mean を +1 bps 改善が必須。
逆に PnL mean +1 bps を達成しても fill rate 17% では月額利益は約 +3,000 JPY 程度にとどまる。

**両方の同時改善が必要** — P0-P2 で fill rate 30-40% + AS 削減で PnL +0.5-1.0 bps を狙う。

---

## §7 約定品質のポジティブ面

悲観だけではない。直近データのポジティブな兆候：

1. **中央値は黒字** (median PnL = +0.10 bps) — テール損失を除けば半分以上の取引はプラス
2. **AS rate 25.6%** は改善傾向 (474# P0 修正前は 28.1%)
3. **Queue wait 12.1s** は健全 (≤60s 基準に十分マージン)
4. **Buy AS rate 19.5%** は ≤20% 基準をほぼ達成
5. **Sell AS 32.1%** が要改善だが、offset_ceiling 0.20 への修正 (474#) はまだ効果測定期間中

---

## §8 結論

fill rate 17.7% の主因は **3つの「巨大な壁」**:
1. **ranging_low_vol_skip (29.1%)** — 通常の市場状態を排除する自己矛盾
2. **preflight_insufficient (19.1%)** — 資金制約（構造的、476# 一部修正済み）
3. **timeout + no_feasible_quote + spread_too_narrow (30.5%)** — 発注しても約定せず or 発注不可

これらの改善で fill rate 30-40% は現実的。PnL 改善（AS 削減、offset 最適化）と合わせて月次黒字達成を目指す。

---

## 付録: データソース

- fill_records: `results/v460/fill_test/fill_records_20260317.jsonl` + `fill_records_20260318.jsonl`
- ログ: `results/v460/fill_test/logs/fill_test.log` (期間: 2026-03-17 19:28 ～ 2026-03-18 21:14)
- G3 MC 結果: `results/v460/g3_pnl_mc_476.json`
- Bot: PID=68232, git_sha=0dd7bacaa (476# 3rd commit。本ログは 476# 適用前と適用後に跨る)
- Regime 分布: ranging 96%, trending_down 3.8%, trending_up 0.3%
