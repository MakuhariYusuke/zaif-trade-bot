# 245# 本番ログ分析: 2026-02-13 〜 2026-03-03 (sha: c141be4a2947 / 241#)

## 概要

fill test が `c141be4a2947` (241# Toxicity Budget Review) で稼働した
約18日間の本番データを分析し、構造的問題と改善方針を纏める。

- **期間**: 2026-02-13 03:37 〜 2026-03-03 16:27 (前回 run) + 16:30 〜 現在 (再起動後)
- **稼働 sha**: `c141be4a2947` (241# — 242#〜244# 未反映)
- **取引所**: coincheck BTC/JPY
- **総サイクル**: 5,726 / **約定**: 2,280 (FR=39.8%)
- **累積 PnL**: **-792 JPY** (avg -0.341bps/fill)
- **quarantine**: 379 records → clean 5,347

---

## 1. 現在のボット状態 (3月3日 18:19 JST)

| 項目 | 値 |
|------|-----|
| PID | 24852 (稼働中) |
| Run ID | `1772523008_b7f1ace2` |
| DD Halt | **継続中** (10:51 JST 発動, PnL=-50.00bps) |
| 残高 | JPY 1,100.61 + BTC 0.002 |
| Buy 可否 | **不可** (最低 ~10,900 JPY 必要) |
| Sell dynamic kill | rolling50 mean=-1.840bps, total_kills=18 |
| Toxic veto | sell=1, buy=2 |
| 次回リセット | 3月4日 09:00 JST (UTC day boundary) |

---

## 2. 3月3日タイムライン

```
09:02  Day reset (前日3/2: -14.49bps, halt無し)
09:53  sell kill cooldown=8, JPY不足→balance_forced_rescue
       → skip_gate SKIP (score=-4.266)
09:57  Cycle 5697 sell: filled @ wait=5.9s
09:59  ★ PnL = -22.54bps (AS=True)
       → per-side DD halt (sell -31.46bps)
       → soft DD (-34.80bps), toxic veto sell
       → fast_fill_defense L1 activated
10:09  Cycle 5698 buy: spread too narrow (471 < 1000)
10:15  Cycle 5699 buy: spread too narrow (921 < 1000)
10:21  Cycle 5700 buy: +0.24bps (wait=6.0s)
       → fast_fill_defense L1 activated (buy)
       cumPnL=-775.5 JPY
10:29  skip_gate hot-reload (retrained model)
10:35  JPY不足 (1099 < 10968) → 強制sell → skip_gate SKIP
10:41  Cycle 5703 sell: -7.75bps (AS=True, wait=54.7s)
       → toxic veto sell
10:50  sell dynamic kill (rolling50=-1.840bps, 18th kill)
10:50  Cycle 5704 buy: -7.70bps (AS=True, wait=5.8s)
       → fast_fill_defense L2 activated
10:51  ★★ DD HALT: daily PnL -50.00bps (hard=-50.0bps, 14 fills)
       buy_pnl=-10.79bps, sell_pnl=-39.21bps
       → 以降16:27まで約5.5h停止
16:27  前回run完了 (24h timer), results_analyzer実行
16:30  再起動 (DD state復元 → halt継続)
18:10  DD halt cycle #10 (依然停止中)
```

---

## 3. 日別パフォーマンス

| 日付 | Records | Filled | Avg PnL (bps) | AS率 | DD Halt | 備考 |
|------|---------|--------|---------------|------|---------|------|
| 2/13 | 211 | 163 | -0.441 | 48.5% | - | 初日、高AS |
| 2/14 | 220 | 161 | -0.724 | 31.1% | - | |
| 2/15 | 60 | 49 | -0.875 | 34.7% | - | cycle少 |
| 2/16 | 21 | 14 | -1.123 | 35.7% | - | 極小activity |
| 2/17 | 205 | 137 | **+0.449** | 28.5% | - | ✅ 黒字 |
| 2/18 | 277 | 149 | **+0.353** | 18.8% | - | ✅ 黒字 |
| 2/19 | 250 | 176 | -0.552 | 29.5% | - | |
| 2/20 | 217 | 132 | -0.198 | 20.5% | - | 微損 |
| 2/21 | 377 | 164 | -0.603 | 14.0% | - | AS低いのに赤字 |
| 2/22 | 401 | 127 | -0.182 | 12.6% | - | AS最低 |
| 2/23 | 592 | 52 | -0.375 | 34.6% | - | FR極低9% |
| 2/24 | 481 | 157 | **+0.665** | 24.8% | - | ✅ 最良日 |
| 2/25 | 504 | 167 | -0.894 | 33.5% | - | |
| 2/26 | 472 | 174 | -1.044 | 33.3% | **YES** | -53.34bps |
| 2/27 | 488 | 204 | **+0.219** | 31.4% | No | ✅ 黒字 |
| 2/28 | 469 | 117 | -0.664 | 31.6% | **YES** | -51.21bps |
| 3/1 | 106 | 29 | **-3.826** | 37.9% | **YES** | ⚠ -110.94bps |
| 3/2 | 343 | 94 | -0.316 | 34.0% | No | |
| 3/3 | 34 | 14 | **-3.572** | 42.9% | **YES** | -50.00bps |

**集計**:
- 黒字日: 4日 (2/17, 2/18, 2/24, 2/27)
- 赤字日: 15日
- DD Halt日: 4日 (2/26, 2/28, 3/1, 3/3) — **直近6日で4回**
- 累積: -792 JPY / 2,280 fills

---

## 4. Regime別パフォーマンス

| Regime | Cycles | FR | PnL (bps) | AS |
|--------|--------|----|-----------|------|
| ranging | 2,952 | 48.4% | -0.390 | 0.257 |
| trending | 521 | 45.3% | -0.043 | 0.280 |
| trending_down | 266 | 46.6% | **+0.385** | 0.298 |
| trending_up | 476 | 27.7% | **-0.919** | 0.326 |
| unknown | 1,132 | 23.9% | -0.390 | 0.351 |

- **trending_down のみ黒字** — ほぼ全レジームで赤字
- **trending_up が最悪**: FR最低 + AS最高 + PnL最悪

---

## 5. Side別分析

### 全期間 (results_analyzer)
- **all_run**: n=5347, filled=2191, FR=0.410, PnL=-0.341bps, AS=0.278
- **current_run**: n=193, filled=69, FR=0.357, PnL=-0.580bps

### 3月3日 (14 fills)
- **Buy**: n=7, avg=-1.542bps
- **Sell**: n=7, avg=**-5.602bps**

### 3月1日 (29 fills, -110.94bps day)
- **Buy**: n=15, avg=-2.442bps
- **Sell**: n=14, avg=**-5.308bps**
- 壊滅的fill: sell -23.32bps, buy -19.74bps, sell -17.27bps

### Sell Dynamic Kill
- total_kills = 18 (cooldown = 10 cycles / kill)
- rolling50 mean = -1.840bps (threshold = -0.5bps)
- **kill が頻繁に発動するが、解除 → 即再損 → 再kill の繰り返し**

---

## 6. Retrain Monitor 判定

```
[online_monitor] 141# P1-12: n=100 (pass=64, skip=36),
  pass_mean_pnl=-0.827bps,
  pass_win_rate=42.2%,
  skip_precision=100.0%
  [DEGRADED: pass_mean_pnl=-0.827bps < threshold=-0.3bps]

  buy:  n=46 (pass=32, skip=14, skip_rate=30.4%), pass_pnl=-0.338bps, win_rate=50.0%
  sell: n=54 (pass=32, skip=22, skip_rate=40.7%), pass_pnl=-1.316bps, win_rate=34.4%
```

- sell モデルが DEGRADED（pass_pnl = -1.316bps, win_rate = 34.4%）
- buy は比較的健全（pass_pnl = -0.338bps, win_rate = 50%）
- skip_precision = 100% — skip した取引は全て正しく回避できている

---

## 7. Event Contribution Analysis

| Event | Active PnL | Inactive PnL | Δ |
|-------|-----------|-------------|------|
| FFD | -0.755 (n=182) | -0.404 (n=1179) | -0.352bps ⚠ |
| VG | -0.352 (n=505) | -0.509 (n=856) | **+0.157bps** ✅ |
| SG | -0.058 (n=280) | +0.032 (n=280) | -0.089bps |

- **FFD**: 発動群の方が PnL 悪化 — FFD 自体が逆効果の可能性あり
  - FFD は fast fill (wait < 10s) 後の offset 拡大だが、offset 拡大 → FR低下 → 機会コスト
  - あるいは FFD 発動自体が adverse 環境の指標 (交絡)
- **VG (VolatilityGuard)**: 効果的 — VG 発動で +0.157bps 改善
- **SG (SkipGate)**: 略中立

---

## 8. 構造的問題の特定

### CRITICAL-1: JPY 残高不足 → 死のスパイラル
- **現象**: JPY 1,100 で buy 不可能 (最低 ~10,900 JPY)
- **因果**: buy不可 → 強制sell → 不利sell → JPY増加するもBTC減少 → sellも不可に
- **影響**: balance_forced_rescue が連発、one_sided_consecutive_limit (5) に頻繁到達
- **市場理論**: Inventory risk (Avellaneda & Stoikov 2008) — 在庫偏重は
  adverse selection コストを指数関数的に増大させる。両サイド取引による
  在庫中立維持がマーケットメイキングの基本命題。

### CRITICAL-2: Sell 側の慢性的出血
- sell pass_pnl = -1.316bps (DEGRADED)、win_rate = 34.4%
- sell dynamic kill が 18回発動 — kill→解除→即再損→再kill のループ
- **根本原因**: sell 側の skill edge が消失、もしくはモデルの sell 予測精度が劣化
- **市場理論**: Glosten-Milgrom (1985) の情報非対称性モデルでは、sell 側の
  systematic loss は informed trader の sell-side activity を示唆。
  スプレッドの非対称拡大 (sell offset > buy offset) で対応すべき。

### HIGH-1: DD Halt 頻発 → 機会損失
- 直近6日で4回 DD Halt — 1日の大半が idle
- **3/1 の -110.94bps**: 29 fills で daily PnL が hard limit (-50bps) の2倍超
  → DD halt は日次リセットだが、fill 毎の PnL 加算型なので
    短期間に大損 fill が集中すると止めきれない
- **市場理論**: optimal stopping theory — halt が早すぎる場合は mean reversion opportunity を逃し、
  遅すぎる場合は tail risk に晒される。現在の -50bps hard limit は
  lot=0.001 BTC でも effective stop loss = ~5 JPY/fill × 10 fills 程度で発動し、
  充分 conservative だが、**halt 後の全日 idle が真の損失源**。

### HIGH-2: trending_up でのSell処理
- trending_up: PnL = -0.919bps, AS = 0.326 (最悪レジーム)
- skip_sell_trending_up_only = true だが soft mode (offset boost ×2.0)
- **3/1 は trending_up が多く、sell の AS 被害が甚大**
- **市場理論**: trending market での逆方向 MM は Kyle (1985) の
  informed trading model で禁忌。トレンドと逆方向のポジションは
  情報優位者の餌食になる。

### HIGH-3: FFD (Fast Fill Defense) が逆効果の可能性
- FFD active 群: -0.755bps vs inactive: -0.404bps (Δ=-0.352bps)
- FFD は offset 拡大でスプレッド稼ぎを狙うが、
  FR 低下による機会コストの方が大きい可能性
- **ただし交絡**: FFD 発動 = adverse 環境の指標であり、FFD が無ければ
  もっと悪化していた（counterfactual は不明）

### MEDIUM-1: unknown レジームの長期化
- unknown = 1,132 cycles (19.8%) — FR = 23.9% と最低
- regime_detector が不安定期間で "unknown" に留まり過ぎ
- 242# で `toxic_kill_stale_multiplier` を導入済み（本番未反映）

---

## 9. 既存実装の活用診断

### 活用されているもの（有効）
| 機能 | 状態 | 評価 |
|------|------|------|
| VG (VolatilityGuard) | Active | ✅ +0.157bps 効果 |
| toxic_fill_veto | Active | ✅ 正常動作 (3cycle封鎖) |
| per-side DD halt | Active | ✅ sell封鎖が機能 |
| loss_cooldown | Active | ✅ 定常的に発動 |
| skip_gate (buy model) | Active | ○ pass_pnl=-0.338, win50% |

### 活用されているが効果不十分 / 逆効果疑い
| 機能 | 状態 | 評価 |
|------|------|------|
| skip_gate (sell model) | Active | ⚠ DEGRADED, pass_pnl=-1.316 |
| sell_dynamic_kill | Active | ⚠ kill→解除→再損ループ |
| FFD | Active | ⚠ 逆効果の可能性 (Δ=-0.352bps) |
| balance_forced_rescue | Active | ⚠ sell-only で損失スパイラル |
| trending_sell_offset_boost | Active | △ ×2.0 では不足か |

### 未活用だが有望
| 機能 | 状態 | 改善案 |
|------|------|--------|
| 242# toxic_kill_stale_multiplier | **未デプロイ** | > resume_window 延長で再損防止 |
| 242# quiescence escalation | **未デプロイ** | > DD halt中のsleep段階的延長 |
| 244# guard_reason_classification | **未デプロイ** | > ガード理由の可視化 |
| 243# YAML wiring fix | **未デプロイ** | > 242# の設定値が実際に効かない bug fix |
| degraded_liquidation | Active だが | > balance_forced + kill blocked 時のみ — もっと積極活用 |
| sell_guard offset_floor=0.20 | Active | > 0.20 → 0.30 に引き上げ検討 |
| sell_guard_inv_bypass_threshold | Active | > ×.3 → 引き下げで sell 抑制強化 |

---

## 10. 市場理論に基づく改善提案

### 10.1 情報非対称性への対応 (Glosten-Milgrom)
- sell 側の systematic loss → informed trading の sell-side 偏重
- **対策**: sell offset floor を 0.20 → 0.30 に引き上げ (逆選択プレミアムの増加)
- 加えて、sell dynamic kill の threshold を -0.5 → -0.3 に厳格化

### 10.2 在庫リスク管理 (Avellaneda-Stoikov)
- JPY 枯渇 → sell-only は在庫中立性の崩壊
- **対策**: 残高入金が最優先。システム側では lot=0.001 (最小) で
  両サイド取引を維持することが理論的に最適
- balance_forced_rescue offset_mult 1.3 → 1.5 に引き上げ

### 10.3 Optimal Stopping / Regret Theory
- DD halt 後の全日 idle → 機会損失
- **対策**: DD halt の部分解除メカニズム（例: halt後 2h 経過で half-lot 再開）
- 現状の 242# liveness relaxation (quiescence + stale_multiplier) は
  この方向の施策だが、DD halt 自体の緩和は別途必要

### 10.4 テール・リスク管理
- 3/1 の -110.94bps、3/3 の -22.54bps — fat tail の実現
- **対策**: per-fill PnL cap の導入 — 1 fill で -15bps 超過した場合、
  そのサイドを N 時間ではなく日次リセットまで封鎖
- toxic_fill_veto_threshold を -5.0 → -3.0 に厳格化

### 10.5 Kyle (1985) — トレンドフォロー原則
- trending_up での sell は情報優位者への逆張り
- **対策**: trending_up 時の sell を hard skip (offset boost ではなく完全封鎖)
  に戻す検討。または offset_boost を 2.0 → 4.0 に大幅引き上げ

---

## 11. 次ステップ

| # | 優先度 | 施策 | 種類 |
|---|--------|------|------|
| 1 | **P0** | JPY 残高入金 (最低 10,000 JPY) | 運用 |
| 2 | **P0** | 242#-244# を production にデプロイ | 運用 |
| 3 | **P1** | sell offset floor 0.20→0.30 引き上げ | 設定変更 |
| 4 | **P1** | sell dynamic kill threshold -0.5→-0.3 厳格化 | 設定変更 |
| 5 | **P1** | trending_up sell の hard skip 復帰 or offset ×4.0 | 実装/設定 |
| 6 | **P1** | toxic_fill_veto_threshold -5.0→-3.0 厳格化 | 設定変更 |
| 7 | **P2** | DD halt 部分解除メカニズム (2h 後 half-lot) | 実装 |
| 8 | **P2** | FFD 効果の詳細な因果分析 (counterfactual) | 分析 |
| 9 | **P2** | sell モデルの緊急再訓練 + 特徴量追加 | ML |

---

## 12. Codex / Gemini レビューへの指示

**レビュアーへ**: 234# 〜 245# のドキュメントと実装を確認し、以下を評価してください。

1. **sell 側損失の根本原因**: skip_gate sell モデルの精度劣化か、market microstructure の変化か
2. **FFD の因果効果**: 交絡を排除した上で FFD の真の効果を推定可能か
3. **DD halt の最適設計**: 現在の -50bps hard limit + 日次リセットは妥当か
4. **242#-244# の本番適用リスク**: 未デプロイの 3 commit が本番環境で問題を起こす可能性
5. **市場構造の変化に対するロバスト性**: モデルの stale 化を検出・自動対応する仕組みの評価
6. **残高不足時の最適戦略**: one-sided trading の理論的限界と現実的対応策

---

*Generated: 2026-03-03 by automated log analysis*
*Running SHA: c141be4a2947 (241#)*
*Analysis period: 2026-02-13 〜 2026-03-03*
