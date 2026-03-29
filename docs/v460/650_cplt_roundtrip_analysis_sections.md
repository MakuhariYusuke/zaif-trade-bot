# 650# Roundtrip Analysis & Financial Engineering Sections

## 概要
`analyze_fill_logs.py` に金融工学的知見に基づく4つの新分析セクションを追加。
個別 fill の 30s post-fill PnL ではなく、実際の買→売 / 売→買 ペアの
実現損益 (Gross Roundtrip PnL) を計算する。
Market Making の収益は roundtrip 単位で評価すべき (Avellaneda & Stoikov 2008)。

## 追加セクション

### 1. `section_roundtrip` — Roundtrip PnL (Avellaneda-Stoikov 2008)
- 連続する異サイド約定をペアリングし Gross Roundtrip PnL を算出
- WR、PF、avg win/loss、hold time、regime/entry-side/worst 分解
- 未ペアリング fill 数も表示 (奇数 fill / 同一サイド連続の検出)

### 2. `section_inventory_health` — 在庫非対称性 (Kyle 1985)
- `preflight_insufficient` の集中度から buy/sell 非対称性を定量化
- `balance_jpy/btc_at_order` から 50/50 乖離度を算出 (LOW/MEDIUM/HIGH)
- 連続 preflight_insufficient の max run length (deadlock indicator)
- 時間帯別の集中度

### 3. `section_mcb_impact` — MCB HALT 影響 (Foucault et al. 2007)
- MCB halt の時間帯分布
- MCB 前後の fill PnL 比較 (pre-MCB / during-post-MCB / outside window)
- MCB regime 分布

### 4. `section_spread_fill_quality` — Spread vs PnL (Glosten-Milgrom 1985)
- spread_bps を quartile 分割し各帯の avg PnL / AS率 / avg wait を比較
- 低スプレッド fill の逆選択リスクを定量化
- Side 別の spread-PnL 相関係数 (paired data で正しく計算)

## バグ修正 (本セッション)
- `section_spread_fill_quality` の Side Breakdown 相関計算:
  `s_spreads` と `s_pnls` の配列長不一致バグを修正。
  `post_fill_30s_pnl` が None のレコードを除外する際に spread 側と
  対応がずれる問題。zip ペアリング方式に変更。
- `section_roundtrip`: `spread_bps` が None の場合に `float(None)` で
  クラッシュする可能性。`float(f.get("spread_bps") or 0)` に修正。
- `section_roundtrip`: 未ペアリング fill 数を表示するよう追加。

---

## 2026-03-29 分析結果 (SHA 5832c87fe, 09:04 再起動後)

### 全体サマリ
| 項目 | 値 |
|---|---|
| 全レコード | 341 (サイクル数 ~335) |
| Fill 数 | 27 (Fill 率 7.9%) |
| Roundtrip | 13 RT, 6W / 7L, WR 46.2% |
| Total RT PnL | -0.81 bps (Avg -0.06 bps/RT) |
| PF (Profit Factor) | 0.98 |
| Avg Win / Loss | +6.20 / -5.43 bps |
| 未ペアリング fill | 1/27 (F26: 23:06 buy@10662281, 最終fill) |
| 取引時間帯 | 10:16–23:06 JST (13時間) |

### 価格推移
```
10:16 mid=10,677,505 (ranging)
  ↓ +35,000 JPY上昇
13:13 mid=10,731,279 (ranging, 元trending_upからの転換直後)
  ↓ -35,000 JPY下降
14:36 mid=10,684,226 (ranging)
  ↓ +30,000 JPY上昇
15:19 mid=10,720,726 (trending_up)
  ↓ -57,000 JPY下降
19:43 mid=10,664,189 (trending_down)
  ↓ +25,000 JPY回復
22:41 mid=10,691,477 (trending_down)
  ↓ -28,000 JPY下降
23:06 mid=10,663,358 (ranging)
```
セッション中の値幅: ~68,000 JPY (≈64bps)。大きな振幅の中でtrendingレジームでの
sell entry に苦戦。

### 全13ラウンドトリップ詳細

| RT# | Entry | Entry Time | Exit Time | PnL(bps) | Hold(min) | Regime In→Out | Spread In/Out | MCB間 | PI間 | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | sell | 10:16 | 10:33 | **+4.55** | 17.6 | ranging→ranging | 3.0/2.8 | 0 | 3 | WIN |
| 1 | sell | 11:23 | 11:28 | **-2.45** | 5.0 | ranging→trending_up | 3.0/3.6 | 0 | 0 | LOSS |
| 2 | sell | 12:04 | 13:13 | **-14.27** | 69.7 | trending_up→ranging | 4.4/2.4 | 4 | 8 | LOSS (最大) |
| 3 | sell | 13:22 | 13:26 | **+8.36** | 3.6 | ranging→ranging | 2.6/3.4 | 0 | 0 | WIN |
| 4 | sell | 13:37 | 13:40 | **-0.78** | 3.6 | ranging→ranging | 2.7/2.4 | 0 | 0 | LOSS |
| 5 | sell | 13:48 | 14:00 | **+3.44** | 12.6 | ranging→ranging | 0.8/2.3 | 0 | 0 | WIN |
| 6 | sell | 14:14 | 14:23 | **+2.05** | 9.4 | ranging→ranging | 2.7/2.7 | 0 | 0 | WIN |
| 7 | sell | 14:36 | 14:46 | **-9.14** | 9.4 | ranging→ranging | 1.6/1.1 | 0 | 0 | LOSS |
| 8 | sell | 14:57 | 15:00 | **-3.59** | 3.0 | ranging→ranging | 2.4/3.1 | 0 | 0 | LOSS |
| 9 | sell | 15:19 | 15:52 | **+14.40** | 32.9 | trending_up→ranging | 3.4/3.1 | 1 | 3 | WIN (最大) |
| 10 | sell | 19:43 | 19:55 | **-3.64** | 11.9 | trending_down→ranging | 2.9/1.9 | 0 | 1 | LOSS |
| 11 | buy | 22:27 | 22:30 | **+4.38** | 3.4 | trending_down→trending_down | 1.9/2.7 | 0 | 0 | WIN |
| 12 | buy | 22:41 | 22:46 | **-4.13** | 4.9 | trending_down→trending_down | 3.3/2.5 | 0 | 0 | LOSS |

### RT個別因果分析

#### RT#2 — 最大損失 -14.27bps (-15,299 JPY)
- **Entry**: sell@10,715,579 (12:04), trending_up, sidecar=stale, offset clamped 0.60→0.40
- **Exit**: buy@10,730,878 (13:13), requote 3/4, wait=71.9s
- **Between**: 21 cancels (mcb_halt×4, preflight_insufficient×8, no_feasible_quote×6, spread_too_narrow×2)
- **根本原因 3重複合**:
  1. **trending_up で sell entry** — trending stage (×1.5) は offset を 0.60 まで引き上げたが ceiling 0.40 でクランプ。実質的に trending guard が機能していない。offset 情報の 33% が clamping で損失。
  2. **MCB HALT 連鎖** — 12:17, 12:27, 12:41, 12:53 の4回 HALT (各 cooldown=300s)。合計~50min の取引停止。Foucault et al. (2007) が理論化した「halt induced inventory risk」の典型例。
  3. **preflight_insufficient** — buy 注文を出したくても JPY 残高不足で8回ブロック。在庫リバランスが構造的に不可能。
- **教訓**: MCB HALT は正しく高ボラ環境を検知したが、open position の存在を考慮していない。HALT 突入前の緊急ポジション縮小、または HALT 中の例外的ヘッジ注文が必要。

#### RT#7 — 第2位損失 -9.14bps (-9,767 JPY)
- **Entry**: sell@10,684,401 (14:36), ranging, spread=1.63bps, wait=5.8s
- **Exit**: buy@10,694,168 (14:46), requote 1/4, wait=38.5s
- **Key indicators at entry**: ob_imbalance=+0.285 (買い圧優位), vpin=0.702 (高毒性), **AS=True**
- **根本原因**:
  1. **低スプレッド (1.63bps)** — Glosten-Milgrom の逆選択理論通り、タイトスプレッドは informed trader の存在を示唆。Q1 (<2.4bps) の avg PnL は -0.86bps。
  2. **OB imbalance +0.285** — 板の買い圧力が売り圧力を大幅に上回る = 価格上昇圧力。この環境で sell は不利。
  3. **VPIN 0.702** — 高い toxicity indicator。50% 超は「informed flow の存在が疑われる」水準。
  4. **即時約定 (5.8s)** — 市場参加者がこの sell を即座に取った = 自分の提示価格が安すぎることの傍証。fast_fill_defense multiplier=2.14 が発動したが exit 側のコスト増加にしか寄与しない。
- **教訓**: OB imbalance > 0.25 かつ VPIN > 0.65 の環境では sell offset を追加加算すべき。現在の VG (vg_boost=1.00) は VPIN が高くても sell に対して追加防御を提供していない。

#### RT#10 — -3.64bps (19:43~19:55)
- **Entry**: sell@10,664,496 (19:43), trending_down, ob_imb=-0.391 (売り圧), vpin=0.739
- **Exit**: buy@10,668,375 (19:55), wait=27.6s, ranging に遷移
- **Between**: 6 cancels (pi=1, narrow=2)
- **問題**: trending_down で sell entry は方向的にはアラインしているが、レジーム遷移中に買い戻しコストが上昇。exit spread=1.89bps と狭く、逆選択に晒された。

#### RT#9 — 最大利益 +14.40bps (+15,430 JPY)
- **Entry**: sell@10,719,315 (15:19), trending_up, offset clamped 0.60→0.40
- **Exit**: buy@10,703,885 (15:52), wait=60.1s, requote 2/4, vg_boost=2.12
- **成功要因**: trending_up で sell した後、32.9min のホールド中に ~16,000 JPY の価格下落が発生。clamping にも関わらず利益。MCB HALT 1回を挟んだが影響は限定的。
- **注意**: このWINはRTの構造的戦略ではなく「たまたま方向転換」による利益。同条件のRT#2は-14.27bps。trending_up sell は high variance / low expectation。

#### RT#0 — +4.55bps (10:16~10:33, 唯一のfresh sidecar)
- **Entry**: sell@10,677,923, sidecar=**fresh**, sc_bias=0.923, vg_boost=2.00 (velocity), spread=3.03
- **注目**: セッションで唯一 sidecar=fresh で取引されたRT。sidecar bias 0.923 は sell に favor。VG velocity boost=2.00 が高い offset を提供し、3.03bps の比較的広いスプレッドで約定。
- **示唆**: sidecar fresh 時の取引品質は stale 時と比べ有意に良い可能性。ただし n=1。

### キャンセル理由分析 (314 cancels)

| Cancel Reason | Count | % | 主原因 |
|---|---|---|---|
| preflight_insufficient | 145 | 46.2% | JPY=216円、buy 注文の balance_margin 充足不可 |
| no_feasible_quote | 67 | 21.3% | spread_too_narrow 3回連続 → no_feasible_quote 昇格 |
| spread_too_narrow | 49 | 15.6% | bid-ask < ATR/σ 最低幅 (例: 1731 < 3197 JPY) |
| skip_gate | 18 | 5.7% | ML model が負 PnL 予測 |
| timeout | 17 | 5.4% | 注文発注済み → micro_timeout 内に未約定 |
| mcb_halt | 9 | 2.9% | σ > 2.0 (5m/15m/1h の2窓以上) |
| status_unknown_fast | 1 | 0.3% | API応答異常 |

#### preflight_insufficient 構造分析
- **JPY残高=216円、BTC残高=0.00203 BTC ≈ 21,657 JPY** → JPY比率 **1.0%**
- JPY比率 1% は Inventory imbalance **49.0pp (HIGH)**
- buy 注文の balance_margin_ratio チェックで `jpy_free < lot × price × margin` が恒常的に不成立
- **全145件が buy-side blocked** — sell-side blocked は 0件
- これにより **売りしかできない一方通行取引** が構造化。RT#2 の 69.7min hold はこの直接的帰結。

#### MCB HALT タイミング
12:17, 12:27, 12:41, 12:53 (4回), 15:19前後 (残り)
- 12時台の4連続 HALT が RT#2 の 69.7min 閉塞を直接引き起こした
- MCB regime 分布: trending_up × ranging 混在

### 構造的問題の深堀り

#### 問題1: inv_skew_factor が実質無効 (25/27 fills = 0.0000)

**現在の設定**:
```yaml
inventory_skewing:
  enabled: true
  neutral_band: 0.1
  decay_tau_sec: 1800.0
  regime_gate_enabled: true
```

**無効化の3経路**:
1. `neutral_band=0.1`: fill 間隔が 3-30min の場合、`exp(-elapsed/1800)` の減衰で `|decayed_imbalance|` が 0.1 以内に収束 → 即座にスキップ
2. `decay_tau_sec=1800`: 30分で e^-1 ≈ 0.37 まで減衰。fill 密度が低い(27fill/13h) と imbalance 情報が急速に蒸発
3. `regime_gate_enabled=true`: trending 中は全面無効化

**結果**: 在庫が JPY:BTC = 1:99 に偏重しているにも関わらず、inventory skewing は offset を全く調整していない。これは Avellaneda-Stoikov (2008) のインベントリリスク管理の根幹を否定する状態。

**非ゼロだった2件の分析**:
- F01 (10:33 buy): inv_skew=-0.2654 — 直前 F00 (10:16 sell) から 17min、decay=exp(-1020/1800)=0.568 → imbalance が neutral_band を超過
- F03 (11:28 buy): inv_skew=-0.1185 — 直前 F02 (11:23 sell) から 5min、decay=exp(-300/1800)=0.846 → 超過ギリギリ

つまり **fill 間隔が < 20min かつ直前に反対 side の fill がないと inv_skew は発動しない**。

#### 問題2: Sidecar の 93% stale 状態

**現在のメカニズム**:
- retrain_scheduler が retrain 完了時に `cache/sidecar_signal.json` を書き込み
- TTL = 7800秒 (2h10m)、retrain 間隔 = 7200秒 (2h)
- retrain が遅延または失敗すると signal が stale 化

**3/29 の状況**:
- F00, F01 (10:16, 10:33) は sidecar=fresh (sc_bias=0.923)
- F02 以降 (11:23~23:06) は全て stale
- 649# で retrain_scheduler のデータ鮮度チェックを分離したが、retrain 自体が 1回も成功していない可能性

**影響**: sidecar offset が適用されない → 全 fill が `decision_path=primary_only` → ML の方向性アドバイスなしで純粋に rule-based 取引。sidecar fresh 時の唯一の RT (RT#0) は +4.55bps で最良ではないが安定。

#### 問題3: Sell offset が ceiling 0.40 に張り付き

**27 fills の sell offset 分布**:
- sell (13 fills): **12/13 が offset=0.4000** (ceiling固定)
  - 唯一の例外: F25 (22:46) offset=0.2948 (ev_offset_mult で縮小)
- buy (14 fills): 0.1608 ~ 0.3500 に分散

**clamping されていた3 fills**:
| Fill | Time | Pre-clamp | Effective | 損失 | Stage |
|---|---|---|---|---|---|
| F02 | 11:23 | 0.6219 | 0.4000 | 2219bps相当の情報損失 | velocity=1.554 |
| F04 | 12:04 | 0.6000 | 0.4000 | 2000bps相当 | trending=1.5 |
| F18 | 15:19 | 0.6000 | 0.4000 | 2000bps相当 | trending=1.5 |

pipeline が 0.60 の offset を推奨 → ceiling 0.40 で 33% 情報損失。
ただし F18 は clamped にもかかわらず +3.70bps (30s pnl) → RT#9 = +14.40bps。
**∴ clamping が必ずしも損失を意味するわけではないが、情報が捨てられている構造は問題。**

#### 問題4: VG triggered on 26/27 fills (96%)

ほぼ全ての fill で Volatility Guard が発火。VG reason の大半は `vpin` (VPIN > threshold)。
- VG boost > 1.00 (offset を広げた) のは 14/26 fills (54%)
- VG boost = 1.00 (発火したが boost なし) が 12/26 fills (46%)

**VG boost=1.00 の意味**: VG は発火条件を満たしたが、sell side に対して boost を適用していない。
これは sell の offset が既に ceiling (0.40) に達しているため、boost を乗せても ceiling でクランプされるから。
→ **VG と ceiling の相互作用により、sell 側の volatility 防御が形骸化している**。

### Spread vs Fill Quality (Glosten-Milgrom 分析)

| Quartile | Spread 範囲 | N | Avg PnL30 | AS率 | Avg Wait |
|---|---|---|---|---|---|
| Q1 | < 2.4bps | 7 | **-0.86bps** | 14% | 21.6s |
| Q2 | 2.4–2.7bps | 6 | **+1.66bps** | 0% | 21.5s |
| Q3 | 2.7–3.1bps | 7 | **+1.72bps** | 0% | 13.2s |
| Q4 | ≥ 3.1bps | 7 | **+0.17bps** | 14% | 35.1s |

- **Q1 (タイトスプレッド)** が唯一の負 PnL 帯。逆選択理論と完全一致。
- Q2-Q3 が sweet spot。2.4–3.1bps のスプレッドが最適収益帯。
- Q4 は wait time が長い (35s) = 市場が約定を渋る価格帯 → regression to mean で利益は薄い。
- **Side別 spread-PnL 相関**: buy=-0.106, sell=+0.188
  - buy 側は弱い負の相関 (広いスプレッド → やや悪い PnL) — counterintuitive だが n=14 で統計的有意性なし
  - sell 側は弱い正の相関 (広いスプレッド → やや良い PnL) — spread capture の恩恵

---

## 具体的改善案

### P0 (即座に実施可能・収益トレード保護)

#### I1: inv_skew neutral_band 引下げ + decay_tau 延長
- **変更**: `neutral_band: 0.1 → 0.03`, `decay_tau_sec: 1800 → 3600`
- **根拠**: 現状 25/27 fills で inv_skew=0.0。neutral_band が広すぎかつ減衰が速すぎて在庫管理が機能していない。BTC:JPY = 99:1 の極端な偏重でも skew 補正が働かない。
- **リスク**: ranging regime の sell 戦略 (WR 62%, +2.44bps) への影響。ただし inv_skew は sell offset を減らす方向に作用するため、sell の spread capture が若干低下する可能性。段階的に `neutral_band: 0.05` から試行。
- **期待効果**: buy-side の offset がインベントリ圧力で拡大 → fill 率向上 → 手仕舞い能力改善

#### I2: MCB HALT 時の open position 警告ログ
- **変更**: MCB HALT 判定時、open inventory (BTC ≠ 0) があれば WARNING ログに「HALT with open position: inventory_risk_jpy=XXX」を出力
- **根拠**: RT#2 の -14.27bps は MCB HALT × open position の複合。現状 MCB は position を考慮せずに HALT。
- **リスク**: なし (ログ追加のみ)
- **実装箇所**: `scripts/v460/lib/micro_circuit_breaker.py` の HALT 判定部

### P1 (要検討・影響範囲あり)

#### I3: low-spread sell ガード
- **変更**: spread_bps < 2.0 かつ regime != trending_down のとき、sell offset に追加 margin (例: +0.02)
- **根拠**: Q1 (< 2.4bps) avg_pnl = -0.86bps。RT#7 (spread=1.6bps, -9.14bps) が典型。低スプレッド環境は informed trading の可能性が高い。
- **リスク**: **RT#5 (entry spread=0.78bps, +3.44bps) をキルする可能性**。RT#5 は Q1 環境にも関わらず WIN。低スプレッド全体をブロックするのは危険。
- **代替案**: spread < 2.0 かつ ob_imbalance > 0.25 かつ vpin > 0.65 の「トリプル条件」でのみ発動。これなら RT#7 (obi=0.285, vpin=0.702) は捕捉し、RT#5 (obi=0.023, vpin=0.639) は通過。

#### I4: sell offset ceiling 0.40 → 0.45 (VG 形骸化対策)
- **変更**: `offset_ceiling_ratio_sell: 0.40 → 0.45`
- **根拠**: sell 12/13 fills が ceiling 固定。VG boost が ceiling で頭打ちになり volatility 防御が形骸化。trending stage (×1.5) も 0.60→0.40 で 33% 情報損失。
- **リスク**: offset 拡大 = fill 率低下。現在 sell fill 率は既に低い。ceiling 拡大で約定確率がさらに下がると取引機会が減少。
- **代替案**: 段階的に 0.42 → 0.45。VG boost > 1.5 のときのみ ceiling を 0.50 に動的拡張。

#### I5: sidecar TTL とretrain間隔の整合性確認
- **変更**: retrain_scheduler の実行頻度を確認し、TTL 7800秒内に retrain が完了する保証を追加。もしくは TTL を 10800秒 (3h) に延長。
- **根拠**: 93% stale は sidecar の意味がない。RT#0 (fresh, +4.55bps) vs 残り全て stale。
- **リスク**: TTL 延長は古い signal を使い続けるリスク。retrain 頻度向上の方が望ましい。
- **前提**: 649# で data freshness check が分離されたが、retrain 自体が成功しているか検証が必要。

### P2 (長期的改善)

#### I6: MCB HALT 前の position pre-close メカニズム
- HALT 判定が WARNING → HALT に昇格する際、open position があれば事前に成行的な close 注文を出す
- 市場理論的背景: Foucault et al. (2007) の halt-induced inventory risk。先進取引所の volatility auction に相当。
- 実装複雑度: HIGH (fill_cycle_executor との連携、partial fill 処理)

#### I7: 動的スプレッドゲート (toxicity-adjusted)
- spread_too_narrow (16%) と no_feasible_quote (21%) の合計 37% が spread guard によるキャンセル
- 現在の ATR/σ ベース最低幅を VPIN で動的調整: 高 VPIN 時は最低幅を引上げ (保守的)、低 VPIN 時は引下げ (積極的)
- Easley et al. (2012) の VPIN-based dynamic quoting と整合

## テスト
- 既存 21 + 新規 11 = 32 テスト全通過
- バグ修正後の再テストも全通過
