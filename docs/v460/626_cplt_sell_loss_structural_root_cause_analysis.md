# 626# Sell 損失の構造的根本原因分析

## 概要

3/22–3/25 の 4 日間 fill データに基づき、sell 側の構造的損失要因を特定。
**時間帯（欧州・NY・東京）という表層的説明に代えて、防御レイヤーの不発火を定量的に立証する。**

## 要約

| 要因 | 深刻度 | 損失寄与 | 修正可能性 |
|------|:------:|:--------:|:----------:|
| Velocity 閾値過大 (6.0bps) | **S** | 直接 | 容易 |
| Regime trending 閾値過大 (0.5%) | **S** | 直接 | 中 |
| Skip Gate 偽陰性 | **A** | 直接 | 要再学習 |
| Post-fill 上方バイアス | **A** | 間接 | 構造的 |
| VG supplement 閾値過大 (12.0bps) | **B** | 軽微 | 容易 |
| Offset pipeline 多段不発火 | — | 上記の帰結 | — |

深刻度: **S** = 即座に対処すべき, **A** = 次回リリースで対処, **B** = 改善推奨

---

## 1. 実績データ概要

### 4 日間 PnL_120s サマリ

| 日付 | Buy n | Buy PnL | Buy avg | Buy WR | Buy AS% | Sell n | Sell PnL | Sell avg | Sell WR | Sell AS% |
|------|------:|--------:|--------:|-------:|--------:|-------:|---------:|---------:|--------:|--------:|
| 3/22 | 51 | -26.2 | -0.51 | 37% | 57.6% | 30 | -58.2 | -1.94 | 40% | 48.3% |
| 3/23 | 38 | +6.9 | +0.18 | 53% | 52.2% | 25 | -13.1 | -0.52 | 48% | 37.3% |
| 3/24 | 27 | -29.7 | -1.10 | 48% | 46.0% | 25 | -108.4 | -4.33 | 40% | 55.3% |
| 3/25 | 19 | +35.9 | +1.89 | 58% | 40.0% | 16 | -61.5 | -3.85 | 44% | 53.6% |

**4 日間合計**: Buy = -13.1 JPY, Sell = -241.2 JPY → **Sell が損失の 95%**

### 3/25 Tail 損失の集中度

| 指標 | Buy | Sell |
|------|----:|-----:|
| Tail 20% 平均 PnL | -6.29 | **-24.24** |
| Worst 単発 | -8.1 | **-26.1** |
| Best 単発 | +17.4 | +35.8 |

Sell の tail 損失が buy の **3.9 倍** 重い。

---

## 2. 根本原因 #1 — AS 被害の非対称性

### 3/25 AS (Adverse Selection) サイド別

| | AS=Y 件数 | AS=Y 平均損失 | AS=N 件数 | AS=N 平均利益 |
|---|:---:|:---:|:---:|:---:|
| **Sell** | 11/16 (69%) | **-11.34 JPY** | 5/16 | +12.63 JPY |
| **Buy** | 7/19 (37%) | -1.79 JPY | 12/19 | +4.03 JPY |

**4 つの構造差:**

1. **AS 比率**: Sell 69% vs Buy 37% — sell は 2 回に 1 回以上が逆選択される
2. **AS 単価被害**: Sell -11.34 vs Buy -1.79 — **6.3 倍**
3. **AS=Y 累計**: Sell -124.69 vs Buy -12.50 — sell AS だけで全損失超過
4. **AS=N 収益性**: Sell AS=N は +12.63（健全）→ AS さえ防げれば sell は利益サイド

**結論**: 問題はスプレッドやオフセットの水準ではなく、**AS fill を事前に拒否できていないこと**。

---

## 3. 根本原因 #2 — Velocity 段の閾値過大

### 現行設定と実測値の乖離

| パラメータ | 閾値 | 実測値 (3/25 sell fill) |
|-----------|:----:|:----------------------:|
| `sell_velocity_skip_threshold_bps` | **6.0 bps** | avg +0.66, max +8.6 |
| `volatility_guard.velocity_threshold_bps` | **12.0 bps** | avg +0.66, max +8.6 |

### Fill 時 velocity と PnL の相関 (3/25 sell)

| Velocity 区間 | fill 数 | 平均 PnL_120s |
|:---:|:---:|:---:|
| vel > 0 (上昇中 = sell 危険) | 10 | **-5.14 JPY** |
| vel ≤ 0 (下降中 = sell 安全) | 6 | -1.68 JPY |

**問題の構造:**

1. **Velocity 段 (195#)** は `vel ≥ 6.0 bps` で発火するが、実際に損失を出す sell の velocity は **+0.5 〜 +4.5 bps（全件閾値未満）**
2. **VG supplement 段 (202#)** は `|vel| ≥ 12.0 bps` で発火 — 観測最大値 8.6 bps の 1.4 倍高く、**到達不可能**
3. 最悪 5 件の fill すべてで velocity stage = `None`（未発火）

### 定量的影響

Velocity 段が +2.0 bps 閾値で発火していた場合に防げた損失:

| 時刻 | vel (bps) | PnL_120s | 防御可否 |
|------|:---------:|---------:|:--------:|
| 13:54 | +2.0 | -25.3 | ☑ 防御 |
| 14:03 | +1.8 | -21.2 | △ 境界 |
| 15:16 | +2.7 | -14.4 | ☑ 防御 |
| 16:29 | +0.5 | -20.0 | ☒ 防御不可 |
| 12:27 | +0.7 | -26.1 | ☒ 防御不可 |

閾値を 2.0 bps に下げた場合の回避可能損失: **約 -61 JPY** (3 件、全売損の 50%)

**ただし偽陽性リスク**: vel > 0 でも +12.63 JPY 利益の AS=N fill 5 件が存在。閾値引き下げは skip 増加 ↔ 損失回避のトレードオフを伴う。

---

## 4. 根本原因 #3 — Regime 検出器のマイクロトレンド盲点

### 現行閾値と実測データ

| パラメータ | 値 | 意味 |
|-----------|:---:|------|
| `trend_threshold_pct` | **0.5%** (50 bps) | 20 観測窓での price return |
| `hysteresis_count` | 3 | 3 連続一致で状態遷移 |
| `min_confidence` | 0.2 | 信頼度ゲート |

### 3/25 sell 最悪 fill の 120s mid 移動量

| 時刻 | mid 移動 (bps/120s) | Regime 判定 | 移動 vs 閾値 |
|------|:-------------------:|:-----------:|:------------:|
| 12:27 | **+26.1** | ranging | 50 bps に**未到達** |
| 13:54 | **+25.3** | ranging | 50 bps に**未到達** |
| 14:03 | **+21.2** | ranging | 50 bps に**未到達** |
| 16:29 | **+20.0** | ranging | 50 bps に**未到達** |
| 11:35 | **+18.6** | ranging | 50 bps に**未到達** |

**問題の構造:**

1. Regime detector は 20 観測窓で price return が 0.5% (50 bps) を超えないと trending と分類しない
2. しかし 120 秒で 20–26 bps の一方向移動は、maker にとって +25 JPY 級の AS 損失を生む
3. 全 28 件の sell fill が `regime=ranging` — trending offset boost (1.8×) は**一度も発火していない**
4. ranging 判定のため `ranging_offset_discount_sell: 0.85` が適用され、offset が**むしろ縮小**される
5. trending_up + sell なら 1.8× boost が掛かるところ、0.85× discount が掛かっている — **実効倍率の差は 2.1 倍**

### 閾値の妥当性検証

Coincheck BTC/JPY の通常ボラティリティ:
- 日次 σ ≈ 1.5–3.0% → 120 秒あたり √(120/86400) × 1.5% ≈ 0.056% ≈ **5.6 bps**
- 20–26 bps/120s は σ の **3.6–4.6 倍** — これは統計的に trending と判定すべき水準

**0.5% (50 bps) は日次スケールの閾値が流用されている可能性**。120 秒スケールでは 10–15 bps が適切な trending 閾値。

---

## 5. 根本原因 #4 — Skip Gate 偽陰性

### 最悪 sell fill の Skip Gate スコア

| 時刻 | PnL_120s | SG score | SG 結果 | vel (bps) |
|------|:--------:|:--------:|:-------:|:---------:|
| 12:27 | -26.1 | -0.52 | ☒ 通過 | +0.7 |
| 13:54 | -25.3 | **+1.96** | ☑ **誤通過** | +2.0 |
| 14:03 | -21.2 | **+2.96** | ☑ **誤通過** | +1.8 |
| 15:16 | -14.4 | -1.39 | ☒ 通過 | +2.7 |
| 16:29 | -20.0 | -2.49 | ☒ 通過※ | +0.5 |

※ 16:29 は SG score -2.49 で閾値以下だが fill されている → 閾値 avg -0.33 より低いので通常はスキップされるはずだが、何らかの理由で通過

**致命的問題:** -25.3, -21.2 JPY の損失 fill が SG score **+1.96, +2.96** で「良好」と判定。Skip Gate LGBM モデルが「上昇トレンド中に売りのAS損失が出る」パターンを学習できていない。

### Skip Gate 閾値の非対称性

| | avg threshold |
|---|:---:|
| Buy | -0.03 |
| Sell | **-0.33** |

Sell 側の閾値が buy より 0.30 低い = **sell は甘い基準でfillを許可**している。

---

## 6. 根本原因 #5 — Post-fill 価格上方バイアス

### 3/25 fill 後の mid 移動方向

| | avg mid 移動 (bps) | 上昇件数 | 下降件数 | 上昇率 |
|---|:---:|:---:|:---:|:---:|
| Sell | **+3.85** | 9 | 7 | 56% |
| Buy | **+1.89** | 11 | 8 | 58% |

**両サイドとも fill 後に価格上昇** — BTC がマイクロ上昇基調にある市場構造的な問題。

- Buy fill 後に上昇 → buy は利益方向 → avg +1.89 JPY
- Sell fill 後に上昇 → sell は損失方向 → avg -3.85 JPY

**差分 5.74 bps** が sell 側の構造的ハンディキャップ。これは offset 非対称化でしか対処できない。

---

## 7. 因果連鎖図

```
BTCマイクロ上昇トレンド (post-fill mid +3.85bps up)
    ↓
Regime detector: 20obs窓で0.5%未達 → 「ranging」判定
    ↓
① trending offset boost 1.8× 未発火
② ranging_offset_discount_sell 0.85× が適用 (逆効果)
    ↓
Velocity: +0.5〜+4.5bps → 閾値 6.0bps 未達 → 未発火
VG supp: 閾値 12.0bps → 到達不可能 → 未発火
    ↓
9段offsetパイプラインの4段 (velocity, trending, toxicity, vg_supp) が
全て None → offset 防御的拡大なし
    ↓
Skip Gate: +1.96, +2.96 で「良好」判定 → fill 許可
    ↓
Sell fill 成立 → 120s後 mid +20〜26bps 上昇
    ↓
AS判定 = Y → 平均損失 -11.34 JPY/件
AS=Y 11件 × -11.34 = -124.69 JPY (全損失の根源)
```

**5 段の防御レイヤーが全て不発火**というのが構造的問題の本質。

---

## 8. Offset pipeline 段別 — 最悪 sell fill 5 件の発火状態

| 段 | 閾値 / 条件 | 12:27 | 13:54 | 14:03 | 16:29 | 11:35 |
|----|:---:|:---:|:---:|:---:|:---:|:---:|
| ev | ev_score ベース | 1.01 | 1.00 | 1.08 | 0.94 | 0.86 |
| velocity | vel ≥ 6.0bps | **None** | **None** | **None** | **None** | **None** |
| trending | regime=trending | **None** | **None** | **None** | **None** | **None** |
| toxicity | warn_level ≥ 0.3 | **None** | **None** | **None** | **None** | **None** |
| vg_supp | vel ≥ 12.0bps | **None** | **None** | **None** | **None** | **None** |
| alert | alert mode on | None | None | 1.5 | None | None |

**5 段中 4 段が全件 None**。EV 段のみ発火しているが multiplier は 0.86–1.08 の中立域 — 防御機能を果たしていない。

---

## 9. 環境コンテキスト (3/25 売り fill の VPIN・OBI)

| 時刻 | PnL_120s | VPIN | OBI | vel (bps) |
|------|:--------:|:----:|:---:|:---------:|
| 12:27 | -26.1 | 0.59 | -0.10 | +0.7 |
| 13:54 | -25.3 | 0.53 | -0.02 | +2.0 |
| 14:03 | -21.2 | 0.62 | -0.37 | +1.8 |
| 16:29 | -20.0 | 0.80 | -0.36 | +0.5 |
| 11:35 | -18.6 | 0.73 | -0.03 | -4.5 |

- VPIN: 0.53–0.80 → 情報取引の存在を示唆するが toxicity 段が拾えていない
- OBI: 負値 (売り板優位) → sell 側の AS リスクを高める側
- 11:35 は vel=-4.5 で sell 安全方向のはずが AS=Y → **velocity だけでは防御不十分**の証左

---

## 10. 推奨アクション（優先順）

### P0: Velocity 閾値の引き下げ

| 変更 | 現行 | 提案 |
|------|:----:|:----:|
| `sell_velocity_skip_threshold_bps` | 6.0 | **2.5–3.0** |
| `volatility_guard.velocity_threshold_bps` | 12.0 | **6.0** |

**根拠**: 実測の AS売 velocity 中央値 +0.74bps、損失域は +0.5–4.5bps。6.0bps 閾値では実データ域のほぼ全件を逃す。2.5bps に引き下げれば最悪 fill の 50% を回避可能（セクション 3 参照）。

**リスク**: vel 2.5–6.0bps の sell skip 増加による機会損失。AS=N かつ vel>0 の sell 5 件 (avg +12.63) を一部失う可能性 → velocity_skip_as_offset_enabled (soft mode) で offset boost に変換すれば skip ではなくスプレッド拡大で対処可能。

### P1: Regime trending 閾値の時間スケール適正化

| 変更 | 現行 | 提案 |
|------|:----:|:----:|
| `trend_threshold_pct` | 0.5% (50bps) | **0.15–0.20%** (15–20bps) |

**根拠**: 120s で 20–26bps の一方向移動を ranging と分類する現行閾値は、日次スケールの閾値がそのまま流用されている。120s σ ≈ 5.6bps に対し、20bps = 3.6σ は統計的に有意な trending。trending 判定時は `trending_up_sell_offset_boost: 1.8` が有効化され、現行の `ranging_offset_discount_sell: 0.85` との差分 2.1× の防御が得られる。

**リスク**: false positive trending 増加 → `hysteresis_count: 3` が安全弁として機能。trending skip が増えすぎる場合は soft mode (`trending_sell_as_offset_enabled: true`) で既に offset boost に変換される。

### P2: Skip Gate モデル再学習

**課題**: SG score +1.96, +2.96 が -25 JPY の損失 fill を「良好」と判定。LGBM の特徴量に方向性 AS リスク（velocity × side のインタラクション項）が不足している可能性。

**アクション案:**
1. 学習用 fill_records に 3/22–3/25 データを追加
2. 特徴量追加検討: `velocity_bps × is_sell` のインタラクション項
3. sell 側 threshold を buy 同等 (-0.03) に引き上げ（暫定）

### P3: Post-fill 上方バイアスへの非対称 offset

**課題**: BTCが上昇基調である限り sell は構造的に不利。velocity/regime 段の改善で大部分は吸収されるが、本質的には AS 費用の方向性要素。

**アクション案**: P0+P1 の効果を確認した後で検討。vel ≤ 0 でも sell avg PnL = -1.68 であり、ゼロ以下 → offset 基本水準の sell 側引き上げも要検討。

---

## 11. 駄目出し・自己批判

### 本分析の限界

1. **サンプルサイズ**: 4 日間 sell fill 96 件。統計的有意性の観点からは n ≥ 200 が望ましい
2. **Velocity → PnL 因果の不確実性**: velocity +0.66bps は「上昇局面で fill された」結果であり、velocity 自体がASの原因かは確定できない。velocity は結果指標であり先行指標ではない可能性
3. **Regime 閾値 0.15% の根拠**: 120s σ ≈ 5.6bps からの逆算で、Coincheck の tick 構造・板厚を反映していない。実測 σ は Parkinson/Roll 推定値であり、理論 σ とは異なる
4. **Skip Gate 偽陰性**: +1.96, +2.96 が「誤り」と断定しているが、SG は 30 秒 PnL を予測しており、120 秒 PnL とは時間軸が異なる可能性がある
5. **AS=N vel>0 fill の機会損失**: 閾値引き下げで失う +12.63 JPY/件 × 5 件 = +63 JPY は、防御で回避する -61 JPY とほぼ等価 → **損益分岐を精密に計算する必要がある**
6. **11:35 の vel=-4.5bps AS=Y**: 売りに安全な方向の velocity でも AS が発生 → velocity 以外の情報（VPIN=0.73）が重要なケースの存在を示す。velocity 閾値のみの対処は万能ではない

### 検証が必要な仮定

- [ ] Regime detector の 20 obs 窓は何秒相当か？（cycle 間隔 × 20）
- [ ] `ranging_offset_discount_sell: 0.85` が実際に適用されているか？（コードパス確認）
- [ ] Skip Gate の予測対象は PnL_30s か PnL_120s か？（モデル仕様確認）
- [ ] toxicity 段が None な理由 — VPIN 0.53–0.80 で warn_level 0.3 を超えているはずだが、toxicity assessment が独立のメトリクスの場合は VPIN と無関係

---

## 12. 分析に使用したデータ

- `data/fill_records_20260322.jsonl` (152 records)
- `data/fill_records_20260323.jsonl` (128 records)
- `data/fill_records_20260324.jsonl` (97 records)
- `data/fill_records_20260325.jsonl` (68 records: 40 buy + 28 sell)
- 分析スクリプト: `temp/analyze_625_deep.py`
