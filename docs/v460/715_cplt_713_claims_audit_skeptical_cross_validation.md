# 715# 713# Claims Audit — Skeptical Cross-Validation

**目的**: 713# の主張を懐疑的視点から検証する。エビデンスの統計的妥当性・交絡因子・論理的飛躍を洗い出し、改善提案の実効性を問い直す。

---

## §0 方法論

713# は 3つの Phase (A/B/C) の比較に基づき「712# Fix で +42% 改善」と主張する。本文書では以下の 5 つの軸で各主張を監査する:

1. **統計的有意性**: サンプルサイズと信頼区間
2. **交絡因子**: 時間帯・市場条件・regime分布の差異
3. **生存者バイアス**: fill rate の変化が見かけの改善を生んでいないか
4. **因果の方向**: 相関を因果と断定していないか (AGENTS.md 原則)
5. **反実仮想の妥当性**: 「もし○○なら」の推論に根拠があるか

---

## §1 「Phase C は Phase A 比 +42% 改善」の検証

### 1.1 生データ

| | Phase A | Phase C |
|---|---------|---------|
| n | 100 | 139 |
| sc_avg | −0.684 | −0.425 |
| std | 1.427 | 1.376 |
| median | −0.413 | −0.371 |

### 1.2 統計的有意性: **不足**

Welch's t-test: **t = 1.405, df = 208.7**

|t| < 1.96 であり、**p > 0.05**。Phase A → C の差は **5% 水準で統計的に有意ではない**。

713# の「+42% 改善」は点推定の比較に過ぎず、標本のばらつき (std ≈ 1.4) を考慮すると、**偶然の範囲内で説明可能**。

### 1.3 中央値ではほぼ差がない

- A median: −0.413
- C median: −0.371

差はわずか 0.042 bps。平均値の差 (0.259 bps) は少数の外れ値に引きずられている可能性がある。

### 1.4 Trimmed mean (5% 両端除去)

- A: −0.627 (n=90)
- C: −0.452 (n=126)

差は 0.175 bps に縮小。極端な外れ値を除いても方向性は一致するが、改善幅は「+42%」ではなく **+28%** 相当。

### 1.5 判定

> **主張「+42% 改善」は過大表現**。中央値ベースでは +10%、trimmed mean で +28%、かつ統計的有意性は未達。「改善傾向が示唆される (suggestive)」が妥当な表現。

---

## §2 交絡因子: 時間帯分布の非対称

### 2.1 データ

| | 日中 (9-20h) | 夜間 |
|---|-------------|------|
| Phase A | 70 (70%) | 30 (30%) |
| Phase C | 86 (61%) | 53 (38%) |

Phase C は Phase A より**夜間比率が高い** (38% vs 30%)。

### 2.2 時間帯の影響

Phase C の夜間 fills (02-04h) は計 21 fills を含む。Phase A はこの帯域に fill がゼロ。夜間は一般にスプレッドが広がり、マーケットメイカーには有利にも不利にも働きうる。

### 2.3 時間帯マッチング

Phase A が存在する時間帯のみで Phase C を抽出:

| | n | sc_avg |
|---|---|--------|
| A (全時間帯) | 100 | −0.684 |
| C (A と同一時間帯のみ) | 118 | **−0.382** |

マッチング後もC優位は保持。ただし A の最悪時間帯 (00h: n=11) が C では n=4 に減少しており、完全なマッチングとは言えない。

### 2.4 判定

> 時間帯分布の差は存在するが、マッチング後も差が残る。交絡の可能性は完全には棄却できないが、時間帯だけでは Phase C の改善を説明しきれない。

---

## §3 交絡因子: 市場環境

### 3.1 スプレッド

| | avg spread (bps) | median spread |
|---|-----------------|---------------|
| Phase A | 2.46 | 2.41 |
| Phase C | 2.28 | 2.20 |

Phase C の方がスプレッドが**狭い**。マーケットメイカーにとり狭いスプレッドは不利（利幅が薄い）方向。つまりこの交絡は **Phase C 不利に作用** しており、712# Fix の改善効果を過小評価する方向。

### 3.2 Regime 分布

| | ranging | trending_down | trending_up |
|---|---------|--------------|-------------|
| A | 73% | 18% | **9%** |
| C | 66% | 15% | **17%** |

Phase C は trending_up が **倍増** (9% → 17%)。trending_up は全 regime 中最悪 (buy/trending_up sc=−0.62)。

713# の「regime 改善」は改善ではなく、trending_up の比率増にもかかわらず全体平均が悪化しなかった可能性がある。これは hard_skip_mult (sell/trending_up: 5.0) の寄与とも取れるが、trending_up 自体が異なる市場フェーズを反映している可能性も排除できない。

### 3.3 判定

> スプレッド交絡は Phase C 不利方向。regime 交絡は trending_up 増加が全体を引き下げうるが、712# Fix がそれを相殺した可能性。**総合すると市場交絡は 712# Fix の改善を否定する方向には働かない**。

---

## §4 生存者バイアス: Fill Rate の変化

### 4.1 データ

| | filled | total | fill rate |
|---|--------|-------|-----------|
| Phase A | 100 | 199 | **50.3%** |
| Phase C | 139 | 230 | **60.4%** |

Phase C の fill rate は +10pt 高い。

### 4.2 キャンセル理由の変化

| 理由 | A | C |
|------|---|---|
| timeout | 48 | 38 |
| spread_too_narrow | 24 | 34 |
| final_clamp_hard_skip | **7** | **0** |
| entry_gate_ev_negative | **5** | **0** |

`final_clamp_hard_skip` と `entry_gate_ev_negative` が Phase C で消滅。これは 712# Fix の直接的効果:
- F1 (ceiling 引上げ) → hard_skip が減少 → 本来キャンセルされていた fill が通過
- F2 (entry_gate 無効化) → ev_negative ブロック消滅

### 4.3 含意: 良い生存者バイアスか、悪い生存者バイアスか

hard_skip による 7 件のキャンセルは「pipeline が高 offset を出力 → ceiling でカット → hard_skip mult に到達して拒否」というフロー。ceiling 引上げによりこのフローが消えた。

**問題**: この 7 件が通過していたら sc は改善したか悪化したか？

Phase C の clamped buy fills (7件): sc_avg = −0.791。clamped buy は依然として悪い。つまり 712# Fix で「通過するようになった」fills の質は低い可能性がある。

しかし Phase C の clamped sell fills (23件): sc_avg = −0.113。sell 側では clamp が保護的に機能。

### 4.4 判定

> Fill rate +10pt は 712# Fix の直接効果だが、通過した追加 fills の質は検証が必要。**少なくとも buy 側では「低品質 fill が通過」している可能性がある**。Sell 側は ceiling が保護的に機能。

---

## §5 「Skip Gate モデルは機能している」の検証

### 5.1 713# の主張

> organic_pass (8 fills) の sc_avg = +0.028。モデル自体は機能している。

### 5.2 統計的信頼性: **壊滅的**

n = 8 での 95% 信頼区間: **[−0.643, +0.699]**

ゼロを大幅にまたぐ。8 件のサンプルから「モデルが機能している」と結論するのは統計的に不当。

### 5.3 ベースライン比較

04/04-06 (SHA: 352e3b7, 712# Fix 前) でも forced_pass は **55%** 存在した。

| | forced_pass率 | bypass率 |
|---|-------------|---------|
| Baseline (352e3b7) | 55% | 31% |
| Phase C (712# Fix) | 60% | 34% |

forced_pass 率に大差はない。**712# Fix が skip_gate の動作を変えたのではなく、以前から forced_pass は支配的だった**。

### 5.4 forced_pass と non-forced の sc 差

| | n | sc_avg |
|---|---|--------|
| Non-forced (organic + bypass) | 55 | −0.339 |
| Forced | 84 | −0.481 |

差は 0.142 bps。方向性は forced が悪いが、**この差も t-test で有意かは疑問** (n=55 vs n=84, std ≈ 1.4)。

### 5.5 Bypass の sc が悪い問題

bypass_mode=true (side-aware bypass) の 47 fills: sc = **−0.401**。「bypass してモデル判断を迂回した fills」が −0.401 という結果は、bypass_mode 自体の存在意義を問う。

bypass はモデルが「skip すべき」と言った fill を「side 的にはOK」として通している。その結果が −0.401 であるなら、bypass 基準も不適切である可能性。

### 5.6 判定

> - 「モデルは機能している」: n=8, CI がゼロをまたぎ、**根拠不足**
> - forced_pass 問題は 712# Fix 固有ではなく構造的
> - bypass_mode (sc=−0.401) も改善対象に含めるべき
> - **max_skip_rate 引上げは妥当な方向だが、bypass_mode の検証も必要**

---

## §6 「Sell Clamp が保護的に機能」の検証

### 6.1 713# の主張

> sell clamped (−0.113) > sell unclamped (−0.530)。ceiling がセーフティネット。

### 6.2 これは本当に逆説か？

ceiling は offset を引き上げる（保守的にする）方向に作用する。clamped fills の offset = 0.650 (ceiling上限) で、unclamped fills の avg offset = 0.466。

offset が高い → 注文は mid price から遠い → fill しにくいが fill すればスリッページが少ない。

**これは逆説ではなく、offset の基本力学どおりの結果**。高 offset = 低 sc_loss は当然。

### 6.3 問題は unclamped 側

unclamped sell (offset avg = 0.466) は pipeline の「自然な判断」の結果。その sc = −0.530 は、pipeline が **sell 側で一貫して offset を低く設定しすぎている** ことを示す。

713# I-3 の sell side_offset 引上げ (0.18→0.22) 提案は、この pipeline 基礎入力の底上げとして方向は正しい。

### 6.4 ただし ceiling = 0.65 の妥当性は検証不足

pipeline の pre_clamp avg = 1.122 は ceiling 0.65 の **1.73倍**。pipeline は 1.12 を「適正」と判断しているが、ceiling でカットされている。

二つの仮説:
- **H1**: pipeline の 1.12 は過大 → ceiling のカットが正しい保護
- **H2**: pipeline の 1.12 は妥当 → ceiling が利益機会を潰している

Phase C clamped sell の sc = −0.113 は「ceiling 0.65 でも損失」であり、H2 (pipeline が正しい) を支持する方向。**もし ceiling を 1.0 にしたら、さらに改善する可能性がある**。

### 6.5 判定

> - 「逆説的発見」は offset の基本力学の再確認であり逆説ではない
> - unclamped sell pipeline の低 offset 問題は正しい指摘
> - **ceiling 0.65 → さらに引上げの余地がある（ただしデータ不足で判断保留）**

---

## §7 「sell_hour_offset_boost が全く寄与していない」— 713# の見落とし

### 7.1 データ

713# offset pipeline 分析で指摘:

> sell の sell_hour と loss_boost ステージは全く寄与していない

追加検証の結果: Phase C の sell fills **68/69 件** (98.6%) で sell_hour ステージの出力が vol_guard ステージと同一。つまり **sell_hour_offset_boost は事実上不作動**。

### 7.2 原因仮説

sell_hour_offset_boost は YAML に 17 時間帯が設定されている (0h, 2h, 3h, 4h, 7h, 8h, 9h, 11h, 12h, 13h, 14h, 15h, 16h, 17h, 19h, 20h, 21h)。

しかし pipeline stages で sell_hour が vol_guard と同値ということは、**乗算対象が 0 か、乗算方式が加算方式に変わっているか、適用条件が満たされていない**可能性がある。

713# はこの問題を「注目」と記載したが、改善候補 I-1 ～ I-3 には含めなかった。**17 個の時間帯設定が全く効かない状態で放置されているのは構造的バグの疑い**。

### 7.3 判定

> **P0-P1 の改善候補よりも先に調査すべき可能性がある**。sell_hour_offset_boost が過去 (310#) では AS 41-62% の防御として設計されたのに不作動なら、sell 損失の根本原因の一つとなりうる。

---

## §8 改善提案の実効性監査

### 8.1 I-1: max_skip_rate 0.30 → 0.40

**713# の論拠**: forced_pass (sc=−0.462) を減らせば全体 sc が改善する。

**反論**:
- ベースライン (352e3b7) でも forced_pass 55% で sc = −0.349 だった。Phase C の forced_pass 60% は大差ない
- max_skip_rate を上げると fill 数が減る。現在の 120秒サイクルで 139 fills/26h ≈ 5.3 fills/h。skip 率が増えると **収入機会そのものが減少**
- 「forced_pass sc=−0.481 vs non-forced sc=−0.339」の差 0.142 bps は、fill 数減少による収入機会損失と全体損に吸収される可能性がある

**対案**: fill 数を保ったまま質を上げるなら、max_skip_rate ではなく **threshold 引下げ (0.8→0.5)** の方が organic pass を増やしつつ fill 数を維持できる。ただし 710# が指摘する sell model inversion 問題への対処が前提。

**結論**: 方向性は妥当だが、**fill 数 × 単価の積 (期待収益)** で評価すべき。rate 引上げ幅は 0.40 ではなく 0.35 から段階的に開始を推奨。

### 8.2 I-2: SAG redesign 有効化

**713# の論拠**: flat 0.5 bps tax → 逆比例ペナルティで +9.4%。

**反論**:
- +9.4% は 710# のシミュレーション値であり、ライブ環境での検証はゼロ
- 133/134 fills で SAG triggered。つまり SAG は「全 fill に適用される base tax」と化している。redesign しても全 fill に適用される点は変わらない
- redesign の narrow spread ペナルティ増は、Phase C の spread avg=2.28 環境で penalty = 2.0/2.28 ≈ 0.877 bps → 現行 0.5 bps より **+75% 増加**。narrow spread fills は悪化する可能性

**対案**: SAG を概念的に再考。133/134 triggered であるなら、SAG の条件判定 (`spread_threshold_bps: 15.0`) は事実上無意味。**SAG の penalty を side_offset に吸収して SAG を無効化** する方がシンプル。

**結論**: redesign 有効化自体は hot-reload で回収可能なので低リスクだが、**narrow spread 環境での penalty 増加に注意**。有効化後に spread_bps < 2.0 の fills を重点監視すべき。

### 8.3 I-3: sell side_offset 0.18 → 0.22

**713# の論拠**: unclamped sell offset avg=0.466 が低すぎる。

**反論**: 
- side_offset は pipeline の base 入力。0.18 → 0.22 は +22% 増だが、sell pipeline は base → as_shift → regime → ... → final で多段増幅される
- Phase C の sell final avg = 0.492。base を 0.04 上げると final は単純加算ではなく乗算段を経て **0.04 × 増幅率** だけ上がる
- sell の spread bucket 分析: spread>3.5 bps の 2 fills は sc=+1.67 (明確に正)。問題は spread<1.5 bps の 7 fills (sc=−1.09, win=0%)。side_offset 引上げは narrow spread fills を減らす（fill しなくなる）方向に作用する

**結論**: 方向は正しいが、+0.04 は控えめ。**sell_hour_offset_boost の不作動問題を先に解決すべき**。sell_hour が動いていれば、特定時間帯で動的に offset が上がり、side_offset の一律引上げが不要になる可能性。

---

## §9 713# が見落とした論点

### 9.1 Baseline との同一条件比較の欠如

713# は Phase A (750cd71) と Phase C (90de90f) を比較するが、SHA が異なる。750cd71 → 90de90f 間には YAML 変更以外に**コード変更 (refactor: split fill record builders)** が含まれる。

真の比較は「同一コード、同一市場条件、YAML のみ異なる」条件で行うべき。Phase B がそれに近いが、n=50 と深夜帯バイアスで使えない。

### 9.2 ベースライン SHA (352e3b7) との比較

| | n | sc_avg | spread_avg |
|---|---|--------|-----------|
| Baseline (352e3b7, 04/04-06) | 487 | **−0.349** | 2.05 |
| Phase C (90de90f, 712# Fix) | 139 | **−0.425** | 2.28 |

**Phase C はベースラインより悪い** (−0.425 vs −0.349)。

ただし:
- ベースラインの spread が狭い (2.05 vs 2.28) → ベースラインの方が狭 spread 環境
- ベースラインの forced_pass 率は 55% (Phase C は 60%)
- SHA (コード) が異なる — 750cd71 は CX4-CX6 変更を含み、352e3b7 は含まない

### 9.3 正直な現状認識

04/04 以降バージョンを重ねているが、sc_avg は改善していない:

```
04/04 (352e3b7):  -0.315
04/05 (352e3b7):  -0.252  ← 最良
04/06 (352e3b7):  -0.460
04/07 (mixed):    -0.618
04/08 (mixed):    -0.659
04/09 (90de90f):  -0.508  (途中)
```

**04/05 の −0.252 はコード変更以前の SHA**。CX4-CX6 (750cd71) や 712# Fix (90de90f) を経て、sc_avg は 04/05 のレベルに戻っていない。

> **市場条件の日次変動が、コード変更の効果を圧倒している可能性が高い**。

### 9.4 判定

> 713# は「712# Fix の効果」を示そうとしているが、最も説明力のある変数は **日次の市場条件変動** であり、YAML 変更の寄与は **二次的** である可能性を認めるべき。

---

## §10 総合評価と推奨

### 713# の主張に対する判定一覧

| 主張 | 判定 | 根拠 |
|------|------|------|
| Phase C +42% 改善 | **過大表現** | t=1.405 (p>0.05), median差 0.04bps, trimmed +28% |
| skip_gate モデル機能 | **根拠不足** | n=8, CI=[−0.64, +0.70] |
| sell clamp が保護的 | **offset力学の再確認** (逆説ではない) | 高offset=低AS は基本理論通り |
| hard_skip_mult 有効 | **支持** | trending_down buy: −1.44→−0.39 (regime内比較で有意) |
| entry_gate 無効化は正当 | **支持** | 全EV<0, 0件を正しくブロックで88件を不正にブロック |

### 改善提案の優先順序（修正案）

| 優先 | 内容 | 根拠 |
|------|------|------|
| **P0** | sell_hour_offset_boost 不作動調査 | 98.6% 不作動は構造的バグの疑い。17時間帯の防御が全て無効 |
| P1 | max_skip_rate 0.30→0.35 (段階的) | forced_pass 問題は方向性正しいが、fill数減を考慮し幅を半減 |
| P1 | bypass_mode の効果検証 | sc=−0.401 は bypass の存在意義に疑問 |
| P2 | SAG redesign 有効化 | 低リスク (hot-reload回収可能) だが narrow spread 悪化に注意 |
| P2 | sell side_offset 引上げ | sell_hour 修正後に再判断 |

### 713# 著者へのフィードバック

1. **統計的主張には信頼区間を添付すべき**。n=100 vs n=139 の比較で「+42% 改善」は点推定の比較で不十分
2. **ベースライン (352e3b7) との比較が欠落**。Phase A (750cd71) は CX4-CX6 変更後であり、真のベースラインではない
3. **sell_hour 不作動問題を「注目」で済ませず、改善候補に含めるべきだった**
4. **日次市場変動の寄与を過小評価**。04/05 の −0.252 はコード変更なしで達成されており、コード変更の限界を示す

---

## 付録: 再現コマンド

```bash
.venv/Scripts/python.exe temp/analyze_715.py
```
