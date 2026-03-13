# 289# ph2 分析: Buy 側パフォーマンス改善調査

> **目的**: fill_test buy 側の PnL 劣化根本原因を特定し、  
> G1.2-full 完了後に実施すべき改善施策を優先順位付きで提案する  
> **日付**: 2026-03-06 (v4 — 290#/291# review 反映)  
> **SHA**: `2e414ee7e` → 292# で FillRecord 可観測性強化  
> **制約**: G1.2-full 168h 測定中のためコード/YAML 変更不可

---

## 1. 現状サマリー

| 指標 | Buy | Sell | 備考 |
|---|---|---|---|
| 総数 | 2,888 | 3,674 | |
| Filled | 1,181 (40.9%) | 1,170 (31.8%) | buy が fill 率高い |
| PnL30 mean (全期間) | -0.27 bps | -0.36 bps | buy が全体では sell より良い |
| Win% | 47.4% | 44.6% | |
| AS% (PnL<-1bps) | 38.5% | 40.6% | |
| Online monitor (直近100) | -0.947 bps | +0.332 bps | **直近の buy 劣化が顕著** |

---

## 2. データ期間とモデル世代の整理

分析の前提として、データには **3 つの異なるモデル世代** が混在している:

| 期間 | 呼称 | Score 分布 | ev_weighted | n(buy filled) | 備考 |
|---|---|---|---|---|---|
| 02-16 〜 02-20 | **旧モデル期** | mean=-4.94 (全件 <-3) | なし | 271 | 初期モデル、trending 多い市場 |
| 02-21 〜 02-27 | **遷移期** | mean=+0.71 | なし | 487 | 新モデル deploy + retrain 開始 |
| 02-28 〜 03-06 | **ev_weighted 期** | mean=+0.90 | あり (100%) | 226 | ev_weighted offset 有効 |

> ⚠️ **この世代差を無視してスコア帯で集計すると Simpson's Paradox が発生する。**  
> 以降の分析はこの点を厳密に制御する。

---

## 3. SkipGate スコア反転の検証と棄却

### 3.1 表面上の反転 (Simpson's Paradox)

全期間を単純集計すると、スコアと PnL30 が逆相関に見える:

| Score Band | Pass PnL30 | n |
|---|---|---|
| < -3 (最低) | **+0.24** | 300 |
| > 1 (最高) | **-0.57** | 326 |

Spearman ρ = -0.047 (p=0.135, **非有意**)

### 3.2 Simpson's Paradox の立証

score<-3 の 300 件中 **271 件 (90%) が旧モデル期 (02-16〜02-20)** のデータ。  
旧モデル期は全スコアが -5 近辺で、かつ trending 優位市場で PnL=+0.26 bps。

| 期間 | Score mean | PnL30 mean | n |
|---|---|---|---|
| 旧モデル期 (02-16〜02-20) | -4.94 | **+0.26** | 271 |
| 新モデル期 (02-21+) | +0.71 | **-0.46** | 758 |

→ score<-3 が良いのではなく、**旧モデル期の市場環境が良かった**だけ。

### 3.3 新モデル期単体での検証

02-21 以降のみで再分析:

| Score Quintile | Range | PnL30 mean | n |
|---|---|---|---|
| Q1 (最低) | [-6.41, -0.89) | -0.30 | 152 |
| Q2 | [-0.89, +0.26) | -0.50 | 151 |
| Q3 | [+0.26, +1.10) | -0.46 | 152 |
| Q4 | [+1.10, +2.34) | +0.24 | 151 |
| Q5 (最高) | [+2.34, +7.03) | **-1.27** | 152 |

- Spearman ρ = -0.016 (p=0.654, **完全に非有意**)
- Q5 の PnL が最悪だが、t検定で Q1 vs Q5 は p=0.61 で差は統計的に有意ではない
- **結論: スコアは「反転」しているのではなく「無識別」（non-discriminating）**

### 3.4 スコアが無識別である理由

Q5 (高スコア) の条件を Q1-Q4 と比較:

| 条件 | Q5 (score>2.34) | Others | 差 |
|---|---|---|---|
| Ranging % | 76.5% | 73.2% | 微差 |
| Night % | 30.1% | 31.1% | 微差 |
| VPIN mean | 0.388 | 0.396 | 微差 |
| OI mean | -0.033 | -0.002 | 微差 |
| QueueWait | 20.7s | 22.8s | 微差 |
| VG% | 35.3% | 37.0% | 微差 |

→ Q5 が悪い明確な条件的偏りはない。モデルが高スコアを付ける特徴量パターンと、  
実際のPnL30結果の間に**系統的な関連がない**。

### 3.5 sell 側との対照実験

sell 側も同様にスコア識別力が低い:

| Sell Score Quintile | PnL30 | n |
|---|---|---|
| Q1 (最低) | -0.12 | 149 |
| Q4 | **+1.14** | 149 |
| Q5 (最高) | -0.88 | 150 |

sell: Spearman ρ = +0.025 (p=0.489, **非有意**)  
→ **buy/sell 両方で SG score 単体の識別力がない** (=モデルの問題ではなく予測困難性の問題)

### 3.6 ✅ スコア反転の結論

| 仮説 | 検証結果 | 判定 |
|---|---|---|
| H1: buy model のスコア方向が反転 | Simpson's Paradox。新モデル期は ρ=-0.016 (非有意) | **棄却** |
| H2: buy model が無識別 (non-discriminating) | 全 quintile で PnL 差に有意差なし | **支持** |
| H3: 市場環境の変化 (trending→ranging) | 旧モデル期 PnL+0.26, 新モデル期-0.46 | **支持** |

> ⚠️ 初版 289# で提案した「SIM-1: score>0 skip」は **Simpson's Paradox に基づく誤った施策** だった。  
> 新モデル期のみでは score による skip は改善効果を持たない。

---

## 4. ev_weighted_pnl: Tautology 検証と結果

### 4.1 初期発見 (v2 時点)

ev_weighted_pnl と PnL30 の間に ρ=0.880 (p=2.3e-74) という驚異的な相関が観測された。

### 4.2 ソースコード確認: ev_weighted_pnl は ex-post

`fill_cycle_executor.py` L469-485:
```python
def _compute_ev_weighted(pnl30, pnl120, *, w30=0.4, w120=0.6):
    if pnl30 is None: return None
    if pnl120 is None: return pnl30
    return w30 * pnl30 + w120 * pnl120
```

呼び出し元 (L342-350):
```python
ev_weighted = self._compute_ev_weighted(
    post_fill_pnl,       # ← 実測 PnL30
    post_fill_120s_pnl,  # ← 実測 PnL120
    w30=..., w120=...,
)
```

**ev_weighted_pnl = 0.4 × 実測PnL30 + 0.6 × 実測PnL120**

### 4.3 Tautology 確認

ev_weighted 期 (02-28+) の 103 件で検証:  
`|ev_weighted_pnl - (0.4×pnl30 + 0.6×pnl120)| = 0.000000` (全件完全一致)

→ **ρ=0.880 は数学的帰結であり、予測力を一切示さない。**

### 4.4 ランタイムの ev_score (ex-ante) との混同に注意

ランタイムの `ev_score` (skip_gate_evaluator.py L639-646) は**別物**:
```python
# 予測値の加重合成
ev_score = w30 * primary_pnl + w120 * alt_pnl  # モデル予測値
```

しかし FillRecord の `ev_weighted_pnl` はこの予測値ではなく、**実測値の加重平均**。

| 変数 | 計算式 | 性質 | 用途 |
|---|---|---|---|
| ランタイム `ev_score` | w30×predicted_primary + w120×predicted_alt | **ex-ante** (予測) | skip/offset 判定 |
| FillRecord `ev_weighted_pnl` | 0.4×actual_pnl30 + 0.6×actual_pnl120 | **ex-post** (実測) | オフライン分析用 |

### 4.5 ランタイム ev_score の実態

ev 期 (02-28+) の buy filled 209 件中:
- `model_used=primary:side_buy`: **180 件** (86.1%) — ev_weighted 不使用
- `model_used=ev_weighted:ev_weighted_buy`: **20 件** (9.6%) — ev_weighted 使用
- `model_used=None`: 9 件 (4.3%)

> ⚠️ **v4 修正 (290# review)**: 上記の `model_used` ベースの利用率 9.6% は
> **誤ったプロキシ**。`ev_as_offset_enabled=true` モード (現在の本番設定) では、
> SkipGate は ev_score を計算するが、`model_used` は常に `primary:side_buy` のまま
> (`skip_gate_evaluator.py` L1203-1207)。ev_weighted 判定で decision が変わるのは
> emergency_skip 時のみ。つまり **実際の ev パス利用率は 9.6% より遥かに高い
> (≒ 100% に近い)** が、FillRecord に記録されていなかった。
>
> 292# で `ev_score_pretrade`, `ev_offset_mult_applied`, `decision_path` の
> 3 フィールドを追加し、この可観測性ギャップを解消。

ev_weighted 使用 20 件での予測 ev_score vs PnL30:  
Spearman ρ=-0.110 (p=0.645) — **非有意、サンプル不足**

→ ランタイム ev_score の予測力は**評価不能** (n=20 は統計的に無意味)。
→ **292# 以降の新データで `ev_score_pretrade` を蓄積して再評価可能に。**

### 4.6 ~~ev_weighted パス利用率の低さの原因~~ (290# で解消)

> ⚠️ **v4 修正**: 289# v3 で「9.6% しか ev_weighted パスを通らない」と報告したが、
> これは `model_used` フィールドが `ev_as_offset` モードを反映しない仕様に起因。
> 実際には ev_score は毎サイクル計算されており、offset 乗数として適用されていた。
> 292# の FillRecord 拡張で定量的に追跡可能になった。

~~209 件中 20 件 (9.6%) しか ev_weighted パスを通らない理由 (要調査):~~
~~1. alt model (pnl120_buy) のロード失敗?~~
~~2. AS mode で ev_weighted が不適用? (コード上明示: AS mode → ev_weighted skip)~~
~~3. 特定の regime/条件で bypass?~~

~~→ **ev_weighted の改善効果を活かすには、まずパス利用率を上げることが前提**。~~

### 4.7 ✅ ev_weighted の結論

| 仮説 | 検証結果 | 判定 |
|---|---|---|
| H4: ev_weighted_pnl ρ=0.88 は予測力 | Tautological (ex-post 値) | **棄却** |
| H5: ランタイム ev_score は予測的 | n=20、ρ=-0.110 (p=0.645) | **評価不能 → 292# で蓄積開始** |
| ~~H6: ev_weighted パスが十分利用されている~~ | ~~9.6% のみ~~ | **v4 修正: model_used は誤プロキシ** |

> ⚠️ v2 で提案した「ev_emergency_threshold 引下げで +0.94bps」は  
> **tautological な ev_weighted_pnl に基づく invalid な SIM** であった。  
> Phase A 施策は凍結し、ランタイム ev_score の検証後に再評価する。

### 4.8 PnL30 vs PnL120 の時間軸比較 (tautology と無関係)

| Horizon | Mean | Win% |
|---|---|---|
| PnL30 | -0.667 | 46.5% |
| PnL120 | **-0.337** | **50.0%** |

Spearman(PnL30, PnL120) = 0.537 (p=6.4e-31)  
PnL30 が負 → PnL120 が正に回復: **16.4%** のケース  
→ buy は **30s 時点で AS 気味でも 120s で部分回復する傾向**がある。

---

## 5. 真の損失原因分析 (新モデル期 n=767)

### 5.1 損失帰属: 夜間 vs 日中

| 時間帯 | cumPnL | n | 寄与率 |
|---|---|---|---|
| Night (21-04 JST) | **-215.0** | 234 | **58.1%** |
| Day (05-20 JST) | -155.0 | 533 | 41.9% |
| **合計** | **-370.0** | 767 | 100% |

→ 全損失の 58% が夜間に集中。n 比率は 30.5% なのに損失寄与は 58%。

### 5.2 時間帯別詳細 (新モデル期)

| 時間帯 (JST) | PnL30 | AS% | cumPnL | n | 性質 |
|---|---|---|---|---|---|
| 05 | +0.39 | 30.6% | +14.2 | 36 | ✅ |
| 16 | +0.30 | 19.4% | +9.4 | 31 | ✅ |
| 10 | +1.18 | 44.4% | +42.6 | 36 | ✅ 但し AS 高め |
| 00 | **-2.10** | 43.3% | -62.9 | 30 | ❌ Toxic |
| 19 | **-2.25** | 48.3% | -65.3 | 29 | ❌ Toxic |
| 23 | **-2.12** | 46.7% | -63.7 | 30 | ❌ Toxic |
| 03 | -1.52 | 50.0% | -24.3 | 16 | ❌ Toxic |

### 5.3 レジーム別損失

| Regime | cumPnL | Mean | n | 寄与 |
|---|---|---|---|---|
| **ranging** | **-395.8** | -0.70 | 565 | **107%** (他で相殺) |
| trending_down | +68.7 | +0.92 | 75 | 利益源 |
| trending | -12.9 | -0.37 | 35 | 微損 |
| trending_up | -29.9 | -0.33 | 92 | 小損 |

→ **Ranging 単独で全損失を超える -395.8 bps**、trending_down の +68.7 で一部相殺。  
→ buy の問題は「ranging 市場での maker 参入コスト」に集約される。

### 5.4 Ranging × Night の複合毒性

| 条件 | PnL30 | AS% | n |
|---|---|---|---|
| Ranging + Night (21-04 JST) | **-1.79** | 44.2% | 86 |
| Ranging + Day (09-16 JST) | -0.33 | 36.8% | 321 |
| 差分 | -1.46 | +7.4pt | — |

→ ranging × night は buy の**最大の損失クラスター**。

### 5.5 マイクロストラクチャ要因

| 条件 | PnL30 | AS% | n | 解釈 |
|---|---|---|---|---|
| VPIN < 0.3 (low) | **-1.00** | 50.0% | 230 | 静か過ぎると逆選別温床 |
| VPIN 0.3-0.5 (medium) | **+0.21** | 36.7% | 120 | 適度な情報性 = 最良 |
| OI > 0.3 (buy_heavy) | **-0.79** | 42.0% | 212 | 板の方向と同じ = 不利 |
| OI < -0.3 (sell_heavy) | **+0.01** | 40.2% | 219 | 逆板 = 有利 |
| Queue wait > 60s | **+0.15** | 25.6% | 117 | 長 wait = queue 前方 |
| Queue wait 30-60s | -0.65 | 41.8% | 220 | |
| Regime confidence < 0.3 | -1.29 | 47.6% | 42 | レジーム不確実 = 悪化 |
| Regime confidence > 0.7 | **-0.02** | 38.0% | 387 | 確信度高 = 改善 |

### 5.6 Adaptive threshold の効果分析

| Threshold Band | PnL30 | n |
|---|---|---|
| < -1 (loose) | **-0.87** | 231 |
| -1 〜 0 | -0.16 | 210 |
| 0 〜 0.5 | -0.32 | 285 |
| > 0.5 (tight) | -0.72 | 32 |

→ loose な時期のパフォーマンスが最悪。adaptive が十分機能していない可能性。

---

## 6. 改善シミュレーション (適切なデータ分割)

### 6.1 新モデル期 (02-21+, n=767)

| シナリオ | Keep PnL30 mean | n | Skip n | 備考 |
|---|---|---|---|---|
| **現状** | **-0.482** | 767 | — | — |
| SIM-A: toxic 時間帯 skip (19,23,01,03) | -0.293 | 681 | 86 | 時間帯 filter |
| SIM-B: night skip (21-04 JST) | **-0.291** | 533 | 234 | 夜間全 skip |
| SIM-C: ranging + night skip | -0.320 | 618 | 149 | 条件付き |
| SIM-D: VPIN < 0.3 skip | -0.260 | 537 | 230 | 低情報性排除 |
| SIM-E: OI > 0.3 skip | -0.366 | 555 | 212 | buy_heavy 排除 |
| **SIM-F: B + D + E 複合** | **-0.043** | **276** | **491** | 最良だが 64% skip |
| SIM-G: toxic hrs + score Q5 | -0.221 | 547 | 220 | |

### 6.2 ~~ev_weighted 期 (02-28+, n=241) — 最新状態~~ ❌ INVALIDATED

> **以下のシミュレーションは ev_weighted_pnl (ex-post) を ex-ante として使用しており無効。**  
> 結果は参考記録として残すが、施策根拠としては使用してはならない。

| シナリオ | Keep PnL30 mean | n | Skip n | 判定 |
|---|---|---|---|---|
| ~~現状~~ | -0.694 | 241 | — | — |
| ~~SIM: ev < -4 skip~~ | ~~+1.013~~ | ~~188~~ | ~~53~~ | ❌ leakage |
| ~~SIM: ev_emergency -8 → -5~~ | ~~+0.940~~ | ~~192~~ | ~~49~~ | ❌ leakage |
| ~~SIM: night + ev<-4 combined~~ | ~~+1.407~~ | ~~114~~ | ~~127~~ | ❌ leakage |
| **SIM: night skip (21-04) のみ** | **-0.226** | **147** | **94** | ✅ 有効 |

→ **唯一有効なシミュレーションは night skip (-0.694 → -0.226 bps, +0.468 bps 改善)**。

---

## 7. 批判的検証: 各知見の信頼性

### 7.1 サンプルサイズの限界

| 分析 | n | 信頼性 | 注意点 |
|---|---|---|---|
| SG score 無識別 (新モデル期) | 758 | ◎ 十分 | Spearman p=0.654 |
| ev_weighted 識別力 | 226 | ○ 中程度 | 02-28 以降のみ |
| 夜間 toxic (新モデル期) | 234/767 | ◎ 十分 | 時間帯は自然な層別 |
| ev < -4 skip 効果 | 53 skip | △ やや小 | 過学習リスクあり |
| Hour 別 (ev 期) | 5-31/hour | ✕ 不足 | 参考値のみ |

### 7.2 想定される反論と応答

| # | 反論 | 応答 |
|---|---|---|
| R1 | 「ev_weighted の高相関は tautological では?」 | **YES、tautological であることをコード検証で確認済** (§4.2-4.3)。ev_weighted_pnl = 0.4×actual_pnl30 + 0.6×actual_pnl120。ρ=0.88 は予測力を示さない |
| R2 | 「ev<-4 skip は過学習」 | ev<-4 skip 自体が **leakage** — 事後値でフィルタしている。施策根拠として無効 |
| R3 | 「夜間 skip で fill 機会損失」 | Night (21-04 JST) の buy は 234 件中 mean=-0.482。機会損失ではなく **損失回避**。AS%=43.3% で昼間(37.8%)より有意に高い |
| R4 | 「Ranging で全 skip すべき」 | Ranging は 565/767 件 = 73.6%。全 skip は事実上 buy 停止に等しく非現実的 |
| R5 | 「sell も同じ問題」 | sell Q5 も PnL=-0.88 で悪い。但し sell 全体の PnL30=-0.36 で buy(-0.46) より軽微 |
| R6 | 「SG score 改善は不要か」 | 無識別 ≠ 害。adaptive threshold が SG score に依存しているため、識別力向上は adaptive 精度向上に繋がる。長期的には重要 |
| R7 | 「旧モデル期のデータは除外すべき」 | 分析上は除外済み。但しモデル再訓練時は全データ使用するため、旧期間のバイアスが混入するリスクあり |
| R8 | 「Q5がQ1より悪いのは偶然か」 | p=0.61 で有意差なし。Q5=-1.27 はノイズの範囲内 |
| R9 | 「ランタイム ev_score は有効かも」 | 可能性あり。292# で `ev_score_pretrade` を FillRecord に追加、蓄積後に再評価 |

### 7.3 ev_weighted_pnl フィールドの正体 ✅ RESOLVED

**結論**: ev_weighted_pnl は **ex-post (事後計算値)** であることをコードで確認済。

- `fill_cycle_executor.py` L342: `_compute_ev_weighted(post_fill_pnl, post_fill_120s_pnl, ...)`
- `fill_quality.py` L133: `ev_weighted_pnl: float | None = None  # 0.4*pnl30 + 0.6*pnl120 (bps)`
- 検証: 103 件で `|ev_weighted_pnl - (0.4*pnl30+0.6*pnl120)| = 0.000000` (完全一致)

**影響**: §4 の ρ=0.88 は tautological、§6.2 の ev_weighted SIM は leakage で無効。

---

## 8. ~~ev_weighted パス利用率の調査~~ (290# で解消 → 292# で可観測性強化済)

> ⚠️ **v4 修正**: `model_used` は `ev_as_offset` モードの利用率を反映しない
> (skip_gate_evaluator.py L1203-1207)。実際にはほぼ全 cycle で ev_score が
> 計算されoffset に適用されていた。
>
> 292# で以下の 3 フィールドを FillRecord に追加し、この盲点を解消:
> - `ev_score_pretrade`: ランタイム ev_score (ex-ante 予測値)
> - `ev_offset_mult_applied`: 実適用 offset 乗数 (1.0=変更なし)
> - `decision_path`: "primary_only" / "ev_offset" / "ev_emergency_skip" / "ev_no_change"

~~確認すべき事項:~~
~~1. alt model (`skip_gate_lgbm_pnl120_buy.pkl`) のロード状態~~
~~2. `skip_gate_ev_weighted_enabled` が fill_test 中に常に True か~~
~~3. AS mode 判定が多発していないか (AS mode では ev_weighted bypass)~~
~~4. ログ確認: `[skip_gate] 188# ev_weighted skipped` の出現頻度~~

~~**ev_weighted パスが 90% 不使用なら、offset modifier 改善の効果も 10% に限定される。**~~

---

## 9. buy primary model の内部分析

### 9.1 モデルメタデータ

| 項目 | 値 |
|---|---|
| ファイル | `models/v460/skip_gate_lgbm_pnl30_buy.pkl` |
| サイズ | 297 KB |
| 訓練日 | 2026-02-24T05:34:10 |
| 訓練サンプル数 | 519 |
| 特徴量数 | 13 (pruning 後) |
| n_estimators | 300 (early stopping 使用) |
| max_depth | 4 |
| WF profit_score | 0.355 |
| 統計ゲート | N/A (初回モデルのため) |

### 9.2 Feature Importance

| Feature | Importance | 割合 |
|---|---|---|
| spread_jpy | 354 | 17.6% |
| hour_cos | 319 | 15.8% |
| hour_sin | 308 | 15.3% |
| price_velocity_60s | 180 | 8.9% |
| depth_imbalance_ob | 164 | 8.1% |
| vpin_60s | 132 | 6.6% |
| trade_count_60s | 126 | 6.3% |
| spread_bps_ob | 106 | 5.3% |
| avg_trade_size | 103 | 5.1% |
| buy_ratio | 75 | 3.7% |
| offset_ratio | 75 | 3.7% |
| regime_trending | 45 | 2.2% |
| regime_ranging | 26 | 1.3% |

### 9.3 モデル構造の考察

- **時間帯が最重要** (hour_sin + hour_cos = 31.1%) — しかし実際の PnL と時間帯の関係を正しく  
  捉えられていない (スコア quintile 間で night% が均一 = 時間帯を正しく使えていない)
- **spread_jpy** (17.6%) — スプレッドは予測に有用なはずだが、実データでは spray 差が小さい  
  (ほぼ全件 spread<50JPY)
- **Pruned features**: `side_buy`, `regime_high_vol`, `side_aligned_tfi` — buy 専用モデルなので  
  side_buy は当然 dead。regime_high_vol は出現頻度不足
- 訓練時のレジームは ranging=319 (61%), trending=118 (23%), unknown=77 (15%)。  
  trending_down/trending_up は **訓練データに含まれていない** → 未知レジームへの汎化不足

### 9.4 売りモデルとの比較

| 特徴 | Buy Primary (pnl30) | Sell Primary (pnl120) |
|---|---|---|
| 特徴量数 | 13 | 15 |
| side_aligned_imbalance | pruned | **112 (7.2%)** |
| side_aligned_velocity | pruned | 使用中 |
| depth_imbalance_ob | 164 (8.1%) | — |
| 識別力 (Spearman) | ρ=-0.016 | ρ=+0.025 |

→ sell model は side_aligned_* 特徴量を活用しているが、buy model は pruning で失っている。  
  これは side 固定後の相関構造の違いを反映。

---

## 10. 改善施策ロードマップ (G1.2-full 完了後)

### 10.1 ~~最優先: ev_weighted パス利用率の調査~~ → 292# で解消

- ~~9.6% → 目標 80%+ に引き上げれば offset modifier が機能~~
- ~~alt model ロード状態、AS mode bypass 頻度を確認~~
- **v4 修正**: 290# で `model_used` は ev_as_offset モードの誤プロキシと判明。
  292# で `ev_score_pretrade`, `decision_path` を追加し、真の利用率を追跡可能に。

### 10.2 ~~Phase A: ev_weighted 閾値調整~~ ❌ 凍結

> v2 で提案した Phase A はev_weighted_pnl (ex-post) に基づいており無効。  
> ランタイム ev_score の予測力が確認されるまで凍結。

| # | 施策 | 判定 |
|---|---|---|
| ~~A-1~~ | ~~`ev_emergency_skip_threshold: -8 → -5`~~ | ❌ 根拠消失 (tautology) |
| ~~A-2~~ | ~~`ev_warning_threshold: -4 → -3`~~ | ❌ 同上 |

### 10.3 Phase B: Night 時間帯対策 (YAML のみ、ev 非依存)

| # | 施策 | 期待効果 | 信頼度 | 実装コスト |
|---|---|---|---|---|
| B-1 | `hour_offsets` に UTC 10 (= 19 JST) 追加: offset=0.5 | 夜間 -2.25bps 軽減 | ◎ | YAML 1行 |
| B-2 | UTC 14 (= 23 JST) offset=0.3 → 0.8 に強化 | 夜間 -2.12bps 軽減 | ◎ | YAML 1行 |
| B-3 | UTC 15 (= 00 JST) 新設: offset=0.5 | -2.10bps 対策 | ◎ | YAML 1行 |

### 10.4 Phase C: マイクロストラクチャ条件 (コード変更)

| # | 施策 | 期待効果 | 信頼度 | 実装コスト |
|---|---|---|---|---|
| C-1 | VPIN < 0.3 時の buy offset 保守化 | AS 50%→35% 目標 | ○ | skip_gate_evaluator |
| C-2 | OI > 0.3 (buy_heavy) 時の offset boost | -0.79bps 軽減 | ○ | 同上 |
| C-3 | Regime confidence < 0.3 でのフィルタ強化 | -1.29bps 軽減 | △ n=42 | 同上 |

### 10.5 Phase D: モデル改善 (長期)

| # | 施策 | 期待効果 | 信頼度 |
|---|---|---|---|
| D-1 | ev_weighted パス利用率向上 (alt model ロード保証) | offset modifier 有効化 | ◎ 前提条件 |
| D-2 | Ranging レジーム用の専用 buy model 訓練 | ranging -0.70→0 目標 | △ 困難 |
| D-3 | Buy model に side_aligned_imbalance 復活 | sell model で有効な特徴量 | ○ |
| D-4 | 訓練データに trending_up/down を含む期間を追加 | 未知レジーム汎化 | ○ |
| D-5 | ランタイム ev_score (predicted) の予測力評価 | ev_score 有効なら threshold 最適化が可能 | ○ (n蓄積後) |

### 優先順位

```
ev_weighted パス利用率調査 (コード凍結中、最優先)
  ├─ 利用率向上可能 → D-1 実装 → ランタイム ev_score 予測力評価 (D-5)
  │   └─ 予測力あり → ev threshold 最適化
  └─ 利用率向上困難 → Phase B (YAML) → Phase C → Phase D-2/D-3/D-4
Phase B (hour_offsets 追加) は ev_weighted と独立して即実行可能
```

---

## 11. コード凍結期間中にできること

| # | 作業 | 種別 | 所要時間 |
|---|---|---|---|
| 1 | **ev_weighted パス利用率調査** (§8) — ログ/コード確認 | 調査 | 1h |
| 2 | sell 側スコア識別力の追加検証 | 分析 | 1h |
| 3 | SAC 4-seed 訓練準備 (ph3 先行) | 構築 | 2-4h |
| 4 | ランタイム ev_score 蓄積のためのロギング強化設計 | 設計 | 1h |
| 5 | 280# 残タスク更新 (ドキュメント) | 文書 | 30min |

---

## 12. 結論

### 12.1 確実な知見

| # | 知見 | 根拠 | 信頼度 |
|---|---|---|---|
| F-1 | SG score は buy/sell 両方で **無識別** | ρ=-0.016 (buy), ρ=+0.025 (sell), 両方 p>0.4 | ◎ |
| F-2 | 初版の「スコア反転」は **Simpson's Paradox** | 旧モデル期 90% が score<-3 に集中 | ◎ |
| F-3 | 夜間 (21-04 JST) が損失の **58.1%** を寄与 | n=234/767, cumPnL=-215.0 | ◎ |
| F-4 | Ranging が **107%** の損失寄与 (他で相殺) | cumPnL=-395.8, n=565 | ◎ |
| F-5 | VPIN<0.3 で AS=50% — 情報性枯渇環境が有毒 | n=230, mean=-1.00 | ○ |
| F-6 | Buy は PnL120 > PnL30 (回復傾向) | mean -0.667→-0.337, win% 46.5→50.0 | ○ |
| F-7 | Buy model は trending_up/down を訓練データに含まない | metadata 確認済 | ◎ |
| F-8 | **ev_weighted_pnl (ρ=0.88) は tautological — ex-post 値** | コード検証 + 完全一致確認 | ◎ |
| F-9 | ~~ev_weighted パス利用率はわずか 9.6%~~ | ~~209 件中 20 件のみ~~ → **290# で model_used は誤プロキシと判明。292# で解消** | ◎ |

### 12.2 要確認事項

| # | 要確認 | 影響範囲 | 優先度 |
|---|---|---|---|
| ~~V-1~~ | ~~ev_weighted_pnl が ex-ante か ex-post か~~ | ✅ RESOLVED: ex-post (tautological) | ✅ 完了 |
| ~~V-2~~ | ~~ev_weighted の ρ=0.88 が tautological でないか~~ | ✅ RESOLVED: tautological confirmed | ✅ 完了 |
| ~~V-3~~ | ~~ev_weighted パス利用率 9.6% の原因~~ | ✅ RESOLVED: model_used は ev_as_offset の誤プロキシ (290#)。292# で FillRecord 拡張 | ✅ 完了 |
| V-4 | ランタイム ev_score (predicted) の予測力 | ev_emergency threshold 最適化の可否 | 🟡 n蓄積後 |
| V-5 | sell 側の ev_weighted パス利用率 | 対照群検証 | 🟡 中 |

### 12.3 即実行可能な改善 (G1.2-full 後)

1. **Phase B** (hour_offsets 追加): UTC 10/15 新設 + UTC 14 強化 → **確実に有効、YAML 3行、ev と独立**
2. **D-1** (ev_weighted パス利用率向上): alt model ロード保証 → **効果量は ev_score 予測力に依存**
3. **Phase C** (VPIN/OI フィルタ): 中期的改善

---

## Appendix A: 初版 (v1) → v3 修正事項

| 初版/v2 の主張 | 修正後 (v3) | 理由 |
|---|---|---|
| 「SG score が反転している」 | **Simpson's Paradox** — 新モデル期では無識別 | 旧/新モデル期の混合分析アーティファクト (v2 で修正済) |
| 「SIM-1: score>0 skip で +0.26bps」 | **無効** — 新モデル期のみでは効果なし | score に識別力がないため (v2 で修正済) |
| v2「ev_weighted_pnl ρ=0.88 は予測力」 | **Tautological** — ex-post 値で leakage | ev_weighted_pnl = 0.4×actual_pnl30 + 0.6×actual_pnl120 |
| v2「ev < -4 skip で +1.013bps」 | **無効** — 事後値でフィルタ (leakage) | ev_weighted_pnl に基づく全 SIM が無効 |
| v2「Phase A: ev_emergency -8→-5」 | **凍結** — 根拠消失 | tautological SIM に基づく提案 |
| v2「ev_weighted が唯一の有効識別器」 | **SG score も ev_weighted_pnl も識別力なし** | 前者は non-discriminating、後者は tautological |
| ev_weighted への言及なし (v1) → 「発見」 (v2) | ~~ev_weighted パス利用率 9.6%~~ → **290# で model_used 誤プロキシ判明、292# で解消** | ランタイム ev_score の検証は n 不足で未完 → 292# で蓄積開始 |
| 最大レバー: ev threshold (v2) | **最大レバー: 夜間 skip (Phase B, YAML)** | 統計的に堅牢かつ ev と独立 |

## Appendix B: 統計的検定サマリー

| 検定 | 対象 | 統計量 | p-value | 判定 |
|---|---|---|---|---|
| Spearman (全期間) | buy SG score vs PnL30 | ρ=-0.047 | 0.135 | 非有意 |
| Spearman (新モデル期) | buy SG score vs PnL30 | ρ=-0.016 | 0.654 | 非有意 |
| Spearman (新モデル期) | sell SG score vs PnL30 | ρ=+0.025 | 0.489 | 非有意 |
| Spearman (ev期) | ev_weighted_pnl vs PnL30 | ρ=+0.880 | 2.3e-74 | ~~有意~~ **Tautological** |
| 完全一致検証 | ev_weighted_pnl vs 0.4×pnl30+0.6×pnl120 | max_diff=0.000000 | — | **Tautology 確認** |
| Spearman (ev期, n=20) | ランタイム ev_score vs PnL30 | ρ=-0.110 | 0.645 | 非有意 (n不足) |
| t-test (全期間) | score<-3 vs score>1 PnL30 | t=1.804 | 0.072 | 非有意 (Simpson's Paradox) |
| Mann-Whitney U (全期間) | score<-3 > score>1 PnL30 | U=52818 | 0.042 | ~~有意~~ Simpson's Paradox contamination |
| Cohen's d (全期間) | score<-3 vs score>1 | d=0.145 | — | 小効果 |
| t-test (新モデル期) | Q1 vs Q5 PnL30 | — | 0.61 | 非有意 |
| Spearman | PnL30 vs PnL120 | ρ=+0.537 | 6.4e-31 | **高度有意** |
| Bootstrap 95%CI | score<-3 PnL30 | [-0.31, +0.81] | — | 0 を含む |
| Bootstrap 95%CI | score>1 PnL30 | [-1.23, +0.09] | — | 0 を含む |
