# 696# 多視点フィルテスト分析レポート（深堀り版）

## AIエージェントレビュー用

### 分析の目的
1. `sell_hour_offset_boost` 2.5 → 5.0 の効果検証（4/2 適用）
2. `entry_gate` ランタイム稼働状況確認（691#/693# 実装）
3. `skip_gate_bypass_mode` 下での挙動検証
4. 収益性改善の構造的要因の特定
5. **追加**: 因果分離の試行、risk-adjusted 評価、regime×spread 交差分析

### データ概要
- **比較期間**: baseline 3日間 (3/29-3/31), 4/1 (boost=2.5), 4/2 (boost=5.0, partial ~12h)
- **ボット SHA**: `b5f7828b1` → `04390da32` (4/2 起動 07:54 JST)
- **構成変更**: entry_gate_enabled=true, skip_gate_bypass_mode=true, sell_hour_boost UTC2/4: 5.0

### 再現コマンド
```bash
python scripts/v460/analysis/analyze_694_multi_perspective.py
```

---

## 視点1: 全体パフォーマンス比較

| 指標 | Baseline 3日 | 4/1 | 4/2 (boost=5.0) | 4/1→4/2 変化 |
|------|-------------|-----|-----------------|-------------|
| Total records | 1607 | 502 | 204 | - |
| Fill rate | 9.8% | 27.5% | **47.5%** | **+20.0pp** |
| Mean PnL (bps) | -0.388 | -0.306 | **+0.367** | **+0.673** |
| Median PnL | -0.053 | +0.023 | +0.030 | +0.007 |
| PnL stdev | 4.786 | 4.879 | 4.438 | -0.441 |
| p10 | -5.198 | -4.985 | -5.766 | **-0.781** |
| p90 | +4.476 | +5.339 | +4.575 | -0.764 |
| Positive rate | 49.4% | 50.7% | 50.5% | -0.2pp |

**解釈**: fill rate が大幅改善（+20pp）は主に `preflight_insufficient` の激減（45.6%→2.8%）と `skip_gate` バイパスによる。PnL mean が正転（+0.367 bps）は有意だが partial day のため統計的頑健性は限定的。

**批判的観点**:
- 4/2 は 204 records のみ（4/1 の 40%）。時間帯バイアスが強い（UTC 0-10 に集中）
- p10 が悪化（-5.766 vs -4.985）は**尾部リスク増加の兆候**
- positive rate が 50.5% でほぼ変わらないのに mean PnL が改善 → **正方向の tail が伸びた**のではなく**負方向の平均影響が軽減**されたことを示唆
- stdev 改善は母数減少による偶然の可能性あり

### 深堀り: Sharpe 近似比較

| 期間 | Mean | Stdev | Sharpe 近似 (mean/stdev) |
|------|------|-------|--------------------------|
| Baseline | -0.388 | 4.786 | **-0.081** |
| 4/1 | -0.306 | 4.879 | **-0.063** |
| 4/2 | +0.367 | 4.438 | **+0.083** |

Sharpe が負から正に転換。しかし n=97 fills での 95% CI は ±0.90 bps（$SE = 4.438/\sqrt{97} \approx 0.451$, $CI = \pm 1.96 \times 0.451 = \pm 0.884$）。つまり真の mean PnL は $[-0.52, +1.25]$ bps の範囲にあり、**正転の統計的有意性は確認できない**（p > 0.05）。

---

## 視点2: Side 別分析

| 指標 | 4/1 BUY | 4/1 SELL | 4/2 BUY | 4/2 SELL |
|------|---------|----------|---------|----------|
| n | 68 | 70 | 50 | 47 |
| Mean PnL | +0.776 | -1.357 | +0.542 | **+0.181** |
| Median PnL | - | - | - | - |
| p10 | -3.700 | -7.641 | -3.331 | -6.568 |
| AS rate | 39.7% | 58.6% | 46.0% | 53.2% |

**解釈**:
- **SELL PnL が正転**（-1.357 → +0.181 bps）は sell_hour_boost 効果の可能性が高い
- BUY PnL は微減（+0.776 → +0.542）だが依然正
- SELL AS rate が改善 (58.6% → 53.2%) は offset 拡大による AS 回避効果

**批判的観点**:
- BUY AS rate は**悪化**（39.7% → 46.0%, +6.3pp）。buy-side cross-venue protection 不在の影響か
- BUY p10 は微改善（-3.700 → -3.331）なのに AS 悪化 → **AS の深刻度（severity）が軽減**された可能性
- SELL p10 改善（-7.641 → -6.568）は 1.07bps の改善だが依然として-6.5bps は深刻

### 深堀り: Side 別 PnL の構造分解

**BUY** の PnL 減少（+0.776 → +0.542）を分解:
- AS rate 悪化 +6.3pp → AS trade の増加が mean を引き下げ
- しかし p10 改善 → AS 被害の個別深度は軽減
- skip_gate bypass により低品質 BUY trade が通過した可能性

**SELL** の PnL 正転（-1.357 → +0.181）を分解:
- sell_hour_boost = 5.0 が UTC4 で劇的効果（後述 視点5）
- trend_5s_sell_guard_veto が最悪 SELL を除外（14% cancel）
- AS rate 改善 -5.4pp → offset 拡大で informed flow を回避

---

## 視点3: Cancel Reason の構造変化

| 理由 | 4/1 (% of cancels) | 4/2 (% of cancels) | 変化 | 4/2 実数 |
|------|---------------------|---------------------|------|----------|
| preflight_insufficient | 45.6 | **2.8** | -42.8pp | 3 |
| timeout | 6.9 | **27.1** | +20.2pp | 29 |
| spread_too_narrow | 19.0 | 16.8 | -2.2pp | 18 |
| trend_5s_sell_guard_veto | 0.8 | **14.0** | +13.2pp | 15 |
| sell_dynamic_kill | 3.8 | 8.4 | +4.6pp | 9 |
| final_clamp_hard_skip | 1.9 | 8.4 | +6.5pp | 9 |
| mcb_halt | 0.3 | **8.4** | +8.1pp | 9 |
| entry_gate_ev_negative | 0.0 | **4.7** | +4.7pp | 5 |
| stale_adverse_drift | 0.3 | 4.7 | +4.4pp | 5 |
| postonly_crossing_skip | 0.8 | 3.7 | +2.9pp | 4 |
| skip_gate | 16.8 | **0.0** | -16.8pp | 0 |
| cross_venue_lead_lag_veto | 0.5 | 0.0 | -0.5pp | 0 |
| mcb_sad_escalation | 0.0 | 0.9 | +0.9pp | 1 |

**解釈**: cancel reason の構造が**根本的に変わった**。preflight_insufficient → timeout + guards として再配分。

**批判的観点**:
- `trend_5s_veto` 14% は過剰？ veto 対象の counterfactual PnL が不明
- `final_clamp_hard_skip` 8.4% は offset 拡大による quote 妥当性問題の兆候
- `mcb_halt` 8.4% 増加は市場ボラティリティ要因か構造的問題か要分離

### 深堀り: Cancel 構造の因果チェーン

preflight_insufficient の激減（166→3）の原因:
1. 4/1 では skip_gate が 61 件 cancel → その前に preflight が catch
2. 4/2 で skip_gate bypass → preflight の出番が減少
3. **仮説**: preflight_insufficient は「skip_gate 以前に落ちた」のではなく、skip_gate による時間消費後に balance check が通らなかった可能性。bypass で高速化 → balance が間に合う

mcb_halt の増加（1→9）:
- 4/2 UTC 時間帯分布を見ると、MCB halt は特定時間に集中している可能性
- ボラティリティ由来なら一時的、構造的なら設定調整が必要
- **要追加調査**: MCB halt の時刻分布と市場ボラティリティの相関

---

## 視点4: Spread Bucket 分析

| Bucket (JPY) | Baseline n | Baseline AS | 4/1 n | 4/1 AS | 4/2 n | 4/2 AS | 4/2 PnL |
|--------------|-----------|-----------|-------|--------|-------|--------|---------|
| 0-1500 | 9 | 44.4% | 5 | 20.0% | **14** | **64.3%** | **-0.461** |
| 1500-2500 | 42 | 40.5% | 42 | 47.6% | **47** | 42.6% | +0.240 |
| 2500-3500 | 72 | 56.9% | 72 | 47.2% | **32** | 56.2% | +0.637 |
| 3500+ | 35 | 51.4% | 19 | 68.4% | **4** | 25.0% | +2.610 |

**解釈**:
- **0-1500 bucket で AS 急悪化**（20%→64%）。min_spread_atr_cap 緩和 (686#) で狭スプレッド参入増 → 逆選択リスク急増
- 1500-2500 が「スイートスポット」（n=47, PnL +0.24, AS 42.6%）
- 3500+ は n=4 に激減 → 市場スプレッドが縮小傾向に

**批判的観点**:
- 0-1500 bucket n=14 で AS=64% は**深刻なリスクシグナル**
- 686# min_spread_atr_cap 2.0→1.2 の効果が**逆効果**の可能性
- 4/1 で 0-1500 AS=20% だったのは skip_gate が狭スプレッドの bad trade を選別していたから？

### 深堀り: Spread bucket の fill 数変化メカニズム

| Bucket | 4/1 n | 4/2 n | 変化 | 変化率 |
|--------|-------|-------|------|--------|
| 0-1500 | 5 | 14 | +9 | **+180%** |
| 1500-2500 | 42 | 47 | +5 | +12% |
| 2500-3500 | 72 | 32 | -40 | -56% |
| 3500+ | 19 | 4 | -15 | -79% |

0-1500 の fill 数が +180% 増加。これは min_spread_atr_cap 緩和で spread filter を通過するようになったため。しかし AS=64% は「通すべきでなかった」ことを示唆。

**計量的リスク評価**: 0-1500 bucket の期待 PnL = -0.461 bps × 14 fills = **-6.45 bps 累計損失**。
仮にこの bucket を全拒否していたら mean PnL は $(0.367 \times 97 - (-0.461) \times 14) / (97 - 14) = (35.6 + 6.45) / 83 = +0.507$ bps に改善。

**結論**: 0-1500 bucket のフィルタリング or offset 拡大は**即座に着手すべき**。

---

## 視点5: sell_hour_boost 効果

| UTC 時間 | 4/1 sell fills | 4/1 PnL | 4/2 sell fills | 4/2 PnL | 4/2 AS rate |
|----------|---------------|---------|---------------|---------|-------------|
| UTC2 (JST 11) | 6 | -5.191 | 1 | -0.505 | 100% |
| UTC4 (JST 13) | 3 | -8.853 | 7 | **+3.506** | **14.3%** |

**解釈**:
- **UTC4 が劇的改善**: -8.853 → +3.506 bps、AS rate 100% → 14.3%
- boost=5.0 による wide spread が informed flow を回避

**批判的観点**:
- UTC4 の改善は n=7 vs n=3 で統計的に弱い
- UTC2 の sell fill が 6→1 に**激減**。boost=5.0 が aggressive すぎて fill しにくくなった可能性
- UTC の時間帯効果と boost 効果の分離ができていない（交絡要因）

### 深堀り: boost=5.0 の fill rate 影響

UTC2 で sell fills が 6→1 の理由:
- boost=5.0 は sell offset を 5 倍に拡大 → quote price が大幅乖離
- fill するには market が大きく動く必要 → fill 機会損失
- **機会損失推定**: 失った 5 fills の 4/1 平均 PnL = -5.191 bps（全損）
  - 仮に 4/2 条件下でも同程度の損失なら、逃した損失 = 5 × 5.191 = 25.96 bps
  - boost 効果で 1 fill が -0.505（マイナス縮小）
  - **net**: 機会損失<損失回避 → UTC2 での boost=5.0 は**正しい判断**の可能性が高い

UTC4 での fill 数増加（3→7）:
- boost=5.0 でも fill する = **市場が十分に動いた**
- AS rate 14.3% は通常の半分以下 → wide spread が quality filter として機能

---

## 視点6: entry_gate 稼働状況

| 指標 | 4/1 | 4/2 |
|------|-----|-----|
| entry_gate blocked (ev_negative) | 0 | **5** |
| skip_gate suppressed (bypassed) | 9 | **27** |
| suppressed / total records | 1.8% | 13.2% |

**解釈**: entry_gate が 5 件 EV negative でブロック。27 件の skip_gate bypass は「ML がスキップ判定したが bypass_mode で通過」。

**批判的観点**:
- entry_gate blocked = 5/204 = 2.5% は少なく、ゲートとしての寄与は限定的
- 693# staleness fix 前は auto_disable で entry_gate が無効化されていた可能性

### 深堀り: entry_gate blocked 5 件の counterfactual 価値

entry_gate が block した 5 件が仮に通過していたら？
- cancel reason = `entry_gate_ev_negative` → EV < 0 と推定された注文
- これらの counterfactual PnL は計測不能（fill されていない）
- しかし EV negative = 期待損失のある注文 → block は正しい可能性が高い
- **推定 impact**: 5 件 × 仮平均損失 -2.0 bps = 10 bps の損失回避（粗い推定）

entry_gate の次段階:
- observe mode → enabled=true + block で稼働中だが、EV threshold の最適化が未実施
- CalibrationMap のキャリブレーション精度が entry_gate の有効性を決定
- **提案**: EV threshold をパラメトリックに探索（-0.5, 0.0, +0.5 bps）

---

## 視点7: Regime 別分析

| Regime | 4/1 n | 4/1 PnL | 4/1 AS | 4/2 n | 4/2 PnL | 4/2 AS |
|--------|-------|---------|--------|-------|---------|--------|
| ranging | 63 | +0.600 | 36.5% | 46 | **-0.362** | **56.5%** |
| trending_up | 32 | -1.771 | 65.6% | 14 | **+0.689** | 57.1% |
| trending_down | 43 | -0.543 | 55.8% | 37 | **+1.152** | **37.8%** |

**解釈**:
- **trending_down が大幅改善** (-0.543 → +1.152, AS 55.8%→37.8%)。sell_hour_boost でトレンド方向に逆らう sell のoffset が拡大 → quality 改善
- **ranging PnL 悪化** (+0.600 → -0.362) かつ **AS 増加** (36.5%→56.5%)
- trending_up も改善 (-1.771 → +0.689) だが n=14 で信頼性低

**批判的観点**:
- ranging は最大母集団（n=46/97=47.4%）→ 全体影響が最大
- ranging の悪化は skip_gate bypass の直接的影響か？

### 深堀り: Ranging regime 悪化の因果分析

**仮説 A**: skip_gate bypass が原因
- 4/1: skip_gate が ranging × low-quality trade を 61 件 cancel
- 4/2: bypass で全通過 → ranging AS rate 上昇
- **検証方法**: 4/1 の skip_gate cancelled trades の regime 分布を確認（データあれば）

**仮説 B**: ranging × 0-1500 spread の交差効果
- 0-1500 bucket で AS=64.3%。ranging regime は spread が狭い時間帯に集中する傾向
- ranging × 0-1500 の交差セルが全体を引き下げている可能性

**仮説 C**: 市場状態の変化
- 4/2 のUTC 0-10 は ranging が多い時間帯。4/1 は 24h データ
- 時間帯バイアスにより ranging の比重が高くなっただけの可能性

**推定 impact**: ranging の PnL 悪化分 = $(-0.362 - 0.600) \times 46 = -44.3$ bps 累計。
trending_down の改善分 = $(1.152 - (-0.543)) \times 37 = +62.7$ bps 累計。
trending_up の改善分 = $(0.689 - (-1.771)) \times 14 = +34.4$ bps 累計。
**Net regime shift**: +62.7 + 34.4 - 44.3 = **+52.8 bps 累計改善**。trending 改善が ranging 悪化を上回る。

---

## 視点8: 時間帯別分析（Hourly）

### 4/2 UTC 時間帯別（n, AS rate, mean PnL）

| UTC | n | AS rate | PnL (bps) | 評価 |
|-----|---|---------|-----------|------|
| 0 | 8 | 37.5% | +2.472 | 優良 |
| 1 | 1 | 0.0% | +3.917 | (n=1) |
| 2 | 3 | 33.3% | +0.519 | 良好 |
| 3 | 14 | **64.3%** | -0.327 | 要注意 |
| 4 | 14 | **21.4%** | **+2.475** | **最良** |
| 5 | 7 | 71.4% | -0.691 | 危険 |
| 6 | 12 | **66.7%** | -0.589 | 危険 |
| 7 | 7 | 57.1% | -1.804 | 危険 |
| 8 | 8 | 37.5% | +0.801 | 良好 |
| 9 | 11 | 45.5% | +0.295 | 中立 |
| 10 | 12 | 58.3% | -0.402 | 要注意 |

### 深堀り: 時間帯クラスタリング

**良好時間帯** (PnL > 0, AS < 50%): UTC 0, 2, 4, 8
- 共通点: sell_hour_boost が効いている or 市場が穏やか
- UTC4 は boost=5.0 の直接効果

**危険時間帯** (AS > 60%): UTC 3, 5, 6, 7
- UTC 5-7 はアジア市場開場時間（JST 14-16）。informed flow が増加する時間帯
- UTC 3（JST 12）もランチタイム前後の thin market
- **提案**: UTC 5-7 での追加 offset or 参入制限の検討

### 4/1 との時間帯比較（重複する時間帯のみ）

| UTC | 4/1 PnL | 4/2 PnL | 4/1 AS | 4/2 AS | 改善? |
|-----|---------|---------|--------|--------|-------|
| 0 | -1.267 | +2.472 | 42.9% | 37.5% | 改善 |
| 2 | -3.302 | +0.519 | 70.0% | 33.3% | 大幅改善 |
| 3 | +1.470 | -0.327 | 16.7% | 64.3% | **悪化** |
| 4 | -5.156 | +2.475 | 60.0% | 21.4% | **劇的改善** |
| 5 | +0.627 | -0.691 | 60.0% | 71.4% | 悪化 |
| 6 | -1.021 | -0.589 | 60.0% | 66.7% | 微改善 |
| 7 | +1.290 | -1.804 | 60.0% | 57.1% | **悪化** |
| 8 | +2.850 | +0.801 | 60.0% | 37.5% | AS改善/PnL悪化 |
| 9 | -1.110 | +0.295 | 75.0% | 45.5% | 改善 |
| 10 | +1.506 | -0.402 | 25.0% | 58.3% | **悪化** |

UTC 3, 5, 7, 10 で悪化。**これらは boost=5.0 が直接関係しない時間帯**であり、skip_gate bypass or 市場条件の変化が原因の可能性。

---

## 視点9: 構造的因果の推定（注意: 観察データのため因果推論は限定的）

### fill rate +20pp の因果分解（推定）

| 要因 | 推定寄与 | 根拠 |
|------|----------|------|
| preflight_insufficient 解消 | +10-15pp | 166→3 cancel reduction |
| skip_gate bypass | +5-8pp | 61→0 cancel reduction |
| 市場条件の差異 | ±2-5pp | 4/2 はスプレッド分布が異なる |
| entry_gate block | -1pp | 5 件 block |

### PnL +0.67bps の因果分解（推定）

| 要因 | 推定寄与 (bps) | 確信度 |
|------|----------------|--------|
| sell_hour_boost 5.0 (UTC4) | +0.3-0.5 | 中 |
| trending_down 改善 | +0.1-0.2 | 中 |
| trend_5s_sell_guard_veto | +0.05-0.15 | 低 |
| entry_gate EV negative block | +0.05-0.1 | 低 |
| ranging 悪化（相殺） | -0.1-0.3 | 中 |
| 0-1500 spread AS 増（相殺） | -0.05-0.1 | 中 |

**注意**: これらは**相関ベースの推定**であり、因果ではない。4/1 と 4/2 の間には複数の同時変更（boost=5.0, bypass_mode, entry_gate 稼働）があり、各効果の分離は厳密には不可能。

---

## 視点10: Risk-Adjusted 評価と統計的頑健性

### Sortino 比近似

| 期間 | Mean | Downside Dev (p10 base) | Sortino 近似 |
|------|------|------------------------|--------------|
| Baseline | -0.388 | 5.198 | **-0.075** |
| 4/1 | -0.306 | 4.985 | **-0.061** |
| 4/2 | +0.367 | 5.766 | **+0.064** |

Sortino も正転だが**有意ではない**。特に downside deviation が 4/2 で最大（5.766）なのは、尾部リスクが意味するものが変わった（fill 数増加に伴う tail exposure）。

### Bootstrap 95% CI 推定（正規近似）

| 期間 | n | Mean | SE | 95% CI |
|------|---|------|----|--------|
| 4/1 | 138 | -0.306 | 0.415 | [-1.12, +0.51] |
| 4/2 | 97 | +0.367 | 0.451 | [-0.52, +1.25] |

CI が重複 → 4/1 と 4/2 の差は**統計的に有意ではない**。ただし方向性は一貫して改善。

### 必要サンプル数の推定

現在の effect size = 0.673 bps、pooled stdev ≈ 4.7 bps の場合:
$n = (2.8 \times 4.7 / 0.673)^2 \approx 385$ (per group, power=0.8)

95% 信頼で効果を確認するには**各群 385 fills**（現在の 4 倍）が必要。

---

## 改善提案（優先度順）

### 即座に着手（P0）
1. **0-1500 spread bucket AS 防御** — 狭スプレッド時の追加ガード or offset 拡大（AS 64%, -0.461 bps）
2. **trend_5s_veto counterfactual** — 14% cancel の価値検証。threshold 0.5→0.3 or 0.8 の探索

### 1 週間以内（P1）
3. **ranging regime AS rate 悪化の根因分析** — regime × spread × skip_gate の三重交差分析
4. **UTC 5-7 時間帯防御** — AS > 60% 時間帯の追加 offset or 参入制限検討
5. **entry_gate EV threshold 最適化** — 現在は binary（positive/negative）、段階的 threshold へ

### 長期改善（P2）
6. **fill record observability 強化** — guard pipeline 判断メタデータの記録
7. **A/B テスト基盤** — boost 効果と market 効果の因果分離
8. **mcb_halt 頻度の原因解析** — 0.3% → 8.4% の構造的要因の特定

---

## 生データ参照
- `analysis_results/694_multi_perspective_analysis.json`
- `results/v460/fill_test/fill_records_20260401.jsonl` (502 records)
- `results/v460/fill_test/fill_records_20260402.jsonl` (204 records)
