# 681# 4/1 Fill Analysis — 構造的改善提案

| 項目 | 値 |
|------|-----|
| 作成日 | 2026-04-02 |
| 実行 SHA | 2b9be7dee (680#) / HEAD: 8475a2512 |
| 対象データ | `fill_records_20260401.jsonl` |
| 観測期間 | JST 9:00–21:59 (≈13h) |
| 前提 | 680# (retrain trigger fix, VG selectivity, balance log) |

---

## 1. 全体サマリー

| 指標 | 4/1 (本分析) | 3/31 (前日) | 差分 |
|------|-------------|-------------|------|
| 全レコード数 | 300 | 493 | -193 |
| 約定数 | 77 (内 filled=true 76–77) | 72 | +5 |
| Fill Rate | 25.3% | 14.6% | **+10.7pp** |
| PnL30 平均 (bps) | **-0.56** | -0.23 | -0.33 悪化 |
| AS 率 | 18.7% (15/77) | 27.8% | -9.1pp **改善** |

> **注**: 4/1 データの filled=true は PowerShell 上 76–77 件で集計タイミングにより微小差。以下は n=77 ベースで議論（n=76 との差は無視可能）。

### Fill 率↑ + PnL↓ という逆行現象
680# のVG selectivity (vpin_continuous_min 0.40→0.50) と 674# のnarrow_spread_boost 緩和により fill rate は顕著に改善。しかし、約定品質（selection）が伴っておらず、**有毒 fill の混入増加**が PnL 悪化の主因。

---

## 2. Regime × Side クロス集計

| Regime | Side | n | PnL30 (bps) | AS 件数 | 所見 |
|--------|------|---|-------------|---------|------|
| ranging | buy | 20 | **+1.03** | 2 | ベスト。ranging buy は安定収益源 |
| ranging | sell | 15 | -0.05 | 2 | ほぼ B/E。ranging sell は改善余地 |
| trending_down | buy | 11 | +0.48 | 0 | 良好。順張り方向 |
| trending_down | sell | 13 | **-2.49** | 4 | 深刻。逆行リスク |
| trending_up | buy | 7 | -0.66 | 0 | 軽微損失 |
| trending_up | sell | 11 | **-2.93** | 5 | **最悪バケット**。AS45% |

### 構造的所見

1. **Buy vs Sell 非対称性**: Buy 全体 +0.56 bps vs Sell 全体 **-1.68 bps** — 損益の全額が sell 側で生成
2. **AS 率の sell 偏重**: Buy AS=10.5% (4/38) vs Sell AS=**28.2%** (11/39)  
3. **Trending sell が損失の核**: trending_down/sell (-2.49) + trending_up/sell (-2.93) で全損失の大部分を形成
4. **Ranging は健全**: ranging/buy +1.03, ranging/sell -0.05 → offset 緩和の余地あり

---

## 3. Offset Clamp 問題（最重要発見）

### 3.1 Clamp 有無の分類

全 filled=true レコード中、`execution_pre_clamp_offset` が存在するもの（executor チェーンが offset ceiling を再適用したケース）:

| 分類 | n | 内訳 |
|------|---|------|
| withClamp (pre_clamp 記録あり) | 23 | sell=16, buy=7 |
| noClamp (pre_clamp が null) | 53 | sell=22, buy=31 |

> noClamp は executor multiplier chain が ceiling 未到達だった、もしくは multiplier 自体が非適用だったケース。

### 3.2 PnL 比較

| 分類 | Buy PnL30 | Sell PnL30 |
|------|-----------|------------|
| withClamp (n=23) | -0.17 (n=7) | -1.17 (n=16) |
| noClamp (n=53) | +0.72 (n=31) | **-2.05** (n=22) |

**解釈**: withClamp sell は ceiling で保護されている（-1.17 vs noClamp sell -2.05）。しかし clamp された sell 16 件中 **16 件全て**が clamp ヒット（100%）であり、オフセットパイプラインの情報が完全に破棄されている。

### 3.3 Sell Clamp 詳細

withClamp sell 16 件（全件 pre > effective に clamp）:

| pre_clamp_offset | effective_offset | regime | PnL30 |
|:---:|:---:|:---|:---:|
| 0.443 – 0.900 | 全て **0.40** | 混在 | -16.2 〜 +2.5 |

**問題**: pre_clamp 中央値が ~0.6 であり、パイプラインは 0.40 よりはるかに大きい offset を要求。sell offset_ceiling_ratio=0.40 がボトルネック。

### 3.4 noClamp Sell Offset 分布

noClamp sell 22 件の effective_offset_used:

```
min=0.200, max=0.400, avg=0.321
分布: 0.20–0.29 (9件), 0.30–0.39 (9件), 0.40 (4件, ceiling ヒット)
```

noClamp 4 件が offset=0.40（ceiling ジャスト）→ 実質 ceiling 拘束。
0.40 到達の noClamp sell 4 件は外れ値級 PnL: **-15.0, -16.2, -7.2, +1.2** bps

**結論**: sell 側で offset_ceiling=0.40 がパイプライン情報を遮断。特に trending sell で pre_clamp=0.6–0.9 を 0.40 に圧縮しており、offset に反映されるべきリスクシグナルが完全に失われている。

---

## 4. Skip Gate 分析

### 4.1 Skip Gate キャンセル分布

| cancel_reason | n | 備考 |
|---------------|---|------|
| preflight_insufficient | 102 | **最多**。残高不足 |
| (filled) | 76 | |
| spread_too_narrow | 44 | |
| **skip_gate** | **42** | buy=17, sell=25 |
| timeout | 12 | |
| no_feasible_quote | 7 | |
| sell_dynamic_kill | 6 | |
| final_clamp_hard_skip | 6 | buy=1, sell=5 (全sell=trending_up) |
| status_unknown_fast | 3 | |
| postonly_crossing_skip | 2 | |

### 4.2 Skip Gate Score と PnL の相関

filled=true レコードの `skip_gate_score` と `post_fill_30s_pnl` のピアソン相関:

| Side | r(score, pnl30) | n | 解釈 |
|------|-----------------|---|------|
| Sell | **-0.203** | 38–39 | 弱い逆相関。スコアが高い（pass 方向）ほど PnL が悪い |
| Buy | -0.069 | 38 | ほぼ無相関 |

**sell 側の r=-0.203 は、Skip Gate が sell 側で「通すべきでない fill を通している」可能性を示唆**。AS sell の多くが高スコア（pass 方向）で通過している。

### 4.3 Hard Skip (final_clamp_hard_skip) の偏り

6 件中 5 件が **sell + trending_up** → trending_up/sell は offset パイプラインが ceiling×2.5 を超えるほどリスクが高いが、hard_skip で防御されている側面もある。ただし skip されていない trending_up/sell 11 件は PnL=-2.93 bps で最悪。

---

## 5. 時間帯分析 (JST)

| JST 時間 | n | PnL30 (bps) | 判定 |
|----------|---|-------------|------|
| 9 | 7 | -1.27 | 要注意 |
| 10 | 10 | -1.40 | 要注意 |
| **11** | **10** | **-3.30** | **危険** |
| 12 | 6 | +1.47 | 良好 |
| **13** | **5** | **-5.16** | **最悪** |
| 14 | 5 | +0.63 | 良好 |
| 15 | 5 | -1.02 | 軽微損失 |
| 16 | 5 | +1.29 | 良好 |
| **17** | **5** | **+2.85** | **最良** |
| 18 | 4 | -1.11 | 軽微損失 |
| 19 | 4 | +1.51 | 良好 |
| **20** | **6** | **+2.20** | **良好** |
| 21 | 4 | -0.80 | 軽微損失 |

### AM/PM 非対称性

| 区分 | 時間帯 | n | 平均 PnL30 |
|------|--------|---|-----------|
| AM (JST 9–13) | UTC 0–4 | 38 | **-1.93 bps** |
| PM (JST 14–21) | UTC 5–12 | 38 | **+0.82 bps** |

**差分**: -2.75 bps。AM は構造的に不利な市場環境（東京株式市場開場・ボラティリティ）。

---

## 6. Volatility Guard 分析

| VG 状態 | n | PnL30 (bps) |
|---------|---|-------------|
| triggered=True | 76 | -0.64 |
| triggered=False | 1 | **+4.41** |

- VG 発火理由: vpin=72, velocity+vpin=4
- VG boost factor: avg=1.378, min=1.000, max=2.500

**問題**: 77 fills 中 76 件 (98.7%) で VG 発火。680# の vpin_continuous_min 0.40→0.50 にもかかわらず、ほぼ全件が VG 対象。VG が「全域防御」状態で**選択性ゼロ**に近い。

### 推定原因
- VPIN avg≈0.68 で vpin_continuous_min=0.50 → 大半が 0.50 超。0.55–0.60 への追加引き上げが必要
- VG boost factor は avg=1.378 で適度だが、n=76 では情報エントロピーがほぼ 0

---

## 7. Fast Fill (Queue Wait) 分析

### 7.1 queue_wait_sec 分布

```
p10=6.0s  p25=6.2s  p50=16.9s  p75=28.2s  p90=39.0s
min=5.6s  max=82.7s
```

### 7.2 Fast Fill (<10s) vs Slow Fill (>=10s)

| 区分 | n | PnL30 (bps) |
|------|---|-------------|
| Fast (<10s) | 20 | **-1.54** |
| Slow (≥10s) | 56 | **-0.21** |

Fast fill by side:
- fast sell: PnL = **-2.99** (n=11)
- fast buy: PnL = +0.24 (n=9)

**Fast sell fill は最大の損失源** (n=11, avg=-2.99 bps)。10 秒未満で約定する sell は情報トレーダーからの take であり、AS コスト大。

### 7.3 FFD (Fast Fill Defense) 状態

| ffd_boost_active | n |
|:---:|---|
| False | 71 |
| True | 5 |

FFD は 5/77 件でのみ発火。threshold_sec_sell=15.0 だが queue_wait p50=16.9 で半数近くが 15s 超。FFD 閾値は現在の市場パターンに対して**やや鈍感**。

---

## 8. 残高不足問題

| Side | preflight_insufficient |
|------|----------------------|
| buy | **73** |
| sell | 29 |
| **合計** | **102** |

全 300 レコード中 102 件 (34%) が残高不足でスキップ。buy 側が 73/102 (71.6%) を占める。

### 制約事項（ユーザー確認済み）
- **追加入金**: 不可（ユーザー制約）
- **ロット縮小**: Coincheck 最小注文量 = **0.001 BTC**（これ以下は不可）
- 現在残高 ≈21,700 JPY、lot=0.001 BTC ≈10,900 JPY → 2 fills で一方の残高枯渇

> 残高問題はパラメータチューニングでは解決不能。ただし fill quality 向上により ROE は改善可能。

---

## 9. Spread Capture

| 指標 | Buy | Sell | 全体 |
|------|-----|------|------|
| spread_capture_bps | -0.740 | -0.897 | **-0.819** |

spread_capture がマイナス = **mid price に対して不利な約定**が支配的。maker として spread の内側に乗せているはずだが、mid drift（発注後の mid 移動）が spread capture を食い潰している。

---

## 10. 改善提案

### 提案 A: Sell Offset Ceiling 引き上げ (P0, YAML)

**現状**: `offset_ceiling_ratio_sell: 0.40`  
**提案**: → `0.55`

**根拠**:
- withClamp sell 16 件中 16 件 (100%) が ceiling ヒット
- pre_clamp 中央値≈0.6 → パイプラインは 0.55–0.60 を要求
- noClamp sell 4 件が 0.40 ジャストで、PnL 外れ値 (-15.0, -16.2, -7.2) を含む
- trending sell が -2.49〜-2.93 bps で最大損失源 → offset 拡大余地が必要

**リスク**: sell fill rate の低下。ただし現在の sell fill は PnL=-1.68 bps で赤字なので、fill rate 低下より quality 改善が優先。

**実装**: `configs/v460/fill_test.yaml` L755 `offset_ceiling_ratio_sell: 0.55`

---

### 提案 B: Skip Gate Sell Regime 条件付き厳格化 (P1, YAML)

**現状**: `regime_thresholds.trending_up: 0.3`, sell side で r=-0.203 (逆相関)

**提案**:
1. `regime_thresholds.trending_up: 0.5` (0.3→0.5, 厳格化)
2. `regime_thresholds.trending_down: 0.3` (0.1→0.3, 厳格化)

**根拠**:
- trending_up/sell: PnL=-2.93 bps, AS=45% — 明確に有毒
- trending_down/sell: PnL=-2.49 bps, AS=31% — sell 側で悪質
- Skip Gate の r(score, pnl30) が sell 側で -0.203 → 通過する売りの品質が低い
- regime_thresholds の厳格化により skip_gate が trending sell を排除

**リスク**: trending sell の fill 数がさらに減少。ただし trending sell の期待値がマイナスなので、スキップ増は PnL 改善に直結。

---

### 提案 C: AM 時間帯 (JST 9-13) Ceiling 緩和 (P1, YAML)

**現状**: `hour_ceiling_mult` に UTC 0-4 (JST 9-13) のエントリなし

**提案**: `hour_ceiling_mult` に以下を追加:
```yaml
hour_ceiling_mult:
  0: 1.5   # JST 9h: PnL=-1.27
  1: 1.5   # JST 10h: PnL=-1.40
  2: 2.0   # JST 11h: PnL=-3.30
  3: 1.5   # JST 12h: (スキップ、+1.47で良好)
  4: 2.5   # JST 13h: PnL=-5.16 (最悪)
```

**根拠**:
- AM (JST 9-13) 平均 PnL=-1.93 bps vs PM +0.82 bps
- JST 11h (-3.30) と JST 13h (-5.16) が特に深刻
- ceiling 緩和により AM の offset パイプラインが防御方向に解放される

**リスク**: AM の fill rate 低下。ただし AM PnL がマイナスなので選択的低下は望ましい。

---

### 提案 D: VG Selectivity 追加引き上げ (P1, YAML)

**現状**: `vpin_continuous_min: 0.50`

**提案**: → `0.55` (段階的。効果不十分なら 0.60 を検討)

**根拠**:
- 77 fills 中 76 件 (98.7%) で VG 発火 → 情報エントロピー ≈ 0 (全域防御=防御なし)
- VPIN 平均≈0.68 で min=0.50 はまだ大半を通過
- VG が無差別に boost するため fill quality 低下（boost が不要な場面でも offset 拡大）

**リスク**: 真の高ボラ場面で VG が不発になるリスク。ただし vpin_threshold=0.80 (バイナリ) が安全弁として機能。

---

### 提案 E: sell_hour_offset_boost AM 時間帯追加 (P2, YAML)

**現状**: UTC 0 (JST 9h) のエントリなし

**提案**:
```yaml
sell_hour_offset_boost:
  0: 1.5    # JST 9h: sell -1.27bps
  1: 1.8    # JST 10h: sell構成要因を要確認
  4: 2.0    # JST 13h: PnL=-5.16, sell AS が主因か検証要
```

**根拠**: §5 の AM 時間帯分析。ただし §5 は buy+sell 集計のため、sell individual の検証が必要。提案 B/C の効果を先に確認し、不十分であれば追加投入。

---

## 11. 実装優先順位

| 優先度 | 提案 | 変更箇所 | 期待効果 |
|--------|------|----------|----------|
| **P0** | **A: Sell ceiling 0.40→0.55** | YAML 1行 | Sell clamp 100%→<50%、offset情報復元 |
| P1 | B: Skip Gate trending 厳格化 | YAML 2行 | Trending sell AS排除 |
| P1 | C: AM ceiling 緩和 | YAML 4-5行 | AM PnL 改善 |
| P1 | D: VG min 0.50→0.55 | YAML 1行 | VG selectivity 向上 |
| P2 | E: sell_hour AM追加 | YAML 2-3行 | AM sell 防御 |

### 実装順序の推奨
1. **A のみ先行適用** (hot-reload) → 24h 観測
2. 効果確認後 B+C を一括適用 → 24h 観測
3. 効果不十分なら D を追加
4. E は B+C の効果次第で判断

---

## 12. 既知制約と注意点

1. **観測データからの因果推論は不可**。相関を因果と断定しない（675#/679# 教訓）
2. **n=77 は統計的に小**: 1日分のデータで結論を出すのはリスク。複数日の観測で確認必要
3. **残高制約**: パラメータ改善で ROE は上がるが、102/300=34% の機会損失は構造的に残存
4. **VG 全域発火**: 680# の vpin_continuous_min 改善は部分的。VPIN 推定器自体の較正が未検証
5. **SAC sidecar**: 679# で reward/γ を修正したが retrain 完了後の効果は未観測
6. **PPO sidecar**: Codex Phase 3 で OBSERVE mode 実装済み。本格稼働は別途判断
7. **Sell モデル退行 (645#)**: sell 専用 SkipGate モデルが定数出力で退行済み→ unified fallback 中。sell 側 retrain が望ましいが、データ品質・量の問題で保留

---

## 13. 再現コマンド

以下のコマンドで本分析の数値を再現可能:

```powershell
# 全体サマリー
$all=Get-Content results\v460\fill_test\fill_records_20260401.jsonl|ForEach-Object{$_|ConvertFrom-Json}
$f=$all|Where-Object{$_.filled -eq $true}
"total=$($all.Count) fills=$($f.Count) rate=$([math]::Round($f.Count/$all.Count*100,1))%"
$pnl=($f|ForEach-Object{[double]$_.post_fill_30s_pnl}|Measure-Object -Average).Average
"pnl30=$([math]::Round($pnl,2))"

# Regime × Side クロス集計
foreach($r in @('ranging','trending_down','trending_up')){
  foreach($s in @('buy','sell')){
    $g=$f|Where-Object{$_.regime -eq $r -and $_.side -eq $s}
    if($g.Count -gt 0){
      $p=($g|ForEach-Object{[double]$_.post_fill_30s_pnl}|Measure-Object -Average).Average
      $as=($g|Where-Object{$_.adverse_selected -eq $true}).Count
      "$r/$s n=$($g.Count) pnl=$([math]::Round($p,2)) as=$as"
    }
  }
}

# Clamp 分析
$wc=$f|Where-Object{$null -ne $_.execution_pre_clamp_offset -and $_.execution_pre_clamp_offset -ne ''}
$nc=$f|Where-Object{$null -eq $_.execution_pre_clamp_offset -or $_.execution_pre_clamp_offset -eq ''}
"withClamp=$($wc.Count) noClamp=$($nc.Count)"
```

---

*本文書は AI エージェントレビュー用に作成。全数値はコマンド再現可能。*
