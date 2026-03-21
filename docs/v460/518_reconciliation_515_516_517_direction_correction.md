# 518# 方向修正 — 515#/516#/517# 三文書検証と統合分析

> **作成**: 2026-03-21 16:00 JST
> **対象 run**: `1774021842_804f05db` (SHA: `20d4f778ef67`, 332 cycles / 111 fills)
> **PnL30**: −75.6 JPY

## §1 結論要旨

**515# の5つの根本原因仮説のうち、§3.4 (sell_dynamic_kill) と §3.5 (FFD) は事実誤認。§3.2 (XV cascade) は因果関係が逆。§3.1 (EV amplifier) と §3.3 (sell_offset) は部分的に正しいが、根本原因の特定に失敗している。**

516# (Codex) と 517# (PHG) による外部レビューは複数の誤認を正しく指摘したが、両者とも独自の事実誤認を含む。

**本ドキュメントが新たに特定した真の構造的原因:**

| 発見 | 数値 |
|------|------|
| **native sell (offset_ceiling=0.20) が全損失の源泉** | 25 fills, PnL30=**−87.4** (全損失 75.6 の 116%) |
| **offset pipeline が正しくリスクを検出 → ceiling が override** | 91/111 fills (82%) でPL出力 > ceiling |
| **sell ceiling 0.20 が sell AS を増幅** | native sell AS: 13 fills, avg=**−7.31** (forced sell AS: avg=−4.42) |
| **buy 側は健全** | 45 fills, PnL30=**+39.0** |
| **balance_switch 由来は黒字** | 47 fills, PnL30=**+26.0** |

---

## §2 三文書事実検証マトリクス

### §2.1 515# の主張と検証結果

| # | 515# 主張 | 検証結果 | 判定 |
|---|----------|---------|------|
| §3.1 | EV offset が逆選択増幅器 | EV mult 平均 0.96 (ほぼ中立)。EV[2,5) bucket は確かに悪いが、ceiling cap が支配的で EV mult の影響は副次的 | **△ 部分的** |
| §3.2 | XV basis_correction → no_feasible_quote cascade (67件) | **67/67 は sell-side の balance_switch が原因**。XV vetoed = 0/67。cascade 仮説は **完全に誤り** | **✗ 否定** |
| §3.3 | sell_offset 0.14 が損失バケット | base=0.14 でも pipeline が 0.274 まで押し上げ → ceiling=0.20 で cap。**実効 offset は 0.19-0.20 であり 0.14 は実効値ではない** | **△ 間接的** |
| §3.4 | sell_dynamic_kill が 0 cancels で死亡 | **4 cancels を実際に確認**。完全不活性は事実誤認 | **✗ 否定** |
| §3.5 | FFD/VG が 0% 稼働 | **FFD boost active = 12 records (10 filled)**。0% は事実誤認 | **✗ 否定** |

### §2.2 516# (Codex) の主張と検証結果

| # | 516# 主張 | 検証結果 | 判定 |
|---|----------|---------|------|
| 1 | FFD は稼働中 (12 boost records) | **正しい**。10 filled, 5 AS, 5 non-AS | **○ 正確** |
| 2 | sell_dynamic_kill は稼働中 | **正しい**。4 cancels | **○ 正確** |
| 3 | recovery_skew: config=1.5 だが runtime=2.0 | config は 1.5。ログ検証で runtime 2.0 の証拠なし | **△ 未確認** |
| 4 | 「buy: no_feasible_quote=67」| **no_feasible_quote 67/67 は全て SELL-side**。516# のサイド帰属は**完全に逆** | **✗ 重大誤認** |
| 5 | 「buy over-filtering + sell toxic fill の二重問題」 | buy は PnL30=+39.0 で健全。over-filtering ではなく適切に機能 | **✗ 誤認** |
| 6 | sell_offset 0.14 は単一犯人ではない (pipeline 飽和) | **正しい**。ceiling cap が支配的 | **○ 正確** |

### §2.3 517# (PHG) の主張と検証結果

| # | 517# 主張 | 検証結果 | 判定 |
|---|----------|---------|------|
| 1 | 515# は事実誤認を含み信頼できない | FFD/SDK の 0% 主張は確かに誤り。正しい指摘 | **○ 正確** |
| 2 | sell AS 23 fills (−139.2) が単一原因 | **概ね正しい**。ただし sell AS の内訳に native vs forced の区別が必要 | **○ 概ね正確** |
| 3 | FFD は 12 boost records で稼働中 | **正しい** | **○ 正確** |
| 4 | 防御メカニズムの再構築は不要 | FFD は稼働しているが AS 防止に失敗 (5/10 AS)。「再構築不要」は部分的にのみ正しい | **△ 部分的** |
| 5 | sell offset 0.14 が原因 | 実効 offset は 0.19-0.25 であり 0.14 は base 値に過ぎない。**真因は ceiling cap** | **✗ 不正確** |

---

## §3 新発見: Sell Ceiling Cap が真の構造的原因

### §3.1 Offset Pipeline vs Ceiling の衝突

```
offset_pipeline 出力 (全 sell fills の典型):
  base=0.14 → regime(0.09→0.23) → spread_adapt(0.23) → kyle(0.23)
  → vol_guard(0.27) → ffd(0.27) → final=0.274

ceiling 適用:
  offset_ceiling_ratio_sell = 0.20  ← 474# で 0.50→0.20 に引き下げ
  final(0.274) > ceiling(0.20) → CAPPED to 0.20

EV mult 適用:
  effective_offset_used = ceiling × ev_mult = 0.20 × 0.96 = 0.192
```

**91/111 fills (82%) で pipeline 出力が ceiling を超過してキャップされている。**
Pipeline のリスク評価は正しいが、ceiling が上書きする。

### §3.2 Ceiling 別の PnL 帰属

| カテゴリ | ceiling | n fills | PnL30 | avg |
|---------|---------|---------|-------|-----|
| **Native sell** | **0.20** | **25** | **−87.4** | **−3.50** |
| Forced sell (balance_switch + recovery_skew) | 0.25 | 41 | −27.3 | −0.67 |
| Buy (all) | 0.25 | 45 | +39.0 | +0.87 |

**Native sell (ceiling=0.20) は forced sell (ceiling=0.25) の 5.2 倍悪い** (avg −3.50 vs −0.67)。

### §3.3 Sell AS の ceiling 別分解

| カテゴリ | AS fills | AS PnL30 | AS avg |
|---------|----------|----------|--------|
| **Native sell AS (ceil=0.20)** | **13** | **−95.0** | **−7.31** |
| Forced sell AS (ceil=0.25) | 10 | −44.2 | −4.42 |

Native sell AS は forced sell AS より **65% 多い損失** (avg −7.31 vs −4.42)。
ceiling=0.20 → 0.25 の 5% 差が avg PnL を 2.89 JPY/fill 改善する可能性。

### §3.4 なぜ ceiling=0.20 が 474# で設定されたか

```yaml
# 474# sell: 0.50→0.20 (473#検証: ratio=0.50→mid+0, 936JPY逆行に耐えられず。
#   0.20→mid+718JPY buffer)
offset_ceiling_ratio_sell: 0.20
```

474# は ceiling=0.50 で mid+0 になる（offset が大きすぎてスプレッド内に収まらない）問題を修正した。しかし 0.20 はパイプライン出力 (0.274) を大幅にカットしている。

**Buy ceiling は 491# で 0.20→0.25 に引き上げ済み** (`offset_ceiling_ratio_buy: 0.25`)。Sell はまだ 0.20 のまま。

---

## §4 no_feasible_quote 67 件の真の原因

515# は「XV veto cascade」、516# は「buy: no_feasible_quote=67」と主張。**両方とも誤り。**

### 実データ

| 属性 | 値 |
|------|-----|
| requested_side | **sell: 67/67 (100%)** |
| resolved_side_reason | **balance_switch: 67/67 (100%)** |
| XV vetoed | **0/67** |
| 時間帯 (UTC) | 11h: 23, 12h: 21, 13h: 23 |
| 時間帯 (JST) | **20h: 23, 21h: 21, 22h: 23** |

### メカニズム

20-22h JST にバランスリバランスが sell を強制するが、市場条件が sell quote を InfeasibleQuoteError にする。XV veto は関与せず、スプレッド/流動性条件が原因。

### 影響の限定性

67 no_feasible_quote は **cancel (非約定)** なので直接的な PnL 損失はゼロ。fill rate を下げるが、これらの cycle で quote が成立しないこと自体は防御機能として正しく動作している。

---

## §5 FFD の機能評価: 稼働しているが AS 防止に失敗

FFD boost active = 10 filled records:

| Side | AS | PnL30 | n |
|------|----|-------|---|
| sell | True | −29.7 | 5 |
| sell | False | +0.7 | 4 |
| buy | False | −0.3 | 1 |

**FFD boost active 5/10 が AS → FFD は発火するが AS をブロックできていない。**

FFD の設計問題 (515# §3.5 の部分的再評価):
- FFD は offset を boost するが、**ceiling cap で boost が吸収される** (offset 0.274 + FFD → ceiling=0.20 で caps)
- 結果として FFD の boost effort が ceiling に打ち消される
- これは 515# が指摘した「0% 稼働」とは異なるメカニズムだが、**実効的に FFD が無力化されている** という結論は一致

---

## §6 Cycle Origin 別の完全 PnL 分解

| 起源 | cycles | fills | PnL30 | Buy PnL | Sell PnL |
|------|--------|-------|-------|---------|----------|
| **native (自発的)** | 144 | 53 | **−91.0** | −3.6 (28) | **−87.4** (25) |
| balance_switch | 163 | 47 | **+26.0** | +39.4 (16) | −13.4 (31) |
| recovery_skew | 25 | 11 | −10.6 | +3.2 (1) | −13.9 (10) |
| **合計** | **332** | **111** | **−75.6** | **+39.0** (45) | **−114.6** (66) |

**核心的発見**: 
- **native sell (25 fills, −87.4) が全損失の 116%** を占める
- **balance_switch 由来は黒字 (+26.0)** — バランスリバランスは正しく機能している
- **buy 側は全起源で黒字 (+39.0)**

---

## §7 515# P0 推奨の再評価

### 515# P0-1: `ev_as_offset_enabled: false`

**再評価**: EV mult は平均 0.96 で ceiling に近い値を更に 4% 下げるだけ。sell AS 23 fills のうち EV mult による offset 削減は最大でも 0.20 × 0.04 = 0.008 (8JPY/BTC 相当)。ceiling cap (0.274→0.20 で 0.074 削減) の 1/9 に過ぎない。

**判定**: 効果はゼロではないが、ceiling 修正に比べ桁違いに小さい。**P1 に降格。**

### 515# P0-2: `sell_offset: 0.14 → 0.18`

**再評価**: base=0.14 でも base=0.18 でも pipeline 出力は 0.274 (volume guard + regime で飽和)。ceiling=0.20 が支配的なため、base を上げても effective_offset_used はほぼ変わらない。

**判定**: **効果なしに近い。P0 不適格。**

---

## §8 修正 P0 アクション

### P0-1 (即時): `offset_ceiling_ratio_sell: 0.20 → 0.25`

| 項目 | 内容 |
|------|------|
| **根拠** | native sell AS の avg=−7.31 vs forced sell AS (ceil=0.25) の avg=−4.42。ceiling 5% 差が 2.89 JPY/fill の改善を示唆 |
| **リスク** | fill rate 低下の可能性。ただし pipeline 出力 0.274 → ceiling 0.25 でもまだ cap される |
| **既存実績** | buy ceiling は 491# で 0.20→0.25 に引き上げ済みで問題なし |
| **理論的妥当性** | 474# の 0.50→0.20 は過剰引き下げ。pipeline のリスク評価を 0.05 分だけ多く反映 |

### P0-2 (即時): `sell_dynamic_kill.window: 50 → 30`

| 項目 | 内容 |
|------|------|
| **根拠** | 現在 4 cancels。window=50 に対し sell fill=66 で到達はしているが、threshold 判定が甘い。window 縮小で直近の損失パターンに早く反応 |
| **リスク** | 過剰な sell キャンセル → fill rate 低下 |

### P1 (次回テスト)

| # | アクション | 根拠 |
|---|-----------|------|
| 1 | `ev_as_offset_enabled: false` | EV mult の影響は小さいが、ceiling 修正後に EV が ceiling 内で作用する余地が増えるため、併せて評価 |
| 2 | FFD boost が ceiling に吸収される問題の修正 | FFD offset boost を ceiling 適用 **後** に加算する方式を検討 |
| 3 | hour_ceiling_mult の拡大 (20-22h JST) | 67 no_feasible_quote 時間帯で ceiling 緩和がすでに一部設定済み (468#) だが、カバー不十分 |

### P2 (中期)

| # | アクション | 根拠 |
|---|-----------|------|
| 4 | Sell AS の market microstructure 分析 | sell AS avg queue_wait=19.0s (5.6s〜78.4s)。fast fill だけでなく全帯域で AS 発生。情報トレーダーの sell-side picking off パターンの特定が必要 |
| 5 | Side 別 ceiling の動的調整 | AS 率が高い side の ceiling を適応的に引き上げるフィードバック機構 |

---

## §9 三文書の盲点まとめ

### 515# の盲点
1. **データ確認不足**: FFD/sell_dynamic_kill の活動記録を確認せず「0%」と断定
2. **因果の取り違え**: no_feasible_quote を XV cascade と結論づけたが、実際は balance_switch。XV vetoed=0/67
3. **offset pipeline 構造の無視**: base=0.14 が実効値であるかのように扱ったが、ceiling cap が支配的

### 516# (Codex) の盲点
1. **サイド帰属の逆転**: no_feasible_quote を「buy: 67」と記載したが、実際は **100% sell-side**
2. **buy over-filtering 仮説の誤り**: buy は +39.0 で健全。過剰フィルタリングではない
3. **ceiling cap 問題の未検出**: pipeline 飽和は正しく指摘したが、ceiling が真因であることに到達できず

### 517# (PHG) の盲点  
1. **sell offset 0.14 帰属の誤り**: 「0.14 が原因」としたが、実効値は 0.19-0.25 (ceiling 後)
2. **native vs forced sell の未区分**: sell AS 23 fills を一括で扱ったが、native (ceil=0.20) と forced (ceil=0.25) で損失の深刻度が 65% 異なる
3. **ceiling cap 問題の未検出**: 516# と同様、pipeline → ceiling の構造を見落とし

### 全三文書の共通盲点
- **offset_ceiling_ratio_sell=0.20 が pipeline の保護機能を打ち消している** ことに、いずれの文書も到達していない
- **買い側が一貫して黒字 (+39.0)** という好材料を活かした方向性が議論されていない

---

## §10 構造的因果グラフ (修正版)

```
offset pipeline (regime, vol_guard, etc.)
  └─→ sell offset 出力 = 0.274 (適切なリスク評価)
        │
        ↓ ceiling cap
  offset_ceiling_ratio_sell = 0.20 ← 474# (0.50→0.20)
        │
        ↓ EV mult (平均 0.96)
  effective_offset_used ≈ 0.192
        │
        ├─→ native sell (ceil=0.20): 25 fills, PnL30=−87.4
        │     └─→ AS 13 fills, avg=−7.31 ← 最悪カテゴリ
        │
        └─→ forced sell (ceil=0.25): 41 fills, PnL30=−27.3
              └─→ AS 10 fills, avg=−4.42 ← ceiling 5% 差で 65% 改善

buy (ceil=0.25): 45 fills, PnL30=+39.0 ← 健全

FFD boost → ceiling に吸収 → 実効なし
  └─→ boost 後も ceiling=0.20 で cap → AS 防止失敗 (5/10 AS)

no_feasible_quote (67 件)
  ├─→ 原因: balance_switch (100%), NOT XV cascade
  ├─→ side: sell (100%), NOT buy
  ├─→ 時間帯: 20-22h JST (100%)
  └─→ PnL 直接影響: 0 (非約定のため)
```

---

## Appendix A: 検証に使用したデータクエリ

すべての検証は `results/v460/fill_test/fill_records_*.jsonl` の run_id=`1774021842_804f05db` レコードに対して実施。

| 検証項目 | 方法 | 結果 |
|---------|------|------|
| FFD 活動 | `ffd_boost_active == True` | 12 records (10 filled) |
| sell_dynamic_kill | `cancel_reason == 'sell_dynamic_kill'` | 4 cancels |
| no_feasible_quote side | `cancel_reason == 'no_feasible_quote'` × `requested_side` | sell: 67, buy: 0 |
| nfq resolved_side_reason | 上記 × `resolved_side_reason` | balance_switch: 67/67 |
| XV vetoed in nfq | 上記 × `cross_venue_lead_lag_xv_vetoed` | 0/67 |
| ceiling cap | `offset_stages.final > offset_stages.ceiling` | 91/111 (82%) |
| PnL by origin | `resolved_side_reason` 別 PnL30 集計 | native=−91.0, bs=+26.0, rs=−10.6 |

## Appendix B: 515# 訂正対照表

| 515# 箇所 | 原記述 | 訂正 |
|-----------|--------|------|
| 結論要旨 #4 | sell_dynamic_kill の事実上の不活性化 | 4 cancels で稼働中 |
| 結論要旨 #5 | FFD/VG 0% 稼働 | FFD 12 records (10 filled) で稼働中 |
| §2.4 表 | sell_dynamic_kill: 0 (0%) | sell_dynamic_kill: 4 |
| §3.2 | no_feasible_quote 67 が XV veto cascade | balance_switch 67/67。XV vetoed=0 |
| §3.3 | sell_offset 0.14 が損失バケット | base=0.14 → pipeline→ceiling=0.20 で実効 0.19-0.20。base 値は支配的でない |
| §3.4 | sell_dynamic_kill 0 cancels | 4 cancels。完全不活性ではない |
| §3.5 | FFD active 0% | FFD boost active 10 fills。ただし ceiling に吸収されて AS 防止には失敗 |
| §7 P0-1 | ev_as_offset_enabled: false | P1 に降格。ceiling 修正が先決 |
| §7 P0-2 | sell_offset: 0.14→0.18 | P0 不適格。ceiling cap が支配的 |
