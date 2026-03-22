# 540# 536-539 独立検証: Pipeline 実態分析と Phase 1 施策

- **日付**: 2026-03-22
- **目的**: 536#-539# の「風水渙に基づく断捨離」方針をコード・実データで検証し、盲点を指摘し、即時実行可能な Phase 1 施策を実装する
- **入力**: 536#-539#, fill_records_20260320-22, configs/v460/fill_test.yaml, 現行コード

---

## §1 総合評価

536# の問題提起は正当であり、538# の「まず測れ、次に動かせ」は的確な方法論。しかし **4文書すべてが共有する重大な盲点** がある:

1. **Pre-order Pipeline と Executor Pipeline の混同** — 537# が 9段と述べるパイプラインは executor 側のみ。実際の offset 生成は MakerPrice の **14 段** pre-order pipeline が先行し、そこに executor が乗算する二重構造
2. **sell_dynamic_kill 削除が Toxicity Budget を道連れにする危険** — 539# Phase 1 で「遅行型の防衛線を取り払う」としているが、Toxicity Budget（4-tier graduated offset）は sell_dynamic_kill の内部機能。fills の 8.7% で発火中
3. **535# pre-emptive CV kill との関係が未整理** — 直前の 535# で CV velocity ベースの事前 kill を追加済み。sell_dynamic_kill の「事後性」は部分的に解消済み

---

## §2 Pipeline 二重構造の実態（最大の盲点）

### 2.1 Pre-order Pipeline (MakerPrice — 14 stages)

fill_records の `offset_stages` フィールドから累積 offset ratio を分析（N=497 fills, 3/20-22）:

| Stage | 中央値 | 備考 |
|-------|--------|------|
| base | 0.0500 | 固定 or sell=0.18 (506# sell_offset 引き下げ) |
| as_shift | 0.1400 | +0.09。AS 防衛の主要寄与 |
| regime | 0.1472 | +0.007。regime による微調整 |
| **spread_adapt** | **0.3000** | **+0.15。ここで一気に倍増 → 主犯** |
| kyle | 0.3000 | identity (0.00 寄与) |
| amihud | 0.3000 | identity |
| vol_guard | 0.3000 | identity |
| cross_venue | 0.3000 | identity |
| imb_risk | 0.3000 | identity |
| buy_as_guard | 0.3000 | identity |
| sell_hour | 0.3000 | identity |
| loss_boost | 0.3000 | identity |
| ffd | 0.3000 | identity |
| final | 0.3000 | identity |

**発見**: 14 段中 **11 段が identity**（中央値で寄与ゼロ）。offset を実質的に決めているのは `base + as_shift + spread_adapt` の 3 段のみ。

**536# が述べる「乗算チェーンの暴走」は pre-order pipeline では発生していない。問題は spread_adapt が単一ステップで offset を 0.15→0.30 に引き上げることにある。**

### 2.2 Executor Pipeline (9 stages)

fill_records の `executor_offset_stages` から分析:

| Stage | 発火率 | 中央値 | 実態 |
|-------|--------|--------|------|
| EV | 100% | 1.017 | ほぼ identity |
| Velocity | 8.3% | 1.603 | 稀に発火 |
| Trending | 1.2% | 1.500 | 極めて稀 |
| Toxicity | 8.7% | 2.000 | 売り側で一定頻度 |
| VG | **0%** | — | **完全に死亡** |
| Macro | **0%** | — | **完全に死亡** |

**発見**: Executor 乗数は中央値ほぼ 1.0。537# が懸念する「乗算の指数的膨張」は executor でも起きていない。典型的な経路:
```
pre-order(0.30) × EV(1.02) × others(1.0) = 0.306 → ceiling 0.25 でクランプ
```

### 2.3 真の Clamp 構造

| 指標 | 値 |
|------|-----|
| Clamp 率（全体） | **76.7%** (381/497) — 537# の「100%」は特定期間 |
| Clamp 率（Buy） | 72.7% (208/286) |
| Clamp 率（Sell） | 82.0% (173/211) |
| Pre-clamp 中央値 | 0.2996 |
| Pre-clamp 最大値 | 0.6226 |
| Ceiling 0.35 で収まる割合 | 72.3% |
| Ceiling 0.50 で収まる割合 | 93.0% |

---

## §3 Block 理由の定量分析

全 1,572 cycles 中 1,075 がブロック（**68.4%**）。

| Block 理由 | 件数 | 割合 | 分類 |
|-----------|------|------|------|
| preflight_insufficient | 334 | 31.1% | 残高不足（構造的） |
| skip_gate | 173 | 16.1% | ML gate |
| no_feasible_quote | 146 | 13.6% | 板状態 |
| timeout | 124 | 11.5% | fill 待ちタイムアウト |
| spread_too_narrow | 85 | 7.9% | min_spread |
| sell_dynamic_kill | 63 | 5.9% | PnL kill |
| final_clamp_hard_skip | 62 | 5.8% | offset 極端 |
| buy_dynamic_kill | 27 | 2.5% | PnL kill |
| cross_venue_lead_lag_veto | 24 | 2.2% | CV veto |

**発見**: sell_dynamic_kill による block は全体の **4.0%** (63/1572) に過ぎない。539# Phase 1 でこれを「まず散らす」としているが、**影響は限定的であり、撤去の優先度は高くない**。

最大の fill rate 改善余地は:
1. `no_feasible_quote` (13.6%) — 板の品質 / market state
2. `spread_too_narrow` (7.9%) — min_spread_jpy 500 でもまだブロック
3. `final_clamp_hard_skip` (5.8%) — pipeline 出力が extreme

---

## §4 539# Reduction Scorecard の実測ベースライン

539# §3 のスコアカードを実データで埋める:

| 計測項目 | 539# 推定 | **実測値** | 539# 目標 | 判定 |
|----------|----------|-----------|----------|------|
| Hard Gate 数 | 4 | **4** (kill×2, spread, precheck) | — | 正確 |
| Soft Gate 数 | 5 | **6** (regime×3, trend, vel, regime_sell) | — | +1 漏れ |
| Pipeline 段数 (executor) | 9 | **9** (但し VG, Macro は死亡) | 3 以下 | 実効 5 |
| Pipeline 段数 (pre-order) | — | **14** (但し 11 が identity) | — | **未計測** |
| Clamp 率 | ほぼ 100% | **76.7%** | 5% 未満 | 過大推定 |
| Block 率 | — | **68.4%** | — | **未計測** |
| sell_dynamic_kill Block 率 | — | **4.0%** | 0% (撤去) | 低影響 |

---

## §5 536#-539# の盲点一覧

| # | 盲点 | 影響 |
|---|------|------|
| 1 | Pre-order pipeline (14段) の存在と identity 段の実態 | 「乗算暴走」論の前提が不正確 |
| 2 | spread_adapt が単一の主犯（0.15→0.30 倍増） | stage_max_mult は spread_adapt に効かない（pre-order は加算型） |
| 3 | sell_dynamic_kill 撤去 → Toxicity Budget 消失 | 8.7% の fills で offset 2.0× を提供中 |
| 4 | 535# pre-emptive CV kill の存在 | sell 事後対応の部分的解決済み |
| 5 | VG (vol_guard) と Macro が executor で完全死亡 | 最優先の掃除対象 |
| 6 | composite_risk threshold 1.5 が保守的すぎる | 重み 0.4-0.7 で単一 gate は max 0.7。2 gate 同時で初めて 1.1。3 gate で 1.8 → 事実上 3 gate同時発火が必要 |
| 7 | no_feasible_quote (13.6%) と final_clamp_hard_skip (5.8%) が最大の改善余地 | Kill 撤去より板状態・offset制御が先 |
| 8 | 537# の buy/sell 非対称 ceiling 意味論の逆転 | 高 ceiling = 保守的（mid から遠い）。buy aggressive には低 ceiling が必要 |

---

## §6 修正版アクションプラン

538# の「測ってから動かせ」を踏まえ、536#-539# の Phase 構造を再構築する。

### Phase 1: Safe Pruning + Measurement（本実装）

| # | 施策 | 種別 | 根拠 |
|---|------|------|------|
| 1-1 | **sell max_kill_duration: 1800→600** | YAML | 30min deadlock を 10min に短縮。buy は既に 900s |
| 1-2 | **composite_risk_threshold: 1.5→1.0** | YAML | 2 gate 同時で block。現行 1.5 は事実上 3 gate 必要で効果なし |
| 1-3 | **executor 死亡段の可視化ログ** | 観測 | 本分析で VG, Macro 死亡を確認済 |

### Phase 2: Pipeline 正規化（次回）

| # | 施策 | 種別 | 根拠 |
|---|------|------|------|
| 2-1 | **spread_adapt 段の挙動分析** | 調査 | 0.15→0.30 倍増の原因特定。ceiling 引き上げの前提条件 |
| 2-2 | **executor per-stage max_mult 導入** | Code | 537# P4。但し executor は既に tame なので効果は保険的 |
| 2-3 | **ceiling 0.25→0.30 (ladder step 1)** | YAML | spread_adapt 分析後。一気に 0.35 ではなく 0.30 から |
| 2-4 | **pre-order identity 段の dead code 掃除** | Code | 14段中 11段が identity。コード simplicity 向上 |

### Phase 3: Prediction Hub への移行（中期）

| # | 施策 | 種別 | 根拠 |
|---|------|------|------|
| 3-1 | **OFI-Lite**: cycle 間の depth delta | Code | Cont-Kukanov-Stoikov (2014) 近似。OB snapshot は既に取得可能 |
| 3-2 | **Toxicity Budget の独立化**: sell_dynamic_kill から分離 | Refactor | kill binary は撤去可能だが toxicity graduated response は保存 |
| 3-3 | **A-S reference spread** (参照値のみ) | Code | まず計測。live 接続は Phase 4 |

### Phase 4: 動的最適化（長期）— 目標のみ

- A-S 最適スプレッドの live 接続
- SAC 影響力拡大 (±0.15→±5.0bps)
- 低次元 learned calibration (538# §6「第三の道」)

---

## §7 Phase 1 実装の詳細

### 1-1: sell max_kill_duration: 1800→600

**理由**: sell_dynamic_kill の kill 持続が最大 30 分は長すぎる。buy は既に 900s (15 min)。532# で指摘された 3h8m deadlock は 534# で構造的に解消済み（max_consecutive + BTC=0 緩和）だが、kill 自体の持続時間も短縮すべき。

**リスク**: 600s (10 min) は sell PnL が回復する前に kill が解除されるリスクがある。しかし 535# の pre-emptive CV kill が事前防衛を担うため、事後 kill の持続時間は短くてよい。

### 1-2: composite_risk_threshold: 1.5→1.0

**理由**: 現行の重み配分:
- unknown_regime: 0.6
- ranging_low_vol: 0.5
- trending_sell: 0.7
- velocity: 0.4

threshold=1.5 では、最大重み 0.7 の trending_sell + 次点 0.6 の unknown_regime = 1.3 でもまだ通過。**3 gate 同時発火でようやくブロック**というのは composite の意味がない。

threshold=1.0 にすれば、2 gate の合計で block 可能になり、「単一 gate では block しないが複合リスクでは block する」という composite_risk の本来の設計意図が機能する。

---

## §8 539# への応答: 「散らす」の順序修正

539# の 3 フェーズ（散らす→渙める→渙発する）を支持するが、「散らす」対象の優先順位を修正する:

**散らすべきもの（優先順）**:
1. ~~sell_dynamic_kill~~ → **kill 持続時間の短縮** (max_duration 600s)。完全撤去は Toxicity Budget 消失のため不可
2. Pre-order pipeline の **identity 11 段** → dead code として削除候補
3. Executor の **VG, Macro 死亡段** → config で disabled 化 or 削除

**散らしてはいけないもの**:
1. **Toxicity Budget** — sell_dynamic_kill の中にあるが、kill とは独立に機能する graduated response
2. **cross_venue_lead_lag_veto** — 唯一の予測的防衛指標
3. **535# pre-emptive CV kill** — 直前に導入した事前防衛。kill 短縮と相補的

---

## §9 後続実装

- **[541#](541_phg_pipeline_optimization_lazy_import_stage_skip.md)**: lazy import 引き上げ (8 箇所) + pre-order disabled stage 5 段のスキップ最適化 + spread_adapt 挙動解明
- **[542#](542_phg_ceiling_030_identity_audit_memleak.md)**: ceiling 0.25→0.30 + 残存 identity 段 6 段のスキップ不可確認 + メモリリーク監査 + test_405 陳腐化修正
- **[543#](543_phg_phase3_ofi_toxicity_as_delta.md)**: Phase 3 実装 — OFI-Lite (CKS 2014) + Toxicity Budget 独立化 (GM 1985) + A-S δ* 計測

### spread_adapt 挙動メモ

- `narrow_spread_bps: 2.5` — スプレッド < 2.5bps で boost ×2.0（buy）/ ×2.5（sell）
- `wide_spread_bps: 4.5` — スプレッド > 4.5bps で ×0.5（縮小）
- Coincheck の典型スプレッドは 1-2.5bps → **ほぼ全サイクルで narrow boost が発火**
- offset 0.15 × 2.0 = 0.30 → ceiling 0.25 でクランプ → **clamp 飽和の直接原因**
- これは意図的設計（狭スプレッド時の AS リスク軽減）だが、ceiling との組み合わせで情報が失われる

**提案**: ceiling 引き上げ（0.25→0.30）は spread_adapt の narrow boost 効果を活かすために必要だが、段階的に実施すべき（540# §6 Phase 2-3）
