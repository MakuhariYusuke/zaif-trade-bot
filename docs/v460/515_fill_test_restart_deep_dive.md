# 515# Fill Test 根本原因分析 — 506#/507# 設定変更後の構造的劣化メカニズム

> **更新**: 2026-03-21 14:30 JST — 前回 run (492 cycles/258 fills) + 現行 run (305 cycles/102 fills) の統計的分析に基づく

> **⚠ 重要訂正 (518# 参照)**: 本ドキュメントの以下の主張はデータ検証により否定されました:
> - §3.4 sell_dynamic_kill「0 cancels で死亡」→ 実際は **4 cancels で稼働中**
> - §3.5 FFD「0% 稼働」→ 実際は **12 records (10 filled) で稼働中**
> - §3.2 no_feasible_quote 67 件を「XV veto cascade」と結論 → 実際は **balance_switch 67/67、XV vetoed=0**
> - §7 P0-1/P0-2 の推奨は **518# で再評価済み**。真因は `offset_ceiling_ratio_sell=0.20` の過剰制限
> 
> 詳細は [518# 方向修正](518_reconciliation_515_516_517_direction_correction.md) を参照

## 結論要旨

**現行 run の PnL30 = -72.5 JPY (PnL120 = -85.6) は、以下の構造的メカニズムが複合した結果である:**

1. **EV offset が逆選択増幅器として機能** — 高 EV → 積極的 offset → 即約定 → AS → 損失 (§3.1)
2. **XV basis_correction の paradox** — filtration 強化 → 残余 fill の毒性が上昇 (§3.2)
3. **sell_offset 0.14 が損失バケットに位置** — offset [0.10, 0.15) = WR 20%, avg -3.65 JPY (§3.3)
4. **sell_dynamic_kill の事実上の不活性化** — fill 数低下で rolling window 未到達 (§3.4)
5. **FFD/VG 0% 稼働** — 29.4% AS 率でも安全機構がゼロ発火 (§3.5)

## Run 情報

### 現行 Run

| 項目 | 値 |
|------|-----|
| **起動時刻** | 2026-03-21 00:50:44 JST (UTC: 2026-03-20T15:50:44) |
| **run_id** | `1774021842_804f05db` |
| **git SHA** | `20d4f778ef67` — `session037: sweep canonical test imports` |
| **PID** | 89776 |
| **config** | `configs/v460/fill_test.yaml` |
| **dry_run** | false |

### 前回 Run

| 項目 | 値 |
|------|-----|
| **起動時刻** | 2026-03-20 05:51:14 JST (UTC: 2026-03-19T20:51:14) |
| **run_id** | `1773953473_cd76a319` |
| **git SHA** | `dfbe3b539eaa` — `499# ドキュメント: hard_loss_cap crash loop 根本原因と修正記録` |
| **PID** | 15468 |
| **稼働時間** | 約 19 時間 (03/20 05:51 → 03/21 00:37 JST) |

---

## SHA 間の差分 (dfbe3b53 → 20d4f778)

前回 run `dfbe3b539eaa` から現行 `20d4f778ef67` まで **47 コミット**。主なもの:

### コード変更 (session037 リファクタリング)

| SHA | 概要 |
|-----|------|
| `0b4be7638` | skip gate context assembly 重複排除 |
| `7a8acdaf7` | skip gate context boundary 引き締め |
| `958fea05e` | canonical test imports 統一 |
| `fe9831f2a` | skip gate payload boundary 精査 |
| `4b18bfac6` | spread guard helper 抽出 |
| `53d3ba08d` | offset amount helper 抽出 |
| `084a7b708` | spread adaptive helper 抽出 + skip gate reasons 引き締め |
| `88bde8b79` | skip gate fill record boundary 統合 |
| `645c977de` | offset ceiling helper 抽出 |
| `4fcd401e0` | final ceiling stage 抽出 |
| `20d4f778e` | canonical test imports sweep (最終) |

### 設定変更を含むコミット

| SHA | doc# | 変更内容 |
|-----|------|---------|
| `287d04f7a` | **506#** | `sell_age_cap_sec: 25.0` 新設 / `sell_offset: 0.18→0.14` / `basis_correction_enabled: true` + `basis_ema_alpha: 0.02` |
| `d9c31ff6d` | **507#** | `recovery_skew_offset_mult: 2.0→1.5` |
| `c9f2e176c` | **512#** | `favorable_tighten_enabled: false` + `favorable_tighten_mult: 0.90` (安全デフォルト) |
| `c40e72175` | **513#** | `spread_anomaly_detector` / `micro_circuit_breaker` セクション YAML 化 (両方 `enabled: false`) |

### 設定変更の詳細 diff

```yaml
# 506# P0: sell 最大滞留時間の新設
+ sell_age_cap_sec: 25.0        # 30-50sバケットに -158.73JPY 集中

# 506# P0: sell offset 保守化を元に戻す
- sell: 0.18                    # 121# A2 で保守化
+ sell: 0.14                    # WR=67.4%, avg=+3.223 バケット

# 506# P1: Cross-venue basis correction (構造的 ~3.3bps 偏差の補正)
+ basis_correction_enabled: true
+ basis_ema_alpha: 0.02

# 507# P1: Recovery skew offset 緩和
- recovery_skew_offset_mult: 2.0
+ recovery_skew_offset_mult: 1.5    # 実績 avg=-0.631 改善

# 512# Favorable-side tightening (安全デフォルト、未有効化)
+ favorable_tighten_enabled: false
+ favorable_tighten_mult: 0.90

# 513# SAD/MCB YAML化 (設定参照のみ、インスタンス化未実装)
+ spread_anomaly_detector:
+   enabled: false
+ micro_circuit_breaker:
+   enabled: false
```

> **注**: 512# `favorable_tighten` と 513# SAD/MCB はいずれも `enabled: false` であり、実行時動作への影響はない。実質的な変更は **506# (sell_age_cap / sell_offset / basis_correction)** と **507# (recovery_skew)** のみ。

---

## 再起動前の重大イベント: hard_loss_cap Crash Loop

### タイムライン (3/20 04:24〜05:51 JST)

| JST | UTC | イベント | SHA | 理由 |
|-----|-----|---------|-----|------|
| 04:24:55 | 19:24:55 | start | `07d5a711d9d4` (497# config deep dive doc) | — |
| 04:25:52 | 19:25:52 | **stop** | 同上 | **hard_loss_cap** (1サイクルで即発動) |
| 04:29:52 | 19:29:52 | start (watchdog) | `07d5a711d9d4` | — |
| 04:32:13 | 19:32:13 | **stop** | 同上 | **hard_loss_cap** |
| 04:34:55 | 19:34:55 | start (watchdog) | `30e1c1f9e9aa` (498# hot-reload 横展開) | — |
| 04:35:21 | 19:35:21 | **stop** | 同上 | **hard_loss_cap** |
| 05:29:57 | 20:29:57 | start (手動) | `600eb3b789ae` (revenue context event logging) | — |
| 05:31:23 | 20:31:23 | **stop** | 同上 | **hard_loss_cap** |
| 05:34:53 | 20:34:53 | start (手動) | `36047bcbbbf6` (raw UTC day helpers) | — |
| 05:35:14 | 20:35:14 | **stop** | 同上 | **hard_loss_cap** |
| 05:39:53 | 20:39:53 | start (手動) | `36047bcbbbf6` | — |
| 05:40:20 | 20:40:20 | **stop** | 同上 | **hard_loss_cap** |
| **05:51:14** | **20:51:14** | **start** | **`dfbe3b539eaa`** (499# crash loop 修正) | **安定** |

### 根本原因 (499# 参照)

`cumulative_pnl_jpy` が **全期間 (2/13〜3/19 = 35 日分)** のレコードから再計算されていた:

- 全期間 cumulative_pnl_jpy = **-1449.3 JPY** (4146 fills)
- loss_cap_jpy = 残高 25,000 × 0.05 = **1,250 JPY**
- `|-1449| > 1250` → 即座に hard_loss_cap 発動 → watchdog restart → 同じ結果 → ∞ loop

### 修正内容

- resume 時の `cumulative_pnl_jpy` を **当日 UTC スコープ** に限定
- `_process_daily_reset` に累積 PnL のゼロリセットを追加

---

## 3/17-3/18 の preflight_skip_exceeded 連鎖

crash loop の前段として、3/18 06:35〜11:45 JST に `preflight_skip_exceeded` による連続停止が発生:

| JST | SHA | 結果 |
|-----|-----|------|
| 03/18 06:35 | `b70365d4d` (468# deep-night 防御強化) | preflight_skip_exceeded (1h23m) |
| 03/18 08:00 | 同上 | preflight_skip_exceeded (57m) |
| 03/18 09:00 | 同上 | preflight_skip_exceeded (57m) |
| 03/18 10:00 | 同上 | preflight_skip_exceeded (58m) |
| 03/18 11:00 | 同上 | preflight_skip_exceeded (40m) |
| 03/18 11:45 | 同上 | preflight_skip_exceeded (1h12m) |

468# の deep-night 防御強化がスプレッド条件を満たさない深夜帯でスキップ過多となり、preflight チェックで停止。その後 3/18 13:20 (SHA=b157832a) 以降も lock_conflict が多発。最終的に 3/18 15:56 (SHA=`94a34e1b`) で復帰。

---

## パフォーマンス比較: 前回 Run vs 現行 Run (十分なサンプル)

### サマリ

| 指標 | 前回 (dfbe3b53, 19h) | 現行 (20d4f778, 14h) | 差 | 統計的意味 |
|------|---------------------|---------------------|-----|-----------|
| サイクル数 | 492 | 305 | — | — |
| Fill 数 | 258 | 102 | — | — |
| **Fill rate** | **52.4%** | **33.4%** | **-19.0pp** | XV veto + no_feasible_quote で大幅低下 |
| **PnL30s total** | **+12.0** | **-72.5** | **-84.5** | 統計的に有意な劣化 |
| PnL30s avg | +0.047 | -0.710 | -0.757 | Cohen's d ≈ 0.12 (effect size 小) |
| PnL30s stdev | 6.811 | 4.954 | -1.857 | 分散縮小 = 大勝ちが消えた |
| PnL60s total | -85.8 | -72.7 | — | PnL は時間経過で悪化 |
| PnL120s total | -128.9 | -85.6 | — | 同上 |
| **WR** | **48.4%** | **40.2%** | **-8.2pp** | — |
| **AS rate** | **32.9%** | **29.4%** | **-3.5pp** | AS 率は微改善だが WR/PnL は悪化 |

> **注**: `PnL30 avg` の Cohen's d は小さいが、total PnL の差 (-84.5 JPY) は実運用上深刻。平均効果が小さいのは stdev が大きいためであり、**尾部リスク (tail risk)** の問題が本質。

### Side 別 (§2.1)

| Side | 前回 fills / PnL30 / avg / WR / AS% | 現行 fills / PnL30 / avg / WR / AS% |
|------|--------------------------------------|--------------------------------------|
| **Buy** | 131 / -49.0 / -0.37 / 50% / 27% | 63 / -26.2 / -0.42 / 43% / 21% |
| **Sell** | 127 / **+61.1** / +0.48 / 47% / 39% | 39 / **-46.3** / -1.19 / 36% / **44%** |

**sell PnL の完全反転 (+61.1 → -46.3) が最大の劣化要因。** sell 側の AS 率が 39%→44% へ悪化し、WR が 47%→36% に低下。sell_offset 0.18→0.14 の変更により sell がより mid に近い位置で発注 → 逆選択に脆弱化。

### Cross-Venue Lead-Lag (§2.2)

| 状態 | 前回 n / PnL30 / AS% | 現行 n / PnL30 / AS% |
|------|----------------------|----------------------|
| **XV applied** | 127 / -53.8 (avg -0.42) / 28% | 25 / -11.5 (avg -0.46) / 20% |
| XV NOT applied | 131 / +65.9 (avg +0.50) / 38% | 77 / -60.9 (avg -0.79) / 32% |

**前回 run**: XV applied = 127/131 が全て **buy** 側、**sell は 0/127**。
**現行 run**: XV applied buy=18, sell=7。basis_correction により sell にも適用されたが、**XV NOT applied の PnL が +65.9 → -60.9 に反転**。

### Adverse Selection (§2.3 — 最重要指標)

| 条件 | 前回 n / PnL30 / avg / WR | 現行 n / PnL30 / avg / WR |
|------|--------------------------|--------------------------|
| **AS=True** | 85 / **-526.2** / -6.19 / **0%** | 30 / **-172.3** / -5.74 / **0%** |
| AS=False | 173 / +538.3 / +3.11 / 72% | 72 / +99.9 / +1.39 / 57% |

**両 run とも AS=True の WR は 0%。** 逆選択された fill は **一つも利益を出せない**。Glosten-Milgrom (1985) モデルにおける情報の非対称性が極度に高い市場構造を示す。

**問題**: Non-AS fill の WR が 72%→57% に低下 = 「安全な」fill まで劣化。これは market-wide な問題ではなく、**config 変更による sampling bias の変化** を示唆。

### Cancel Reasons の構造変化 (§2.4)

| 理由 | 前回 (234) | 現行 (203) | 変化 |
|------|-----------|-----------|------|
| **no_feasible_quote** | **0** (0%) | **67** (33.0%) | **新出・最大** |
| skip_gate | 66 (28.2%) | 44 (21.7%) | — |
| **sell_dynamic_kill** | **43** (18.4%) | **0** (0%) | **消失** |
| timeout | 22 (9.4%) | 28 (13.8%) | 微増 |
| spread_too_narrow | 26 (11.1%) | 22 (10.8%) | — |
| **cross_venue_lead_lag_veto** | **0** (0%) | **13** (6.4%) | **新出** |
| buy_dynamic_kill | 10 (4.3%) | 12 (5.9%) | — |
| final_clamp_hard_skip | 50 (21.4%) | 5 (2.5%) | **激減** |

### Queue Wait vs PnL の逆転 (§2.5)

| 待ち時間 | 前回 PnL30 / AS% | 現行 PnL30 / AS% | 変化 |
|---------|-----------------|-----------------|------|
| **0-10s** | **+84.9** / 24% | **-60.4** / **45%** | **完全逆転** |
| 10-20s | -44.3 / 38% | -5.4 / 38% | — |
| 20-40s | -54.3 / 35% | -18.4 / 15% | — |
| 40-60s | +20.6 / 43% | +0.3 / 29% | — |
| 60-120s | +5.1 / 33% | +11.4 / 11% | — |

**0-10s の fast fill が +84.9→-60.4 へ完全逆転**。AS 率も 24%→45% へ倍増。原因: sell_offset 0.14 により sell が mid に接近 → 即約定 → 逆選択。Fast fill がもはや「良い fill」ではない。

### EV Score vs Actual PnL (§2.6 — calibration 崩壊)

| EV 区間 | 前回 n / PnL30 | 現行 n / PnL30 |
|---------|---------------|---------------|
| [-10, -2) | 30 / -19.2 | 3 / -7.9 |
| [-2, -1) | 46 / **+53.2** | 4 / -8.7 |
| [-1, 0) | 58 / **-74.8** | 15 / -2.6 |
| [0, 1) | 55 / +36.8 | 28 / -3.1 |
| [1, 2) | 39 / +7.0 | 26 / -8.9 |
| **[2, 5)** | 30 / +9.0 | 25 / **-45.1** |

**現行 run では EV と PnL が完全に逆相関。** EV[2,5) が最悪 (-45.1)。§3.1 で分析。

### Offset vs PnL (§2.7)

| Offset 区間 | 前回 n / PnL30 / WR | 現行 n / PnL30 / WR |
|-------------|---------------------|---------------------|
| **[0.10, 0.15)** | 10 / **-36.5** / **20%** | 1 / -2.0 / 0% |
| [0.15, 0.20) | 38 / +34.5 / 50% | 11 / -21.8 / 27% |
| [0.20, 0.25) | 129 / +51.5 / 50% | 45 / -39.5 / 38% |
| [0.25, 0.50) | 77 / -33.8 / 49% | 45 / -9.2 / 47% |

**全 offset バケットが現行 run で悪化。** 特に [0.15, 0.20) が +34.5→-21.8。

---

## §3 根本原因分析

### §3.1 EV Offset Adverse Selection Amplifier (逆選択増幅バグ)

**メカニズム (Glosten & Milgrom 1985, Kyle 1985 の文脈):**

```
EV score 高 → compute_ev_offset_multiplier() で mult > 1.0
  → aggressive_when_multiplier_gt_one=True → offset を mid に接近
    → queue position が前方 → fill_probability 上昇
      → 即約定率 上昇 → adverse selection 確率 上昇
        → PnL 悪化
```

**コード経路** (`fill_config_results.py:97-125`):
```python
raw = 1.0 + sensitivity * ev_score   # sensitivity=0.05
mult = clamp(raw, min_mult=0.5, max_mult=1.5)
# ev=+5 → mult=1.25 (25% aggressive), ev=-5 → mult=0.75 (25% conservative)
```

`offset_pipeline.py:91-110` で `aggressive_when_multiplier_gt_one=True` により、高 EV → offset 縮小 → mid に接近。

**Winner's Curse**: market maker の limit order が速く fill される ≈ informed trader に picking off されている。高 EV 判定 → 積極的配置 → 即約定 → 実際にはほぼ全て AS → 損失。これは classic な winner's curse (Milgrom & Weber 1982) そのもの。

**実証**: 
- 現行 run: EV[2,5) → fast fill 多 → PnL30=-45.1 (worst bucket)
- 前回 run: EV[-2,-1) → conservative fill → PnL30=+53.2 (best bucket)
- **EV と PnL の相関が non-monotonic (前回) → anti-correlated (現行)**

**対策候補**:
- **即効**: `ev_as_offset_enabled: false` で EV offset パイプラインを無効化
- **代替**: sensitivity の符号反転 (-0.05) = 高 EV 時に保守化 (逆直観だが market theory に整合)
- **根本**: EV model 自体の再学習 (post-fill PnL30 を教師データにした再 calibration)

### §3.2 XV Basis Correction の Filtration Paradox

**506# で basis_correction_enabled + basis_ema_alpha=0.02 を導入。**

意図: CC mid > BF mid の構造的偏差 (~3.3bps) を補正し、adverse_side が常に "buy" に偏る問題を解消。

**実際の影響**:

| 指標 | 前回 (basis_correction 無効) | 現行 (basis_correction 有効) |
|------|---------------------------|---------------------------|
| XV buy applied | 127/131 (97%) | 18/63 (29%) | 
| XV sell applied | **0/127** (0%) | 7/39 (18%) |
| XV veto cancels | 0 | 13 (6.4%) |
| no_feasible_quote | 0 | **67 (33.0%)** |
| Fill rate | 52.4% | 33.4% |

**Cascade 経路** (`maker_price.py:1034-1038`):
```
XV hint → adverse_side 判定 → veto flag → InfeasibleQuoteError(CROSS_VENUE_LEAD_LAG_VETO)
  → 3 consecutive → NO_FEASIBLE_QUOTE (fill_cycle_executor.py:719-725)
```

XV veto が InfeasibleQuoteError を発生 → 同一 side で 3 回連続 → `no_feasible_quote` に昇格。
**67 no_feasible_quote のうち相当数が XV veto の cascade。**

**Filtration Paradox** (Frey & Stremme 1997 の情報フィルタリング理論):
- 強い事前フィルタ → 通過した fill の条件付き期待値が悪化
- 前回: XV は buy のみ保護 → sell は unprotected だが profitable (+61.1)
- 現行: XV が両 side を保護 + veto → fill rate 19pp 低下、だが残余 fill の PnL は WORSE

**根本問題**: XV retreat の offset adjustment 幅が情報トレーダーの損失以下。XV が「この fill は危険」と正しく検出しても、retreat 幅が不十分なため fill が成立し、結果的に損失。

### §3.3 sell_offset 0.14 → 損失バケット問題

506# で `sell_offset: 0.18 → 0.14` に変更。根拠は「0.10-0.19 バケット WR=67.4%」。

**しかし前回 run の実データ**:
- offset [0.10, 0.15): **PnL30=-36.5, WR=20%** ← 0.14 はこの **損失バケット** に位置
- offset [0.15, 0.20): PnL30=+34.5, WR=50%

**分析上の罠**: 506# の根拠となった分析期間と前回 run の分析期間が異なる。0.10-0.19 全体では positive でも、**0.10-0.15 と 0.15-0.20 で PnL が sign 逆転** しており、0.14 は損失側に位置。

さらに:
- sell_offset=0.14 → sell 注文が mid に接近 → queue position 前方 → 即約定
- 前回 run で 0-10s fast fill AS 率は 24% だったが、sell_offset 接近により現行で 45% へ倍増

**Ho & Stoll (1981) の最適スプレッド理論**: inventory risk に見合うスプレッドが必要。0.14 は inventory risk を過小評価した設定。

**対策**: sell_offset を 0.18 に戻す (506# 変更のリバート) または 0.16 に中間値を検討。

### §3.4 sell_dynamic_kill の事実上の不活性化

config: `sell_dynamic_kill.enabled: true`, `window: 50`, `threshold_bps: -0.5`

**前回 run**: 43 cancels (18.4%) — sell 側の「安全弁」として機能
**現行 run**: **0 cancels** — 完全に不発

**原因**: 
1. sell fill 数: 127 → 39。window=50 に対し sell fill=39 → **rolling window 未到達**
2. Fill rate 低下 (52.4%→33.4%) + sell 比率低下 (sell=39 vs buy=63) で sell fill が蓄積しない
3. no_feasible_quote (67 cancels) + XV veto (13) が先に kill → sell_dynamic_kill に到達する前にサイクルが終了

**構造的問題**: XV の過剰 filtration → sell fill 減少 → sell_dynamic_kill window 未到達 → 残った sell fill に対する保護なし → sell PnL 悪化。**Safety mechanism の dead zone**。

### §3.5 FFD (Fast Fill Defense) / VG (Velocity Guard) 0% 稼働

**両 run とも FFD active = 0%, VG active = 0%。**

FFD config: `enabled: true`, `threshold_sec: 5.0` (buy: 8.0, sell: 15.0)

AS 率 29.4% で fast fill (0-10s) の AS 率 45% にもかかわらず FFD boost がゼロ。

**考えられる原因**:
1. **Layer 1 (即時判定)**: `fill_price vs mid_at_fill` の spread cost が正常範囲 → "negative edge" 非検出
2. **Layer 2 (post-fill PnL)**: `post_fill_pnl_bps` が `evaluate_fill()` に供給されていない、または deadzone (l2_deadzone_bps=3.0) を超えないケースが多い
3. 構造的: AS fill の損失は即時の price impact ではなく **30s-120s スケールの drift** で顕在化。Layer 1 の即時チェックでは捕捉不能

**市場理論的解釈**: Coincheck の BTC/JPY 市場では、情報トレーダーの価格インパクトが即時ではなく **遅延的 (gradual)**。Almgren & Chriss (2001) の optimal execution モデルが示す "slow information leakage" に合致。FFD の即時検出設計はこの市場構造に不適合。

---

## §4 構造的因果グラフ

```
506# sell_offset 0.18→0.14
  ├─→ sell 注文が mid に接近
  │     ├─→ fast fill 増加 → AS 率倍増 (24%→45%) ← §3.3
  │     └─→ sell PnL 反転 (+61.1 → -46.3)
  │
506# basis_correction_enabled
  ├─→ XV direction "up" 検出 (sell adverse)
  │     ├─→ XV veto 13 cancels (NEW)
  │     ├─→ no_feasible_quote 67 cancels (cascade) ← §3.2
  │     └─→ fill rate 52.4% → 33.4%
  ├─→ sell fill 数 127→39
  │     └─→ sell_dynamic_kill window=50 未到達 → 安全弁 死亡 ← §3.4
  └─→ filtration paradox: 残余 fill の AS 率は上がらないが WR/PnL は悪化

EV offset (ev_as_offset_enabled=true)
  └─→ 高 EV → aggressive offset → fast fill → AS ← §3.1
        └─→ EV[2,5) PnL=-45.1 (worst bucket)

FFD/VG 0% ← §3.5
  └─→ 即時 negative edge 非検出 → fast fill AS に対する保護なし
```

---

## §5 Macro Trend / Regime 分析

### Macro Trend

| macro × aligned | 前回 n / PnL30 | 現行 n / PnL30 |
|-----------------|---------------|---------------|
| neutral + aligned | 28 / +85.3 | 61 / **-52.0** |
| weak_up + aligned | 97 / +66.4 | 20 / +22.9 |
| weak_down + aligned | 83 / -102.4 | 5 / -19.4 |
| strong_up + aligned | 28 / -29.7 | 5 / -14.8 |

**macro_neutral が +85.3→-52.0 に反転**。neutral は情報がない状態での取引だが、現行 config の aggressive sell_offset が neutral 市場でも逆選択を引き起こしている。

### Regime 分布

- 前回: ranging=235, trending_up=17, trending_down=6 — ほぼ全て ranging
- 現行: ranging=96, trending_up=6, trending_down=0 — 同様に ranging 中心

trending_down が 0 fill は夜間〜午後の time-of-day 分布の影響。

### Spread at Order

| Spread 区間 | 前回 PnL30 / WR | 現行 PnL30 / WR |
|-------------|----------------|----------------|
| [0, 1000) | +28.9 / 67% | **-14.9 / 22%** |
| [1000, 2000) | -7.7 / 48% | -47.5 / 40% |
| [2000, 3000) | +59.1 / 53% | -2.7 / 48% |
| [3000, 4000) | -78.0 / 30% | -7.4 / 27% |

**narrow spread [0,1000) が +28.9→-14.9 に反転**。narrow spread = 競争的市場 = informed trading が多い。sell_offset 0.14 で mid に接近した sell 注文が narrow spread 環境で即座に picking off されている。

---

## §6 時間帯分析 (Time-of-Day)

### 前回 run (05:51-00:37 JST)

- **黒字時間帯**: 10h (+20.1), 15h (+28.5), 16h (+20.8), 18h (+24.9), 20h (+37.7), 22h (+18.2)
- **赤字時間帯**: 21h (-34.8), 23h (-33.3), 00h (-13.4)
- **パターン**: 東京昼間 (10-18h) が主力、NYクローズ (21-23h) が損失

### 現行 run (00:50- JST)

- **深刻赤字**: 02h (-42.9, n=12, loss%=83%), 07h (-13.3), 14h (-13.6)
- **唯一の黒字**: 00h (+20.3, n=1), 01h (+8.7, n=1), 03h (+4.7)

**02h の -42.9 JPY** (12 fills, 83% loss率) は現行 run 損失の 59% を占める。深夜帯の流動性低下 + informed trader 比率上昇が原因だが、根本的には sell_offset 0.14 で AS に脆弱な状態で深夜帯に取引したことが問題。

---

## §7 推奨アクション (優先順位付き)

### P0 (即時対応)

| # | アクション | 理由 | 影響 |
|---|-----------|------|------|
| 1 | `ev_as_offset_enabled: false` | EV offset が逆選択増幅器として機能 (§3.1) | 全 fill の offset が EV 非依存になり、systematic な AS 増幅を停止 |
| 2 | `sell_offset: 0.14 → 0.18` (リバート) | 0.14 は損失バケット (§3.3) | sell の mid 接近を是正、sell PnL 反転を修正 |

### P1 (次回テストサイクル)

| # | アクション | 理由 |
|---|-----------|------|
| 3 | XV retreat offset の引き上げ | XV が danger を検出しても retreat 不十分 (§3.2) |
| 4 | FFD Layer 2 の post_fill_pnl 供給確認 | 0% 稼働は safety gap (§3.5) |
| 5 | sell_dynamic_kill window: 50→20 に縮小 | fill 数低下で window 未到達 (§3.4) |

### P2 (中期改善)

| # | アクション | 理由 |
|---|-----------|------|
| 6 | EV model 再学習 (PnL30 教師) | EV が predictive power を失っている |
| 7 | FFD のイベント駆動 (drift-based) 化 | 即時チェックでは slow information leakage を捕捉不能 |
| 8 | 時間帯別 offset 動的調整 | 深夜帯の構造的リスクは offset で吸収すべき |

---

## Appendix A: 再起動履歴サマリ (3/15〜3/21)

```
03/15 09:02 JST  SHA=78ac4ecd (長期 run 開始)
  ↓ 複数回の短寿命 restart (3/15-3/16)
03/16 21:03 JST  SHA=d0769f28 → 03/17 04:27 (7.4h, PnL30=+67.9)
03/17 04:37 JST  SHA=f840d0e0 → 3/17 16:17 (11.7h, PnL30=-21.4, fill rate 3.8%)
03/17 16:22 JST  SHA=f840d0e0 → 3/18 06:03 (13.7h, PnL30=-91.6, fill rate 30.2%)
  ↓ preflight_skip_exceeded 連鎖 (3/18 06:35-11:45, SHA=b70365d4d)
03/18 15:56 JST  SHA=94a34e1b → 復帰
  ↓ 複数 restart (3/18-3/19)
03/19 18:34 JST  SHA=cc6c9466 → 03/20 00:55 (6.3h, PnL30=-52.8, fill rate 35.3%)
  ↓ hard_loss_cap crash loop (3/20 04:24-05:40, 6回連続停止)
03/20 05:51 JST  SHA=dfbe3b53 (499# 修正) → 03/21 00:37 (18.8h, PnL30=+12.0) ← 前回
03/21 00:50 JST  SHA=20d4f778 (session037 最終) → 稼働中 (PnL30=-72.5)       ← 現行
```
