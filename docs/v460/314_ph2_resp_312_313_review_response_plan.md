# 314# resp: 312# Codex / 313# Gemini レビュー応答 — 妥当性検証・盲点・計画

> **日付**: 2026-03-07  
> **種別**: resp (レビュー応答)  
> **対象**: [312#](312_ph2_rev_308_311_multifaceted_validation.md) (Codex), [313#](313_ph2_gemini_31_pro_review_309_312_pricing_math_inversion.md) (Gemini 3.1 Pro)  
> **検証方法**: コード直接読了 + 実データ分布検査  

---

## 目次

- [§1 結論](#1-結論)
- [§2 312# (Codex) 各 Finding の妥当性検証](#2-312-codex-各-finding-の妥当性検証)
- [§3 313# (Gemini) 各主張の妥当性検証](#3-313-gemini-各主張の妥当性検証)
- [§4 盲点: 両レビューが見落とした問題](#4-盲点-両レビューが見落とした問題)
- [§5 影響範囲の整理: 分析コード vs プロダクションコード](#5-影響範囲の整理-分析コード-vs-プロダクションコード)
- [§6 実行計画](#6-実行計画)

---

## §1 結論

312# / 313# のレビューは **核心的な問題を正確に特定している**。特に F2 (spread capture 分解式の転倒) は分析基盤を根底から揺るがす CRITICAL 問題であり、即時修正が必要。

一方で、313# の表現は一部過剰であり、「311# 白紙撤回」は不要。影響を受けるのは §6 (Spread/AS 分解) と `derive_improvement_proposals()` の efficiency 依存部分のみ。regime 分析・時間帯分析・offset quintile・AB 判定は spread capture に依存しておらず有効。

また、**両レビューが見落とした盲点を 4 件特定** した。特に B1 (offset ratio のセマンティクス全体転倒) は 313# §2.2 の指摘をさらに深刻化させるもので、プロダクションコード自体への影響がある。

---

## §2 312# (Codex) 各 Finding の妥当性検証

### F1: Mixed-SHA 汚染 [HIGH] — ✅ 妥当

**検証結果**: `analysis/311_observational_rerun.py` は全 fill record を無差別に読み込み、`git_sha` / `date_from` フィルタを一切使用していない。`ztb/metrics/fill_quality.py` の `apply_fill_record_filters()` は既に存在するが未活用。

**影響**: 311# の数値は「現行システムの評価」ではなく「22 日間の混合 SHA 回顧分析」。312# の指摘通り Gate 根拠としては弱い。

**312# 推奨の妥当性**: ✅ `apply_fill_record_filters(git_sha="dcc3064")` で再集計すべき。ただし dcc3064a8 のデータは n≈5 sell と極少で、統計的に有意な結論は出せない。48–72h 置いて再集計が現実的。

### F2: Spread Capture / AS Cost 分解式の転倒 [CRITICAL] — ✅ 妥当・確認済み

**検証結果**: コード精査で確認。

価格計算式 (`maker_price.py` L588–605):
```
buy:  price = best_bid + offset    (offset = spread × ratio)
sell: price = best_ask - offset
```

mid 基準の spread capture:
$$\text{spread\_capture\_mid} = \text{spread} \times (0.5 - \text{ratio})$$

best price 基準の spread capture:
$$\text{spread\_capture\_best} = \text{spread} \times (1 - \text{ratio})$$

311# の式 (`analysis/311_observational_rerun.py` L237):
```python
sc_bps = spread_bps * offset  # offset = effective_offset_ratio
```

これは **ratio が大きいほど spread capture が大きい** と計算している。実態は ratio が大きいほど price が mid に近づき spread capture は **減少する**。完全に転倒。

**実データ検証**: 直近 5 ファイルの sell filled records (n=197):
- Mean effective_offset_used = **0.446** (mid の手前 5.4% でしか capture していない!)
- Max = **2.088** (best_bid 超え → spread guard が発動したか、repricing 後の値)
- Median = **0.313**

ratio=0.446 の場合、正しい mid spread capture = spread × (0.5 - 0.446) = **spread × 0.054**。
311# の誤った計算: spread × 0.446 = **spread × 0.446**。8.3 倍の過大評価。

**影響範囲**: 
- 311# §6 の spread_capture / AS cost / efficiency 全数値 → **無効**
- `derive_improvement_proposals()` の D-buy P0 優先度判定 → **無効**
- 311# の他のセクション (regime, hour, quintile, AB) → **spread capture に非依存、有効**

### F3: sell_hour_offset_boost 効果測定の誤り [HIGH] — ✅ 妥当

**検証結果**: `sell_hour_boost_analysis()` (311_observational_rerun.py L169–223) は UTC 8/13/14/16 vs 他を比較。データの大部分は pre-310# であり、boost 実装 (310# A) 前のデータで「boost 対象時間帯」と「非対象」を比較しているだけ。

```
比較しているもの: 「元々 AS が高い時間帯」vs「元々 AS が低い時間帯」
比較すべきもの:  「同一時間帯の pre-310# vs post-310#」
```

312# の推奨「同一時間帯内の pre/post 比較」は正当。

### F4: 動的フロア割引の因果推論の弱さ [HIGH] — ✅ 妥当

**検証結果**: offset quintile Q1 (0.136–0.268) の劣後は観察事実として強いが、Q1 には以下の交絡因子が混在:
- 在庫 buy 偏重時 (inv_discount 発動条件) のマーケット状態
- regime 分布の偏り
- 時間帯の偏り

312# の推奨「ranging × 危険時間帯 × buy 偏重在庫」での層別分析は正当。

### F5: 308# L1 批判の過度一般化 [MEDIUM] — ✅ 妥当

L2 (microprice side) は明確な理論倒錯 → 309# で修正済み。
L1 (dynamic cycle interval) は「この bot では妥当だが一般論としては絶対ではない」という 312# の整理が正確。

### F6: None regime 追加対策の前提 [MEDIUM] — ✅ 妥当

303# で `passive_mm_enabled` 実装、306# で有効化済み。311# の none 分析はこれ以前のデータを含み、`312-C (none × 1.3)` の根拠は弱い。312# の推奨「post-303/post-310 限定で再集計」は正当。

### F7: 309#/310# の安全性と収益性の区別 [MEDIUM] — ✅ 妥当

統計的安全性確認 (p=0.96) ≠ 収益改善確認。「安全化として評価、収益改善はまだ」と整理すべき。

### F8: 308# 制御文字混入 [LOW] — ✅ 妥当

文書品質の問題。優先度は低い。

---

## §3 313# (Gemini) 各主張の妥当性検証

### 2.1 Spread Capture 転倒 — ✅ 妥当 (F2 と同一)

312# F2 と同じ指摘。コード検証で確認済み。

ただし 313# は mid 基準 (`0.5 - ratio`) ではなく best price 基準 (`1.0 - ratio`) を推奨している。MM 理論の文脈では **mid 基準が標準** (Avellaneda-Stoikov)。どちらを採用するかは明確に定義して統一すべき。

### 2.2 Volatility Guard「自発的自殺」— ⚠️ 方向性は妥当、表現は過剰

**検証結果**: VG は `effective_offset_ratio` を `_raw_boost` (> 1.0) 倍にする (maker_price.py L1191)。

ratio が増加すると:
- buy: `price = best_bid + spread × ratio` → 価格上昇 → mid に接近 → **より攻撃的**
- sell: `price = best_ask - spread × ratio` → 価格下降 → mid に接近 → **より攻撃的**

高 vol 時に VG がクオートをより攻撃的にする = Avellaneda-Stoikov の推奨 (高 vol → スプレッド拡大) と逆方向。

**ただし以下の理由で「自発的自殺」は過剰表現**:

1. VG の実際の boost 係数は `volatility_guard_offset_boost_factor` (YAML 設定、デフォルト ~1.5) × VPN damping。ratio を 0.05→0.075 にする程度で、0.5 超え (mid 超え) には通常至らない
2. 実運用データでは sell ratio の mean=0.446、つまり VG がなくても既に mid 近辺。VG が追加する分は限定的
3. 本当の問題は VG 単体ではなく、**パイプライン全体の加算方向** (後述 §4 B1)

### 3.1 Fill Rate 過剰最適化 — ⚠️ 部分的に妥当

AB テストは fill_rate / avg_pnl / downside_p10 の 3 基準で判定しており、fill_rate 単体への最適化ではない。ただし offset パイプライン全体が「ratio を上げる = 攻撃的にする = fill_rate を上げる」方向に偏っているのは事実。

### 3.2 Inventory Skewing の Time Horizon 無視 — ✅ 妥当

在庫偏重時の offset 調整は比例制御 (`imbalance × factor`) で、保持期間を考慮していない。理論的な gap として妥当な指摘。ただし即座の修正優先度は低い。

### 「311# 白紙撤回」— ❌ 過剰

311# で spread capture が関与するのは §6 (Spread/AS 分解) と `derive_improvement_proposals()` のみ。以下は spread capture に依存しておらず引き続き有効:
- §3 理論修正検証 (Bootstrap/Matched → p, PnL 直接比較)
- §4 AB 判定結果 (fill_rate, avg_pnl, downside_p10)
- §5 Regime 別深堀り (n, fill_rate, PnL, p10)
- §7 時間帯別 AS 構造 (PnL, p10, AS 率)
- §8 Offset 分位点 (PnL, AS 率)
- §9 None regime (PnL, AS 率)

---

## §4 盲点: 両レビューが見落とした問題

### B1: offset ratio セマンティクスの全体的転倒 [CRITICAL]

312# F2 は「分析コードの数式」に焦点を当て、313# §2.2 は「VG の方向」に焦点を当てた。しかし問題は VG だけではない。**プロダクションコードの offset パイプライン全体のコメントと意図が、コードの実際の動作と逆方向である**。

| 機能 | コメントの意図 | ratio への操作 | 実際の効果 | 方向 |
|---|---|---|---|---|
| VG (high vol) | AS 防御: offset 拡大 | ratio × boost (↑) | mid 接近 → 攻撃的 | ❌ 逆 |
| high_vol regime | AS 防御: offset boost | ratio × boost (↑) | mid 接近 → 攻撃的 | ❌ 逆 |
| trending_up sell boost 5.4x | 逆選択防御 | ratio × 5.4 (↑) | mid 接近 → 攻撃的 | ❌ 逆 |
| sell_hour_boost | 危険時間帯防御 | ratio × 1.3–1.5 (↑) | mid 接近 → 攻撃的 | ❌ 逆 |
| ranging discount | 安定時に利幅確保 | ratio × discount (↓) | mid 離反 → 保守的 | ❌ 逆 |
| AS reservation shift | AS 防御 | ratio ↑ | mid 接近 → 攻撃的 | ❌ 逆 |
| sell_offset_floor 0.30 | 最低 AS 保護 | ratio ≥ 0.30 | mid の 20% 手前 | ⚠️ 保護は薄い |

**全ステージが「防御のつもりで攻撃的にしている」可能性がある。**

ただし重要な留保: この解釈が正しければ、offset floor 0.30 は「ratio 0.30 以上 → sell price が best_ask の 30%+ 内側」を保証するもので、spread capture = spread × (0.5 - 0.30) = spread × 0.20。これは mid から 20% の capture であり、**ゼロではない**。つまりシステムは「壊滅的に破綻」しているわけではなく、「防御が意図の半分しか効いていない」状態。

**プロダクションへの影響評価**:
- 現行のクオート挙動そのものは変更されていない (分析式のエラーとは独立)
- ただし「高リスク時に防御を強化する」つもりのロジックが「高リスク時にさらに攻撃的にする」結果になっている可能性
- 実際の sell ratio mean=0.446 は **mid まであと spread×0.054 しかない**。これは正常な MM ではない

### B2: offset_ceiling_ratio (0.15) が未適用の疑い [HIGH]

**実データ検証**: 直近 197 sell filled records のうち、effective_offset_used ≤ 0.15 は **わずか 1 件**。残り 196 件は全て > 0.15。最新 SHA (dcc3064a8, n=5) でも 5/5 が > 0.15。

```
YAML: offset_ceiling_ratio: 0.15
実データ: min=0.136, max=2.088, median=0.313
```

ceiling 0.15 が正常に機能していれば、全 sell record は ≤ 0.15 のはず。しかしデータは ceiling が **適用されていない**ことを示す。

可能性:
1. **ceiling 適用後に repricing が ratio を再拡大** (fill_cycle_executor L562: `effective_offset_ratio * offset_mult`)
2. **none_regime passive MM がパイプラインをバイパス** (ceiling スキップ)
3. **config のロード順序で ceiling が後から上書き**

ceiling が死んでいる場合、sell floor (0.30) + 各種 boost が制限なく積算され、ratio が 0.4–2.0 に達している。これが sell p10 = -6.87 bps の構造的原因の可能性。

### B3: 312# / 313# の recommended formula の不一致 [MEDIUM]

312# は `(0.5 - ratio) × spread_bps` (mid 基準)、313# は `(1.0 - ratio) × spread_bps` (best price 基準) を推奨。同一レビューチェーン内で formula が不一致。

Avellaneda-Stoikov 理論では **mid 基準** が標準。ただし:
- `post_fill_30s_pnl` は fill 価格と 30 秒後の mid の差 → PnL は mid 基準
- 分解式: `realized_pnl = spread_capture_mid - AS_cost_mid`
- よって `spread_capture = spread × (0.5 - ratio)` が整合する

### B4: Spread 変動下での spread_capture 分解の限界 [LOW]

`spread_bps` は fill 時点での spread。しかし `post_fill_30s_pnl` は 30 秒後の mid 変動を含む。高 vol 時は spread 自体が急変し、fill 時の spread 基準の分解は近似にすぎない。両レビューともこの点に触れていない。

---

## §5 影響範囲の整理: 分析コード vs プロダクションコード

| 問題 | 分析コード | プロダクションコード | 備考 |
|---|---|---|---|
| F2: spread capture 式転倒 | ✅ 修正必要 | ❌ 無関係 | 分析結果のみ影響 |
| B1: ratio セマンティクス転倒 | — | ⚠️ 要調査 | **全 boost ロジックの方向性** |
| B2: ceiling 未適用 | — | ⚠️ 要調査 | パイプライン制限が死んでいる |
| F1: mixed-SHA | ✅ フィルタ追加 | ❌ 無関係 | |
| F3: hour boost 測定設計 | ✅ redesign | ❌ 無関係 | |
| F4: offset quintile 交絡 | ✅ 層別追加 | ❌ 無関係 | |

**最も重要な区別**: F2 は分析コードのみの issue。B1/B2 はプロダクションコードへの影響がある。B1 が確定すれば **パイプラインの方向転換** が必要になり、v460 のアーキテクチャ変更レベルの作業。

---

## §6 実行計画

### Phase 0: 即時対応 (本セッション)

| # | タスク | 対象 | リスク | 根拠 |
|---|---|---|---|---|
| **T0-1** | spread capture 式の修正 | `analysis/311_observational_rerun.py` L237 | 低 (分析のみ) | F2/2.1 |
| **T0-2** | efficiency ベース優先度判定の無効化 | `analysis/311_observational_rerun.py` derive_improvement_proposals | 低 | F2 派生 |
| **T0-3** | SHA/date フィルタ導入 | `analysis/311_observational_rerun.py` | 低 | F1 |
| **T0-4** | sell_hour_boost_analysis redesign | `analysis/311_observational_rerun.py` | 低 | F3 |
| **T0-5** | 修正スクリプトの再実行 + 311# 更新 | analysis + docs | 低 | 全体 |

### Phase 1: プロダクションコード調査 (次セッション)

| # | タスク | 対象 | リスク | 根拠 |
|---|---|---|---|---|
| **T1-1** | ceiling 未適用の根本原因調査 | `maker_price.py` + `fill_cycle_executor.py` | 中 | B2 |
| **T1-2** | offset ratio セマンティクス精査 | `maker_price.py` 全パイプライン | 高 | B1 |
| **T1-3** | ratio 方向逆転の影響シミュレーション | バックテスト or 分析 | 中 | B1 |
| **T1-4** | post-310# データでの各指標再集計 | `analysis/311_observational_rerun.py` | 低 | F1 |

### Phase 2: プロダクション修正 (T1 結果次第)

| # | タスク | 対象 | リスク | 根拠 |
|---|---|---|---|---|
| **T2-1** | ceiling 修正 (実際に ceiling を適用させる) | `maker_price.py` or `fill_cycle_executor.py` | 高 | B2 |
| **T2-2** | boost 方向の修正検討 (`ratio ↑` → `ratio ↓`) | `maker_price.py` パイプライン | **極高** | B1 |
| **T2-3** | None regime post-310 限定再評価 + veto 検討 | YAML + コード | 中 | F6 |
| **T2-4** | offset quintile 層別分析 (regime × hour × inv) | 分析 | 低 | F4 |

### G1.2 への影響

| Phase | G1.2 168h リセット | 判断 |
|---|---|---|
| Phase 0 | なし (分析コードのみ) | ✅ 即実行可 |
| Phase 1 | なし (調査のみ) | ✅ 即実行可 |
| Phase 2 T2-1/T2-2 | **リセット** | ⚠️ T1 結果次第で判断 |

### T2-2 (boost 方向転換) の判断基準

B1 が確定した場合の対応方針:

**Option A: 全 boost を ratio ↓ 方向に反転**
- 理論的に正しい (A-S: 高リスク → スプレッド拡大)
- 影響範囲が広大 → テスト工数大
- fill_rate 低下リスク

**Option B: ceiling を修正して ratio を 0.15 以下に強制**
- boost 方向はそのまま (上向き) だが ceiling で抑制
- 影響範囲が小さい
- 問題の本質を解決しない

**Option C: 段階的切替 — まず ceiling 修正、次に boost 方向転換**
- リスク最小
- A の効果を後段で検証可能
- 推奨

---

## 付録: 検証で参照したコード

| ファイル | 行 | 内容 |
|---|---|---|
| `scripts/v460/lib/maker_price.py` | L588–605 | `_finalize_price_with_spread_guard` — buy=bid+offset, sell=ask-offset |
| `scripts/v460/lib/maker_price.py` | L1107–1210 | `_apply_volatility_guard` — ratio × boost |
| `scripts/v460/lib/maker_price.py` | L1515–1682 | offset パイプライン全体 (13 ステージ + ceiling) |
| `scripts/v460/lib/maker_price.py` | L1294–1330 | `_apply_sell_hour_boost` — ratio × mult |
| `scripts/v460/lib/maker_price.py` | L391–413 | `_effective_sell_offset_floor` — 動的フロア |
| `scripts/v460/lib/maker_price.py` | L774–870 | `_apply_regime_boosts` — regime 別 boost |
| `scripts/v460/lib/fill_cycle_executor.py` | L315, L329 | fill record への ratio 記録 |
| `scripts/v460/lib/fill_cycle_executor.py` | L525–562 | repricing による ratio 変更 |
| `scripts/v460/lib/fill_config.py` | L292, L456, L1918 | config defaults + YAML binding |
| `analysis/311_observational_rerun.py` | L237 | **転倒した sc_bps 計算** |
| `analysis/311_observational_rerun.py` | L169–223 | 時間帯比較 (F3 の問題箇所) |
| `ztb/metrics/fill_quality.py` | L1383–1430 | `apply_fill_record_filters` (既存、未使用) |
| `configs/v460/fill_test.yaml` | L454, L506 | sell floor=0.30, ceiling=0.15 |

## 付録: 実データ検証結果

```
直近 5 JSONL の sell filled records (n=197):
  Min effective_offset_used:  0.136427
  Max effective_offset_used:  2.087689
  Mean:                       0.446004
  Median:                     0.313029
  > 0.15 (ceiling 超え):      196/197 (99.5%)
  = 0.30 (floor 固定):        1/197

SHA dcc3064a8 (310# 最新): n=5, 全て > 0.15
```
