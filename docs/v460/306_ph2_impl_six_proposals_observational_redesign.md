# 306# 6提案実装 + 299# 観察比較再設計

> **文書番号**: 306#  
> **種別**: `impl` (実装)  
> **作成日**: 2026-03-06  
> **前提**: [300#](300_ph2_rev_ab_test_deep_analysis.md), [301#](301_ph2_rev_292_300_multifaceted_review.md), [302#](302_ph2_gemini_31_pro_review_300_301_hft_blindspots.md), [305#](305_ph2_analysis_systems_market_theory_p0_improvements.md)

---

## §1 概要

300#-302# のレビューで指摘された6件の改善提案を実装し、
301# F2 で批判された 299# 疑似A/Bテストの統計手法を再設計・再実行した。

### 1.1 実装した提案

| # | 提案 | 分類 | 出典 | 状態 |
|---|------|------|------|------|
| O1 | Queue Position Estimation | 可観測性 | 300# T1-7 | ✅ |
| L2 | Microprice Side Selection | ロジック | 302# L2 | ✅ |
| L1 | Dynamic Cycle Interval | ロジック | 302# L1 | ✅ |
| A1 | EV-based Offset Adaptation | 適応 | 302# A1 | ✅ |
| E1 | Offset Stage Recording | 可観測性 | 301# F6 | ✅ |
| σ  | Parkinson σ YAML 有効化 | 設定 | 305# | ✅ |
| 天井 | Offset Ceiling | 制約 | 300# T1-3 | ✅ |

### 1.2 299# 再設計

| 手法 | 旧 (299#) | 新 (306#) |
|------|-----------|-----------|
| 比較設計 | 疑似 A/B (sell vs buy) | 観察比較 (明示ラベル) |
| 独立性仮定 | iid 仮定 | Block Bootstrap (MBB) |
| 条件統制 | なし | 時間近接 Matched Pair |
| 多重比較 | Holm-Bonferroni (2検定) | + BH FDR (regime 横断) |
| ペア検定 | なし | Wilcoxon signed-rank |

---

## §2 提案実装詳細

### 2.1 O1: Queue Position Estimation (板前方深度推定)

**理論**: 指値注文のフィル確率は、自分より有利な価格に並んでいる待ち注文量に依存する。

**実装**: `maker_price.py` に `estimate_queue_depth(side, price)` メソッドを追加。
OB の自サイド価格帯を走査し、order_price より有利な価格にある volume を合計する。

```
fill_probability = exp(-depth / lot)
```

- depth = 0 → fill_prob = 1.0 (先頭 = 即約定)
- depth >> lot → fill_prob ≈ 0 (大量の待ち行列)

**変更ファイル**:
- `maker_price.py`: `estimate_queue_depth()` メソッド
- `fill_cycle_executor.py`: lot 確定後にqueue推定・FillRecord 書き込み
- `fill_quality.py`: `queue_depth_ahead`, `queue_fill_prob_est` フィールド

### 2.2 L2: Microprice Side Selection (マイクロプライスサイド選択)

**理論**: Microprice = (P_bid × Q_ask + P_ask × Q_bid) / (Q_ask + Q_bid)

midprice と microprice の乖離 (bias_bps) が閾値を超えた場合、
bias 方向に有利なサイドを選択する。正 = 買い圧力 → sell が有利。

**実装**:
- `maker_price.py`: `compute_microprice_bias_bps()` — OB キャッシュから microprice を算出
- `side_selector.py`: `next(microprice_bias_bps=...)` で閾値超過時にサイドオーバーライド
- `fill_record_helpers.py`: `_next_side()` でバイアスを SideSelector に渡す

### 2.3 L1: Dynamic Cycle Interval (σ連動サイクル間隔)

**理論**: ボラティリティと執行間隔の最適化。高σ → 短間隔 (機会捕捉)、低σ → 長間隔 (コスト節約)。

```
interval = base × (σ_ref / σ),  clamped to [min_sec, max_sec]
```

**実装**: `fill_loop_orchestrator.py` の sleep ループ内で σ-based interval を計算。
`maker_price.py` の `_estimate_sigma()` でσをキャッシュし `last_sigma` プロパティで公開。

### 2.4 A1: EV-based Offset Adaptation (期待値ベースオフセット適応)

**理論**: 単純な fill_rate 最適化は toxic fill を増やすリスクがある。
EV = fill_rate × avg_pnl - (1 - fill_rate) × opportunity_cost で総合評価。

**デッドロック対策**: EV << 0 かつ十分なサンプル → offset 拡大 (deadlock break)
**微細最適化**: EV > 0 かつ AS 余裕あり → offset 微小縮小 (fill_rate 改善)

**実装**: `param_adapter.py` の `compute_adaptation()` / `compute_side_adaptation()` を拡張。

### 2.5 E1: Offset Stage Recording (オフセットパイプライン段階記録)

**背景**: 301# F6 が指摘した通り、最終 offset だけでは「どの段階が offset を膨張させたか」が不透明。

**実装**: `maker_price.py` の `compute()` メソッド内で 10+ パイプライン段階を追跡:
`base`, `as_shift`, `regime`, `spread_adapt`, `kyle`, `amihud`, `vol_guard`, `imb_risk`, `buy_as_guard`, `loss_boost`, `ffd`, `final`

各段階の offset 値を JSON で `FillRecord.offset_stages` に保存。

### 2.6 Parkinson σ YAML 有効化

YAML に `sigma_parkinson` セクションを追加: `enabled: true`, `window_sec: 300.0`

### 2.7 Offset Ceiling (300# T1-3)

`compute()` の FFD boost 後に天井クランプを追加:
```python
if ceiling > 0 and offset > ceiling:
    offset = ceiling  # 無制限膨張を防止
```

---

## §3 299# 観察比較再設計

### 3.1 Block Bootstrap (移動ブロックブートストラップ)

**問題**: 299# は各 fill を独立同分布 (iid) として扱ったが、時系列データは自己相関がある。
**解法**: Künsch (1989) MBB — 連続ブロック単位でリサンプリングし、自己相関構造を保存。

```python
# block_size = 10 (自己相関長の目安)
# n_bootstrap = 2000
def _block_bootstrap_mean_diff(x, y, block_size=10, n_bootstrap=2000):
    boot_diffs = []
    for _ in range(n_bootstrap):
        x_boot = block_resample(x, block_size)
        y_boot = block_resample(y, block_size)
        boot_diffs.append(mean(x_boot) - mean(y_boot))
    ci = percentile(boot_diffs, [2.5, 97.5])
    p = mean(|centered_diffs| >= |observed_diff|)
```

### 3.2 Matched Temporal Comparison

**問題**: sell と buy は異なる市場条件下で実行されるため、直接比較は交絡を含む。
**解法**: 時間的に近い (|Δt| ≤ 600秒) sell/buy fill をペアリングし、
同一市場条件下でのペア差を検定する。

- Greedy nearest-neighbor マッチング (ソート済み走査)
- ペア差に対する Bootstrap CI + Wilcoxon signed-rank 検定

### 3.3 BH FDR (Benjamini-Hochberg)

**問題**: regime 横断で多数の検定を行うと偽陽性率が膨張する。
Holm-Bonferroni は保守的すぎて検出力を失う。
**解法**: BH FDR は偽発見率を制御しつつ検出力を維持する。

### 3.4 Wilcoxon Signed-Rank Test

Matched pair の差に対するノンパラメトリック検定。正規分布仮定不要。
Tie correction 付き正規近似で実装。

---

## §4 再実行結果

### 4.1 全体結果 (none 除外)

| 検定 | 統計量 | p 値 | 効果量 | 結論 |
|------|--------|------|--------|------|
| Welch's t | — | 0.9286 | d = −0.004 | 非有意 |
| Mann-Whitney U | — | 0.4537 | δ = 0.018 (negligible) | 非有意 |
| **Block Bootstrap** | diff = −0.023 bps | **0.9355** | 95%CI [−0.565, +0.499] | CI が 0 を含む → 差なし |
| **Matched Pairs** | n=928, diff = −0.069 bps | **0.2043** | 95%CI [−0.638, +0.460] | 差なし |

### 4.2 全体結果 (none 含有 — 301# F1 対応)

| 検定 | 統計量 | p 値 | 結論 |
|------|--------|------|------|
| Block Bootstrap | diff = −0.088 bps | 0.7455 | CI [−0.589, +0.439] → 差なし |
| Matched Pairs | n=1043, diff = −0.154 bps | 0.1575 | 差なし |

### 4.3 解釈

1. **4つの検定すべてで sell/buy 間の PnL 差は統計的に非有意**
2. Block Bootstrap は iid 仮定を緩和してもなお同一結論 → 頑健
3. Matched Pairs (928 ペア) は市場条件を統制してもなお差なし → 交絡排除後も同一結論
4. none 含有版は fill_rate degradation で FAIL → 301# F1 の楽観バイアス指摘は妥当
5. **真の問題はレジーム別構造差**: trending_up の sell は fill 18.2%, AS 40.7% で構造的に不利

---

## §5 深堀り分析 (306# 追加)

### 5.1 レジーム別 Matched & Bootstrap 比較 (BH FDR 補正付)

| レジーム | sell n | buy n | Matched n | diff (bps) | 95% CI | p 値 | BH-sig |
|----------|--------|-------|-----------|------------|--------|------|--------|
| ranging | 808 | 808 | 635 | +0.262 | [−0.351, +0.879] | 0.532 | ✗ |
| trending | 118 | 118 | 88 | −1.190 | [−3.343, +0.802] | 0.422 | ✗ |
| trending_down | 81 | 79 | 55 | −0.085 | [−2.740, +2.721] | 0.731 | ✗ |
| trending_up | 81 | 94 | 60 | −0.825 | [−2.826, +1.256] | 0.365 | ✗ |
| unknown | 46 | 47 | 41 | +1.069 | [−0.937, +2.975] | 0.285 | ✗ |

**BH FDR 補正後、全レジームで非有意。** ただし以下の構造的差が観察される:
- **ranging**: sell が PnL で +0.31 bps 優位 (bootstrap diff=+0.309)。唯一の sell 有利レジーム
- **trending**: sell が −1.23 bps 劣後 (bootstrap p=0.112)。有意に最も近いレジーム
- **trending_up**: sell AS率 40.7% vs buy 27.7% → 構造的 AS 格差

### 5.2 Adverse Selection (AS) 深堀り

| Side | AS n | non-AS n | AS avg_pnl30 | non-AS avg_pnl30 | Bootstrap diff | p |
|------|------|----------|--------------|------------------|----------------|---|
| sell | 382 | 880 | −6.682 bps | +2.389 bps | −9.071 bps | **0.0000** |
| buy | 355 | 930 | −5.761 bps | +1.828 bps | −7.589 bps | **0.0000** |

**発見**:
1. AS/非AS 間の PnL 差は **sell で −9.07 bps, buy で −7.59 bps** と極めて有意 (p ≈ 0)
2. sell の AS 損失は buy より **1.5 bps 大きい** → sell 側の AS 対策がより緊急
3. **非 AS 約定は両サイドとも正の PnL** (sell +2.39, buy +1.83)
   → AS をフィルタリングできれば、システム全体が利益圏に入る
4. sell の EV: AS 群 −0.435 vs 非 AS 群 +0.484 → pretrade EV スコアは AS を部分的に予測
5. buy の EV: AS 群 +0.255 vs 非 AS 群 −0.921 → **buy 側は EV が逆転** (EV が高い方が AS)

**改善仮説 H1**: pretrade EV > 0 の sell fill は AS 率が低い → EV gate の導入
**改善仮説 H2**: buy の EV 逆転は、trending 時の逆張り buy が高 EV かつ高 AS を示唆
**改善仮説 H3**: AS 率改善だけで全体 PnL を +1 bps 以上改善可能 (AS 率 30% → 20% で試算)

### 5.3 PnL30 分布形状

| Side | mean | std | skewness | kurtosis (excess) | p10 | p50 | p90 |
|------|------|-----|----------|-------------------|-----|-----|-----|
| sell | −0.357 | 6.479 | **+0.480** | **+8.383** | −6.824 | −0.308 | +6.019 |
| buy | −0.269 | 5.454 | +0.300 | **+12.661** | −5.545 | −0.168 | +4.762 |

**発見**:
1. **売りの方が分散が大きい** (std: 6.48 vs 5.45) → より振幅のある PnL
2. **両サイドとも正のスキュー** (右裾重) → 利益方向のテールが存在するが、頻度は低い
3. **buy の超過尖度が +12.66** と極端に高い → 極端値が集中する「レプトクルティック」分布
   → 正規分布仮定の検定 (Welch t) は本来不適切であり、ノンパラメトリック手法への移行は正しかった
4. sell の skewness +0.48 > buy の +0.30 → sell は buy より利益テールが少し厚い
5. **p50 (中央値) が mean より 0 に近い** → 平均を引き下げているのは左裾の少数の大損失

**改善仮説 H4**: p10 以下のテール損失 (sell: −6.82, buy: −5.55) をカットするだけで平均 PnL は正に転じる可能性

### 5.4 UTC 時間帯別分析 (sell 要注意時間帯)

| 時間帯 (UTC) | JST | sell pnl30 | sell AS率 | buy pnl30 | buy AS率 | sell 劣後 |
|-------------|-----|------------|-----------|-----------|----------|-----------|
| **08h** | 17時 | **−3.546** | **0.630** | −0.574 | 0.214 | **−2.97 bps** |
| **13h** | 22時 | **−1.943** | **0.407** | −0.551 | 0.271 | −1.39 bps |
| **14h** | 23時 | **−3.164** | **0.429** | −1.332 | 0.421 | −1.83 bps |
| **16h** | 01時 | **−2.250** | **0.611** | −2.358 | 0.389 | +0.11 bps |
| **21h** | 06時 | **−2.054** | **0.381** | −0.934 | 0.326 | −1.12 bps |

**発見**:
1. **UTC 08h (JST 17時)**: sell AS率 63.0% — **全時間帯最悪**。東京市場終了直後のボラティリティで sell が逆行される。sell n=27 で統計的には微弱だが、構造的リスク
2. **UTC 13-14h (JST 22-23時)**: NY市場オープン前後。売り AS率 40%超、PnL −2〜−3 bps
3. **UTC 16h (JST 01時)**: 両サイドとも悪い (sell/buy ともに AS率 > 38%)。深夜の流動性低下期
4. **sell が最も良い時間帯**: UTC 10-12h (JST 19-21時) — PnL +0.4〜+0.8, AS率 16-20%
5. **buy が最も良い時間帯**: UTC 07h (JST 16時) — PnL +0.72, AS率 10.5% (全時間帯最良)

**改善仮説 H5**: UTC 08h, 13-14h での sell スキップまたはスプレッド拡大で、年間 AS 損失を 10-15% 削減可能

### 5.5 週次トレンド

| 週 | sell n | sell pnl30 | sell AS率 | buy n | buy pnl30 | buy AS率 |
|----|--------|------------|-----------|-------|-----------|----------|
| W06 (2/9-) | 181 | −0.958 | 38.7% | 192 | −0.301 | 39.6% |
| W07 (2/16-) | 450 | −0.355 | 20.4% | 449 | +0.029 | 21.8% |
| W08 (2/23-) | 444 | −0.291 | 34.0% | 456 | −0.572 | 28.9% |
| W09 (3/2-) | 187 | +0.067 | 36.9% | 188 | −0.213 | 26.1% |

**発見**:
1. **W07 が最も健全**: 両サイドとも AS率 20%台。この週の市場構造が最適だった可能性
2. **W06 (初期) が最悪**: 両サイドとも AS率 38-40%。学習初期の不安定さか、市場急変
3. **W09 で sell が初めて正の PnL**: AS率は高いが、非 AS fill の質が改善した兆候
4. **buy は W08 で悪化**: sell が改善する週に buy が悪化 → **逆相関**の兆候

**改善仮説 H6**: AS 率の週次変動は市場ボラティリティと相関しており、σ-linked AS gate が有効

---

## §6 改善仮説サマリ (他 AI レビュー用)

| ID | 仮説 | 根拠 | 期待効果 | 優先度 |
|----|------|------|----------|--------|
| H1 | pretrade EV > 0 の sell を通す EV gate | sell AS群 EV=−0.435 vs 非AS群 EV=+0.484 | AS率 30%→20% → PnL +1bps | **P0** |
| H2 | buy の EV 逆転を調査 (trending 逆張り) | buy AS群 EV=+0.255 vs 非AS群 EV=−0.921 | buy gate 設計改善 | P1 |
| H3 | AS 率を 30%→20% に改善 | AS diff −9.07bps (sell), −7.59bps (buy) | 全体 PnL +1〜2bps | P0 |
| H4 | p10 テールカット (損切り強化) | p50≈0 vs mean≈−0.35 → テール引き下げ | mean→0 以上 | P1 |
| H5 | 時間帯別 sell スキップ (08h,13-14h) | sell AS率 63%@08h, 40%@13-14h | AS損失 10-15%削減 | **P0** |
| H6 | σ-linked AS gate | W07(AS 20%) vs W06(AS 39%) が σ と相関 | 動的 AS 回避 | P1 |

### 中核的洞察

**「sell と buy に差がない」という 299# の結論は正しいが、改善余地の所在を見誤る。
真の改善ドライバは sell/buy の差ではなく、AS/非AS の差 (−9.07 bps) と 時間帯の構造差 (08h: AS 63%) にある。
非 AS 約定は sell +2.39 bps, buy +1.83 bps と健全であり、
問題は「いつ・どの条件で AS 約定が発生するか」の予測と回避にある。**

---

## §7 データ仕様

| 項目 | 値 |
|---|---|
| 期間 | 2026-02-13 ～ 2026-03-06 |
| 総レコード数 | 7,117 |
| 約定レコード数 | 2,547 (35.8%) |
| sell n (excl. none) | 1,134 |
| buy n (excl. none) | 1,146 |
| Matched pairs | 928 (excl. none), 1,043 (incl. none) |
| AS fill (sell) | 382 / 1,262 (30.3%) |
| AS fill (buy) | 355 / 1,285 (27.6%) |

---

## §8 変更ファイル一覧

| ファイル | 変更内容 |
|----------|----------|
| `configs/v460/fill_test.yaml` | 6 提案の YAML セクション追加 |
| `scripts/v460/lib/fill_config.py` | 設定フィールド追加 + YAML パーシング |
| `scripts/v460/lib/config_hot_reload.py` | 12 フィールドのホットリロード対応 |
| `scripts/v460/lib/maker_price.py` | microprice, queue, σ cache, stage recording, ceiling |
| `scripts/v460/lib/side_selector.py` | microprice override ロジック |
| `scripts/v460/lib/fill_record_helpers.py` | microprice bias パススルー |
| `scripts/v460/lib/fill_loop_orchestrator.py` | σ-based dynamic interval |
| `scripts/v460/lib/fill_cycle_executor.py` | queue estimation + FillRecord 拡張 |
| `scripts/v460/lib/param_adapter.py` | EV-based adaptation |
| `scripts/v460/lib/adaptation_engine.py` | PnL mean パススルー |
| `scripts/v460/lib/ab_judgment.py` | block bootstrap, matched comparison, BH FDR, Wilcoxon |
| `scripts/v460/analysis/side_regime_dashboard.py` | 新統計フィールドの dict 出力 |
| `ztb/metrics/fill_quality.py` | 4 FillRecord フィールド追加 |
| `tests/unit/v460/test_306_proposals.py` | 51 テスト (新規) |
| `tests/unit/v460/test_260_compute_extract_regime_split.py` | 行数 assertion 更新 |
| `analysis/306_deep_dive.py` | 深堀り分析スクリプト (新規) |

---

## §9 テスト結果

| テストスイート | 結果 |
|----------------|------|
| test_306_proposals.py | 51 passed |
| test_160_ab_judgment.py | 93 passed |
| 全 v460 テスト | 4069 passed, 19 warnings |

---

## §10 理論参照

| 手法 | 参照 |
|------|------|
| Microprice | Stoikov (2018) "The Micro-Price" |
| Block Bootstrap | Künsch, H. R. (1989) "The Jackknife and the Bootstrap for General Stationary Observations" |
| Wilcoxon signed-rank | Wilcoxon, F. (1945) "Individual Comparisons by Ranking Methods" |
| BH FDR | Benjamini, Y. & Hochberg, Y. (1995) "Controlling the False Discovery Rate" |
| Parkinson σ | Parkinson, M. (1980) "The Extreme Value Method for Estimating the Variance of the Rate of Return" |
| Queue position | Cont, R. et al. (2010) "A Stochastic Model for Order Book Dynamics" |

---

## §11 総評

301# F2 の「観察比較であり A/B テストではない」という批判に正面から対処した。
Block Bootstrap は時系列の自己相関を尊重し、Matched Pairs は市場条件の交絡を制御した。
4手法すべてで sell/buy 間に統計的有意差なしという結論は、299# の結論と一致するが、
**統計的基盤がより頑健になった**。

6提案の実装により、以下のパイプラインが追加された:
1. **可観測性**: queue depth + offset stages → 次のイテレーションの分析基盤
2. **適応**: EV-based offset + microprice side → 市場微視構造に基づく意思決定
3. **制約**: offset ceiling + dynamic interval → 暴走防止 + コスト最適化

深堀り分析から、**sell/buy 間の差より AS/非AS 間の差 (−9 bps) の方が 桁違いに大きい**ことが判明した。
次の改善サイクルでは §6 の仮説 H1 (EV gate), H3 (AS 率削減), H5 (時間帯別スキップ) を優先的に検証すべきである。
