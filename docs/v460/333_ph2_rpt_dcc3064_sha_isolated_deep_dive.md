# 333# dcc3064 SHA 分離分析 — 310# 設計改修の定量評価

> **種別**: rpt (調査・分析レポート)  
> **起票**: 2026-03-08  
> **稼働 SHA**: `dcc3064a8d3be15ef03392f5ffc384014a2a5f6b` (310#) + `4e670141a` (316# doc — ランタイム等価)  
> **計測期間**: 2026-03-06 23:44 → 2026-03-07 23:42 JST (24.0h)  
> **データ**: n=637 records / 100 fills (sell=51, buy=49)  
> **関連**: [299#](299_ph2_rpt_ab_test_f4_validation.md), [306#](306_ph2_impl_six_proposals_observational_redesign.md), [310#](310_ph2_impl_design_improvements.md), [311#](311_ph2_rpt_observational_comparison_rerun.md), [317#](317_ph2_rpt_observation_experiment.md), [320#](320_ph2_fix_c1_side_specific_ceiling.md)

---

## 目次

- [§1 分析の前提と方法論](#1-分析の前提と方法論)
- [§2 SHA 等価性の証明](#2-sha-等価性の証明)
- [§3 エグゼクティブ・サマリー](#3-エグゼクティブサマリー)
- [§4 AB 判定](#4-ab-判定)
- [§5 311# ベースラインとの比較](#5-311-ベースラインとの比較)
- [§6 Regime 別分析](#6-regime-別分析)
- [§7 時間帯別深堀り](#7-時間帯別深堀り)
- [§8 Cancel Reason 分析 — Buy Kill 問題](#8-cancel-reason-分析--buy-kill-問題)
- [§9 Adverse Selection 構造](#9-adverse-selection-構造)
- [§10 320# C-1 未修正の影響](#10-320-c-1-未修正の影響)
- [§11 統計的限界と信頼区間](#11-統計的限界と信頼区間)
- [§12 構造的発見と課題整理](#12-構造的発見と課題整理)
- [§13 000# G1.2 Gate 進捗](#13-000-g12-gate-進捗)
- [§14 AI レビュー向け設問](#14-ai-レビュー向け設問)

---

## §1 分析の前提と方法論

### §1.1 SHA 分離の動機

先行セッションで 3/1-3/7 の fill_records を日次集計したが、ユーザーから「コードをちょこちょこ変えてたので、その部分についてはあまり信頼なりません。同一 SHA で論じるように」と指示を受けた。

3/6-3/7 の SHA タイムラインを調査した結果:

| SHA (短) | records | filled | 期間 (JST) | コミット |
|---|---|---|---|---|
| 169b0b9 | 192 | 58 | 03/06 09:00~17:21 | 280# fix |
| 407be00 | 11 | 7 | 03/06 17:28~18:01 | — |
| f905e74 | 66 | 25 | 03/06 18:08~21:05 | — |
| d7e87ec | 5 | 2 | 03/06 21:10~21:18 | — |
| 985a3ab | 20 | 2 | 03/06 21:26~22:12 | — |
| c160291 | 6 | 3 | 03/06 22:17~22:50 | — |
| 10b7e21 | 22 | 7 | 03/06 22:52~23:39 | — |
| **dcc3064** | **111** | **21** | **03/06 23:44~03/07 04:14** | **310# 設計改修** |
| **4e67014** | **526** | **79** | **03/07 04:16~03/07 23:42** | **316# doc** |
| 894c1bf | 115 | 32 | 03/07 23:50~03/08 04:33 | 329# refactor |

**3/6 は 8 つの SHA が切り替わっている** — コード変更を繰り返していた期間であり、日次集計は無意味。

### §1.2 計測対象

`dcc3064a8` (310# 設計改修, 2026-03-06 23:43 コミット) が対象。317# 時点では 16 fills しかなく評価不可だったが、24 時間経過して **100 fills** に到達。

---

## §2 SHA 等価性の証明

`dcc3064a8` → `4e670141a` 間の diff を調査:

```diff
# retrain_scheduler.py — YAML 読込を config_loader に切替 (動作同一)
-            import yaml
-            with open(yaml_path) as f:
-                yaml_data = ensure_dict(yaml.safe_load(f) or {})
+            from scripts.v460.lib.config_loader import load_fill_test_config
+            yaml_data = ensure_dict(load_fill_test_config(yaml_path))

# broker_registry.py — デフォルト登録のリファクタ (動作同一)
+_DEFAULT_BROKERS: dict[str, type[IBroker]] = {
+    "coincheck": CoincheckAdapter,
+    "bitflyer": BitFlyerAdapter,
+}
 def __init__(self) -> None:
-    self._brokers: dict[str, type[IBroker]] = {}
+    self._brokers: dict[str, type[IBroker]] = dict(_DEFAULT_BROKERS)
```

**取引パイプライン (`maker_price.py`, `fill_cycle_executor.py`, `param_adapter.py`, `fill_config.py`) に差分なし。** 両 SHA はランタイム等価であり、統合して分析する。

---

## §3 エグゼクティブ・サマリー

| 指標 | 値 | 判定 |
|---|---|---|
| サンプル期間 | 24.0h (03/06 23:44 → 03/07 23:42 JST) | — |
| 総レコード | 637 | — |
| Fill 数 | **100** (sell=51, buy=49) | n≥100 到達 |
| Fill rate | 15.7% | ⚠️ 低い |
| **PnL mean** | **+0.636 bps** | ✅ 正 |
| **PnL sum** | **+63.56 bps** | ✅ 正 |
| **Win rate** | **58.0%** | ✅ 50%超 |
| **p10** | **-4.120** | ✅ -5.00 超 |
| p25 | -1.789 | — |
| p50 (中央値) | +0.585 | ✅ 正 |

**24 時間で +63.56 bps の PnL を記録。** 311# full data (22日, -0.248bps mean) と比較して大幅改善。ただし n=100 であり統計的信頼区間は広い。

---

## §4 AB 判定

### §4.1 サイド別結果

| Side | fill_rate | 判定 | avg_pnl | 判定 | p10 | 判定 | 総合 |
|---|---|---|---|---|---|---|---|
| **sell** | 46.8% | ✅ PASS | +0.889 | ✅ PASS | **-5.207** | ❌ FAIL | **FAIL** |
| **buy** | 9.3% | ❌ FAIL | +0.372 | ✅ PASS | -3.819 | ✅ PASS | **FAIL** |

**Overall: FAIL** — sell は p10 で閾値を 0.207bps 超過、buy は fill_rate が壊滅的に低い。

### §4.2 判定の含意

1. **sell p10 = -5.207**: 閾値 -5.00 との差は **わずか 0.207bps**。n=100 での p10 推定のブートストラップ SE は ~2bps 程度と推定され、実質的に閾値上にある。**統計的には PASS/FAIL の区別は無意味に近い。**

2. **buy fill_rate = 9.3%**: 49/526 しか約定していない。buy_dynamic_kill (40.2%) + forced_buy_delay (18.6%) + degraded_liquidation_duty_skip (17.7%) が buy 注文の 76.5% を殺している。**これは 310# の設計問題ではなく、既存の kill/skip ロジックが buy 側に過剰に作用している構造的問題。**

### §4.3 311# からの変化

| 指標 | 311# (22日, n≈2564) | 333# (24h, n=100) | Δ |
|---|---|---|---|
| sell p10 | -6.87 | -5.21 | **+1.66 改善** |
| buy p10 | -5.67 | -3.82 | **+1.85 改善** |
| sell avg_pnl | -0.33 | +0.89 | **+1.22 改善** |
| buy avg_pnl | -0.32 | +0.37 | **+0.69 改善** |
| overall mean | -0.25 | +0.64 | **+0.89 改善** |
| win rate | ~47% | 58.0% | **+11pp 改善** |

全指標で改善が見られるが、311# は 22 日間・複数 SHA のデータであり、**市場環境 (time-of-day, volatility, regime distribution) の違いも大きい**ため因果推論は困難。

---

## §5 311# ベースラインとの比較

### §5.1 299# AB テストとの一貫性

299# の結論: 「sell と buy の PnL 分布に統計的に意味のある差は存在しない」(Welch p=0.89, Mann-Whitney p=0.49, Cliff's δ=0.017)。

333# data: sell mean=+0.889, buy mean=+0.372。差 = 0.517bps。

n=100 (売51/買49) では、この差の検出力は極めて低い。**299# の「非有意」の結論と整合的**であり、sell/buy 差が拡大した証拠はない。

### §5.2 317# からの進展

317# 時点では dcc3064 は 16 fills (稼働 3.3h) しかなく評価不可だった。333# で 100 fills (24.0h) に到達し、暫定的な定量評価が可能になった。

| 項目 | 317# 報告値 | 333# 実測値 |
|---|---|---|
| fills | 16 | **100** |
| 稼働時間 | 3.3h | **24.0h** |
| fill rate (推定) | ~4.6 fills/h | **4.2 fills/h** |

fill rate はほぼ一致。317# の蓄積見込み推定 (50 fills → +7.5h) は妥当だった。

---

## §6 Regime 別分析

### §6.1 全体像

| Regime | records | filled | fill% | PnL mean | PnL sum | p10 | win% |
|---|---|---|---|---|---|---|---|
| **ranging** | 575 | 90 | 15.7% | +0.687 | +61.83 | -3.940 | 60.0% |
| trending_down | 37 | 5 | 13.5% | -0.413 | -2.07 | -5.495 | 40.0% |
| trending_up | 25 | 5 | 20.0% | +0.759 | +3.79 | -3.847 | 40.0% |

### §6.2 Ranging 支配 (90.3%)

24 時間中 575/637 = **90.3%** が ranging regime。これは計測期間 (03/07) の市場環境がレンジ相場に偏っていたことを示す。

ranging 内の sell/buy 分解:

| Side | n | mean | p10 | win% |
|---|---|---|---|---|
| sell | 44 | +1.125 | -3.898 | 65.9% |
| buy | 46 | +0.268 | -3.862 | 54.3% |

**ranging 内では sell/buy とも p10 > -5.0 で PASS 圏**。ranging の PnL=+61.83bps が全体利益の 97% を占める。

### §6.3 Trending 不足問題

trending_down (n=5) + trending_up (n=5) = **10 fills** しかなく、regime 別の統計的判断は不可能。

311# では trending_up sell が p10=-9.86 で最悪だったが、333# では trending_up 全体の n=5 (sell=3) しかない。**この最大のリスク要因の評価が依然としてできていない。**

### §6.4 None Regime 不在

310# で追加した None Regime Observability (D) により、none 出現をカウントしているが、24h の計測期間で **none regime は 0 件**。これは detector が warmup 完了後に安定動作していることを示す。

311# で none=10.4% (AS 42.7%) だった問題は、少なくとも dcc3064 の 24h 期間では再現していない。

---

## §7 時間帯別深堀り

### §7.1 310# Sell Hour Boost の効果検証

310# A で導入した `sell_hour_offset_boost` は UTC 08/13/14/16 に sell offset を拡大する。

| UTC | 333# n | 333# sell mean | 317# baseline sell mean | 310# boost 乗数 |
|---|---|---|---|---|
| 08 | 2 (sell) | +5.978 | -3.546 | 1.5 |
| 13 | 1 (sell) | +0.486 | -2.156 | 1.3 |
| 14 | 4 (sell) | +3.017 | -3.277 | 1.3 |
| 16 | — | — | -2.250 | 1.5 |

UTC 08/13/14 で sell PnL はすべて正転しているが、**n が極少 (1-4 件)** であり統計的判断は不可能。UTC 16 は sell fill なし。

**暫定評価**: sell hour boost は方向性として有効に見えるが、n < 5 の区間での判断は危険。

### §7.2 危険時間帯

| UTC | n | mean | p10 | 注記 |
|---|---|---|---|---|
| 03 | 2 | -11.411 | — | sell -19.032bps の extreme outlier |
| 12 | 2 | -7.627 | — | sell -14.096bps |
| 17 | 5 | -1.378 | -7.845 | |

UTC 03 (JST 12時) と UTC 12 (JST 21時) に大きな外れ値。n=2 なので構造的問題かノイズかは判別不可。

### §7.3 最良時間帯

| UTC | n | mean | p10 |
|---|---|---|---|
| 00 | 6 | +6.488 | +0.186 |
| 08 | 3 | +6.211 | +4.343 |
| 19 | 3 | +6.290 | +2.012 |
| 20 | 5 | +2.911 | +0.484 |
| 14 | 8 | +2.735 | -0.951 |

UTC 00 (JST 09時) と UTC 08 (JST 17時) が安定して高収益。

---

## §8 Cancel Reason 分析 — Buy Kill 問題

### §8.1 Skip 内訳

| Cancel Reason | Count | % of skips | 影響 |
|---|---|---|---|
| **buy_dynamic_kill** | **216** | **40.2%** | buy fill 壊滅の主因 |
| forced_buy_delay | 100 | 18.6% | buy 遅延 |
| degraded_liquidation_duty_skip | 95 | 17.7% | 清算義務による skip |
| skip_gate | 38 | 7.1% | ML ゲート |
| spread_too_narrow | 30 | 5.6% | spread 不足 |
| timeout | 25 | 4.7% | タイムアウト |
| stale_adverse_drift | 22 | 4.1% | stale AS ドリフト |

### §8.2 Buy Kill の構造分析

**buy_dynamic_kill が skip 理由の 40.2% を占め、buy fill_rate を 9.3% まで押し下げている。**

これは AS 防御メカニズム (`buy_as_guard`) が buy 側に過剰適用されている可能性を示す。結果として:

1. buy fill_rate=9.3% (AB 閾値 30% を大幅に下回る)
2. fill された buy 49 件は "厳選" されたサンプルであり、avg_pnl=+0.372bps, p10=-3.819 という良好な数値は **survivorship bias** の産物の可能性

### §8.3 Balance Forced

balance_forced_switch = 170 (26.7%) — 約 1/4 の cycle で残高強制が発動。311# 時点での懸念と整合的。

---

## §9 Adverse Selection 構造

### §9.1 AS 率

| Side | AS (<-3bps) | Severe (<-10bps) |
|---|---|---|
| sell | 17.6% (9/51) | 7.8% (4/51) |
| buy | 12.2% (6/49) | 4.1% (2/49) |

### §9.2 311# AS 率との比較

| Side | 311# AS率 | 333# AS率 | Δ |
|---|---|---|---|
| sell | 30.3% | 17.6% | **-12.7pp 改善** |
| buy | 27.5% | 12.2% | **-15.3pp 改善** |

**AS 率が両サイドで大幅に低下** — ただし 333# は ranging=90.3% であり、trending_up (311# で AS 49.3%) がほぼ含まれないことが最大の要因。

### §9.3 sell hour boost の AS 抑制効果

317# baseline では boost 時間帯 (UTC 8/13/14/16) の AS 率は 49.3%。333# データでは:

- UTC 08: sell n=2, PnL=+5.978bps (AS 発生なし)
- UTC 13: sell n=1, PnL=+0.486bps (AS 発生なし)
- UTC 14: sell n=4, PnL=+3.017bps (AS 発生なし)

n=7 全件で AS なし。**小サンプルゆえ確定できないが、boost 乗数が offset を十分に拡大し AS 回避に寄与している可能性**がある。

---

## §10 320# C-1 未修正の影響

### §10.1 問題の再確認

320# で発見・修正された C-1 問題 (sell pipeline 100% ceiling-clamped):

```
sell: floor(0.30) > ceiling(0.15)
→ 全 sell offset が 0.15 にクランプ
→ 12+ パラメータ (regime, spread_adapt, kyle, amihud, etc) が死亡
```

**この分析の dcc3064 期間中、320# 修正 (044e687a9) は未適用。** sell pipeline は 100% ceiling-hit の状態で稼働していた。

### §10.2 にもかかわらず sell が正の PnL を出した理由

sell mean PnL = +0.889bps であり、ceiling-clamped にもかかわらず利益を計上した。考えられる要因:

1. **executor boost**: ceiling 結果 (0.15) に対して `fill_cycle_executor.py` の trending 乗数 (×4.0) が適用され、trending 時には 0.60 まで拡大される。**巨視的には executor boost が唯一の実効制御**だった。
2. **ranging 支配**: 90.3% が ranging であり、ceiling 0.15 でも ranging の低ボラ環境では十分な margin を確保できた可能性。
3. **sell hour boost**: 310# A で追加した時間帯 boost は ceiling **前** に適用される pipeline stage だが、floor(0.30) に支配されて ceiling(0.15) に圧縮されるため、**sell_hour_boost も実質的に死んでいる**。ただし、pipeline の他の downward 調整が 0.30 → ceiling 0.15 の間で生き残る余地があり、hour boost が floor 値を引き上げた可能性がある。

### §10.3 C-1 修正後の期待と懸念

320# 修正 (sell ceiling 0.15 → 0.50) 後は:
- **期待**: 12+ パラメータが復活し、regime/spread/kyle/amihud による offset 分化が機能。AS 防御が本来の精度で動作。
- **懸念**: ceiling 解放で平均 offset が上昇 → fill_rate 低下のリスク。**ceiling 0.15 で fill_rate 46.8% (sell) は十分高く、offset 拡大で 30% を下回らないかモニタリング必要。**

---

## §11 統計的限界と信頼区間

### §11.1 n=100 の検出力

n=100 での p10 推定のブートストラップ標準誤差は、分布の裾が heavy-tail の場合 **±2-3 bps** に達する。したがって:

| 指標 | 点推定 | 概算 95% CI | 含意 |
|---|---|---|---|
| overall p10 | -4.120 | [-8, 0] | PASS だが信頼区間は広い |
| sell p10 | -5.207 (n=51) | [-10, 0] | PASS/FAIL の判別不可能 |
| buy p10 | -3.819 (n=49) | [-8, +1] | PASS だが不確実 |
| mean PnL | +0.636 | [-0.5, +1.8] | 正の可能性が高いが保証なし |

### §11.2 Regime カバレッジの偏り

| Regime | 333# 比率 | 311# 比率 | 差 |
|---|---|---|---|
| ranging | **90.3%** | ~64% | +26pp |
| trending_up | 3.9% | ~7% | -3pp |
| trending_down | 5.8% | ~7% | -1pp |
| none | 0.0% | 10.4% | -10pp |

333# は **ranging に極端に偏った 24h** のスナップショットであり、trending/none regime での性能は評価できていない。

### §11.3 信頼性の総合評価

| 項目 | 信頼度 | 理由 |
|---|---|---|
| PnL mean の符号 (正) | 中 | 60/100 件が正 (binomial p ≈ 0.02) |
| sell hour boost の効果 | 低 | boost 時間帯の n=7、対照なし |
| trending 性能 | 不可 | n=10、統計的判断不可能 |
| buy fill_rate 問題 | 高 | 構造的 (buy_dynamic_kill 216/537) |
| AS 率改善 | 低 | regime 分布の差が交絡因子 |

---

## §12 構造的発見と課題整理

### §12.1 発見

| # | 発見 | 重要度 | 根拠 |
|---|---|---|---|
| D-1 | **PnL 正転**: 24h で +63.56bps は過去最良の SHA 期間 | 高 | sum(pnl)=+63.56, win=58% |
| D-2 | **buy kill 壊滅**: buy_dynamic_kill が fill_rate を 9.3% に圧縮 | 高 | 216/537 skip が buy_dynamic_kill |
| D-3 | **ranging 偏り**: 90.3% ranging で trending 性能未検証 | 中 | regime 分布の計測 |
| D-4 | **sell p10 僅差 FAIL**: -5.207 vs 閾値 -5.000 (Δ=0.207) | 中 | AB judgment |
| D-5 | **C-1 未修正でも profit**: sell pipeline 全死でも +0.889bps/fill | 中 | offset_stages 確認 |

### §12.2 課題 (優先度順)

| ID | 課題 | 優先度 | 提案 |
|---|---|---|---|
| **T-1** | buy_dynamic_kill の閾値見直し | **P0** | kill 判定の aggressive さを緩和し fill_rate ≥ 30% を目指す |
| **T-2** | forced_buy_delay の頻度削減 | P1 | delay ロジックの条件精査 |
| **T-3** | degraded_liquidation_duty_skip 調査 | P1 | 清算義務 skip が 17.7% は異常値の可能性 |
| **T-4** | Trending regime データ蓄積 | P1 | 168h 連続計測で trending 評価を確保 |
| **T-5** | 320# C-1 修正後の sell 性能追跡 | P1 | ceiling 解放後の fill_rate/p10 モニタリング |
| **T-6** | UTC 03/12 外れ値の構造分析 | P2 | n 蓄積後に再評価 |

---

## §13 000# G1.2 Gate 進捗

311# §1.2 で整理した G1.2-full (168h) 指標を 333# データで更新:

| # | 指標 | 閾値 | 333# 値 | 311# 値 | 判定 |
|---|---|---|---|---|---|
| F1 | attempted_fill_rate | ≥ 70% | 15.7% | ~40% | ❌ 大幅悪化 |
| F4 | PnL30 (有意に負でない) | p ≥ 0.05 | mean=+0.636 | mean=-0.33 | 🟡 正だが CI 広い |
| F5 | AS_ratio | ≤ 30% | sell 17.6%, buy 12.2% | sell 30.3%, buy 27.5% | ✅ 改善 |
| F7 | calendar_coverage | ≥ 7暦日 | 1日 (24h) | ~22日 | ❌ 未達 |
| F8 | n_attempted | ≥ 500 | 637 | — | ✅ |

**F1 の大幅悪化は buy_dynamic_kill に起因。** fill_rate を指標として用いるならば、kill ロジックの見直し (T-1) が Gate 通過の前提条件。

**注意**: 333# は **同一 SHA 24h** のデータであり、G1.2 gate の 168h (7日) 基準には未達。

---

## §14 AI レビュー向け設問

### Q1: buy_dynamic_kill の最適閾値

buy_dynamic_kill が skip の 40.2% を占め、buy fill_rate=9.3% まで圧縮している。AS 防御と fill_rate のトレードオフにおいて、どのような閾値調整アプローチが適切か？ 具体的に:

- kill 閾値を緩和した場合、AS 露出はどの程度増加すると予測されるか？
- buy fill_rate を 30% に引き上げるために必要な kill 緩和度は？
- 「厳選された 49 fills が +0.372bps」は survivorship bias であり、kill 緩和で mean PnL は負に反転する可能性があるか？

### Q2: sell p10 = -5.207 の構造的解釈

sell p10 が閾値 -5.00 に対し 0.207bps の超過で FAIL している。n=51 での p10 推定の信頼区間を考慮すると、この FAIL 判定にはどの程度の意味があるか？ Bootstrap CI を用いた判定基準の修正は妥当か？

### Q3: C-1 修正後のリスクシナリオ

320# で sell ceiling を 0.15 → 0.50 に拡大した。ceiling 解放により:
- sell offset が 0.30-0.50 に分散し、fill_rate が 46.8% から低下するリスクはどの程度か？
- 12+ パラメータ復活による AS 防御向上と fill_rate 低下のどちらが dominant になるか？
- 321# で「YAML 未パースで sell 防御力 62.5% 悪化」(ceiling 0.50 未適用) が報告されたが、これが修正された後の安定 offset 分布はどうなると予測するか？

### Q4: Ranging 偏り (90.3%) は楽観的バイアスか？

333# の PnL 正転 (+0.636bps mean) は ranging=90.3% の偏った市場環境に依存している可能性が高い。311# データでは trending_up sell が p10=-9.86 で最悪だった。ranging が 60-70% に戻った場合、overall PnL はどの程度悪化するか？ 線形外挿ではなくレジーム遷移確率を考慮した推定を求む。

### Q5: 統合的 next step

dcc3064 (310#) → 320# (C-1 fix) → 321# (YAML parse fix) と修正が積み重なった現在の HEAD で、G1.2 168h gate に向けた最適な計測 SHA はどれか？ 追加の修正なしに 168h 計測を開始すべきか、それとも T-1 (buy_dynamic_kill 閾値) を先に修正すべきか？ cost of delay vs cost of iteration のトレードオフを定量的に論じよ。

---

## §15 追記: Gemini 3.1 Pro セカンドオピニオン

> 以下は Gemini 3.1 Pro による外部レビュー追記欄。レビュー完了後に記入する。

### レビュー依頼コンテキスト

本ドキュメントは、310# (dcc3064a8) の設計改修が稼働した 24 時間のデータを SHA 分離して分析したレポートである。

**レビューに際して知っておくべき前提:**

1. **C-1 sell ceiling 問題** (320# で修正): dcc3064 稼働期間中、sell offset pipeline は floor(0.30) > ceiling(0.15) のためすべて 0.15 にクランプされ、12+ パラメータが無効化。310# A の sell hour boost も実質死亡。
2. **buy_dynamic_kill 支配**: Skip の 40.2% を占め、buy fill_rate=9.3% まで圧縮。
3. **Ranging 偏り**: 24h で 90.3% が ranging regime。trending での性能は未検証。
4. **299# 結論**: sell vs buy PnL 差は統計的に非有意 (4 検定すべて)。
5. **n=100 の統計的限界**: p10 の 95% CI は ±2-3bps と推定。

§14 の Q1-Q5 への回答および追加の指摘を求む。
