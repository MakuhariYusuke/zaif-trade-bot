# 005: G1-info 検証報告

**Phase**: 1 (検証)  
**Type**: rpt (report)  
**Gate**: G1-info  
**実施日**: 2026-02-13  
**先行文書**: [000# Project Proposal](000_ph0_plan_project_proposal.md) §3.2, [004# Review Fix](004_ph0_rev_fix.md)

---

## §1 概要

### G1-info 公式判定: **FAIL**

XGBoost Walk-Forward 評価により、OHLCV プロキシ特徴量 10 種の情報量を検証した。
g1_judgment (p-mean → Holm-Bonferroni → AND) の結果、全 9 ターゲットで Cliff's Delta が閾値 0.33 を下回り、正式には **FAIL** である。

ただし、個別メトリクスでは **7/9 ターゲットが閾値を超過**しており、結果は単純な棄却ではなく慎重な解釈を要する。

---

## §2 実験設定

| 項目 | 値 |
|------|-----|
| データ | `data/v460/features/btc_jpy_1m_v460_features.parquet` |
| 行数 | 1,216,930 rows × 11 cols (close + 10 proxy features) |
| SHA-256 prefix | `9494e9cf59834c56` |
| Walk-Forward | 5-fold blocked time-split, train_ratio=0.80 |
| モデル | XGBoost (n_est=200, depth=6, lr=0.05) |
| ベースライン | Logistic (classification) / Ridge (regression) |
| ターゲット | 3 horizons (h1/h5/h15) × 3 types (direction/magnitude/volatility) = 9 |
| Seed | 42 |
| 結果ファイル | `results/v460/v460_g1info_seed42_20260213_080522.json` |
| マニフェスト | `results/v460/manifest.jsonl` (gate_result=FAIL) |

### G1 閾値 (000# §3.2)

| 条件 | 閾値 |
|------|------|
| OOS Spearman IC | > 0.02 |
| OOS Accuracy | > 51% |
| 有意 fold 数 | ≥ 2/5 (p < 0.05) |
| Cliff's Delta | \|d\| > 0.33 |
| 多重比較補正 | Holm-Bonferroni (family=9) |

---

## §3 結果サマリー

### §3.1 XGBoost 個別メトリクス

| Target | IC mean | IC std | Sig folds | Acc mean | Acc std | IC>0.02 | Acc>51% | Sig≥2 | 判定 |
|--------|---------|--------|-----------|----------|---------|---------|---------|-------|------|
| direction_h1 | 0.0393 | 0.0726 | 1/5 | 51.46% | 2.58% | ✅ | ✅ | ❌ | FAIL |
| magnitude_h1 | 0.0401 | 0.0774 | 1/5 | 50.98% | 2.46% | ✅ | ❌ | ❌ | FAIL |
| **volatility_h1** | **0.3222** | 0.0613 | **5/5** | **99.66%** | 0.12% | ✅ | ✅ | ✅ | **PASS** |
| direction_h5 | 0.0356 | 0.0644 | 2/5 | 51.41% | 2.51% | ✅ | ✅ | ✅ | **PASS** |
| magnitude_h5 | 0.0306 | 0.0440 | 3/5 | 51.22% | 1.99% | ✅ | ✅ | ✅ | **PASS** |
| **volatility_h5** | **0.4654** | 0.0432 | **5/5** | **99.88%** | 0.16% | ✅ | ✅ | ✅ | **PASS** |
| direction_h15 | 0.0298 | 0.0387 | 3/5 | 51.56% | 1.97% | ✅ | ✅ | ✅ | **PASS** |
| magnitude_h15 | 0.0225 | 0.0238 | 3/5 | 51.08% | 1.60% | ✅ | ✅ | ✅ | **PASS** |
| **volatility_h15** | **0.6129** | 0.0549 | **5/5** | **99.98%** | 0.05% | ✅ | ✅ | ✅ | **PASS** |

**個別結果**: 7/9 PASS, 2/9 FAIL (direction_h1, magnitude_h1)

### §3.2 Logistic/Ridge ベースライン

| Target | IC mean | IC std | Acc mean | Acc std |
|--------|---------|--------|----------|---------|
| direction_h1 | -0.0067 | 0.0202 | 49.26% | 2.01% |
| magnitude_h1 | -0.0115 | 0.0266 | 49.75% | 0.28% |
| volatility_h1 | 0.2343 | 0.1031 | 98.44% | 2.46% |
| direction_h5 | -0.0020 | 0.0171 | 49.61% | 1.47% |
| magnitude_h5 | -0.0042 | 0.0180 | 49.40% | 1.20% |
| volatility_h5 | 0.3337 | 0.1987 | 98.55% | 2.82% |
| direction_h15 | 0.0036 | 0.0179 | 50.80% | 0.58% |
| magnitude_h15 | -0.0011 | 0.0240 | 49.59% | 0.90% |
| volatility_h15 | 0.4571 | 0.2787 | 98.72% | 2.55% |

### §3.3 g1_judgment 結果

| Target | p_geo | pmean_pass | holm_pass | cliff_d | \|d\| > 0.33 |
|--------|-------|------------|-----------|---------|---------------|
| direction_h15 | 0.0001 | ✅ | ✅ | -0.1008 | ❌ |
| volatility_h1 | 0.0001 | ✅ | ✅ | -0.0440 | ❌ |
| volatility_h5 | 0.0001 | ✅ | ✅ | -0.1071 | ❌ |
| volatility_h15 | 0.0001 | ✅ | ✅ | -0.1107 | ❌ |
| direction_h1 | 0.0001 | ✅ | ✅ | -0.0236 | ❌ |
| direction_h5 | 0.0003 | ✅ | ✅ | -0.0302 | ❌ |
| magnitude_h5 | 0.6183 | ❌ | ❌ | -0.1773 | ❌ |
| magnitude_h15 | 0.6520 | ❌ | ❌ | -0.1863 | ❌ |
| magnitude_h1 | 1.0000 | ❌ | ❌ | -0.1997 | ❌ |

**全ターゲットで cliff_d < 0.33**: g1_pass = **false**

---

## §4 批判的考察

### §4.1 Fold 4 異常値 — 非定常性の直接証拠

direction/magnitude 全ターゲットで、fold 4 (データ末尾 20%) の IC が他 fold を 10〜50 倍上回る。

| Target | Fold 0-3 IC 範囲 | Fold 4 IC | 倍率 |
|--------|-------------------|-----------|------|
| direction_h1 | -0.002 〜 +0.008 | **0.1842** | ~23x |
| magnitude_h1 | -0.005 〜 +0.007 | **0.1947** | ~28x |
| direction_h5 | -0.005 〜 +0.010 | **0.1639** | ~16x |
| direction_h15 | +0.001 〜 +0.022 | **0.1055** | ~5x |

**意味**: 全期間の平均 IC (0.02〜0.04) は fold 4 が支配的に引き上げている。fold 0-3 の IC は実質ゼロに近く、特定の市場レジーム期間でのみ情報性がある可能性が極めて高い。このレジームが将来にわたって持続する保証はない。

### §4.2 Volatility ターゲットの疑似 PASS — ボラティリティ・クラスタリング

volatility ターゲットの IC 0.32〜0.61、精度 99.6%〜100% は**ボラティリティの自己相関（clustering）の反映であり、マイクロストラクチャ情報ではない**。

- 特徴量 `micro_return_vol` = `log_ret.rolling(20).std()` (過去 20 期間の対数リターン標準偏差)
- ターゲット `target_volatility_hN` = `log_ret.rolling(N).std().shift(-N)` (将来 N 期間の対数リターン標準偏差)
- ボラティリティは市場の普遍的性質として強い自己相関を持つ (GARCH/SV 効果)
- **Logistic/Ridge でも IC 0.23〜0.46 を達成** → XGBoost の非線形性は不要

この 3 ターゲットの PASS は「ボラティリティは予測可能」という自明な事実の確認に過ぎず、**方向予測能力の証拠とはならない**。

### §4.3 OHLCV プロキシ特徴量の原理的限界

v460 の 10 特徴量はすべて OHLCV データから算出された**プロキシ**である。

| 特徴量 | 構成 | 限界 |
|--------|------|------|
| bid_ask_spread | (H-L)/mid | 実スプレッドは板の最良気配差。OHLCV は高低差のみ |
| depth_imbalance | CLV (Close Location Value) | 板深度の直接情報なし |
| trade_flow_imbalance | 符号付き出来高 | 実際の約定方向（maker/taker）は不明 |
| vwap_deviation | VWAP 乖離 | 近似 VWAP (TP) で真の VWAP は不取得 |
| order_flow_toxicity | VPIN proxy | 分類基準が真のフローと乖離 |
| micro_return_vol | 対数リターン σ | 真のマイクロ構造ではなく価格変動性 |

**v459 K2 の教訓**: v459 では RSI×7 の OHLCV 由来特徴量が G1 相当の検証で方向情報の欠如が確認された。v460 プロキシも同一の OHLCV データから導出しており、**情報源の本質的拡張はない**。

### §4.4 Cliff's Delta の解釈問題

g1_judgment の cliff_d は XGBoost シグナルと Logistic/Ridge シグナルの**分布比較**であり、**予測品質の比較ではない**。

- cliff_d が全て負 → Logistic/Ridge のシグナル値が XGBoost より系統的に大きい
- これはモデルの calibration 差であり、予測精度の優劣を意味しない
- 一方、IC では XGBoost が direction/magnitude で Logistic を上回っている

**検討課題**: g1_judgment の比較対象を raw signal → per-fold IC 等のパフォーマンス指標に変更する設計変更の是非。ただし n=5 fold では統計的検出力が不十分となるジレンマがある。

### §4.5 v459 との構造的類似性

| 項目 | v459 K2 | v460 G1 |
|------|---------|---------|
| データソース | OHLCV | OHLCV (同一データベース) |
| 特徴量由来 | RSI×7 | Proxy×10 |
| 方向 IC | ≈ 0 | 0.03〜0.04 (fold 4 除外で ≈ 0) |
| 結論 | 方向情報なし | **方向情報は実質的に同様に欠如** |

fold 4 除外時の direction IC は 0〜0.01 であり、v459 K2 の結果と本質的に一致する。
**OHLCV データからの方向予測は、手法を変えても不可能である可能性が高い。**

---

## §5 方向転換判定

### §5.1 判定: 条件付き続行 + データ拡張必須

| 選択肢 | 根拠 | 推奨 |
|--------|------|------|
| A: G1 FAIL → 全面停止 | cliff_d 未達、非定常性 | ❌ 過度に保守的 |
| B: G1 PASS 扱いで G2 突入 | 7/9 個別 PASS | ❌ volatility 疑似 PASS を無視 |
| **C: G1.1-exec 並行 + 実データ収集** | **方向情報は不十分だが板データで再検証余地** | **✅ 推奨** |

### §5.2 推奨アクション

1. **即時**: 取引所 API (Coincheck/Bitflyer/Zaif) からリアルタイム板・約定データの収集開始
2. **並行**: G1.1-exec (maker 約定可能性の実測) — fill rate + adverse selection の確認
3. **中期**: 実板データベースの蓄積 (2〜4 週間) → 真のマイクロストラクチャ特徴量で G1 再検証
4. **G1 再検証条件**: 実板データ由来の特徴量で G1 threshold を再度判定

### §5.3 000# 方向性との整合

000# §6 リスク表の最高重要度リスク「マイクロストラクチャ特徴量にも情報がない」が**部分的に顕在化**。

000# の対処方針「G1-info を Phase 1 で即実行。FAIL なら即座に特徴量再設計」に基づき、
OHLCV プロキシから実板データへの**特徴量再設計**を開始すべきである。

---

## §6 発見されたバグと修正

G1 実行過程で 4 つの重大バグを発見・修正：

| # | ファイル | バグ | 修正 |
|---|---------|------|------|
| B1 | `evaluator.py` | IC 計算で `.diff()` の先頭 NaN が `spearmanr` を完全に NaN 化 → 全 IC=0.0 | `y_test` を直接使用 + `nan_policy="omit"` |
| B2 | `evaluator.py` | High-confidence IC で `price_changes` 変数の残参照 | `ic_target[]` に統一 |
| B3 | `data_loader.py` | `rolling(1).std()` (h=1 volatility) が NaN → 0 行 | `max(h, 2)` で最小窓幅保証 |
| B4 | `gate_checks.py` | `cliffs_delta` が O(n²) ネストループ → 243k² ≈ 590 億反復で実行不能 | Mann-Whitney U から O(n log n) で導出 |
| B5 | `run_experiment.py` | `gate_thresholds.yaml` を `load_config` で読込 → validation エラー | `yaml.safe_load` で直接読込 |

**注**: B1 が最も重大。修正前は全実験の IC が 0.0 として報告され、G1 判定自体が無意味だった。

---

## §7 ログ・成果物所在

| 成果物 | パス |
|--------|------|
| 最終結果 JSON | `results/v460/v460_g1info_seed42_20260213_080522.json` |
| マニフェスト | `results/v460/manifest.jsonl` |
| 実験設定 | `configs/v460/experiments/g1_full_9targets.yaml` |
| Gate 閾値 | `configs/v460/gate_thresholds.yaml` |
| データ | `data/v460/features/btc_jpy_1m_v460_features.parquet` |
| 分析スクリプト | `temp/analyze_g1.py`, `temp/debug_g1_judgment.py` |
| 修正ファイル | `lib/evaluator.py`, `lib/data_loader.py`, `ztb/metrics/gate_checks.py`, `run_experiment.py` |

---

## §8 次フェーズへの接続

### 確定事項

- G1-info (OHLCV proxy): **FAIL** — cliff_d 未達
- OHLCV 由来プロキシでの方向予測は不十分
- ボラティリティ予測は自明 (volatility clustering) であり、maker 戦略の方向判断には使えない

### 要検討

1. g1_judgment の cliff_d がシグナル分布比較であることの是非 → 次回レビューで議論
2. 実板データ収集の具体的実装計画（取引所選定、データスキーマ、保存形式）
3. G1.1-exec の実行可否 — G1 FAIL でも maker 約定特性の調査は独立に有用

---

## Appendix A: 改訂履歴

| 日付 | 変更内容 |
|------|---------|
| 2026-02-13 | 初版作成 — G1-info 検証結果報告 |
