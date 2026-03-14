# v460 Project Proposal: "Microstructure Edge"

**Date**: 2026-02-13
**Status**: Draft
**Predecessor**: v459 "Alpha Resurrection" (No-Go confirmed)

---

## 目次

- [§0 大義と目的](#0-大義と目的)
- [§1 v459 教訓サマリ](#1-v459-教訓サマリ)
- [§2 Phase 定義](#2-phase-定義)
- [§3 Gate 定義](#3-gate-定義)
  - [§3.1 G0-data](#31-g0-data) / [§3.2 G1-info](#32-g1-info) / [§3.3 G1.1-exec](#33-g11-exec)
  - [§3.4 G2-train](#34-g2-train) / [§3.5 G3-pnl](#35-g3-pnl) / [§3.5.1 G3.1-stress](#351-g31-stress) / [§3.6 G4-live](#36-g4-live)
  - [§3.7 統計検定仕様](#37-統計検定仕様)
  - [§3.8 Gate 枝番規則](#38-gate-枝番規則)
- [§4 技術概要](#4-技術概要)
- [§5 命名規則・運用規約](#5-命名規則運用規約)
- [§6 リスク](#6-リスク)
- [Appendix A: 改訂履歴](#appendix-a-改訂履歴)

---

## §0 大義と目的

> **本プロジェクトの大義は「短期間での高収益性システム」の実現である。**
> ただし短期的成果と長期的な健全性のバランスを取ることが重要である。

v459 は OHLCV 派生特徴量 (RSI×7 + ReturnStdDev) に方向予測情報が存在しないことが K2 実験で確定し、No-Go となった。v460 はマイクロストラクチャ特徴量（板情報・約定フロー）の導入と maker-only 執行戦略により、v459 の根本原因を直接解消する。

### §0.1 SAC の役割定義（422#-426# 改訂）

> **SAC は「運転手 (Driver)」ではなく「助言者 (Sidecar Navigator)」である。**

426# S1/S1' 実験により、SAC 単体での収益性には構造的限界が確認された:

- **レジーム汎化限界**: 20K step 学習は ~60 日以降の市場レジーム変動に対応できない（8 seed 中 6 seed で OOS mid 期崩壊）
- **評価と本番の乖離**: OOS シミュレーションと本番環境（clamp 100% 発火, Sidecar 停止）は根本的に異なる（431#）
- **学習容量の天井**: 100K step に増やしても改善せず悪化（422#）。step 数ではなく方策空間（1 次元連続行動）の表現力が制約

したがって v460 のアーキテクチャは以下の三層分離を採る（428# 提案, 430# 監査で既存資産の存在を確認済み）:

| 層 | 責務 | 実装 |
|---|---|---|
| **Alpha 層** | 方向バイアス生成 | SAC → Sidecar Signal（directional_bias [-1,1]）|
| **Execution 層** | 注文価格決定・約定管理 | offset chain + Final Clamp（421# 実装済み）|
| **Safety 層** | リスク制御・緊急停止 | SkipGate, Circuit Breaker, retrain_scheduler（430# 発見: 実装済み・未起動）|

SAC の出力は直接注文価格を決定しない。Sidecar v2（比例変換, sidecar_types.py 実装済み）を介して offset に変換され、Final Clamp で安全範囲内に制約される。

**前提条件**: 全取引は maker 注文（手数料 0%）で執行する。taker 注文は禁止。対象取引所は Coincheck（主）/ Bitflyer / Zaif を含む maker 手数料 0% の国内取引所とし、API 品質・流動性に応じて切替可能な設計とする。

---

## §1 v459 教訓サマリ

v459 の 119 文書・全 Phase 実験から得られた 4 教訓:

| # | 教訓 | 根拠 |
|---|------|------|
| 1 | **特徴量を先に検証せよ** | K2 (XGBoost) を Phase A で実行していれば Phase B–E は不要だった |
| 2 | **単一 seed の成功を信じるな** | E2α +41.95% が E2β で +3.93% に崩壊 |
| 3 | **Oracle テストを早期に行え** | 完全予測でも taker 0.1% では費用負け |
| 4 | **手数料構造は戦略の前提条件** | taker 0.1% × 1min 足は Oracle でも不成立。maker 0% が必須。Zaif の API 品質低下により Coincheck / Bitflyer へ急遽移行した経緯あり — 取引所非依存設計が必須 |

詳細: [v459/index.md](../v459/index.md) / [v459/116# §17](../v459/116_phase_e0_diagnostic_report.md)

---

## §2 Phase 定義

| Phase | 目的 | Gate | 成果物 |
|-------|------|------|--------|
| **ph0** | データ取得基盤・特徴量候補定義・Gate 仕様確定 | G0-data | データパイプライン、001#–003# |
| **ph1** | 特徴量情報量検証 (非 RL) | G1-info | XGBoost Walk-Forward 結果 |
| **ph2** | maker 執行可能性検証 | G1.1-exec | fill rate 実測データ |
| **ph3** | SAC 学習安定性検証 | G2-train | 4 seed 訓練結果 |
| **ph3.1** | Sidecar 統合・retrain 起動 | — | retrain_scheduler 稼働, Sidecar v2 有効化 |
| **ph4** | 収益性検証 (コスト込み) | G3-pnl | PF / Sharpe / DD レポート |
| **ph4.1** | 摩擦耐性検証 (stress) | G3.1-stress | slippage / miss 感度レポート |
| **ph5** | Paper trading 運用検証 | G4-live | 1 週間連続稼働データ |

Phase 枝番規則: `.1` 刻み整数連番 (e.g. ph1.1)、ファイル名では `ph1-1`。最大 2 枝。詳細は [v459/119# §8.2](../v459/119_v460_launch_integrated_policy.md)

---

## §3 Gate 定義

### §3.1 G0-data

**目的**: データ品質と再現性基盤の保証。学習開始の前提条件。

| 条件 | 閾値 | 判定 |
|------|------|------|
| データハッシュ一致 | config 記載値と実データが一致 | 不一致 → 即停止 |
| 特徴量カラム数 | ≥ 4 | 不足 → 即停止 |
| NaN 比率 | ≤ 1% | 超過 → 前処理見直し |
| manifest.jsonl 記録 | 全実行が自動記録済み | 未記録 → 実行無効 |

**FAIL 時**: 学習禁止。データパイプラインを修正して再実行。

### §3.2 G1-info

**目的**: 特徴量に方向予測情報があるかを非 RL で検証。v460 最重要 Gate。

| 条件 | 閾値 | 測定方法 |
|------|------|---------|
| OOS Spearman IC | > 0.02（少なくとも 1 horizon） | blocked time split 5 窓 + OOS |
| OOS Accuracy | > 51% | 同上 |
| 有意 fold 数 | ≥ 2/5 (p < 0.05) | binomial test |

**Multi-target**: h1 (1min), h5 (5min), h15 (15min) の 3 horizon × direction / magnitude / volatility の最大 9 組合せ。1 組合せでも PASS すれば G1 通過。ただし multi-target 判定は Holm-Bonferroni 補正後の p 値を用いる（family = 9 組合せ）。

**FAIL 時**: 特徴量再設計へ戻る。RL には進まない。

### §3.3 G1.1-exec

**目的**: maker 注文の約定可能性を実測で確認。

**116# 改訂: 二段階ゲート化** (115# 外部レビュー反映)

原初の単一ゲート (E1-E5) は SkipGate 導入前に策定され、attempted/overall 分母の区別がなかった。
SkipGate 導入後の実測データ分析により、G1.1-quick (Kill) + G1.2-full (Qualification) の二段階に分割。

#### G1.1-quick (72h Kill Gate)

**判定時点**: 72 時間経過 or n(attempted) ≥ 300、いずれか早い方。

| # | 指標 | 閾値 | 測定ベース | 根拠 |
|---|------|------|----------|------|
| K1 | attempted_fill_rate | ≥ 60% | skip_gate 除外 | §3.9 中止ルール 70% より緩く、Kill は「明らかに不成立」の検出が目的 |
| K2 | attempted_cancel_ratio | ≤ 40% | skip_gate 除外 | K1 の裏 |
| K3 | queue_wait_median | ≤ 120 sec | 約定のみ | 300s timeout の半分 |
| K4 | PnL30 複合条件 | p < 0.02 **かつ** mean ≤ -0.8 bps で FAIL | 片側 t 検定 | 115#: 単独 p 値判定は不十分。効果量条件を併設 |
| K5 | 累積実損 | < 10,000 JPY | — | §3.9 キャップ |
| K6 | skip_gate_ratio | ≤ 25% | 全体 | SkipGate 過剰ブレーキ防止 |

**Watch 層**: K1-K6 全 PASS だが PnL が `p < 0.05 かつ mean < -0.3 bps` の場合は WATCH (黄信号)。パラメータ凍結・監視強化を推奨。

**FAIL 時**: fill_test 即時停止。戦略クラスの変更（e.g. aggressive maker, IOC 併用）を検討。

#### G1.2-full (168h Qualification Gate)

**判定時点**: 168 時間 (7 暦日) 完了後。

| # | 指標 | 閾値 | 測定ベース | 根拠 |
|---|------|------|----------|------|
| F1 | attempted_fill_rate | ≥ 70% | skip_gate 除外 | §3.9 中止ルールと一致 |
| F1b | overall_fill_rate | ≥ 62% | 全体 | 115#: SkipGate 過剰回避の安全弁 |
| F2 | attempted_cancel_ratio | ≤ 30% | skip_gate 除外 | 原初 E2 を attempted ベースで維持 |
| F3 | queue_wait_median | ≤ 60 sec | 約定のみ | 原初 E3 維持 |
| F4 | PnL30 | 有意に負でないこと (p ≥ 0.05) | 片側 t 検定 | 原初 E4 維持 |
| F5 | AS_ratio | ≤ 30% | 約定のみ | 115#: 20→35 は緩すぎ、30% を推奨 |
| F6 | skip_gate_ratio | ≤ 20% | 全体 | SkipGate 過剰スキップ監視 |
| F7 | calendar_coverage | ≥ 7 暦日 | — | 全曜日の市場特性をカバー |
| F8 | n_attempted | ≥ 500 | skip_gate 除外 | 統計的検出力保証 (MDE ≈ ±0.5 bps at 80% power) |

**attempted ベースの前提**: SkipGate 有効性について F6 (skip_gate_ratio ≤ 20%) で監視。将来的に S0 (SkipGate 有効性ゲート: OOT AUC ≥ 0.55 等) の前置きを検討。

**FAIL 時**: 個別指標により対応分岐。
- F1/F2 FAIL → offset 戦略見直し or aggressive maker
- F4/F5 FAIL → SkipGate / sell_guard の再設計
- F6 FAIL → SkipGate 閾値の緩和

#### 分母定義 (117# / 115# Q10.6)

Gate 指標の算出に用いる 3 つの母集団定義。判定出力では常に 3 系列を並記する。

| 系列 | 定義 | 分母 | 用途 |
|------|------|------|------|
| **raw (全体)** | 全注文サイクル | `total_orders` | overall_fill_rate, skip_gate_ratio |
| **clean** | quarantine 除外済み (データ品質フィルタ適用後) | `total_orders - quarantine_count` | legacy E1-E5 判定 |
| **attempted** | clean のうち skip_gate 除外 (実際に発注したもの) | `total_orders - quarantine - skip_gate` | K1-K2, F1-F2 判定 |

**注意**:
- `quarantine`: データ不整合レコード (047# A2)。run_id 欠損, 異常 timestamp 等。
- `skip_gate`: SkipGate ML 分類器が「高リスク」と判断し **意図的にスキップ** したサイクル。
- attempted ベースの正当性は SkipGate 有効性 (S0) に依存。F6/K6 の skip_gate_ratio 上限が暫定ガード。

#### cancel reason 内訳 (117# / 115# Q10.6)

cancel レコードの `cancel_reason` フィールドで分類。判定出力に内訳を付記。

| reason | 発生原因 | 改善アクション |
|--------|---------|--------------|
| `timeout` | 300s 以内に約定せず | offset 調整 / timeout 短縮検討 |
| `skip_gate` | SkipGate が高リスクと判断 | SkipGate 閾値 / 特徴量見直し |
| `postonly_reject` | Post-only 注文が即約定扱いで拒否 | offset 引き上げ |
| `status_unknown` | API 応答不明 | リトライ / API 安定性確認 |
| `stale_reprice_max` | stale order reprice 上限到達 | reprice_max / drift_bps 見直し |
| `unknown` (null) | cancel_reason 未設定 | データ品質改善 |

**測定期間**: デフォルト 7 日間。ただし n ≥ 200 サイクル かつ 3 暦日以上をカバーしていれば暫定判定可。最終判定は 7 日間データで確定する。

#### Gate 判定前提条件 (130# E.5 #3)

Gate 判定データの品質を保証するため、以下を前提とする:

- **同一 Git SHA ・同一 YAML で連続 72 時間以上** のデータが収集されていること。
- 設定変更やコード更新後のデータは、変更前データと混在させず run_id で分離する。
- YAML のみの変更 (hot-reload 対応の retrain パラメータ) は、fill_test 本体に影響しない場合は例外として許容。
- `反映確認済み` ステータス: YAML 変更は「反映済み」ではなく「**反映確認済み**」(ログで実際に新値が使用されていることを確認) で管理する。

### §3.4 G2-train

**目的**: RL (SAC) の学習が安定し、seed に依存しないことの確認。

| 条件 | 閾値 | 測定方法 |
|------|------|---------|
| gross > 0 の seed 比率 | ≥ 3/4 (75%) | 4 seed × 指定 steps |
| ROI の seed 間標準偏差 | ≤ 0.03 | 4 seed の OOS ROI 分散 |
| 学習曲線の収束 | 後半 50% で ROI 変動 ≤ 5% | checkpoint 別評価 |
| worst-seed 下限 | ROI > −2% | 最低 seed でも大幅な損失を出さないこと |

**val_ratio 規定** (426# 改訂): G2 判定の OOS は `val_ratio=0.10` を標準とする。実験では任意の val_ratio を使用可能だが、Gate 判定時は結果 JSON に `val_ratio` を記録し、比較可能性を担保する。val_ratio=0.02 の結果のみで Gate PASS としない（422#-426# で楽観バイアスが確認済み）。

**FAIL 時**: 学習器・報酬設計の見直し。特徴量を疑う前に G1-info を再確認。

### §3.5 G3-pnl

**目的**: コスト込みの実収益性を検証。

| 条件 | 閾値 | 測定方法 |
|------|------|---------|
| Profit Factor (median) | > 1.05 | maker 0% 前提、全 seed の中央値 |
| Profit Factor (worst-seed) | > 0.95 | 最低 seed でも致命的損失を回避 |
| avg gross/trade > avg fee/trade | true | 取引あたり収益 > 取引あたり費用（pooled） |
| Max Drawdown | < 15% | equity curve の最大下落（worst-seed） |
| Sharpe (年率) | > 0.8 | 日次リターン（median） |
| reward-profit alignment | corr median > 0 | 報酬とPnLの相関が正（425# 追加） |

#### G3 評価規定（426# 大幅改訂）

422#-426# の S1/S1' 実験で、val_ratio=0.02 の楽観バイアスと評価系の欺矇が確認された。以下の規定を追加する:

| 規定 | 内容 | 根拠 |
|------|------|------|
| **val_ratio 標準化** | G3 判定は `val_ratio ≥ 0.10` で実施。val_ratio=0.02 の結果のみでは PASS としない | 426# §3.1: val_ratio=0.02 → OOS 17 日ではレジーム変動が露呈しない |
| **multi-slice 診断** | OOS を early/mid/late 3 分割し、各期間の PF/Sharpe/DD を記録。Gate 判定自体には使わないが、崩壊パターンの診断に必須 | 426# §3.2: 8 seed 中 6 seed で mid 期崩壊。multi-slice なしでは根因特定不能 |
| **best_model 並行評価** | final_model と best_model の両方で OOS 評価を実施し、結果 JSON に両方記録。Gate 判定は final_model で行うが、best_model が優勢な場合は報告 | 422# A1, 426# §3.5: seed 789 で best_model のみ黒字帰還 |
| **結果 JSON 必須フィールド** | `val_ratio`, `model_dir`, `slice_metrics`, `best_model_eval_metrics`, `train_val_split.train_rows/val_rows` を必須記録 | 比較可能性・再現性の担保 |

**reward 関数方針** (426# §3.3, §7.2): reward_clean（純粋 PnL 信号のみ）を標準とする。補助ペナルティ（balance_penalty, consistency_penalty 等）は PnL 勾配を汚染し、学習効率を低下させることが実証された。

**FAIL 時**: モデル改善 or 取引頻度調整。全 seed FAIL かつ reward_clean でも改善しない場合は、SAC 単体の限界と判断し Sidecar+retrain アーキテクチャ（§0.1）への移行を検討する。v461 検討は Sidecar 統合後を判断時点とする。

### §3.5.1 G3.1-stress

**目的**: G3-pnl PASS 済みモデルを現実的な摩擦条件下で再評価し、paper trading (G4) 投入前に収益耐性を確認する。

**前提**: G3-pnl PASS が確定していること。G3.1 は G3 の枝番であり、G3 PASS 条件を変更しない（§3.8 枝番規則に準拠）。

**背景**: maker 0% 環境でも queue miss（注文が板に残らず約定しない）、adverse selection（約定直後の逆行）、約定遅延による slippage は実コストとして存在する。G3 は理想的な即時約定を前提としており、これらの friction を織り込んでいない（392# P1-2）。

| # | 条件 | 閾値 | stress パラメータ | 根拠 |
|---|------|------|------------------|------|
| S1 | PF (median) under slippage | > 1.00 | slippage_ticks: 1 (= 1 JPY) | 1tick slippage は maker 最小摩擦。PF>1.0 で損益分岐以上 |
| S2 | PF (worst-seed) under slippage | > 0.90 | slippage_ticks: 1 | worst-seed でも壊滅的損失を回避 |
| S3 | MaxDD under slippage | < 15% | slippage_ticks: 1 | G3 と同一基準を維持 |
| S4 | maker miss 感度 | PF > 0.95 at miss_rate=30% | maker_miss_rate: 0.30 | G1.2-full F1 の fill_rate≥70% と整合（30% miss は worst case） |
| S5 | 複合 stress | PF > 0.90 | slippage_ticks: 1 + miss_rate: 0.15 | slippage と miss の同時発生を想定した保守的条件 |

**測定方法**: G3 PASS 時と同一の OOS データ・同一モデルで、stress パラメータを注入して再評価。4 seed 全てで実施。

**stress パラメータの注入方法**:
- **slippage_ticks**: 各約定時に `slippage_ticks × tick_size` (1 JPY) を不利方向に加算。Buy なら価格+1、Sell なら価格-1。
- **maker_miss_rate**: 各約定判定時に確率 `miss_rate` で約定を無効化（ポジション変動なし）。seed 固定の乱数で再現性を保証。

**FAIL 時**: S1-S3 FAIL → slippage 耐性不足。取引頻度削減・エントリ閾値引き上げで改善。S4-S5 FAIL → maker miss 耐性不足。fill_rate 改善施策（offset 調整等）を G4 前に実施。全条件 FAIL でも v461 には移行せず、パラメータ調整で対処（G3 PASS 済みのため基本収益性は確認済み）。

**判定出力**: G3 判定 JSON に `g3_1_stress_judgment_cache` として追記。`stress_params` と各 seed の stress 下指標を記録。

### §3.6 G4-live

**目的**: Paper trading で実運用条件を検証。

| 条件 | 閾値 | 測定方法 |
|------|------|---------|
| 連続稼働 | ≥ 7 日 | ダウンタイム < 1% |
| Circuit Breaker テスト | 発動確認済 | 手動テスト |
| G3 指標の維持 | Paper 期間中も G3 閾値内 | リアルタイム監視 |
| 緊急停止応答 | < 1 秒 | 手動テスト |

**FAIL 時**: 本番投入禁止。インフラ修正後に再 Paper。

### §3.7 統計検定仕様

| 項目 | 仕様 |
|------|------|
| 検定方法 | Mann-Whitney U 検定（ノンパラメトリック） |
| 有意水準 | α = 0.05 |
| 多重比較補正 | Holm-Bonferroni 法 |
| 効果量 | Cliff's Delta（\|d\| > 0.33 で中程度） |
| p値統合 | **p平均法**（幾何平均）— 複数 fold の p 値を統合判定 |
| サンプル数 | 各条件 n ≥ 16（4 seed × 4 split）以上 |
| G1 最小サンプル | n ≥ 20（5 folds × 4 seeds） |

**p平均法**: データを N 分割し、各分割で Mann-Whitney U 検定を実行、得られた p 値の幾何平均で統合判定する手法。既存実装 `ztb.metrics.metrics.p_mean_method()` を使用。Holm-Bonferroni と併用し、両方 PASS で Gate 通過とする。

参照: [P_MEAN_METHOD_README.md](../../docs/guides/P_MEAN_METHOD_README.md) / [v459/24#](../v459/24_phase3_specification.md)

v459 00# §5.6 を踏襲。実装コードは [001#](001_ph0_plan_technical_specs.md) に記載。

### §3.8 Gate 枝番規則

| 規則 | 内容 |
|------|------|
| 枝番は `.1`, `.2`, `.3` の整数連番 | `.5` や `.25` 禁止 |
| 再帰禁止 | `G1.1.1` は禁止 |
| 名前必須 | `G1.1-exec` のように用途名を付ける |
| 上限 | 1 親 Gate あたり最大 3 枝番。超えたら Gate 体系再定義 |

### §3.9 継続中止ルール

**目的**: 見込みのない戦略への時間投下を防止し、早期撤退を制度化する。012# §3 #5 で要求、014# §3.2 で具体化。

| 条件 | n 最低要件 | 判断 | 根拠 |
|------|-----------|------|------|
| fill_rate < 70% | n ≥ 200 | **中止** — maker 戦略不成立 | 約定しない maker は収益機会ゼロ |
| AS_ratio > spread/2 が継続 | n ≥ 500 | **中止** — 逆選択コスト過大 | 約定しても逆選択でコスト割れ |
| G1 再検証で全 9 ターゲット FAIL | — | **v461 移行** | 特徴量に情報なし |
| G1 再検証で方向 IC > 0.04 (BE ライン) | — | **続行** — 収益性の可能性あり | 010# §4 損益分岐分析に基づく |
| 累積実損 > 10,000 JPY (fill_test 中) | — | **一時停止** — 原因分析 | 実損キャップ |

**運用**: 各条件を満たした時点で 014# 実施ログに記録し、判断結果を明示。中止判断は 000# への改訂 (§3.9 行の状態更新) で確定とする。

### §3.10 Gate 例外ルール

**目的**: Gate FAIL 判定時に、一定条件下で限定的にテスト継続を許可するルール。Gate 制度の空洞化を防ぎつつ、構造的に収益可能性がある局面での早期打切りを回避する。170# §10.3 で定義、Codex R1 + Gemini 9.2 が必要性を指摘。

#### Exception-1: Oracle Gap 継続例外

| 項目 | 内容 |
|------|------|
| 発動条件 | K1/F1 FAIL かつ Oracle PnL mean > 0 かつ 累積実損 < 3,000 JPY |
| 有効期限 | Gate FAIL 判定日から 72h |
| 再判定日 | 有効期限終了時 (自動) |
| 解除条件 | Oracle PnL ≤ 0 or 累積実損 ≥ 3,000 JPY |
| 適用上限 | 連続 2 回まで (2 回連続例外後は強制停止) |
| 記録義務 | 例外発動ごとに docs/v460 に理由と判定を記録 |

**根拠**: Oracle PnL 正 = 情報量が存在する (ph1 PASS 継続) の実測証拠。執行精度が追いついていないだけで情報量の毀損ではない。ただし無制限継続は Gate 制度の空洞化になるため有効期限と適用上限を設ける。

---

## §4 技術概要

### §4.1 基本仕様

| 項目 | 決定事項 | 備考 |
|------|---------|------|
| 対象市場 | BTC/JPY（国内取引所） | Coincheck（主）/ Bitflyer / Zaif。API 品質・流動性で選定 |
| 時間足 | 1 分足 (OHLCV) + tick/板 | マイクロストラクチャ特徴量の追加 |
| 特徴量 | 板情報・約定フロー・VWAP 等 | v459 の RSI×7 を置換。候補は 001# |
| 情報量検証 | XGBoost Walk-Forward → G1-info | K2 パイプライン流用 |
| 手数料モデル | ExchangeFeeModel（取引所別） | maker-only 前提。Coincheck maker 0% がデフォルト |
| 実験管理 | 1 ランナー + N 設定 YAML + manifest.jsonl | 119# §3–§5 準拠 |

### §4.2 アーキテクチャ: 三層分離モデル（426#/428#/430# 改訂）

422#-426# の実験と 427#-431# の実装監査により、SAC End-to-End アーキテクチャから三層分離モデルへ転換する。

```
+------------------+     +--------------------+     +------------------+
| Alpha 層          |     | Execution 層        |     | Safety 層          |
|                  |     |                    |     |                  |
| SAC [-1,1]       | --> | Sidecar v2         | --> | Final Clamp      |
| (directional     |     | (比例変換)         |     | (ceiling/floor)  |
|  bias 生成)      |     |    +               |     |    +             |
|                  |     | offset chain       |     | SkipGate         |
|                  |     | (regime/VG/kyle/   |     |    +             |
|                  |     |  amihud/EV)        |     | Circuit Breaker  |
+------------------+     +--------------------+     +------------------+
         ^                                                    |
         |            +---------------------+                 |
         +------------|  retrain_scheduler  |<----------------+
                      |  (定期再訓練)        |
                      |  430# 発見:         |
                      |  実装済み・未起動   |
                      +---------------------+
```

| 層 | 責務 | 主要コンポーネント | 状態 |
|---|---|---|---|
| **Alpha** | 方向バイアス生成 | SAC (1次元 continuous [-1,1]) | 学習済み・運用中 |
| **Execution** | 注文価格決定 | Sidecar v2 + offset chain | Sidecar v2 実装済み (430#)、v1→v2切替未完 |
| **Safety** | リスク制御 | Final Clamp (421#) + SkipGate | Clamp 稼働中、buy ceiling 100%発火 (431#) |
| **適応** | レジーム追従 | retrain_scheduler (~700 LOC) | 完全実装済み・未起動 (430#) |

**重要**: SAC の出力は直接注文価格を決定しない。Sidecar v2 で offset に変換され、Final Clamp で安全範囲に制約される。SAC 単体のレジーム汎化限界（426# §3.2: mid 期崩壊）を、retrain_scheduler による定期再訓練で補完する。

### §4.3 SAC 学習仕様（426# 確定）

| 項目 | 仕様 | 根拠 |
|------|------|------|
| アルゴリズム | SAC (G1 通過後のみ) | v459 基盤を継承 |
| 行動空間 | continuous_1d [-1, 1] | directional bias 。End-to-End 価格決定ではない |
| reward 関数 | **reward_clean** を標準 | `step_pnl × 100`、補助ペナルティ全無効。426# §3.3 で reward_tuned より頼健と実証 |
| ent_coef | 0.01 (固定) | auto より 20K step 短期学習では exploitation 集中が合理的 (426# §6.3) |
| val_ratio | **0.10 を標準** | 0.02 は楽観バイアス (426# C1)。0.20 は 20K 学習に対して過酷 |
| 評価 | multi-slice OOS + best_model 並行 | 425#-426# で実装済み |

### §4.4 適応運用

段階導入: A (パラメータ適応, ph2–) → B (定期再訓練, ph3.1–) → C (リアルタイム学習) は v461 判断。028#–030# 参照。

**方策B の具体化** (430# 改訂): `sac_retrain_scheduler.py` が完全実装済み (L267-447 retrain loop, L730-793 signal update, L489-545 trigger logic)。ops script 一つで Sidecar pipeline が全通する。ph3.1 で起動する。

詳細なアーキテクチャ・データ仕様・特徴量候補は [001#](001_ph0_plan_technical_specs.md) に記載。

---

## §5 命名規則・運用規約

### 番号体系

- **ドキュメント番号** (例: 123#): `docs/v460/` 配下の通し番号。正式な記録単位。
- **セッション番号** (例: 125.1#): 対話セッションの識別子。**枝番号は小数点**で表記する (例: 124.2#, 124.3#)。
- ドキュメント番号とセッション番号は独立。1セッションで複数ドキュメントを生成可能、逆も然り。
- **原則**: 参照はドキュメント番号を使用する。セッション番号は経締の追跡が必要な場合のみ記載。

### コード内コメントの番号付与 (130# Q5 / D.1)

- コード内コメントでは **ドキュメント番号** (`130#`, `129#` 等) を参照する。
- セッション番号 (`129.1#` 等) をコード内で使用しない。
- ドキュメントが未作成の段階では、仮番号 (`TBD-YYYYMMDD` 形式) を使用し、ドキュメント確定時に一括置換する。
- コミットメッセージでは `D130#` (ドキュメント) と `S129.1#` (セッション) を接頭辞で区別するか、ドキュメント番号のみを使用する。

### ファイル名形式

```
NNN_phX_type_subject.md
```

| 要素 | 規則 |
|------|------|
| `NNN` | 3 桁ゼロ埋め連番 |
| `phX` | Phase 略号（小文字: ph0, ph1, phg） |
| `type` | plan / rev / resp / rpt / fix / ext / meta |
| `subject` | 英語スネークケース（日本語禁止。本文は日本語可） |

### レビューチェーン

- `rev` は対象番号を含む: `006_ph1_rev_005.md`
- `resp` も対象番号を含む: `007_ph1_resp_006.md`
- 同名禁止（subject は一意）

### ディレクトリ分離

| 種別 | 配置先 | docs/v460 には含めない |
|------|-------|----------------------|
| AI プロンプト | prompts/v460/ | ✓ |
| 日次実験ログ | experiments/v460/ | ✓ |
| 実験結果 | results/v460/ | ✓ |
| 実験データ JSON | results/v460/ | ✓ |

### アーカイブ方針

`docs/v460/archived/` への移動は **文書数ではなく以下の基準** で判断する。重要文書は数に関係なく `docs/v460/` に残す。

| 基準 | 説明 | 例 |
|------|------|-----|
| **被代替** | 後続文書に内容が完全に包含された | 065_as_lr_prep → 065_ph1_impl に統合 |
| **rev/resp チェーン完結** | rev → resp 往復が終了し結論が impl/rpt に反映済 | 006_rev → 007_resp → impl 済 |
| **命名規則違反 (修正不能)** | `NNN_phX_type_subject.md` に準拠しない旧文書 | phX/type 欠落の生成レポート |

**アーカイブしない文書**:
- 000#/001# (プロジェクト定義・技術仕様)
- Gate 判定結果 (rpt 型)
- Codex レビューパッケージ
- 現在進行中の Phase の plan/impl 文書

詳細: [v459/117# §3](../v459/117_v460_doc00_design_and_naming_reform.md) / [v459/119# §8](../v459/119_v460_launch_integrated_policy.md)

---

## §6 リスク

| 重要度 | リスク | 緩和策 | 状態 |
|--------|--------|--------|--------|
| ⭐⭐⭐ | マイクロストラクチャ特徴量にも情報がない | G1-info を Phase 1 で即実行。FAIL なら即座に特徴量再設計 | G1 PASS 済 |
| ⭐⭐⭐ | maker 注文が約定しない（薄い板） | G1.1-quick (72h) で早期棄却 + G1.2-full (168h) で品質検証 | G1.1 PASS 済 |
| ⭐⭐⭐ | 取引所 API 品質低下・仕様変更 | 取引所非依存設計 (ExchangeFeeModel) | 実装済 |
| ⭐⭐⭐ | **SAC のレジーム汎化限界** (426# 新規) | retrain_scheduler による定期再訓練。SAC を Sidecar Navigatorに降格し単体依存を排除 | **対策実装済み・未起動** (430#) |
| ⭐⭐⭐ | **評価系と本番の乖離** (431# 新規) | clamp 条件付き OOS 評価の導入 (426# P3)、Sidecar 有効レコードの積み上げ | **対策未着手** |
| ⭐⭐ | v459 と同じ「設計は綺麗だが実験で崩れる」パターン | manifest 強制・Gate 順序厳守。G1 FAIL 時点で止まる設計 | 運用中 |
| ⭐⭐ | 見込みのない戦略への時間沈没 | §3.9 継続中止ルールで早期撤退を制度化 | 運用中 |
| ⭐⭐ | 市場レジーム変化によるモデル陳腐化 | 定期再訓練 (方策B) + G3 指標モニタリング | retrain_scheduler 実装済 (430#) |
| ⭐⭐ | **val_ratio 交絡による評価楽観** (422# 新規) | G3 判定は val_ratio≥0.10 を標準化。結果 JSON に val_ratio 必須記録 | §3.5 改訂済 |
| ⭐ | **reward 関数の補助ペナルティ汚染** (426# 新規) | reward_clean (純粋 PnL) を標準とし、reward_tuned は終了 | §4.3 確定済 |

---

## Appendix A: 改訂履歴

| 日付 | §番号 | 変更内容 | 理由 |
|------|-------|---------|------|
| 2026-02-13 | — | 初版作成 | v459 No-Go 確定、v460 始動 |
| 2026-02-13 | §3.2–§3.7 | 120# レビュー反映: Holm 補正明記、fill quality 追加、worst-seed 指標追加、G1 最小サンプル規定 | v459/120# |
| 2026-02-13 | §0,§1,§3.3,§4,§6 | Zaif 専用記述を取引所汎用化。Coincheck/Bitflyer/Zaif の多取引所対応。ExchangeFeeModel 採用 | Zaif API 品質問題での移行実績を反映 |
| 2026-02-13 | §3.7 | p平均法を統計検定仕様に追加。既存実装 `p_mean_method()` を活用 | 複数 fold/seed の p 値統合判定の補強 |
| 2026-02-13 | (001#) §6 | 実装ロードマップ追加: 全 Phase のギャップ分析・タスク分解・クリティカルパス・既存コード流用計画 | 実装要件の体系化 |
| 2026-02-13 | (001#) §1,§3,§4,§5,§6,§7 | 002# レビュー反映: tick raw 二層保存方針、IBroker 互換レイヤ設計、async/sync I/O 方針、DataScheduler 流用範囲明確化、G1 判定アルゴリズム厳密仕様化 (§5.3)、技術的負債事前解消 (§6.6)、章番号再採番 | v460/002# |
| 2026-02-13 | §3.3 | G1.1 測定期間の柔軟化: 暫定判定条件 (n≥200, 3暦日以上) を追加。デフォルト 7 日間は維持 | 012# §5, 011# #2 |
| 2026-02-13 | §3.9, §6 | 継続中止ルール追加。fill_rate<70% で中止、AS>spread/2 で中止、実損キャップ 10,000 JPY | 012# §3 #5, 014# §3.2 |
| 2026-02-14 | §4, §6 | 適応運用方針 (A/B/C段階導入) を§4に追記、モデル陳腐化リスクを§6に追加 | 028#–030# |
| 2026-02-14 | §3.4-§3.6, §3.3 | Gate G2/G3/G4 スタブ完成。方策A パラメータ適応 core 実装・fill_test 統合。品質改善6件 (セキュリティ, JSONL耐性, deprecated API修正等) | 032# |
| 2026-02-14 | §3.3, §3.9, §4 | 方策B 動的ロットサイジング実装 (0.001→0.005 BTC段階). 累積PnL 10K JPYキャップのランタイム監視・自動停止. Fill test 295件分析: AS改善トレンド確認 | 033# |
| 2026-02-14 | §3.3, §4 | fill_test 全設定を configs/v460/fill_test.yaml に外部化。FillTestConfig.from_yaml() + CLI>YAML>defaults 優先チェーン。v459反省 (config形骸化) 解消。10テスト追加 (396 total) | 034# |
| 2026-02-15 | §5 | アーカイブ方針改定: 「40文書以内」数値基準を廃止。被代替・rev/resp完結・命名違反の3基準に変更。重要文書はアーカイブしない方針を明記 | 065# 整理時 |
| 2026-02-19 | §3.3, §6 | G1.1-exec を二段階化: G1.1-quick (72h Kill) + G1.2-full (168h Qualification)。SkipGate attempted ベース導入、PnL Kill 複合条件、Watch 層追加。115# 外部レビュー反映 | 114#, 115#, 116# |
| 2026-02-19 | §3.3 | 分母定義 (raw/clean/attempted) 一覧表追加。cancel reason 内訳表追加。3 系列同時出力ルール明記。115# Q10.6 対応 | 117# |
| 2026-02-21 | §5 | 番号体系 (ドキュメント番号 vs セッション番号) を追記。123# から移設 | 128# |
| 2026-02-21 | §5, §3.3 | コード内コメント番号付与ポリシー追記 (Q5)。Gate 判定前提条件 (同一 SHA・YAML 72h) 追記 (E.5 #3)。YAML 反映を「反映確認済み」で管理する運用規約追記 | 130# D.1 |
| 2026-02-27 | §3.10 | Gate 例外ルール (Exception-1: Oracle Gap 継続例外) 追加。Gate FAIL 時の限定的継続許可条件を制度化 | 170# §10.3, 172# |
| 2026-03-13 | §3.5.1 | G3.1-stress Gate 正式定義。slippage 1tick / maker miss 30% / 複合 stress の 5 条件。G3 PASS 後の friction 耐性検証を制度化 | 392# P1-2, 409# |
| 2026-03-15 | §0, §2, §3.4, §3.5, §4, §6 | **422#-431# 大幅改訂**: §0.1 SAC役割定義（Driver→Sidecar Navigator）追加。§2 ph3.1 Sidecar統合Phase追加。§3.4 val_ratio規定・step数汎用化。§3.5 val_ratio標準化(≥0.10)・multi-slice診断・best_model並行評価・reward_clean標準化。§4 三層分離アーキテクチャ(Alpha/Execution/Safety)に全面改訂、SAC学習仕様(§4.3)・適応運用(§4.4)分離。§6 レジーム汎化限界・評価本番乖離・val_ratio交絡・reward汚染を新規リスクとして追加 | 422#-426# S1/S1'実験, 427#-431# 実装監査 |



---

## 運用方針追補（VS Code同期制限回避）

- VS Code 拡張の同期制限（実質 50MB 前後）を前提に、巨大ファイルは直接全文読取を行わない。
- 調査レビューは PowerShell の部分抽出（Get-Content -Skip -First, Select-String -Context）を標準手順とする。
- ログ分析の最終判定は run_id / git_sha / 期間固定で再現可能にし、ワイルドカード横断のみで結論を出さない。
- 成果物は肥大化を避け、章分割日次分割アーカイブ分離（docs/v460/, reports/, temp/）を徹底する。
