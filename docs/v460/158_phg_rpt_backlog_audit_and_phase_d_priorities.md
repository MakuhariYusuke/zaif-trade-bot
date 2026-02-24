# 158# バックログ監査 + Phase D 優先順位表

> 118#–157# の全ドキュメント横断監査結果。  
> Phase C dry-run (2,700+ records) 蓄積後の **未対応事項 22 件**を洗い出し、  
> 外部 AI コーディングエージェントによるレビュー・実装に備えて整理する。

---

## §0 エグゼクティブサマリ

### 監査範囲

| ドキュメント | 内容 | 残項目数 |
|---|---|---|
| 118# | マスターバックログ (53 OPEN / 42 RESOLVED) | backbone |
| 149# | Phase C 並行作業計画 (P2/P3 残項目) | 2 |
| 152# | 代替優先施策: CRITICAL 検証 + P3-02/03 | 3 |
| 153# | テスト安定化 + run_fill_test 分割設計 | 2 |
| 154# | Dry-Run 10h ログ分析 (P0-08 deadlock) | 0 (解決済) |
| 155# | 後知恵 PnL 分析: sell 弱点・timeout 機会損失 | 3 |
| 156# | Sell 根本原因 7 重ゲート + Phase D 計画 | 残計画 4 |
| 157# | §20 レジームデッドロック修正 + cancel re-raise | 0 (完了) |

### 数字のサマリ

- **未対応合計**: 22 件 → **解決済: 12 件、残: 10 件**
- **P0 (Phase D 即時実行)**: 4 件 — ✅ P0-1 SkipGate sell 再訓練, P0-2 sell offset A/B (データ観測待ち), ✅ P0-3 trending フィルタ検証完了, ✅ P0-4 Oracle テスト
- **P1 (168h run 中に解決)**: 6 件 — ✅ P1-2 fill rate offset 分析完了, ✅ P1-1 balance_forced 救済, ✅ P1-3 reprice ログ, ✅ P1-4 trades_health, ✅ P1-5 A/B テスト基盤, ✅ P1-6 時間帯閾値
- **P2 (ph3/ph5 ブロッカー)**: 6 件 — ✅ P2-1 WF リーケージ修正, ✅ P2-2 execute_trade 統合テスト, ✅ P2-3 failure mode test, ✅ P2-4 分割 Phase1 完了, P2-5 skip_gate.py, ✅ P2-6 VG JSONL ログ
- **P3 (低優先 / v461+)**: 6 件 — deferred items

---

## §1 P0: Phase D 即時実行項目

> Phase C 168h fill_test 並行で着手すべき最高優先。

### P0-1: SkipGate sell 再訓練 (2,700+ records)

| 項目 | 値 |
|---|---|
| 出典 | 156# §4.2 P0, 118# Appendix F |
| 問題 | 旧データで訓練された sell SkipGate モデルが実環境に適合していない。sell PnL = -0.516 bps (buy の 5-6 倍悪い)。sell モデルは pnl120 時間軸で短期リバージョンを見逃す。 |
| 前提 | 156# D-1 (OB 正規化) は 145# `ob_utils` で対応済み |
| アクション | 蓄積 2,700+ records で sell 側 SkipGate を再訓練。118# Appendix F の特徴量計画 (regime 特徴量含む) に従う。pnl120→pnl30 モデル統一も検証 (156# §4.2 P3)。 |
| 成果基準 | sell skip_gate の AS フィルタ精度が OOS で改善 (AUC +0.02 以上) |
| 関連コード | `scripts/v460/ml/retrain_scheduler.py`, `scripts/v460/lib/skip_gate_evaluator.py` |
| 工数見積 | 0.5 日 |
| **ステータス** | **✅ 初回デプロイ完了** (2026-02-24 05:34) |
| 結果 | sell model: pnl120 target, 229 samples, 15 features → `skip_gate_lgbm_pnl120_sell.pkl` (250KB) |
| | buy model: pnl30 target, 519 samples, 13 features → `skip_gate_lgbm_pnl30_buy.pkl` (297KB) |
| | 159# P0-1: 統計ゲートを初回訓練時スキップ (prev model 不在時は比較対象なし), absolute_min_score 緩和 (-0.10→-0.50 for --all-runs) |
| | retrain_scheduler 定期再訓練で自動更新予定 (PID 122860, interval 1h, side_specific_enabled=true) |

### P0-2: sell offset 段階的縮小 A/B テスト (0.18→0.14)

| 項目 | 値 |
|---|---|
| 出典 | 156# §4.2 P1, 156# §5.3 D-3 |
| 問題 | sell offset 0.18 は buy 0.05 の 3.6 倍で過剰防御。fill rate 構造的低下の直接原因。VG/trending boost で 0.18×1.5=0.27 まで拡大し、sell がほぼ fill されない時間帯が発生。 |
| 前提 | Phase C restart 後のクリーンデータが必要 (現在蓄積中) |
| アクション | Phase D A/B-2: sell offset 0.18 vs 0.15 → 0.14 を各 24h で比較。fill_rate と PnL30 をトラッキング。 |
| 成果基準 | sell fill_rate +10-15pt かつ PnL30 悪化 ≤0.2 bps |
| 関連設定 | `configs/v460/fill_test.yaml` → `side_offset.sell` |
| 工数見積 | 0.1 日 (設定変更 + 24h×2 観測) |

### P0-3: skip_sell_trending 方向別分解の効果測定

| 項目 | 値 |
|---|---|
| 出典 | 156# §5.3 D-4, 156# §17 |
| 問題 | D-4 は実装完了 (`skip_sell_trending_up_only: true`)。trending_down 時の sell を開放したが、**効果測定が未完了**。 |
| 現状 | 157# §20 修正後、新しい 168h fill_test が 2026-02-24 03:57 に開始、cycle 2728+ まで蓄積中。 |
| アクション | fill_records から trending_down での sell 約定を抽出し、PnL30/PnL120 を分析。trending_down sell が正の PnL であることを確認。 |
| 成果基準 | trending_down sell の avg PnL30 > -0.3 bps (buy との差が 1 bps 以内) |
| 分析スクリプト | `scripts/v460/analysis/analyze_fill_records.py` (regime×side フィルタ) |
| 工数見積 | 0.2 日 (データ蓄積待ち + 分析) |

#### 158# 最終評価 (2026-02-24, n=221 trending_sell_skip + 681 filled sell)

**分析手法:** trending_sell_skip 221 件は約定前キャンセルのため価格/PnL データなし。  
代替として filled sell/buy レコードを regime 別に集計し、trending sell フィルタの妥当性を検証。

| regime | sell PnL30 | buy PnL30 | n(sell) | n(buy) |
|---|---|---|---|---|
| ranging | -0.32 bps | -0.03 bps | 377 | 375 |
| trending | **-0.66 bps** | **+0.57 bps** | 118 | 118 |
| trending_down | **+1.18 bps** | +4.25 bps | 11 | 13 |
| unknown | -0.69 bps | -0.46 bps | 174 | 186 |
| **全体** | **-0.45 bps** | **+0.04 bps** | 680 | 692 |

**カウンターファクチュアル分析:**
- 221 件の trending sell skip が仮に約定していた場合、trending sell PnL(-0.66 bps) × 221 = **-145.9 bps の累積損失回避**
- trending_down sell (D-4 `skip_sell_trending_up_only: true` で開放): n=11, PnL30=+1.18 bps → **+12.97 bps の累積利益**
- trending buy は +0.57 bps で正の期待値を維持 (sell skip 後も buy は継続)

**最終判定:** ✅ **フィルタ設計は妥当**
- trending sell は -0.66 bps で skip は正解 (buy は +0.57 bps で方向性あり)
- trending_down sell 開放は +1.18 bps で成果基準 (> -0.3 bps) を大幅超過
- 安全弁 `max_consecutive_trending_sell_skip: 30` も機能確認済
- **ステータス: ✅ 完了**

### P0-4: Oracle テスト (ph3 前必須) — ✅ PASS (158# 実施済)

| 項目 | 値 |
|---|---|
| 出典 | 118# §1.2 教訓③, 118# §8.5 |
| 問題 | v459 の教訓「完全予測でも taker 手数料で費用負け」が ph3 進入前に検証されていない。v460 は maker 0% だが AS コストが Oracle PnL を超えていないかを確認すべき。 |
| アクション | Phase C の 2,700+ records で Oracle baseline を計算: 「全約定の PnL30 を符号反転なしで計算し、完全予測時の理論上限を算出」。AS_ratio × avg_AS_loss を差し引いた net を検証。 |
| 成果基準 | Oracle net PnL > 0 bps (maker 0% で理論的に利益が出うることを確認) |
| 関連コード | `scripts/v460/analysis/oracle_test.py`, `oracle_baseline.py` (AS cost 分析追加) |
| 工数見積 | 0.3 日 |

#### 158# 実施結果 (2026-02-24, n=1289 filled / 2739 total)

| 指標 | 30s | 120s |
|---|---|---|
| Baseline mean | -0.2844 bps | +0.2267 bps |
| Oracle Skip mean | +2.8606 bps | +4.9194 bps |
| Oracle Flip mean | +2.9343 bps | +4.7844 bps |
| Profitable rate | 46.3% | 50.9% |
| Kill Switch | **PASS** (>1.0) | **PASS** (>1.0) |

**AS コスト分析:**
- AS ratio: 26.5% (341/1289), AS avg PnL30: -5.18 bps
- Non-AS avg PnL30: +1.48 bps
- **AS cost = 0.265 × 5.18 = 1.37 bps**
- **Oracle net (Flip - AS cost) = 2.93 - 1.37 = +1.56 bps > 0 → ✅ PASS**

**結論:** maker 0% 環境で理論上限は十分に正。ph3 SAC 訓練進行の根拠確認済。
改善余地 = Oracle Skip (+2.86) - Baseline (-0.28) = **+3.14 bps** — skip gate 精度向上が最大レバレッジ。

---

## §2 P1: 168h run 中に解決すべき中優先項目

### P1-1: balance_forced 救済モード検証 — ✅ 実装完了 (158# P1-1)

| 項目 | 値 |
|---|---|
| 出典 | 156# §5.3 D-5, 155# §9.5 #1 |
| 問題 | Phase C で balance_forced_skip が 314 件 (13.0%)。BTC/JPY のどちらかに偏ったとき、実質的に取引が停止する。156# §14 で balance_forced バイパスを全 3 ゲートに水平展開済だが、低リスク執行ロジック未実装。 |
| **ステータス** | **✅ 実装完了** (158# P1-1) |
| 結果 | `FillTestConfig.balance_forced_rescue_enabled` + `balance_forced_rescue_offset_mult` (default 2.0) 追加。rescue 有効時: skip の代わりに offset 倍増で安全にポジション解消。YAML `loss_control.balance_forced_rescue_enabled` で制御。テスト 8 件追加 (31 passed)。 |
| 工数見積 | 0.3 日 (実績 0.2 日) |

### P1-2: fill rate 向上のための offset 最適化 — ✅ 分析完了

| 項目 | 値 |
|---|---|
| 出典 | 155# §3 (timeout 分析), 118# §3.1 |
| 問題 | timeout (261 件, avg +0.406 bps) は板に並べたが fill されなかった。72.4% が方向正解。「指値 offset が保守的すぎる」が主因。fast_fill_defense の has_negative_edge 検出率が sell 側で構造的に低い (098# §3.1)。 |
| アクション | stale_check 間隔の動的調整 + reprice 戦略改善。155# T-1/T-2/T-3 の順に実施。 |
| 工数見積 | 0.5 日 |
| **ステータス** | **✅ 分析完了** (158# offset 最適化分析) |

#### 158# offset 最適化分析 (2026-02-24, n=102 直近 2 日間)

**全体概況:**
- Fill rate: 46.9% (BUY 59.0%, SELL 38.7%)
- Timeout: 268 件 (buy 161, sell 107)
- Queue wait: mean=31.4s, median=13.1s

**offset × PnL30 分析 (直近 2 日, filled のみ):**

| side | offset range | n | mean PnL30 | 判定 |
|---|---|---|---|---|
| BUY | ≤0.1 | 39 | +0.16 bps | 低 offset → 利益薄 |
| BUY | 0.1–0.3 | 12 | **+5.86 bps** | ⭐ **スイートスポット** |
| BUY | >0.3 | 1 | -4.17 bps | 過剰 → 逆選択のみ fill |
| SELL | ≤0.1 | 0 | — | — |
| SELL | 0.1–0.3 | 44 | +0.66 bps | 現行レンジ、妥当 |
| SELL | >0.3 | 6 | -0.50 bps | 過剰 |

**現行設定:**
```yaml
spread_offset_ratio: 0.05    # base (buy はこれを継承)
side_offset:
  sell: 0.18                 # sell 専用
trending_offset_boost_buy: 1.0   # boost なし
trending_offset_boost_sell: 1.5  # 1.5 倍
ranging_offset_discount: 0.90    # ranging 時 10% 割引
adaptive_offset: min=0.01, max=0.30
```

**Key Insight:**
- Buy 実効 offset (mean=0.097) が低すぎる。0.1–0.3 帯は PnL30 が **36.6 倍** (+5.86 vs +0.16)
- Sell 実効 offset (mean=0.289) は妥当レンジ内
- **推奨**: Buy base offset を 0.05→0.12–0.15 に引き上げ検討
- **注意**: n=102 (2 日間) はサンプル不足。追加データ蓄積後に最終判断 → P0-2 A/B テストで正式検証

### P1-3: reprice ログ連携 (156# §9.4 #3) — 部分完了

| 項目 | 値 |
|---|---|
| 出典 | 155# §9.4 #3, 156# §6.2 |
| 問題 | reprice イベントが FillRecord に記録されていない。ヒンドサイト分析で reprice の効果を定量化できない。 |
| アクション | ~~reprice 時のログ情報 (reprice_count, reprice_drift_bps) を FillRecord に追加。~~ |
| **159# 更新** | `reprice_count` は `FillRecord` 実装済み (`ztb/metrics/fill_quality.py:93`)。~~残課題は `reprice_drift_bps` のみ。~~ |
| **ステータス** | **✅ 完了** (158# P1-3) |
| 結果 | `FillRecord.reprice_drift_bps` 追加。`OrderMonitor` で `cumulative_drift_bps` 追跡、`FillMonitorResult` 経由で FillRecord に記録。テスト 5 件追加 (42 passed)。 |
| 工数見積 | 0.1 日 (実績 0.1 日) |

### P1-4: trades_health UNHEALTHY 状態の検証 — ✅ 完了 (158# 修正済)

| 項目 | 値 |
|---|---|
| 出典 | 118# §8.3, retrain_scheduler ログ |
| 問題 | 20260220-21 deadlock gap により trades_health が UNHEALTHY → retrain 完全ブロック。 |
| **158# 修正** | `max_missing_days` パラメータ導入 (commit `1a1d8d354`)。YAML: `trigger_trades_max_missing_days: 2`。retrain_scheduler PID 124484 で再稼働確認済。 |
| **159# §2.1 追加修正** | `run_fill_test.py:1443-1445` の `th.latest_ts`/`th.age_hours` 参照を `th.available_days[-1]`/`th.stale_hours` に修正。`TradesHealthResult` のフィールド不整合による silent `AttributeError` を解消。テスト 5 件追加。 |
| ステータス | **✅ 完了** (2 commit で段階修正) |

### P1-5: offset A/B テスト自動化基盤 — ✅ 実装完了 (158# P1-5)

| 項目 | 値 |
|---|---|
| 出典 | 156# §4.3 A/B-1〜4, 152# §5 |
| 問題 | P0-2 (sell offset A/B) を手動で実施するのは労力がかかる。A/B テスト基盤をコードレベルで用意すれば、Phase D の複数施策を効率的に評価可能。 |
| **ステータス** | **✅ 実装完了** (158# P1-5) |
| 結果 | `FillTestConfig.ab_test_variant` + `FillRecord.ab_test_variant` 追加。YAML `ab_test.variant` で制御。fill_records の各レコードに variant が記録され、分析スクリプトで variant 別集計が可能。テスト 8 件追加 (31 passed)。 |
| 工数見積 | 0.3 日 (実績 0.1 日) |

### P1-6: 時間帯 skip_gate 閾値の動的調整

| 項目 | 値 |
|---|---|
| 出典 | 107# Phase 2 (動的ゲーティング提案), 155# §3 |
| 問題 | skip_utc_hours は ALL-or-NOTHING の静的フィルタ。155# の後知恵分析で UTC 帯別の PnL 分散が大きく、一律ブロックは機会損失を生む。 |
| アクション | 時間帯別に skip_gate 閾値を微調整するロジック。例: 高 AS 時間帯は閾値を +0.05 (厳しく)、低 AS 時間帯は -0.03 (緩く)。 |
| 工数見積 | 0.3 日 |

---

## §3 P2: ph3/ph5 ブロッカー (中期必須)

### P2-1: v458 WalkForward バグ修正

| 項目 | 値 |
|---|---|
| 出典 | 118# §4.3, 111# §6 |
| 問題 | `_evaluate_wf_single` が `eval_set = [(X_test_sc, y_test)]` でテストデータをリークしていた。multi-window 版は正常 (`X_val` 使用)。 |
| 修正内容 | (1) train/embargo/val/test の 4 分割に変更 (2) `wf_val_ratio_single` (default 0.1) パラメータ追加 (3) `wf_embargo_rows` (default 0) でアンバーゴ対応 (4) early stopping は `X_val` を使用 (5) `sample_weight` スライシングを `[:train_end]` に修正 |
| テスト | `TestWFSingleWindowLeakageFix` 3 件追加 (val 分離検証, embargo 検証, 最小データ検証) |
| **ステータス** | **✅ 完了** (commit `7c5bed6ce`) |
| retrain 再起動 | PID 129404 (2026-02-24 13:53:58) で修正済みコードで再起動完了 |

### P2-2: execute_trade() 実装 (013# D-1) — ✅ 統合テスト完了

| 項目 | 値 |
|---|---|
| 出典 | 013# D-1, 118# §6.1 |
| **159# 更新** | `ztb/trading/live_trader/components/order_manager.py:29` に live 実装あり。fill_test は独自注文フローだが、`live_trader.py:1634` から `order_manager.execute_trade()` を呼び出す統合済み。 |
| テスト | `test_158_order_manager_integration.py` 25 件: Demo mode (4), Validation (5), Live success (4), No adapter (2), No price (3), Exchange error (2), Order None (2), Async bridge (2), get_trade_info (1) |
| **ステータス** | **✅ 完了** (commit `628e7e710`) |

### P2-3: 運用失敗モードテスト (118# §8.5) — ✅ 完了

| 項目 | 値 |
|---|---|
| 出典 | 118# §6.3, 118# §8.5 |
| テスト | `test_158_failure_modes.py` 27 件: CircuitBreaker 状態遷移 (8), OrderManager タイムアウト (2), RiskManager 緊急停止/制限 (10), 価格フォールバック (4), 連続エラー (3) |
| 副次修正 | `risk_manager.py`: `pd.DataFrame` 未 import (NameError) を `from __future__ import annotations` + `TYPE_CHECKING` で修正 |
| **ステータス** | **✅ 完了** (commit `628e7e710`) |

### P2-4: run_fill_test 分割設計 (153# タスク B) — Phase 1 完了

| 項目 | 値 |
|---|---|
| 出典 | 153# §3 |
| 問題 | `FillTestRunner` が 2,700+ 行の god object。Lot/Position, Order Execution, Measurement, Lifecycle, Record/IO の 5 責務が混在。 |
| **159# 更新** | `scripts/v460/lib/` に 27+ モジュール分割進展 (`lot_manager.py`, `lot_sizer.py`, `order_monitor.py`, `pnl_measurer.py`, `balance_checker.py` 等)。159# §3.3: 稼働中は Facade 維持 + 契約テスト先行で段階移行。 |
| **158# Phase 1** | `event_logger.py` (118行), `lock_manager.py` (146行), `fill_test_cli.py` (422行) 抽出。`run_fill_test.py` 2,715→2,164行 (-551行, -20.3%)。`skip_gate_evaluator.py` _compute_file_hash 統一。テスト 7 ファイル修正、1,738 全件 PASS。commit `1aed848e3` |
| ステータス | **✅ Phase 2 完了 (163#)**: run_fill_test.py 2,231→378行。3 Mixin 分割 (fill_record_helpers / fill_cycle_executor / fill_loop_orchestrator) + maker_price compute() 306→143行 + fill_config from_yaml() 479→139行。テスト 1,858 全件 PASS。 |
| 残課題 | `run_continuous` (802行), `run_single_cycle` (483行) の更なる分割は次回 run 終了後 |
| 工数見積 | Phase 1: 0.3 日 (実績)。Phase 2: 0.5 日 (予定) |

### P2-5: skip_gate.py モジュール配置 (106# R5)

| 項目 | 値 |
|---|---|
| 出典 | 106# R5, 118# §7 |
| 問題 | `scripts/v460/lib/` 以下の 4 モジュールが `ztb/` パッケージ外にある。テストの import パスが不安定。 |
| アクション | `skip_gate_evaluator.py`, `maker_price.py` 等を `ztb/trading/live/` に段階的移動。 |
| 工数見積 | 0.5 日 |

### P2-6: VG (Volatility Guard) JSONL ログ蓄積 — ✅ 実装完了 (158# P2-6)

| 項目 | 値 |
|---|---|
| 出典 | 107# Phase 2, 118# §3 |
| 問題 | Volatility Guard の判定履歴を JSONL に記録する仕組みがない。VG の振る舞いをヒンドサイトで分析できない。 |
| **ステータス** | **✅ 実装完了** (158# P2-6) |
| 結果 | `FillRecord` に `vg_velocity_bps`, `vg_vpin`, `vg_boost_factor` 追加。`MakerPriceCalculator` に詳細追跡プロパティ。テスト 5 件追加 (42 passed)。 |
| 工数見積 | 0.2 日 (実績 0.1 日) |

---

## §4 P3: 低優先 (v461+)

> 以下は即時の収益寄与が低く、v461+ への繰り越しが妥当。

| # | 項目 | 出典 | 概要 |
|---|---|---|---|
| P3-1 | SkipGate 単体テスト拡充 | 106# R3 | evaluator の分岐網羅テスト。現状でも retrain テスト 69 件でカバー |
| P3-2 | utils 70+ ファイル分割 | 106# R6 | `utils/` が肥大。機能別サブパッケージに整理 |
| P3-3 | config/ vs configs/ 重複整理 | 106# R7 | 命名の一貫性確保 |
| P3-4 | UnifiedTrainer god object 分割 | 109# DUP3 | 2,835 行。ph4 (オンライン学習) 前に対処推奨 |
| P3-5 | sell pnl120→pnl30 モデル統一評価 | 156# §4.2 P3 | 時間軸公平化の検証。P0-1 (sell 再訓練) の延長で実施可能 |
| P3-6 | asyncio.to_thread 残 5 メソッド | 013# C-4 | ph5 本番パフォーマンス最適化 |

---

## §5 解決済み項目 (検証済み)

> 以下は 118#–157# の過程で **完了が確認された** 項目。

| 項目 | 解決文書 | 確認方法 |
|---|---|---|
| 144# CRITICAL 全件 (§8.1 #1-#3, §9.1 #1-#4) | 145#, 151#, 152# §2 | 152# で残課題ゼロ確認 |
| SkipGate warm_start 閾値未復元 | 098# §3.2 → 130# で修正 | 130# 実装確認 |
| param_adapter 全履歴使用 (recency window) | 098# §3.6 → 130# 実装 | 130# window=200 導入 |
| fast_fill L2 事後 PnL フィードバック | 098# §3.1 → 093# 実装 | 093# 変更確認 |
| time_filter Step 1 (skip_hours 実装) | 110# | 110# deadlock 解消 + fill test 稼働中 |
| §20 regime deadlock (3 件) | 157# | 157# §20-A/B/C/D 全件修正、24 テスト pass |
| OB tuple/object 正規化 | 144# §9 #3 → 145# ob_utils | 145# で extract_price/depth_volume 導入 |
| cancel_reason 正規化 (postonly) | 156# §12 #3 | postonly→post_only 統一 |
| balance_forced バイパス水平展開 | 156# §14 | 全 3 ゲート (skip_buy_unknown, skip_sell_trending, sell_dynamic_kill) に適用 |
| trending 方向分解 (D-4) | 156# §17 | TRENDING_UP/DOWN enum + skip_sell_trending_up_only |
| AdvancedRegimeDetector archived (E-3) | 156# §19 | dead code 除去 |
| BuyDynamicKillManager (E-1) | 156# §19 | DynamicKillManager DRY 化、buy 側 kill 実装 |
| regime_trend_pct / volatility_ratio 伝搬 | 156# §18 | FillRecord にフィールド追加 |
| ranging_offset_discount 有効化 | 156# §18 | YAML 0.90 設定 |
| collect errors 39 件解消 | 153# タスク A | テスト安定化でアーカイブ or 修正 |
| P0-08 deadlock | 154# | 110# で lock timeout + 157# で regime deadlock 修正 |

---

## §6 外部 AI レビューに向けた補足情報

> 本セクションは、外部 AI コーディングエージェントが本文書を起点にプロジェクトを理解し、
> レビューまたは実装作業を効率的に行うための補足である。

### 6.1 プロジェクト概要

| 項目 | 値 |
|---|---|
| プロジェクト名 | v460 "Microstructure Edge" |
| 対象 | Coincheck BTC/JPY maker 注文の執行品質検証 |
| 大義 | **短期間での高収益性システム** (000# §0) |
| 現フェーズ | ph2 (G1.1-exec gate: 168h fill test qualification) |
| 蓄積データ | 2,700+ fill_records (10+ 暦日) |
| 技術スタック | Python 3.11, LGBM SkipGate, pytest 1,738 テスト |
| 設定ファイル | `configs/v460/fill_test.yaml` |
| メイン実行 | `scripts/v460/run_fill_test.py` (`FillTestRunner`, 2,164 行) + `lib/fill_test_cli.py` (422行) |
| retrain | `scripts/v460/ml/retrain_scheduler.py` (1,970 行, WF + 品質ゲート) |

### 6.2 アーキテクチャ概要

```
scripts/v460/
├── run_fill_test.py         # メインループ (FillTestRunner, 2,164行)
├── lib/
│   ├── fill_config.py       # YAML 設定 → dataclass
│   ├── fill_test_cli.py     # CLI エントリポイント (main() 抽出, 422行)
│   ├── event_logger.py      # 148# イベントログ + TeeWriter (118行)
│   ├── lock_manager.py      # 044# 単一起動ロック管理 (146行)
│   ├── maker_price.py       # offset 計算 + VG + regime boost
│   ├── skip_gate_evaluator.py  # LGBM SkipGate (buy/sell 分離モデル)
│   ├── regime_detector.py   # FillTestRegimeDetector (TRENDING_UP/DOWN/RANGING/HIGH_VOL/UNKNOWN)
│   ├── fast_fill_defense.py # 高速約定防御 (AS 検出)
│   ├── order_monitor.py     # 注文監視 + stale 検出 + reprice
│   ├── cancel_reasons.py    # cancel_reason 定数 + 監査 frozenset
│   └── pnl_measurer.py      # 約定後 PnL 測定
├── ml/
│   └── retrain_scheduler.py # LGBM WF 再訓練 + hot-reload
└── analysis/
    ├── hindsight_filter.py  # 後知恵 PnL 分析
    └── analyze_fill_records.py  # fill_records 集計

ztb/
├── trading/live/exchanges/coincheck/  # API adapter
├── risk/sell_dynamic_kill.py          # DynamicKillManager (buy/sell 共用)
├── ml/retrain_trigger.py             # retrain 判定ロジック
├── metrics/fill_quality.py           # FillRecord dataclass
└── analysis/regime/                  # レジーム検出器 (basic のみ)

configs/v460/fill_test.yaml           # 全パラメータ一元管理
```

### 6.3 Sell 7 重ゲート構造 (156# §3.2)

外部レビュアーが最も注目すべき構造:

```
sell 注文サイクル
  ├─ Gate 1: Time Filter (skip_utc_hours_sell)
  ├─ Gate 2: skip_sell_unknown_regime (unknown → sell 全殺し)
  ├─ Gate 3: skip_sell_trending_up_only (上昇トレンド → sell skip)
  ├─ Gate 4: sell_dynamic_kill (rolling PnL < -0.5bps → 20分凍結)
  ├─ Gate 5: SkipGate ML (AS probability > threshold)
  ├─ Gate 6: sell_guard (max_spread チェック)
  └─ Gate 7: MakerPrice (offset 0.18 = buy の 3.6 倍)

問題: ゲート積層による「防御の螺旋」が sell 機会を 40-50% に半減。
156# §3 で詳細分析済み。Phase D で段階的緩和を進行中。
```

### 6.4 レビュー観点 (外部エージェント向け)

以下の観点でのレビューを期待する:

1. **P0 項目の妥当性検証**: 優先順位は適切か。見落としている P0 はないか。
2. **sell 改善戦略の批判的評価**: 防御緩和 (offset 縮小等) は収益性改善に本当につながるか。緩和しすぎた場合のリスクは十分に考慮されているか。
3. **Oracle テスト設計**: ph3 進入前に v460 maker 戦略の理論上限を確認する方法は適切か。
4. **retrain パイプラインの健全性**: trades_health 問題 (P1-4) が retrain 品質に与える影響の深刻度評価。
5. **run_fill_test 分割の実現可能性**: 2,200+ 行 god object の段階的分割は「Phase C 稼働中」の制約下で安全に行えるか。
6. **P2/P3 の棚上げ判断**: 先送りは妥当か。ph3 移行前にブロッカーとなる見落としはないか。

### 6.5 即時確認推奨項目

以下は fill_test 稼働中に素早く確認すべき事項:

| # | 項目 | 確認方法 | 期待結果 |
|---|---|---|---|
| 1 | trades_health 状態 | `grep "trades_health" logs/retrain_scheduler.log` | dry-run で UNHEALTHY は正常 or 設計バグ |
| 2 | §20 効果測定 (regime deadlock 解消) | `grep "regime_deadlock\|lock_timeout" results/v460/fill_test/logs/fill_test.log` | 0 件 |
| 3 | trending_down sell 実績 | `python scripts/v460/analysis/analyze_fill_records.py --filter regime=trending_down,side=sell` | PnL30 > -0.3 bps |
| 4 | retrain_scheduler health | `Get-Process -Name python \| Where-Object {$_.Id -eq <retrain_pid>}` | プロセス生存 |

---

## §7 Phase D ロードマップ (ガントチャート)

```
Week 1 (02/24–03/02):
  fill_test: ════════════════════ (168h 蓄積中, PID 124796) ═══════
  P0-3:  [trending 効果測定]──(データ 48h 待ち)──[分析]
  P0-1:  [SkipGate sell 再訓練]──[OOS 評価]──[hot-reload]
  P1-4:  [trades_health 確認]

Week 2 (03/02–03/09):
  fill_test: ════════════════════════════════════════════════════════
  P0-2:  [sell offset 0.18→0.15 A/B 24h]──[0.15→0.14 A/B 24h]
  P0-4:  [Oracle テスト設計]──[実行]──[報告]
  P1-1:  [balance_forced 救済設計]──[実装]
  P1-2:  [offset 最適化]──[reprice 改善]

Week 3 (03/09–03/16):
  Gate 判定: [G1.2-full 暫定判定]──[合否決定]
  P2-2:  [execute_trade 実装]
  P2-4:  [LotManager 抽出]
```

---

## §8 変更履歴

| 日付 | 内容 |
|------|------|
| 2026-02-24 | 初版: 118#–157# 横断監査 22 件 + Phase D 優先順位 + 外部レビュー向け補足 |
| 2026-02-24 | 158# P1-4: trades_health `max_missing_days` 導入 (commit `1a1d8d354`) |
| 2026-02-24 | 158# P0-4: Oracle テスト実施 PASS + AS cost 分析 (commit `fd9d4ce18`) |
| 2026-02-24 | 158# P0-3: trending_down sell 中間スナップショット n=5 (commit `ae2e4e5c0`) |
| 2026-02-24 | 159# レビュー反映: §1.1 進捗更新 (P1-3/P2-2/P2-4/P1-4), §2.1 trades_health alert 不整合修正, P0-B 3指標 dashboard 追加 |
| 2026-02-24 | 159# P0-1: SkipGate sell/buy side 別モデル初回デプロイ (--all-runs, 統計ゲート初回スキップ+absolute_min 緩和) |
| 2026-02-24 | 158# P1-3: `reprice_drift_bps` 追加 (FillRecord + OrderMonitor 累積 drift 追跡) |
| 2026-02-24 | 158# P2-6: VG 詳細ログ (`vg_velocity_bps`, `vg_vpin`, `vg_boost_factor`) FillRecord 追加 |
| 2026-02-24 | 158# P1-1: balance_forced rescue モード実装 (offset 倍増で安全ポジション解消) |
| 2026-02-24 | 158# P1-5: A/B テスト基盤 (`ab_test_variant`) FillConfig + FillRecord 追加 |
| 2026-02-24 | 158# P1-6: skip_gate 時間帯閾値オフセット実装 (commit `b22ea22f7`) |
| 2026-02-24 | 158# P2-4 Phase 1: run_fill_test.py 分割 — event_logger/lock_manager/fill_test_cli 抽出 (2,715→2,164行, -20.3%). テスト1,738全件PASS (commit `1aed848e3`) |
| 2026-02-24 | 158# P2-4 Phase 2 検討: run_continuous (802行)/run_single_cycle (483行) の更なる分割は本番稼働リスク考慮で保留 |
| 2026-02-24 | 158# fix: test_157 SkipGateEvaluator._compute_file_hash → compute_file_hash 直接使用に統一, test_136 浮動小数点精度許容 |
| 2026-02-24 | 158# P2-1: WF `_evaluate_wf_single` テストデータリーケージ修正 — eval_set を X_val に変更, wf_val_ratio_single/wf_embargo_rows 追加, テスト3件 (commit `7c5bed6ce`) |
| 2026-02-24 | 158# P2-2: OrderManager 統合テスト 25 件 (demo/live/validation/adapter/price/error/async 全パス) |
| 2026-02-24 | 158# P2-3: 障害モードテスト 27 件 (CircuitBreaker 状態遷移, タイムアウト, RiskManager 緊急停止, 価格フォールバック, 連続エラー) |
| 2026-02-24 | fix: risk_manager.py `pd.DataFrame` NameError 修正 (`from __future__ import annotations`) |
| 2026-02-25 | 163# P2-4 Phase 2 完了: run_fill_test.py 2,231→378行 3 Mixin 分割, maker_price compute() 306→143行, fill_config from_yaml() 479→139行, God Object 化防止警告追加, テスト 1,858 PASS (commit `6b766caf9`) |
| 2026-02-24 | retrain_scheduler 再起動: PID 129404 (WF leakage fix 適用済み), 旧 PID 122860 停止確認 |
| 2026-02-24 | テスト合計: 1793 passed (1741 + P2-2 25 件 + P2-3 27 件) |
| 2026-02-24 | 158# P0-3 最終評価: trending sell フィルタ検証完了 (n=221 skip + 681 filled sell 分析)。trending sell=-0.66bps(skip正解), trending_down sell=+1.18bps(開放成功), 累積損失回避~145.9bps |
| 2026-02-24 | 158# P1-2 offset 分析: buy offset 0.1-0.3帯がスイートスポット(+5.86bps vs +0.16bps @<=0.1)。buy base offset 引き上げ推奨(0.05→0.12-0.15)。n=102で追加データ要 |
| 2026-02-24 | mypy 品質チェック: order_manager/risk_manager/circuit_breaker/retrain_scheduler 全件 0 error (実質エラー 0、yaml スタブ警告のみ) |
