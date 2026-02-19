# 111# v456–v459 レガシー資産・教訓 調査レポート — v460 fill_test 活用視点

| key | value |
|-----|-------|
| type | rpt (調査レポート) |
| scope | cross-version (v456–v459 → v460 ph2) |
| date | 2026-02-19 |
| purpose | 過去バージョンの技術成果・教訓・再利用可能モジュールを体系的に棚卸しし、v460 fill_test への適用可能性を評価する |

---

## §0 Executive Summary

v456–v459 の 4 バージョン（合計 ~230 文書、~120 スクリプト）を調査した結果:

- **再利用価値の高い資産**: 12 モジュール（Exchange Adapter, CircuitBreaker, DrawdownController, PositionManager 等）
- **v460 で既に活用中**: 6 モジュール
- **即時活用可能だが未統合**: 4 モジュール
- **根本的教訓**: 5 件（手数料構造、報酬関数設計、Gate 順序、検証プロセス、スクリプト肥大化）

---

## §1 バージョン別 技術成果サマリ

### §1.1 v456 "MTF Integration" (2026-01-13~16, 4日間)

**目標**: マルチタイムフレーム特徴量 + 統合シグナルシステム
**最終判定**: インフラ成功・収益性未達 (ROI: -70.96% → -2.04%)

| 成果 | ファイル | 状態 | v460適用性 |
|------|---------|------|-----------|
| 88次元観測空間設計 | `ztb/trading/environment/fast_intraday_env_v456.py` | ALIVE (v458/v459で使用) | LOW — v460はマイクロストラクチャ特徴量に移行 |
| EnvironmentFactory | `ztb/trading/environment/factory_v456.py` (450L) | ALIVE | MEDIUM — ファクトリーパターンは継承可能 |
| 型安全環境 (95%+ Type Hints) | 環境コード全体 | ALIVE | HIGH — 型安全方針はv460で継続 |
| 訓練インフラ安定化 (50K完走) | `scripts/v456/train_v456_optimized.py` | ALIVE | LOW — v460はSACをph3で扱う |
| 環境Config一元化 | `ztb/config/environment_config.py` | ALIVE | MEDIUM — ConfigパターンはYAML化で発展 |
| ロギングスロットル (98.7% I/O削減) | 環境内 `last_log_step` | ALIVE | HIGH — fill_testのログI/Oにも適用すべき |
| Causal Rolling Mean (Look-ahead修正) | 環境内 | ALIVE | HIGH — 因果性保証は全バージョン共通基盤 |

**Key Files** (docs):
- [docs/v456/59_V456_FINAL_RETROSPECTIVE.md](../v456/59_V456_FINAL_RETROSPECTIVE.md) — 統括レトロスペクティブ
- [docs/v456/28_phase1_completion_summary.md](../v456/28_phase1_completion_summary.md) — Phase 1完了報告
- [docs/v456/24_final_execution_report.md](../v456/24_final_execution_report.md) — 実行報告

---

### §1.2 v457 "Diagnostic & Frequency Control" (2026-01-16~20, 5日間)

**目標**: 既存モデルのポテンシャル引出し + 学習失敗原因特定
**最終判定**: モデルにAlpha不足確定。再学習が必要 (v458へ)

| 成果 | ファイル | 状態 | v460適用性 |
|------|---------|------|-----------|
| Profit Factor / Expectancy 計算 | `scripts/analysis/analyze_backtest_v456.py` | ALIVE | MEDIUM — fill_testの収益性評価に流用可 |
| BacktestReporter (trades.json出力) | `scripts/v457/backtest_v457.py` | ALIVE | MEDIUM — 統一Reporterに統合推奨 |
| Frequency Control Wrapper | `backtest_v456.py` (cooldown/threshold) | ALIVE | HIGH — fill_testの取引頻度制御に直結 |
| Trend Guidance System (Ichimoku) | `fast_intraday_env_v456.py` 内 | ALIVE | LOW — v460はRule-basedシグナルに依存しない |
| Cyclical Time Features (sin/cos) | `fast_intraday_env_v456.py` 内 | ALIVE | MEDIUM — 時間帯効果はfill_testでも有用 |
| Causal MTF Features (lookahead修正) | `fast_intraday_env_v456.py` 内 | ALIVE | HIGH — lookahead bias防止は普遍的教訓 |
| Seed Stability Test Framework | `docs/v457/32_seed_stability_test.md` | 文書のみ | MEDIUM — 再現性検証フレームワーク |
| Legacy Asset Analysis | `docs/v457/01_legacy_asset_analysis.md` | 文書のみ | HIGH — v451 "Golden Era" の発見 |

**Critical Finding**: γ=0.80 (v451)、Hold Penalty=0、シンプルPnL報酬が最も有望。

**Key Files** (docs):
- [docs/v457/17_v457_enhancement_roadmap.md](../v457/17_v457_enhancement_roadmap.md) — Enhancement roadmap
- [docs/v457/21_v457_1_phase2_frequency_control.md](../v457/21_v457_1_phase2_frequency_control.md) — Frequency Control結果
- [docs/v457/37_v457_summary_and_v458_prep.md](../v457/37_v457_summary_and_v458_prep.md) — v457総括

---

### §1.3 v458 "Lost Alpha Integration & Stabilization" (2026-01-20~22, 3日間)

**目標**: Walk-Forward評価パイプライン確立 + Entry Gate統合
**最終判定**: ~80%完成。Critical bugs残存 (Gate未接続、コスト二重カウント等)

| 成果 | ファイル | 状態 | v460適用性 |
|------|---------|------|-----------|
| Walk-Forward Evaluation Pipeline | `ztb/evaluation/walk_forward/` | ALIVE | MEDIUM — ph3/ph4で使用可能 |
| Multi-Seed Training Framework | config-driven seed管理 | ALIVE | HIGH — 4-seed検証はv460のGate2で必須 |
| Baseline Comparison (BuyHold/SMA) | `ztb/analysis/baseline_comparison.py` | ALIVE | MEDIUM — ベースライン比較はGate3で有用 |
| IntegratedEntrySystem | `ztb/trading/signal/entry_system.py` | ALIVE (未接続) | LOW — v460はSkipGate MLに移行 |
| AB Testing Infrastructure | `tools/ab_test_runner.py` | ALIVE (非機能) | LOW — v460はmanifest.jsonlで管理 |

**Remaining Bugs (v458 Doc19/20で特定)**:
1. Entry Gate crash: `gate_result.allowed` → `gate_result["should_enter"]` 未修正
2. Entry Gate config未接続 → Gate無効化
3. Fee/Slippage二重カウント → PnL過小評価
4. Val/Test汚染 → 同一Reporter再利用
5. Trade種別誤分類 → "close"を"short"と判定
6. BacktestReporter 3重定義 → メトリクス不整合
7. CalibrationMap未ロード → 学習不能

**Key Files** (docs):
- [docs/v458/20_phase5_7_final_summary.md](../v458/20_phase5_7_final_summary.md) — 最終サマリ
- [docs/v458/13_phase5_4_review_and_remaining_gaps.md](../v458/13_phase5_4_review_and_remaining_gaps.md) — Gap分析
- [docs/v458/19_phase5_6_final_review.md](../v458/19_phase5_6_final_review.md) — Stop-Ship指摘

---

### §1.4 v459 "Alpha Resurrection" (2026-01-24~02-13, 21日間)

**目標**: SACベースBTC/JPY 1min戦略の収益化
**最終判定**: **No-Go確定** — 特徴量情報量不足 (K2実験で確定)

| 成果 | ファイル | 状態 | v460適用性 |
|------|---------|------|-----------|
| DrawdownController reset修正 | `ztb/trading/environment/components/position_manager.py` | ALIVE | HIGH — reset漏れはv460でも再発リスク |
| Oracle Test (理論上限検証) | `scripts/v459/run_phase_e1_counterfactual.py` | ARCHIVED | HIGH — 手数料構造検証のパターン |
| BUY/SELLカウント実測化 | `PositionManager.__init__` | ALIVE | HIGH — 計測精度はv460でも必須 |
| Gate C0-C3テストスイート | `tests/unit/trading/components/test_gate05_reward_purity.py` | ALIVE | MEDIUM — 報酬純度テストは再利用可 |
| Multi-horizon IC診断 | `scripts/v459/run_phase_e0_diagnostic.py` | ARCHIVED | HIGH — 方向予測力検証はG1-infoの基盤 |
| Holm-Bonferroni補正適用 | 設計のみ (000# §3.7) | 未実装 | HIGH — v460のG1判定に必須 |
| ベースライン比較 (Random/BuyHold/Momentum) | `scripts/v459/run_phase_c_subprocess.py` | ARCHIVED | HIGH — Gate判定の参照基準 |
| Counterfactual実験 (cost=0学習) | `run_phase_e1_counterfactual.py` | ARCHIVED | **CRITICAL** — maker 0%戦略の根拠 |

**Critical Findings**:
1. **COST_STRUCTURE_FATAL**: taker 0.1%では完全予測(Oracle)でも利益不可 (ROI=-18.25%)
2. **SAC_HAS_WEAK_EDGE**: cost=0環境で微弱な方向予測力あり (CF3: ROI=+2.82%)
3. **FEATURES_NO_INFO**: K2 (XGBoost) でOHLCV派生特徴量に情報量なし確定
4. **損益分岐手数料率**: ~0.02% → maker 0%なら成立可能

**Key Files** (docs):
- [docs/v459/116_phase_e0_diagnostic_report.md](../v459/116_phase_e0_diagnostic_report.md) — E0/E1診断
- [docs/v459/100_phase45_completion_report.md](../v459/100_phase45_completion_report.md) — Phase4.5完了
- [docs/v459/104_phase_c_comprehensive_report.md](../v459/104_phase_c_comprehensive_report.md) — Phase C統合実験
- [docs/v459/119_v460_launch_integrated_policy.md](../v459/119_v460_launch_integrated_policy.md) — v460始動方針
- [docs/v459/120_v460_doc0_doc1_cross_perspective_review.md](../v459/120_v460_doc0_doc1_cross_perspective_review.md) — v460設計レビュー

---

## §2 クロスバージョン教訓 (Lessons Learned)

### §2.1 手数料構造は戦略の前提条件 [CRITICAL]

| Version | 発見 | 影響 |
|---------|------|------|
| v456 | 手数料がPnLを圧倒 → HOLD 100%に収斂 | 訓練失敗 |
| v457 | Frequency Controlでも手数料負け解消不可 | モデル救済失敗 |
| v459 §100 | SAC ≈ Random (-15%) — 手数料支配 | Gate C3 No-Go |
| v459 §116 | **Oracle (完全予測) でもtaker 0.1%で負け** | maker 0%必須確定 |

**v460への教訓**: maker 0% 前提は正しい。fill_test で fill quality を実測し、maker 注文が実際に機能するか検証する現行アプローチは v459 の教訓を直接適用している。

### §2.2 報酬関数の過剰設計は致命的 [CRITICAL]

| Version | 問題 | 学習結果 |
|---------|------|---------|
| v456 | 9項目ペナルティがPnLを圧倒 | SELL 100% or BUY 100% |
| v457 | v451のシンプル報酬 (Hold Penalty=0) が最良 | 設計指針確立 |
| v459 §100 | hold_penalty_multiplier=0.0 → 98%の報酬がゼロ | HOLD学習しかできず |
| v459 §104 | position_change_penalty ハードコード → 設定無効化不能 | 意図しない行動抑制 |

**v460への教訓**: fill_test は SAC モデルを使わず rule-based シグナルで運用中。報酬関数問題は ph3 (SAC訓練) で再発リスクあり。v451 の γ=0.80 + Hold Penalty=0 + PnL直結を基本設計とすべき。

### §2.3 Gate順序の是正 [HIGH]

| Version | 問題 |
|---------|------|
| v459 | 特徴量検証(K2)をPhase Eで実行 → Phase B-Dの実験が全て無駄に |
| v460 | G1-info を早期化、G1.1-exec を挿入 → 正しい順序 |

**v460での適用**: G1-info PASS済。G1.1-exec (fill_test) が現在進行中。順序は正しい。

### §2.4 「良すぎる結果」は必ず疑え [HIGH]

| Version | 事例 |
|---------|------|
| v456 | バックテスト勝率63.2%/シャープ14.72 → 実際は0.3%勝率 (コードバグ) |
| v458 | ROI=-0.098 なのに is_robust=true → 判定ロジックと結果の矛盾 |
| v459 | 単一seed +41.95% → 別seedで+3.93%に崩壊 |

**v460での適用**: fill_test は複数 Codex レビュー (043#, 067#, 082#, 090#, 095#) を通じて検証強化。しかし依然として楽観バイアスに注意。

### §2.5 スクリプト肥大化と God Object [MEDIUM]

| Version | 問題 |
|---------|------|
| v459 | 45スクリプト / 12,458行 / run_phase_c.py が 1,277行のGod Object |
| v459 | 実験定義がPython dictとしてハードコード → config形骸化 |

**v460での適用**: 119# 方針に基づき YAML 駆動 + 単一ランナー (run_experiment.py 200行以内) を採用。fill_test は `run_fill_test.py` + `lib/` 分割で対応済み (042# で God Object 分割)。

---

## §3 再利用可能モジュール詳細

### §3.1 Exchange Adapter (v460 ACTIVE)

| 項目 | 内容 |
|------|------|
| パス | `ztb/trading/live/exchanges/coincheck/adapter.py` (853L) |
| 機能 | Coincheck REST API + dry-run / real trading 切替、IBroker準拠 |
| 状態 | **v460 fill_test で現在使用中** |
| 備考 | 013# で Signature修正、async統一、post_only対応、rate limit修正済 |

### §3.2 CircuitBreaker (v460 AVAILABLE)

| 項目 | 内容 |
|------|------|
| パス | `ztb/utils/circuit_breaker.py` (229L) |
| ラッパー | `ztb/risk/circuit_breakers.py` (189L) — KillSwitch含む |
| 互換 | `archived/circuit_breaker.py` — compatibility shim |
| 機能 | CLOSED/OPEN/HALF_OPEN 3状態、failure_threshold / recovery_timeout / success_threshold |
| 状態 | **ALIVE — テスト参照あり** |
| v460適用 | **HIGH — fill_test のAPI障害時の自動遮断に有用。未統合** |

### §3.3 DrawdownController (v460 AVAILABLE)

| 項目 | 内容 |
|------|------|
| パス | `ztb/risk/drawdown_controller.py` (255L) |
| 機能 | 段階的DD制御: warning(5%) → reduction(8%) → emergency_stop(15%) |
| 状態 | **ALIVE** |
| v460適用 | **HIGH — fill_test のmax_daily_loss制御に適用可能。現在はfill_test.yamlの`max_daily_loss_jpy`で制御** |
| 注意 | v459 §104で発見: `is_emergency_stop` ラッチがreset()で解除されないバグ → 修正済 (commit 21ec3b82) |

### §3.4 DynamicPositionSizer (v460 PARTIAL)

| 項目 | 内容 |
|------|------|
| パス | `ztb/risk/dynamic_position_sizer.py` (260L) |
| 機能 | ボラティリティ・DD・市場レジームに基づく動的ポジションサイジング |
| 状態 | **ALIVE** |
| v460適用 | **MEDIUM — fill_test の方策B (lot_sizer.py) が類似機能を提供中。統合検討の余地** |

### §3.5 PositionManager (v460 ACTIVE)

| 項目 | 内容 |
|------|------|
| パス | `ztb/trading/position_manager.py` (592L) |
| 機能 | マルチ取引所対応、最小取引単位管理、BUY/SELLカウント実測 |
| 状態 | **ALIVE — v460 fill_test系列で使用** |
| 注意 | v459 §104: `reset()` → `risk_manager.reset()` カスケード修正済 |

### §3.6 HealthMonitor (v460 AVAILABLE)

| 項目 | 内容 |
|------|------|
| パス | `ztb/trading/live/core/health_monitor.py` (119L) |
| 機能 | CPU/メモリ/ディスク/プロセス情報の定期取得 |
| 状態 | **ALIVE** |
| v460適用 | **MEDIUM — fill_test の長時間稼働 (168h) でのリソース監視に有用** |

### §3.7 Reconciliation (v460 AVAILABLE)

| 項目 | 内容 |
|------|------|
| パス | `ztb/trading/live/core/reconciliation.py` (602L) |
| 機能 | ポジション・残高・注文の取引所照合 |
| 状態 | **ALIVE** |
| v460適用 | **HIGH — fill_test でのポジション整合性検証に有用** |

### §3.8 Production Modules (v460 AVAILABLE)

| モジュール | パス | 機能 | v460適用 |
|-----------|------|------|----------|
| EmergencyStop | `ztb/trading/production/emergency_stop.py` | 緊急停止 | MEDIUM |
| HealthChecker | `ztb/trading/production/health_checker.py` | ヘルスチェック | LOW (HealthMonitorで代替) |
| RecoverySystem | `ztb/trading/production/recovery_system.py` | 自動復旧 | MEDIUM |
| RollbackManager | `ztb/trading/production/rollback_manager.py` | ロールバック | LOW |
| StatePersistence | `ztb/trading/production/state_persistence.py` | 状態永続化 | HIGH — fill_test再起動時の状態復元 |
| RiskBasedAllocator | `ztb/trading/production/risk_based_allocator.py` | リスクベース配分 | LOW |
| PaperTradingManager | `ztb/trading/production/paper_trading_manager.py` | ペーパートレード | MEDIUM — G4-live |

### §3.9 v460 固有モジュール (fill_test用, 現在 ACTIVE)

| モジュール | パス | 機能 |
|-----------|------|------|
| ParamAdapter (方策A) | `scripts/v460/lib/param_adapter.py` | spread_offset自動調整 |
| LotSizer (方策B) | `scripts/v460/lib/lot_sizer.py` | 動的ロットサイジング |
| RegimeDetector | `scripts/v460/lib/regime_detector.py` | 軽量レジーム検知 |
| FastFillDefense | `scripts/v460/lib/fast_fill_defense.py` | 即約定防御 |
| FillMetrics | `ztb/metrics/fill_quality.py` | Fill品質メトリクス + SkipGate統合 |
| ConfigLoader | `scripts/v460/lib/config_loader.py` | YAML設定読込 |
| Manifest | `scripts/v460/lib/manifest.py` | Run manifest記録 |

---

## §4 Dead Code / 未統合モジュール

### §4.1 完全Dead (参照ゼロ、v460で不使用)

| モジュール | パス | 理由 |
|-----------|------|------|
| OnlineLearningPipeline | `ztb/adaptation/online_learning/` | circular import で disabled |
| RetrainingTrigger | `ztb/adaptation/retraining/` | 型定義のみ |
| ConceptDriftDetector | `ztb/adaptation/concept_drift/` | 設定のみ |
| AB Testing (adaptation) | `ztb/adaptation/ab_testing/` | 参照ゼロ |
| Safety (adaptation) | `ztb/adaptation/safety/` | 参照ゼロ |
| Operations (adaptation) | `ztb/adaptation/operations/` | 参照ゼロ |
| OnlineLearningEngine (V433) | `ztb/training/online_learning_engine.py` (808L) | `enable_v433_adaptive=False` デフォルト |
| V433 Adaptive Training | `unified_trainer.py` 内 | 246行削除済 (063# SAC cleanup) |

### §4.2 部分Dead (一部参照あるが機能不全)

| モジュール | パス | 状態 |
|-----------|------|------|
| IntegratedEntrySystem | `ztb/trading/signal/entry_system.py` | CalibrationMap未ロード、gate_result属性名不一致 |
| AB Test Runner | `tools/ab_test_runner.py` | 2結果以上が必要だが単一結果で常にno-op |
| Walk-Forward Adapter | `ztb/evaluation/walk_forward/` | types.py / result.py 二重定義 |

---

## §5 既知の未解決バグ

### §5.1 v458由来 (v460にも影響しうるもの)

| # | バグ | ファイル | 深刻度 | v460影響 |
|---|------|---------|--------|---------|
| 1 | Entry Gate crash (属性名不一致) | `fast_intraday_env_v456.py:500-506` | P0 | LOW — v460 fill_test はEntry Gateを使わない |
| 2 | Fee/Slippage二重カウント | `evaluator.py:528` / `fast_intraday_env_v456.py:579` | P0 | MEDIUM — ph3でWalk-Forward使用時に再発 |
| 3 | Val/Test Reporter汚染 | `evaluator.py:371-399` | P0 | MEDIUM — ph3以降 |
| 4 | Trade種別誤分類 | `evaluator.py:521` / `reporter.py:230` | P1 | MEDIUM — ph3以降 |
| 5 | Entry Price on Reversals未更新 | `fast_intraday_env_v456.py:579` | P1 | LOW — fill_testは環境非使用 |
| 6 | BacktestReporter 3重定義 | 複数箇所 | P1 | MEDIUM — 統一前にph3移行不可 |

### §5.2 v459由来

| # | バグ | 修正状況 | v460影響 |
|---|------|---------|---------|
| 1 | DrawdownController ラッチ | ✅ 修正済 (commit 21ec3b82) | 影響なし |
| 2 | BUY/SELLカウント推定値 | ✅ 修正済 | 影響なし |
| 3 | hold_penalty_multiplier=0.0 | ✅ 設計レベルで排除 | ph3で注意 |
| 4 | Seed非決定性 (cuDNN等) | ⚠️ 未解決 | MEDIUM — ph3の再現性に影響 |
| 5 | チェックポイント共有/上書き | ⚠️ 未解決 | MEDIUM — ph3の評価信頼性 |

### §5.3 v460 fill_test 固有 (086#, 110# で発見)

| # | バグ | 修正状況 |
|---|------|---------|
| 1 | time_filter 片側蓄積→デッドロック (49%アイドル) | ✅ 110# で修正 |
| 2 | sell_offset_balance 不整合 | ✅ 105# で修正 |
| 3 | cancel race条件 + Gate不整合 | ✅ 047# で修正 |
| 4 | JSONL データロス | ✅ 022# / 042# で修正 |
| 5 | ゾンビプロセス | ✅ 042# で修正 |

---

## §6 v460 fill_test への具体的活用提案

### §6.1 即時適用推奨 (Priority: HIGH)

| # | 提案 | 根拠 | 実装量 |
|---|------|------|--------|
| 1 | CircuitBreaker を fill_test のAPI通信に適用 | API障害時の自動遮断。現在はtry/exceptのみ | ~50行 (統合コード) |
| 2 | Reconciliation を定期実行 (1h毎) | fill_test の168h長時間稼働でのポジション/残高ドリフト検出 | ~30行 (呼出し追加) |
| 3 | StatePersistence で fill_test 状態を永続化 | 再起動時の状態復元 (既に079#で棚卸し済) | ~100行 |
| 4 | HealthMonitor を fill_test に統合 | OOM/ディスク枯渇の早期検出 (018#のメモリリーク防止と連動) | ~20行 |

### §6.2 中期的検討 (Priority: MEDIUM)

| # | 提案 | 根拠 |
|---|------|------|
| 1 | DrawdownController → fill_test の max_daily_loss 統合 | 現在はYAML値との単純比較。段階的制御 (warning/reduction/stop) が有用 |
| 2 | DynamicPositionSizer → lot_sizer.py 統合 | ボラティリティ適応機能の共通化 |
| 3 | Walk-Forward Pipeline の v458 バグ修正 | ph3移行前の必須作業 |
| 4 | BacktestReporter 統一 | ph3移行前の必須作業 |

### §6.3 Phase 3+ で必須 (Priority: DEFERRED)

| # | 提案 | 根拠 |
|---|------|------|
| 1 | 報酬関数: PnL直結 + Hold Penalty=0 + γ=0.80 | v451/v457/v459の教訓集約 |
| 2 | Holm-Bonferroni補正をG1判定に実装 | v459 §116 / 120# §1 の指摘 |
| 3 | Oracle Test をph3の早期段階で実施 | v459 §116の教訓 (完全予測でもコスト負け→早期中止) |
| 4 | 4-seed検証の厳格化 (worst-seed下限) | v459 §100, 120# §1 の指摘 |
| 5 | 方策B (定期バッチ再訓練) の設計着手 | 028# §3.2 の方策 |

---

## §7 バージョン間の技術的進化マップ

```
v456 (4日)          v457 (5日)           v458 (3日)           v459 (21日)          v460 (進行中)
───────────         ───────────          ───────────          ───────────          ───────────
88次元観測空間  ──→ 収益性検証        ──→ Walk-Forward     ──→ HP探索大規模    ──→ Microstructure
EnvironmentFactory   Frequency Control    Entry Gate (未完)    Oracle Test          fill_test
型安全化             v451 Legacy発掘      Multi-seed           K2特徴量検証         SkipGate ML
訓練安定化           報酬簡素化方針       Baseline比較         Gate体系確立         maker 0%実測
                     Lost Alpha修正       AB Test基盤          No-Go確定            パラメータ適応

  インフラ確立         診断・方針確立       評価基盤構築         根本原因特定          実測検証
  (報酬設計失敗)       (Alpha不足確定)      (バグ多数残存)       (手数料構造問題確定)   (fill quality)
```

---

## §8 結論

### 最大の教訓 (v456-v459 統合)

> **「手数料構造 × 取引頻度」が収益性の根本制約。いかなるモデル改善も、この制約を超えない限り無意味。** v460 の maker 0% 前提 + fill quality 実測 (G1.1-exec) は、この教訓を正面から解決するアプローチであり、方向性は正しい。

### 最大のリスク

1. **fill_test が PASS しても ph3 (SAC訓練) で v456-v459 の失敗パターンを繰り返すリスク** — 報酬関数設計、seed非決定性、HP探索の肥大化
2. **v458 の Walk-Forward バグが ph3 で再発するリスク** — 修正されていない 6 件のP0/P1バグ
3. **Dead code の存在によるメンテナンスコスト** — 6+ dead modules in `ztb/adaptation/`

### 推奨アクション (優先順)

1. **即時**: CircuitBreaker / HealthMonitor / Reconciliation を fill_test に統合
2. **ph2完了前**: v458 Walk-Forward バグ 6件の修正
3. **ph3開始時**: 報酬関数をv451ベースで設計、Oracle Testを早期実施
4. **継続**: Dead code (adaptation/) の整理・削除
---

## §9 追加調査 — 見落とし資産 (2026-02-19 補完)

初回調査 (§3) で漏れていた再利用可能モジュールの追加棚卸し。`ztb/risk/`, `ztb/ops/`, `ztb/cache/`, `ztb/data/` を横断的に再スキャン。

### §9.1 fill_test 即時統合候補 (HIGH)

| # | モジュール | パス | 行数 | 機能 | 統合目的 |
|---|-----------|------|------|------|---------|
| 1 | **PnL Monte Carlo** | `ztb/risk/pnl_monte_carlo.py` | 412L | fill_records から経験分布を構築し n=10,000 月次PnLシミュレーション。G1.1判定指標も出力 | **014# T5 の直接実装** — fill_test 実測データから収益予測 |
| 2 | **AdvancedAutoStop** | `ztb/risk/advanced_auto_stop.py` | 445L | ボラティリティ急騰・DD限度・連続損失・時間制限による多重自動停止。`StopReason` enum + cooldown管理 | 168h長時間稼働の安全装置 |
| 3 | **RiskRuleEngine** | `ztb/risk/checks.py` (195L) + `ztb/risk/rules.py` (332L) | 527L | Pre/Post-trade リスクバリデーション。ハードストップ・クールダウン・最大取引頻度・日次損失限度 | fill_test 注文発行前のプリトレードチェック |
| 4 | **RiskProfiles** | `ztb/risk/profiles.py` | 139L | CONSERVATIVE / BALANCED / AGGRESSIVE 3プリセット（ポジション上限、日次損失限度、取引頻度、SL/TP一括定義） | fill_test はCONSERVATIVEプロファイルで運用すべき |
| 5 | **DataValidation** | `ztb/data/data_validation.py` | 934L | 金融時系列の包括的バリデーション（型・範囲・一貫性、スキーマ検証、品質メトリクス、異常検知、分布変化検出） | リアルタイムデータ品質保証 |
| 6 | **watch_1m** | `ztb/ops/monitoring/watch_1m.py` | 364L | 長時間稼働ウォッチャー：ステップ停滞>10分・RSS>2GB・エラー率>2%・報酬急落でアラート。kill-file対応。JSONL出力 | 168h fill_test の生存監視 |
| 7 | **DiscordNotifier** | `ztb/ops/alerts/notifications.py` | 191L | Discord webhook通知クラス | fill_test アラート配信基盤 |
| 8 | **GatesToAlerts** | `ztb/ops/alerts/gates_to_alerts.py` | 158L | gates.json 読込 → Gate失敗時にwebhookアラート送信 | Gate G0-G1 判定結果の自動通知 |
| 9 | **MemoryCache** | `ztb/cache/memory_cache.py` | 444L | TTLCache + 動的バッファサイズ調整 + メモリ使用量監視・最適化 + LRU | 168h稼働のメモリリーク防止 |

> **特筆**: `pnl_monte_carlo.py` は 014# T5 の直接実装であり、§3 で見落とされていた重大な漏れ。

### §9.2 中期活用候補 (MEDIUM)

| # | モジュール | パス | 行数 | 機能 | 適用フェーズ |
|---|-----------|------|------|------|------------|
| 1 | StatisticalValidator | `ztb/metrics/statistical_validator.py` | 492L | 多重検定補正(Holm-Bonferroni)、信頼区間、統計的有意性検定 | ph2-ph3: fill_test結果の統計検証 |
| 2 | CalibrationMap | `ztb/trading/signal/calibration_map.py` | 340L | 階層的フォールバック (Specific→Regime→Global) + EWMA WinRate追跡 | ph3: パフォーマンス追跡 |
| 3 | QualityScorer | `ztb/trading/signal/quality_scorer.py` | 764L | テクニカル指標ベース決定論的シグナルスコアリング (0-100) | ph3: Rule-based信号品質フィルタ |
| 4 | MarketAdaptationManager | `ztb/risk/market_adaptation_manager.py` | 333L | レジーム動的検知・適応、遷移追跡・安定性評価 | ph2: レジーム条件付き行動分析 |
| 5 | KellyPositionSizer | `ztb/analysis/kelly_position_sizer.py` | 496L | Kelly基準による動的ポジションサイズ最適化 | ph3: 実績データからKelly最適ロット |
| 6 | CheckVenueHealth | `ztb/ops/health/check_venue_health.py` | 273L | Coincheck REST/WebSocket接続・レイテンシ・レート制限状態チェック | fill_test起動前プリフライトチェック |
| 7 | SystemHealth | `ztb/ops/health/system_health.py` | 512L | CPU/メモリ/ディスク/依存関係の包括ヘルスチェック | fill_test起動前プリフライトチェック |
| 8 | DiskHealth | `ztb/ops/monitoring/disk_health.py` | 197L | ディスク容量・inode使用率・I/Oレイテンシ監視 | 168h稼働中のディスク枯渇防止 |
| 9 | StreamingPipeline | `ztb/data/streaming_pipeline.py` | 714L | CoinGecko→特徴量生成ストリーミングパイプライン (ThreadPoolExecutor) | ph3: リアルタイムデータ取得基盤 |
| 10 | StreamBuffer | `ztb/data/stream_buffer.py` | 430L | チャンク式循環バッファ(DataFrame)、圧縮・ゼロコピースライス | 高頻度データバッファリング |
| 11 | AnomalyDetection | `ztb/data/anomaly_detection.py` | 638L | データ品質管理・異常値検知 (Autoencoder対応) | 異常市場データフィルタリング |
| 12 | IntegrityChecker | `ztb/data/integrity_checker.py` | 299L | Parquetデータ欠損・重複検査・自動修復 + Discord通知 | データパイプライン品質保証 |
| 13 | TimeSeriesCV | `ztb/analysis/cv.py` | 179L | 時系列交差検証 + FDR多重検定補正 | ph3: SAC再学習時の検証 |
| 14 | CheckDataLeakage | `scripts/v459/check_data_leakage.py` | 373L | OnlineScaler fit範囲、MTF因果性、look-ahead bias検査 | ph3: 汎用リーク検査ツール |
| 15 | K2 NonRL Upper Bound | `scripts/v459/run_k2_nonrl_upper_bound.py` | 372L | XGBoost/Logistic で特徴量方向予測力検証 (IC≈0 診断) | ph3: 新特徴量の upper-bound テスト |

### §9.3 統合優先度マトリクス (§3 + §9 統合)

```
fill_test 即時統合 (§3 + §9.1 統合, 全13モジュール)
──────────────────────────────────────────────────────
Priority A (安全基盤): CircuitBreaker + AdvancedAutoStop + RiskRuleEngine + RiskProfiles
Priority B (監視通報): HealthMonitor + watch_1m + DiscordNotifier + GatesToAlerts
Priority C (データ品質): Reconciliation + DataValidation + MemoryCache
Priority D (状態管理): StatePersistence
Priority E (収益分析): PnL Monte Carlo (014# T5)
```

### §9.4 見落とし原因の分析

| 見落としカテゴリ | 件数 | 原因 |
|----------------|------|------|
| `ztb/risk/` 下層 | 4 | §3 は CircuitBreaker / DrawdownController / DynamicPositionSizer のみスキャン。checks, rules, profiles, advanced_auto_stop を見落とし |
| `ztb/ops/` 全体 | 4 | ops/ ディレクトリが §3 の調査対象外 |
| `ztb/cache/` | 1 | cache/ ディレクトリが §3 の調査対象外 |
| `ztb/data/` バリデーション | 1 | data/ 下にバリデーションモジュールが存在すること自体を見落とし |

---

## §10 改訂版 推奨アクション (§8 + §9 統合)

### 即時 (ph2 進行中)

| # | アクション | 根拠 | 実装見込み |
|---|-----------|------|-----------|
| 1 | **Priority A 安全基盤4点** を fill_test に統合 | API障害自動遮断 + 多重停止 + プリトレードチェック + プロファイル適用 | ~150行の統合コード |
| 2 | **Priority B 監視通報4点** を fill_test に統合 | 168h長時間稼働の無人監視。DiscordでGate結果・障害を即時通知 | ~100行の統合コード |
| 3 | **Priority C データ品質3点** を統合 | ポジションドリフト検出 + 入力データ品質保証 + メモリリーク防止 | ~80行の統合コード |
| 4 | **StatePersistence** で再起動時状態復元 | 079# で棚卸し済。HWM, offset, adaptation state の永続化 | ~100行 |
| 5 | **PnL Monte Carlo** を定期実行 (1日1回) | 014# T5 → fill_test 実測データから G1.1 月次収益予測を更新 | ~30行の呼出し追加 |

### ph2 完了前

| # | アクション | 根拠 |
|---|-----------|------|
| 6 | v458 Walk-Forward バグ 6件修正 | ph3 移行前の必須作業 |
| 7 | BacktestReporter 統一 | 3重定義 → 単一定義に統合 |
| 8 | CheckVenueHealth を fill_test 起動時プリフライトに追加 | 取引所接続異常の早期検出 |

### ph3 開始時

| # | アクション | 根拠 |
|---|-----------|------|
| 9 | 報酬関数をv451ベースで設計 (γ=0.80, Hold Penalty=0, PnL直結) | v451/v457/v459 の教訓集約 |
| 10 | Oracle Test を早期実施 | 完全予測でもコスト負け → 早期中止判断に必須 |
| 11 | Holm-Bonferroni 補正を G1 判定に実装 | フォールスポジティブ防止 |
| 12 | check_data_leakage + K2 upper-bound テスト | 新特徴量の情報量検証 |

### 継続

| # | アクション | 根拠 |
|---|-----------|------|
| 13 | Dead code (`ztb/adaptation/` 6+ モジュール) 整理・削除 | メンテナンスコスト削減 |
| 14 | StatisticalValidator を Gate 判定パイプラインに正式統合 | 統計的厳密性の確保 |