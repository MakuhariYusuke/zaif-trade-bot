# v460 ドキュメント索引

> **v460 "Microstructure Edge"** — Coincheck BTC/JPY maker 執行品質検証  
> 最終更新: 2026-03-23 (555# CalibrationMap runtime integration)

---

## 運用方針（VS Code同期制限の回避）

- 一部の巨大ファイルは VS Code 拡張の同期制限（実質 50MB 前後）により直接読取できない前提で運用する。
- 調査レビューは Get-Content -TotalCount / Select-String -Context / Get-Content -Skip -First を使った部分抽出を標準手順とする。
- 集計検証は run_id / git_sha / date_from,to を固定して再現可能性を担保し、ワイルドカード横断集計のみで最終判定しない。
- 長大な成果物は章分割または日次分割し、単一 Markdown/JSONL の肥大化を避ける。
- 一時抽出物は temp/、恒久成果物は docs/v460/ と reports/ に分離して管理する。
---

## フェーズ定義

| フェーズ | Gate | 目的 | 状態 |
|---|---|---|---|
| **ph0** | — | プロジェクト提案・技術仕様策定 | ✅ 完了 |
| **ph1** | G1-info | 情報ゲート: データ品質・特徴量検証 | ✅ PASS |
| **ph2** | G1.1-exec | 執行ゲート: fill quality 実測検証 | 🔄 **進行中** |
| **ph3** | G2 | コード整理・SAC 4-seed 訓練 | 🔄 **進行中** (G2 PASS 達成) |
| **ph4** | G3 | オンライン学習設計・実装 | ⏳ 待機 (調査済) |
| **ph5** | — | 本番デプロイ | ⏳ 未着手 |
| **phg** | — | フェーズ横断: 型安全・品質改善・分析 | 🔄 継続中 |

---

## 全ドキュメント一覧

### ph0 — プロジェクト計画

| # | ファイル | タイトル |
|---|---|---|
| 000 | [000_ph0_plan_project_proposal.md](000_ph0_plan_project_proposal.md) | v460 Project Proposal: "Microstructure Edge" |
| 001 | [001_ph0_plan_technical_specs.md](001_ph0_plan_technical_specs.md) | 技術仕様: データ契約・アーキテクチャ・特徴量候補・実験基盤 |

### ph1 — G1-info ゲート

| # | ファイル | タイトル |
|---|---|---|
| 005 | [005_ph1_rpt_g1_info.md](005_ph1_rpt_g1_info.md) | G1-info 検証報告 |
| 064 | [064_ph1_g1_info_verify.md](064_ph1_g1_info_verify.md) | G1-info 再検証結果 |
| 065a | [065_ph1_rev_064.md](065_ph1_rev_064.md) | 064 レビュー: ph2 再 fill test 投入モデル選定 |
| 065b | [065_ph1_impl_as_lr_prep.md](065_ph1_impl_as_lr_prep.md) | 対応: ph2 モデル再評価 + AS-LR SkipGate 学習 |
| 070 | [070_ph1_rpt_model_search.md](070_ph1_rpt_model_search.md) | 徹底モデルサーチ (72+ classifiers, 20 regressors, 12 rules) |

### ph2 — G1.1-exec ゲート (現在のメインフェーズ)

| # | 種別 | ファイル | タイトル |
|---|---|---|---|
| 009 | plan | [009_ph2_plan_g1_1_exec.md](009_ph2_plan_g1_1_exec.md) | Phase 2 計画: G1.1-exec |
| 010 | rpt | [010_ph2_rpt_profitability_assessment.md](010_ph2_rpt_profitability_assessment.md) | 収益性批判的評価 |
| 013 | rpt | [013_ph2_rpt_exchange_api_audit.md](013_ph2_rpt_exchange_api_audit.md) | Exchange API 実装調査 |
| 014 | plan | [014_ph2_plan_completion_and_transition.md](014_ph2_plan_completion_and_transition.md) | ph2 完遂計画 + ph3 移行条件 |
| 019 | rpt | [019_ph2_rpt_fill_test_analysis.md](019_ph2_rpt_fill_test_analysis.md) | Fill Test 分析・対策実装 |
| 022 | ext | [022_ph2_ext_fill_test_data_loss.md](022_ph2_ext_fill_test_data_loss.md) | Fill Test JSONL データロス調査依頼 |
| 023 | plan | [023_ph2_plan_parallel_tasks.md](023_ph2_plan_parallel_tasks.md) | G1.1 待機中の並行タスク実行計画 |
| 040 | rpt | [040_ph2_rpt_regime_integration_fill_test.md](040_ph2_rpt_regime_integration_fill_test.md) | レジーム検知統合と fill test 中間報告 |
| 041 | rpt | [041_ph2_rpt_profitability_improvements.md](041_ph2_rpt_profitability_improvements.md) | 高収益改善施策 — 動的 loss_cap + AS 最適化 5 施策 |
| 043 | ext | [043_ph2_codex_review_package.md](043_ph2_codex_review_package.md) | Codex レビュー用情報パッケージ |
| 046 | fix | [046_ph2_remaining_fixes_and_log_analysis.md](046_ph2_remaining_fixes_and_log_analysis.md) | 残タスク完了・ログ分析・再起動準備 |
| 047 | fix | [047_ph2_fix_cancel_race_and_gate_alignment.md](047_ph2_fix_cancel_race_and_gate_alignment.md) | Cancel Race 修正・Gate 整合・ログ最適化 |
| 048 | rpt | [048_ph2_exit_timing_analysis_and_e3_data_collection.md](048_ph2_exit_timing_analysis_and_e3_data_collection.md) | Exit Timing 分析と E3 Multi-Timeframe データ収集 |
| 050 | impl | [050_ph2_impl_log_analysis.md](050_ph2_impl_log_analysis.md) | 実装完了報告 + ログ分析 |
| 051 | impl | [051_ph2_proactive_impl.md](051_ph2_proactive_impl.md) | Phase 2 残課題の先行実装 |
| 052 | impl | [052_ph2_data_driven_optimization.md](052_ph2_data_driven_optimization.md) | Fill Test 改善 — データ駆動パラメータ最適化 |
| 053 | rpt | [053_ph2_rpt_g11_interim_and_cleanup.md](053_ph2_rpt_g11_interim_and_cleanup.md) | G1.1 暫定判定・Monte Carlo PnL・リポジトリ整理 |
| 054 | plan | [054_ph2_plan_profitability_improvement.md](054_ph2_plan_profitability_improvement.md) | 収益性改善計画: AS 低減 + 最適エントリー・エグジット |
| 057 | impl | [057_ph2_ml_baseline.md](057_ph2_ml_baseline.md) | ML-1/ML-2 ベースライン分類器 |
| 058 | impl | [058_ph2_ml_enrichment_skip_gate.md](058_ph2_ml_enrichment_skip_gate.md) | ML 強化: マイクロストラクチャ特徴量 + PnL 回帰 Skip Gate |
| 059 | impl | [059_impl_summary.md](059_impl_summary.md) | 058# レビュー対応 + 追加見落とし修正 |
| 060 | impl | [060_ph2_ml_improvement.md](060_ph2_ml_improvement.md) | ML パイプライン改善: バグ修正 + 特徴量 v2 + チューニング |
| 061 | rpt | [061_ph2_rpt_ml_improvement.md](061_ph2_rpt_ml_improvement.md) | ML パイプライン改善: バグ修正 + 特徴量 v2 + チューニング |
| 062 | impl | [062_ph2_skip_gate_fill_test_integration.md](062_ph2_skip_gate_fill_test_integration.md) | AS SkipGate → fill_test ライブ統合 |
| 066 | rpt | [066_ph2_rpt_trade_only_two_tier.md](066_ph2_rpt_trade_only_two_tier.md) | Trade-Only 比較検証 + Two-Tier SkipGate |
| 067 | ext | [067_ph2_codex_review_package.md](067_ph2_codex_review_package.md) | Codex レビュー: AS-LR SkipGate + 次の一手 |
| 068 | rev | [068_ph2_rev_067.md](068_ph2_rev_067.md) | レビュー: AS-LR SkipGate と板情報なしモデル投入方針 |
| 069 | resp | [069_ph2_resp_068.md](069_ph2_resp_068.md) | レスポンス: SkipGate 実装整合性修正 |
| 071 | impl | [071_ph2_impl_ob_removal.md](071_ph2_impl_ob_removal.md) | 板情報 (OB) 除去 — 価格ベース回帰 |
| 072 | impl | [072_ph2_impl_ob_toggle.md](072_ph2_impl_ob_toggle.md) | OB 特徴量トグル実装 |
| 073 | rpt | [073_ph2_rpt_strategy_analysis.md](073_ph2_rpt_strategy_analysis.md) | 戦略分析 & パラメータチューニング |
| 074 | rev | [074_ph2_rev_073.md](074_ph2_rev_073.md) | レビュー: 073 戦略分析の再点検と vXXX 資産の総動員計画 |
| 075 | impl | [075_ph2_impl_review_response.md](075_ph2_impl_review_response.md) | レビュー指摘対応 + 50K ステップ検証 + 批判的再考察 |
| 076 | rev | [076_ph2_rev_075.md](076_ph2_rev_075.md) | レビュー: 075 結果検証と v458 以前資産の再利用提案 |
| 077 | impl | [077_ph2_impl_076_review_response.md](077_ph2_impl_076_review_response.md) | 076# レビュー指摘対応 |
| 078 | impl | [078_ph2_impl_deeper_verification.md](078_ph2_impl_deeper_verification.md) | 検証深化: Permutation Test / 時系列安定性 / 検出力分析 |
| 079 | rpt | [079_ph2_rpt_inventory_and_restart.md](079_ph2_rpt_inventory_and_restart.md) | 情報棚卸し + fill_test 再開 + ph3 先行作業整理 |
| 082 | ext | [082_ph2_fill_test_deep_dive_for_codex.md](082_ph2_fill_test_deep_dive_for_codex.md) | Fill Test データ深掘り (Codex レビュー用) |
| 083 | rev | [083_ph2_rev_082.md](083_ph2_rev_082.md) | レビュー: 082 Fill Test 深掘りの再点検 |
| 084 | impl | [084_ph2_impl_083_review_response.md](084_ph2_impl_083_review_response.md) | 083# レビュー指摘対応 + 盲点 8 項目特定 |
| 085 | impl | [085_ph2_084_blind_spot_impl.md](085_ph2_084_blind_spot_impl.md) | 084 盲点指摘の実装 |
| 086 | rpt | [086_ph2_rpt_time_filter_bug_and_session_analysis.md](086_ph2_rpt_time_filter_bug_and_session_analysis.md) | time_filter 片側蓄積バグ修正 + 085# セッション考察 |
| 087 | rev | [087_ph2_rev_086.md](087_ph2_rev_086.md) | レビュー: 086# 外部 Codex 分析 — 構造的損失原因の特定 |
| 088 | impl | [088_ph2_impl_087_review_response.md](088_ph2_impl_087_review_response.md) | 087# レビュー対応: SkipGate 動的較正 + sell ガード + データ品質修正 |
| 090 | ext | [090_ph2_deep_dive_v2_for_codex.md](090_ph2_deep_dive_v2_for_codex.md) | fill_test 深掘り分析 v2 — Codex レビュー用資料 |
| 091 | rev | [091_ph2_rev_090.md](091_ph2_rev_090.md) | 090 深掘り分析 v2 の整合レビューと修正提案 |
| 092 | impl | [092_ph2_impl_gap_analysis.md](092_ph2_impl_gap_analysis.md) | 対応漏れ点検と先行実装 |
| 093 | impl | [093_ph2_spread_adaptive_fast_fill.md](093_ph2_spread_adaptive_fast_fill.md) | spread_adaptive / fast_fill_defense サイド別パラメータ追加 |
| 094 | impl | [094_ph2_stale_order_cancel_replace.md](094_ph2_stale_order_cancel_replace.md) | stale order 検出 & cancel-replace |
| 095 | ext | [095_ph2_codex_review_v3.md](095_ph2_codex_review_v3.md) | fill_test Codex レビュー v3 — 構造損失の根本原因と状態管理バグ |
| 096a | rev | [096_ph2_rev_095.md](096_ph2_rev_095.md) | 095 事後諸葛亮レビュー（ログ逆算 + 収益改善） |
| 096b | impl | [096_ph2_impl.md](096_ph2_impl.md) | 095# Codex Review Response 実装 |
| 097 | impl | [097_ph2_skipgate_retrain_preorder.md](097_ph2_skipgate_retrain_preorder.md) | SkipGate AS モデル再訓練（preorder-only features） |
| 098 | rpt | [098_ph2_post_097_deep_analysis.md](098_ph2_post_097_deep_analysis.md) | 097 後の構造診断 + 収益改善戦略 |
| 099 | rev | [099_ph2_rev_098.md](099_ph2_rev_098.md) | 098 改善点レビュー（トレーダー視点込み） |
| 100 | impl | [100_ph2_impl_codex_review_fixes.md](100_ph2_impl_codex_review_fixes.md) | 098#/099# Review Implementation — Post-Review Fix Bundle |
| 101 | rpt | [101_ph2_rpt_additional_structural_issues.md](101_ph2_rpt_additional_structural_issues.md) | 追加構造問題（098/099 未カバー） |
| 102 | fix | [102_ph2_fix_structural_fixes.md](102_ph2_fix_structural_fixes.md) | Structural Fixes Implementation |
| 103 | fix | [103_ph2_fix_yaml_externalization.md](103_ph2_fix_yaml_externalization.md) | YAML 設定外部化 — マジックナンバー一掃 |
| 104 | fix | [104_ph2_fix_self_review_retrain.md](104_ph2_fix_self_review_retrain.md) | Self-Review + SkipGate 再訓練 + \_\_post_init\_\_ バリデーション |
| 105 | fix | [105_ph2_fix_sell_offset_balance.md](105_ph2_fix_sell_offset_balance.md) | SELL offset 引上げ + balance insufficient 削減 |
| 106 | fix | [106_ph2_fix_refactoring_r1_r10.md](106_ph2_fix_refactoring_r1_r10.md) | リファクタリング調査 + 即時修正 (R1–R10) |
| 107 | rpt | [107_ph2_analysis_time_filter_dynamic_gating.md](107_ph2_analysis_time_filter_dynamic_gating.md) | Time Filter 分析 — 動的ゲーティングへの移行提案 |
| 110 | fix | [110_ph2_fix_086_time_filter_deadlock.md](110_ph2_fix_086_time_filter_deadlock.md) | 086# time_filter デッドロック修正 (49%アイドル解消) |
| 113 | impl | [113_ph2_impl_resilience_r1_split.md](113_ph2_impl_resilience_r1_split.md) | Resilience 統合 + R1 God Method 分割 (755→307行) |
| 114 | ext | [114_ph2_ext_gate_redesign_review.md](114_ph2_ext_gate_redesign_review.md) | G1.1 二段階ゲート再設計 — 外部 AI レビュー依頼 |
| 115 | resp | [115_ph2_ext_gate_redesign_review_response.md](115_ph2_ext_gate_redesign_review_response.md) | G1.1 二段階ゲート再設計 — 外部レビュー回答 |
| 116 | impl | [116_ph2_impl_two_stage_gate.md](116_ph2_impl_two_stage_gate.md) | 二段階ゲート実装 — 115# レビュー反映 |
| 117 | impl | [117_ph2_impl_import_chain_fix.md](117_ph2_impl_import_chain_fix.md) | Import Chain Fix + Fill Test 二重キャンセル防止 |
| 119 | rpt | [119_fill_test_161h_analysis.md](119_fill_test_161h_analysis.md) | Fill Test 161h 中間分析レポート |
| 120 | rev | [120_ph2_rev_119.md](120_ph2_rev_119.md) | 119 Fill Test 161h 分析の妥当性検証と追加改善提案 |
| 121 | plan | [121_ph2_plan_model_replacement.md](121_ph2_plan_model_replacement.md) | モデル差し替え・次期改善計画 |
| 122 | rev | [122_ph2_rev_121.md](122_ph2_rev_121.md) | 121# モデル差し替え計画 — 外部レビュー |
| 123 | rpt | [123_ph2_rpt_sell_structural_analysis.md](123_ph2_rpt_sell_structural_analysis.md) | Sell 構造問題分析と改善提案 (S125.1#) |
| 124 | rev | [124_ph2_rev_123.md](124_ph2_rev_123.md) | 123# Sell 構造分析レビュー — 外部 AI 検証 |
| 125 | impl | [125_ph2_impl_lgbm_pnl120_model.md](125_ph2_impl_lgbm_pnl120_model.md) | LGBM PnL120 回帰モデル構築・S1 適用 (S125.1#) |
| 126 | impl | [126_ph2_impl_retrain_hot_reload.md](126_ph2_impl_retrain_hot_reload.md) | SkipGate 定期再学習 + Hot-Reload (S125.1#) |
| 127 | rev | [127_ph2_rev_126.md](127_ph2_rev_126.md) | 126# レビュー — 契約統一・AS 非依存・run_id 分離 (S127.1#) |
| 128 | rpt | [128_ph2_rpt_log_review_and_strategy.md](128_ph2_rpt_log_review_and_strategy.md) | ログレビュー分析と改善方策（妥当性検証 + 追加提案追記） |
| 129 | rpt | [129_ph2_rpt_log_analysis_and_backlog.md](129_ph2_rpt_log_analysis_and_backlog.md) | ログ分析・改善指針・残課題統合レビュー (131# 外部レビュー用) |
| 130 | rpt | [130_ph2_rpt_implementation_and_retrain.md](130_ph2_rpt_implementation_and_retrain.md) | 6 施策実装 + retrain 改善 |
| 131 | rpt | [131_ph2_rpt_y3_retrain_efficiency.md](131_ph2_rpt_y3_retrain_efficiency.md) | Y3 retrain 効率化報告 |
| 132 | rpt | [132_ph2_rpt_fill_test_log_analysis.md](132_ph2_rpt_fill_test_log_analysis.md) | fill_test ログ分析・収益性最大化計画 |
| 133 | rev | [133_ph2_rev_132_profitability_max_plan.md](133_ph2_rev_132_profitability_max_plan.md) | 132# 収益性最大化計画レビュー |
| 134 | rev | [134_ph2_rev_133_validity_evaluation.md](134_ph2_rev_133_validity_evaluation.md) | 133# 妥当性評価 — Phase A-E ロードマップ |
| 135 | impl | [135_ph2_impl_data_infra_gate_perrun.md](135_ph2_impl_data_infra_gate_perrun.md) | Phase A/B: データインフラ復旧 + Gate per-run |
| 136 | impl | [136_ph2_impl_p1_retrain_kill.md](136_ph2_impl_p1_retrain_kill.md) | P1 retrain kill 施策実装 |
| 137 | impl | [137_ph2_impl_review_fixes_p1.md](137_ph2_impl_review_fixes_p1.md) | レビュー指摘対応 P1 |
| 138 | impl | [138_ph2_impl_p1_preflight_calibration.md](138_ph2_impl_p1_preflight_calibration.md) | P1 preflight calibration 実装 |
| 139 | fix | [139_ph2_fix_review_137_138.md](139_ph2_fix_review_137_138.md) | 137#/138# レビュー修正 |
| 140 | fix | [140_ph2_fix_critical_fillrecord.md](140_ph2_fix_critical_fillrecord.md) | FillRecord critical fix |
| 141 | impl | [141_ph2_impl_side_separation_regime_monitor.md](141_ph2_impl_side_separation_regime_monitor.md) | Side 分離 + Regime モニター |
| 142 | plan | [142_ph2_plan_regime_utilization.md](142_ph2_plan_regime_utilization.md) | Regime 活用計画 |
| 143 | impl | [143_ph2_impl_regime_utilization.md](143_ph2_impl_regime_utilization.md) | Regime 活用実装 |
| 144 | impl | [144_ph2_impl_regime_reprice_timeout.md](144_ph2_impl_regime_reprice_timeout.md) | Regime reprice + timeout 実装 |
| 145 | impl | [145_ph2_impl_regime_retrain_adaptation.md](145_ph2_impl_regime_retrain_adaptation.md) | Regime retrain adaptation (R-2a config) |
| 146 | impl | [146_ph2_impl_multi_exchange_registry.md](146_ph2_impl_multi_exchange_registry.md) | Multi-exchange registry 分離 |
| 147 | rpt | [147_ph2_rpt_phase_c_24h_run_start.md](147_ph2_rpt_phase_c_24h_run_start.md) | Phase C 24h 連続 run 開始・停止原因調査 |
| 148 | rev | [148_ph2_rev_147_phase_c_stop_cause_and_side_issues.md](148_ph2_rev_147_phase_c_stop_cause_and_side_issues.md) | 147# 補足レビュー: 停止原因再点検 + lock heartbeat 修正 |
| 149 | plan | [149_ph2_plan_phase_c_parallel_work.md](149_ph2_plan_phase_c_parallel_work.md) | Phase C 並行作業計画 (P2/P3 残項目検討) |
| 150 | plan | [150_ph2_plan_fill_test_auto_restart.md](150_ph2_plan_fill_test_auto_restart.md) | P2-B 自動再起動設計 |
| 151 | impl | [151_ph2_plan_dynamic_position_sizer.md](151_ph2_plan_dynamic_position_sizer.md) | P3-03 confidence_lot 実装 |
| 152 | plan | [152_ph2_plan_priority_improvements.md](152_ph2_plan_priority_improvements.md) | 代替優先施策: 144# CRITICAL検証 + P3-03判定 + P3-02 unknown削減 |
| 153 | refactor | [153_ph2_refactor_test_stabilization.md](153_ph2_refactor_test_stabilization.md) | P2 品質改善: テスト安定化 + run_fill_test 分割設計 |
| 154 | analysis | [154_ph2_dryrun_10h_analysis.md](154_ph2_dryrun_10h_analysis.md) | Dry-Run 10h ログ分析 & 改善提案 (P0-08 deadlock 発見) |
| 155 | rpt | [155_ph2_rpt_hindsight_filter_analysis.md](155_ph2_rpt_hindsight_filter_analysis.md) | Phase C ヒンドサイト分析: sell弱点・時間帯・regime×side |
| 156 | rpt | [156_ph2_rpt_sell_root_cause_and_phase_d_plan.md](156_ph2_rpt_sell_root_cause_and_phase_d_plan.md) | Sell根本原因7重ゲート分析 + 168h総括 + Phase C/D並行計画 |
| 157 | fix | [157_ph2_fix_regime_deadlock_and_cancel.md](157_ph2_fix_regime_deadlock_and_cancel.md) | §20 レジームデッドロック修正 + cancel re-raise + spread_too_narrow 分類 |
| 169 | rpt | [169_ph2_rpt_deep_analysis_and_scaling_plan.md](169_ph2_rpt_deep_analysis_and_scaling_plan.md) | 深堀り分析: G1.1 ゲート診断 + ロットスケーリング計画 + 外部 AI レビュー |
| 170 | rpt | [170_ph2_rpt_log_deep_analysis_and_type_safety.md](170_ph2_rpt_log_deep_analysis_and_type_safety.md) | ログ深堀り + 型安全強化 + Config Hot-Reload + AI レビュー準備 |
| 171 | rpt | [171_ph2_rpt_sell_guard_paradox_deep_dive.md](171_ph2_rpt_sell_guard_paradox_deep_dive.md) | Sell Guard Paradox 技術精査 + balance_forced_skip 正フィードバックループ発見 |
| 172 | fix | [172_ph2_fix_guard_paradox_and_ev_per_cycle.md](172_ph2_fix_guard_paradox_and_ev_per_cycle.md) | Guard Paradox 根本対策 (InvSkew bypass) + EV_per_cycle 実装 |
| 173 | fix | [173_ph2_fix_code_review_sweep.md](173_ph2_fix_code_review_sweep.md) | 包括的コードレビュー Sweep — CRITICAL 1 / HIGH 5 / MED 6 / 機能改善 1 |
| 174 | fix | [174_ph2_fix_fresh_code_review.md](174_ph2_fix_fresh_code_review.md) | Fresh Code Review — CRITICAL 1 / HIGH 5 / MED 1 |
| 175 | fix | [175_ph2_fix_code_review_sweep2.md](175_ph2_fix_code_review_sweep2.md) | Code Review Sweep #2 — HIGH 2 / MED 5 / LOW 4 |
| 176 | impl | [176_ph2_impl_trending_offset_asymmetry.md](176_ph2_impl_trending_offset_asymmetry.md) | Trending方向×サイド別Offset Asymmetry + 横展開 |
| 177 | rev | [177_ph2_rev_176_trending_capture_root_solution.md](177_ph2_rev_176_trending_capture_root_solution.md) | 176レビュー: 大値動き取り逃し是正 + 追加方策 + vXXX再利用計画 |
| 178 | rev | [178_ph2_rev_177_evaluation.md](178_ph2_rev_177_evaluation.md) | 177レビュー評価: Codex/Gemini 提案精査 + CycleStrategy 方針 |
| 179 | impl | [179_ph2_impl_regime_policy_cycle_strategy.md](179_ph2_impl_regime_policy_cycle_strategy.md) | RegimePolicyConfig + CycleStrategy + _effective_sleep + Chase |
| 180 | impl | [180_ph2_impl_watchdog_hidden_review.md](180_ph2_impl_watchdog_hidden_review.md) | Watchdog 非表示化 + 179# Self-Review + from_yaml 堅牢化 |
| 181 | impl | [181_ph2_impl_cd_chase_enable_ev_weighted.md](181_ph2_impl_cd_chase_enable_ev_weighted.md) | C/D/Chase 有効化 + EV_weighted + Stop Condition Monitor |
| 182 | impl | [182_ph2_impl_trend_strict_ev_ext_deadlock.md](182_ph2_impl_trend_strict_ev_ext_deadlock.md) | Trend Mode 厳格化 + EV_weighted外部化 + Deadlock regime別緩和 |
| 183 | impl | [183_ph2_impl_log_analysis_adverse_guard.md](183_ph2_impl_log_analysis_adverse_guard.md) | ログ分析ベース逆選択防御強化 (hour_offsets + velocity_skip + narrow_spread_guard + VG/VPIN) |
| 184 | ext | [184_ph2_ext_adverse_guard_review.md](184_ph2_ext_adverse_guard_review.md) | 逆選択防御施策レビュー依頼 — 外部 AI レビュー |
| 185 | rev | [185_ph2_rev_184_adverse_guard_macro_trend_review.md](185_ph2_rev_184_adverse_guard_macro_trend_review.md) | 184レビュー: 逆選択防御の妥当性 + 大値動き追随不足の根本分析 |
| 186 | rev | [186_ph2_rev_185_evaluation_and_plan.md](186_ph2_rev_185_evaluation_and_plan.md) | 185レビュー評価 + Trend Mode ヒステリシス + Strictness Clamp |
| 187 | impl | [187_ph2_impl_chase_direction_guard_trace.md](187_ph2_impl_chase_direction_guard_trace.md) | Chase 方向制御 + guard_trace 記録 + clamp YAML外部化 |
| 188 | impl | [188_ph2_impl_split_evc_macro.md](188_ph2_impl_split_evc_macro.md) | ファイル分割 + Phase C ev_weighted SkipGate + Phase D Macro Regime 基盤 |
| 189 | impl | [189_alt_horizon_macro_integration.md](189_alt_horizon_macro_integration.md) | Alt horizon モデル訓練 + ev_weighted SkipGate + MacroRegime 基盤 |
| 190 | fix | [190_ph2_fix_ev_weighted_deadlock.md](190_ph2_fix_ev_weighted_deadlock.md) | ev_weighted デッドロック修正 + min_spread_jpy 緩和 + pnl_threshold 調整 |
| 191 | rev | [191_ph2_rev_guard_complexity_analysis.md](191_ph2_rev_guard_complexity_analysis.md) | Guard Layer 複雑性分析 + 簡素化提案 (AI レビュー用) |
| 192 | rev | [192_ph2_rev_191_guard_simplification_validation.md](192_ph2_rev_191_guard_simplification_validation.md) | 191レビュー: Guard複雑性分析の検証 + 根本簡素化方針 |
| 193 | impl | [193_ph2_impl_ev_weighted_to_offset.md](193_ph2_impl_ev_weighted_to_offset.md) | ev_weighted gate→offset modifier 変換 (192# §5.2 + Gemini §9.4) |
| 194 | impl | [194_ph2_impl_cycle_gate_aggregator.md](194_ph2_impl_cycle_gate_aggregator.md) | CycleGateAggregator per-cycle skip 判定一元化 (192# §3 対応) |
| 195 | impl | [195_ph2_impl_velocity_b1_soft_gate.md](195_ph2_impl_velocity_b1_soft_gate.md) | velocity_skip ソフト化 + B1' offset 統合 (193# 横展開) |
| 196 | impl | [196_ph2_impl_velocity_proportional_trending_soft.md](196_ph2_impl_velocity_proportional_trending_soft.md) | velocity offset 比例化 + trending_sell ソフト化 |
| 197 | impl | [197_ph2_impl_boost_optimization_gate_integration.md](197_ph2_impl_boost_optimization_gate_integration.md) | boost 最適化 + balance_forced offset + Gate 8-9 統合 |
| 198 | rpt | [198_ph2_rpt_drawdown_postmortem_20260301.md](198_ph2_rpt_drawdown_postmortem_20260301.md) | 事後分析: 2026-03-01 朝セッション -53bps ドローダウン |
| 199 | rev | [199_ph2_rev_198_drawdown_and_hidden_risks.md](199_ph2_rev_198_drawdown_and_hidden_risks.md) | 198レビュー: ドローダウン分析の検証 + 隠れた再発要因 |
| 200 | resp | [200_ph2_resp_199_codex_gemini_review_eval.md](200_ph2_resp_199_codex_gemini_review_eval.md) | 199 Codex/Gemini レビュー評価 + P0 実装 |
| 201 | impl | [201_ph2_impl_an_comprehensive_improvements.md](201_ph2_impl_an_comprehensive_improvements.md) | A–N 包括的改善: postonly skip, 比例 boost, velocity SSOT, ev warning, daily reset bugfix |
| 202 | impl | [202_ph2_impl_log_based_improvements.md](202_ph2_impl_log_based_improvements.md) | ログ分析ベース改善: loss cooldown, one-sided rescue offset, VG sell supplement |
| 203 | fix | [203_ph2_fix_dd_state_persistence.md](203_ph2_fix_dd_state_persistence.md) | P0: DD状態HALT中未保存バグ修正 + fill records warmup + halt counter修正 |
| 204 | rpt | [204_ph2_rpt_comprehensive_trade_analysis.md](204_ph2_rpt_comprehensive_trade_analysis.md) | 包括的トレード分析 — MM理論・一目均衡表・市場微細構造理論11理論適用 + vXXX資産活用計画 |
| 205 | rev | [205_ph2_rev_200_204_root_cause_progress_and_blind_spots.md](205_ph2_rev_200_204_root_cause_progress_and_blind_spots.md) | 200–204レビュー: 根本解決への進捗評価 + 盲点整理 |
| 206 | impl | [206_ph2_impl_205_review_response_p0.md](206_ph2_impl_205_review_response_p0.md) | 205レビュー応答: P0 3施策実装 (Hard Skip §9.4 / Toxic Veto §9.2 / 片側DD §9.5) |
| 207 | fix | [207_ph2_fix_206_robustness_and_one_sided_limit.md](207_ph2_fix_206_robustness_and_one_sided_limit.md) | 206堅牢性修正 5件 + 片側連続実行制限 (205# §4.2) |
| 208 | refactor | [208_ph2_refactor_velocity_ssot.md](208_ph2_refactor_velocity_ssot.md) | Velocity SSOT 強化 — instant velocity 計算を velocity_math に移動 (205# §3.2/§9.1) |
| 209 | fix | [209_ph2_fix_self_review_and_audit.md](209_ph2_fix_self_review_and_audit.md) | セルフレビュー + コード監査: vetoデッドロック防止, config検証, sleep上限, health監視修正 |
| 210 | fix | [210_ph2_fix_remaining_203_204_issues.md](210_ph2_fix_remaining_203_204_issues.md) | 203#/204# 残課題解消: FFD hot-reload同期, velocity配線, one-sided永続化, spread staleness, DRY snapshot |
| 211 | fix | [211_ph2_fix_204i_offset_boost_sleep_clamp.md](211_ph2_fix_204i_offset_boost_sleep_clamp.md) | 204# I offset boost 3層防御完成, _effective_sleep clamp, halt可視化ログ, 198# link修正 |
| 212 | audit | [212_ph2_audit_codebase_quality.md](212_ph2_audit_codebase_quality.md) | コードベース品質監査 — Codex/Gemini 外部レビュー用改善ポイント一覧 |
| 213 | rev | [213_ph2_rev_205_212_validation_and_proposals.md](213_ph2_rev_205_212_validation_and_proposals.md) | 205# Gemini追加分〜212# 横断レビュー: 実装検証, DD状態移行穴, velocity混線, 211# 外部イベント監査 |
| 214 | resp | [214_ph2_resp_213_codex_gemini_verification.md](214_ph2_resp_213_codex_gemini_verification.md) | 213# Codex/Gemini指摘に対する実コード・実データ検証: DD state 5フィールド不整合確認, hot-reload 7漏れ確認, velocity名称問題評価 |
| 215 | fix | [215_ph2_fix_dd_hotreload_alertmode.md](215_ph2_fix_dd_hotreload_alertmode.md) | P0実装: DD state整合性修復 (if/elif+warmup), hot-reload 13フィールド追加, alert_mode.json DEFCONスイッチ |
| 216 | fix | [216_ph2_fix_velocity_rename_guard_counters.md](216_ph2_fix_velocity_rename_guard_counters.md) | P1実装: velocity引数リネーム(19ファイル), guard発火カウンタ永続化, 211#§8事実/仕様分離 |
| 217 | rev | [217_ph2_self_review_211_214_216.md](217_ph2_self_review_211_214_216.md) | セルフレビュー: 211# MCB/SAD + 214#-216# 実装検証 (CRITICAL 1, SIGNIFICANT 3, LOW-RISK 2) |
| 218 | fix | [218_ph2_fix_anti_deadlock_probe.md](218_ph2_fix_anti_deadlock_probe.md) | Anti-Deadlock: DynamicKill probe cycle + per-side halt + deadlock detection log |
| 219 | fix | [219_ph2_fix_progressive_probe_force_release.md](219_ph2_fix_progressive_probe_force_release.md) | Progressive Probe + Force Release: DynamicKill回復の高速化 (max_stale 30→10, 半減interval, 5probe強制解除) |
| 220 | fix | [220_ph2_fix_gate_level_deadlock.md](220_ph2_fix_gate_level_deadlock.md) | Gate-level Deadlock 3fixes: Gate7 balance_forced対称性, dual-kill breaker, unknown連続bypass |
| 221 | rev | [221_ph2_rev_ai_review_deadlock_and_pnl.md](221_ph2_rev_ai_review_deadlock_and_pnl.md) | AI Review: 218#-220# デッドロック対策総合レビュー + PnL構造分析 + 改善提案6件 |
| 222 | rev | [222_ph2_rev_213_221_deadlock_validation_and_residual_risks.md](222_ph2_rev_213_221_deadlock_validation_and_residual_risks.md) | 213#–221# レビュー: デッドロック対策の実証, per-side halt破り, SHA混在評価の補正, 残存リスク |
| 223 | fix | [223_ph2_fix_222_review_response.md](223_ph2_fix_222_review_response.md) | 222# レビュー対応: CRITICAL halt bypass修正, guard_fire_counts 7種追加, skip-time state save, DUAL KILL廃止ロードマップ |
| 224 | fix | [224_ph2_fix_halt_recovery_and_kill_reset.md](224_ph2_fix_halt_recovery_and_kill_reset.md) | 後続作業: B1/B2 halt解除後ソフトリカバリ + 盲点修正 |
| 225 | fix | [225_ph2_fix_warmup_state_save_recovery_market_theory.md](225_ph2_fix_warmup_state_save_recovery_market_theory.md) | warmup日付フィルタ + state save強化 + recovery復元 + 市場理論補強 |
| 226 | fix | [226_ph2_fix_loss_boost_decay_mcb_ffd_state_inv_skew.md](226_ph2_fix_loss_boost_decay_mcb_ffd_state_inv_skew.md) | loss_boost指数減衰 + MCB/FFD state永続化 + inv_skew O(1) + toxic_veto修正 + halt中MCB/SAD更新 |
| 227 | fix | [227_ph2_fix_ranging_obi_velocity_ema_import_optimization.md](227_ph2_fix_ranging_obi_velocity_ema_import_optimization.md) | Ranging×OBI方向非対称 + Velocity EMAフィルタ + import最適化 + getattr排除 + Config検証 |
| 228 | fix | [228_ph2_fix_inv_decay_hasattr_removal.md](228_ph2_fix_inv_decay_hasattr_removal.md) | Inventory Time-Decay + hasattr排除 |
| 229 | fix | [229_ph2_fix_code_hygiene_counter_rename.md](229_ph2_fix_code_hygiene_counter_rename.md) | コード衛生 + M-5 unknown counter fix + M-2 consume rename |
| 230 | fix | [230_ph2_fix_ffd_deadzone_streak_guards.md](230_ph2_fix_ffd_deadzone_streak_guards.md) | FFD deadzone/streak + MCB/SAD guard + hasattr排除 |
| 231 | fix | [231_ph2_fix_ffd_logic_hardening_null_safety.md](231_ph2_fix_ffd_logic_hardening_null_safety.md) | FFDロジック強化 + import_state None安全 |
| 232 | rev | [232_ph2_rev_222_231_predeployment_risk_review.md](232_ph2_rev_222_231_predeployment_risk_review.md) | 222#–231# レビュー: 本筋投入前の先回り点検, FFD妥当性, 実運用SHA乖離, feasible set collapse, 残存リスク |
| 233 | rev | [233_ph2_gemini_31_pro_final_judgement_and_breakthrough.md](233_ph2_gemini_31_pro_final_judgement_and_breakthrough.md) | Gemini 3.1 Pro 最終審判: 1時間破綻の真因 + アーキテクチャ根本欠陥 + Liveness制約 |
| 234 | fix | [234_ph2_fix_gate_bypass_degraded_liquidation.md](234_ph2_fix_gate_bypass_degraded_liquidation.md) | Gate bypass + 縮退清算モード + one-sided エスカレーション |
| 235 | fix | [235_ph2_fix_234_self_review_cleanup.md](235_ph2_fix_234_self_review_cleanup.md) | 234# セルフレビュー: FFD状態復元 + 清算モード改善 + CycleStrategy統合 |
| 236 | fix | [236_ph2_fix_state_persistence_cqs_hasattr.md](236_ph2_fix_state_persistence_cqs_hasattr.md) | State永続化 + CQS分離 + hasattr排除 + per-side no_feasible |
| 237 | fix | [237_ph2_fix_phantom_position_guard.md](237_ph2_fix_phantom_position_guard.md) | PhantomPositionGuard: status_unknown 幽霊ポジション検知・遅延照合 (232# §1.6) |
| 238 | fix | [238_ph2_fix_237_self_review.md](238_ph2_fix_237_self_review.md) | 237# セルフレビュー: 型安全 + 残高スナップショット + TTL + サイドベトー |
| 239 | fix | [239_ph2_fix_feasible_quote_proactive.md](239_ph2_fix_feasible_quote_proactive.md) | Feasible Quote Proactive: InfeasibleQuoteError + 制約前方移動 + fallback dedup (232# §1.5) |
| 240 | fix | [240_ph2_fix_toxicity_budget.md](240_ph2_fix_toxicity_budget.md) | Toxicity Budget: binary skip → continuous adverse-selection budget (232# §2.2) |
| 241 | fix | [241_ph2_fix_toxicity_budget_review.md](241_ph2_fix_toxicity_budget_review.md) | 240# セルフレビュー: dead code 修正 + 評価順序 + 型安全 + config バリデーション |
| 242 | fix | [242_ph2_fix_liveness_constraint_relaxation.md](242_ph2_fix_liveness_constraint_relaxation.md) | Liveness Constraint Relaxation: dual_kill/片側硬直の緩和で No Trade を正常化 |
| 243 | fix | [243_ph2_fix_yaml_wiring.md](243_ph2_fix_yaml_wiring.md) | 242# 追加設定の YAML 配線漏れ修正 + バリデーション補強 |
| 244 | impl | [244_ph2_impl_guard_reason_classification.md](244_ph2_impl_guard_reason_classification.md) | Guard reason を MARKET/SYSTEM/RECOVERY に分類し、集計可視化を追加 |
| 245 | rpt | [245_ph2_production_log_analysis_mar03.md](245_ph2_production_log_analysis_mar03.md) | 本番ログ分析 (2026-02-13〜2026-03-03): sell劣後・在庫中立前提・DD挙動を再点検 |
| 246 | fix | [246_ph2_fix_dd_cooldown_release_sell_defense.md](246_ph2_fix_dd_cooldown_release_sell_defense.md) | DD Halt Cooldown Release + Sell Defence Hardening |
| 247 | rev | [247_ph2_rev_234_246_functionality_market_theory_review.md](247_ph2_rev_234_246_functionality_market_theory_review.md) | 234#–246# レビュー: 未デプロイ検証, DD cooldown再武装不足, inventory中立前提の見直し |
| 248 | rev | [248_ph2_gemini_31_pro_review_234_246_directional_alpha.md](248_ph2_gemini_31_pro_review_234_246_directional_alpha.md) | Gemini 3.1 Pro レビュー: Directional Alpha パラダイムシフト + P0提案5件 |
| 249 | impl | [249_ph2_impl_directional_alpha_dd_rearm.md](249_ph2_impl_directional_alpha_dd_rearm.md) | Directional Alpha + DD Re-arm + Quiescence: 247#/248# P0 全5件実装 |
| 250 | impl | [250_ph2_impl_pl_split_freeze_side_probe.md](250_ph2_impl_pl_split_freeze_side_probe.md) | P/L 3分離・freeze side紐付け・quiescence deadlock防御・probe廃止基盤 |
| 251 | rev | [251_ph2_pre_impl_review_report.md](251_ph2_pre_impl_review_report.md) | Pre-Implementation Review: 247#/248# 残 P1/P2 項目の実装準備レビュー |
| 252 | impl | [252_ph2_impl_sell_asymmetric_phantom_ternary.md](252_ph2_impl_sell_asymmetric_phantom_ternary.md) | Sell Asymmetric Gate + PhantomGuard 三値化 + 型安全化 |
| 253 | rev+impl | [253_phg_rev_pre_impl_codebase_sweep.md](253_phg_rev_pre_impl_codebase_sweep.md) / [253_ph2_impl_hot_reload_dead_config_getattr_bare_except.md](253_ph2_impl_hot_reload_dead_config_getattr_bare_except.md) | 252# self-review sweep + hot_reload 配線漏れ, dead config 削除, getattr 排除, bare except 改善 |
| 254 | impl | — | frozen_side 永続化, orchestrator getattr 排除, bare except 改善 |
| 255 | impl | [255_phg_rev_codebase_sweep.md](255_phg_rev_codebase_sweep.md) | skip_gate_evaluator/order_monitor getattr 排除, bare except → debug log |
| 256 | impl | [256_phg_impl_recent_records_fix_self_review.md](256_phg_impl_recent_records_fix_self_review.md) | _recent_records 累積バグ修正, セルフレビュー |
| 257 | rpt | [257_phg_rpt_codebase_sweep.md](257_phg_rpt_codebase_sweep.md) | Codebase Sweep: ドキュメント整合・市場理論・再利用・技術的負債 (P1×9, P2×47) |
| 258 | impl | [258_phg_impl_as_reservation_vpin_continuous_protocol.md](258_phg_impl_as_reservation_vpin_continuous_protocol.md) | AS Reservation Price, VPIN Continuous, RegimeDetectorLike Protocol |
| 259 | rpt+impl | [259_phg_rpt_codebase_sweep.md](259_phg_rpt_codebase_sweep.md) | Sweep + AS σ² vol_ratio 統合, adaptation_engine hasattr 排除 |
| 260 | refactor | [260_phg_refactor_compute_extract_regime_split.md](260_phg_refactor_compute_extract_regime_split.md) | compute() extract method (_apply_loss_boost, _apply_ffd_boost) + _apply_regime_boosts 5-split |
| 261 | impl | [261_phg_impl_protocol_type_safety.md](261_phg_impl_protocol_type_safety.md) | P2-1/5/6/7: OrderBookLevelLike, OrderBookSnapshot, BalanceAdapterProtocol, config_hot_reload getattr 排除 |
| 262 | impl | [262_phg_impl_protocol_cancel_recheck_dry.md](262_phg_impl_protocol_cancel_recheck_dry.md) | adaptation_engine Protocol化 (type:ignore×4 排除) + order_monitor cancel-recheck DRY化 |
| 263 | impl | [263_phg_impl_optional_xnone_unification.md](263_phg_impl_optional_xnone_unification.md) | Optional[X]→X|None 統一 + 未使用import削除 (18ファイル, 87箇所) |
| 264 | impl | [264_phg_impl_kelly_criterion_lot_sizing.md](264_phg_impl_kelly_criterion_lot_sizing.md) | Kelly Criterion lot sizing (f*=(pb-q)/b, Fractional Kelly 天井) |
| 265 | refactor | [265_phg_refactor_run_continuous_extract_skip_gate_protocol.md](265_phg_refactor_run_continuous_extract_skip_gate_protocol.md) | run_continuous extract methods (1694→1221行) + SkipGateAdapter Protocol + P3-3 docs |
| 266 | impl | [266_phg_impl_market_theory_protocol.md](266_phg_impl_market_theory_protocol.md) | GLFT τ動的化 + AS δ* + Kyle λ + Amihud ILLIQ + Protocol型安全化 (type:ignore×4, getattr×8 排除) |
| 267 | bugfix | [267_phg_fix_delta_star_depth_dry.md](267_phg_fix_delta_star_depth_dry.md) | δ* 次元修正 (σ_return→σ_abs) + _get_depth DRY + docstring 正確化 |
| 268 | bugfix | [268_phg_fix_dd_halt_jst_reset.md](268_phg_fix_dd_halt_jst_reset.md) | DD halt JST日付リセット: _utc_today→_today (day_reset_utc_offset_hours=9.0) |
| 269 | rev | [269_ph2_rev_249_268_blocking_architecture_and_next_moves.md](269_ph2_rev_249_268_blocking_architecture_and_next_moves.md) | 249#–268# レビュー: side-halt deadlock 継続, degraded 到達不能, state stale, Inventory Escape Mode 提案 |
| 270 | rev | [270_ph2_gemini_31_pro_review_249_269_bureaucratic_deadlock.md](270_ph2_gemini_31_pro_review_249_269_bureaucratic_deadlock.md) | Gemini 3.1 Pro レビュー: Bureaucratic Deadlock, Debt Trap, Sleeping Giants, 在庫エスケープ提言 |
| 271 | rev | [271_ph2_rev_269_270_review_validity_assessment.md](271_ph2_rev_269_270_review_validity_assessment.md) | 269#/270# レビュー妥当性評価 + Inventory Escape Mode + PnL reanchor + 市場理論YAML配線 実装完了 |
| 272 | impl | [272_ph2_impl_dry_refactor_and_residual_analysis.md](272_ph2_impl_dry_refactor_and_residual_analysis.md) | DRY リファクタ: `_tick_toxic_veto` / `_maybe_skip_state_save` / `_feed_mcb_sad` / `_opposite_side` + 269# 残指摘の掘り下げ検証 |
| 273 | impl | [273_ph2_impl_268_incident_resolution.md](273_ph2_impl_268_incident_resolution.md) | 268# インシデント残課題: I3 空サイクル halt 除外 / I5 kill 時間上限 / I6 halt 後 gate grace / Pattern B 解消 |
| 274 | impl | [274_ph2_impl_theory_strengthening_and_pattern_c.md](274_ph2_impl_theory_strengthening_and_pattern_c.md) | 市場理論補強 (Stoll/Ho-Stoll/Glosten-Milgrom/Kyle/Avellaneda-Stoikov) + MacroRegime 観測有効化 + Kelly YAML + Pattern C 3層検証 + deprecated CLI 削除 |
| 275 | impl | [275_ph2_impl_dry_separation_and_theory_expansion.md](275_ph2_impl_dry_separation_and_theory_expansion.md) | 責務分離 DRY: side パラメータ化 (_is_side_killed/_track_side_pnl) + toxic veto DRY + 市場理論 8モジュール拡大 (Hamilton/Lo/Garman/Brunnermeier/Copeland-Galai/Amihud/Hasbrouck/Greenwald-Stein) |
| 276 | analysis | [276_ph2_analysis_blocking_policy_extraction.md](276_ph2_analysis_blocking_policy_extraction.md) | BlockingPolicy 抽出分析: 22 BP マッピング + 6 クラスタ + 市場理論コード活用候補 |
| 277 | impl | [277_ph2_impl_blocking_policy_dry.md](277_ph2_impl_blocking_policy_dry.md) | BlockingPolicy DRY: _execute_skip ヘルパー (14箇所統一) + halt_sleep_multiplier config化 (Brunnermeier-Pedersen) |
| 278 | impl | [278_ph2_impl_magic_number_grounding.md](278_ph2_impl_magic_number_grounding.md) | マジックナンバー根拠化 (5 config化 + 3 __post_init__検証 + B1 warmup TZ fix) + 271#-277# セルフレビュー |
| 279 | fix | [279_ph2_fix_degraded_liquidation_min_lot.md](279_ph2_fix_degraded_liquidation_min_lot.md) | CRITICAL fix: degraded_liquidation config.min_lot → config.min_order_btc (234# 属性名取違え) |
| 280 | rpt | [280_ph2_rpt_position_and_remaining_tasks.md](280_ph2_rpt_position_and_remaining_tasks.md) | 0番ドキュメント立ち位置確認 + 残課題浚い上げ (R-1〜R-26) |
| 281 | fix | [281_ph2_fix_halt_persist_interval_nameError.md](281_ph2_fix_halt_persist_interval_nameError.md) | CRITICAL fix: NameError `_HALT_PERSIST_INTERVAL` — 278# config化の参照漏れ (halt 時プロセス即死) |
| 282 | fix | [282_ph2_fix_balance_forced_halt_deadlock.md](282_ph2_fix_balance_forced_halt_deadlock.md) | CRITICAL fix: balance_forced + per-side halt 永久デッドロック修正 (untick除去×2 + IE双方向化 + 15tests) |
| 283 | rev | [283_ph2_rev_271_282_deadlock_and_preincident_market_analysis.md](283_ph2_rev_271_282_deadlock_and_preincident_market_analysis.md) | 271〜282レビュー: デッドロック再検証 + 発生前ログの市場理論分析 + 見落とし是正提案 |
| 284 | rev | [284_ph2_gemini_31_pro_review_271_283_split_brain_and_buy_toxicity.md](284_ph2_gemini_31_pro_review_271_283_split_brain_and_buy_toxicity.md) | Gemini 3.1 Pro セカンドオピニオン: 282# 修正評価 + Split-Brain P0確認 + buy 毒性分析 |
| 285 | fix | [285_ph2_fix_split_brain_guard_and_config_constraint.md](285_ph2_fix_split_brain_guard_and_config_constraint.md) | 283#/284# P0 対応: FillRecord pid 追加 + per_side_dd/IE 相互制約 + 282# doc 修正 |
| 286 | fix | [286_ph2_fix_282_284_comprehensive_resolution.md](286_ph2_fix_282_284_comprehensive_resolution.md) | 282#–284# 課題包括的解決: Lock portalocker強化, Split-Brain検知, KPI分離, AS防御, Guard再分類 |
| 287 | fix | [287_ph2_fix_balance_forced_switch_attribute.md](287_ph2_fix_balance_forced_switch_attribute.md) | CRITICAL fix: `record.balance_forced` → `balance_forced_switch` AttributeError修正 (286# P1-5 属性名誤記) |
| 288 | rpt | [288_ph2_ph3_rpt_skipgate_retrain_assessment.md](288_ph2_ph3_rpt_skipgate_retrain_assessment.md) | SkipGate全データ再訓練評価 + retrain_scheduler(126#)整合性確認。全5モデル品質ゲート棄却=既存最適化済 |
| 289 | analysis | [289_ph2_analysis_buy_side_improvement.md](289_ph2_analysis_buy_side_improvement.md) | Buy側PnL改善深堀v4: SGスコア反転=Simpson's Paradox, ev_weighted_pnl=tautological, 290#でmodel_used誤プロキシ判明→292#で解消 |
| 290 | rev | [290_ph2_rev_289_buy_side_systems_market_corrections.md](290_ph2_rev_289_buy_side_systems_market_corrections.md) | 289レビュー補正: ev_as_offset前提で「ev利用犇9.6%」解釈を修正。buy不振を在庫修復交絡+低情報帯参加+観測不足の複合問題として再定義 |
| 291 | rev | [291_ph2_gemini_31_pro_review_289_290_buy_side_blindspots.md](291_ph2_gemini_31_pro_review_289_290_buy_side_blindspots.md) | Gemini 3.1 Proレビュー: Queue Position放棄・強制買い毒性・観測不足の指摘 |
| 292 | impl | [292_ph2_impl_ev_weighted_observability_enhancement.md](292_ph2_impl_ev_weighted_observability_enhancement.md) | 290#/291#レビュー実装: FillRecord 3フィールド追加(ev_score_pretrade/offset_mult/decision_path) + reprice deadband + forced_buy_delayレジーム強化 |
| 293 | analysis | [293_ph2_blind_spot_analysis_buy_side_execution.md](293_ph2_blind_spot_analysis_buy_side_execution.md) | 290#/291#査読補完: セルフレビューによるblind spot分析 |
| 294 | fix | [294_ph2_fix_forced_buy_delay_deadlock.md](294_ph2_fix_forced_buy_delay_deadlock.md) | CRITICAL fix: forced_buy_delay リアームループによる永久buyブロック。max_consecutive上限で解消 |
| 295 | impl | [295_ph2_hot_reload_comprehensive_coverage.md](295_ph2_hot_reload_comprehensive_coverage.md) | Config hot-reload包括的カバレッジ修正: 157フィールド追加 (312/368=84.8%) + 290#/291#対応漏れ調査 + セルフレビュー |
| 296 | impl | [296_ph2_impl_p2p3_cleanup_and_v459_asset_survey.md](296_ph2_impl_p2p3_cleanup_and_v459_asset_survey.md) | P2/P3 cleanup: B-14 except as e (16箇所), F-2 cancel_reason CR定数化 (5新規+18置換), B-17 MCB/SAD型安全化, v459資産調査 |
| 297 | rpt | [297_ph2_rpt_f4_g2_pre_analysis.md](297_ph2_rpt_f4_g2_pre_analysis.md) | F-4/G-2 事前調査: 統計検定スタック分析 + 168# P3残タスク評価 |
| 298 | impl | [298_ph2_impl_f4_nonparametric_tests.md](298_ph2_impl_f4_nonparametric_tests.md) | F-4: Mann-Whitney U + Cliff's δ + Holm-Bonferroni を ab_judgment に統合。G-2: CB/DD 統合済確認 |
| 299 | rpt | [299_ph2_rpt_ab_test_f4_validation.md](299_ph2_rpt_ab_test_f4_validation.md) | A/Bテスト実施報告: F-4検定バリデーション + レジーム別考察 (6,952レコード/22日間) |
| 300 | rev | [300_ph2_rev_ab_test_deep_analysis.md](300_ph2_rev_ab_test_deep_analysis.md) | A/Bテスト深堀り: 5構造的矛盾の特定 + Glosten-Milgrom/Kyle/A-S理論 + 検出力分析 + 外部AIレビュー用Q5 |
| 301 | rev | [301_ph2_rev_292_300_multifaceted_review.md](301_ph2_rev_292_300_multifaceted_review.md) | 292-300横断レビュー: `none` 除外によるF-4楽観化、sell-vs-buy擬似A/B、hot-reload過信、forced_buy_delayの位置付け、統計/可観測性の残課題を整理 |
| 302 | rev | [302_ph2_gemini_31_pro_review_300_301_hft_blindspots.md](302_ph2_gemini_31_pro_review_300_301_hft_blindspots.md) | Gemini 3.1 Pro HFT盲点レビュー: none regime過剰エンジニアリング、Taker執行欠落、AlternationTrap+DD Guard死のスパイラル |
| 303 | resp/impl | [303_ph2_resp_301_302_review_response.md](303_ph2_resp_301_302_review_response.md) | 301#/302#レビュー応答+実装: Side Comparison表記修正, none含有版二系統出力, DD soft lot side分離, none regime Passive MMバイパス |
| 304 | refactor | [304_ph2_refactor_bps_ssot_dry_helpers.md](304_ph2_refactor_bps_ssot_dry_helpers.md) | BPS_FACTOR SSOT + DRY ヘルパー + マジックナンバー排除: 定数集約, mid逆推定ヘルパー, side別PnL計算共通化, hot_swap PID修正 |
| 305 | rpt | [305_ph2_analysis_systems_market_theory_p0_improvements.md](305_ph2_analysis_systems_market_theory_p0_improvements.md) | システム工学+市場理論分析: spread capture / AS cost 分解, Parkinson σ, EV-based offset, microprice side, queue/cancel latency 改善提案 |
| 306 | impl | [306_ph2_impl_six_proposals_observational_redesign.md](306_ph2_impl_six_proposals_observational_redesign.md) | 6提案実装 + 299# 観察比較再設計: queue推定, microprice side, dynamic interval, EV-based adaptation, offset stage recording, ceiling, deep dive再分析 |
| 307 | rev | [307_ph2_rev_303_306_systems_market_review.md](307_ph2_rev_303_306_systems_market_review.md) | 303-306再レビュー: 306 deep dive の観測設計ギャップ, side差解釈の限界, AS/session本丸論, none/repair交絡, A1ロジック整合性を整理 |
| 308 | rev | [308_ph2_gemini_31_pro_review_306_307_inverted_microstructure.md](308_ph2_gemini_31_pro_review_306_307_inverted_microstructure.md) | Gemini 3.1 Pro レビュー: L1/L2 理論倒錯発見 + AS Seeker 指摘 |
| 309 | resp | [309_ph2_review_response_307_308_fixes.md](309_ph2_review_response_307_308_fixes.md) | 307#/308# レビュー対応: L1/L2 理論倒錯修正 + deep dive スキーマ是正 |
| 310 | impl | [310_ph2_impl_design_improvements.md](310_ph2_impl_design_improvements.md) | 設計面改修: 時間帯ブースト + Decision Path + L2 Safety + None Observability + Spread/AS 分解 |
| 311 | rpt | [311_ph2_rpt_observational_comparison_rerun.md](311_ph2_rpt_observational_comparison_rerun.md) | 観測比較再実行 + 309#/310# 理論修正検証 + 深堀り分析 |
| 312 | rev | [312_ph2_rev_308_311_multifaceted_validation.md](312_ph2_rev_308_311_multifaceted_validation.md) | 308-311再レビュー: mixed-SHA分析の限界, spread/AS分解式の再検証, sell_hour_boost評価の交絡, floor discount仮説の扱い, none/L2の優先順位を再整理 |
| 313 | rev | [313_ph2_gemini_31_pro_review_309_312_pricing_math_inversion.md](313_ph2_gemini_31_pro_review_309_312_pricing_math_inversion.md) | Gemini 3.1 Pro レビュー: spread capture 転倒確認 + VG「自発的自殺」指摘 + fill rate 過剰最適化 |
| 314 | resp | [314_ph2_resp_312_313_review_response_plan.md](314_ph2_resp_312_313_review_response_plan.md) | 312#/313# レビュー応答: 全 Finding 検証 + 盲点 4 件 (B1 ratio セマンティクス転倒, B2 ceiling 未適用) + 3 Phase 実行計画 |
| 315 | rpt | [315_ph2_rpt_ceiling_ratio_semantics.md](315_ph2_rpt_ceiling_ratio_semantics.md) | Ceiling / Ratio Semantics 調査報告: ceiling 正常動作確認 + post-ceiling multiplier による ratio 膨張 + effective_offset_used 信頼不可 |
| 316 | fix | [316_ph2_fix_self_review_and_observation.md](316_ph2_fix_self_review_and_observation.md) | セルフレビュー修正 6 件 + 317# 観測比較実験結果 + 先行施策 7 件 (S-1 trending_up boost, S-3 mid_at_order, S-7 テール分析) |
| 317 | rpt | [317_ph2_rpt_observation_experiment.md](317_ph2_rpt_observation_experiment.md) | 観測比較実験報告: 全データ n=2575 + dcc3064 n=16(不足) + 構造的課題 6 件特定 |
| 318 | fix | [318_ph2_fix_307_f5_none_regime.md](318_ph2_fix_307_f5_none_regime.md) | 307# F5 none レジーム問題修正: Passive MM 死亡バグ修正 + regime_at_order/observation_count 追加 + 分析スクリプト改善 |
| 319 | fix/rpt | [319_ph2_fix_deep_analysis_and_s1_s3_s5_s7.md](319_ph2_fix_deep_analysis_and_s1_s3_s5_s7.md) | 深層分析: sell パイプライン全死 C-1 発見 + S-1 修正版 (boost 3.0→4.0) + S-3 mid_at_order + S-5 YAML 整合 + S-7 テール分析 |
| 320 | fix | [320_ph2_fix_c1_side_specific_ceiling.md](320_ph2_fix_c1_side_specific_ceiling.md) | C-1 根本対策: サイド別 ceiling (sell 0.15→0.50) + executor ×4.0→×1.5 + パイプライン 12+ パラメータ復活 + dcc3064 暫定評価 |
| 321 | fix | [321_ph2_fix_critical_yaml_parse_and_tasks.md](321_ph2_fix_critical_yaml_parse_and_tasks.md) | CRITICAL: 320# sell ceiling YAML 未パース修正 + M-3 consecutive 10→5 + M-5 offset_bps 2→1 + H-1 skip_rate 矛盾 + God Object 分割検討 |
| 322 | refactor | [322_phg_refactor_maker_price_god_object_split.md](322_phg_refactor_maker_price_god_object_split.md) | God Object 分割: maker_price.py 1,692→996 行 (3 Mixin 抽出) |
| 323 | refactor | [323_executor_split_improvements.md](323_executor_split_improvements.md) | God Object 分割: fill_cycle_executor.py 1,502→1,090 行 (2 Mixin 抽出) |
| 324 | fix | [324_phg_fix_residual_tasks_and_regime_reuse.md](324_phg_fix_residual_tasks_and_regime_reuse.md) | 未達事項消化: M-2 per-side counter + L-3/L-4 YAML 文書化 + Regime RSI 統合 |
| 325 | refactor | [325_phg_refactor_orchestrator_god_object_split.md](325_phg_refactor_orchestrator_god_object_split.md) | God Object 分割: fill_loop_orchestrator.py 2,849→1,594 行 (3 Mixin 抽出) |
| 326 | fix | [326_phg_fix_mixin_audit_and_encapsulation.md](326_phg_fix_mixin_audit_and_encapsulation.md) | 325# Mixin Audit: 型安全修正 + DD guard warmup 委譲 + DRY + 未使用 import 削除 |
| 327 | fix | [327_phg_fix_proactive_bug_hunt.md](327_phg_fix_proactive_bug_hunt.md) | Proactive bug fix: loss_cap_ratio ZeroDivisionError 防止 + ファイルハンドルリーク修正 |
| 328 | rpt | [328_phg_rpt_task_audit_and_god_object_analysis.md](328_phg_rpt_task_audit_and_god_object_analysis.md) | タスク棚卸し 47 件 + fill_config.py / orchestrator God Object 分割戦略 |
| 329 | refactor | [329_phg_refactor_fill_config_god_object_split.md](329_phg_refactor_fill_config_god_object_split.md) | fill_config.py 2046→724 行 God Object 分割 (4 ファイル) |
| 330 | refactor | [330_phg_refactor_orchestrator_pre_cycle_and_bugfixes.md](330_phg_refactor_orchestrator_pre_cycle_and_bugfixes.md) | run_continuous pre-cycle 抽出 (1595→1223 行) + σ floor + ゼロ除算ガード |
| 333 | rpt | [333_ph2_rpt_dcc3064_sha_isolated_deep_dive.md](333_ph2_rpt_dcc3064_sha_isolated_deep_dive.md) | dcc3064 SHA 分離分析: 24h n=100 fills, PnL +63.56bps, AB FAIL (sell p10 僅差 + buy fill_rate 壊滅), buy_dynamic_kill T-1 提起 |
| 334 | rev | [334_ph2_rev_313_333_profitability_design_market_review.md](334_ph2_rev_313_333_profitability_design_market_review.md) | 313#–333# 横断レビュー: 収益性最優先で buy suppressor 過剰, side-specific ceiling 評価, refactor 凍結線引き, ranging/trending 分離を整理 |
| 335 | rev | [335_ph2_gemini_31_pro_review_314_334_comprehensive_audit.md](335_ph2_gemini_31_pro_review_314_334_comprehensive_audit.md) | Gemini 3.1 Pro 総括レビュー: 生存者バイアス警告→自己訂正, buy kill -0.8bps=過敏スプリンクラー, P0 緩和必須 |
| 336 | rev | [336_ph2_rev_334_335_claims_validation_and_measures.md](336_ph2_rev_334_335_claims_validation_and_measures.md) | 334#/335# 主張検証: カスケード増幅メカニズム解明, T-1〜T-5 施策策定, YAML-only Phase 1 即時実行計画 |
| 336 | fix | — | drift fix: fill_config.py 12コードデフォルトをYAML値に整合 (`a3e2750`) |
| 336 | fix | — | drift fix: CycleGateAggregator.UNKNOWN_REGIME_MAX_CONSECUTIVE + test_157/196/197/220 assertions (`a35e881`) |
| 336 | impl | [analysis/333_sha_isolated_analysis.py](../../analysis/333_sha_isolated_analysis.py) | 333# SHA分析スクリプト promotion (334# P1-5): CLI汎用化 + JSON出力 (`31883c0`) |
| 336 | test | — | YAML↔Code drift prevention test: 125-field allowlist, God Object growth monitor (`0cbf7b9`) |
| 336 | cleanup | — | temp/ 36ファイル整理: 11→archived/, 5→tools/, 残り削除; root txt 11件削除 (`f468711`) |
| 337 | rpt | [337_ph2_rpt_sell_side_degradation_countermeasures.md](337_ph2_rpt_sell_side_degradation_countermeasures.md) | Sell-side 損益悪化分析 & 対策設計: buy 緩和後の sell 崩壊仮説, rolling-50 自己強化ループ, threshold/relaxation 提案 |
| 338 | rev | [338_ph2_rev_337_sell_side_countermeasure_audit.md](338_ph2_rev_337_sell_side_countermeasure_audit.md) | 337# レビュー: inv_relaxation 符号逆転, metric 混在, sell relief 重複, filter stack 過小評価, threshold overfit を指摘 |
| 339 | rev | [339_ph2_gemini_31_pro_review_337_338_critical_audit.md](339_ph2_gemini_31_pro_review_337_338_critical_audit.md) | Gemini 3.1 Pro: 338# 符号逆転バグ全面同意, 二重緩和ルート整理要請, forced 完全除外ロールバック提案 |
| 340 | resp | [340_ph2_resp_338_339_sign_fix_and_finding_review.md](340_ph2_resp_338_339_sign_fix_and_finding_review.md) | CRITICAL: threshold_offset_bps 符号逆転修正 (286#以降), テスト assertion 逆転修正, 全7 Finding 妥当性判定 |
| 341 | impl | [341_ph2_threshold_revert_and_horizontal_analysis.md](341_ph2_threshold_revert_and_horizontal_analysis.md) | 閾値復元: 336#/337# calibration は符号バグ前提→sell/buy とも pre-336# 値に revert, 横展開チェック |
| 342 | rpt | [342_ph2_design_and_market_theory_deep_investigation.md](342_ph2_design_and_market_theory_deep_investigation.md) | 設計・市場理論面の深掘り調査: forced PnL 処理, inv_bypass 不連続, skip_gate/kill 二重抑制, EWMA 化, sell wait 非対称 |
| 343 | impl | [343_ph2_impl_p1_improvements.md](343_ph2_impl_p1_improvements.md) | (A) forced fill downweight 0.5, (B) sell forced KPI 分離, (C) skip_gate/kill release grace window, (D) regime_min_confidence default sync, (E) getattr→直接参照 |
| 344 | impl | [344_ph2_impl_improvements.md](344_ph2_impl_improvements.md) | (A) velocity_ema_alpha 0.3, (B) ranging_obi_asymmetry 0.3, (C) inv_decay_tau 1800, (D) inv_bypass gradual 化, (E) EWMA mode |
| 345 | fix | [345_proactive_fixes.md](345_proactive_fixes.md) | (A) warmup forced_fill_pnl_downweight 整合, (B) CircuitBreaker sync Py3.12+ 互換, (C) 324# M-1/L-2 完了反映 |
| 346 | rpt | [346_phg_rpt_test_coverage_and_tail_loss_analysis.md](346_phg_rpt_test_coverage_and_tail_loss_analysis.md) | テストカバレッジ拡充 (fill_config_validation 67 + pre_order_adjustments 18) + S-7 テール損失分析スクリプト (33) = 118 tests |
| 347 | rpt | [347_ph2_rpt_min_lot_constraint_analysis.md](347_ph2_rpt_min_lot_constraint_analysis.md) | 1mBTC 最低ロット制約分析: 6 死蔵メカニズム, satoshi 精度, Phase 0-4 ロードマップ |
| 348 | impl | [348_ph2_impl_satoshi_and_balance_forced_removal.md](348_ph2_impl_satoshi_and_balance_forced_removal.md) | Satoshi 精度化 (lot_step 1e-8) + balance_forced 全面撤廃 (~20 ソース, ~30 テスト) |
| 349 | rpt | [349_phg_refactor_analysis_dedup.md](349_phg_refactor_analysis_dedup.md) | 分析ツール整理 + 重複削減 + EWMA 永続化/seed/TIME LIMIT 修正 + 再起動後/修正後トレード分析 |
| 350 | rev | [350_ph2_rev_339_349_bugfix_trade_multifaceted_review.md](350_ph2_rev_339_349_bugfix_trade_multifaceted_review.md) | 339#–349# 総合レビュー: kill 制御正常化の評価, EWMA fallback 数学的不整合, balance_forced 撤廃後の観測課題, buy_ranging 本丸論 |
| 351 | rev | [351_ph2_gemini_31_pro_review_340_350_profitability_audit.md](351_ph2_gemini_31_pro_review_340_350_profitability_audit.md) | Gemini 3.1 Pro レビュー: EWMA rebuild look-ahead bias, effective_threshold 無視, buy_ranging 市場理論 3 盲点, 時間減衰 EWMA 提案 |
| 352 | impl | [352_ph2_impl_ewma_strict_buy_ranging_deep_dive.md](352_ph2_impl_ewma_strict_buy_ranging_deep_dive.md) | 350#/351# P0 実装: EWMA rebuild 厳密化 (history[0] seed), TIME LIMIT effective_threshold 修正, テスト 13→18, buy_ranging 深堀り分析 |
| 353 | impl | [353_ph2_impl_vpin_asymmetric_buy_boost.md](353_ph2_impl_vpin_asymmetric_buy_boost.md) | VPIN 非対称 buy boost + EWMA 時間減衰 (351# 盲点1-2) + buy 防御パラメータ調整, テスト 25→29+25 |
| 354 | fix | [354_phg_fix_code_quality_sweep.md](354_phg_fix_code_quality_sweep.md) | コード品質改善: 型安全性 (mypy 12→0), DRY (parser 共通化), サイレント例外修正, hot-reload drift 修正 |
| 355 | impl | [355_ph2_impl_api_dedup_and_ph3_blockers.md](355_ph2_impl_api_dedup_and_ph3_blockers.md) | SkipGate API 呼出し重複排除 (OB+Trades prefetch 共有, 200-500ms/cycle 削減) + ph3 ブロッカー 5 件特定 |
| 356 | plan | [356_ph3_plan_sac_training.md](356_ph3_plan_sac_training.md) | ph3 SAC 訓練計画: vXXX 資産活用 + B1-B5 ブロッカー解消 + 4-7 日ロードマップ |
| 357 | rpt | [357_ph3_rpt_sac_asset_inventory.md](357_ph3_rpt_sac_asset_inventory.md) | ph3 SAC 資産インベントリ: v430-v459 設定・モデル・特徴量・環境の網羅的棚卸し |
| 358 | impl | [358_ph3_impl_g2_sac_blockers.md](358_ph3_impl_g2_sac_blockers.md) | G2 SAC ブロッカー実装: B1 YAML + B3 feature_columns 注入 + B4 multi-seed + G2 gate 判定 |
| 359 | impl | [359_ph2_impl_g2_gate_metrics_fix.md](359_ph2_impl_g2_gate_metrics_fix.md) | ph2 G2 gate メトリクス修正: checkpoint ROI + eval gross_roi + セルフレビュー品質改善 (ph3 先行準備) |
| 360 | rpt | [360_ph2_rpt_fill_test_analysis.md](360_ph2_rpt_fill_test_analysis.md) | ph2 Fill Test ログ分析: 5日間集計 + SDK/BDK 影響 + クラッシュ分析 + G1.1 ゲート判定 + 改善提案 |
| 361 | rev | [361_ph2_rev_351_360_ph3_readiness_and_profit_priority_review.md](361_ph2_rev_351_360_ph3_readiness_and_profit_priority_review.md) | 351#–360# Codex レビュー: ph3 gate in-sample 問題, E2 `ic_seed_std` 空洞化, ph2 live 不整合, mixed-SHA 補正, profit-first 次手整理 |
| 362 | rev | [362_gemini_final_answer_profit_architecture.md](362_gemini_final_answer_profit_architecture.md) | Gemini 3.1 Pro レビュー: 収益最大化アーキテクチャ + Sidecar 提案 |
| 363 | rev | [363_ph2_rev_361_362_review_validation.md](363_ph2_rev_361_362_review_validation.md) | 361/362 AI レビュー妥当性検証: コード検証 + 統合アクション + Codex メモリ診断プロンプト |
| 364 | rpt | [364_ph2_sha_analysis_tune3.md](364_ph2_sha_analysis_tune3.md) | SHA別分析レポート + TUNE-3 SDK閾値緩和 |
| 365 | plan | [365_ph3_design_sac_sidecar_and_env_obstacles.md](365_ph3_design_sac_sidecar_and_env_obstacles.md) | ph3 設計: SAC Sidecar アーキテクチャ + HeavyTradingEnv 阻害要因分析 |
| 366 | rpt | [366_ph3_market_theory_proposals_and_tech_debt.md](366_ph3_market_theory_proposals_and_tech_debt.md) | 市場理論システム提案 + 技術課題 + 計算高速化 |
| 367 | rpt | [367_ph2_deep_analysis_360_ai_review.md](367_ph2_deep_analysis_360_ai_review.md) | ph2 深堀り分析: 360# AIレビュー準備 |
| 368 | rev | [368_ph2_rev_362_367_sac_linkage_and_urgent_blockers.md](368_ph2_rev_362_367_sac_linkage_and_urgent_blockers.md) | 362#-367# レビュー: SAC接続・市場理論・緊急ブロッカー再整理 |
| 369 | rev | [369_gemini_verification_and_next_steps.md](369_gemini_verification_and_next_steps.md) | Gemini 3.1 Pro: Codex 368# 徹底検証と収益再構築への緊急指令 (362#-368#) |
| 370 | plan | [370_ph2_pivot_368_369_review_acceptance.md](370_ph2_pivot_368_369_review_acceptance.md) | ph2 方向転換: 368#/369# レビュー受容 + 緊急対応 |
| 371 | impl | [371_ph2_366_market_theory_wiring.md](371_ph2_366_market_theory_wiring.md) | ph2 366# 市場理論システム配線 (M2-M5) |
| 372 | plan | [372_ph2_dust_sweep_refinement_and_sac_wiring_plan.md](372_ph2_dust_sweep_refinement_and_sac_wiring_plan.md) | ph2 Dust Sweep 洗練 + SAC sidecar 配線計画 |
| 372 | rpt | [372_ph3_audit_report.md](372_ph3_audit_report.md) | Phase 3 Post-Implementation Audit Report (**同番号**: ph2/ph3 別文書) |
| 373 | fix | [373_phg_fix_post_sac_audit_and_safety_hardening.md](373_phg_fix_post_sac_audit_and_safety_hardening.md) | Post-SAC 安全性監査 & ハードニング: CRITICAL 2件 (order_quantity ゼロ除算防止, balance_checker 例外安全化) + IMPORTANT 7件 (SACRetrainConfig validation, TOCTOU, type:ignore 排除, max_lot clamp, poll error limit) |
| 374 | plan | [374_ph3_design_sac_continuous_value_and_market_theory_integration.md](374_ph3_design_sac_continuous_value_and_market_theory_integration.md) | SAC 連続値活用 + 市場理論統合 4 Phase 設計: Proportional boost, Regime-Aware obs, Parameter modulation, Closed-loop reward |
| 375 | rev | [375_ph3_rev_369_374_profit_first_and_sac_scope_review.md](375_ph3_rev_369_374_profit_first_and_sac_scope_review.md) | 369#-374# Codex レビュー: profit-first 再整理, 374# の縮小GO/HOLD/NO-GO判定, sidecar 数値感・obs契約・closed-loop reward の危険性点検 |
| 376 | rev | [376_gemini_ph3_comprehensive_review_and_v456_revival.md](376_gemini_ph3_comprehensive_review_and_v456_revival.md) | Gemini 3.1 Pro: 374#–375# 深層レビュー + v456/v454 資産復活 + データパリティ是正 |
| 377 | plan | [377_ph3_unified_direction_course_correction.md](377_ph3_unified_direction_course_correction.md) | 375#/376# レビュー統合方針: 軌道修正 + データパリティロードマップ + FastIntradayEnvV456 評価 |
| 374 | impl | — | Phase 3.1 Proportional Boost 実装: `compute_sidecar_offset_bps_v2` + `_shaping_fn` + FillConfig 5フィールド + YAML parser + hot-reload + CycleGateAggregator v1/v2 切替 + validation 横展開 + 55 tests |
| 379 | fix | — | watchdog UNKNOWN→NOT_RUNNING エスカレーション修正 + fill_test 再起動 |
| 380 | analysis | [380_fill_test_trading_improvement_analysis.md](380_fill_test_trading_improvement_analysis.md) | Fill Test クリーンデータ取引改善分析 (Rev.2): sell EV破綻 (30s=-0.12bps), 時間帯依存性, SkipGate非単調性, offset ceiling問題 — 8施策提案 |
| 381 | review | [381_fill_test_analysis_codex_review.md](381_fill_test_analysis_codex_review.md) | Gemini Codex Review: Ghost Metric Paradox (forced_buy_delay撤廃済), データepoch汚染, JST/UTC混同 — 380# Rev.2 + P0/P1 YAML適用 |
| 382 | rev | [382_ph3_rev_379_sb3_stub_and_g2_pipeline_review.md](382_ph3_rev_379_sb3_stub_and_g2_pipeline_review.md) | 379# SB3スタブ修正・G2 SAC パイプライン再点検: checkpoint eval の env 汚染, OOS 評価の同一区間反復, seed failure PASS 漏れ, scaler 契約 drift を指摘 |
| 383 | rev | [383_gemini_sb3_pipeline_and_codex_review.md](383_gemini_sb3_pipeline_and_codex_review.md) | Gemini second opinion: SB3 スタブ purge, G2/OOS 評価経路の致命欠陥支持, curriculum/gamma/cost 設計の再提案 |
| 384 | rev | [384_ph3_rev_cross_validation_codex_gemini.md](384_ph3_rev_cross_validation_codex_gemini.md) | Codex/Gemini cross validation: SB3 修正と G2 パイプライン論点の相互検証 |
| 385 | rpt | [385_ph3_rpt_config_audit.md](385_ph3_rpt_config_audit.md) | 設定監査: P0-1~P0-8 CRITICAL 6件修正 (threshold統一, reward_scaling, YAML→env伝播, behavior_optimization, マージ順序, from_dict) |
| 386 | plan | [386_ph3_plan_fee_fix_gamma_experiments.md](386_ph3_plan_fee_fix_gamma_experiments.md) | 次ステップ計画: 手数料矛盾修正 + Gamma比較実験 + 科学的アプローチ |
| 387 | rpt | [387_ph3_rpt_reward_analysis.md](387_ph3_rpt_reward_analysis.md) | 報酬関数分析: ペナルティ過大問題, 4実験比較 (baseline/γ=0.95/reward-tuned/warm-start), **G2 PASS達成** |
| 388 | plan | [388_ph3_plan_g3_pnl_preparation.md](388_ph3_plan_g3_pnl_preparation.md) | G3-pnl Gate 準備計画: 評価パイプライン構築, PF/MaxDD/Sharpe 計測, FAIL 時対策 |
| 389 | rev | [389_ph3_rev_383_388_profit_systems_market_review.md](389_ph3_rev_383_388_profit_systems_market_review.md) | 383#-388# Profit-first 再整理: G2 PASS の artifact/curr threshold 整合, reward-PnL 逆相関, G3 二重実装回避, friction 未考慮リスクを点検 |
| 390 | rev | [390_comprehensive_review_and_second_opinion.md](390_comprehensive_review_and_second_opinion.md) | Gemini second opinion: 384#-389# 総合レビュー, パイプライン修復の評価, G3 DRY原則警告, 500K/γ=0.99 提案 |
| 391 | rpt | [391_fill_test_deep_dive_root_cause.md](391_fill_test_deep_dive_root_cause.md) | Fill Test 根本原因分析: sell tail 2.7x, EV confident-wrong, rapid-fire −0.43bps, offset ceiling 59%張付き, **P0施策で+100bps改善余地** |
| 392 | resp | [392_ph3_resp_389_390_review_validity.md](392_ph3_resp_389_390_review_validity.md) | 389#/390# レビュー妥当性評価: P0-3 reward-PnL逆相関(corr=-0.38)を最重要指摘として採用, G3 checker新設撤回, 500K提案は時期尚早 |
| 393 | rev | [393_ph3_rev_391_fill_test_root_cause_validation.md](393_ph3_rev_391_fill_test_root_cause_validation.md) | 391# 妥当性監査: mixed-SHA汚染, headline再現不能, EV=price-direction誤読, rapid-fire 5分禁止の過剰さを補正 |
| 394 | rev | [394_ph3_rev_391_393_fill_test_second_opinion.md](394_ph3_rev_391_393_fill_test_second_opinion.md) | Gemini second opinion: 393#全面支持, EV confident-wrong是正, HFT哲学堅持, Ghost Metric再発警告, 雷火豊の符合 |
| 395 | rpt | [395_ph3_rev_393_394_empirical_validation.md](395_ph3_rev_393_394_empirical_validation.md) | **SHA-fenced実証**: 393#/394#の全指摘をコード+データで裏付け。391#因果推論5本中4本がcurrentSHAで消失。構造的問題はconfidence[0.7,0.9)のみ |
| 397 | impl | [397_ph3_confidence_band_jst23h_fix.md](397_ph3_confidence_band_jst23h_fix.md) | confidence[0.7,0.9) offset×1.2 boost + JST 23h sell_hour_boost 1.3→1.5 / skip_gate 0.3→0.5 強化 |
| 398 | fix | [398_ph3_g3_pipeline_selfreview_fix.md](398_ph3_g3_pipeline_selfreview_fix.md) | G3パイプライン selfreview 修正 |
| 399 | impl | [399_gate_judgment_consolidation.md](399_gate_judgment_consolidation.md) | ゲート判定統合 |
| 400 | rpt | [400_reward_clean_analysis.md](400_reward_clean_analysis.md) | Reward Clean 分析 |
| 401 | rpt | [401_deep_investigation_findings.md](401_deep_investigation_findings.md) | 深層調査知見 |
| 402 | rpt | [402_time_guard_root_cause_and_397_review.md](402_time_guard_root_cause_and_397_review.md) | 時間帯パフォーマンス根本原因分析: AS率が最大ドライバー、JST 09h sell tail壊滅、confidence≥0.9が全データ最悪(-1.69bps)、397# sell ceiling制約で実質無効 |
| 403 | rev | [403_ph3_rev_402_time_guard_fill_test_multifaceted_review.md](403_ph3_rev_402_time_guard_fill_test_multifaceted_review.md) | 402# 再点検: mixed-SHA注意, AS rawは事後ラベル, 397#の真の制約はsell ceilingよりglobal 0.30 cap, hard skip解除は選択バイアスあり |
| 404 | rev | [404_ph3_rev_402_403_time_guard_second_opinion.md](404_ph3_rev_402_403_time_guard_second_opinion.md) | Gemini セカンドオピニオン: _scale_offset_ratio 中間キャップのデッドロック証明, AS予測モデルへの統合推奨 |
| 405 | impl | [405_offset_ceiling_pipeline_fix_selfreview.md](405_offset_ceiling_pipeline_fix_selfreview.md) | Offset Ceiling Pipeline Fix: sell側中間キャップを0.30→0.50に拡大, 14箇所修正, 14新規テスト |
| 406 | rev | [406_selfreview_400_405_deep_dive.md](406_selfreview_400_405_deep_dive.md) | セルフレビュー: 400#–405# 深堀り分析 — session037ゴーストファイル問題発見, CHANGELOG上書き修正, 未コミット成果物整理 |
| 407 | fix/perf | [407_phg_fix_ghost_file_cleanup.md](407_phg_fix_ghost_file_cleanup.md) | Ghost File Cleanup: 71ファイル再追跡, デッドコード11ファイル削除, S4タプルバグ修正, P1設定キャッシュ, P3二重GC統合, 11新規テスト |
| 408 | rpt/fix | [408_phg_rpt_dead_code_analysis.md](408_phg_rpt_dead_code_analysis.md) | F6 OOS best-checkpoint実装 + F4デフォルト統一 + B1-B5盲点修正 + Codexデッドコード調査 (RewardCalculator 2252行/50メソッド God Object 分割提案) |
| 409 | rpt/fix | [409_phg_rpt_broad_discovery_scan.md](409_phg_rpt_broad_discovery_scan.md) | 広域課題スキャン (6カテゴリ24項目) + Codex T1-T16実行 (17項目対処) + G3 reward_profit_corr gate追加 + SAC reward-clean **G2+G3 PASS** (40テスト追加) |
| 410 | rpt | [410_ph4_rpt_g3_pass_deep_dive.md](410_ph4_rpt_g3_pass_deep_dive.md) | G3 PASS 深堀り分析: seed456 corr=-0.20 根本分析, slippage 耐性推算 (PF≈0.99), 100K検証ポイント定義, G3.1-stress 橋渡し |
| 411 | rpt/impl | [411_phg_rpt_seed_sensitivity_analysis.md](411_phg_rpt_seed_sensitivity_analysis.md) | Seed感度構造分析: 5乱数注入箇所特定, 過パラメータ化10.8x定量化, M1-M5対策提案, policy_kwargs転送実装, reward-clean-small config作成 |
| 412 | rev | [412_phg_rev_410_411_sac_hparam_seed_review.md](412_phg_rev_410_411_sac_hparam_seed_review.md) | 410-411レビュー: reward_profit_corr解釈補正, [128,128]は本命候補だが最適断定は保留, weight_decay/learning_starts/M5の採用優先度整理, actor-critic非対称化提案 |
| 413 | rev | [413_phg_rev_410_412_gemini_second_opinion_and_self_review.md](413_phg_rev_410_412_gemini_second_opinion_and_self_review.md) | Geminiセカンドオピニオン: レジーム多様性観点, Q値平滑化リスク, 初期経験偏向, Reward Hacking仮説, state_dict平均危険性 |
| 414 | rpt | [414_ph4_rpt_20k_attribution_experiment.md](414_ph4_rpt_20k_attribution_experiment.md) | 20K Attribution実験 A/B/C 深堀り分析: M1(net_arch縮小)20Kで棄却, M2(weight_decay)corr改善だがROI82%減, Baseline[256,256]が最安定, 100K A単独推奨 |
| 415 | rpt | [415_ph2_rpt_fill_test_log_analysis.md](415_ph2_rpt_fill_test_log_analysis.md) | fill_test ログ分析 (3/11-3/14): sell_dynamic_kill爆発, 405# offset fix後のsell AS悪化仮説, SkipGate/VG/時間帯の整理, 改善案 P0-P3 |
| 416 | rev | [416_ph2_rev_415_fill_test_improvement_and_blindspots.md](416_ph2_rev_415_fill_test_improvement_and_blindspots.md) | 415# レビュー: pre-405から存在するpost-ceiling offset leakを特定, SDKをroute-to-kill相互作用として再定義, SkipGate提案の一部は既実装, same-SHA再集計を最優先化 |
| 417 | rev | [417_ph2_rev_415_416_gemini_second_opinion_and_re_aggregation.md](417_ph2_rev_415_416_gemini_second_opinion_and_re_aggregation.md) | Gemini セカンドオピニオン & 再集計: 415#/416# fill_test ログ深堀り |
| 418 | rpt | [418_ph2_rpt_execution_final_clamp_deadlock.md](418_ph2_rpt_execution_final_clamp_deadlock.md) | Execution Final Clamp + Route-to-Kill Deadlock修正: post-ceiling offset leak (×4.35倍膨張) + buy→sell デッドスピラル (416#/417# 起票) |
| 419 | rev | [419_ph2_rev_421_final_clamp_deadlock.md](419_ph2_rev_421_final_clamp_deadlock.md) | Self-Review: 421# Execution Final Clamp + Route-to-Kill Deadlock (9項目検査, 不備修正) |
| 420 | impl | [420_ph2_impl_observability_deferred_items.md](420_ph2_impl_observability_deferred_items.md) | 416#/417# 先送り事項: 可観測性改善 + hard_skip_mult 有効化 |
| 421 | impl | [421_ph2_impl_execution_final_clamp_and_route_to_kill.md](421_ph2_impl_execution_final_clamp_and_route_to_kill.md) | Execution Final Clamp 実装: SHA `4aa779d27`, DRYヘルパー, 10テスト追加, 419# self-review反映済 |
| 422 | rpt | [422_ph4_rpt_100k_forensic_analysis.md](422_ph4_rpt_100k_forensic_analysis.md) | 100K Forensic分析: G3 FAIL根因=val_ratio交絡 (20K:0.02→100K:0.20), F6 best_model未使用, 5000-step OOS切詰め盲点 |
| 423 | rev | [423_ph4_rev_422_100k_training_review_and_next_options.md](423_ph4_rev_422_100k_training_review_and_next_options.md) | 422# 再点検: val_ratio単独犯ではなく評価設計+レジーム脆弱性の複合と整理, best_model差替え単独では不足, walk-forward / full-OOS比較 / sidecar化を提案 |
| 424 | rev | [424_ph4_rev_422_423_gemini_second_opinion_and_self_review.md](424_ph4_rev_422_423_gemini_second_opinion_and_self_review.md) | Gemini second opinion: 時間スケールのミスマッチ, 5K step=OOS先頭への過学習, SAC万能主義限界, セルフレビュー付き |
| 425 | rev | [425_ph4_rev_423_424_validity_and_blind_spots.md](425_ph4_rev_423_424_validity_and_blind_spots.md) | 423#/424# 妥当性評価: val_ratio=0.02でもG3 FAIL 3/6, reward設計×val_ratio交互作用, S1'併走提案, corr paradox |
| 426 | rpt | [426_ph4_rpt_s1_s1prime_val_ratio_experiment_results.md](426_ph4_rpt_s1_s1prime_val_ratio_experiment_results.md) | S1/S1' val_ratio分離実験: S1 marginal FAIL (pf 1.049), S1' 構造的FAIL (pf 1.031). mid期崩壊パターン6/8seed. 根因=20K SACのレジーム汎化限界, val_ratio=0.02は隠蔽 |
| 427 | plan | [427_global_quant_cultures_and_sidecar_plan.md](427_global_quant_cultures_and_sidecar_plan.md) | 世界のクオンツ文化とSidecar/Final Clamp構想: 日中ハイブリッド方針, 実装計画案 |
| 428 | rev | [428_ph4_rev_427_global_direction_and_sidecar_review.md](428_ph4_rev_427_global_direction_and_sidecar_review.md) | 427# 再整理: 文化論は補助線に留め、既存Sidecar/Final Clampの live証明と効能検証を最優先化。walk-forward/ztb再利用, ML責務限定, clamp観測強化を提案 |
| 429 | rev | [429_ph4_rev_428_actionable_sidecar_roadmap.md](429_ph4_rev_428_actionable_sidecar_roadmap.md) | 428# アクション化: Sidecar蘇生Step1-4, Clamp observabilityフィールド追加, SAC責務縮小。ただし修辞過剰・実態乖離あり |
| 430 | rev | [430_ph4_rev_428_429_validity_and_implementation_audit.md](430_ph4_rev_428_429_validity_and_implementation_audit.md) | 428#/429# 妥当性評価+既存実装監査: retrain_scheduler完全実装済み発見, Sidecar v2(比例変換)存在, fill_recordsフィールド既存, 統合ロードマップ |
| 431 | impl | [431_ph2_impl_clamp_observability_and_data_analysis.md](431_ph2_impl_clamp_observability_and_data_analysis.md) | Clamp Observability実装 + fill data分析: ceiling clamp rate 100%発見, EV相関0.017, offset chain装飾的, 428#/430# P3完了 |
| 432 | rpt | [432_ph2_rpt_fill_records_deep_dive_analysis.md](432_ph2_rpt_fill_records_deep_dive_analysis.md) | Fill Records深掘り分析: Skip Gate相関 -0.010, buy+ranging PF 0.766, reprice悪化, balance_forcedの短期悪化/中期改善 |
| 433 | plan | [433_ph4_advanced_microstructure_edge_ideas.md](433_ph4_advanced_microstructure_edge_ideas.md) | Advanced Microstructure Edge Ideas: toxicity veto, BitFlyer lead-lag, queue heuristic, committee sidecar の次期バックログ案 |
| 434 | rev | [434_ph2_ph4_rev_426_432_433_multifaceted_validation.md](434_ph2_ph4_rev_426_432_433_multifaceted_validation.md) | 426#/432#/433# 横断レビュー: SAC OOS・live fill・sidecar backlog を profit-first に再整列。toxicity veto 優先, ceiling再測定, walk-forward化を提案 |
| 435 | rev | [435_ph4_gemini_second_opinion_on_434_cross_review.md](435_ph4_gemini_second_opinion_on_434_cross_review.md) | Gemini second opinion: 434# 全結論に追認、Ceiling Observability 急務性を強調 |
| 436 | rpt | [436_ph4_rpt_cross_review_synthesis_and_sac_mandate.md](436_ph4_rpt_cross_review_synthesis_and_sac_mandate.md) | 統合検証+SAC責務再定義: 全7資産production-ready検証、Phase 1-3ロードマップ、正誤表 |
| 437 | rev | [437_ph4_rev_432_436_cross_review_and_val_ratio_curve.md](437_ph4_rev_432_436_cross_review_and_val_ratio_curve.md) | 432#-436# 横断レビュー: val_ratio感度曲線完成(崩壊cliff edge 42-84日)、retrain推奨21-30日、Toxicity Veto最優先 |
| 438 | impl | [438_ph2_impl_buy_ranging_defense_triple_config.md](438_ph2_impl_buy_ranging_defense_triple_config.md) | buy+ranging 防御3重設定の実装 |
| 439 | impl | [439_ph4_cross_venue_lead_lag_guard.md](439_ph4_cross_venue_lead_lag_guard.md) | Cross-Venue Lead-Lag Guard: BitFlyer 参照信号基盤構築 (disabled default) |
| 440 | impl | [440_ph4_toxicity_veto_investigation_and_regime_side_offset.md](440_ph4_toxicity_veto_investigation_and_regime_side_offset.md) | Toxicity Veto調査: AS Classifier ROC-AUC≈0.50で棄却、代替としてregime-side offset非対称化を実装 |
| 441 | impl | [441_ph2_impl_sac_retrain_and_ab_offset_comparison.md](441_ph2_impl_sac_retrain_and_ab_offset_comparison.md) | SAC Retrain YAML設定完成 + 440# A/B Offset比較ツール構築、ベースライン取得済 |
| 442 | impl | [442_ph4_impl_cross_venue_activation_ob_depth.md](442_ph4_impl_cross_venue_activation_ob_depth.md) | Cross-Venue有効化 + L5板深度 + Microprice + Depth Imbalance |
| 443 | fix | [443_ph4_fix_cross_venue_threshold_logging.md](443_ph4_fix_cross_venue_threshold_logging.md) | Cross-Venue 閾値修正 + ログ・FillRecord バグ修正 |
| 444 | fix | [444_ph4_fix_cross_venue_threshold_tuning.md](444_ph4_fix_cross_venue_threshold_tuning.md) | Cross-Venue 閾値チューニング + hint=None ログ可視化 |
| 445 | impl | [445_ph4_impl_cross_venue_ema_confidence.md](445_ph4_impl_cross_venue_ema_confidence.md) | Cross-Venue EMA平滑化 + Confidence Scoring: sign_disagree解消 + 比例ブースト |
| 446 | rev | [446_ph2_ph4_rev_435_445_profitability_and_systemic_gaps.md](446_ph2_ph4_rev_435_445_profitability_and_systemic_gaps.md) | 435#-445# 横断レビュー: mixed-SHAでA/B因果が未分離, cross_venueはlive presence薄くno-op多発, 既存 toxicity / deadlock 系の再監査を優先と整理 |
| 447 | rev | [447_ph4_gemini_second_opinion_and_new_paradigms.md](447_ph4_gemini_second_opinion_and_new_paradigms.md) | Gemini Second Opinion: 446# 検証確認 + EMAねじれ自己批判 + 新パラダイム提案 (Inventory Sponging / Micro-Timeout / Spread Shadowing) |
| 448 | resp | [448_ph4_resp_446_447_ema_spread_fix_and_noop_visibility.md](448_ph4_resp_446_447_ema_spread_fix_and_noop_visibility.md) | 446/447# レビュー応答: F3 EMA spread 混線修正 + F2 cap_hit no-op 可視化 |
| 449 | impl | [449_ph4_impl_cross_venue_dry_stability_config.md](449_ph4_impl_cross_venue_dry_stability_config.md) | DRY + 安定性: spread_bps 3重計算排除 + confidence_floor/depth_imb_threshold config化 + getattr統一 |
| 450 | rev | [450_ph2_ph4_rev_447_449_validation_and_remaining_gaps.md](450_ph2_ph4_rev_447_449_validation_and_remaining_gaps.md) | 447#-449# 検証レビュー: repo上の修正は確認したが live未反映と FillRecordスキーマ穴が残存、same-SHA再実走と観測整備を優先と整理 |
| 451 | impl | [451_ph2_p04_p12_p13_git_sha_compound_toxicity.md](451_ph2_p04_p12_p13_git_sha_compound_toxicity.md) | P0-4 git_sha フィルタ / P1-2 compound suppression 分析 / P1-3 toxicity_budget 有効化 |
| 452 | plan | [452_ph2_plan_micro_timeout_design.md](452_ph2_plan_micro_timeout_design.md) | Micro-timeouts (TIF Emulation) 詳細設計案 |
| 453 | impl | [453_micro_timeout_implementation.md](453_micro_timeout_implementation.md) | Micro-timeouts (TIF Emulation) 実装結果 (disabled by default) |
| 454 | plan | [454_ph2_plan_uptrend_sell_loss_countermeasures.md](454_ph2_plan_uptrend_sell_loss_countermeasures.md) | 上昇トレンドにおける Sell 損失対策: 10案の深堀り + 3フェーズ推奨実装順序 |
| 455 | rev | [455_ph2_rev_454_uptrend_sell_loss_countermeasures_review.md](455_ph2_rev_454_uptrend_sell_loss_countermeasures_review.md) | 454# Codex レビュー: R3 表現修正 (micro 保護は既存)、S1/S2 再定式化、F-lite→B→H 推奨順序 |
| 456 | plan | [456_ph2_plan_uptrend_sell_centric_paradigm.md](456_ph2_plan_uptrend_sell_centric_paradigm.md) | Sell-Centric パラダイム: 上昇トレンド = seller's market、A Liquidity Mirage / B Dynamic Offloading / C Cross-Venue |
| 457 | rev | [457_ph2_rev_456_sell_centric_paradigm_review.md](457_ph2_rev_456_sell_centric_paradigm_review.md) | 456# レビュー: A-lite のみ抽出 (×1.2-1.8)、B/C 棄却、"sell はプレミアム機会" |
| 458 | impl | [458_ph2_impl_macro_sell_protection.md](458_ph2_impl_macro_sell_protection.md) | マクロ連動 Sell 防御: 455#/456#/457# 統合実装 — Hysteresis + F-lite boost + threshold 引下げ + micro-timeout 連動 + 観測フィールド |
| 459 | impl | [459_ph2_impl_hot_reload_lateral_and_restart_resilience.md](459_ph2_impl_hot_reload_lateral_and_restart_resilience.md) | Hot-Reload 横展開 + 再起動耐性 + ranging_buy ソフト化: _effective_sleep reload 統合, _cleanup_sync 最終状態保存 |

### ph3 — コード整理・SAC (先行調査・一部実装済)

| # | 種別 | ファイル | タイトル | 前倒し理由 |
|---|---|---|---|---|
| 015 | plan | [015_ph3_plan_sac_investigation.md](015_ph3_plan_sac_investigation.md) | SAC 実装調査 & オンライン学習設計 | fill test 待ち時間活用 |
| 018 | rpt/impl | [018_ph3_rpt_perf_memleak.md](018_ph3_rpt_perf_memleak.md) | メモリリーク防止・パフォーマンス最適化 | ブロッカー (OOM リスク) |
| 021 | rpt | [021_ph3_rpt_code_duplication.md](021_ph3_rpt_code_duplication.md) | コード重複 & リファクタリング分析 | 018# 作業中に発見 |
| 063 | impl | [063_ph3_sac_cleanup.md](063_ph3_sac_cleanup.md) | SAC 重複実装の整理 (246行削除) | 衛生管理・ph3 負荷軽減 |
| 108 | fix | [108_ph3_fix_ahead_of_schedule.md](108_ph3_fix_ahead_of_schedule.md) | 018# 残課題前倒し (M5/C3/M1/DUP2) | fill test 待ち時間活用 |
| 109 | fix | [109_ph3_phg_fix_resilience_and_any.md](109_ph3_phg_fix_resilience_and_any.md) | 耐障害性強化 + Any型完全撤去 (H3/032#16,17/036#) | fill test 待ち時間活用 |

### phg — フェーズ横断 (型安全・品質・分析)

| # | 種別 | ファイル | タイトル |
|---|---|---|---|
| 028 | rpt | [028_phg_rpt_online_learning_and_gaps.md](028_phg_rpt_online_learning_and_gaps.md) | 取引中学習の方策検討 + コードベースギャップ分析 |
| 031 | rpt | [031_phg_rpt_fill_test_improvement.md](031_phg_rpt_fill_test_improvement.md) | Fill Test 分析 & 改善 |
| 032 | rpt | [032_phg_rpt_gate_stubs_and_quality.md](032_phg_rpt_gate_stubs_and_quality.md) | Gate スタブ完成・方策 A 実装・品質改善 |
| 033 | rpt | [033_phg_rpt_dynamic_lot_sizing.md](033_phg_rpt_dynamic_lot_sizing.md) | 方策B: 動的ロットサイジング + 安全キャップ |
| 034 | rpt | [034_phg_rpt_action_space_analysis.md](034_phg_rpt_action_space_analysis.md) | エージェント行動空間・執行パラメータ制御の歴史的分析 |
| 036 | plan | [036_phg_plan_any_reduction_preparation.md](036_phg_plan_any_reduction_preparation.md) | Any 削減マスター (計画・進捗・方針 一元化) |
| 037 | rpt/master | [037_phg_rpt_refactoring_session_log.md](037_phg_rpt_refactoring_session_log.md) | リファクタリングセッションログ運用ハブ（036参照） |
| 038 | rpt | [038_phg_rpt_any_cleanup_step3_completion.md](038_phg_rpt_any_cleanup_step3_completion.md) | 統合済み通知 (036へ集約) |
| 042 | fix | [042_phg_fix_fill_test_3bugs_3improvements.md](042_phg_fix_fill_test_3bugs_3improvements.md) | fill test 3バグ修正 + 3追加改善 + ゾンビプロセス発見 |
| 080 | rpt | [080_phg_dedup_and_inheritance.md](080_phg_dedup_and_inheritance.md) | 重複排除 & 継承ベース統合 (~3,000行削減) |
| 081 | fix | [081_phg_deep_scan_bug_memory_fix.md](081_phg_deep_scan_bug_memory_fix.md) | 深層スキャン — 不具合修正 & メモリ効率改善 |
| 111 | rpt | [111_phg_rpt_legacy_asset_research.md](111_phg_rpt_legacy_asset_research.md) | v456–v459 レガシー資産・教訓 調査レポート |
| 112 | rev | [112_phg_rev_111_legacy_asset.md](112_phg_rev_111_legacy_asset.md) | 111# レビュー + 追加提案 + 見落とし補完 |
| 118 | rpt | [118_phg_rpt_backlog_deep_analysis.md](118_phg_rpt_backlog_deep_analysis.md) | 残課題・未検討提案の深掘り考察 (53→39 RESOLVED, §5/§8全件disposition, 特徴量再訓練計画 Appendix F) |
| 158 | rpt | [158_phg_rpt_backlog_audit_and_phase_d_priorities.md](158_phg_rpt_backlog_audit_and_phase_d_priorities.md) | バックログ監査 (118#–157# 横断) + Phase D 優先順位 22 件 + 外部 AI レビュー向け補足 |
| 159 | rev | [159_phg_rev_158_phase_d_backlog_review.md](159_phg_rev_158_phase_d_backlog_review.md) | 158# レビュー: 実装照合・優先度再編・範囲外改善提案 |
| 160 | rpt | [160_phg_rpt_analysis_and_regime_tuning.md](160_phg_rpt_analysis_and_regime_tuning.md) | P0-3/P1-2 分析 + YAML外部化 + regime=None根本修正 + P0-B A/B判定3指標固定 + P0-C trending_down sell実測評価 |
| 161 | impl | [161_phg_impl_code_quality_and_structural_improvements.md](161_phg_impl_code_quality_and_structural_improvements.md) | 複雑性監査 + SIGTERM graceful shutdown + DRY統合 + asyncio安全化 |
| 162 | rpt | [162_phg_rpt_fill_test_10day_log_analysis.md](162_phg_rpt_fill_test_10day_log_analysis.md) | Fill Test 10日間ログ分析: AS率27%, Fill Rate急落, 改善提案 |
| 163 | rpt | [163_phg_rpt_stopgap_measures_catalog.md](163_phg_rpt_stopgap_measures_catalog.md) | 止血施策カタログ: 17 件のストップギャップ措置を文書化 + FillTestRunner mixin 分割 |
| 164 | rpt | [164_phg_rpt_skip_gate_shap_analysis.md](164_phg_rpt_skip_gate_shap_analysis.md) | SkipGate SHAP 特徴量重要度分析: 3 モデル TreeExplainer + Stopgap 退出基準表 |
| 165 | rpt | [165_phg_rpt_as_root_cause_and_velocity_rule.md](165_phg_rpt_as_root_cause_and_velocity_rule.md) | AS Root Cause Analysis + Velocity Skip Rule + Daily Health Report |
| 166 | rpt | [166_phg_rpt_reviewer_response_and_remaining_tasks.md](166_phg_rpt_reviewer_response_and_remaining_tasks.md) | レビュー対応 + 162/163 残課題消化: SR-1~SR-4 安定性修正, ログベース改善観測 |
| 167 | fix | [167_phg_fix_sell_loop_dl4_dl5.md](167_phg_fix_sell_loop_dl4_dl5.md) | sell ループ構造修正 (DL-4/DL-5) + カウンタ永続化: 166# §11.5 全提案を汎用原則で解消 |
| 168 | rpt | [168_phg_rpt_comprehensive_improvement_hodl_vs_trading.md](168_phg_rpt_comprehensive_improvement_hodl_vs_trading.md) | HODL vs Trading 定量比較 + 未検討提案棚卸し + 既存資産活用計画 |
| 331 | rev | [331_phg_review_329_330_self_audit.md](331_phg_review_329_330_self_audit.md) | Self-Review: 329#/330# 自己監査 — BUG-1/2 修正, CycleContext cleanup, validation 追加 |
| 332 | refactor | [332_phg_refactor_run_continuous_phase4.md](332_phg_refactor_run_continuous_phase4.md) | run_continuous Phase 4: Balance/MidCycle Mixin 抽出 (1228→407 行, 908→~80 行) |
| 460 | refactor | [460_phg_refactor_offset_pipeline_extraction.md](460_phg_refactor_offset_pipeline_extraction.md) | run_single_cycle 分割 + 重複排除: offset pipeline 9段チェーン抽出, lot chain 統合, 継承チェーン整理 |
| 461 | refactor/rpt | [461_phg_refactor_skip_gate_split_fill_test_analysis.md](461_phg_refactor_skip_gate_split_fill_test_analysis.md) | skip_gate_evaluator Mixin 分割 (1362→866行) + MAX LINES 3件 + Fill Test 5日間ログ分析 |
| 470 | analysis | [470_phg_analysis_fundamental_root_cause.md](470_phg_analysis_fundamental_root_cause.md) | 根本原因分析: 売り側オフセット設計のセマンティック反転バグ解明 (1,342レコード/192約定) |
| 471 | rev | [471_phg_rev_470_fundamental_root_cause_deep_dive.md](471_phg_rev_470_fundamental_root_cause_deep_dive.md) | 470# レビュー: 根本原因分析の妥当性確認 + 追加深掘り |
| 472 | rev/plan | [472_phg_rev_470_471_phantom_midpoint_and_market_theory.md](472_phg_rev_470_471_phantom_midpoint_and_market_theory.md) | セカンドオピニオン: ファントム・ミッドポイントの数学的解明 + 市場設計の死角 |
| 473 | rev | [473_phg_rev_verification_471_472_claims.md](473_phg_rev_verification_471_472_claims.md) | 検証レポート: 471#/472# 主張のコード・データ照合 |
| 474 | fix | [474_phg_fix_triple_bugfix_and_orphan_cleanup.md](474_phg_fix_triple_bugfix_and_orphan_cleanup.md) | Triple Bugfix: sell パラメータ修正 + orphan process cleanup (retrain_scheduler 8重起動解消) |
| 475 | rev | [475_phg_rev_472_proposals_and_474_selfreview.md](475_phg_rev_472_proposals_and_474_selfreview.md) | 472# 三提案評価 + 474# セルフレビュー + 残課題改修 |
| 476 | fix | [476_phg_fix_dust_sweep_and_truncation_removal.md](476_phg_fix_dust_sweep_and_truncation_removal.md) | Dust sweep 修正 + 0.001 単位切り捨て廃止 + 残高連動ロット |
| 477 | analysis | [477_phg_analysis_suppression_paradox_and_fill_rate.md](477_phg_analysis_suppression_paradox_and_fill_rate.md) | 抑制パラドックス分析: 20層防御 × fill rate 17.7% の構造的矛盾と改善提案 |
| 478 | rev | [478_phg_rev_477_improvements_and_fill_suppression_review.md](478_phg_rev_477_improvements_and_fill_suppression_review.md) | 475-477改善レビュー: コホート混在・条件付き確率構造・資本制約の見落とし |
| 479 | rev | [479_phg_rev_474_478_feasibility_and_liquidity_paradox.md](479_phg_rev_474_478_feasibility_and_liquidity_paradox.md) | Liquidity Paradox 仮説: 価格フィージビリティ制約が真の主犯という立論 |
| 480 | verify | [480_phg_verify_478_479_review_claims.md](480_phg_verify_478_479_review_claims.md) | 478/479レビュー検証 + veto閾値8.0bps + min_spread 700JPY 段階緩和 |
| 481 | fix | [481_phg_fix_veto_spread_nfq_log.md](481_phg_fix_veto_spread_nfq_log.md) | veto 6→8bps + min_spread 1000→700 + NFQ log reason tag |
| 482 | impl | [482_phg_impl_sac_100k_vr02_results_and_deploy.md](482_phg_impl_sac_100k_vr02_results_and_deploy.md) | SAC 100K×4シード最終結果 + Seed456 bestデプロイ + 深掘り分析 |
| 483 | rev | [483_phg_rev_479_482_feasibility_tuning_and_sac_deploy_review.md](483_phg_rev_479_482_feasibility_tuning_and_sac_deploy_review.md) | 479-482レビュー: feasibility緩和妥当、SAC投入は探索段階 |
| 484 | rev | [484_phg_deep_dive_482_sac_selection_and_live_effect_limits.md](484_phg_deep_dive_482_sac_selection_and_live_effect_limits.md) | 深掘り: seed選定不安定、deploy gate緩い、live sidecar効き幅小 |
| 485 | rev | [485_phg_verify_483_sac_paradox_and_sidecar_limits.md](485_phg_verify_483_sac_paradox_and_sidecar_limits.md) | 総括: OOS過適合分析 + sidecar実質無風化 + 寄与分解P0提案 |
| 486 | verify | [486_phg_verify_483_484_485_review_claims.md](486_phg_verify_483_484_485_review_claims.md) | 483-485レビュー検証: 全数値・コード・設定を突合、483/484=A、485=B+ |
| 487 | fix | [doc](487_phg_fix_sac_memory_leak_and_dedup.md) | SAC メモリリーク防止 (gc.collect/model del) + 重複削減 (create_env_from_config/create_sac_model 統合) + P0 sidecar attribution 可観測性 |
| 488 | review | [488_phg_fill_pipeline_deep_review.md](488_phg_fill_pipeline_deep_review.md) | Fill Test パイプライン総合コードレビュー: 33日ログを踏まえ、logging/例外/設定/guard/DRY 観点から根本原因候補を広く棚卸し |
| 489 | rev | [489_phg_rev_488_fill_pipeline_multifaceted_review.md](489_phg_rev_488_fill_pipeline_multifaceted_review.md) | 488レビュー: 有効論点と古くなった指摘を分離し、3/19 NameError・cross_venue veto→NO_FEASIBLE_QUOTE・防御過積載を追加補強 |
| 490 | opinion | [490_phg_second_opinion_architectural_pivot.md](490_phg_second_opinion_architectural_pivot.md) | 488/489レビューへのセカンドオピニオン: 防御過積載を踏まえ、事前ロック型予算管理・複合リスク統合・SAC移管を提案 |
| 491 | fix | [491_session_remediation_deep_dive.md](491_session_remediation_deep_dive.md) | Sidecar復旧, VPIN修正, 再学習装置堅牢性4件, 防御レイヤー定量分析, Composite Risk Score実装(490# Level2) |
| 492 | audit | [492_codex_review_pipeline_audit.md](492_codex_review_pipeline_audit.md) | 491全修正と実運用ログを監査し、slow fill逆選択・二重クランプ・Composite Riskの未実証部分を棚卸し |
| 493 | rev | [493_phg_rev_490_492_profit_first_fill_test_review.md](493_phg_rev_490_492_profit_first_fill_test_review.md) | 490-492レビュー: 新機構導入論を是々非々で整理し、runtime drift・inventory制約・slow fill対策を優先課題として再構成 |
| 494 | rev | [494_phg_verify_493_action_plan_and_market_theory.md](494_phg_verify_493_action_plan_and_market_theory.md) | 493検証+アクションプラン: micro-timeout TTL cut, inventory recovery skew, runtime drift cold restart の3戦術 |
| 495 | fix | [495_retrain_scheduler_crash_resilience.md](495_retrain_scheduler_crash_resilience.md) | Retrain Scheduler crash resilience: auto-restart, loop protection, training timeout, memory leak prevention |
| 496 | impl | [496_review_implementation_and_blind_spots.md](496_review_implementation_and_blind_spots.md) | 493/494レビュー実装: micro-timeout TTL cut, recovery skew, blind spot分析 (ranging_low_vol 16.6%, spread_too_narrow 9.1%) |
| 497 | rpt | [497_config_change_impact_deep_dive.md](497_config_change_impact_deep_dive.md) | 3/10-3/19 config変更の効果検証: micro-timeoutはプラス、ranging hard skipはマイナス、3/19 regression を slow fill / sdk / deadlock で再整理 |
| 498 | fix | [498_hot_reload_lateral_expansion.md](498_hot_reload_lateral_expansion.md) | Hot-reload 対象を 348→409 へ拡張し、micro_timeout / recovery_skew / cross_venue 等 61 field を追加 |
| 499 | fix | [499_hard_loss_cap_crash_loop_fix.md](499_hard_loss_cap_crash_loop_fix.md) | hard_loss_cap crash loop 修正: cumulative_pnl_jpy を UTC 当日スコープへ是正し、resume 即死を停止 |
| 500 | rev | [500_phg_rev_497_499_503_sell_side_breakdown_and_asymmetric_actions.md](500_phg_rev_497_499_503_sell_side_breakdown_and_asymmetric_actions.md) | 497-499/503レビュー: sell 崩壊を fast/slow・offset・kill の非対称構造で再整理し、cross_venue sell常時化を補正 |
| 503 | rpt | [503_sell_buy_pnl_factor_analysis.md](503_sell_buy_pnl_factor_analysis.md) | 3/14-3/20 Buy/Sell 損益要因分析: CV非対称性、ranging集中、fast/slow adverse selection の整理 |
| 504 | rev | [504_phg_rev_502_lib_to_ztb_plan.md](504_phg_rev_502_lib_to_ztb_plan.md) | 502# `scripts/v460/lib` → `ztb` 移行計画レビュー |
| 505 | resp | [505_phg_resp_504_lib_to_ztb_plan_adjustment.md](505_phg_resp_504_lib_to_ztb_plan_adjustment.md) | 504# レビュー応答: lib→ztb 移行計画調整 |
| 506 | verify | [506_phg_verify_500_501_reviews.md](506_phg_verify_500_501_reviews.md) | 500#/501# レビュー検証: フィルレコード (773 fills) + コード解析による独立検証 |
| 507 | fix | [507_ph2_fix_confidence_velocity_demeaning_and_recovery_skew.md](507_ph2_fix_confidence_velocity_demeaning_and_recovery_skew.md) | Confidence/Velocity De-meaning 統一 + Recovery Skew 縮小 |
| 508 | impl | [508_ph2_impl_logging_observability_improvements.md](508_ph2_impl_logging_observability_improvements.md) | ログ・可観測性改善: de-meaning basis_bps/adjusted_spread + sell_age_cap/regime_timeout ログレベル修正 |
| 509 | fix | [509_ph2_fix_sell_age_cap_micro_timeout_guard.md](509_ph2_fix_sell_age_cap_micro_timeout_guard.md) | sell_age_cap × micro_timeout ガード: cumulative cap + stale reprice 残時間チェック |
| 510 | impl | [510_ph2_impl_inv_skew_periodic_summary_vg_reason.md](510_ph2_impl_inv_skew_periodic_summary_vg_reason.md) | inv_skew_factor fill_record 追加 + 周期的サマリ統計 + VG boost 理由粒度向上 |
| 511 | docs | [511_docs_phase_type_prefix_completion.md](511_docs_phase_type_prefix_completion.md) | 504-510 フェーズ・Type 接頭辞補完リネーム + index.md 更新 |
| 512 | impl | [512_ph2_impl_cross_venue_favorable_tighten_dry_summary.md](512_ph2_impl_cross_venue_favorable_tighten_dry_summary.md) | cross-venue favorable-side tighten + DRY microprice + 周期サマリ統合 |
| 513 | audit | [513_ph2_audit_dormant_mechanisms.md](513_ph2_audit_dormant_mechanisms.md) | 実装済み機構の活用度監査 + SAD/MCB YAML化: SAD/PPG/MCB がデッドコード |
| 514 | plan | [514_phg_plan_deferred_docs_refresh_and_carryforward_audit.md](514_phg_plan_deferred_docs_refresh_and_carryforward_audit.md) | deferred docs 棚卸し: v461/後日表現の現状追随方針と更新優先順位 |
| 515 | rpt | [515_fill_test_restart_deep_dive.md](515_fill_test_restart_deep_dive.md) | Fill Test 再起動深掘り: 506#/507# 後の構造的劣化を EV/XV/sell_offset/kill の観点から分析 |
| 516 | rev | [516_phg_rev_501_513_515_multifaceted_validation.md](516_phg_rev_501_513_515_multifaceted_validation.md) | 501-513/515レビュー: runtime不整合・buy過剰抑制・sell toxic fill・cross_venue飽和を是々非々で再整理 |
| 517 | rev | [517_phg_second_opinion_515_516_validation.md](517_phg_second_opinion_515_516_validation.md) | PHG second opinion: 515#事実誤認指摘 + sell AS 23fillsが喪失の単一原因 |
| 518 | rpt | [518_reconciliation_515_516_517_direction_correction.md](518_reconciliation_515_516_517_direction_correction.md) | 方向修正: 515/516/517三文書検証→sell ceiling cap(0.20)が真因特定、P0再編成 |
| 519 | impl | [519_ph2_impl_sell_ceiling_favorable_tighten_sdk_window.md](519_ph2_impl_sell_ceiling_favorable_tighten_sdk_window.md) | sell ceiling 0.25 + favorable_tighten 有効化 + sell_dynamic_kill window 30 |
| 520 | plan | [520_phg_plan_remaining_deferred_actions_screening.md](520_phg_plan_remaining_deferred_actions_screening.md) | remaining deferred 項目の実行可否スクリーニング: 今やるもの / future 維持の分離 |
| 521 | plan | [521_phg_master_deferred_and_architecture_carryforward.md](521_phg_master_deferred_and_architecture_carryforward.md) | deferred docs とコード基本設計の一元化された living document |
| 522 | impl | [522_ph2_impl_balance_forcing_abolish_and_truncation_audit.md](522_ph2_impl_balance_forcing_abolish_and_truncation_audit.md) | balance_switch/recovery_skew/inventory_escape 完全撤廃 + 切り捨て/丸め無効化パラメータ監査 |
| 523 | impl | [523_ph2_impl_double_ceiling_fix_and_dead_code_cleanup.md](523_ph2_impl_double_ceiling_fix_and_dead_code_cleanup.md) | 二重 ceiling 撤廃 (maker_price中間ceiling→offset_pipeline一本化) + 522# dead code cleanup |
| 524 | rpt | [524_phg_rpt_preflight_skip_exceeded_analysis.md](524_phg_rpt_preflight_skip_exceeded_analysis.md) | preflight_skip_exceeded 停止分析: sell注文スタック→両側膠着→open order cancel提案 |
| 525 | rev | [525_phg_rev_520_524_maintainability_and_dedup_review.md](525_phg_rev_520_524_maintainability_and_dedup_review.md) | 520-524レビュー: legacy残置・stale-order cleanup共有化・double ceiling後のdead code整理を提案 |
| 526 | impl | [526_phg_impl_log_observability_and_dead_code.md](526_phg_impl_log_observability_and_dead_code.md) | ログ可観測性改善 (order_id・残高コンテキスト追加) + 525#指摘dead code削除 |
| 527 | impl | [527_phg_impl_jpy_precision_and_silent_except.md](527_phg_impl_jpy_precision_and_silent_except.md) | JPY精度改善 (繊単位:.0f→:.2f、adapter int()→round()) + silent except可観測化 |
| 528 | impl | [528_phg_impl_codex_import_cleanup_and_review_docs.md](528_phg_impl_codex_import_cleanup_and_review_docs.md) | Codex追加import掃除 + review docs整備 |
| 529 | rpt | [529_phg_rpt_fill_test_trade_analysis_20260322.md](529_phg_rpt_fill_test_trade_analysis_20260322.md) | 2026-03-22 fill_test取引別分析: 最悪/最良トレード・CV veto deadlock・final_clamp飽和の整理 |
| 530 | rev | [530_phg_rev_526_529_trade_level_multifaceted_validation.md](530_phg_rev_526_529_trade_level_multifaceted_validation.md) | 526-529レビュー: 数値母集団補正・sell本丸再確認・favorable tighten支持・在庫ゼロdeadlock是正 |
| 531 | rev | [531_phg_second_opinion_526_530_validation.md](531_phg_second_opinion_526_530_validation.md) | セカンドオピニオン: 526-530検証、JSONL独立分析、clamp 100%飽和・CV cap_hit 73%確認 |
| 532 | verify | [532_phg_rev_530_531_blind_spot_verification.md](532_phg_rev_530_531_blind_spot_verification.md) | 530#/531#盲点検証: 時間帯クラスタリング・隠れデッドロック・veto機会損失・fill数乖離 |
| 533 | rev | [533_phg_evaluation_532_blind_spots.md](533_phg_evaluation_532_blind_spots.md) | 532#盲点の市場微視構造検証: 時間帯ハードコード懸念・理論的反論 |
| 534 | impl | [534_phg_impl_veto_deadlock_prevention_and_log_improvements.md](534_phg_impl_veto_deadlock_prevention_and_log_improvements.md) | vetoデッドロック防止(max_consecutive+BTC=0緩和) + log_cycle_no join key + ログ可観測性改善 |
| 535 | impl | [535_phg_impl_preemptive_cv_kill_and_min_spread.md](535_phg_impl_preemptive_cv_kill_and_min_spread.md) | pre-emptive CV kill (532# P1-5) + min_spread_jpy 700→500 (P1-6) |
| 536 | rpt | [536_phg_architecture_simplification_and_reduction.md](536_phg_architecture_simplification_and_reduction.md) | 風水渙を起点にしたアーキテクチャ断捨離シナリオA/B/Cと「何を捨て何を束ねるか」の全体図 |
| 537 | rev | [537_phg_review_536_architecture_simplification.md](537_phg_review_536_architecture_simplification.md) | 536# 検証と拡張: clamp飽和・OFI-Lite・A-S参照・段階ロードマップ化 |
| 538 | rev | [538_phg_second_opinion_536_537_architecture_review.md](538_phg_second_opinion_536_537_architecture_review.md) | 536-537セカンドオピニオン: 現況差分、ceiling先行の危険、composite_risk再位置付け、簡素化指標を補強 |
| 539 | rpt | [539_phg_huan_simplification_scorecard.md](539_phg_huan_simplification_scorecard.md) | 風水渙断捨離スコアカード: 4指標定義(gate数・pipeline段数・clamp率・block理由)+アクション |
| 540 | verify | [540_phg_verify_536_539_pipeline_reality_check.md](540_phg_verify_536_539_pipeline_reality_check.md) | 536-539独立検証: pipeline二重構造(pre-order 14段+executor 9段)発見、spread_adapt主犯特定、Phase 1 YAML施策 |
| 541 | impl | [541_phg_pipeline_optimization_lazy_import_stage_skip.md](541_phg_pipeline_optimization_lazy_import_stage_skip.md) | lazy import引き上げ(8箇所) + pre-order disabled stage 5段スキップ最適化 + spread_adapt解明 |
| 542 | impl | [542_phg_ceiling_030_identity_audit_memleak.md](542_phg_ceiling_030_identity_audit_memleak.md) | ceiling 0.25→0.30 + 残存identity段6段分析 + メモリリーク監査 + test_405修正 |
| 543 | impl | [543_phg_phase3_ofi_toxicity_as_delta.md](543_phg_phase3_ofi_toxicity_as_delta.md) | Phase 3: OFI-Lite (CKS 2014) + Toxicity Budget独立化 (GM 1985) + A-S δ*計測 |
| 544 | impl | [544_phg_phase4_delta_star_ofi_window_sidecar_ladder.md](544_phg_phase4_delta_star_ofi_window_sidecar_ladder.md) | Phase 4: δ*→spread_adapt動的バインド + OFI rolling window + sidecar ladder 0.20 |
| 545 | impl | [545_phg_phase4b_ofi_boost_toxicity_sidecar_dynamic.md](545_phg_phase4b_ofi_boost_toxicity_sidecar_dynamic.md) | Phase 4b: OFI→boost変調 + Toxicity→sidecar confidence減衰 + δ*→sidecar動的天井 |
| 546 | review | [546_phg_review_request_next_steps.md](546_phg_review_request_next_steps.md) | レビュー依頼: sidecar拡大 / CalibrationMap / δ*→executor / drift detection |
| 547 | rev | [547_phg_rev_540_546_spec_dilemmas_and_multifaceted_recommendations.md](547_phg_rev_540_546_spec_dilemmas_and_multifaceted_recommendations.md) | 540-546レビュー: preflight_insufficient優先、same-SHA live検証先行、546のA-Eへ判定を付与 |
| 548 | analysis | [548_phg_phase4_dilemma_and_true_nature_of_time_restrictions.md](548_phg_phase4_dilemma_and_true_nature_of_time_restrictions.md) | Phase 4葛藤と本質論: 時間制約の多角的再解釈とレビュー盲点 |
| 549 | fix+analysis | [549_log_analysis_ewma_clamp_sidecar_as_pattern.md](549_log_analysis_ewma_clamp_sidecar_as_pattern.md) | ログ分析深堀り: EWMA汚染修正(Winsorization) + サイドカー死因(neutral fallback) + ASパターン |

---

## 種別凡例

| 種別 | 説明 |
|---|---|
| **plan** | 計画・設計 |
| **rpt** | 調査・分析レポート |
| **impl** | 実装完了報告 |
| **fix** | バグ修正・改善 |
| **rev** | 外部レビュー (外部 AI による批判的検証) |
| **resp** | レビューへの対応実装 |
| **ext** | 外部依頼用資料 (Codex レビューパッケージ等) |
| **refactor** | リファクタリング・構造改善 |
| **analysis** | 深堀り分析・定量調査 |

---

## 欠番一覧

以下の番号はスキップまたは統合済み:

002–004, 006–008, 011–012, 016–017, 020, 024–027, 029–030, 035, 039,
044–045, 049, 055–056, 089

> 欠番は主にセッション内作業ノートの非文書化、番号統合 (038→036)、
> または Copilot セッション間の連番断絶により発生。

### 旧番ファイル (非索引)

| ファイル | 備考 |
|---|---|
| [065_as_lr_prep.md](065_as_lr_prep.md) | 065_ph1_impl_as_lr_prep.md の旧名 |

---

## 番号体系

```
NNN_phX_TYPE_description.md
 │   │    │
 │   │    └── plan / rpt / impl / fix / rev / resp / ext / refactor / analysis
 │   └── ph0-ph5 / phg (フェーズ横断)
 └── 3桁連番 (セッション単位で採番)
```

> **注意**: 初期ドキュメント (~000-090) では TYPE トークンが省略・不統一の
> ケースが多い (例: `059_impl_summary.md`, `119_fill_test_...`)。
> また Gemini レビューは `NNN_ph2_gemini_...` 形式が慣例化している (正式には rev)。
> リネームは後方互換性を考慮し段階的に実施する方針。

---

## 現在の重点課題

### 最優先 (Gate 判定関連)

1. ~~**fill_test 168h 再実測**~~: 121# 改善版で 168h 蓄積 → ✅ G1.1 判定完了
2. ~~**G1.1-exec gate 判定**~~: ✅ PASS (gate_judgment.py)
3. ~~**Holm-Bonferroni 補正**~~: ✅ 122# B2 で g1_2_full_judgment に実装済み
4. ~~**G2 SAC 4-seed 訓練**~~: ✅ **PASS** — reward-tuned (γ=0.95, E4=-3.5% 緩和) で達成 (386#/387#)
5. ~~**G3 PnL Monte Carlo**~~: ✅ **PASS** — reward-clean (400#) で G3 全条件クリア (409#, PF=1.145, Sharpe=5.70, MaxDD=0.26%)
6. **100K 拡大訓練**: ❌ **G3 FAIL** — val_ratio交絡 (0.02→0.20) で20K G3 PASSが偽陽性の疑い, F6 best_model未使用 (422#). 次: val_ratio統一再実験
7. **G3.1 stress 条件**: slippage 1tick, maker miss penalty の感度分析 (392# P2-A) — **000# §3.5.1 で正式定義済み** (410#)

### 高優先 (収益性直結)

8. **SkipGate 再訓練/見直し**: データ 500 到達時に preorder features で再訓練 (097#/095# M1)
9. **spread_adaptive AB テスト**: narrow_spread_bps 探索 (093#/092#)
10. **Volatility Guard**: 107# Phase 2 提案の動的ゲーティング
11. **SAC sidecar 起動**: Phase 3.1 scheduler 初回実行 → sidecar_signal.json 生成 (377#)

### 中期 (ph5 本番前に必須)

12. **013# D-1**: `OrderManager.execute_trade()` — 実取引パス実装
13. **013# D-3**: `post_only` 対応 — maker 保証
14. **013# C-4**: `asyncio.to_thread` 残 5 メソッド
15. **Tier-2/3 統合**: PnL Monte Carlo, RiskRuleEngine, Reconciliation (113#)

### 低優先 (v461+)

16. **106# R3**: SkipGate 単体テスト拡充
   - session037 で大幅補強済み。残りは future 大項目というより継続保守
17. **106# R5**: lib → ztb 移動 (残 4 モジュール)
   - 主要 canonical 化は session037 で前倒し済み。残りは `maker_price` / `skip_gate_evaluator` / `order_monitor` / `ab_judgment`
18. **106# R6**: utils 70+ ファイル分割
19. **106# R7**: config/ vs configs/ 重複ディレクトリ整理
20. **109# DUP3**: UnifiedTrainer God Object (2835行)
21. **408# RewardCalculator 分割**: God Object 2252行 → RewardSettingsAccessor / StageRouter / PostProcessor 段階分割
| 550 | plan | [550_phg_plan_maker_price_state_and_stage_boundary.md](550_phg_plan_maker_price_state_and_stage_boundary.md) | MakerPrice state 分類 + compute stage 境界設計 |
| 551 | plan | [551_phg_plan_post_550_remaining_waves.md](551_phg_plan_post_550_remaining_waves.md) | Post-550 残課題の Wave2-5 実行計画 |
| 552 | report | [552_sac_retrain_investigation_and_toxicity_counter.md](552_sac_retrain_investigation_and_toxicity_counter.md) | SAC retrain OOS gate 持続失敗調査 + 546#D toxicity counter 実装 |
| 553 | impl | [553_phg_impl_ohlcv_auto_update_pipeline.md](553_phg_impl_ohlcv_auto_update_pipeline.md) | OHLCV 自動更新パイプライン: yfinance→FeatureRegistry→parquet + stale data guard |
| 554 | impl | [554_phg_impl_raw_gap_fill_and_calibration_batch.md](554_phg_impl_raw_gap_fill_and_calibration_batch.md) | Raw gap fill + CalibrationMap offline batch (546# §B 推奨) |
| 555 | impl | [555_phg_impl_calibrationmap_runtime_integration.md](555_phg_impl_calibrationmap_runtime_integration.md) | CalibrationMap ランタイム統合: entry gate EV check + online update |
| 556 | rpt | [556_phg_rpt_git_optimization_and_architectural_proposals.md](556_phg_rpt_git_optimization_and_architectural_proposals.md) | Git リポジトリの健全化と構造的負債の深掘り・改善提案 |
