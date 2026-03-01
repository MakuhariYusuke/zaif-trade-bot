# v460 ドキュメント索引

> **v460 "Microstructure Edge"** — Coincheck BTC/JPY maker 執行品質検証  
> 最終更新: 2026-03-02 (211# 204# I offset boost 3層防御完成, sleep clamp, halt可視化ログ)

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
| **ph3** | G2 | コード整理・SAC 4-seed 訓練 | ⏳ 待機 (一部先行) |
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
| 212 | audit | [212_codebase_audit_review.md](212_codebase_audit_review.md) | コードベース品質監査 — Codex/Gemini 外部レビュー用改善ポイント一覧 |

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

---

## 欠番一覧

以下の番号はスキップまたは統合済み:

002–004, 006–008, 011–012, 016–017, 020, 024–027, 029–030, 035, 037, 039,
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
 │   │    └── plan / rpt / impl / fix / rev / resp / ext
 │   └── ph0-ph5 / phg (フェーズ横断)
 └── 3桁連番 (セッション単位で採番)
```

---

## 現在の重点課題

### 最優先 (ph2 Gate 判定関連)

1. **fill_test 168h 再実測**: 121# 改善版で 168h 蓄積中 → SLO/Gate 判定 (現在 ~84%)
2. **G1.1-exec gate 判定**: SLO 閾値表 (111# §10) で最終判定 → ✅ gate_judgment.py (122# B1)
3. ~~**Holm-Bonferroni 補正**~~: ✅ 122# B2 で g1_2_full_judgment に実装済み (F4/F4b/F4c 3TF)

### 高優先 (収益性直結)

4. **SkipGate 再訓練/見直し**: データ 500 到達時に preorder features で再訓練 (097#/095# M1)
5. **spread_adaptive AB テスト**: narrow_spread_bps 探索 (093#/092#)
6. **Volatility Guard**: 107# Phase 2 提案の動的ゲーティング

### 中期 (ph5 本番前に必須)

7. **013# D-1**: `OrderManager.execute_trade()` — 実取引パス実装
8. **013# D-3**: `post_only` 対応 — maker 保証
9. **013# C-4**: `asyncio.to_thread` 残 5 メソッド
10. **Tier-2/3 統合**: PnL Monte Carlo, RiskRuleEngine, Reconciliation (113#)

### 低優先 (v461+)

11. **106# R3**: SkipGate 単体テスト拡充
12. **106# R5**: lib → ztb 移動 (残 4 モジュール)
13. **106# R6**: utils 70+ ファイル分割
14. **106# R7**: config/ vs configs/ 重複ディレクトリ整理
15. **109# DUP3**: UnifiedTrainer God Object (2835行)
