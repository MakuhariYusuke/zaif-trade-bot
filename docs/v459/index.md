# v459 ドキュメント索引

> **最終判定: No-Go** — v460 へ移行  
> 根本原因: 特徴量情報量不足 (K2 実験で確定)  
> 最終コミット: `1f932e510` (2026-02-13)

---

## 全体構成

| Phase | 文書番号 | 概要 | 結論 |
|-------|---------|------|------|
| Phase 0 | 00#–08# | プロジェクト定義・既存コード分析 | 完了 |
| Phase 1 | 09#–11# | 基盤実装仕様 | 完了 |
| Phase 2 | 12#–23# | 環境・報酬・SAC 統合 | 完了 |
| Phase 3 | 24#–39# | 行動空間・実行モデル・特徴量最適化 | 完了 |
| Phase 4 | 40#–86# | SAC ハイパーパラメータ探索・AB テスト | 収束せず |
| Phase 4.5 | 87#–101# | 収益化ピボット・Gate0 検証 | Gate0 FAIL |
| Phase C | 102#–104# | 統合実験・d2 ベースライン確立 | PF < 1.0 |
| Phase D | 105#–113# | HFT・Swing・創造的再評価 | 10+ 実験全 FAIL |
| **Phase E** | **114#–116#** | **診断・反実仮想・K2 最終検証** | **No-Go 確定** |

---

## Phase 0: プロジェクト定義 (00#–08#)

| # | 文書 | 要旨 |
|---|------|------|
| 00# | [00_project_proposal_v459.md](00_project_proposal_v459.md) | v459 "Alpha Resurrection" プロジェクト提案。SAC ベース BTC/JPY 1min 戦略 |
| 01# | [01_review_and_gaps_v459.md](01_review_and_gaps_v459.md) | 00# レビュー、ギャップ分析 |
| 02# | [02_evaluation_design_and_causality.md](02_evaluation_design_and_causality.md) | 評価設計と因果推論フレームワーク |
| 03# | [03_rereview_v459.md](03_rereview_v459.md) | 再レビュー |
| 04# | [04_phase0_specification.md](04_phase0_specification.md) | Phase 0 仕様 |
| 05# | [05_phase0_specification_review.md](05_phase0_specification_review.md) | Phase 0 仕様レビュー |
| 06# | [06_phase0_2_existing_code_analysis.md](06_phase0_2_existing_code_analysis.md) | 既存コード分析 |
| 07# | [07_phase0_completion_report.md](07_phase0_completion_report.md) | Phase 0 完了報告 |
| 08# | [08_phase0_completion_review.md](08_phase0_completion_review.md) | Phase 0 完了レビュー |

## Phase 1: 基盤実装 (09#–11#)

| # | 文書 | 要旨 |
|---|------|------|
| 09# | [09_phase1_specification.md](09_phase1_specification.md) | Phase 1 仕様 |
| 10# | [10_phase1_specification_review.md](10_phase1_specification_review.md) | Phase 1 仕様レビュー |
| 11# | [11_phase1_completion_report.md](11_phase1_completion_report.md) | Phase 1 完了報告 |

## Phase 2: 環境・報酬・SAC 統合 (12#–23#)

| # | 文書 | 要旨 |
|---|------|------|
| 12# | [12_phase2_specification.md](12_phase2_specification.md) | Phase 2 仕様 |
| 13# | [13_phase2_specification_self_review.md](13_phase2_specification_self_review.md) | 自己レビュー |
| 14# | [14_phase2_specification_review.md](14_phase2_specification_review.md) | 外部レビュー |
| 15# | [15_doc14_review_response.md](15_doc14_review_response.md) | 14# への対応 |
| 16# | [16_phase2_specification_rereview.md](16_phase2_specification_rereview.md) | 再レビュー |
| 17# | [17_phase2_specification_review.md](17_phase2_specification_review.md) | 仕様最終レビュー |
| 18# | [18_phase2_completion_report.md](18_phase2_completion_report.md) | Phase 2 完了報告 |
| 19# | [19_phase2_completion_review.md](19_phase2_completion_review.md) | 完了レビュー |
| 20# | [20_doc19_implementation_review_response.md](20_doc19_implementation_review_response.md) | 19# 対応 |
| 21# | [21_doc20_implementation_review.md](21_doc20_implementation_review.md) | 実装レビュー |
| 22# | [22_doc21_implementation_review_response.md](22_doc21_implementation_review_response.md) | 21# 対応 |
| 23# | [23_phase2_completion_verification.md](23_phase2_completion_verification.md) | Phase 2 最終検証 |

## Phase 3: 行動空間・特徴量最適化 (24#–39#)

| # | 文書 | 要旨 |
|---|------|------|
| 24# | [24_phase3_specification.md](24_phase3_specification.md) | Phase 3 仕様 |
| 25# | [25_phase3_specification_review.md](25_phase3_specification_review.md) | 仕様レビュー |
| 26# | [26_doc25_response.md](26_doc25_response.md) | 25# 対応 |
| 27# | [27_phase3_implementation_plan_phase4_ready.md](27_phase3_implementation_plan_phase4_ready.md) | 実装計画 |
| 28# | [28_phase3_day1_implementation_complete.md](28_phase3_day1_implementation_complete.md) | Day1 完了 |
| 29# | [29_phase3_existing_implementation_review.md](29_phase3_existing_implementation_review.md) | 既存実装レビュー |
| 30# | [30_phase3_day3_reward_config_complete.md](30_phase3_day3_reward_config_complete.md) | 報酬設定完了 |
| 31# | [31_phase3_action_space_analysis.md](31_phase3_action_space_analysis.md) | 行動空間分析 |
| 32# | [32_phase3_action_space_fix_complete.md](32_phase3_action_space_fix_complete.md) | 行動空間修正 |
| 33# | [33_phase3_execution_status.md](33_phase3_execution_status.md) | 実行状況 |
| 34# | [34_windows_sigint_resolution_and_first_success.md](34_windows_sigint_resolution_and_first_success.md) | Windows SIGINT 解決・初回成功 |
| 35# | [35_feature_generation_optimization_plan.md](35_feature_generation_optimization_plan.md) | 特徴量生成最適化計画 |
| 36# | [36_feature_generation_optimization_review.md](36_feature_generation_optimization_review.md) | 最適化レビュー |
| 37# | [37_existing_implementation_audit.md](37_existing_implementation_audit.md) | 既存実装監査 |
| 38# | [38_review_feature_opt_plan_and_audit.md](38_review_feature_opt_plan_and_audit.md) | レビュー |
| 39# | [39_review_response_修正計画.md](39_review_response_修正計画.md) | 修正計画 |

## Phase 4: ハイパーパラメータ探索 (40#–86#)

<details>
<summary>40#–86# (47文書) — クリックで展開</summary>

| # | 文書 | 要旨 |
|---|------|------|
| 40# | [40_phase4_planning.md](40_phase4_planning.md) | Phase 4 計画 |
| 41# | [41_phase4_planning_review.md](41_phase4_planning_review.md) | 計画レビュー |
| 42# | [42_phase4_planning_review.md](42_phase4_planning_review.md) | 計画レビュー (2) |
| 43# | [43_phase3.5_verification_results.md](43_phase3.5_verification_results.md) | Phase 3.5 検証 |
| 44# | [44_phase4_week1_implementation_report.md](44_phase4_week1_implementation_report.md) | Week1 実装報告 |
| 45# | [45_phase4_day5_ab_test_results.md](45_phase4_day5_ab_test_results.md) | Day5 AB テスト |
| 46# | [46_phase4_metrics_bug_fix_report.md](46_phase4_metrics_bug_fix_report.md) | メトリクスバグ修正 |
| 47# | [47_current_position_summary.md](47_current_position_summary.md) | 現状サマリ |
| 48# | [48_external_advice_request.md](48_external_advice_request.md) | 外部助言依頼 |
| 49# | [49_external_advice_response.md](49_external_advice_response.md) | 外部助言回答 |
| 50# | [50_phase4_week2_implementation_plan.md](50_phase4_week2_implementation_plan.md) | Week2 計画 |
| 51# | [51_phase4_week2_implementation_plan_review.md](51_phase4_week2_implementation_plan_review.md) | Week2 レビュー |
| 52# | [52_phase4_week2_implementation_plan_revised.md](52_phase4_week2_implementation_plan_revised.md) | Week2 改訂 |
| 53# | [53_time_optimization_strategies.md](53_time_optimization_strategies.md) | 時間最適化戦略 |
| 54#–61# | 54#–61# | AI レビュープロンプト・回答群 |
| 62# | [62_day6_reward_tuning_analysis.md](62_day6_reward_tuning_analysis.md) | Day6 報酬チューニング |
| 63#–66# | 63#–66# | レビュー妥当性評価 |
| 67# | [67_day7_causal_separation_results.md](67_day7_causal_separation_results.md) | Day7 因果分離 |
| 68# | [68_day7_causal_separation_review.md](68_day7_causal_separation_review.md) | Day7 レビュー |
| 69# | [69_day8_scale_deconfounding_results.md](69_day8_scale_deconfounding_results.md) | Day8 スケール交絡除去 |
| 70#–71# | 70#–71# | Day8 レビュー |
| 72# | [72_day8_ent_coef_ablation_results.md](72_day8_ent_coef_ablation_results.md) | エントロピー係数アブレーション |
| 73# | [73_day8_ent_coef_ablation_review.md](73_day8_ent_coef_ablation_review.md) | アブレーションレビュー |
| 74# | [74_feature_expansion_plan.md](74_feature_expansion_plan.md) | 特徴量拡張計画 |
| 75# | [75_day9_gamma_ablation_results.md](75_day9_gamma_ablation_results.md) | γ アブレーション |
| 76#–77# | 76#–77# | Day9 レビュー・対応 |
| 78# | [78_day9b_50k_validation_results.md](78_day9b_50k_validation_results.md) | 50K 検証 |
| 79# | [79_day9b_50k_validation_review.md](79_day9b_50k_validation_review.md) | 50K レビュー |
| 80# | [80_day10_comprehensive_experiment_plan.md](80_day10_comprehensive_experiment_plan.md) | Day10 総合計画 |
| 81# | [81_day9b_review_response.md](81_day9b_review_response.md) | Day9b 対応 |
| 82# | [82_day10_comprehensive_results.md](82_day10_comprehensive_results.md) | Day10 結果 |
| 83# | [83_day10_comprehensive_review.md](83_day10_comprehensive_review.md) | Day10 レビュー |
| 84# | [84_day10_review_response_and_fix_plan.md](84_day10_review_response_and_fix_plan.md) | Day10 修正計画 |
| 85# | [85_day11_verification_plan.md](85_day11_verification_plan.md) | Day11 検証計画 |
| 86# | [86_day11_verification_results.md](86_day11_verification_results.md) | Day11 検証結果 |

</details>

## Phase 4.5: 収益化ピボット (87#–101#)

| # | 文書 | 要旨 |
|---|------|------|
| 87# | [87_phase4.5_profitability_plan.md](87_phase4.5_profitability_plan.md) | 収益化計画 |
| 88# | [88_phase4.5_profitability_review.md](88_phase4.5_profitability_review.md) | 収益化レビュー |
| 89# | [89_phase4.5_detailed_execution_plan.md](89_phase4.5_detailed_execution_plan.md) | 詳細実行計画 |
| 90# | [90_p1_experiment_results.md](90_p1_experiment_results.md) | P1 実験結果 |
| 91# | [91_pivot_plan_based_on_legacy_analysis.md](91_pivot_plan_based_on_legacy_analysis.md) | レガシー分析ピボット |
| 92# | [92_pivot_plan_review.md](92_pivot_plan_review.md) | ピボットレビュー |
| 93# | [93_revised_pivot_plan.md](93_revised_pivot_plan.md) | 改訂ピボット |
| 94# | [94_gate0_phaseb_verification_results.md](94_gate0_phaseb_verification_results.md) | Gate0 検証結果 |
| 95# | [95_gate0_phaseb_review.md](95_gate0_phaseb_review.md) | Gate0 レビュー |
| 96# | [96_revised_execution_plan.md](96_revised_execution_plan.md) | 改訂実行計画 |
| 97# | [97_phase_b_results_analysis.md](97_phase_b_results_analysis.md) | Phase B 結果分析 |
| 98# | [98_phase_b_critical_reassessment_and_roadmap.md](98_phase_b_critical_reassessment_and_roadmap.md) | Phase B 再評価 |
| 99# | [99_98_review_validation_and_execution_plan.md](99_98_review_validation_and_execution_plan.md) | 98# レビュー |
| 100# | [100_phase45_completion_report.md](100_phase45_completion_report.md) | Phase 4.5 完了報告 |
| 101# | [101_phase45_followup_reuse_recommendations.md](101_phase45_followup_reuse_recommendations.md) | 再利用推奨事項 |

## Phase C: 統合実験 (102#–104#)

| # | 文書 | 要旨 |
|---|------|------|
| 102# | [102_phase_c_experiment_log.md](102_phase_c_experiment_log.md) | Phase C 実験ログ |
| 103# | [103_phase_c_review_and_next_steps.md](103_phase_c_review_and_next_steps.md) | Phase C レビュー |
| 104# | [104_phase_c_comprehensive_report.md](104_phase_c_comprehensive_report.md) | Phase C 総合報告 |

## Phase D: HFT・Swing 実験 (105#–113#)

| # | 文書 | 要旨 |
|---|------|------|
| 105# | [105_phase_d_plan_for_codex_review.md](105_phase_d_plan_for_codex_review.md) | Phase D 計画 |
| 106# | [106_phase_d_critical_replan_50k_feature_reuse.md](106_phase_d_critical_replan_50k_feature_reuse.md) | 50K 再計画 |
| 107# | [107_106_review_and_revised_phase_d.md](107_106_review_and_revised_phase_d.md) | 106# レビュー |
| 108# | [108_phase_d1_experiment_results.md](108_phase_d1_experiment_results.md) | D1 実験結果 |
| 109# | [109_phase_d1_critical_review_and_aggressive_improvements.md](109_phase_d1_critical_review_and_aggressive_improvements.md) | D1 レビュー |
| 110# | [110_d2_swing_experiments_and_109_validation.md](110_d2_swing_experiments_and_109_validation.md) | D2 Swing 実験 |
| 111# | [111_d2_hft_analysis_and_next_actions.md](111_d2_hft_analysis_and_next_actions.md) | D2 HFT 分析 |
| 112# | [112_111_review_and_perf_fix.md](112_111_review_and_perf_fix.md) | 111# レビュー・性能修正 |
| 113# | [113_d2_creative_reassessment_and_discovery_map.md](113_d2_creative_reassessment_and_discovery_map.md) | 創造的再評価 (外部レビュー) |

## Phase E: 診断・最終検証 (114#–116#) ★最重要

| # | 文書 | 要旨 |
|---|------|------|
| 114# | [114_113_review_phase_realignment_and_next.md](114_113_review_phase_realignment_and_next.md) | 113# レビュー・Phase 再整合・Phase E 定義 |
| 115# | [115_114_policy_review_and_phase_e_corrections.md](115_114_policy_review_and_phase_e_corrections.md) | 114# レビュー・Phase E 補正 (外部レビュー) |
| **116#** | [**116_phase_e0_diagnostic_report.md**](116_phase_e0_diagnostic_report.md) | **Phase E 全実験報告 (§1–§18)** |
| **117#** | [**117_v460_doc00_design_and_naming_reform.md**](117_v460_doc00_design_and_naming_reform.md) | **v460 00# 設計方針・命名規則改革** |
| **118#** | [**118_v460_launch_additional_review.md**](118_v460_launch_additional_review.md) | **v460 始動追加レビュー (Gate 再設計・manifest)** |
| **119#** | [**119_v460_launch_integrated_policy.md**](119_v460_launch_integrated_policy.md) | **v460 始動統合方針 (118# 評価 + スクリプト・設定改革)** |

### 116# 内部構成

| セクション | 内容 | 結論 |
|-----------|------|------|
| §1–§5 | E0 診断 (Q1 IC / Q2 学習曲線 / Q3 Threshold) | NO_USEFUL_EDGE / NO_LEARNING_SIGNAL / COST_DOMINATED |
| §6–§7 | 不整合記録・コミット対象 | — |
| §8 | E1 反実仮想 (CF1/CF3/Oracle) | COST_STRUCTURE_FATAL / SAC_HAS_WEAK_EDGE |
| §9 | E2 方針決定 | 軸 B+D (コスト分離 + TTL) |
| §10 | E2α TTL 実験 | base +41.95% (単一 seed), break-even fee = 0.019% |
| §11 | E2β 設計 | Gate: ≥3/4 seeds gross > 0% |
| §12 | E2β Multi-seed 結果 | **2/4 正 → Gate FAIL** |
| §13 | Phase E 最終結論 | **No-Go** |
| §14 | 外部レビュー意見 | K1/K2 実験提案、分岐表 |
| §15 | §14 への評価・K2 設計 | K1 不要判断、K2 優先実行 |
| §16 | **K2 非RL 上限テスト結果** | **FEATURES_NO_INFO** (XGBoost IC=0.004, OOS IC=0.000) |
| §17 | v459 最終総括 | 特徴量情報量不足が根本原因 |
| §18 | §15 レビュー (進行中) | K1 優先度低に修正、D0 先送り注記 |

---

## 主要スクリプト

| パス | 用途 |
|------|------|
| `scripts/v459/run_phase_c.py` | Phase C/D/E 共通の SAC 訓練・評価基盤 |
| `scripts/v459/run_phase_e0_diagnostic.py` | E0 診断 (IC/学習曲線/Threshold) |
| `scripts/v459/run_phase_e1_counterfactual.py` | E1 反実仮想 (CF1/CF3/Oracle) |
| `scripts/v459/run_phase_e2a_ttl.py` | E2α TTL 実験 |
| `scripts/v459/run_phase_e2b_multiseed.py` | E2β Multi-seed 検証 |
| `scripts/v459/run_k2_nonrl_upper_bound.py` | K2 非 RL 上限テスト (XGBoost/Logistic) |

---

## v459 の教訓 (v460 へ)

1. **特徴量を先に検証せよ** — K2 (XGBoost walk-forward) を Phase A で実行していれば、Phase B–E の全実験 (数百時間) は不要だった
2. **単一 seed の成功を信じるな** — E2α +41.95% は E2β で +3.93% に崩壊した
3. **Oracle テストを早期に行え** — 完全予測でも費用負けする構造なら、いかなるモデル改善も無意味
4. **手数料構造は戦略の前提条件** — taker 0.1% × 1min 足は Oracle でも不成立。maker 0% が必須前提
