# v460 ドキュメント索引

> **v460 "Microstructure Edge"** — Coincheck BTC/JPY maker 執行品質検証  
> 最終更新: 2026-02-19 (113# Resilience + R1 分割)

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
| 082 | rpt | [082_ph2_fill_test_deep_dive_for_codex.md](082_ph2_fill_test_deep_dive_for_codex.md) | Fill Test データ深掘り (Codex レビュー用) |
| 083 | rev | [083_ph2_rev_082.md](083_ph2_rev_082.md) | レビュー: 082 Fill Test 深掘りの再点検 |
| 084 | impl | [084_ph2_impl_083_review_response.md](084_ph2_impl_083_review_response.md) | 083# レビュー指摘対応 + 盲点 8 項目特定 |
| 085 | impl | [085_ph2_084_blind_spot_impl.md](085_ph2_084_blind_spot_impl.md) | 084 盲点指摘の実装 |
| 086 | rpt | [086_ph2_rpt_time_filter_bug_and_session_analysis.md](086_ph2_rpt_time_filter_bug_and_session_analysis.md) | time_filter 片側蓄積バグ修正 + 085# セッション考察 |
| 087 | rev | [087_ph2_rev_086.md](087_ph2_rev_086.md) | レビュー: 086# 外部 Codex 分析 — 構造的損失原因の特定 |
| 088 | impl | [088_ph2_impl_087_review_response.md](088_ph2_impl_087_review_response.md) | 087# レビュー対応: SkipGate 動的較正 + sell ガード + データ品質修正 |
| 090 | rpt | [090_ph2_deep_dive_v2_for_codex.md](090_ph2_deep_dive_v2_for_codex.md) | fill_test 深掘り分析 v2 — Codex レビュー用資料 |
| 091 | rev | [091_ph2_rev_090.md](091_ph2_rev_090.md) | 090 深掘り分析 v2 の整合レビューと修正提案 |
| 092 | impl | [092_ph2_impl_gap_analysis.md](092_ph2_impl_gap_analysis.md) | 対応漏れ点検と先行実装 |
| 093 | impl | [093_ph2_spread_adaptive_fast_fill.md](093_ph2_spread_adaptive_fast_fill.md) | spread_adaptive / fast_fill_defense サイド別パラメータ追加 |
| 094 | impl | [094_ph2_stale_order_cancel_replace.md](094_ph2_stale_order_cancel_replace.md) | stale order 検出 & cancel-replace |
| 095 | ext | [095_ph2_codex_review_v3.md](095_ph2_codex_review_v3.md) | fill_test Codex レビュー v3 — 構造損失の根本原因と状態管理バグ |
| 096 | rev | [096_ph2_rev_095.md](096_ph2_rev_095.md) | 095 事後諸葛亮レビュー（ログ逆算 + 収益改善） |
| 097 | impl | [097_ph2_skipgate_retrain_preorder.md](097_ph2_skipgate_retrain_preorder.md) | SkipGate AS モデル再訓練（preorder-only features） |
| 098 | rpt | [098_ph2_post_097_deep_analysis.md](098_ph2_post_097_deep_analysis.md) | 097 後の構造診断 + 収益改善戦略 |
| 099 | rev | [099_ph2_rev_098.md](099_ph2_rev_098.md) | 098 改善点レビュー（トレーダー視点込み） |
| 110 | fix | [110_ph2_fix_086_time_filter_deadlock.md](110_ph2_fix_086_time_filter_deadlock.md) | 086# time_filter デッドロック修正 (49%アイドル解消) |
| 113 | impl | [113_ph2_impl_resilience_r1_split.md](113_ph2_impl_resilience_r1_split.md) | Resilience 統合 + R1 God Method 分割 (755→307行) |

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
| 111 | rpt | [111_phg_rpt_legacy_asset_research.md](111_phg_rpt_legacy_asset_research.md) | v456–v459 レガシー資産・教訓 調査レポート |
| 112 | rev | [112_phg_rev_111_legacy_asset.md](112_phg_rev_111_legacy_asset.md) | 111# レビュー + 追加提案 + 見落とし補完 |

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
044–045, 049, 055–056, 059–060

> 欠番は主にセッション内作業ノートの非文書化、番号統合 (038→036)、
> または Copilot セッション間の連番断絶により発生。

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

1. **113# 完了**: Resilience 統合 (CircuitBreaker / HealthMonitor / StatePersistence) + R1 God Method 分割
2. **fill_test 168h 再実測**: 113# 耐障害機能込みで 168h 再起動 → SLO/Gate 判定
3. **G1.1-exec gate 判定**: fill_test データ蓄積中 — SLO 閾値表 (111# §10) で判定
4. **111# Tier-2/3 残統合**: PnL Monte Carlo, RiskRuleEngine, watch_1m 等
5. **106# R3-R7**: SkipGate テスト、ドキュメント命名、lib/ztb 統合、utils 分割
