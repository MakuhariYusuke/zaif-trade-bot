# 055# v459 資産棚卸しレポート — v460 再利用可能性評価

| 項目 | 内容 |
|------|------|
| 日付 | 2026-03-06 |
| 目的 | v459 の全資産を棚卸しし、v460 への再利用候補を特定 |
| 依拠 | v459 index.md, 101#, 117#, 119# |
| 結論 | **v459 は No-Go 判定で終了したが、コア基盤 (ztb/) に取り込まれた成果は既に v460 で稼働中。scripts/v459 は実験固有であり移植不要。統計ツールと診断手法に参考価値あり** |

---

## §0 エグゼクティブサマリー

v459 の資産は以下の 6 カテゴリに分類される:

| カテゴリ | ファイル数 | v460 との関係 | 再利用推奨 |
|---------|-----------|-------------|-----------|
| scripts/v459 (実験スクリプト) | 45 本 | v460 は `run_experiment.py` 統一ランナーに移行済み | **低** — 個別参考のみ |
| tests/unit/v459 | 7 本 | v460 独自テスト 100+ 本が既存 | **低〜中** — CausalScaler テストは保守対象 |
| tests/integration | 1 本 | v460 統合テスト未確立 | **中** — パターンが参考になる |
| configs/v459 | 1 ファイル | v460 は `configs/v460/` に base.yaml + experiments/ を整備済み | **低** — 構造的に旧式 |
| docs/v459 | 121 文書 | v460 は 055 文書目 (本ドキュメント含む) | **中** — 教訓・反省は参照価値大 |
| ztb/ コア (v459 由来) | 5 箇所 | **既に v460 で稼働中** | **完了済み** |

---

## §1 scripts/v459 — 実験スクリプト群 (45 本, 12,458 行)

### §1.1 カテゴリ別一覧

#### A. 中核実験ランナー (God Object 群)

| ファイル | 行数 | 内容 | v460 対応 | 再利用 |
|---------|------|------|----------|--------|
| [run_phase_c.py](../../scripts/v459/run_phase_c.py) | 1,277 | 統一実験ランナー (Gate2 KPI全収集) | v460 `run_experiment.py` + `lib/` で分割済み | **不要** |
| [run_phase_e0_diagnostic.py](../../scripts/v459/run_phase_e0_diagnostic.py) | 578 | SAC 学習品質の多面的 IC 評価 | v460 ML パイプラインに未移植 | **参考** — IC 診断ロジック |
| [run_phase_e1_counterfactual.py](../../scripts/v459/run_phase_e1_counterfactual.py) | 406 | 反実仮想実験 (cost=0, threshold感度) | v460 未移植 | **参考** — 実験設計パターン |
| [run_phase_e2a_ttl.py](../../scripts/v459/run_phase_e2a_ttl.py) | 327 | 最小保有期間 (TTL) 実験 | v460 未移植 | **参考** — TTL 概念 |
| [run_phase_e2b_multiseed.py](../../scripts/v459/run_phase_e2b_multiseed.py) | 238 | Multi-seed 検証 + OOS | v460 `run_experiment.py` がseed対応済み | **不要** |
| [run_baselines.py](../../scripts/v459/run_baselines.py) | 354 | ベースライン比較 (Random/BuyHold/RSI) | v460 未移植 | **中** — ベースライン定義は参考 |
| [run_k2_nonrl_upper_bound.py](../../scripts/v459/run_k2_nonrl_upper_bound.py) | 373 | XGBoost/Logistic で特徴量情報量検証 | v460 `ml/` に同等機能あり | **不要** |

#### B. ハイパーパラメータ探索 (Day 6–11)

| ファイル | 行数 | 内容 | 再利用 |
|---------|------|------|--------|
| [run_day6_reward_tuning.py](../../scripts/v459/run_day6_reward_tuning.py) | — | 報酬チューニング | **不要** — v460 は報酬体系が異なる |
| [run_day7_causal_separation.py](../../scripts/v459/run_day7_causal_separation.py) | — | 因果分離実験 | **不要** |
| [run_day8_ent_coef_ablation.py](../../scripts/v459/run_day8_ent_coef_ablation.py) | — | エントロピー係数探索 | **不要** |
| [run_day8_scale_deconfounding.py](../../scripts/v459/run_day8_scale_deconfounding.py) | — | スケール交絡除去 | **不要** |
| [run_day9_gamma_ablation.py](../../scripts/v459/run_day9_gamma_ablation.py) | — | γ アブレーション | **不要** |
| [run_day9b_50k_validation.py](../../scripts/v459/run_day9b_50k_validation.py) | — | 50K ステップ検証 | **不要** |
| [run_day10_comprehensive.py](../../scripts/v459/run_day10_comprehensive.py) | — | 包括実験 | **不要** |
| [run_day11_verification.py](../../scripts/v459/run_day11_verification.py) | — | 最終検証 | **不要** |

#### C. ユーティリティ

| ファイル | 行数 | 内容 | v460 対応 | 再利用 |
|---------|------|------|----------|--------|
| [json_compat.py](../../scripts/v459/json_compat.py) | 66 | JSON 互換性ヘルパー | `ztb/io/json_io.py` が基盤 | **不要** — ztb層で解決済み |
| [generate_v459_features.py](../../scripts/v459/generate_v459_features.py) | 369 | 22特徴量生成 (RSI/SMA/EMA/BB等) | v460 `build_features.py` が大幅拡張済み | **不要** |
| [gate_c3_comparison.py](../../scripts/v459/gate_c3_comparison.py) | 451 | Mann-Whitney / Cliff's Delta 統計検定 | v460 未移植 | **中** — 統計検定ロジック |
| [check_data_leakage.py](../../scripts/v459/check_data_leakage.py) | 374 | データリーク検査 | v460 未移植 (101# で非推奨) | **低** — プレースホルダ多い |
| [precompute_optimized_features.py](../../scripts/v459/precompute_optimized_features.py) | 155 | 特徴量事前計算 | v460 `build_features.py` が担当 | **不要** |
| [precompute_optimized_features_memory_safe.py](../../scripts/v459/precompute_optimized_features_memory_safe.py) | 319 | メモリセーフ版特徴量計算 | チャンク処理パターンが参考 | **低** — パターンのみ |
| [prepare_cached_data.py](../../scripts/v459/prepare_cached_data.py) | 62 | キャッシュ事前投入 | v460 データパイプラインに内蔵 | **不要** |
| [profile_feature_generation.py](../../scripts/v459/profile_feature_generation.py) | 129 | cProfile による特徴量プロファイル | v460 未移植 | **低** — 必要時に再作成可 |
| [report_cache_statistics.py](../../scripts/v459/report_cache_statistics.py) | 120 | FeatureCache 統計表示 | v460 未移植 | **低** |

#### D. AB テスト・バッチ実行

| ファイル | 内容 | 再利用 |
|---------|------|--------|
| run_ab_batches.py | AB テストバッチ | **不要** — v460 は YAML ベース実験管理 |
| run_ab_feature_test.py | 特徴量 AB テスト | **不要** |
| run_ab_minimal.py | 最小 AB テスト | **不要** |
| run_ab_reward_experiments.py | 報酬 AB 実験 | **不要** |
| run_gatec1c3_all.py | Gate C1/C3 統合実行 | **不要** |
| run_phase_c_batch.py | Phase C バッチ | **不要** |
| run_phase_c_subprocess.py | subprocess 分離実行 | **参考** — メモリリーク回避パターン |
| run_phase45_p1.py | Phase 4.5 P1 実験 | **不要** |
| run_phase45_p1_subprocess.py | subprocess 版 | **不要** |
| run_p1_background.py | バックグラウンド P1 | **不要** |

#### E. 分析・検証

| ファイル | 内容 | 再利用 |
|---------|------|--------|
| analyze_consistency.py | 実験再現性分析 | **低** — v460 分析基盤が既存 |
| analyze_single_experiment.py | 単一実験分析 | **低** |
| check_cache.py | キャッシュ検証 | **不要** |

#### F. テスト用スクリプト (scripts/v459 内)

| ファイル | 再利用 |
|---------|--------|
| test_ab_runner_simple.py, test_debug_interrupt.py, test_import.py, test_json_save.py, test_metrics_extraction.py, test_quick_trade.py, test_single_experiment.py, test_ultra_minimal.py, test_ultra_short_metrics.py | **全て不要** — 使い捨てデバッグ用 |

### §1.2 scripts/v459 総評

119# が指摘した通り、v459 スクリプトは `run_phase_c.py` (1,277行) を筆頭に **God Object + コピペ増殖** の典型。v460 は `run_experiment.py` + `lib/` + `ml/` に分割統治されており、構造的に上位互換。**v459 スクリプトの直接移植は不要**。

---

## §2 tests/unit/v459 — 単体テスト (7 本)

| ファイル | 内容 | テスト対象 | v460 対応 | 再利用 |
|---------|------|----------|----------|--------|
| [test_causal_scaler_v459.py](../../tests/unit/v459/test_causal_scaler_v459.py) | CausalOnlineScaler 単体テスト | `ztb/processing/causal_online_scaler.py` | ztb コア → v460 にも影響 | **高** — 保守必須 |
| [test_reporter_v459.py](../../tests/unit/v459/test_reporter_v459.py) | BacktestReporter テスト | `ztb/evaluation/walk_forward/reporter.py` | ztb コア → v460 にも影響 | **高** — 保守必須 |
| [test_entry_gate_safety_v459.py](../../tests/unit/v459/test_entry_gate_safety_v459.py) | Entry Gate 安全性テスト | `FastIntradayEnvV456._is_entry_action` | v460 は別 Gate 体系 | **低** |
| [test_config_validation_v459.py](../../tests/unit/v459/test_config_validation_v459.py) | config 検証テスト | `validate_env_config()` | v460 config 体系とは異なる | **低** |
| [test_p01_p02_completion.py](../../tests/unit/v459/test_p01_p02_completion.py) | P0-1/P0-2 バグ修正検証 | Entry Gate + Config | 回帰テストとして維持 | **中** |
| [test_p03_cost_double_count.py](../../tests/unit/v459/test_p03_cost_double_count.py) | コスト二重計上防止テスト | Reporter + Env | 重要な回帰テスト | **高** — コスト計算の保証 |
| [test_p04_val_test_leakage.py](../../tests/unit/v459/test_p04_val_test_leakage.py) | Val/Test リーケージ防止テスト | Evaluator | 重要な回帰テスト | **中** — ファイル存在チェック方式 |

### §2.1 tests/unit/v459 総評

`test_causal_scaler_v459.py` と `test_reporter_v459.py` は ztb コアモジュールのテストであり、**v460 でも引き続き有効**。`test_p03_cost_double_count.py` はコスト計算の健全性保証に重要。これらは削除せず保守対象とすべき。

---

## §3 tests/integration — 統合テスト (1 本)

| ファイル | 内容 | 再利用 |
|---------|------|--------|
| [test_v459_phase0_integration.py](../../tests/integration/test_v459_phase0_integration.py) | Reporter/EntryGate/CausalScaler/Config の統合動作検証 (361行) | **中** — パターンが参考になるが、v460 コンポーネントとの統合テストは別途必要 |

---

## §4 configs/v459 — 設定ファイル (1 ファイル)

| ファイル | 行数 | 内容 | 再利用 |
|---------|------|------|--------|
| [base/config.yaml](../../configs/v459/base/config.yaml) | 267 | SAC HP / 環境 / Walk-Forward / 評価基準 / Phase 制御 | **不要** |

### §4.1 比較

| 項目 | v459 config | v460 config |
|------|------------|------------|
| 構造 | 単一 `base/config.yaml` (267行) | `base.yaml` + `experiments/*.yaml` + `fill_test.yaml` + `gate_thresholds.yaml` |
| 実験管理 | Python dict ハードコード (スクリプト内) | YAML ファイルで宣言的 |
| 評価基準 | config 内に `success_criteria` 記述 | `gate_thresholds.yaml` に分離 |
| Phase 制御 | config 内に Phase 定義 | ドキュメントレベルで管理 |

v460 の設定体系は v459 の反省 (119# §2.3) を踏まえて根本的に再設計されている。**v459 config の移植は不要**。

v459 config の `success_criteria` セクション (ROI>5%, PF>1.2, Sharpe>1.0, MaxDD<15%, WinRate>35%) は参考値として有用。

---

## §5 docs/v459 — ドキュメント (121 文書)

### §5.1 フェーズ別概要

| Phase | 文書 | 結論 | v460 参考価値 |
|-------|------|------|-------------|
| Phase 0 (00#–08#) | プロジェクト定義・コード分析 | 完了 | **低** — v460 が 000# で刷新 |
| Phase 1 (09#–11#) | 基盤実装仕様 | 完了 | **低** |
| Phase 2 (12#–23#) | 環境・報酬・SAC 統合 | 完了 | **低** |
| Phase 3 (24#–39#) | 行動空間・特徴量最適化 | 完了 | **中** — 行動空間分析が参考 |
| Phase 4 (40#–86#) | HP 探索・AB テスト | 収束せず | **中** — 失敗パターンが教訓 |
| Phase 4.5 (87#–101#) | 収益化ピボット | Gate0 FAIL | **中** — 評価基準 |
| Phase C (102#–104#) | 統合実験 | PF < 1.0 | **低** |
| Phase D (105#–113#) | HFT/Swing 探索 | 全 FAIL | **中** — HFT 分析が参考 |
| Phase E (114#–116#) | 診断・K2 | No-Go 確定 | **高** — 根本原因分析 |
| v460 移行 (117#–120#) | 設計方針・命名改革 | v460 基礎 | **高** — 直接適用済み |

### §5.2 特に参考価値の高い文書

| 文書 | 内容 | 参考価値 |
|------|------|---------|
| [101# 再利用提案](../../docs/v459/101_phase45_followup_reuse_recommendations.md) | 過去 vXXX 資産の再利用マップ | **高** — 本レポートの先行分析 |
| [116# Phase E0 診断報告](../../docs/v459/116_phase_e0_diagnostic_report.md) | SAC 学習障害の根本原因 (IC≈0) | **高** — v460 が特徴量改革に至った根拠 |
| [117# 命名法則改革](../../docs/v459/117_v460_doc00_design_and_naming_reform.md) | v460 文書・命名体系の設計根拠 | **高** — v460 基盤文書 |
| [119# 統合方針](../../docs/v459/119_v460_launch_integrated_policy.md) | v460 スクリプト・設定改革の仕様 | **高** — v460 アーキ設計の原典 |
| [113# 創造的再評価](../../docs/v459/113_d2_creative_reassessment_and_discovery_map.md) | Phase D の探索と発見マップ | **中** — HFT 戦略論 |
| [31# 行動空間分析](../../docs/v459/31_phase3_action_space_analysis.md) | 1D action space の分析 | **中** |

---

## §6 ztb/ コア — v459 由来の統合済み成果

v459 の成果のうち、最も価値のある部分は **既に ztb/ コアに取り込まれて v460 で稼働中**:

| ztb モジュール | v459 由来 | 状態 |
|---------------|----------|------|
| `ztb/processing/causal_online_scaler.py` | Phase 0.2c 因果性保証付き Scaler | **稼働中** |
| `ztb/features/grouping/causal_grouped_scaler.py` | Phase 0.2c Grouped Scaler | **稼働中** |
| `ztb/training/unified_trainer/trainer.py` (L1934) | Parquet 対応最適化 | **稼働中** |
| `ztb/trading/environment/heavy_env/mixins/initialization.py` (L269) | feature_flags 最適化 | **稼働中** |
| `ztb/utils/v4xx_config_converter.py` (L48) | バージョン検出 | **稼働中** |

---

## §7 外部依存 — v459 を参照するファイル

v460 のファイルは v459 スクリプトを **一切 import していない** (確認済み)。

v459 スクリプトを参照するテストが存在:

| テストファイル | 参照先 | 状態 |
|--------------|-------|------|
| `tests/unit/scripts/test_run_phase_c.py` | `scripts.v459.run_phase_c.compute_gate2_metrics` | v459 固有テスト |
| `tests/unit/scripts/test_run_phase_c_d0.py` | `scripts.v459.run_phase_c` (多数) | v459 固有テスト |
| `tests/test_reward_config_integration.py` | `scripts.v459.run_day6_reward_tuning` | v459 固有テスト |

これらは v459 回帰テストとして残置で問題ないが、v460 には影響しない。

---

## §8 結論と推奨アクション

### 即座のアクション不要
v459 の価値ある成果は既に ztb/ コアに統合済み。v460 は v459 の構造的欠陥 (God Object, コピペ, 設定体系の形骸化) を 119# の設計に従い解消済み。

### 参考として保持すべきもの

| 参考資産 | 用途 |
|---------|------|
| Phase E 診断ロジック (IC 多面評価) | v460 ML モデル評価時に手法を参照 |
| `gate_c3_comparison.py` の統計検定 | Mann-Whitney / Cliff's Delta の scipy 不要実装 |
| Phase E2α の TTL (最小保有期間) 概念 | 取引頻度最適化の参考 |
| subprocess 分離実行パターン | メモリリーク回避の設計パターン |
| `run_baselines.py` のベースライン定義 | 将来のベースライン実験設計 |
| docs/v459 の失敗分析 (特に 116#) | 「なぜ SAC は学習しなかったか」の記録 |

### 移植の必要なし
v459 スクリプト群を v460 に移植する必要はない。v460 は `run_experiment.py` 統一ランナー + YAML 宣言的実験管理 + `lib/` 分割アーキテクチャにより、v459 の教訓を構造レベルで解決済み。

### テスト保守
`tests/unit/v459/test_causal_scaler_v459.py`, `test_reporter_v459.py`, `test_p03_cost_double_count.py` は ztb コアの回帰テストとして **削除せず保守継続**。
