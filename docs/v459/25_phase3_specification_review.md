# v459 Phase 3 仕様書レビュー (25)

**Date**: 2026-01-25  
**Status**: 📝 Review  
**Targets**: `docs/v459/24_phase3_specification.md`

---

## Findings (Critical -> Major -> Minor)
- [Critical] Phase 2で「ABTestingComparator基本クラス」「compute_descriptive_stats実装済み」と記載されていますが、実コード側に該当実装が見当たりません。Phase 3の拡張前提が崩れるため、まず実装有無の確認が必要です。`docs/v459/24_phase3_specification.md:67` `docs/v459/24_phase3_specification.md:68`
- [Major] Circuit Breaker統合の疑似コードが既存APIと不一致です。`CircuitBreaker`は`CircuitBreakerConfig`を受け取る設計で、`should_halt`等は存在しません。仕様のまま実装すると破綻します。`docs/v459/24_phase3_specification.md:1062` `docs/v459/24_phase3_specification.md:1070` `ztb/trading/production/circuit_breaker.py:83`
- [Major] MTF因果性・Scaler境界の対象パスが存在しません。`check_causality.py`は見当たらず、`ztb/features/scaling/online_scaler.py`も実在しないため、実装対象の再定義が必要です。`docs/v459/24_phase3_specification.md:1166` `docs/v459/24_phase3_specification.md:1180` `ztb/processing/online_scaler.py:1`
- [Major] 報酬Stage 1/2/3のコード例が現行Env/Reward設計と噛み合っていません。`_last_observation`や`last_action`は未定義で、実際は`compute_hft_reward`と`ichimoku_signals`が使用されています。統合ポイントの再設計が必要です。`docs/v459/24_phase3_specification.md:951` `docs/v459/24_phase3_specification.md:966` `docs/v459/24_phase3_specification.md:973` `ztb/trading/environment/fast_intraday_env_v456.py:34` `ztb/trading/environment/fast_intraday_env_v456.py:747`
- [Major] サンプル数の説明が矛盾しています。「4 seed × 4 split（Val/Test）= 16」と記載されていますが、評価期間は「Val + Test」とのみ書かれており、2 splitなら8サンプルです。統計検定の前提が揺らぎます。`docs/v459/24_phase3_specification.md:302` `docs/v459/24_phase3_specification.md:1024`

---

## Open Questions / Assumptions
- ABTestingComparatorはどこに実装されている想定ですか？未実装ならPhase 3の最初の成果物として新規作成に切り替えますか？
- Walk-Forwardのsplit数は「2（Val/Test）」か「4」か、どちらで確定しますか？サンプル数要件の整理が必要です。
- 報酬Stage 1/2/3は`compute_hft_reward`側に統合する想定ですか？それともEnv内で分岐させる設計ですか？

---

## 既存実装の活用（推奨）
- **AB実行基盤**: `tools/ab_test_runner.py`, `tools/run_ab_searches.py`, `experiments/v450/run_ab_test_threshold_v450.py`
- **統計比較の既存例**: `ztb/analysis/comparative/analyze_backtest.py`, `ztb/analysis/comparative/model_comparator.py`
- **メトリクス算出の共通資産**: `ztb/metrics/metrics.py`, `ztb/utils/metrics/trading_metrics.py`, `ztb/trading/environment/components/reward/metrics.py`
- **リスク制御・損失管理**: `ztb/trading/risk/backtest_risk_manager.py`, `ztb/risk/rules.py`, `ztb/risk/advanced_auto_stop.py`
- **報酬設計の統合先**: `ztb/trading/rewards/fast_intraday.py`, `ztb/trading/environment/components/calculators/reward_calculator.py`
- **Scaler/境界チェック**: `ztb/processing/online_scaler.py`, `ztb/processing/causal_online_scaler.py`, `ztb/analysis/core/data/check_scaler.py`

---

## 参考になりそうなvXXXシリーズ
- `docs/v456/41_METRICS_INTEGRATION_MEMO.md`
- `docs/v456/47_ZTB_STRUCTURE_ANALYSIS_20260115.md`
- `docs/v456/40_PHASE4_IMPROVEMENT_ANALYSIS_20250114.md`
- `docs/v458/02_training_validation_results.md`
- `docs/v458/00_project_proposal_v458.md`

---

## 改善提案
- AB Testingの出力スキーマ（condition/seed/split/metric）を先に固定し、Comparator/Runner/Reportの入出力を統一する。
- 多重比較補正後の`is_significant`判定を「補正済みαで再計算」に統一し、pairwiseの内部判定と矛盾させない。
- 報酬設計は`compute_hft_reward`にStage別パラメータを渡す方式に寄せ、Env側の分岐増殖を抑える。
- MTF因果性検証は既存のスキーマ/テスト資産を流用し、スクリプト新設なら配置とCI対象を明記する。
- Circuit Breaker相当の挙動は既存の`risk`系ロジック（daily_loss, drawdown）と統合し、二重の保護層を作らない。
- 4 seed実験の工数見積もりに、実行時間と並列数の前提（CPU/GPU、並列ジョブ上限）を追記する。

---

## Change Summary
- Phase 3の前提実装（ABTestingComparator/compute_descriptive_stats）と実コードの整合性が不足。
- Circuit Breaker / Scaler / Rewardの設計が現行APIとズレており、着手前に対象モジュールの再定義が必要。
