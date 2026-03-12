# v459 Phase 2 完了報告レビュー (19)

**Date**: 2026-01-22  
**Status**: 📝 Review  
**Targets**: `docs/v459/18_phase2_completion_report.md`, `docs/v459/12_phase2_specification.md`

---

## Findings (Critical -> Major -> Minor)
- [Critical] Phase 2完了報告ではP1-4（AB Testing）をPhase 3延期と明記していますが、Doc12ではPhase 2で「2条件×2seedの記述統計」まで実施する計画でした。完了報告が仕様未達のままになっています。`docs/v459/18_phase2_completion_report.md:20`, `docs/v459/18_phase2_completion_report.md:258`, `docs/v459/12_phase2_specification.md:44`
- [Critical] Doc12はPhase 2を「記述統計のみ」に限定したのに、Comparator/API/テストに統計検定が残っています。仕様と実装範囲のズレが継続中です。`docs/v459/12_phase2_specification.md:44`, `docs/v459/12_phase2_specification.md:488`, `docs/v459/12_phase2_specification.md:562`, `docs/v459/12_phase2_specification.md:892`
- [Major] AB条件のseed数が2/4で混在しています。Doc12の設定例は`[0,1,2,3]`ですがPhase 2完了条件は2seed前提のままです。比較出力が壊れる可能性があります。`docs/v459/12_phase2_specification.md:44`, `docs/v459/12_phase2_specification.md:444`, `docs/v459/12_phase2_specification.md:846`
- [Major] Doc12の比較スクリプトで`compute_descriptive_stats()`を呼んでいますが、API定義がなく実装不能です。`docs/v459/12_phase2_specification.md:873`
- [Major] Doc12のメトリクス計算に`initial_capital`が未定義のまま使用されています。`net_roi`が計算できません。`docs/v459/12_phase2_specification.md:535`
- [Major] Doc18のテストログに`test_close_reason_priority_reversal_first`が残っており、本文の「tp/sl優先」方針と矛盾しています（命名ミスかテスト基準の不一致）。`docs/v459/18_phase2_completion_report.md:91`, `docs/v459/18_phase2_completion_report.md:174`
- [Major] Doc18はReporter統合を完了と記載していますが、Doc12が求めるTrainingReporter統合（2実装削除・互換API移植）の実施証跡がありません。統合範囲の実績が不明です。`docs/v459/18_phase2_completion_report.md:17`, `docs/v459/12_phase2_specification.md:317`
- [Major] 既存テストで依存関係エラーがある一方で「後方互換性に影響なし」と断定しています。回帰テストの状態が曖昧です。`docs/v459/18_phase2_completion_report.md:190`, `docs/v459/18_phase2_completion_report.md:193`
- [Minor] Doc18の作成年が2025年となっており、v459タイムライン（2026）と不一致です。`docs/v459/18_phase2_completion_report.md:5`, `docs/v459/18_phase2_completion_report.md:318`
- [Minor] Task2のテスト項目は8件あるのに完了条件が「4/4」と書かれており、品質判定が不整合です。`docs/v459/12_phase2_specification.md:754`, `docs/v459/12_phase2_specification.md:766`
- [Minor] Phase 1のテスト数が「103/103」と「94」で混在しています。包含関係を明確にしないと信頼性が落ちます。`docs/v459/12_phase2_specification.md:18`, `docs/v459/12_phase2_specification.md:965`

---

## 既存実装の活用 / 重複の指摘
- **TP/SL判定の重複**: `tp_threshold/sl_threshold`を新設するより、既存のリスク/判定ロジックを再利用した方が保守性が高いです。候補: `ztb/risk/rules.py`, `ztb/trading/risk/backtest_risk_manager.py`, `ztb/trading/risk/risk_manager.py`
- **AB実行基盤の重複**: 新規比較スクリプトを作る前に既存ABランナーを流用し、集計レイヤだけ追加する方が安全です。候補: `tools/ab_test_runner.py`, `tools/run_ab_searches.py`, `experiments/v450/run_ab_test_threshold_v450.py`
- **メトリクス集計の重複**: Reporter/AB comparatorで個別に指標計算を持つと不一致が出ます。既存の評価ユーティリティを再利用するか共通関数化を推奨。候補: `ztb/analysis/baseline_comparison.py`
- **Reporter統合の重複**: TrainingReporter削除は互換API移植完了後に行い、短期はラッパー維持が安全です。対象: `ztb/training/unified_trainer/components/reporter.py`, `ztb/training/unified_trainer/reporting.py`

---

## Extensibility / Maintainability 提案
- **AB結果のスキーマ固定**: seed/condition/metricのCSVスキーマをPhase 2で固定し、Phase 3で統計検定を追加しても互換性を維持。
- **close_reasonデータフロー明確化**: env→evaluator→reporterの伝搬を1本化し、逆流・欠損を防止。
- **指標計算の共通化**: `net_roi`, `win_rate`等を共通ユーティリティで一元管理し、Doc/実装/テストのズレを低減。

---

## Open Questions / Assumptions
- Doc12のAB TestingはPhase 2で「記述統計のみ」に統一して良いですか？（統計検定はPhase 3へ完全移管）
- `initial_capital`はどの設定・CSV・Reporterから取得する想定ですか？
- TrainingReporter削除はPhase 2完了条件に含めるべきですか？それともPhase 3へ繰り延べますか？

---

## Change Summary (Docs 12/18向け)
- Doc18の実績をDoc12のスコープに合わせるか、Doc12側を「P1-4延期」に更新して整合を取る。
- AB Testingのseed数・API定義・統計範囲を一本化し、比較スクリプトを実行可能な形にする。
- テスト数/回帰テストの記載を整理し、互換性の根拠を明確化する。

---

## Implementation Review (Phase 2 code)

### Findings (Critical -> Major -> Minor)
- [Critical] `FastIntradayEnvV456`が`self.recorder.record_trade()`を旧シグネチャで呼び出しており、`WalkForwardReporter`を`eval_env.recorder`に設定するとTypeError/二重記録の可能性があります。`ztb/trading/environment/fast_intraday_env_v456.py:807` `ztb/trading/environment/fast_intraday_env_v456.py:858` `ztb/evaluation/walk_forward/evaluator.py:403` `ztb/evaluation/walk_forward/reporter.py:338`
- [Major] ポジション変更時に常に「前ポジション全量の決済PnL」を計算し、`entry_price`を新価格に上書きしています。`long_add/long_reduce`でも全量決済扱いになり、PnL/統計が歪みます。`ztb/trading/environment/fast_intraday_env_v456.py:667` `ztb/trading/environment/fast_intraday_env_v456.py:686` `ztb/evaluation/walk_forward/reporter.py:20` `ztb/trading/environment/components/fast_intraday_action_processor.py:34`
- [Major] 反転時の`entry_price`が新規側に更新された後に`info`へ格納されるため、Evaluatorが旧ポジションのエントリー価格を失い、クローズ側の記録が不正になります。`ztb/trading/environment/fast_intraday_env_v456.py:686` `ztb/trading/environment/fast_intraday_env_v456.py:843` `ztb/evaluation/walk_forward/evaluator.py:440`
- [Major] 反転取引のPnL/コストを半分ずつ分配しており、`trade_pnl`の意味（クローズ側のNET PnL）とズレます。統計の歪みにつながるため配賦ルールを再検討してください。`ztb/evaluation/walk_forward/reporter.py:387` `ztb/trading/environment/fast_intraday_env_v456.py:667`
- [Minor] `test_close_reason_reversal_on_position_flip`は反転=必ずreversalを期待しますが、実装はTP/SL優先のため価格推移によっては不安定です。`tests/unit/trading/test_close_reason.py:106` `ztb/trading/environment/fast_intraday_env_v456.py:672`

---

## 既存実装の活用 / 重複の指摘 (Implementation)
- **entry_price/部分約定のロジック重複**: 既存の平均取得価格・部分クローズ処理を使えば、`FastIntradayEnvV456`のentry_price上書き問題を回避できます。候補: `ztb/trading/production/virtual_portfolio_manager.py:266` `ztb/trading/production/virtual_portfolio_manager.py:352`
- **PositionManagerの再利用**: `live_trader`側で「entry_price上書きバグ対策」が実装済みです。バックテスト環境でもPositionManagerに寄せると保守性が上がります。`ztb/trading/live_trader/live_trader.py:1663` `ztb/trading/environment/components/position_manager.py:511`
- **trade記録パイプラインの重複**: envとevaluatorの二重記録を避け、単一路に統一した方が安全です。`ztb/trading/environment/fast_intraday_env_v456.py:807` `ztb/evaluation/walk_forward/evaluator.py:448`

---

## Extensibility / Maintainability 提案 (Implementation)
- **Tradeイベントの単一路化**: `env`が`position_before/after`と`prev_entry_price`をinfoに載せ、`reporter`側で一元処理する構成にすると拡張が容易です。
- **add/reduceの明確化**: 「部分増減を許す」なら加重平均entry_priceを導入し、部分決済PnLを正しく分離する設計に寄せるのが安全です。
- **reversalの配賦設計**: 反転は「クローズPnL + 新規エントリーコスト」に分解し、open側にはPnLを載せないルールにすると一貫します。

---

## Open Questions / Assumptions (Implementation)
- `FastIntradayEnvV456`の「ポジション変更は全量クローズ扱い」は意図された仕様でしょうか？（`long_add/long_reduce`分類と整合が必要）
- 反転時に旧entry_priceが必要な分析（例:損益率、指標出力）はありますか？あれば`prev_entry_price`の伝搬が必要です。
