# v459 Phase 2 仕様書 再レビュー (17)

**Date**: 2026-01-22  
**Status**: 📝 Review  
**Target**: `docs/v459/12_phase2_specification.md`

---

## Findings (Critical -> Major -> Minor)
- [Critical] Phase 2は「記述統計のみ」と明記している一方で、Comparator設計とテストにMann-Whitney/Cliff’s Deltaが残っており、スコープが矛盾しています。Phase 2で実装する/しないを明確に統一しないと実装が分岐します。`docs/v459/12_phase2_specification.md:44`, `docs/v459/12_phase2_specification.md:488`, `docs/v459/12_phase2_specification.md:562`, `docs/v459/12_phase2_specification.md:892`, `docs/v459/12_phase2_specification.md:903`
- [Critical] AB条件のseed数が「2 seed」前提の記述と「4 seed」設定で食い違っています。比較スクリプトと出力ディレクトリが一致せず、結果集計が破綻します。`docs/v459/12_phase2_specification.md:44`, `docs/v459/12_phase2_specification.md:444`, `docs/v459/12_phase2_specification.md:846`
- [Major] `compute_descriptive_stats()`が呼ばれているのに定義が無く、比較スクリプトが動きません。`docs/v459/12_phase2_specification.md:873`, `docs/v459/12_phase2_specification.md:874`
- [Major] トレードCSV集計で`initial_capital`が未定義のまま使われており、`net_roi`計算が実装不能です。`docs/v459/12_phase2_specification.md:535`
- [Major] TP/SL判定が`tp_threshold`/`sl_threshold`に依存していますが、設定導線と値の定義がありません（常にFalseになりやすい）。さらに`current_price`の参照元が不明で、実装が曖昧です。`docs/v459/12_phase2_specification.md:714`, `docs/v459/12_phase2_specification.md:716`, `docs/v459/12_phase2_specification.md:728`
- [Major] Entry/Exit/Hold理由付きの「完全分類」をPhase 2成果として記載する一方、entry/hold_reasonはPhase 3延期のまま。成果の表現が矛盾します。`docs/v459/12_phase2_specification.md:52`, `docs/v459/12_phase2_specification.md:1055`
- [Minor] Task2のテスト項目は8件あるのに完了条件が「4/4パス」。工数と品質判定が不整合です。`docs/v459/12_phase2_specification.md:754`, `docs/v459/12_phase2_specification.md:766`
- [Minor] Phase 1のテスト数が「103」と「94」で混在しています。Phase 0/1の包含関係を明確に書き分けないと信頼性が落ちます。`docs/v459/12_phase2_specification.md:18`, `docs/v459/12_phase2_specification.md:965`

---

## 既存実装の活用 / 重複の指摘
- **TP/SL判定の重複**: 簡易`tp_threshold/sl_threshold`を新設するより、既存リスク判定を流用した方が保守性が高いです。候補: `ztb/risk/rules.py`, `ztb/trading/risk/backtest_risk_manager.py`, `ztb/trading/risk/risk_manager.py`  
- **AB実行基盤の重複**: 新規比較スクリプトより、既存のABランナーを再利用して集計層だけ追加する方が安全です。候補: `tools/ab_test_runner.py`, `tools/run_ab_searches.py`, `experiments/v450/run_ab_test_threshold_v450.py`
- **メトリクス集計の重複**: `BacktestReporter`の集計や既存の比較ユーティリティがあるため、`_compute_metrics_from_trades`の独自実装は二重化リスク。候補: `ztb/analysis/baseline_comparison.py`
- **TrainingReporter統合の重複**: 旧2実装は残しつつ互換ラッパーを一定期間維持する方が安全（破壊的変更の拡散を防止）。対象: `ztb/training/unified_trainer/components/reporter.py`, `ztb/training/unified_trainer/reporting.py`

---

## Extensibility / Maintainability 提案
- **メトリクス計算を共通化**: Reporter/AB comparator/基準比較の計算を共通ユーティリティにまとめ、指標定義のズレを防止。
- **close_reasonのデータフロー明確化**: env→evaluator→reporterの伝搬を明記し、逆流や欠損を防止。
- **AB結果のスキーマ固定**: `SeedResult`/summary CSVの項目を固定し、Phase 3の統計検定追加でも後方互換性を保つ。

---

## Open Questions / Assumptions
- `tp_threshold/sl_threshold`はどの設定（既存 `stop_loss_pct/take_profit_pct` 等）を参照する想定ですか？
- `initial_capital`はどこから取得しますか（config/Reporter/CSV）？
- Phase 2は記述統計のみで良い場合、`compare_two_conditions`と統計テストの実装はPhase 3に完全移管して良いですか？

---

## Change Summary (Doc12向け)
- AB TestingのPhase 2スコープ（記述統計のみ）に合わせて、Comparator/テスト/設定の整合を取る。
- TP/SL判定は既存のリスク/ルール実装に寄せ、設定導線を一本化する。
- `compute_descriptive_stats`と`initial_capital`の定義を確定し、比較スクリプトが実行可能な形にする。
