# v459 Phase 2 仕様書 再レビュー (16)

**Date**: 2026-01-22  
**Status**: 📝 Review  
**Target**: `docs/v459/12_phase2_specification.md`

---

## Findings (Critical -> Major)
- [Critical] `close_reason`の判定順が`net_pnl`優先のため、反転（reversal）がほぼ成立しません。さらに`tp/sl`をPnL符号で決めると手動/時間切れのcloseが誤ラベル化します。反転判定を先にし、`tp/sl`は明示的なexitトリガーに紐づける必要があります。`docs/v459/12_phase2_specification.md:217`, `docs/v459/12_phase2_specification.md:642`
- [Major] AB比較スクリプトが`compare_with_condition`を呼んでいますが、ComparatorにそのAPI定義がありません。設計のままでは実装が破綻します。`docs/v459/12_phase2_specification.md:767`, `docs/v459/12_phase2_specification.md:517`
- [Major] AB Testingを2seedで実施しつつ統計検定まで行う設計は過小サンプルで信頼性が低いです。Phase 2は「記述統計のみ」と明記するか、Phase 3へ統計判定を全面移管すべきです。`docs/v459/12_phase2_specification.md:44`, `docs/v459/12_phase2_specification.md:747`, `docs/v459/12_phase2_specification.md:777`
- [Major] `min_action_threshold`がAB条件例に登場しますが、既存GateConfig仕様（min_confidence/edge_threshold）と不一致です。設定が無効化されるリスクがあります。`docs/v459/12_phase2_specification.md:420`, `docs/v459/04_phase0_specification.md:270`
- [Major] 「Entry/Exit/Hold理由付きの完全分類」と書きつつ、entry/hold_reasonはPhase 3延期のままです。完了条件と成果記載が矛盾しています。`docs/v459/12_phase2_specification.md:52`, `docs/v459/12_phase2_specification.md:941`
- [Major] Reporter統合でTrainingReporterを削除する一方、Training側が必要とするメトリクス/呼び出し規約の移植方針が不足しています。互換性が不明なまま削除計画が進みます。`docs/v459/12_phase2_specification.md:317`, `docs/v459/12_phase2_specification.md:340`

## Findings (Minor / Gaps)
- [Minor] 工数見積もりが「5-6日」と「合計4日」で不一致です。`docs/v459/12_phase2_specification.md:54`, `docs/v459/12_phase2_specification.md:611`
- [Minor] テスト総数の記述が103/103・196/196・123/123で揺れています。Phase 1/Phase 0の包含関係を明記しないと混乱します。`docs/v459/12_phase2_specification.md:18`, `docs/v459/12_phase2_specification.md:45`, `docs/v459/12_phase2_specification.md:844`, `docs/v459/12_phase2_specification.md:855`
- [Minor] `close_reason`のテスト計画に「反転（reversal）」「PnL=0」「手動close」ケースがなく、最も壊れやすい分岐が未検証です。`docs/v459/12_phase2_specification.md:674`
- [Minor] AB Testing ComparatorがCSVから`val_metrics`をどう生成するか未定義です（trades CSVかsummary CSVかが不明）。実装/テストの前提が揺れます。`docs/v459/12_phase2_specification.md:381`, `docs/v459/12_phase2_specification.md:589`

---

## Open Questions / Assumptions
- close_reasonは「TP/SLのトリガーイベント」が存在する前提で良いか？ない場合は分類名の再定義が必要。
- close_reasonの生成はenvだけで完結するのか、evaluatorで補足情報（例: 反転分解）を付与するのか？
- AB TestingはPhase 2では探索的（記述統計のみ）として扱う前提で良いか？
- TrainingReporter削除に伴い、trainer側が期待するメトリクス（return/length/success_rate）をどのAPIで提供するか？

---

## 既存実装 / 過去vXXXの活用提案
- AB実験の実行基盤は既存ツールを流用し、比較レポート生成だけ拡張する方が安全: `tools/ab_test_runner.py`, `tools/run_ab_searches.py`, `experiments/v450/run_ab_test_threshold_v450.py`
- seed安定性の基準やリスク記述は既存ドキュメントを再利用すると一貫性が出る: `docs/v457/32_seed_stability_test.md`, `docs/v457/34_seed_stability_lost_alpha_review.md`
- Reporter/評価の既知バグ回避（close扱い・PnL二重計上）は過去レビューのチェックリストを踏襲: `docs/v458/19_phase5_6_final_review.md`
- TrainingReporter統合は既存APIの洗い出しを優先し、移行期間はラッパー残置を推奨: `ztb/training/unified_trainer/components/reporter.py`, `ztb/training/unified_trainer/reporting.py`
- Baseline比較や評価指標の整備は既存実装の再利用で工数圧縮: `ztb/analysis/baseline_comparison.py`

---

## Change Summary (Doc12向け)
- close_reasonの判定順と意味付けを修正（反転優先・TP/SLはトリガー由来に変更）。
- AB TestingはPhase 2では「探索的統計のみ」に限定し、比較API/設定キーを実装整合に合わせる。
- TrainingReporter統合の互換APIを明文化し、削除は移行完了後に実施。
- 工数・テスト数の表記を一本化し、完了条件の信頼性を担保する。
