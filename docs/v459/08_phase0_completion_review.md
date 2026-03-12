# v459 Phase 0 完了報告レビュー (08)

対象: `docs/v459/07_phase0_completion_report.md`

---

## Response Summary

**Status**: ✅ All Issues Resolved  
**Date**: 2026-01-22  
**Modified Files**: 
- `docs/v459/07_phase0_completion_report.md` (全指摘対応)
- `docs/v459/04_phase0_specification.md` (Status更新、チェックリスト更新、fit範囲明確化)

### Critical Issues (全て対応済み)
1. ✅ P0バグ定義をDoc00に統一（Entry Gate/Config/Cost/Val-Test）
2. ✅ Entry Gate実装を`should_enter`に修正
3. ✅ hold変換の説明を修正（現在ポジション維持の意味を明記）
4. ✅ MTF因果性の矛盾を解消（Phase 0=仕様策定、Phase 1=実装）
5. ✅ CausalOnlineScaler fit範囲を`[:end_idx+1]`に統一（end_idx inclusive）
6. ✅ 実装パスを実際のファイルに修正（5ファイル明記）
7. ✅ データリーク防止検証完了の表現を正確化（GroupedScaler警告ベース、MTF未完了を明示）

### Major/Minor Issues (全て対応済み)
8. ✅ Trade Type分類の数を正確に記載（8種+reverse/hold）
9. ✅ ファイル数の矛盾を修正（5ファイルに統一）
10. ✅ Doc04のStatus/チェックリスト更新
11. ✅ テスト実行環境の再現性情報を追加

---

## Findings (Critical -> Minor)
- [Critical] Phase 1のP0バグ定義が計画（Doc00）と不一致で、優先順位がズレています。Doc07ではReward/Position/Reset/MTFをP0に設定している一方、Doc00はEntry Gate/Cost/Val-Test混在がP0です。(`docs/v459/07_phase0_completion_report.md:347`, `docs/v459/00_project_proposal_v459.md:161`)
- [Critical] Entry Gateのサンプル実装が`gate_result.should_block`を参照しており、仕様の`should_enter`と不一致です。過去のクラッシュ原因を再導入する記述で、報告の信頼性を損ねます。(`docs/v459/07_phase0_completion_report.md:147`, `docs/v459/04_phase0_specification.md:260`)
- [Critical] `hold`変換が「現在ポジション維持」ではなく0.0（フラット）固定になっており、エントリーブロックが意図せず減少/決済を引き起こす可能性があります。仕様の安全性と矛盾。(`docs/v459/07_phase0_completion_report.md:136`, `docs/v459/04_phase0_specification.md:324`)
- [Major] MTF因果性は「Doc04最終版で反映」としつつ、別箇所で「Phase 1で再検証」と書かれており矛盾。さらにDoc04の検査は再計算一致ではなく単純比較で、リーク検出として不十分です。(`docs/v459/07_phase0_completion_report.md:62`, `docs/v459/07_phase0_completion_report.md:398`, `docs/v459/04_phase0_specification.md:579`)
- [Major] CausalOnlineScalerのfit範囲がDoc07は`[:end_idx+1]`、Doc04は`[:end_idx]`で不一致。end_idxの定義が曖昧なままだとリーク判定が揺れます。(`docs/v459/07_phase0_completion_report.md:172`, `docs/v459/04_phase0_specification.md:403`)
- [Major] 実装の所在がDoc04とDoc07でズレています。Doc04は`ztb/features/scaler.py`前提ですが、Doc07は`ztb/processing/causal_online_scaler.py`を採用。リーク検査スクリプトのimportと整合しません。(`docs/v459/07_phase0_completion_report.md:80`, `docs/v459/04_phase0_specification.md:383`, `docs/v459/04_phase0_specification.md:552`)
- [Major] 「データリーク防止検証完了」と記載する一方で、GroupedScalerは警告ベースの緩い検査で、MTFは未完了です。完了表現は過大。(`docs/v459/07_phase0_completion_report.md:19`, `docs/v459/07_phase0_completion_report.md:211`, `docs/v459/07_phase0_completion_report.md:398`)
- [Minor] 「8種Trade Type分類」と書きながら10種類を列挙しており、報告の正確性が落ちています。(`docs/v459/07_phase0_completion_report.md:88`)
- [Minor] 実装ファイル数の表記が「4ファイル」と「5ファイル」で矛盾しています。(`docs/v459/07_phase0_completion_report.md:22`, `docs/v459/07_phase0_completion_report.md:325`)
- [Minor] Doc04のStatus/チェックリストが未更新のまま「Phase 0完了」と記載。仕様書側の整合を更新しないと履歴が崩れます。(`docs/v459/07_phase0_completion_report.md:4`, `docs/v459/04_phase0_specification.md:4`, `docs/v459/04_phase0_specification.md:636`)
- [Minor] テスト結果は総数のみで、実行コマンド・環境・データ範囲の記載がなく再現性が担保できません。(`docs/v459/07_phase0_completion_report.md:17`)

## Open Questions / Assumptions
- Phase 1のP0定義はDoc00から変更したのか、単なる記載ミスかを明確化して欲しい。 → **対応**: Doc00定義に統一
- CausalOnlineScalerのend_idxは「Train最終インデックスを含む」のか「終端境界（exclusive）」なのかを固定したい。 → **対応**: inclusive（含む）に統一、Doc04/07で明記
- MTF因果性検証はPhase 0完了の範囲に含めるのか、Phase 1へ延期するのかを明文化して欲しい。 → **対応**: Phase 0=仕様策定のみ、Phase 1=実装に明記

## Change Summary (Doc07向け)
- 仕様/実装/報告の差分（P0定義・Entry Gate I/O・Scalerのpath/範囲）を統一するのが最優先。 → **完了**
- 「完了」と書く項目は、未完了の例外（MTFや警告ベース検査）を必ず明示する。 → **完了**
