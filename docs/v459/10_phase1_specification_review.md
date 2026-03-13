# v459 Phase 1 仕様書レビュー (10)

対象: `docs/v459/09_phase1_specification.md`（+ Doc07の残件）

## Findings (Critical -> Minor)
- [Critical] Phase 1のP0定義がDoc07と食い違ったままで、優先順位が二重管理になっています（Doc09はDoc00準拠、Doc07は別P0セット）。`docs/v459/09_phase1_specification.md:15`、`docs/v459/07_phase0_completion_report.md:347`、`docs/v459/00_project_proposal_v459.md:161`
- [Critical] P0-1/2を「Phase 0対応済み」とする根拠が、Doc07の実装例と仕様に矛盾（`should_block`参照、hold=0.0固定）しており、未修正のまま通過している可能性があります。`docs/v459/09_phase1_specification.md:19`、`docs/v459/07_phase0_completion_report.md:136`、`docs/v459/07_phase0_completion_report.md:147`、`docs/v459/04_phase0_specification.md:298`、`docs/v459/04_phase0_specification.md:324`
- [Major] MTF因果性の扱いがPhase 1→Phase 2へ事実上移動しており、Doc07の「Phase 1で実施」と矛盾します。P0-4の完了定義が揺れるため、スコープ変更を明文化すべきです。`docs/v459/09_phase1_specification.md:165`、`docs/v459/07_phase0_completion_report.md:366`
- [Major] Scalerのfit境界と実装パスがドキュメント間で不一致です（inclusive/exclusive、`ztb/features` vs `ztb/processing`）。リーク検査とテストが同じ対象を見ていない可能性があります。`docs/v459/09_phase1_specification.md:91`、`docs/v459/07_phase0_completion_report.md:80`、`docs/v459/04_phase0_specification.md:383`、`docs/v459/04_phase0_specification.md:552`
- [Major] PnL規約は「pnl=net」を掲げつつ、Reporterの`gross_pnl`計上が同値のままで、将来の分析指標が誤解される構造です。後方互換の影響も未整理。`docs/v459/09_phase1_specification.md:54`、`docs/v459/09_phase1_specification.md:115`、`docs/v459/09_phase1_specification.md:296`
- [Major] Val/Test分離の実装・テストが「環境IDが違う」レベルに留まり、prewarmの境界（train_end_idx）やscalerのfit範囲が明示されていません。これではリークの検出力が弱い。`docs/v459/09_phase1_specification.md:160`、`docs/v459/09_phase1_specification.md:171`

## Open Questions / Assumptions
- Phase 1のP0定義はDoc00を正式採用で良いか、それともDoc07のP0セットを維持するのか。
- MTF因果性検証はPhase 1に残すのか、Phase 2へ正式に延期するのか。
- `trade_pnl`は「取引単位の純PnL」なのか「ステップ差分の純PnL」なのか、どちらを規約化するのか。
- Scalerのfit範囲はinclusive/exclusiveどちらで固定し、実装パスは`ztb/processing`で統一するのか。

## Change Summary (Doc09向け)
- Doc07とDoc09のP0定義・完了状況を統合し、単一の優先順位表に揃える。
- PnL規約に合わせてReporterの`gross_pnl`扱いを整理し、後方互換性の方針を明記する。
- Val/Test分離でのscaler fit境界・prewarm手順を具体化し、リーク検出テストを強化する。
