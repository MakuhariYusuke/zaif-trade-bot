# v459 Phase 2 仕様書レビュー (14)

対象: `docs/v459/12_phase2_specification.md`（前回の積み残し含む）

## Findings (Critical -> Minor)
- [Critical] Phase 1完了を前提に進めていますが、Doc07/Doc04で指摘済みのEntry Gate I/O不整合（`should_block`参照）とhold変換の挙動が残っており、P0完了の根拠が揺れています。Phase 2に入る前提が成立していません。`docs/v459/12_phase2_specification.md:6`、`docs/v459/12_phase2_specification.md:14`、`docs/v459/07_phase0_completion_report.md:136`、`docs/v459/07_phase0_completion_report.md:149`、`docs/v459/04_phase0_specification.md:298`、`docs/v459/04_phase0_specification.md:324`
- [Critical] P1-1の内容がDoc00のP1定義（“close”の明示処理）とズレています。Phase 2の計画では“理由付与”に軸足があり、**本来のP1バグが未解決のまま**になる可能性が高い。`docs/v459/12_phase2_specification.md:65`、`docs/v459/12_phase2_specification.md:172`、`docs/v459/00_project_proposal_v459.md:176`
- [Major] 既存TradeTypeの前提がDoc04と一致していません。Doc12は`long_entry_win/loss`等を既存分類として扱っていますが、Doc04は`long_open/long_close/long_add/...`の設計です。基準がズレたまま拡張するとテストも設計も破綻します。`docs/v459/12_phase2_specification.md:74`、`docs/v459/12_phase2_specification.md:141`、`docs/v459/04_phase0_specification.md:147`
- [Major] `exit_reason/entry_reason/hold_reason`は追加されるものの、どの層（env/evaluator）で生成するかが記載されておらず、**実質的に全てNoneのまま**になる懸念があります。`docs/v459/12_phase2_specification.md:174`、`docs/v459/12_phase2_specification.md:193`
- [Major] AB Testingの統計仕様がDoc00と不整合です。Doc12は2-seed成功・two-sided検定・多重比較補正なしで進めていますが、Doc00は4seed×4split・Holm-Bonferroni・効果量を前提としています。`docs/v459/12_phase2_specification.md:41`、`docs/v459/12_phase2_specification.md:430`、`docs/v459/00_project_proposal_v459.md:282`
- [Major] AB Testingは単一条件のseed統合に留まり、**条件A/Bの定義・保存・比較導線**が不足しています。Entry Gate ON/OFFなどの比較が設計上成立していません。`docs/v459/12_phase2_specification.md:361`、`docs/v459/12_phase2_specification.md:419`
- [Major] Reporter統合はTrainingReporter削除が前提ですが、Training側が必要とするメトリクス/APIをBacktestReporterにどう移植するかの仕様が不足しています（破壊的変更リスク）。`docs/v459/12_phase2_specification.md:287`、`docs/v459/12_phase2_specification.md:323`
- [Major] MTF因果性検証とScaler境界の厳密化がPhase 3に後ろ倒しされています。前回の積み残しがPhase 2計画に反映されておらず、評価基盤の前提が弱いままです。`docs/v459/12_phase2_specification.md:736`、`docs/v459/07_phase0_completion_report.md:366`
- [Minor] テスト総数の整合が取れていません（Phase 1=103/103 vs 196/196、または123/123）。報告の信頼性が下がります。`docs/v459/12_phase2_specification.md:18`、`docs/v459/12_phase2_specification.md:623`、`docs/v459/12_phase2_specification.md:670`
- [Minor] 「基本8種」と書きながら列挙は12種です。表現の誤差が仕様理解を混乱させます。`docs/v459/12_phase2_specification.md:67`、`docs/v459/12_phase2_specification.md:77`

## Open Questions / Assumptions
- Phase 1の完了はDoc07/Doc04の矛盾を解消したうえで確定したのか？
- TradeTypeの基準はDoc04の分類（open/close/add/reduce）を正とするのか、Doc12の勝敗ベース分類に置き換えるのか？
- `exit_reason/entry_reason/hold_reason`はどの層で生成し、どう定義するのか？（Manual/Reversalの判定基準含む）
- AB TestingはDoc00の統計仕様に合わせるのか、Phase 2で簡易版に落とすのか？
- MTF因果性検証はPhase 2に戻すのか、それともPhase 3へ正式延期として明文化するのか？

## Change Summary (Doc12向け)
- Phase 1完了の根拠と未解決P0（Entry Gate/MTF/Scaler境界）を整理し、Phase 2開始条件を再定義する。
- P1-1はDoc00の「close明示処理」を必ず満たす形に戻し、TradeTypeの基準をDoc04と統一する。
- 追加フィールド（exit/entry/hold reason）の**生成経路**と定義を明確化し、実装がno-opにならないよう設計する。
- AB TestingはDoc00の検定仕様とサンプル設計に合わせるか、簡易版なら明確に“暫定”と記載する。
