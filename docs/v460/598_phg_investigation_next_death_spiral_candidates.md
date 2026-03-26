# 598# 次のデススパイラル候補 調査メモ

## 位置付け

この系統の cleanup / 調査は、以後この `598#` に集約して更新する。

- 597# は 190# A/B + `one_sided_balance` の個別 cleanup 記録
- 598# は、その次に再燃しうる候補の調査・判断・追記の正本

## 背景

596# で primary safety valve を導入し、597# で 190# A/B と `one_sided_balance`
に dead-code 注釈を付けた。

ここで次に危ないのは、

- runtime では実質使われない
- しかし config / YAML / parser / テストだけ残る

という「保守判断を誤らせる死コード予備軍」である。

## 調査結果

### A. `inventory_escape_*` / `recovery_skew_*`

対象:

- `scripts/v460/lib/fill_config.py`
- `scripts/v460/lib/fill_config_parser.py`
- `configs/v460/fill_test.yaml`
- `tests/unit/v460/test_285_split_brain_guard.py`
- `tests/unit/v460/test_346_fill_config_validation.py`

観察:

1. runtime 側での参照は実質消えている
2. parser / YAML / validation test は後方互換のため残っている
3. `fill_config.py` 自体にも `522# 撤廃` と明記されている

判断:

- **dead-code 予備軍ではなく、ほぼ dead-code 本体**
- ただし parser と validation test が existence / backward-compat を見ているため、
  いきなり削除するより
  **注釈で「読み取り専用の残置」であることを固定する** 方が安全

今回の対応:

- `fill_config.py` に `DEAD CODE (598#)` 注釈を追加
- `fill_config_parser.py` に後方互換読み取り専用の注釈を追加
- `test_285_split_brain_guard.py` / `test_346_fill_config_validation.py` の説明も
  「runtime で効く前提」から「legacy read-only field の存在確認」へ補正

### B. `skip_gate_ev_*` 旧 hard-gate 系

対象:

- `skip_gate_ev_max_consecutive_skip`
- `skip_gate_ev_one_sided_threshold_shift`
- `skip_gate_ev_weighted.py` の旧 hard-gate 分岐

観察:

1. 本番 YAML では `ev_as_offset_enabled: true`
2. 旧 hard-gate 分岐は本番主経路では通らない
3. ただし hot-reload / YAML drift / legacy existence test が残る

判断:

- runtime 的には dead-code に近い
- ただし **互換性レイヤとしての存在はまだ意図的**
- 597# の dead-code 注釈で、現時点では十分

### C. `execution_additive_enabled` と `experimental_additive_pipeline`

対象:

- `fill_config.py`
- `fill_config_parser.py`
- `config_hot_reload.py`
- `fill_record_builder.py`
- `analyze_fill_logs.py`

観察:

1. logic 分岐は `experimental_additive_pipeline`
2. outward telemetry は `execution_additive_enabled`
3. parser 側に mismatch warning がある
4. hot-reload には両方が残っている

判断:

- これは dead-code ではなく **二重意味の温床**
- ただし現状は
  - ロジック
  - telemetry
  の役割分離が入っているので、直ちに危険ではない
- 次に危険化するなら、docs/test がこの役割分離を忘れた時

## 型安全の確認

次の targeted mypy を実行し、今回の調査対象の config 線に新しい型崩れがないことを確認した。

```bash
.venv/Scripts/python.exe scripts/quality/run_targeted_mypy.py \
  scripts/v460/lib/fill_config.py \
  scripts/v460/lib/fill_config_parser.py \
  scripts/v460/lib/config_hot_reload.py
```

結果:

```text
Success: no issues found in 3 source files
```

## まとめ

優先度順では次の通り。

1. `inventory_escape_*` / `recovery_skew_*`
   - dead-code 注釈を付けた上で、将来削除候補として整理
2. `skip_gate_ev_*`
   - 597# の注釈で現状十分、次は docs/test の意味付け整理
3. additive dual-line
   - dead-code ではなく contract drift 防止が主眼

## 追加確認

### additive dual-line の守り

次の 2 点をテストで固定した。

1. `execution_additive_enabled` は telemetry 契約として hot-reload 対象に残す
2. `inventory_escape_*` / `recovery_skew_*` は legacy read-only 扱いのため hot-reload 対象にしない

また、parser は

- `execution_additive_enabled`
- `experimental_additive_pipeline`

が不一致のとき warning を出すことで、
「logic は additive、telemetry は multiplicative」などの読み違いを検知できる。

### analysis / observability 側の low-risk 型 sweep

`hindsight_filter.py` は TypedDict の整理範囲が大きく、別タスク向きだった。
一方で次の 2 本は low-risk に targeted mypy を進められると判断した。

- `compare_regime_ab.py`
  - detector subclass の内部属性型を宣言
- `side_regime_dashboard.py`
  - YAML judgment mapping と per-regime criteria の cast を明示

## 次の自然な一手

1. `inventory_escape_*` / `recovery_skew_*` を「読み取り専用 legacy field」として docs 側にも明記
2. `skip_gate_ev_*` の existence/hot-reload test を、legacy compatibility 契約として整理
3. additive dual-line については telemetry と logic の責務分離を保つ focused test を追加

### analysis / observability 側の low-risk 型 sweep (追加)

`hindsight_filter.py` は TypedDict の総整理が重いため、まず low-risk な第1段だけ入れた。

- `scripts/v460/analysis/hindsight_filter.py`
  - `_print_report(...)` を section helper 群へ分割
  - `SkipGateCalibrationReport` の union 参照を `cast` で明示
  - JSON 出力は `analysis_common.write_json_output(...)` を再利用

効果:

- `_print_report(...)` 内の TypedDict 混線が解消
- targeted mypy を semantics 変更なしで clean 化
- 次の batch では I/O 共通化 (`args`, record loading, output policy) をより安全に進められる

さらに第2段として、CLI / I/O も `analysis_common` に寄せた。

- `scripts/v460/analysis/hindsight_filter.py`
  - `load_records_from_args(...)` を利用して record loading を共通化
  - `--start/--end/--data-dir` は維持しつつ
    - `date_from/date_to/results_dir`
    へ alias 解決して shared loader に渡す
  - `add_output_args(...)` を再利用して output 契約も統一

効果:

- `hindsight_filter.py` 独自の `_load_records(...)` 依存を外せた
- 既存 CLI 互換を維持したまま、他 analysis script と同じ loader/output policy に寄せられた
- 今後は `analysis_common` の helper 拡張を `hindsight_filter.py` にも横展開しやすい

`analysis_common` の横展開先も見えてきた。

- `stopgap_daily_report.py`
  - loader 自体は `stopgap_health` 固有のため維持
  - ただし
    - `add_common_filter_args(...)`
    - `add_output_args(...)`
    - `write_json_output(...)`
    はそのまま再利用可能
- `print_ab_summary.py`
  - record loader は不要
  - しかし output helper / JSON read helper の共通化余地はある

この時点の方針:

1. loader を shared 化できる script は `load_records_from_args(...)`
2. loader は専用のままでも、filter/output 契約は `analysis_common` に寄せる
3. script 固有の集計や report formatting は無理に共通化しない

2026-03-27 追加棚卸し:

- `tests/unit/v460/_real_data_test_helpers.py`
  - `write_jsonl_sample(...)` / `write_jsonl_gz(...)` は複数 real-data test の実質共通 helper
  - まずは test helper のまま `JsonRow = dict[str, object]` へ寄せて `Any` を減らす
  - その上で reuse 範囲が `tests/` を超えるなら `ztb.io.jsonl` 周辺へ移管を検討
- `analyze_fill_logs.py` / `fill_quality.py`
  - observability / metrics bridge の基準点
  - 次の batch では
    - output helper
    - typed record alias
    - execution telemetry field
    を同じ基準で揃える
- `maker_price.py` / `ab_judgment.py`
  - Wave 2 の残差として、god object 化を再燃させやすい inline ownership 点だけを棚卸し対象にする
  - 大分割ではなく
    - veto/cache/telemetry
    - result/report
    の update ownership を薄くするところで止める

2026-03-26 追加:

- `reproduce_152_metrics.py`
- `oracle_test.py`
- `oracle_baseline.py`
- `tail_loss_analysis.py`

について、人向け report 出力も `analysis_common.write_output(...)` へ寄せた。

判断:

1. JSON 出力はすでに `write_json_output(...)` へ揃っていた
2. 次の drift は text/report 側で起きやすい
3. ここを buffer 化して helper へ統一すると
   - stdout / file 出力の扱いが一貫する
   - test で monkeypatch しやすい
   - targeted mypy にも悪影響がない

このため、analysis batch の次の基準形は

- machine-readable: `write_json_output(...)`
- human-readable: `write_output(...)`

の 2 本立てとする。

2026-03-25 追加棚卸し:

- `print_ab_summary.py`
  - 低リスクな型残差のみ存在
  - `main() -> None` と JSON row/object の判定を入れれば targeted mypy で片付く
- `sha_comparison.py`
  - container の型推論が揺れていて、`dict[str, list[float]]` と `dict[str, list[dict]]` が混ざっている
  - こちらは helper 化より先に集計 container の型整理が必要

したがって次の優先順は:

1. `print_ab_summary.py` の low-risk typing
2. `sha_comparison.py` の集計 container 整理
3. その後に `analysis_common` の追加横展開を検討

## 2026-03-25 追加追記: `sha_comparison.py` first batch

`sha_comparison.py` は当初見立てどおり、helper 再利用より先に
集計 container の型固定が必要だった。

今回の対応:

- `RegimeBuckets`
- `HourlyBuckets`
- `DailyBuckets`

を導入し、

- regime 別 `filled/all`
- UTC hour 別 `pnls/sell_pnls/buy_pnls`
- day 別 `total/filled/pnls/bf`

の shape を固定した。

また、JSON 出力は `analysis_common.write_json_output(...)` に寄せた。

効果:

1. `dict[str, dict]` / `defaultdict` 由来の型崩れを局所的に止められた
2. `sha_comparison.py` が targeted mypy clean になった
3. analysis script の共通化は
   - loader まで shared 化するもの
   - output helper だけ shared 化するもの
   の 2 系統で十分だと再確認できた

追加した focused test:

- `tests/unit/v460/test_sha_comparison_cli.py`
  - `run_analysis()` の hourly/daily 集計が崩れないこと
  - `main(--json)` が `write_json_output(...)` を使うこと

次の候補:

1. `sha_comparison.py` に `add_output_args(...)` を寄せるかの判断
   - ただし現状は `--json` boolean 契約で十分なので急がない
2. `analysis_common` の横展開先として
   - `sha_comparison.py`
   - `print_ab_summary.py`
   - `stopgap_daily_report.py`
   の 3 本を基準形にする
3. より重い script は I/O と report を先に分離してから着手する

## 2026-03-25 追加追記: analysis output contract batch

`analysis_common` の helper をさらに広げる際、
最も低リスクでまとまりやすいのは「出力 contract」の統一だった。

今回対象:

- `hour_matched_comparison.py`
- `tail_loss_analysis.py`
- `vg_and_trend.py`

判断:

1. いずれも集計ロジックはそのままでよい
2. 差し替える価値があるのは
   - `write_json_output(...)`
   - `write_output(...)`
   - 一部 `argv` 受け取り
3. parser や集計 helper を無理に共通化するより、
   まず output contract を揃える方が安全

今回の対応方針:

- `hour_matched_comparison.py`
  - `add_output_args(...)` を導入
  - `--json` 時の保存は `write_json_output(...)`
- `tail_loss_analysis.py`
  - JSON 保存を `write_json_output(...)` に統一
  - `main()` は `Sequence[str]` でも呼べるようにする
- `vg_and_trend.py`
  - JSON/stdout/text file 出力を
    - `write_json_output(...)`
    - `write_output(...)`
    に統一
  - `main(argv)` で focused test をしやすくする

この batch の意味:

- analysis script の「出力の仕方」がかなり揃う
- focused test は I/O contract だけを薄く守ればよくなる
- 次の横展開は
  - `oracle_baseline.py`
  - `reproduce_152_metrics.py`
  - `ab_offset_comparison.py`
  のような script へ自然に広げられる

## 2026-03-25 追加追記: 次の十数ファイル batch 見立て

以後の analysis 系 sweep は、単発ではなく「同じ改善軸」を持つ束で進める。
現時点で最も見返しやすく、安全にまとめやすいのは次の 3 batch である。

### Batch A: analysis output / CLI contract 統一

対象:

1. `scripts/v460/analysis/ab_offset_comparison.py`
2. `scripts/v460/analysis/oracle_baseline.py`
3. `scripts/v460/analysis/reproduce_152_metrics.py`
4. `scripts/v460/analysis/print_ab_summary.py`
5. `scripts/v460/analysis/oracle_test.py`
6. `docs/v460/598_phg_investigation_next_death_spiral_candidates.md`
7. `docs/v460/037_phg_rpt_refactoring_session_log.md`

改善軸:

- `write_json_output(...)`
- `write_output(...)`
- `main(argv)` 化
- output path / stdout 契約の統一

判断:

- 既存集計ロジックに手を入れずに寄せられる
- 後から見たときも「出力 contract を揃えた commit」として追いやすい

### Batch B: analysis loader / filter contract 統一

対象:

1. `scripts/v460/analysis/ab_offset_comparison.py`
2. `scripts/v460/analysis/reproduce_152_metrics.py`
3. `scripts/v460/analysis/compare_regime_ab.py`
4. `scripts/v460/analysis/analysis_common.py`
5. `tests/unit/v460/test_441_ab_offset_comparison.py`
6. `tests/unit/v460/test_511_shared_contracts_migration.py`
7. `docs/v460/598_phg_investigation_next_death_spiral_candidates.md`
8. `docs/v460/037_phg_rpt_refactoring_session_log.md`

改善軸:

- `add_common_filter_args(...)`
- `load_records_from_args(...)`
- `DEFAULT_RESULTS_DIR`
- filter alias (`--start/--end` vs `--date-from/--date-to`)

判断:

- fill-record 系 script の shared loader 再利用を一段進められる
- ただし domain 固有 loader が強い script は無理に寄せない

## 2026-03-25 追加追記: Batch B 第1段

実施対象:

1. `scripts/v460/analysis/analysis_common.py`
2. `scripts/v460/analysis/reproduce_152_metrics.py`
3. `scripts/v460/analysis/compare_regime_ab.py`
4. `tests/unit/v460/test_analysis_loader_contracts.py`
5. `tests/unit/v460/test_152_parallel_tasks.py`
6. `docs/v460/598_phg_investigation_next_death_spiral_candidates.md`
7. `docs/v460/037_phg_rpt_refactoring_session_log.md`

今回の寄せ方:

- `analysis_common.add_common_filter_args(...)`
  - `include_legacy_aliases=True` で
    - `--data-dir`
    - `--start`
    - `--end`
    を吸収できるようにした
- `reproduce_152_metrics.py`
  - main の record loading を `load_records_from_args(...)` に統一
  - 既存 CLI 互換は alias で維持
- `compare_regime_ab.py`
  - main の record loading を `load_records_from_args(...)` に統一
  - 既存 CLI 互換は alias で維持

判断:

- parser/loader の shared contract を先に作ると、他 script へも横展開しやすい
- `_load_records(...)` 相当のローカル helper は test/compat 用に残してもよいが、
  main entrypoint は shared contract を通す方が読みやすい

## 2026-03-25 追加追記: results-dir parser contract の横展開

対象:

1. `scripts/v460/analysis/analysis_common.py`
2. `scripts/v460/analysis/oracle_baseline.py`
3. `scripts/v460/analysis/oracle_test.py`
4. `scripts/v460/analysis/vg_and_trend.py`
5. `scripts/v460/analysis/ab_offset_comparison.py`
6. `tests/unit/v460/test_analysis_results_dir_contracts.py`
7. `tests/unit/v460/test_analysis_batch_a_cli_contracts.py`
8. `tests/unit/v460/test_analysis_output_contracts.py`
9. `docs/v460/598_phg_investigation_next_death_spiral_candidates.md`
10. `docs/v460/037_phg_rpt_refactoring_session_log.md`

今回の寄せ方:

- `analysis_common.add_results_dir_arg(...)` を新設
- `oracle_baseline.py`
- `oracle_test.py`
- `vg_and_trend.py`
- `ab_offset_comparison.py`
  の `--results-dir` 定義を shared helper に統一

判断:

- 変更内容が単純で review しやすい
- parser 契約の drift を減らせる
- 今後 `results-dir` の default/help を直すときに横展開が不要になる

## 2026-03-25 追加追記: typing / container contract 第1段

対象:

1. `scripts/v460/analysis/oracle_baseline.py`
2. `scripts/v460/analysis/reproduce_152_metrics.py`
3. `scripts/v460/analysis/ab_offset_comparison.py`
4. `scripts/v460/analysis/sha_comparison.py`
5. `tests/unit/v460/test_sha_comparison_cli.py`
6. `docs/v460/598_phg_investigation_next_death_spiral_candidates.md`
7. `docs/v460/037_phg_rpt_refactoring_session_log.md`

今回の寄せ方:

- `oracle_baseline.py`
  - `Optional[float]` を `float | None` に統一
- `reproduce_152_metrics.py`
  - shared `Record` alias を採用
  - `MetricsMap` を `TypeAlias` として明示
  - `_to_dict(...)` / `_print_report(...)` / `params` の型を shared contract に寄せた
- `ab_offset_comparison.py`
  - record container を shared `Record` alias に統一
- `sha_comparison.py`
  - `main(argv)` 化して CLI test と entrypoint 契約を他 script と揃えた

判断:

- runtime 振る舞いを変えずに型/entrypoint だけ揃えられる
- mypy と focused pytest の両方で守りやすい

### Batch C: analysis typing / container 整理

対象:

1. `scripts/v460/analysis/ab_offset_comparison.py`
2. `scripts/v460/analysis/oracle_baseline.py`
3. `scripts/v460/analysis/oracle_test.py`
4. `scripts/v460/analysis/reproduce_152_metrics.py`
5. `tests/unit/v460/test_441_ab_offset_comparison.py`
6. `tests/unit/v460/test_467_remaining_issues.py`
7. `docs/v460/598_phg_investigation_next_death_spiral_candidates.md`
8. `docs/v460/037_phg_rpt_refactoring_session_log.md`

改善軸:

- `TypeAlias`
- `TypedDict`
- `cast`
- `main() -> None / argv`
- targeted mypy clean 化

判断:

- `sha_comparison.py` と同じ粒度で進められる
- `dict[str, dict]` / `defaultdict` の型揺れを先に止めるのが主眼

## 今の推奨順

1. **Batch A**
   - 低リスク
   - commit の意味が明瞭
   - review しやすい
2. **Batch C**
   - output contract が揃った後に型を詰めると差分が読みやすい
3. **Batch B**
   - loader/shared args は script ごとの事情差が大きいため最後

## 今回の結論

次の「十数ファイルを一気にまとめる」なら、
まずは **Batch A: analysis output / CLI contract 統一** が最も良い。

## 2026-03-25 追加追記: Batch A 実施結果

実施対象:

1. `scripts/v460/analysis/print_ab_summary.py`
2. `scripts/v460/analysis/reproduce_152_metrics.py`
3. `scripts/v460/analysis/oracle_baseline.py`
4. `scripts/v460/analysis/oracle_test.py`
5. `scripts/v460/analysis/ab_offset_comparison.py`
6. `tests/unit/v460/test_analysis_batch_a_cli_contracts.py`
7. `tests/unit/v460/test_152_parallel_tasks.py`
8. `tests/unit/v460/test_441_ab_offset_comparison.py`
9. `tests/unit/v460/test_158_oracle_test.py`
10. `docs/v460/598_phg_investigation_next_death_spiral_candidates.md`
11. `docs/v460/037_phg_rpt_refactoring_session_log.md`

今回の寄せ方:

- `print_ab_summary.py`
  - `main(argv)` 化
  - `write_output(...)` を使う形に整理
- `reproduce_152_metrics.py`
  - `--output` は `write_json_output(...)` に統一
- `oracle_baseline.py`
  - `run_oracle_baseline(..., output_path=...)` で `write_json_output(...)` を利用
  - `main(argv)` 化
- `oracle_test.py`
  - `main(argv)` 化
  - 最後の JSON 表示を `write_json_output(...)` へ統一
- `ab_offset_comparison.py`
  - baseline/comparison 保存を `write_json_output(...)` に統一
  - `main(argv)` 化

補助的に入れた型整理:

- `reproduce_152_metrics.py`
  - `_to_dict(...)` / `_as_int(...)` の戻り値型を固定
  - hour 集計の変数名を分けて mypy の型揺れを解消
- `oracle_test.py`
  - `_to_finite_float_array(...)` を `NDArray[np.float64]` で固定

Batch A の評価:

- commit として意味がはっきりしている
- review 時に「analysis output/CLI contract の統一」と一目で追える
- 次は Batch C の typing/container 整理へ、そのまま接続しやすい

## 2026-03-25 追加追記: reporting / entrypoint contract の横展開

Batch A の続きとして、reporting 色の強い analysis script も同じ契約へ寄せる。

対象:

1. `scripts/v460/analysis/diagnose_deadlock.py`
2. `scripts/v460/analysis/side_regime_dashboard.py`
3. `scripts/v460/analysis/compare_regime_ab.py`
4. `tests/unit/v460/test_analysis_reporting_entrypoints.py`
5. `tests/unit/v460/test_159_side_regime_dashboard.py`
6. `tests/unit/v460/test_152_parallel_tasks.py`
7. `tests/unit/v460/test_regime_detector.py`
8. `docs/v460/598_phg_investigation_next_death_spiral_candidates.md`
9. `docs/v460/037_phg_rpt_refactoring_session_log.md`

今回の寄せ方:

- `diagnose_deadlock.py`
  - `main(argv)` を維持したまま、最終出力を `write_output(...)` に統一
- `side_regime_dashboard.py`
  - `--json` 出力を `write_json_output(...)` に統一
  - report 本文は従来どおり human-readable print を維持
- `compare_regime_ab.py`
  - summary JSON 保存を `write_json_output(...)` に統一
  - CSV と human-readable report は現行のまま維持

判断:

- script ごとの集計/表示はそのまま
- output boundary だけを揃えるので低リスク
- 後から見返したときも「analysis reporting entrypoint の統一」として意味が読みやすい
