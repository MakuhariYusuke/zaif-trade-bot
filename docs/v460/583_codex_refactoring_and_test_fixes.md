# 583# Codex Refactoring And Test Fixes

## Summary
- `tests/unit/v460` broad で停止していた failure を起点に、`prompt_codex_583` の Task A/B/C/E を実装した。
- `Task D` は現行 repo ではすでに解消済みで、追加実装は不要だった。

## Implemented

### Task A: v460 broad failure fix
- `MakerPriceCalculator.get_robust_inputs()` を復旧し、`574#` / `575#` で前提にしていた robust input path を再接続した。
- `analyze_fill_logs.section_execution_quality_comparison()` は `execution_additive_enabled` を優先し、旧 `executor_offset_stages` JSON にも後方互換で対応した。

### Task B: `offset_pipeline.py` split
- `_apply_offset_pipeline_multiplicative()` を [scripts/v460/lib/multiplicative_pipeline.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/multiplicative_pipeline.py) へ抽出した。
- [scripts/v460/lib/offset_pipeline.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/offset_pipeline.py) は dispatcher + additive pipeline を中心に整理した。
- `OffsetPipelineMixin` は `MultiplicativePipelineMixin` を継承する形に更新した。

### Task C: `fill_cycle_executor.py` split
- `run_single_cycle()` は orchestration のみに整理した。
- phase helper:
  - `_run_pre_order_phase(...)`
  - `_submit_order_phase(...)`
  - `_monitor_fill_phase(...)`
  - `_finalize_cycle(...)`
- phase result dataclass:
  - `_PreOrderPhaseResult`
  - `_SubmissionPhaseResult`
  - `_FillPhaseResult`

### Task D: FillRecord telemetry alignment
- 現行 repo では以下がすでに通っていたため no-op:
  - `execution_sigma`
  - `execution_adverse_ofi`
  - `execution_additive_enabled`
- `FillRecord` と `fill_record_builder` の field alignment は prompt 実施前に整合済みだった。

### Task E: additive pipeline tests
- `tests/unit/v460/test_582_additive_pipeline.py` に以下を追加した。
  - additive pipeline でも final clamp が動く
  - `experimental_additive_pipeline=True` 時に multiplicative path を呼ばない
  - EV score が liquidity buffer に分類される
  - buy side では trending offset を無視する

## Test Follow-up
- `run_single_cycle` 分割後の source-contract test を phase helper 前提へ更新した。
- `offset_pipeline.py` split 後の source-contract test は、dispatcher file ではなく実際の multiplicative method source を検査する形へ更新した。

## Validation
- focused:
  - prompt 583 関連 suite: `236 passed`
- broad:
  - `tests/unit/v460/` は failure を段階的に修正し、最後は assertion failure を解消した状態まで到達
  - 実行末尾では環境側 `KeyboardInterrupt` が混ざることがあるため、完走可否は環境負荷の影響を受ける

