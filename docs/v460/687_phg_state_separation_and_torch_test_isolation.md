# 687# state separation と torch test isolation

## 概要

687# の 2 本をまとめて実装した。

1. `SideSelector` の state を `last_executed_side` と `last_attempted_side` に分離
2. `tools/ab_param_search.py` の torch 連鎖 import を lazy import 化し、`tests/unit/tools/test_ab_param_search.py` を環境依存で落とさないようにした

加えて、直近で切った metrics helper が既存 helper と重複していないかも確認した。

## 実装

### 1. Side state separation

- `scripts/v460/lib/side_selector.py`
  - `_last_side` 単一状態を廃止
  - `_last_executed_side`
  - `_last_attempted_side`
  を追加
  - `next()` は `last_executed_side` を参照
  - `update_after_attempt(...)` を追加
  - `update_after_decision(..., attempt_already_recorded=...)` で fill 成功時のみ executed を更新

- `scripts/v460/lib/fill_loop_orchestrator.py`
  - `_execute_skip(update_last_side=True)` は attempted のみ更新
  - 既存軽量 stub 互換のため、`_side_selector` 不在時は旧 `_last_side` fallback を維持

- `scripts/v460/lib/orchestrator_balance.py`
- `scripts/v460/lib/orchestrator_mid_cycle.py`
  - preflight / exception / network-error の skip 経路を attempted 更新へ統一

- `scripts/v460/lib/fill_cycle_executor.py`
  - pre-order で attempted 更新
  - fill 成功後だけ executed 更新

- `scripts/v460/lib/fill_record_helpers.py`
  - skip record に `last_executed_side` / `last_attempted_side` を自動埋め込み
  - resume 時も両 state を復元

- `scripts/v460/lib/fill_record_builder.py`
- `ztb/metrics/fill_quality.py`
  - `FillRecord` に
    - `last_executed_side`
    - `last_attempted_side`
    を追加

### 2. torch test isolation

- `tools/ab_param_search.py`
  - `UnifiedOptimizer` / `OptimizationConfig` の import を `main()` 内へ移動
  - config 生成テストでは top-level で torch 依存を踏まないようにした
  - あわせて spec load と score path の low-risk 型補強を追加

### 3. helper overlap scan

- `ztb/metrics/fill_metrics_core.py` で切り出した helper は、既存の同等 helper と衝突していないことを確認
- ただし `ztb/metrics/record_metrics.py` が `fill_quality` に少し寄りすぎていたため、
  `format_utc_day` だけは `fill_metrics_core` へ寄せて依存を薄くした

## hidden task と判断

### hidden task 1: skip 経路の更新点は `_execute_skip` に集約すべき

state 分離は `SideSelector` だけ直しても足りない。
実害は orchestrator の skip 経路にあったため、`_execute_skip(update_last_side=True)` を
attempt-only に寄せた。

### hidden task 2: `update_after_decision()` の互換維持

既存の side selector テストは `update_after_decision()` を単独で呼ぶ。
runtime では pre-order で attempted を先に記録するが、テスト互換のため
`attempt_already_recorded=False` を default にした。

### hidden task 3: test harness の軽量 stub 互換

`test_276_blocking_policy_dry.py` の stub は `_side_selector` を持たない。
そのため `_execute_skip()` には fallback を残し、回帰を防いだ。

## 検証

### py_compile

```bash
python3 -m py_compile \
  tools/ab_param_search.py \
  scripts/v460/lib/side_selector.py \
  scripts/v460/lib/fill_record_helpers.py \
  scripts/v460/lib/fill_loop_orchestrator.py \
  scripts/v460/lib/orchestrator_balance.py \
  scripts/v460/lib/orchestrator_mid_cycle.py \
  scripts/v460/lib/fill_cycle_executor.py \
  scripts/v460/lib/fill_record_builder.py \
  ztb/metrics/fill_quality.py \
  ztb/metrics/record_metrics.py \
  tests/unit/v460/test_687_state_separation.py
```

### targeted mypy

```bash
.venv/Scripts/python.exe scripts/quality/run_targeted_mypy.py \
  tools/ab_param_search.py \
  scripts/v460/lib/side_selector.py \
  tests/unit/v460/test_687_state_separation.py
```

- `Success: no issues found in 3 source files`

補足:
- `fill_record_helpers.py` と `record_metrics.py` は既存 baseline が厚いため、
  今回は clean 化の対象にせず `py_compile + pytest` を主証拠にした

### focused / broader pytest

```bash
.venv/Scripts/python.exe -m pytest \
  tests/unit/tools/test_ab_param_search.py \
  tests/unit/v460/test_687_state_separation.py \
  tests/unit/v460/test_634_sell_ranging_suppression.py \
  tests/unit/v460/test_fill_test_config.py \
  tests/unit/v460/test_166_remaining_tasks.py \
  tests/unit/v460/test_421_final_clamp_deadlock.py \
  tests/unit/v460/test_276_blocking_policy_dry.py \
  -x --tb=short --no-cov
```

- `206 passed in 15.53s`

新規:
- `tests/unit/v460/test_687_state_separation.py`
  - fill success: executed/attempted 両更新
  - attempt-only: attempted のみ更新
  - alternation は executed を参照
  - `_execute_skip()` は attempted のみ更新
  - `FillRecord` / skip record roundtrip

## 影響

- `preflight_insufficient` や NFQ で executed side が汚染されなくなった
- `balance_freeze_cycles` と alternation の誤作動を後から `FillRecord` 上で追跡できる
- `ab_param_search` は torch DLL 問題がある環境でも、テスト対象の config 生成 path は安全に通る

## 次の一手

1. `FillRecord` を使う analysis 側で `last_executed_side` / `last_attempted_side` を必要なら surfacing
2. `fill_quality` metrics/judgment 側の残る長大化整理
3. PPO/SAC scheduler 共通 helper の追加共有と warm-start continuity の強化
