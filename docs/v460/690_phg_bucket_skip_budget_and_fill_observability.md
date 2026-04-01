# 690# Bucket 別 Skip Budget と Fill 可観測性

## 目的

- `skip_gate` の連続 skip 安全弁を global counter だけで扱うのをやめ、`regime × side` の bucket で制御する
- `bypass_mode=true` でも skip 統計は失わず、FillRecord 側で追えるようにする
- `primary_max_consecutive_skip` はグローバル緊急ブレーキとして残し、budget と独立させる

## 今回の実装

### 1. runtime

- 新規: [skip_gate_budget.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/skip_gate_budget.py)
  - `BucketKey`
  - `BucketState`
  - `BucketedSkipBudget`
- [skip_gate_evaluator.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/skip_gate_evaluator.py)
  - model decision 後、primary safety valve 前に budget check を挿入
  - budget 枯渇時は `budget_exhausted_pass`
  - budget 統計は final block 可否ではなく raw skip 判定で記録
  - これにより `bypass_mode=true` でも統計だけは積める

### 2. config

- [fill_config.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/fill_config.py)
  - `skip_gate_budget_enabled`
  - `skip_gate_budget_window_min`
  - `skip_gate_budget_limits`
  - `get_budget_limit(...)`
- [fill_config_parser.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/fill_config_parser.py)
- [fill_config_validation.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/fill_config_validation.py)
- [config_hot_reload.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/config_hot_reload.py)

### 3. observability

- [fill_config_results.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/fill_config_results.py)
- [skip_gate_result_fields.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/ztb/ml/skip_gate_result_fields.py)
- [skip_gate_fill_record.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/ztb/ml/skip_gate_fill_record.py)
- [fill_record_builder.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/fill_record_builder.py)
- [fill_cycle_executor.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/fill_cycle_executor.py)
- [fill_quality.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/ztb/metrics/fill_quality.py)

追加した field:

- `skip_gate_budget_regime`
- `skip_gate_budget_remaining`
- `skip_gate_budget_exhausted`

## 設計上の判断

- budget は `skip_gate_primary_max_consecutive_skip` を置き換えない
  - budget: bucket 別制御
  - primary safety valve: global 緊急ブレーキ
- budget 統計は `raw skip` で数える
  - `bypass_mode=true` でも統計が残る
  - budget 枯渇で PASS 強制になっても、skip 意図そのものは統計に残る
- `scripts/` に runtime wiring を残しつつ、generic budget state は小さい独立 helper に分けた

## テスト

- 新規: [test_690_skip_budget.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/tests/unit/v460/test_690_skip_budget.py)
  - budget disabled
  - budget exhaustion
  - window rotation
  - regime×side independence
  - default fallback
  - config mutation/hot-reload 相当で ceiling 更新・counter 維持
  - primary safety valve coexist
  - FillRecord observability
  - bypass mode での budget statistics
- 更新:
  - [test_169_config_hot_reload.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/tests/unit/v460/test_169_config_hot_reload.py)
  - [test_346_fill_config_validation.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/tests/unit/v460/test_346_fill_config_validation.py)
  - [test_336_yaml_code_drift_prevention.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/tests/unit/v460/test_336_yaml_code_drift_prevention.py)
  - [test_fill_quality.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/tests/unit/v460/test_fill_quality.py)

## 今後の残り

1. `fill_quality` の judgment/report 側の残分割
2. heavy test setup の grouped sweep
3. PPO/SAC scheduler の shared safety helper 継続整理
