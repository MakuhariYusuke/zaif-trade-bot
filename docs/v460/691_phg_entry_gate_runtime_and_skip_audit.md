# 691# entry_gate runtime 有効化と skip audit 整備

## 概要

- `690#` で要求された runtime 側の実装をまとめて反映した。
- 主眼は以下の 4 点。
  - `entry_gate` を observe から実 blocking へ移す
  - stale / 連続 block / 高 block rate の guard を追加する
  - timeout regime×side override を runtime と observability に通す
  - `_execute_skip(...)` の監査可能性を上げる

## 実装内容

### 1. entry_gate guard

- 新規: `scripts/v460/lib/entry_gate_guard.py`
- `EntryGateGuard` を導入し、以下を管理する。
  - calibration の最終更新時刻
  - 連続 block 回数
  - 直近 window の block rate
- suppress 条件:
  - calibration が stale
  - `max_consecutive_blocks` 超過
  - `max_block_rate` 超過かつ `min_eval_for_rate` 到達

### 2. runtime wiring

- `scripts/v460/run_fill_test.py`
  - runner に `EntryGateGuard` を保持
  - calibration map 更新時に guard へ通知
- `scripts/v460/lib/orchestrator_mid_cycle.py`
  - `n_eff < entry_gate_n_min` 時は `p_win=0.5` 固定
  - EV<=0 の block 判定前に guard suppress を適用
  - cycle 単位 observability を保持
- `scripts/v460/lib/orchestrator_post_cycle.py`
  - calibration 更新後に guard へ時刻反映

### 3. timeout regime×side override

- `FillTestConfig.regime_timeout_overrides`
- `get_timeout_with_reason(side, regime)`
  - reason 形式を `regime_override_{regime}_{side}` に統一
- `sell_age_cap` による短縮時は `_sell_age_cap` suffix を付与

### 4. skip audit / trace

- `_execute_skip(...)` 呼び出しに `update_last_side=` を明示
- env-halt 系 skip は heartbeat を明示
- cancel reason taxonomy を `scripts/v460/lib/cancel_reason_taxonomy.py` へ集約
- `decision_trace_id` / timeout fields / entry gate fields を `FillRecord` に接続

## hidden task として回収したもの

- flat parser と nested parser の両方へ `entry_gate_*` safety fields を追加
- hot-reload で entry gate guard を reset する経路を追加
- YAML drift allowlist を更新
- early skip record と normal fill record の observability ずれを防止

## テスト

- 新規
  - `tests/unit/v460/test_690_entry_gate_guard.py`
  - `tests/unit/v460/test_690_timeout_priority.py`
  - `tests/unit/v460/test_690_skip_audit.py`
  - `tests/unit/v460/test_690_offset_pipeline.py`
- 更新
  - `test_169_config_hot_reload.py`
  - `test_346_fill_config_validation.py`
  - `test_336_yaml_code_drift_prevention.py`
  - `test_688_timeout_trace_and_skip_audit.py`

### 結果

- focused regression:
  - `164 passed in 3.99s`
- 主要確認:
  - `entry_gate` guard suppress
  - timeout priority
  - skip audit source check
  - offset pipeline stage toggle

## 今後

1. `entry_gate` blocking 実績の layer 別集計を analysis protocol 側へ渡す
2. `skip_gate_bypass` と `bucket budget` の相互作用をさらに長期観測する
3. `fill_quality` judgment/report 側へ trace fields を自然に統合する
