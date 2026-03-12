# 243# YAML Wiring + Validation Fix

## 概要
242# Liveness Constraint Relaxation で追加した新設定フィールドが
YAML → FillTestConfig → DynamicKillConfig ランタイムへ正しくパススルーされていなかった
重大な配線漏れを修正。

## 問題 (242# セルフレビュー subagent で発見)

| # | Severity | 問題 |
|---|----------|------|
| 1 | **HIGH** | `quiescence_gate_blocks_threshold` / `quiescence_sleep_sec` が `flat_keys` に未登録 → YAML 読み込み無効 |
| 2 | **HIGH** | `toxic_kill_stale_multiplier` が `run_fill_test.py` の 3 箇所の DynamicKillConfig 構築で未渡し → 常にデフォルト値 10 |
| 3 | **LOW** | `quiescence_sleep_sec` / `quiescence_gate_blocks_threshold` の `__post_init__` バリデーション欠如 |
| 4 | **LOW** | `DynamicKillConfig.toxic_kill_stale_multiplier` の `__post_init__` バリデーション欠如 |

## 修正内容

### A. fill_config.py
- `flat_keys` セットに `quiescence_gate_blocks_threshold`, `quiescence_sleep_sec` 追加
- `_parse_stopgap_section` の sell_kill / buy_kill ブロックに `toxic_stale_multiplier` → `sell/buy_dynamic_kill_toxic_stale_mult` YAML 配線追加
- FillTestConfig に `sell_dynamic_kill_toxic_stale_mult: int = 10` / `buy_dynamic_kill_toxic_stale_mult: int = 10` フィールド追加
- `__post_init__` に `quiescence_sleep_sec < 0` / `quiescence_gate_blocks_threshold < 0` バリデーション追加

### B. run_fill_test.py
- 初期構築 (L291-305): `SellKillConfig` / `DynamicKillConfig` に `toxic_kill_stale_multiplier` パススルー
- `_rebuild_sell_kill_mgr` (L431): 同上
- `_rebuild_buy_kill_mgr` (L446): 同上

### C. sell_dynamic_kill.py
- `DynamicKillConfig.__post_init__` に `toxic_kill_stale_multiplier < 0` バリデーション追加

## YAML 設定例
```yaml
止血:
  sell_dynamic_kill:
    toxic_stale_multiplier: 10  # probe interval × N (toxicity KILL 時)
  buy_dynamic_kill:
    toxic_stale_multiplier: 10

quiescence_gate_blocks_threshold: 20  # 連続ゲートブロック → quiescence 判定
quiescence_sleep_sec: 1800.0          # quiescence 時のスリープ上限 (秒)
```

## テスト
- 12 テスト追加 (`test_243_yaml_wiring.py`)
  - `TestQuiescenceYAMLWiring243`: 4 tests (flat_keys 配線 + バリデーション)
  - `TestToxicStaleMultYAMLWiring243`: 4 tests (止血 YAML 配線)
  - `TestDynamicKillConfigValidation243`: 2 tests (DynamicKillConfig バリデーション)
  - `TestPassthroughToDynamicKillConfig243`: 2 tests (Config → Runtime パススルー)
- 全 3370 v460 テスト通過
