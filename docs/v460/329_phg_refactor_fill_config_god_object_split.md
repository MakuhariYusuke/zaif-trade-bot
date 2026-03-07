# 329# fill_config.py God Object 分割

| key | value |
|-----|-------|
| session | 329# |
| date | 2026-03-12 |
| type | refactor |
| phase | phg (cross-phase) |
| parent | 328# task audit |
| commit | (この doc と同一コミット) |

## 概要

`fill_config.py` を **2,046 → 724 行** に縮小。
328# タスク監査の Step B (YAML パーサー分離) に基づき、
God Object を 4 ファイルに分割した。

## 分割結果

| ファイル | 行数 | 内容 |
|---------|------|------|
| `fill_config.py` | 724 | FillTestConfig dataclass フィールド定義 + 薄い `from_yaml` / `__post_init__` ラッパー |
| `fill_config_parser.py` | 971 | YAML→kwargs パーサー (5 セクション関数 + `parse_fill_config_yaml`) |
| `fill_config_validation.py` | 313 | `validate_fill_config()` — `__post_init__` バリデーション |
| `fill_config_results.py` | 120 | `SkipGateResult`, `FillMonitorResult`, `PnlMeasurement` + `compute_ev_offset_multiplier()` |

**合計**: 2,128 行 (モジュールヘッダ / import のオーバーヘッド +82 行)

## 設計方針

### 後方互換性の維持
- `fill_config.py` は全シンボルを **re-export** (`from fill_config_results import ... # noqa: F401`)
- 既存の `from scripts.v460.lib.fill_config import SkipGateResult` 等は **一切変更不要**
- `FillTestConfig.from_yaml()` は薄い classmethod ラッパーとして残存

### 循環 import 回避
- `fill_config_validation.py` → `FillTestConfig` は `TYPE_CHECKING` ガード
- `fill_config_parser.py` → `FillTestConfig` は関数内 lazy import
- `__post_init__` → `validate_fill_config` は関数内 lazy import

### テスト修正 (3 件)
- `test_155_hindsight_review.py`: ソース検査先を `fill_config_parser.py` に変更
- `test_175_code_review_sweep2.py`: `_parse_stale_vg_section` を parser モジュールから直接 import
- `test_200_an_improvements.py`: `_parse_stopgap_section` を parser モジュールから直接 import

## テスト結果

```
4105 passed, 14 warnings in 21.64s
```

## 参照

- 328# 分割計画: `docs/v460/328_phg_rpt_task_audit_and_god_object_analysis.md`
- 325# orchestrator 分割: `docs/v460/325_phg_refactor_orchestrator_god_object_split.md`
