# 587# Telemetry Parity & Dead Config Cleanup

## 概要

585# / 586# レビューで見えた、Phase 6 で踏みやすい設定断線と telemetry 断線を先に解消する。

今回の対象は次の 4 点:

1. `execution_additive_enabled` を FillRecord / JSONL まで通す
2. `additive_base_bps` という未使用 config を削除する
3. eDRC final clamp に `get_robust_inputs()` を接続する
4. additive / eDRC / entry-gate の hot-reload 範囲を明示する

## 実装

### A. Telemetry parity

- `ztb/metrics/fill_quality.py`
  - `FillRecord.execution_additive_enabled` を保持
- `scripts/v460/lib/fill_cycle_executor.py`
  - `_build_fill_record(...)` 呼び出しに
    `execution_additive_enabled=self.config.experimental_additive_pipeline`
    を追加

これで additive / multiplicative の実行ラベルが JSONL まで残る。

### B. Dead config cleanup

- `scripts/v460/lib/fill_config.py`
  - `additive_base_bps` を削除
- `scripts/v460/lib/fill_config_parser.py`
  - nested additive config からの parse を削除
- `configs/v460/fill_test.yaml`
  - 設定例から削除

### C. eDRC robust inputs

- `scripts/v460/lib/offset_pipeline.py`
- `scripts/v460/lib/multiplicative_pipeline.py`

final clamp の `resolve_offset_ceiling()` 呼び出しで、

- `self._maker_price.last_sigma`
- `self._maker_price.get_adverse_ofi(side)`

ではなく、

- `self._maker_price.get_robust_inputs(side)`

を使う形へ変更した。

これで sigma / OFI の tail spike が eDRC ceiling にそのまま入る経路を避けられる。

### D. Hot-reload scope

`scripts/v460/lib/config_hot_reload.py` に次を追加:

- `experimental_additive_pipeline`
- `edrc_alpha`
- `edrc_beta`
- `edrc_c_base`
- `edrc_hard_cap`
- `entry_gate_enabled`

一方で `entry_gate_calibration_map_path` はファイルロード境界なので対象外のままにしている。

### E. 二枚看板の整理

- `execution_additive_enabled`
  - deprecated telemetry-only field
- `experimental_additive_pipeline`
  - 実際のロジック分岐

`fill_config_parser.py` では両者が不一致なら warning を出す。

## テスト

- `tests/unit/v460/test_421_final_clamp_deadlock.py`
  - hot-reload 範囲
  - execution telemetry roundtrip
- `tests/unit/v460/test_467_remaining_issues.py`
  - hard cap / additive parse / mismatch warning
- `tests/unit/v460/test_582_additive_pipeline.py`
  - additive final clamp で robust inputs を使う回帰
- `tests/unit/v460/test_585_multiplicative_pipeline.py`
  - multiplicative final clamp で robust inputs を使う回帰
- `tests/unit/v460/test_292_observability.py`
  - `build_fill_record(..., execution_additive_enabled=...)`
- `tests/unit/v460/test_145_structural_fixes.py`
  - fill-cycle source contract
- `tests/unit/v460/test_169_config_hot_reload.py`
  - Phase 6 hot-reload fields

## 判断

- `execution_additive_enabled` は当面残すが、役割は telemetry-only に限定する
- additive / eDRC は「設定・executor・JSONL・分析」の線を切らないことを優先する
- dead config は温存せず、この段階で削る
