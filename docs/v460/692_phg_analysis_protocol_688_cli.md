# 692# analysis protocol 688 CLI 整備

## 概要

- `688#` の layered analysis を単発スクリプト依存ではなく protocol registry で再利用できる形に整理した。
- `scripts/v460/analysis/run_protocol.py` を追加し、CLI から protocol を呼べるようにした。

## 実装内容

### 1. protocol registry

- 新規: `scripts/v460/analysis/protocols/__init__.py`
- 定義:
  - `AnalysisProtocol`
  - `ProtocolResult`
  - `PROTOCOL_REGISTRY`
  - `register_protocol`

### 2. Protocol688

- 新規: `scripts/v460/analysis/protocols/protocol_688.py`
- 既存 section 関数を再利用しつつ、JSON payload を protocol 形式へ整理
- 現在の payload key:
  - `basic`
  - `side`
  - `nfq`
  - `adverse_selection`
  - `spread`
  - `hour`
  - `sha`
  - `regime`
  - `side_regime_cross`
  - `sell_hour_offset_boost`

### 3. 共通 loader / filter

- `scripts/v460/analysis/analysis_common.py`
  - `add_standard_args(...)`
  - `filter_by_date_range(...)`
  - `filter_by_days(...)`
  - `load_records_with_filters(...)`
- 既存の output / filter helper をそのまま使い、CLI drift を避けた

### 4. CLI

- 新規: `scripts/v460/analysis/run_protocol.py`
- 対応:
  - `--list`
  - `--protocol`
  - `--days`
  - `--start`
  - `--end`
  - `--output-dir`
  - `--json`

## 使い方

```bash
.venv/Scripts/python.exe -m scripts.v460.analysis.run_protocol --list
.venv/Scripts/python.exe -m scripts.v460.analysis.run_protocol --protocol 688 --days 3
.venv/Scripts/python.exe -m scripts.v460.analysis.run_protocol --protocol 688 --start 2026-04-01 --end 2026-04-02 --output-dir analysis_results
```

出力:

- `protocol_688.json`
- `protocol_688.txt`

## hidden task として回収したもの

- protocol registry は `Record` dict ベースにして、既存 analysis section 関数との互換を維持
- `--json` 時は text writer を踏まず、JSON writer のみ使うように統一
- date filter は UTC 基準で共通化

## テスト

- 新規: `tests/unit/v460/test_690_analysis_protocol.py`

### 結果

- focused:
  - protocol registry
  - date filter
  - CLI parser
  - `--json` output path

## 今後

1. protocol registry に `688` 以外の analysis bundle を追加できる状態になった
2. `run_protocol` を analysis task の標準再現コマンドとして使える
3. `analysis_common` の filter/output 共通化をさらに横展開できる
