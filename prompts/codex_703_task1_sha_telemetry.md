# 703# Task 1: SHA 別 AS テレメトリ追加

## 背景
Protocol 688 再分析 (702#) で、単一 SHA (b56771a) が全体損失の 88.6% を占める交絡が発覚。
SHA 別 AS rate をリアルタイムで追跡し、異常 SHA を早期検知する機構が必要。

## 修正箇所

### 1. `scripts/v460/analysis/protocols/protocol_688.py`
- 既存の `sha` セクションに `adverse_selection_rate_pct` と `adverse_selection_count` を追加
- 各 SHA について filled records から `is_adverse` フラグを集計
- `is_adverse` は既存の `_record_bool(record, "is_adverse")` で取得可能

### 2. `scripts/v460/lib/fill_record_builder.py`
- `build_fill_record` 内で `sha_as_count` / `sha_fill_count` の累積カウンタはスコープ外（ランタイム状態）
- 代わりに既存の `git_sha` フィールドが fill_record に含まれることを検証

### 3. Protocol 688 出力強化
- `sha` セクションの各 SHA エントリに以下を追加:
  - `adverse_selection_count`: int
  - `adverse_selection_rate_pct`: float
  - `total_pnl_contribution_bps`: float (avg_pnl30 × filled)
- SHA を `total_pnl_contribution_bps` 昇順でソート（最悪 SHA が先頭）

## テスト
- `tests/unit/v460/test_702_sha_telemetry.py`
  - test_sha_as_rate_calculation: mock records で SHA 別 AS 率が正しく計算される
  - test_sha_pnl_contribution: total_pnl_contribution_bps の計算精度
  - test_sha_empty_fills: filled=0 の SHA で rate=0, contribution=0
  - test_sha_sorting: worst SHA が先頭に来る

## 制約
- 既存テスト (`test_700_protocol_688_nfq_fix.py`) が引き続きパスすること
- protocol_688.json の既存キー構造を破壊しない（追加のみ）
- `is_adverse` フラグが存在しない旧レコードでは graceful に None/0 扱い
