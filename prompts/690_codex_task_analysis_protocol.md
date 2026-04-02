# Codex Task: 690# Analysis Scripts 統一 CLI (688# 分析手法の再利用化)

## 目的
688# で手動実施したデータ分析パターン (層別分析: NFQ / AS / spread band / regime)
を再現可能な CLI コマンドとして統合する。
既存の `analyze_fill_logs.py` (25+ section) と `analysis_common.py` を活用し、
688# の分析プロトコルを `--protocol 688` として呼び出せるようにする。

## 背景

### 既存の分析スクリプト (19 ファイル)
```
scripts/v460/analysis/
├── analysis_common.py          # 共通 CLI / データ読み込み
├── analyze_fill_logs.py        # 25+ section (basic, side, regime, cancel, ...)
├── side_regime_dashboard.py    # 層別ダッシュボード
├── vg_and_trend.py             # VG 有効性 + daily/8h trend
├── tail_loss_analysis.py       # tail loss 分析
├── sha_performance_report.py   # SHA 別パフォーマンス
├── sha_comparison.py           # SHA 間比較
├── stopgap_daily_report.py     # 日次レポート
├── deep_analysis_672.py        # 672# 多角分析
├── hour_matched_comparison.py  # 時間帯マッチ比較
└── ... (10+ more)
```

### 688# の分析プロトコル (手動で実施)
1. **基本指標**: n=fill, avg PnL30, side 別 avg PnL, AS rate
2. **NFQ 層別**: cancel_reason 別の skip 回数・fill rate
3. **AS 層別**: adverse_selected 別の PnL 分布、severity (< -10 bps)
4. **Spread Band 層別**: spread_at_order を bins (0-1500, 1500-2500, 2500+) で分類
5. **時間帯分析**: JST 時間別 fill rate / PnL / AS rate
6. **Git SHA 分析**: SHA 別パフォーマンス比較
7. **JSON 出力**: 全結果を analysis_results/ に JSON 保存

### 課題
- 688# の分析は Copilot Chat で手動実行 → 再現性なし
- `analyze_fill_logs.py` は 25+ section だが期間指定が不便
- 層別分析 (NFQ / AS / spread) が散在し、統一プロトコルがない

## タスク

### Task 1: Protocol Registry

**新規作成**: `scripts/v460/analysis/protocols/__init__.py`

```python
"""分析プロトコル登録モジュール.

各プロトコルは AnalysisProtocol を実装し、
PROTOCOL_REGISTRY に登録することで CLI から呼び出せる。
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ztb.metrics.fill_quality import FillRecord


@dataclass
class ProtocolResult:
    """プロトコル実行結果."""
    text_report: str
    json_payload: dict[str, Any]
    warnings: list[str]


class AnalysisProtocol(ABC):
    """分析プロトコルの基底クラス."""

    @property
    @abstractmethod
    def name(self) -> str:
        """プロトコル名 (CLI --protocol 引数)."""

    @property
    @abstractmethod
    def description(self) -> str:
        """プロトコルの説明."""

    @abstractmethod
    def execute(
        self,
        records: list[FillRecord],
        *,
        output_dir: Path | None = None,
    ) -> ProtocolResult:
        """プロトコルを実行し結果を返す."""


PROTOCOL_REGISTRY: dict[str, type[AnalysisProtocol]] = {}


def register_protocol(cls: type[AnalysisProtocol]) -> type[AnalysisProtocol]:
    """プロトコルを登録するデコレータ."""
    PROTOCOL_REGISTRY[cls.name.fget(cls)] = cls  # type: ignore[attr-defined]
    return cls
```

### Task 2: 688# Protocol 実装

**新規作成**: `scripts/v460/analysis/protocols/protocol_688.py`

```python
"""688# データ分析プロトコル.

層別分析パターン:
1. 基本指標 (n, avg PnL30, side別)
2. NFQ 層別 (cancel_reason)
3. AS 層別 (adverse_selected + severity)
4. Spread Band 層別 (spread_at_order bins)
5. 時間帯分析 (JST hour)
6. Git SHA 分析
7. Side × Regime クロス分析
"""

@register_protocol
class Protocol688(AnalysisProtocol):
    name = "688"
    description = "688# 層別分析 (NFQ/AS/spread/hour/regime)"

    def execute(self, records, *, output_dir=None) -> ProtocolResult:
        ...
```

各セクションは既存の `analyze_fill_logs.py` の section 関数を **再利用**:
- `section_basic()` → 基本指標
- `section_side()` → side 別
- `section_cancel()` → NFQ 層
- `section_adverse_selection()` → AS 層
- `section_spread()` → spread band
- `section_hourly()` → 時間帯
- `section_git_sha()` → SHA
- `section_regime()` → regime

新規追加:
- `section_side_regime_cross()` → side × regime クロス (688# で手動実施)
- `section_sell_hour_boost_effectiveness()` → sell_hour_offset_boost の有効性 (688# 固有)

### Task 3: 統合 CLI エントリポイント

**新規作成**: `scripts/v460/analysis/run_protocol.py`

```python
"""分析プロトコル CLI.

Usage:
    python -m scripts.v460.analysis.run_protocol --protocol 688 \\
        --days 4 --output-dir analysis_results/
    
    python -m scripts.v460.analysis.run_protocol --protocol 688 \\
        --start 2026-03-29 --end 2026-04-02
    
    python -m scripts.v460.analysis.run_protocol --list
"""

import argparse
from scripts.v460.analysis.protocols import PROTOCOL_REGISTRY
from scripts.v460.analysis.analysis_common import (
    add_standard_args,
    load_records_with_filters,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="分析プロトコル CLI")
    parser.add_argument("--protocol", type=str, help="プロトコル名")
    parser.add_argument("--list", action="store_true", help="利用可能プロトコル一覧")
    parser.add_argument("--days", type=int, help="直近 N 日のデータ")
    parser.add_argument("--start", type=str, help="開始日 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="終了日 (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=str, default="analysis_results")
    parser.add_argument("--json", action="store_true", help="JSON 出力のみ")
    add_standard_args(parser)  # analysis_common の共通引数
    return parser
```

### Task 4: analysis_common.py 拡張

**対象**: `scripts/v460/analysis/analysis_common.py`

1. `add_standard_args(parser)`: 共通引数 (--results-dir, --side, --regime 等) 追加関数
2. `load_records_with_filters(args) -> list[FillRecord]`: 引数に基づくレコード読み込み + フィルタ
3. `filter_by_date_range(records, start, end) -> list[FillRecord]`: 日付範囲フィルタ
4. `filter_by_days(records, days) -> list[FillRecord]`: 直近N日フィルタ

**注意**: 既存の関数 (`add_results_dir_arg`, `write_json_output`, `write_output`) はそのまま維持。

### Task 5: テスト

**新規作成**: `tests/unit/v460/test_690_analysis_protocol.py`

1. Protocol688 が ProtocolResult を返す (空レコードでもエラーなし)
2. Protocol688 の JSON 出力に必須キーが含まれる
3. PROTOCOL_REGISTRY に "688" が登録されている
4. `--list` オプションで利用可能プロトコル一覧が表示される
5. `filter_by_days()` が正しくフィルタする
6. `filter_by_date_range()` が正しくフィルタする
7. CLI parser が必須引数を正しく処理
8. `run_protocol.py` が `--protocol 688 --days 1` で実行可能
9. `python -m pytest tests/ -x --tb=short` で全テスト pass

## 動作仕様

1. `python -m scripts.v460.analysis.run_protocol --protocol 688 --days 4` で 688# 分析を再現
2. テキストレポート + JSON ファイルを出力
3. `--list` で利用可能プロトコル一覧
4. 既存スクリプト (`analyze_fill_logs.py` 等) は変更しない
5. Protocol クラスは既存の section 関数を内部的に再利用
6. 将来 protocol_690, protocol_692 等を追加可能な extensible 設計

## 受け入れ基準

- [ ] `--protocol 688` で層別分析が実行される
- [ ] JSON 出力に 7 セクション (basic/nfq/as/spread/hour/sha/regime) が含まれる
- [ ] `--days N` / `--start`/`--end` でデータ絞り込み可能
- [ ] 既存 analysis スクリプトを変更しない
- [ ] 新規テスト 9 件以上、全テスト pass
- [ ] `--list` でプロトコル一覧表示

## リスク評価

- **低リスク**: 新規ファイル追加のみ。既存コード変更なし
- **ロールバック**: protocols/ ディレクトリ削除で即時復帰
- **価値**: 688# パターンの再現性確保。今後のセッションで分析手法を標準化
