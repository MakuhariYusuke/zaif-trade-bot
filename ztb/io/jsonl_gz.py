"""JSONL.gz 読み書きユーティリティ.

OBRecorder / TradesRecorder / MarketDataCollector 等で共通利用する
JSONL gzip ファイルの append 書き込みと読み込みヘルパー。

フォーマット: 1行1JSON, gzip 圧縮, ファイル名 YYYYMMDD.jsonl.gz
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

def append_jsonl_gz(
    path: Path,
    records: Sequence[dict[str, object]],
) -> int:
    """レコードを JSONL.gz ファイルに追記.

    Args:
        path: 書き込み先 .jsonl.gz パス (親ディレクトリは事前に存在すること).
        records: JSON-serializable な dict のシーケンス.

    Returns:
        書き込んだレコード数.

    Raises:
        OSError: ファイル書き込み失敗時.
    """
    if not records:
        return 0
    with gzip.open(path, "at", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)

def read_jsonl_gz(path: Path) -> list[dict[str, object]]:
    """JSONL.gz ファイルを読み込み.

    Args:
        path: 読み込み元 .jsonl.gz パス.

    Returns:
        dict のリスト. ファイルが存在しない場合は空リスト.
    """
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipped malformed JSONL line in {path.name}")
    return records
