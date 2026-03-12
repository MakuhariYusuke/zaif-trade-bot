"""304# 共通定数モジュール — 複数モジュールで共有される定数の SSOT.

_BPS_FACTOR が 6 ファイルに独立定義されていた重複を解消し、
変更時の修正漏れリスクを排除する。
"""

from __future__ import annotations

from typing import Final

# 1 basis point = 1e-4, bps 換算: value / base * BPS_FACTOR
BPS_FACTOR: Final[int] = 10_000
