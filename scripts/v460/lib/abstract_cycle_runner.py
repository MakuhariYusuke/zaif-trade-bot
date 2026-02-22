"""145# §10.1-#2: AbstractCycleRunner — cycle runner の抽象基底クラス.

FillTestRunner (v460) をはじめとする fill test ランナーの共通インタフェースを定義。
Template Method パターンにより、サブクラスは以下を実装する:

    - ``run_single_cycle(side)``  — 1 サイクルの発注→監視→FillRecord 構築
    - ``run_continuous(hours)``   — メインオーケストレーションループ

共通ユーティリティ (_new_cycle_id, _get_git_sha) は具象メソッドとして提供。
ロック管理・BatchPersistence 等のインフラは FillTestRunner が保持 (将来的に抽出予定)。
"""

from __future__ import annotations

import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ztb.metrics.fill_quality import FillRecord


class AbstractCycleRunner(ABC):
    """Fill test cycle runner の抽象基底クラス.

    §10.1-#2: FillTestRunner の God Object パターンを段階的に解消する
    第一歩として、共通インタフェースと汎用ユーティリティを定義。

    Subclass Requirements:
        - ``run_single_cycle(side)`` を実装
        - ``run_continuous(hours)`` を実装
        - ``adapter``, ``config`` 属性を持つこと
    """

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    async def run_single_cycle(
        self,
        side: Optional[str] = None,
    ) -> "FillRecord":
        """1 サイクル: 発注 → 約定監視 → PnL 計測 → FillRecord 構築.

        Args:
            side: "buy" or "sell". None の場合はサブクラスが決定.

        Returns:
            FillRecord — サイクルの結果レコード.
        """
        ...

    @abstractmethod
    async def run_continuous(
        self,
        hours: float = 1.0,
    ) -> List["FillRecord"]:
        """メインオーケストレーションループ.

        Args:
            hours: 実行時間 (時間).

        Returns:
            サイクル結果のリスト.
        """
        ...

    # ------------------------------------------------------------------
    # Hook methods (override in subclass if needed)
    # ------------------------------------------------------------------

    def on_cycle_start(self, side: str) -> None:
        """Called before a cycle begins. Override for pre-cycle logic."""

    def on_cycle_end(self, record: "FillRecord") -> None:
        """Called after a cycle completes. Override for post-cycle logic."""

    def should_skip_cycle(self, side: str) -> Optional[str]:
        """Return a cancel_reason string to skip this cycle, or None to proceed."""
        return None

    # ------------------------------------------------------------------
    # Common utilities (concrete, shared by all runners)
    # ------------------------------------------------------------------

    @staticmethod
    def _new_cycle_id(prefix: Optional[str] = None) -> str:
        """Generate a unique cycle ID (timestamp + uuid)."""
        base = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        return f"{prefix}_{base}" if prefix else base

    @staticmethod
    def _get_git_sha() -> str:
        """Get current git commit short hash, or 'unknown' on failure."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"
