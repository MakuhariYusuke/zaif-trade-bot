"""044# 単一起動ロック管理.

run_fill_test.py から分離 (158# P2-4: god object 分割).
047# A4: TOCTOU race 対策 — open(path, 'x') で排他的作成。
129# D.3: heartbeat timestamp をロックファイルに記録。
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class LockManager:
    """fill_test プロセスの単一起動ロック管理.

    同一 results_dir に対して複数プロセスが並行動作することを防止する。
    ロックファイルに PID を記録し、起動時に既存ロックの生死を検証する。
    129# D.3: heartbeat timestamp で stale 検知。
    """

    def __init__(
        self,
        results_dir: Path,
        run_id: str,
        *,
        lock_stale_heartbeat_sec: float = 600.0,
        lock_acquire_retries: int = 3,
        lock_heartbeat_period_sec: float = 60.0,
    ) -> None:
        self._results_dir = results_dir
        self._run_id = run_id
        self._lockfile_path: Path | None = None
        self._lock_stale_heartbeat_sec = lock_stale_heartbeat_sec
        self._lock_acquire_retries = lock_acquire_retries
        self.lock_heartbeat_period_sec = lock_heartbeat_period_sec

    @property
    def lockfile_path(self) -> Path | None:
        return self._lockfile_path

    def acquire(self) -> None:
        """044# Bug7: 単一起動ロック (lockfile + PID + stale 回収).

        047# A4: TOCTOU race 対策 — open(path, 'x') で排他的作成。
        """
        lock_path = self._results_dir / "fill_test.lock"
        self._lockfile_path = lock_path
        now_ts = int(time.time())
        lock_content = f"{os.getpid()}|{now_ts}|{self._run_id}|{now_ts}"

        def _check_stale_and_reclaim() -> bool:
            """既存ロックが stale なら削除して True を返す."""
            try:
                content = lock_path.read_text(encoding="utf-8").strip()
                parts = content.split("|")
                existing_pid = int(parts[0])
                # 129# heartbeat age 検査 (4番目フィールド)
                heartbeat_ts = int(parts[3]) if len(parts) >= 4 else int(parts[1])
                heartbeat_age = time.time() - heartbeat_ts
                import psutil  # type: ignore[import-untyped]
                if psutil.pid_exists(existing_pid):
                    try:
                        proc = psutil.Process(existing_pid)
                        cmdline = " ".join(proc.cmdline())
                        if "fill_test" in cmdline or "run_fill_test" in cmdline:
                            # 129# heartbeat stale 検査
                            if heartbeat_age > self._lock_stale_heartbeat_sec:
                                logger.warning(
                                    f"[lock] PID {existing_pid} alive but "
                                    f"heartbeat stale ({heartbeat_age:.0f}s > "
                                    f"{self._lock_stale_heartbeat_sec:.0f}s). "
                                    f"Treating as stale."
                                )
                            else:
                                raise RuntimeError(
                                    f"別の fill_test プロセスが実行中です "
                                    f"(PID={existing_pid}, "
                                    f"heartbeat={heartbeat_age:.0f}s ago). "
                                    f"強制起動するにはロックファイルを削除: {lock_path}"
                                )
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except (ValueError, ImportError, OSError):
                pass
            # stale lock — 削除して再取得を試みる
            logger.warning(f"[lock] Stale lockfile detected, reclaiming: {lock_path}")
            try:
                lock_path.unlink()
            except OSError:
                pass
            return True

        # 047# A4: open(path, 'x') で排他的にファイル作成 (atomic)
        for _attempt in range(self._lock_acquire_retries):
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, lock_content.encode("utf-8"))
                finally:
                    os.close(fd)
                logger.info(
                    f"[lock] Acquired lockfile: PID={os.getpid()}, run_id={self._run_id}"
                )
                return
            except FileExistsError:
                _check_stale_and_reclaim()
        # リトライ後もダメな場合
        raise RuntimeError(f"ロックファイルの取得に失敗しました: {lock_path}")

    def release(self) -> None:
        """044# ロックファイル解放."""
        if self._lockfile_path and self._lockfile_path.exists():
            try:
                content = self._lockfile_path.read_text(encoding="utf-8").strip()
                if content.startswith(f"{os.getpid()}|"):
                    self._lockfile_path.unlink()
                    logger.info("[lock] Released lockfile")
            except Exception as e:
                logger.warning(f"[lock] Failed to release lockfile: {e}")

    def update_heartbeat(self) -> None:
        """129# D.3: lock ファイルの heartbeat timestamp を更新.

        PID alive だが non-functional な状態を検出可能にする。
        フォーマット: PID|created_ts|run_id|heartbeat_ts
        """
        if not self._lockfile_path or not self._lockfile_path.exists():
            return
        try:
            content = self._lockfile_path.read_text(encoding="utf-8").strip()
            parts = content.split("|")
            if not content.startswith(f"{os.getpid()}|"):
                return  # 自プロセスのロックでない
            # heartbeat_ts (4番目) を更新
            now_ts = str(int(time.time()))
            if len(parts) >= 4:
                parts[3] = now_ts
            else:
                parts.append(now_ts)
            self._lockfile_path.write_text("|".join(parts), encoding="utf-8")
        except Exception:
            pass  # heartbeat 更新失敗は致命的ではない
