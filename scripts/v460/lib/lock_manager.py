"""044# 単一起動ロック管理.

run_fill_test.py から分離 (158# P2-4: god object 分割).
047# A4: TOCTOU race 対策 — open(path, 'x') で排他的作成。
129# D.3: heartbeat timestamp をロックファイルに記録。
286# 283#/284# P0: portalocker による OS レベルロック強化 +
     PID 死活確認ループ (zombie プロセス排除)。
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import psutil  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# 286#: OS レベル排他ロック — portalocker がなければフォールバック
try:
    import portalocker  # type: ignore[import-untyped]
    _HAS_PORTALOCKER = True
except ImportError:
    _HAS_PORTALOCKER = False
    logger.warning("[lock] portalocker not installed — OS-level lock unavailable")


def _read_lockfile_content(path: Path) -> str:
    """Read lockfile text with the shared stripping convention."""
    return path.read_text(encoding="utf-8").strip()


class LockConflictError(RuntimeError):
    """166# HF2: 別プロセス稼働中の正常終了用例外.

    RuntimeError の代わりにこの例外を raise することで、
    fill_test_cli.py 側で crash 扱いせず silent exit できる。
    """
    pass


class LockManager:
    """fill_test プロセスの単一起動ロック管理.

    同一 results_dir に対して複数プロセスが並行動作することを防止する。
    ロックファイルに PID を記録し、起動時に既存ロックの生死を検証する。
    129# D.3: heartbeat timestamp で stale 検知。
    286# 283#/284# P0: portalocker で OS レベル排他 + PID 死活確認ループ。
    """

    # 286#: zombie プロセス待機パラメータ
    _ZOMBIE_WAIT_INTERVAL_SEC: float = 1.0  # PID 消滅確認ポーリング間隔
    _ZOMBIE_WAIT_MAX_SEC: float = 30.0  # 最大待機時間 (超過で強制進行)

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
        # 286#: portalocker ファイルハンドル (プロセス終了時に自動解放)
        self._os_lock_fh: object | None = None

    @property
    def lockfile_path(self) -> Path | None:
        return self._lockfile_path

    def acquire(self) -> None:
        """044# Bug7: 単一起動ロック (lockfile + PID + stale 回収).

        047# A4: TOCTOU race 対策 — open(path, 'x') で排他的作成。
        286# P0: portalocker + zombie 待機で二重起動を完全排除。

        手順:
        1. 既存ロックの PID が生存中なら zombie 待機ループ
        2. PID 消滅確認後に stale lock を回収
        3. O_CREAT|O_EXCL で排他的にロック作成
        4. portalocker で OS レベル LOCK_EX (プロセス死亡時に自動解放)
        """
        lock_path = self._results_dir / "fill_test.lock"
        self._lockfile_path = lock_path
        now_ts = int(time.time())
        lock_content = f"{os.getpid()}|{now_ts}|{self._run_id}|{now_ts}"

        def _check_stale_and_reclaim() -> bool:
            """既存ロックが stale なら削除して True を返す."""
            try:
                content = _read_lockfile_content(lock_path)
                parts = content.split("|")
                existing_pid = int(parts[0])
                # 129# heartbeat age 検査 (4番目フィールド)
                heartbeat_ts = int(parts[3]) if len(parts) >= 4 else int(parts[1])
                heartbeat_age = time.time() - heartbeat_ts
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
                                # 286#: stale プロセスの終了を待機
                                self._wait_for_pid_exit(existing_pid)
                            else:
                                raise LockConflictError(
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
                # 286#: portalocker で OS レベルロックを追加取得
                self._acquire_os_lock(lock_path)
                return
            except FileExistsError:
                _check_stale_and_reclaim()
        # リトライ後もダメな場合
        raise LockConflictError(f"ロックファイルの取得に失敗しました: {lock_path}")

    def _wait_for_pid_exit(self, pid: int) -> None:
        """286# 284# P0: PID がプロセスツリーから消滅するまで待機.

        Watchdog kill 後の zombie プロセスが残存するケースに対応。
        psutil.wait_procs() で最大 _ZOMBIE_WAIT_MAX_SEC 待機し、
        消えなければ警告をログして強制的に進行する。
        """
        logger.info(
            f"[lock] 286# Waiting for PID {pid} to exit "
            f"(max {self._ZOMBIE_WAIT_MAX_SEC:.0f}s)..."
        )
        start = time.monotonic()
        while time.monotonic() - start < self._ZOMBIE_WAIT_MAX_SEC:
            if not psutil.pid_exists(pid):
                elapsed = time.monotonic() - start
                logger.info(
                    f"[lock] 286# PID {pid} exited after {elapsed:.1f}s"
                )
                return
            try:
                proc = psutil.Process(pid)
                cmdline = " ".join(proc.cmdline())
                if "fill_test" not in cmdline and "run_fill_test" not in cmdline:
                    logger.info(
                        f"[lock] 286# PID {pid} is now a different process — "
                        f"treating as exited"
                    )
                    return
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                logger.info(f"[lock] 286# PID {pid} no longer accessible")
                return
            time.sleep(self._ZOMBIE_WAIT_INTERVAL_SEC)
        logger.warning(
            f"[lock] 286# PID {pid} still alive after "
            f"{self._ZOMBIE_WAIT_MAX_SEC:.0f}s — proceeding with lock reclaim"
        )

    def _acquire_os_lock(self, lock_path: Path) -> None:
        """286# 284# P0: portalocker で OS レベル排他ロックを取得.

        プロセスが死亡 (SIGKILL, crash, 電源断等) すると OS が自動で
        ロックを解放する。O_CREAT|O_EXCL の論理ロックの上にハードウェア
        レベルの排他を重畳し、zombie ロックファイル問題を根本排除する。

        portalocker が未インストールの場合は graceful にスキップ。
        """
        if not _HAS_PORTALOCKER:
            logger.debug("[lock] 286# portalocker not available — skipping OS lock")
            return
        try:
            # 排他ロック用の別ファイル (.os_lock) を使用
            # ロックファイル本体は PID 情報の read/write に使うため分離
            os_lock_path = lock_path.with_suffix(".os_lock")
            fh = open(os_lock_path, "w", encoding="utf-8")
            portalocker.lock(fh, portalocker.LOCK_EX | portalocker.LOCK_NB)
            fh.write(f"{os.getpid()}|{self._run_id}\n")
            fh.flush()
            self._os_lock_fh = fh
            logger.info(
                f"[lock] 286# OS-level lock acquired: {os_lock_path}"
            )
        except portalocker.LockException:
            raise LockConflictError(
                f"286# OS レベルロック取得失敗 — "
                f"別プロセスが排他ロックを保持中: {lock_path}"
            )
        except Exception as e:
            logger.warning(f"[lock] 286# OS lock failed (non-fatal): {e}")

    def release(self) -> None:
        """044# ロックファイル解放. 286#: OS ロックも同時解放."""
        # 286#: OS レベルロック解放 (先に)
        if self._os_lock_fh is not None:
            try:
                if _HAS_PORTALOCKER:
                    portalocker.unlock(self._os_lock_fh)
                self._os_lock_fh.close()  # type: ignore[union-attr]
                logger.info("[lock] 286# OS-level lock released")
            except Exception as e:
                logger.warning(f"[lock] 286# OS lock release failed: {e}")
            finally:
                self._os_lock_fh = None
        # 論理ロックファイル解放
        if self._lockfile_path and self._lockfile_path.exists():
            try:
                content = _read_lockfile_content(self._lockfile_path)
                if content.startswith(f"{os.getpid()}|"):
                    self._lockfile_path.unlink()
                    # 286#: .os_lock ファイルも削除
                    os_lock_path = self._lockfile_path.with_suffix(".os_lock")
                    if os_lock_path.exists():
                        try:
                            os_lock_path.unlink()
                        except OSError:
                            pass
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
            content = _read_lockfile_content(self._lockfile_path)
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
        except Exception as e:
            # 255# bare except → debug log (heartbeat 更新失敗は致命的ではないが可観測化)
            logger.debug("lockfile heartbeat update failed: %s", e, exc_info=True)
