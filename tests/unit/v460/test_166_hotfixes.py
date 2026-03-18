"""166# Hotfix tests: HF1 rescue config, HF2 lock conflict, HF3 insufficient cooldown."""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.v460.lib.balance_checker import BalanceChecker
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.lock_manager import LockConflictError, LockManager


@pytest.fixture(scope="module")
def hotfix_yaml(v460_fill_test_yaml_base: dict[str, object]) -> dict[str, object]:
    """fill_test.yaml の共通キャッシュ."""
    return v460_fill_test_yaml_base


@pytest.fixture(scope="module")
def cycle_gate_source() -> str:
    return Path("scripts/v460/lib/cycle_gate_aggregator.py").read_text(encoding="utf-8")


# ======================================================================
# HF2: LockConflictError tests
# ======================================================================


class TestLockConflictError:
    """166# HF2: LockConflictError は RuntimeError のサブクラス."""

    def test_is_runtime_error(self):
        assert issubclass(LockConflictError, RuntimeError)

    def test_can_catch_as_runtime(self):
        with pytest.raises(RuntimeError):
            raise LockConflictError("test")

    def test_can_catch_specifically(self):
        with pytest.raises(LockConflictError):
            raise LockConflictError("test")

    def test_message(self):
        err = LockConflictError("別のプロセスが実行中")
        assert "別のプロセスが実行中" in str(err)


class TestLockManagerConflict:
    """166# HF2: LockManager.acquire() が LockConflictError を raise."""

    def test_raises_lock_conflict_on_active_process(self, tmp_path: Path):
        """既存プロセスが alive + heartbeat valid  LockConflictError."""
        lock_path = tmp_path / "fill_test.lock"
        pid = os.getpid()
        now = int(time.time())
        lock_path.write_text(f"{pid}|{now}|test_run|{now}", encoding="utf-8")

        # Mock psutil so cmdline contains "fill_test"  active process detection
        mock_proc = MagicMock()
        mock_proc.cmdline.return_value = ["python", "run_fill_test.py"]

        mgr = LockManager(
            tmp_path, "new_run",
            lock_stale_heartbeat_sec=600.0,
            lock_acquire_retries=1,
        )
        with patch("scripts.v460.lib.lock_manager.psutil") as mock_psutil:
            mock_psutil.pid_exists.return_value = True
            mock_psutil.Process.return_value = mock_proc
            mock_psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
            mock_psutil.AccessDenied = type("AccessDenied", (Exception,), {})
            with pytest.raises(LockConflictError, match="別の"):
                mgr.acquire()

    def test_stale_lock_reclaimed(self, tmp_path: Path):
        """stale heartbeat  ロック回収して正常取得."""
        lock_path = tmp_path / "fill_test.lock"
        pid = os.getpid()
        old_ts = int(time.time()) - 9999
        lock_path.write_text(f"{pid}|{old_ts}|old_run|{old_ts}", encoding="utf-8")

        mgr = LockManager(
            tmp_path, "new_run",
            lock_stale_heartbeat_sec=60.0,
            lock_acquire_retries=2,
        )
        with patch.object(mgr, "_check_running_fill_test", return_value=None):
            mgr.acquire()
            assert lock_path.exists()
            mgr.release()


# ======================================================================
# HF3: Insufficient balance cooldown tests
# ======================================================================


class TestInsufficientCooldown:
    """166# HF3: Insufficient 警告の side 別クールダウン."""

    @pytest.fixture
    def config(self):
        return FillTestConfig()

    def test_cooldown_state_init(self, config):
        bc = BalanceChecker(config)
        assert bc._insufficient_cooldown_sec == 120.0
        assert bc._last_insufficient_log == {}

    def test_first_log_always_emitted(self, config):
        bc = BalanceChecker(config)
        with patch("scripts.v460.lib.balance_checker.logger") as mock_logger:
            bc._log_insufficient("buy", "test message")
            mock_logger.warning.assert_called_once_with("test message")
            assert "buy" in bc._last_insufficient_log

    def test_second_log_suppressed_within_cooldown(self, config):
        bc = BalanceChecker(config)
        bc._insufficient_cooldown_sec = 120.0
        bc._last_insufficient_log["buy"] = time.time()  # just logged
        with patch("scripts.v460.lib.balance_checker.logger") as mock_logger:
            bc._log_insufficient("buy", "should be suppressed")
            mock_logger.warning.assert_not_called()
            mock_logger.debug.assert_called_once()

    def test_different_side_not_suppressed(self, config):
        bc = BalanceChecker(config)
        bc._last_insufficient_log["buy"] = time.time()
        with patch("scripts.v460.lib.balance_checker.logger") as mock_logger:
            bc._log_insufficient("sell", "sell message")
            mock_logger.warning.assert_called_once_with("sell message")

    def test_log_after_cooldown_expires(self, config):
        bc = BalanceChecker(config)
        bc._insufficient_cooldown_sec = 0.0  # zero cooldown
        bc._last_insufficient_log["buy"] = time.time() - 1  # expired
        with patch("scripts.v460.lib.balance_checker.logger") as mock_logger:
            bc._log_insufficient("buy", "should log again")
            mock_logger.warning.assert_called_once()


# ======================================================================
# HF1: balance_forced_rescue config — 348# 撤廃
# ======================================================================


# ======================================================================
# HF4: trending sell skip rebalance relaxation
# ======================================================================

class TestTrendingSellSkipRebalance:
    """166# HF4: buy 残高不足時に trending sell skip を緩和.

    194#: HF4 ロジックは CycleGateAggregator に集約。
    """

    def test_hf4_code_exists_in_orchestrator(self, cycle_gate_source: str):
        """HF4 コードが CycleGateAggregator に存在する."""
        assert "166# HF4" in cycle_gate_source
        assert (
            "buy side insufficient" in cycle_gate_source
            or "buy_side_insufficient" in cycle_gate_source
        )

    def test_max_consecutive_trending_sell_skip_yaml(
        self,
        hotfix_yaml: dict[str, object],
    ):
        """YAML で max_consecutive_trending_sell_skip=10 が設定されている (171# 20→10)."""
        val = hotfix_yaml.get("loss_control", {}).get("max_consecutive_trending_sell_skip", None)
        assert val == 10, f"Expected 10, got {val}"

    def test_config_loads_max_consecutive(self, hotfix_yaml: dict[str, object]):
        """FillTestConfig が max_consecutive_trending_sell_skip を正しくロードする (171# 10)."""
        cfg = FillTestConfig.from_yaml(hotfix_yaml)
        assert cfg.max_consecutive_trending_sell_skip == 10


# ======================================================================
# HF4: trending sell skip rebalance relaxation
# ======================================================================

class TestTrendingSellSkipRebalance2:
    """166# HF4: buy 残高不足時に trending sell skip を緩和 (duplicate cleanup).

    194#: HF4 ロジックは CycleGateAggregator に集約。
    """

    def test_hf4_code_exists_in_orchestrator(self, cycle_gate_source: str):
        """HF4 コードが CycleGateAggregator に存在する."""
        assert "166# HF4" in cycle_gate_source
        assert (
            "buy side insufficient" in cycle_gate_source
            or "buy_side_insufficient" in cycle_gate_source
        )

    def test_max_consecutive_trending_sell_skip_yaml(
        self,
        hotfix_yaml: dict[str, object],
    ):
        """YAML で max_consecutive_trending_sell_skip=10 が設定されている (171# 20→10)."""
        val = hotfix_yaml.get("loss_control", {}).get("max_consecutive_trending_sell_skip", None)
        assert val == 10, f"Expected 10, got {val}"

    def test_config_loads_max_consecutive(self, hotfix_yaml: dict[str, object]):
        """FillTestConfig が max_consecutive_trending_sell_skip を正しくロードする (169# C3: 20)."""
        cfg = FillTestConfig.from_yaml(hotfix_yaml)
        assert cfg.max_consecutive_trending_sell_skip == 10  # 171# 20→10
