"""166# Hotfix tests: HF1 rescue config, HF2 lock conflict, HF3 insufficient cooldown."""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.v460.lib.lock_manager import LockConflictError, LockManager
from scripts.v460.lib.balance_checker import BalanceChecker


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
        from scripts.v460.lib.fill_config import FillTestConfig
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
# HF1: balance_forced_rescue config tests
# ======================================================================


class TestRescueConfig:
    """166# HF1: rescue 設定が YAML から正しく読み込まれる."""

    def test_default_rescue_disabled(self):
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.balance_forced_rescue_enabled is False
        assert cfg.balance_forced_rescue_offset_mult == 2.0

    def test_yaml_rescue_enabled(self):
        import yaml
        from scripts.v460.lib.fill_config import FillTestConfig
        yaml_cfg = yaml.safe_load(open("configs/v460/fill_test.yaml"))
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.balance_forced_rescue_enabled is True
        assert cfg.balance_forced_rescue_offset_mult == 2.0


# ======================================================================
# HF4: trending sell skip rebalance relaxation
# ======================================================================

class TestTrendingSellSkipRebalance:
    """166# HF4: buy 残高不足時に trending sell skip を緩和."""

    def test_hf4_code_exists_in_orchestrator(self):
        """HF4 コードがオーケストレータに存在する."""
        import inspect
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        source = inspect.getsource(FillLoopOrchestratorMixin)
        assert "166# HF4" in source
        assert "buy side insufficient" in source or "buy_insufficient" in source

    def test_max_consecutive_trending_sell_skip_yaml(self):
        """YAML で max_consecutive_trending_sell_skip=10 が設定されている."""
        import yaml
        from pathlib import Path
        cfg_path = Path("configs/v460/fill_test.yaml")
        with open(cfg_path) as f:
            raw = yaml.safe_load(f)
        val = raw.get("loss_control", {}).get("max_consecutive_trending_sell_skip", None)
        assert val == 10, f"Expected 10, got {val}"

    def test_config_loads_max_consecutive(self):
        """FillTestConfig が max_consecutive_trending_sell_skip を正しくロードする."""
        import yaml
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg_path = "configs/v460/fill_test.yaml"
        with open(cfg_path) as f:
            raw = yaml.safe_load(f)
        cfg = FillTestConfig.from_yaml(raw)
        assert cfg.max_consecutive_trending_sell_skip == 10


# ======================================================================
# HF4: trending sell skip rebalance relaxation
# ======================================================================

class TestTrendingSellSkipRebalance:
    """166# HF4: buy 残高不足時に trending sell skip を緩和."""

    def test_hf4_code_exists_in_orchestrator(self):
        """HF4 コードがオーケストレータに存在する."""
        import inspect
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        source = inspect.getsource(FillLoopOrchestratorMixin)
        assert "166# HF4" in source
        assert "buy side insufficient" in source or "buy_insufficient" in source

    def test_max_consecutive_trending_sell_skip_yaml(self):
        """YAML で max_consecutive_trending_sell_skip=10 が設定されている."""
        import yaml
        from pathlib import Path
        cfg_path = Path("configs/v460/fill_test.yaml")
        with open(cfg_path) as f:
            raw = yaml.safe_load(f)
        val = raw.get("loss_control", {}).get("max_consecutive_trending_sell_skip", None)
        assert val == 10, f"Expected 10, got {val}"

    def test_config_loads_max_consecutive(self):
        """FillTestConfig が max_consecutive_trending_sell_skip を正しくロードする."""
        import yaml
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg_path = "configs/v460/fill_test.yaml"
        with open(cfg_path) as f:
            raw = yaml.safe_load(f)
        cfg = FillTestConfig.from_yaml(raw)
        assert cfg.max_consecutive_trending_sell_skip == 10
