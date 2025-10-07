"""
Tests for GradProbeGuard functionality.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ztb.training.grad_probe_guard import (
    GradProbeConfig,
    GradProbeGuard,
    GradProbeStats,
)


class TestGradProbeConfig:
    """Test GradProbeConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GradProbeConfig()
        
        assert config.zero_threshold == 1e-8
        assert config.consecutive_zeros == 5
        assert config.check_interval == 1000
        assert config.monitor_actions == ["SELL", "BUY", "HOLD"]
        assert config.critical_actions == ["SELL"]
        assert config.save_replay_buffer is True
        assert config.save_model_state is True
        assert config.save_diagnostics is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = GradProbeConfig(
            zero_threshold=1e-6,
            consecutive_zeros=10,
            check_interval=500,
            critical_actions=["SELL", "BUY"],
        )
        
        assert config.zero_threshold == 1e-6
        assert config.consecutive_zeros == 10
        assert config.check_interval == 500
        assert config.critical_actions == ["SELL", "BUY"]


class TestGradProbeStats:
    """Test GradProbeStats dataclass."""
    
    def test_stats_creation(self):
        """Test creating gradient probe stats."""
        stats = GradProbeStats(
            step=1000,
            timestamp=123456.789,
            action_grads={"SELL": 0.001, "BUY": 0.002, "HOLD": 0.003},
            grad_norms={"SELL": 0.001, "BUY": 0.002, "HOLD": 0.003},
            is_zero={"SELL": False, "BUY": False, "HOLD": False},
            consecutive_zero_count={"SELL": 0, "BUY": 0, "HOLD": 0},
        )
        
        assert stats.step == 1000
        assert stats.timestamp == 123456.789
        assert stats.action_grads["SELL"] == 0.001
        assert stats.is_zero["SELL"] is False


class TestGradProbeGuard:
    """Test GradProbeGuard callback."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model with policy."""
        model = MagicMock()
        model.policy = MagicMock()
        model.policy.action_net = MagicMock()
        model.policy.action_net.weight = MagicMock()
        
        # Mock gradient
        mock_grad = MagicMock()
        mock_grad.shape = (64, 3)  # (features, n_actions)
        mock_grad.cpu.return_value.detach.return_value.numpy.return_value = np.random.randn(64, 3)
        model.policy.action_net.weight.grad = mock_grad
        
        return model
    
    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary checkpoint directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_guard_initialization(self, temp_checkpoint_dir):
        """Test GradProbeGuard initialization."""
        config = GradProbeConfig(consecutive_zeros=3)
        guard = GradProbeGuard(
            config=config,
            checkpoint_dir=str(temp_checkpoint_dir),
            session_id="test_session",
            verbose=1,
        )
        
        assert guard.config.consecutive_zeros == 3
        assert guard.session_id == "test_session"
        assert guard.halt_triggered is False
        assert guard.halt_reason is None
        assert len(guard.consecutive_zeros) == 3  # SELL, BUY, HOLD
    
    def test_extract_grad_stats_no_policy(self, temp_checkpoint_dir):
        """Test extracting stats when model has no policy."""
        guard = GradProbeGuard(checkpoint_dir=str(temp_checkpoint_dir))
        guard.model = MagicMock(spec=[])  # No policy attribute
        guard.num_timesteps = 1000
        
        stats = guard._extract_grad_stats()
        assert stats is None
    
    def test_extract_grad_stats_with_gradients(self, temp_checkpoint_dir, mock_model):
        """Test extracting gradient stats from model."""
        guard = GradProbeGuard(checkpoint_dir=str(temp_checkpoint_dir))
        guard.model = mock_model
        guard.num_timesteps = 1000
        
        stats = guard._extract_grad_stats()
        
        assert stats is not None
        assert stats.step == 1000
        assert isinstance(stats.action_grads, dict)
    
    def test_check_zero_gradients_normal(self, temp_checkpoint_dir):
        """Test check when gradients are normal (not zero)."""
        config = GradProbeConfig(consecutive_zeros=5, critical_actions=["SELL"])
        guard = GradProbeGuard(config=config, checkpoint_dir=str(temp_checkpoint_dir))
        
        stats = GradProbeStats(
            step=1000,
            timestamp=123456.789,
            consecutive_zero_count={"SELL": 0, "BUY": 0, "HOLD": 0},
        )
        
        should_halt = guard._check_zero_gradients(stats)
        assert should_halt is False
    
    def test_check_zero_gradients_halt_triggered(self, temp_checkpoint_dir):
        """Test halt when zero gradients exceed threshold."""
        config = GradProbeConfig(consecutive_zeros=5, critical_actions=["SELL"])
        guard = GradProbeGuard(config=config, checkpoint_dir=str(temp_checkpoint_dir))
        
        stats = GradProbeStats(
            step=1000,
            timestamp=123456.789,
            consecutive_zero_count={"SELL": 5, "BUY": 0, "HOLD": 0},
        )
        
        should_halt = guard._check_zero_gradients(stats)
        assert should_halt is True
        assert "SELL" in guard.halt_reason
        assert "zero" in guard.halt_reason.lower()
    
    def test_create_manifest(self, temp_checkpoint_dir):
        """Test manifest creation."""
        config = GradProbeConfig(consecutive_zeros=5)
        guard = GradProbeGuard(
            config=config,
            checkpoint_dir=str(temp_checkpoint_dir),
            session_id="test_session",
        )
        guard.halt_reason = "SELL action gradient stuck at zero"
        
        stats = GradProbeStats(
            step=10000,
            timestamp=123456.789,
            action_grads={"SELL": 0.0, "BUY": 0.001, "HOLD": 0.002},
            grad_norms={"SELL": 0.0, "BUY": 0.001, "HOLD": 0.002},
            is_zero={"SELL": True, "BUY": False, "HOLD": False},
            consecutive_zero_count={"SELL": 5, "BUY": 0, "HOLD": 0},
        )
        
        manifest = guard._create_manifest(stats, "20251007_120000")
        
        assert manifest["session_id"] == "test_session"
        assert manifest["halt_reason"] == "SELL action gradient stuck at zero"
        assert manifest["halt_step"] == 10000
        assert manifest["config"]["consecutive_zeros"] == 5
        assert manifest["final_stats"]["step"] == 10000
        assert manifest["final_stats"]["action_grads"]["SELL"] == 0.0
    
    def test_save_diagnostics(self, temp_checkpoint_dir):
        """Test saving diagnostics."""
        guard = GradProbeGuard(checkpoint_dir=str(temp_checkpoint_dir))
        
        # Add some history
        for i in range(3):
            stats = GradProbeStats(
                step=i * 1000,
                timestamp=123456.0 + i,
                action_grads={"SELL": 0.001, "BUY": 0.002, "HOLD": 0.003},
                grad_norms={"SELL": 0.001, "BUY": 0.002, "HOLD": 0.003},
                is_zero={"SELL": False, "BUY": False, "HOLD": False},
                consecutive_zero_count={"SELL": 0, "BUY": 0, "HOLD": 0},
            )
            guard.history.append(stats)
        
        archive_dir = temp_checkpoint_dir / "test_archive"
        archive_dir.mkdir()
        
        final_stats = GradProbeStats(
            step=3000,
            timestamp=123459.0,
            action_grads={"SELL": 0.0, "BUY": 0.002, "HOLD": 0.003},
            grad_norms={"SELL": 0.0, "BUY": 0.002, "HOLD": 0.003},
            is_zero={"SELL": True, "BUY": False, "HOLD": False},
            consecutive_zero_count={"SELL": 5, "BUY": 0, "HOLD": 0},
        )
        
        guard._save_diagnostics(archive_dir, final_stats)
        
        # Check files exist
        diagnostics_dir = archive_dir / "diagnostics"
        assert diagnostics_dir.exists()
        
        history_file = diagnostics_dir / "gradient_history.json"
        assert history_file.exists()
        
        with open(history_file) as f:
            history_data = json.load(f)
        assert len(history_data) == 3
        
        stats_file = diagnostics_dir / "final_stats.json"
        assert stats_file.exists()
        
        with open(stats_file) as f:
            stats_data = json.load(f)
        assert stats_data["step"] == 3000
        assert stats_data["action_grads"]["SELL"] == 0.0
    
    def test_get_stats_summary_no_data(self, temp_checkpoint_dir):
        """Test stats summary with no data."""
        guard = GradProbeGuard(checkpoint_dir=str(temp_checkpoint_dir))
        
        summary = guard.get_stats_summary()
        assert summary["status"] == "no_data"
    
    def test_get_stats_summary_with_data(self, temp_checkpoint_dir):
        """Test stats summary with data."""
        guard = GradProbeGuard(checkpoint_dir=str(temp_checkpoint_dir))
        
        stats = GradProbeStats(
            step=5000,
            timestamp=123456.789,
            action_grads={"SELL": 0.001, "BUY": 0.002, "HOLD": 0.003},
            grad_norms={"SELL": 0.001, "BUY": 0.002, "HOLD": 0.003},
            is_zero={"SELL": False, "BUY": False, "HOLD": False},
            consecutive_zero_count={"SELL": 0, "BUY": 0, "HOLD": 0},
        )
        guard.history.append(stats)
        
        summary = guard.get_stats_summary()
        
        assert summary["step"] == 5000
        assert summary["timestamp"] == 123456.789
        assert summary["action_grads"]["SELL"] == 0.001
        assert summary["halt_triggered"] is False
        assert summary["history_length"] == 1
    
    def test_on_step_skip_check_interval(self, temp_checkpoint_dir, mock_model):
        """Test that checks are skipped when below check_interval."""
        config = GradProbeConfig(check_interval=1000)
        guard = GradProbeGuard(config=config, checkpoint_dir=str(temp_checkpoint_dir))
        guard.model = mock_model
        guard.num_timesteps = 500
        guard.last_check_step = 0
        
        result = guard._on_step()
        
        # Should return True without checking (below interval)
        assert result is True
    
    def test_on_step_halt_when_zero_gradients(self, temp_checkpoint_dir, mock_model):
        """Test halt when zero gradients detected."""
        config = GradProbeConfig(
            check_interval=1000,
            consecutive_zeros=1,  # Halt after 1 zero
            critical_actions=["SELL"],
        )
        guard = GradProbeGuard(config=config, checkpoint_dir=str(temp_checkpoint_dir))
        guard.model = mock_model
        guard.num_timesteps = 1000
        guard.last_check_step = 0
        
        # Mock extract_grad_stats to return zero gradient for SELL
        def mock_extract():
            return GradProbeStats(
                step=1000,
                timestamp=123456.789,
                action_grads={"SELL": 0.0, "BUY": 0.002, "HOLD": 0.003},
                grad_norms={"SELL": 0.0, "BUY": 0.002, "HOLD": 0.003},
                is_zero={"SELL": True, "BUY": False, "HOLD": False},
                consecutive_zero_count={"SELL": 1, "BUY": 0, "HOLD": 0},
            )
        
        with patch.object(guard, '_extract_grad_stats', side_effect=mock_extract):
            with patch.object(guard, '_handle_zero_gradient_halt'):
                result = guard._on_step()
        
        # Should halt
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
