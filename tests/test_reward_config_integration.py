import sys
import os
import tempfile
import shutil
from pathlib import Path
import dataclasses

import pytest

# Ensure project root is on sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ztb.training.reward_config_schema import RewardConfigSchema, load_reward_config
from scripts.v459.run_day6_reward_tuning import create_experiment_config, save_results
from ztb.trading.environment.utils.config import RewardSettings


def test_stage1_yamls_validate():
    cfg_dir = PROJECT_ROOT / "configs" / "rewards"
    yamls = [
        cfg_dir / "stage1_basic.yaml",
        cfg_dir / "stage1_hold_removed.yaml",
        cfg_dir / "stage1_trade_reduced.yaml",
        cfg_dir / "stage1_exploration_tuned.yaml",
    ]

    for y in yamls:
        assert y.exists(), f"Missing YAML: {y}"
        cfg = RewardConfigSchema.load_and_validate(str(y))
        assert isinstance(cfg, dict)
        assert "name" in cfg
        assert "reward_scale" in cfg


def test_load_reward_config_returns_dataclass():
    cfg_path = PROJECT_ROOT / "configs" / "rewards" / "stage1_hold_removed.yaml"
    rs = load_reward_config(str(cfg_path))
    assert isinstance(rs, RewardSettings)
    # custom param must contain hold_penalty_multiplier
    assert hasattr(rs, "hold_penalty_multiplier")


def test_create_experiment_injects_reward_settings():
    config = create_experiment_config("C_Test", 42, "configs/rewards/stage1_hold_removed.yaml", {})
    env = config["training"]["environment"]
    assert "reward_settings" in env
    assert isinstance(env["reward_settings"], dict)
    assert env["reward_settings"].get("name") == "stage1_hold_removed"
    # behavior_optimization should be propagated into environment for runtime mapping
    assert "behavior_optimization" in env


def test_save_results_serializes_reward_settings(tmp_path):
    # Create a fake result containing RewardSettings dataclass
    rs = RewardSettings()
    results = [
        {
            "experiment_name": "test",
            "status": "completed",
            "timestamp": "2026-01-29T00:00:00",
            "config": {"training": {"environment": {"reward_settings": rs}}},
            "metrics": {"final_reward": 1.23},
            "report": {"training_stats": {"final_reward": 1.23}}
        }
    ]

    out_dir = tmp_path / "out"
    save_results(results, out_dir)

    # Check files created
    files = list(out_dir.glob("*.json"))
    assert any("day6_reward_tuning_" in f.name for f in files)
    assert any("day6_summary_" in f.name for f in files)