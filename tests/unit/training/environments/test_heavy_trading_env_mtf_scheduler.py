import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from ztb.trading.environment.utils.config import EnvironmentConfig as TradingEnvConfig
from ztb.training.environments.heavy_trading_env import HeavyTradingEnv


class FakeDataFrame:
    """Very small fake DataFrame implementing minimal pandas-like subset used by HeavyTradingEnv tests."""

    def __init__(self, rows=None):
        self._rows = rows or [
            {"close": 100.0, "open": 99.0, "high": 101.0, "low": 98.0, "volume": 10}
        ]
        # columns attribute expected by HeavyTradingEnv
        self.columns = list(self._rows[0].keys()) if self._rows else []

    def copy(self):
        return FakeDataFrame(rows=self._rows.copy())

    def ffill(self):
        return self

    def fillna(self, _):
        return self

    @property
    def iloc(self):
        return self

    def __getitem__(self, idx):
        return self._rows[idx]

    def __len__(self):
        return len(self._rows)


def _make_base_df():
    return FakeDataFrame()


def test_env_registers_mtf_scheduler(tmp_path: Path):
    # create minimal base config file for optimizer
    base_cfg = tmp_path / "base_config.json"
    base_cfg.write_text(
        json.dumps(
            {
                "training": {"model_name": "mtf_test"},
                "multi_timeframe": {
                    "feature_weights": {"1min": 0.3, "5min": 0.6, "15min": 0.1}
                },
            }
        )
    )

    # Build training config-like object - we only need the attributes HeavyTradingEnv reads
    cfg = SimpleNamespace()
    cfg.initial_portfolio_value = 100000.0
    cfg.commission = 0.0
    cfg.max_position_size = 1.0
    cfg.reward_scaling = 1.0
    cfg.feature_set = "full"
    cfg.curriculum_stage = "forced_balance"
    cfg.base_action_penalty = 0.02
    cfg.action_bonuses = {
        "buy_action_bonus": 0.0,
        "sell_action_bonus": 0.0,
        "hold_action_bonus": 0.0,
    }

    # behavior attribute which HeavyTradingEnv looks for
    # add arbitrary behavior attribute to configuration - tests can add attributes dynamically
    cfg.behavior = {
        "mtf": {
            "weight_optimizer": {
                "enabled": True,
                "base_config": str(base_cfg),
                "out_dir": str(tmp_path / "mtf_candidates"),
                "candidates": 3,
                "per_seed": 1,
                "timesteps": 10,
                "dry_run": True,
                "stage_filter": ["balanced_transition"],
            }
        }
    }

    env = HeavyTradingEnv(
        data=cast(Any, _make_base_df()),
        config=cast(TradingEnvConfig, cfg),
        feature_columns=["close", "open", "high", "low", "volume"],
        reward_settings={},
    )

    # MTFScheduler should be attached to env and a callback should be registered
    assert hasattr(env, "mtf_scheduler")
    assert hasattr(env.reward_calculator, "mtf_scheduler")

    bcm = env.reward_calculator.curriculum_manager
    assert len(bcm.stage_change_listeners) > 0


def test_stage_change_triggers_optimizer(tmp_path: Path):
    base_cfg = tmp_path / "base_config.json"
    base_cfg.write_text(
        json.dumps(
            {
                "training": {"model_name": "mtf_test"},
                "multi_timeframe": {
                    "feature_weights": {"1min": 0.3, "5min": 0.6, "15min": 0.1}
                },
            }
        )
    )

    cfg = SimpleNamespace()
    cfg.initial_portfolio_value = 100000.0
    cfg.commission = 0.0
    cfg.max_position_size = 1.0
    cfg.reward_scaling = 1.0
    cfg.feature_set = "full"
    cfg.curriculum_stage = "forced_balance"
    cfg.base_action_penalty = 0.02
    cfg.action_bonuses = {
        "buy_action_bonus": 0.0,
        "sell_action_bonus": 0.0,
        "hold_action_bonus": 0.0,
    }

    cfg.behavior = {
        "mtf": {
            "weight_optimizer": {
                "enabled": True,
                "base_config": str(base_cfg),
                "out_dir": str(tmp_path / "mtf_candidates"),
                "candidates": 3,
                "per_seed": 1,
                "timesteps": 10,
                "dry_run": True,
                "stage_filter": ["balanced_transition"],
            }
        }
    }

    env = HeavyTradingEnv(
        data=cast(Any, _make_base_df()),
        config=cast(TradingEnvConfig, cfg),
        feature_columns=["close", "open", "high", "low", "volume"],
        reward_settings={},
    )
    mgr = env.reward_calculator.mtf_weight_manager
    before = mgr.get_weights().copy()
    # trigger stage change to balanced_transition
    env.reward_calculator.curriculum_manager._progress_to_stage(
        "balanced_transition", step=200
    )
    after = mgr.get_weights()
    assert abs(sum(after.values()) - 1.0) < 1e-9
    # We expect weights to be updated away from base
    assert before != after
