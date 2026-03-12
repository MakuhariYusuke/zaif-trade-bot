
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.trading.environment.components.reward_calculator import RewardCalculator
from ztb.trading.environment.utils.config import RewardSettings
from ztb.trading.constants import ACTION_HOLD


def test_mtf_weights_propagate_to_rewardcalc():
    cfg = EnvironmentConfig()
    rs = RewardSettings(
        use_simple_reward=False,
        reward_scale=1.0,
        trading_bonus=0.01,
        profit_bonuses={},
        penalty_coefficients={},
    )
    rc = RewardCalculator(cfg, rs, initial_portfolio_value=10000.0)
    # Sanity: default weights
    initial_weights = rc.mtf_weight_manager.get_weights()
    # Simulate applying metrics that highly favor 5min
    metrics = {
        "tf_metrics": {
            "1min": {"sharpe": 0.01},
            "5min": {"sharpe": 10.0},
            "15min": {"sharpe": 0.01},
        }
    }
    rc.mtf_weight_manager.update(0, metrics)
    # After update, the 5min weight should have increased
    updated_weights = rc.mtf_weight_manager.get_weights()
    assert updated_weights["5min"] >= initial_weights["5min"]
    # And it should be present in last reward telemetry after a reward calc
    rc.calculate_reward(
        action=ACTION_HOLD,
        current_price=100.0,
        position=0.0,
        portfolio_value=10000.0,
        atr=1.0,
        transaction_cost=0.001,
        reward_scaling=1.0,
        pnl=0.0,
        old_position=0.0,
        step=1,
        observation=None,
        reward_history=[],
        portfolio_value_history=[10000.0],
    )
    comps = rc.get_last_reward_components()
    assert "mtf_weights" in comps
    assert comps["mtf_weights"] == updated_weights
