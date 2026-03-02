"""
System Parameter Space Manager

Manages parameter spaces for system-level optimization tasks.
Extends the generic ParameterSpace concept for system configuration optimization.
"""

from typing import Any

from ztb.training.hyperparameter_optimizer import ParameterSpace
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

class SystemParameterSpaceManager:
    """
    Manages parameter spaces for system-level optimization.

    This class provides reusable parameter space definitions for various
    system optimization tasks beyond reward function optimization.
    """

    def __init__(self):
        self.logger = get_logger(__name__)

    def get_system_parameter_spaces(self) -> dict[str, dict[str, ParameterSpace]]:
        """
        Get parameter spaces for different system optimization tasks.

        Returns:
            Dictionary of parameter spaces by optimization task
        """
        return {
            "environment_config": self._get_environment_config_space(),
            "training_hyperparams": self._get_training_hyperparams_space(),
            "risk_management": self._get_risk_management_space(),
            "market_adaptation": self._get_market_adaptation_space(),
        }

    def get_trading_parameter_spaces(self) -> dict[str, dict[str, ParameterSpace]]:
        """
        Get parameter spaces for trading system optimization.

        Returns:
            Dictionary of parameter spaces for trading optimization
        """
        return {
            "position_sizing": self._get_position_sizing_space(),
            "entry_exit_rules": self._get_entry_exit_rules_space(),
            "risk_controls": self._get_risk_controls_space(),
            "market_timing": self._get_market_timing_space(),
        }

    def _get_environment_config_space(self) -> dict[str, ParameterSpace]:
        """Get parameter space for environment configuration optimization."""
        return {
            "transaction_cost": ParameterSpace(
                name="transaction_cost",
                type="float",
                low=0.0001,
                high=0.01,
                log_scale=True,
            ),
            "slippage": ParameterSpace(
                name="slippage", type="float", low=0.0001, high=0.005, log_scale=True
            ),
            "max_position_size": ParameterSpace(
                name="max_position_size", type="float", low=0.01, high=1.0
            ),
            "min_position_size": ParameterSpace(
                name="min_position_size", type="float", low=0.001, high=0.1
            ),
        }

    def _get_training_hyperparams_space(self) -> dict[str, ParameterSpace]:
        """Get parameter space for training hyperparameters."""
        return {
            "learning_rate": ParameterSpace(
                name="learning_rate", type="float", low=1e-6, high=1e-2, log_scale=True
            ),
            "batch_size": ParameterSpace(
                name="batch_size",
                type="int",
                low=32,
                high=512,
                choices=[32, 64, 128, 256, 512],
            ),
            "gamma": ParameterSpace(name="gamma", type="float", low=0.9, high=0.999),
            "tau": ParameterSpace(
                name="tau", type="float", low=0.001, high=0.1, log_scale=True
            ),
        }

    def _get_risk_management_space(self) -> dict[str, ParameterSpace]:
        """Get parameter space for risk management parameters."""
        return {
            "max_drawdown_limit": ParameterSpace(
                name="max_drawdown_limit", type="float", low=0.05, high=0.3
            ),
            "daily_loss_limit": ParameterSpace(
                name="daily_loss_limit", type="float", low=0.01, high=0.1
            ),
            "position_limit": ParameterSpace(
                name="position_limit", type="float", low=0.1, high=2.0
            ),
            "var_limit": ParameterSpace(
                name="var_limit", type="float", low=0.01, high=0.05
            ),
        }

    def _get_market_adaptation_space(self) -> dict[str, ParameterSpace]:
        """Get parameter space for market adaptation parameters."""
        return {
            "regime_sensitivity": ParameterSpace(
                name="regime_sensitivity", type="float", low=0.1, high=2.0
            ),
            "volatility_threshold": ParameterSpace(
                name="volatility_threshold", type="float", low=0.01, high=0.1
            ),
            "trend_strength_min": ParameterSpace(
                name="trend_strength_min", type="float", low=0.1, high=0.8
            ),
            "adaptation_rate": ParameterSpace(
                name="adaptation_rate",
                type="float",
                low=0.001,
                high=0.1,
                log_scale=True,
            ),
        }

    def _get_position_sizing_space(self) -> dict[str, ParameterSpace]:
        """Get parameter space for position sizing optimization."""
        return {
            "base_position_size": ParameterSpace(
                name="base_position_size", type="float", low=0.01, high=0.2
            ),
            "volatility_scaling": ParameterSpace(
                name="volatility_scaling", type="float", low=0.5, high=3.0
            ),
            " Kelly_fraction": ParameterSpace(
                name="kelly_fraction", type="float", low=0.1, high=1.0
            ),
            "max_position_pct": ParameterSpace(
                name="max_position_pct", type="float", low=0.05, high=0.5
            ),
        }

    def _get_entry_exit_rules_space(self) -> dict[str, ParameterSpace]:
        """Get parameter space for entry/exit rules optimization."""
        return {
            "entry_threshold": ParameterSpace(
                name="entry_threshold", type="float", low=0.1, high=2.0
            ),
            "exit_threshold": ParameterSpace(
                name="exit_threshold", type="float", low=-2.0, high=-0.1
            ),
            "stop_loss_pct": ParameterSpace(
                name="stop_loss_pct", type="float", low=0.01, high=0.1
            ),
            "take_profit_pct": ParameterSpace(
                name="take_profit_pct", type="float", low=0.02, high=0.2
            ),
        }

    def _get_risk_controls_space(self) -> dict[str, ParameterSpace]:
        """Get parameter space for risk controls optimization."""
        return {
            "max_consecutive_losses": ParameterSpace(
                name="max_consecutive_losses", type="int", low=3, high=10
            ),
            "max_daily_trades": ParameterSpace(
                name="max_daily_trades", type="int", low=5, high=50
            ),
            "correlation_limit": ParameterSpace(
                name="correlation_limit", type="float", low=0.3, high=0.8
            ),
            "concentration_limit": ParameterSpace(
                name="concentration_limit", type="float", low=0.1, high=0.4
            ),
        }

    def _get_market_timing_space(self) -> dict[str, ParameterSpace]:
        """Get parameter space for market timing optimization."""
        return {
            "bull_threshold": ParameterSpace(
                name="bull_threshold", type="float", low=0.02, high=0.1
            ),
            "bear_threshold": ParameterSpace(
                name="bear_threshold", type="float", low=-0.1, high=-0.02
            ),
            "sideways_band": ParameterSpace(
                name="sideways_band", type="float", low=0.005, high=0.03
            ),
            "momentum_period": ParameterSpace(
                name="momentum_period",
                type="int",
                low=5,
                high=50,
                choices=[5, 10, 20, 30, 50],
            ),
        }

    def create_custom_parameter_space(
        self, name: str, param_definitions: dict[str, dict[str, Any]]
    ) -> dict[str, ParameterSpace]:
        """
        Create a custom parameter space from definitions.

        Args:
            name: Name of the parameter space
            param_definitions: Dictionary of parameter definitions

        Returns:
            Dictionary of ParameterSpace objects
        """
        parameter_space = {}

        for param_name, definition in param_definitions.items():
            param_type = definition.get("type", "float")

            if param_type == "float":
                parameter_space[param_name] = ParameterSpace(
                    name=param_name,
                    type="float",
                    low=definition.get("low"),
                    high=definition.get("high"),
                    log_scale=definition.get("log_scale", False),
                )
            elif param_type == "int":
                parameter_space[param_name] = ParameterSpace(
                    name=param_name,
                    type="int",
                    low=definition.get("low"),
                    high=definition.get("high"),
                    choices=definition.get("choices"),
                )
            elif param_type == "categorical":
                parameter_space[param_name] = ParameterSpace(
                    name=param_name,
                    type="categorical",
                    choices=definition.get("choices"),
                )

        self.logger.info(
            f"Created custom parameter space '{name}' with {len(parameter_space)} parameters"
        )
        return parameter_space
