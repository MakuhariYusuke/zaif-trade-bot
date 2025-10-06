#!/usr/bin/env python3
"""
Ensemble Trading System for Zaif Trade Bot.

Combines multiple trained models for improved prediction accuracy and risk management.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, TypedDict, cast


# Type definitions for ensemble configuration
class ModelConfig(TypedDict, total=False):
    """Configuration for a single model in the ensemble."""

    path: str
    weight: float
    feature_set: str


import numpy as np
from numpy.typing import NDArray
from stable_baselines3 import PPO

from ztb.training.ppo_trainer import PredictorProtocol, TradingSystemProtocol

logger = logging.getLogger(__name__)


class EnsemblePredictor(PredictorProtocol):
    """Ensemble predictor combining multiple trained models."""

    def __init__(self, model_configs: List[ModelConfig]):
        """
        Initialize ensemble predictor.

        Args:
            model_configs: List of model configurations with paths and weights
        """
        self.model_configs = model_configs
        self.models = []
        self.weights = []
        self.feature_sets = []

        for config in model_configs:
            model_path = config.get("path")
            weight = config.get("weight", 1.0)
            feature_set = config.get("feature_set", "full")

            try:
                model = PPO.load(model_path)  # type: ignore[arg-type]
                self.models.append(model)
                self.weights.append(weight)
                self.feature_sets.append(feature_set)
                logger.info(
                    f"Loaded model: {model_path} (weight: {weight}, feature_set: {feature_set})"
                )
            except Exception as e:
                logger.error(f"Failed to load model {model_path}: {e}")
                continue

        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

        logger.info(f"Ensemble initialized with {len(self.models)} models")

    def predict(
        self, observation: NDArray[np.float32], deterministic: bool = True
    ) -> Tuple[Any, Optional[Any]]:
        """
        Make ensemble prediction.

        Args:
            observation: Input observation
            deterministic: Whether to use deterministic prediction

        Returns:
            Tuple of (action, state) where action is ensemble prediction
        """
        if not self.models:
            raise ValueError("No models loaded in ensemble")

        # Get predictions from all models
        actions = []
        states = []

        for model in self.models:
            try:
                action, state = model.predict(observation, deterministic=deterministic)
                actions.append(action)
                states.append(state)
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
                continue

        if not actions:
            raise ValueError("All model predictions failed")

        # Ensemble voting (weighted average for continuous actions)
        if actions[0].dtype in [np.float32, np.float64]:
            # Continuous actions - weighted average
            ensemble_action = np.average(
                actions, weights=self.weights[: len(actions)], axis=0
            )
        else:
            # Discrete actions - weighted voting
            action_counts: Dict[int, float] = {}
            for action, weight in zip(actions, self.weights[: len(actions)]):
                action_val = (
                    int(action[0]) if hasattr(action, "__len__") else int(action)
                )
                action_counts[action_val] = action_counts.get(action_val, 0) + weight

            # Select action with highest weighted vote
            ensemble_action = np.array(
                [max(action_counts, key=lambda k: action_counts.get(k, 0))]
            )

        # Use state from first successful model
        ensemble_state = states[0] if states else None

        return ensemble_action, ensemble_state

    def get_action_probabilities(
        self, observation: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Get action probabilities from ensemble with confidence weights.

        Args:
            observation: Input observation

        Returns:
            Tuple of (ensemble_probabilities, confidence_weights)
        """
        if not hasattr(self.models[0], "policy"):
            raise ValueError("Models must have policy for probability extraction")

        probabilities: List[NDArray[np.float32]] = []
        confidences: List[float] = []

        for model in self.models:
            try:
                # Get action probabilities from policy
                obs_tensor = model.policy.obs_to_tensor(observation)[0]
                _, _, _ = model.policy(obs_tensor)
                distribution = model.policy.get_distribution(obs_tensor)
                if distribution is None or distribution.distribution is None:
                    continue
                probs = distribution.distribution.probs
                prob_array = probs.detach().cpu().numpy()

                probabilities.append(prob_array)

                # Calculate confidence as entropy (lower entropy = higher confidence)
                entropy = -np.sum(prob_array * np.log(prob_array + 1e-10))
                confidence = 1.0 / (1.0 + entropy)  # Normalize to [0, 1]
                confidences.append(confidence)

            except Exception as e:
                logger.warning(f"Failed to get probabilities from model: {e}")
                continue

        if not probabilities:
            raise ValueError("Could not get probabilities from any model")

        # Convert confidences to weights (normalize)
        confidence_array = np.array(confidences)
        weights = confidence_array / np.sum(confidence_array)

        # Weighted average of probabilities
        ensemble_probabilities: NDArray[np.float32] = np.average(
            probabilities, weights=weights, axis=0
        )

        return ensemble_probabilities, weights

    def get_ensemble_info(self) -> Dict[str, Any]:
        """
        Get information about the ensemble.

        アンサンブルに関する情報を取得。
        """
        return {
            "num_models": len(self.models),
            "model_paths": (
                [config.get("path") for config in self.model_configs]
                if hasattr(self, "model_configs")
                else []
            ),
            "weights": self.weights,
            "feature_sets": self.feature_sets,
        }


class EnsembleTradingSystem(TradingSystemProtocol):
    """Complete ensemble trading system with risk management."""

    def __init__(
        self,
        model_configs: List[ModelConfig],
        risk_configs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ensemble trading system.

        Args:
            model_configs: Model configurations for ensemble
            risk_configs: Risk management configurations
        """
        self.ensemble = EnsemblePredictor(model_configs)
        self.risk_configs = risk_configs or self._get_default_risk_configs()

        # Risk management state
        self.consecutive_losses = 0
        self.daily_loss = 0.0
        self.daily_start_balance = 0.0
        self.last_reset_date = None

        # Circuit breaker state
        self.circuit_breaker_triggered = False
        self.price_history: List[float] = []

        logger.info("Ensemble trading system initialized")

    def _get_default_risk_configs(self) -> Dict[str, Any]:
        """Get default risk management configurations."""
        return {
            "max_consecutive_losses": 5,
            "daily_loss_limit": 0.02,  # 2%
            "circuit_breaker_threshold": 0.20,  # ±20%
            "max_position_size": 1.0,
            "min_order_size": 0.001,
            "max_order_size": 1.0,
        }

    def check_risk_limits(self, current_balance: float, current_price: float) -> bool:
        """
        Check if current conditions meet risk management criteria.

        Args:
            current_balance: Current portfolio balance
            current_price: Current market price

        Returns:
            True if trading is allowed, False otherwise
        """
        # Check circuit breaker
        if self._check_circuit_breaker(current_price):
            logger.warning("Circuit breaker triggered - stopping trading")
            return False

        # Check daily loss limit
        if self._check_daily_loss_limit(current_balance):
            logger.warning("Daily loss limit exceeded - stopping trading")
            return False

        # Check consecutive losses
        if self.consecutive_losses >= self.risk_configs["max_consecutive_losses"]:
            logger.warning(
                f"Consecutive losses limit reached ({self.consecutive_losses}) - stopping trading"
            )
            return False

        return True

    def _check_circuit_breaker(self, current_price: float) -> bool:
        """Check if circuit breaker should be triggered."""
        if len(self.price_history) < 2:
            self.price_history.append(current_price)
            return False

        # Calculate price change
        prev_price = self.price_history[-1]
        price_change = abs(current_price - prev_price) / prev_price

        self.price_history.append(current_price)
        if len(self.price_history) > 10:  # Keep last 10 prices
            self.price_history.pop(0)

        if price_change > self.risk_configs["circuit_breaker_threshold"]:
            self.circuit_breaker_triggered = True
            return True

        return False

    def _check_daily_loss_limit(self, current_balance: float) -> bool:
        """Check if daily loss limit is exceeded."""
        # Reset daily tracking if date changed (simplified - in real system use proper date tracking)
        if self.daily_start_balance == 0:
            self.daily_start_balance = current_balance
            return False

        loss_pct = (
            self.daily_start_balance - current_balance
        ) / self.daily_start_balance

        if loss_pct > self.risk_configs["daily_loss_limit"]:
            return True

        return False

    def update_risk_state(self, pnl: float) -> None:
        """
        Update risk management state after trade.

        Args:
            pnl: Profit/Loss from the trade
        """
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def trade(self, observation: Any) -> Any:
        """
        Execute a trade based on observation.

        Args:
            observation: Current market observation

        Returns:
            Trade action and metadata
        """
        # For now, return a simple action based on observation
        # In a real implementation, this would integrate with actual trading execution
        # Extract basic market data from observation if available
        current_price = 100.0  # Default fallback
        current_balance = 1000.0  # Default fallback

        if hasattr(observation, "shape") and len(observation.shape) > 0:
            # Assume observation contains price data
            current_price = (
                float(observation[0]) if observation.shape[0] > 0 else current_price
            )

        action = self.predict_action(observation, current_balance, current_price)

        return {
            "action": action,
            "confidence": self.get_ensemble_confidence(observation),
            "risk_check_passed": self.check_risk_limits(current_balance, current_price),
        }

    def predict_action(
        self,
        observation: NDArray[np.float32],
        current_balance: float,
        current_price: float,
    ) -> int:
        """
        Make ensemble prediction with risk management.

        Args:
            observation: Current market observation
            current_balance: Current portfolio balance
            current_price: Current market price

        Returns:
            Action to take (0=Hold, 1=Buy, 2=Sell)
        """
        # Check risk limits first
        if not self.check_risk_limits(current_balance, current_price):
            return 0  # Hold if risk limits exceeded

        # Get ensemble prediction
        action, _ = self.ensemble.predict(observation, deterministic=True)
        return int(action[0])

    def get_ensemble_confidence(self, observation: NDArray[np.float32]) -> float:
        """
        Get confidence score for ensemble prediction.

        Args:
            observation: Current market observation

        Returns:
            Confidence score (0-1)
        """
        try:
            probabilities = self.ensemble.get_action_probabilities(observation)
            # Return max probability as confidence
            return float(np.max(probabilities))
        except:
            # Fallback to model count based confidence
            return min(
                1.0, len(self.ensemble.models) / 5.0
            )  # Max confidence at 5 models


class EnsemblePredictorLegacy(PredictorProtocol):
    """Legacy ensemble predictor combining multiple trained models."""

    def __init__(self, model_configs: List[Dict[str, Any]]):
        """
        Initialize ensemble predictor.

        Args:
            model_configs: List of model configurations with paths and weights
        """
        self.model_configs = model_configs
        self.models = []
        self.weights = []
        self.feature_sets = []

        for config in model_configs:
            model_path = config.get("path")
            weight = config.get("weight", 1.0)
            feature_set = config.get("feature_set", "full")

            try:
                model = PPO.load(model_path)  # type: ignore[arg-type]
                self.models.append(model)
                self.weights.append(weight)
                self.feature_sets.append(feature_set)
                logger.info(
                    f"Loaded model: {model_path} (weight: {weight}, feature_set: {feature_set})"
                )
            except Exception as e:
                logger.error(f"Failed to load model {model_path}: {e}")
                continue

        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

        logger.info(f"Ensemble initialized with {len(self.models)} models")

    def predict(  # type: ignore[override]
        self, observation: NDArray[np.float32], deterministic: bool = True
    ) -> Tuple[NDArray[np.float32], Optional[np.ndarray[Any, Any]]]:
        """
        Make ensemble prediction.

        Args:
            observation: Input observation
            deterministic: Whether to use deterministic prediction

        Returns:
            Tuple of (action, state) where action is ensemble prediction
        """
        if not self.models:
            raise ValueError("No models loaded in ensemble")

        # Get predictions from all models
        actions = []
        states = []

        for model in self.models:
            try:
                action, state = model.predict(observation, deterministic=deterministic)
                actions.append(action)
                states.append(state)
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
                continue

        if not actions:
            raise ValueError("All model predictions failed")

        # Ensemble voting (weighted average for continuous actions)
        if actions[0].dtype in [np.float32, np.float64]:
            # Continuous actions - weighted average
            ensemble_action = np.average(
                actions, weights=self.weights[: len(actions)], axis=0
            )
        else:
            # Discrete actions - weighted voting
            action_counts: Dict[int, float] = {}
            for action, weight in zip(actions, self.weights[: len(actions)]):
                action_val = (
                    int(action[0]) if hasattr(action, "__len__") else int(action)
                )
                action_counts[action_val] = action_counts.get(action_val, 0) + weight

            # Select action with highest weighted vote
            ensemble_action = np.array(
                [max(action_counts, key=lambda k: action_counts[k])]
            )

        # Use state from first successful model
        ensemble_state = states[0] if states else None

        return ensemble_action, ensemble_state  # type: ignore[return-value]

    def get_action_probabilities(
        self, observation: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Get action probabilities from ensemble with confidence weights.

        Args:
            observation: Input observation

        Returns:
            Tuple of (ensemble_probabilities, confidence_weights)
        """
        if not hasattr(self.models[0], "policy"):
            raise ValueError("Models must have policy for probability extraction")

        probabilities: List[NDArray[np.float32]] = []
        confidences = []

        for model in self.models:
            try:
                # Get action probabilities from policy
                obs_tensor = model.policy.obs_to_tensor(observation)[0]
                _, _, _ = model.policy(obs_tensor)
                probs = model.policy.get_distribution(obs_tensor).distribution.probs  # type: ignore[union-attr]
                prob_array = probs.detach().cpu().numpy()

                probabilities.append(prob_array)

                # Calculate confidence as entropy (lower entropy = higher confidence)
                entropy = -np.sum(prob_array * np.log(prob_array + 1e-10))
                confidence = 1.0 / (1.0 + entropy)  # Normalize to [0, 1]
                confidences.append(confidence)

            except Exception as e:
                logger.warning(f"Failed to get probabilities from model: {e}")
                continue

        if not probabilities:
            raise ValueError("Could not get probabilities from any model")

        # Convert confidences to weights (normalize)
        confidence_array: NDArray[np.float32] = np.array(confidences)
        weights = confidence_array / np.sum(confidence_array)

        # Weighted average of probabilities
        ensemble_probabilities: NDArray[np.float32] = np.average(
            probabilities, weights=weights, axis=0
        )

        return ensemble_probabilities, weights

    def get_ensemble_info(self) -> Dict[str, Any]:
        """
        Get information about the ensemble.

        アンサンブルに関する情報を取得。
        """
        return {
            "num_models": len(self.models),
            "model_paths": (
                [config.get("path") for config in self.model_configs]
                if hasattr(self, "model_configs")
                else []
            ),
            "weights": self.weights,
            "feature_sets": self.feature_sets,
        }


class EnsembleTradingSystemLegacy:
    """Legacy complete ensemble trading system with risk management."""

    def __init__(
        self,
        model_configs: List[Dict[str, Any]],
        risk_configs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ensemble trading system.

        Args:
            model_configs: Model configurations for ensemble
            risk_configs: Risk management configurations
        """
        self.ensemble = EnsemblePredictorLegacy(model_configs)
        self.risk_configs = risk_configs or self._get_default_risk_configs()

        # Risk management state
        self.consecutive_losses = 0
        self.daily_loss = 0.0
        self.daily_start_balance = 0.0
        self.last_reset_date = None

        # Circuit breaker state
        self.circuit_breaker_triggered = False
        self.price_history: List[float] = []

        logger.info("Ensemble trading system initialized")

    def _get_default_risk_configs(self) -> Dict[str, Any]:
        """Get default risk management configurations."""
        return {
            "max_consecutive_losses": 5,
            "daily_loss_limit": 0.02,  # 2%
            "circuit_breaker_threshold": 0.20,  # ±20%
            "max_position_size": 1.0,
            "min_order_size": 0.001,
            "max_order_size": 1.0,
        }

    def check_risk_limits(self, current_balance: float, current_price: float) -> bool:
        """
        Check if current conditions meet risk management criteria.

        Args:
            current_balance: Current portfolio balance
            current_price: Current market price

        Returns:
            True if trading is allowed, False otherwise
        """
        # Check circuit breaker
        if self._check_circuit_breaker(current_price):
            logger.warning("Circuit breaker triggered - stopping trading")
            return False

        # Check daily loss limit
        if self._check_daily_loss_limit(current_balance):
            logger.warning("Daily loss limit exceeded - stopping trading")
            return False

        # Check consecutive losses
        if self.consecutive_losses >= self.risk_configs["max_consecutive_losses"]:
            logger.warning(
                f"Consecutive losses limit reached ({self.consecutive_losses}) - stopping trading"
            )
            return False

        return True

    def _check_circuit_breaker(self, current_price: float) -> bool:
        """Check if circuit breaker should be triggered."""
        if len(self.price_history) < 2:
            self.price_history.append(current_price)
            return False

        # Calculate price change
        prev_price = self.price_history[-1]
        price_change = abs(current_price - prev_price) / prev_price

        self.price_history.append(current_price)
        if len(self.price_history) > 10:  # Keep last 10 prices
            self.price_history.pop(0)

        if price_change > self.risk_configs["circuit_breaker_threshold"]:
            self.circuit_breaker_triggered = True
            return True

        return False

    def _check_daily_loss_limit(self, current_balance: float) -> bool:
        """Check if daily loss limit is exceeded."""
        # Reset daily tracking if date changed (simplified - in real system use proper date tracking)
        if self.daily_start_balance == 0:
            self.daily_start_balance = current_balance
            return False

        loss_pct = (
            self.daily_start_balance - current_balance
        ) / self.daily_start_balance

        if loss_pct > self.risk_configs["daily_loss_limit"]:
            return True

        return False

    def update_risk_state(self, pnl: float) -> None:
        """
        Update risk management state after trade.

        Args:
            pnl: Profit/Loss from the trade
        """
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def predict_action(
        self,
        observation: NDArray[np.float32],
        current_balance: float,
        current_price: float,
    ) -> int:
        """
        Make ensemble prediction with risk management.

        Args:
            observation: Current market observation
            current_balance: Current portfolio balance
            current_price: Current market price

        Returns:
            Action to take (0=Hold, 1=Buy, 2=Sell)
        """
        # Check risk limits first
        if not self.check_risk_limits(current_balance, current_price):
            return 0  # Hold if risk limits exceeded

        # Get ensemble prediction
        action, _ = self.ensemble.predict(observation, deterministic=True)
        return int(action[0])

    def get_ensemble_confidence(self, observation: NDArray[np.float32]) -> float:
        """
        Get confidence score for ensemble prediction.

        Args:
            observation: Current market observation

        Returns:
            Confidence score (0-1)
        """
        try:
            probabilities = self.ensemble.get_action_probabilities(observation)
            # Return max probability as confidence
            return float(np.max(probabilities))
        except:
            # Fallback to model count based confidence
            return min(
                1.0, len(self.ensemble.models) / 5.0
            )  # Max confidence at 5 models


def create_default_ensemble() -> EnsembleTradingSystem:
    """Create default ensemble with available models."""
    model_configs: List[ModelConfig] = [
        {
            "path": "models/trading_optimized_reward_v2_final.zip",
            "weight": 1.0,
            "feature_set": "full",
        },
        # Add more models as they become available
    ]

    return EnsembleTradingSystem(model_configs)


def create_default_ensemble_legacy() -> EnsembleTradingSystemLegacy:
    """Create default legacy ensemble with available models."""
    model_configs: List[ModelConfig] = [
        {
            "path": "models/trading_optimized_reward_v2_final.zip",
            "weight": 1.0,
            "feature_set": "full",
        },
        # Add more models as they become available
    ]

    return EnsembleTradingSystemLegacy(cast(List[Dict[str, Any]], model_configs))


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create ensemble system
    ensemble_system = create_default_ensemble()

    # Example prediction (would need actual observation data)
    print("Ensemble trading system created successfully")
    print(f"Loaded {len(ensemble_system.ensemble.models)} models")
