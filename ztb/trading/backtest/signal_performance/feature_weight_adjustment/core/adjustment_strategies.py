"""
Adjustment Strategies

Base classes and concrete implementations of weight adjustment strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta

from ztb.utils.logging_utils import get_logger

from ..interfaces.adjustment_interface import WeightAdjustmentInterface
from ..utils.data_processor import DataProcessor
from ..utils.validation_utils import ValidationUtils

logger = get_logger(__name__)


class BaseAdjustmentStrategy(WeightAdjustmentInterface):
    """
    Base class for weight adjustment strategies.

    Provides common functionality and validation for all adjustment strategies.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base adjustment strategy.

        Args:
            config: Strategy-specific configuration
        """
        self.config = config or {}
        self.data_processor = DataProcessor()
        self.adjustment_count = 0

    @abstractmethod
    def adjust_weights(
        self,
        current_weights: Dict[str, float],
        performance_data: Dict[str, Any],
        feature_importance: Dict[str, float],
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Adjust feature weights based on performance data.

        Args:
            current_weights: Current feature weights
            performance_data: Performance metrics and analysis
            feature_importance: Feature importance scores
            market_conditions: Current market conditions (optional)

        Returns:
            Adjusted feature weights
        """
        pass

    def get_adjustment_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the adjustment algorithm.

        Returns:
            Dictionary containing algorithm metadata
        """
        return {
            "strategy_name": self.__class__.__name__,
            "adjustment_count": self.adjustment_count,
            "config": self.config.copy(),
        }

    def validate_weights(self, weights: Dict[str, float]) -> bool:
        """
        Validate that weights are properly normalized and valid.

        Args:
            weights: Feature weights to validate

        Returns:
            True if weights are valid, False otherwise
        """
        if not weights:
            return False

        # Check that all weights are finite numbers
        for feature, weight in weights.items():
            if not isinstance(weight, (int, float)) or not np.isfinite(weight):
                logger.error(f"Invalid weight for feature {feature}: {weight}")
                return False

        return True

    def reset(self) -> None:
        """
        Reset the adjustment algorithm to initial state.
        """
        self.adjustment_count = 0
        logger.info(f"{self.__class__.__name__} reset to initial state")

    def _calculate_weight_changes(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        max_change_rate: float = 0.1
    ) -> Dict[str, float]:
        """
        Calculate gradual weight changes with maximum rate limiting.

        Args:
            current_weights: Current weights
            target_weights: Target weights
            max_change_rate: Maximum change rate per adjustment

        Returns:
            Adjusted weights with rate limiting
        """
        adjusted_weights = {}

        for feature in current_weights:
            if feature in target_weights:
                current = current_weights[feature]
                target = target_weights[feature]

                # Calculate maximum allowed change
                max_change = abs(current) * max_change_rate if current != 0 else max_change_rate

                # Apply rate limiting
                if abs(target - current) > max_change:
                    if target > current:
                        new_weight = current + max_change
                    else:
                        new_weight = current - max_change
                else:
                    new_weight = target

                adjusted_weights[feature] = new_weight
            else:
                # Keep current weight if not in target
                adjusted_weights[feature] = current_weights[feature]

        return adjusted_weights

    def _apply_momentum(
        self,
        new_weights: Dict[str, float],
        previous_changes: Optional[Dict[str, float]] = None,
        momentum_rate: float = 0.9
    ) -> Dict[str, float]:
        """
        Apply momentum to weight changes.

        Args:
            new_weights: Newly calculated weights
            previous_changes: Previous weight changes for momentum
            momentum_rate: Momentum coefficient

        Returns:
            Weights with momentum applied
        """
        if not previous_changes:
            return new_weights

        momentum_weights = {}
        for feature, weight in new_weights.items():
            if feature in previous_changes:
                momentum = previous_changes[feature] * momentum_rate
                momentum_weights[feature] = weight + momentum
            else:
                momentum_weights[feature] = weight

        return momentum_weights


class PerformanceDrivenStrategy(BaseAdjustmentStrategy):
    """
    Performance-driven weight adjustment strategy.

    Adjusts weights based on feature performance metrics like
    win rate, return contribution, and risk-adjusted returns.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize performance-driven strategy.

        Args:
            config: Strategy configuration
        """
        super().__init__(config)

        # Default configuration
        self.config.setdefault("performance_window", 100)
        self.config.setdefault("min_weight", 0.01)
        self.config.setdefault("max_weight", 1.0)
        self.config.setdefault("adjustment_rate", 0.05)
        self.config.setdefault("performance_weights", {
            "win_rate": 0.4,
            "return_contribution": 0.3,
            "sharpe_ratio": 0.2,
            "max_drawdown": 0.1,
        })

        self.data_processor = DataProcessor()
        self.validator = ValidationUtils()

        # Performance history
        self.performance_history: List[Dict[str, Any]] = []

    def adjust_weights(
        self,
        current_weights: Dict[str, float],
        performance_data: Dict[str, Any],
        feature_importance: Dict[str, float],
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Adjust weights based on performance metrics.

        Args:
            current_weights: Current feature weights
            performance_data: Performance metrics and analysis
            feature_importance: Feature importance scores
            market_conditions: Current market conditions

        Returns:
            Adjusted feature weights
        """
        try:
            # Validate inputs
            if not self.validator.validate_weight_dict(current_weights):
                logger.error("Invalid current weights")
                return current_weights

            if not self.validator.validate_performance_data(performance_data):
                logger.error("Invalid performance data")
                return current_weights

            # Store performance data
            self.performance_history.append({
                "timestamp": datetime.now(),
                "performance_data": performance_data.copy(),
                "weights": current_weights.copy(),
            })

            # Maintain history size
            if len(self.performance_history) > self.config["performance_window"]:
                self.performance_history = self.performance_history[-self.config["performance_window"]:]

            # Calculate performance scores for each feature
            feature_scores = self._calculate_feature_performance_scores(
                performance_data, feature_importance
            )

            # Calculate target weights based on performance
            target_weights = self._calculate_target_weights(
                current_weights, feature_scores, market_conditions
            )

            # Apply gradual adjustment with rate limiting
            adjusted_weights = self._calculate_weight_changes(
                current_weights, target_weights, self.config["adjustment_rate"]
            )

            # Normalize weights
            adjusted_weights = self._normalize_weights(adjusted_weights)

            self.adjustment_count += 1

            logger.info(f"Performance-driven adjustment completed. Features adjusted: {len(adjusted_weights)}")
            return adjusted_weights

        except Exception as e:
            logger.error(f"Failed to adjust weights: {e}")
            return current_weights

    def _calculate_feature_performance_scores(
        self,
        performance_data: Dict[str, Any],
        feature_importance: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate performance scores for each feature.

        Args:
            performance_data: Performance metrics
            feature_importance: Feature importance scores

        Returns:
            Dictionary of feature performance scores
        """
        scores = {}

        # Extract feature-specific performance data
        feature_performance = performance_data.get("feature_performance", {})

        for feature in feature_importance.keys():
            perf_data = feature_performance.get(feature, {})

            # Calculate weighted performance score
            win_rate = perf_data.get("win_rate", 0.5)
            return_contribution = perf_data.get("return_contribution", 0.0)
            sharpe_ratio = perf_data.get("sharpe_ratio", 0.0)
            max_drawdown = perf_data.get("max_drawdown", 0.0)

            # Normalize max_drawdown (lower is better)
            normalized_drawdown = 1.0 / (1.0 + abs(max_drawdown)) if max_drawdown != 0 else 1.0

            # Calculate composite score
            score = (
                self.config["performance_weights"]["win_rate"] * win_rate +
                self.config["performance_weights"]["return_contribution"] * max(0, return_contribution) +
                self.config["performance_weights"]["sharpe_ratio"] * max(0, sharpe_ratio) +
                self.config["performance_weights"]["max_drawdown"] * normalized_drawdown
            )

            # Weight by feature importance
            importance = feature_importance.get(feature, 0.0)
            scores[feature] = score * (0.5 + 0.5 * importance)  # Importance weighting

        return scores

    def _calculate_target_weights(
        self,
        current_weights: Dict[str, float],
        feature_scores: Dict[str, float],
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calculate target weights based on performance scores.

        Args:
            current_weights: Current weights
            feature_scores: Performance scores for features
            market_conditions: Market conditions

        Returns:
            Target weights
        """
        target_weights = {}

        # Apply market condition adjustments
        market_multiplier = self._get_market_condition_multiplier(market_conditions)

        for feature, score in feature_scores.items():
            current_weight = current_weights.get(feature, 0.0)

            # Base adjustment based on performance score
            if score > 0.6:  # Good performance
                adjustment = self.config["adjustment_rate"] * 1.5
            elif score < 0.4:  # Poor performance
                adjustment = -self.config["adjustment_rate"] * 1.2
            else:  # Neutral performance
                adjustment = 0.0

            # Apply market conditions
            adjustment *= market_multiplier

            # Calculate target weight
            target_weight = current_weight + adjustment

            # Apply bounds
            target_weight = max(self.config["min_weight"], min(self.config["max_weight"], target_weight))

            target_weights[feature] = target_weight

        return target_weights

    def _get_market_condition_multiplier(self, market_conditions: Optional[Dict[str, Any]]) -> float:
        """
        Get market condition adjustment multiplier.

        Args:
            market_conditions: Current market conditions

        Returns:
            Adjustment multiplier
        """
        if not market_conditions:
            return 1.0

        volatility = market_conditions.get("volatility", 0.5)
        trend_strength = market_conditions.get("trend_strength", 0.5)

        # Reduce adjustments in high volatility
        volatility_multiplier = 1.0 - (volatility - 0.5) * 0.5

        # Increase adjustments in strong trends
        trend_multiplier = 0.8 + trend_strength * 0.4

        return volatility_multiplier * trend_multiplier

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to ensure they sum to 1.0.

        Args:
            weights: Weights to normalize

        Returns:
            Normalized weights
        """
        total_weight = sum(weights.values())

        if total_weight == 0:
            # Equal weights if all are zero
            normalized = {feature: 1.0 / len(weights) for feature in weights}
        else:
            # Normalize to sum to 1.0
            normalized = {feature: weight / total_weight for feature, weight in weights.items()}

        return normalized


class CorrelationBasedStrategy(BaseAdjustmentStrategy):
    """
    Correlation-based weight adjustment strategy.

    Adjusts weights based on feature correlations and redundancy.
    Reduces weights for highly correlated features to avoid overfitting.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize correlation-based strategy.

        Args:
            config: Strategy configuration
        """
        super().__init__(config)

        # Default configuration
        self.config.setdefault("correlation_threshold", 0.8)
        self.config.setdefault("redundancy_penalty", 0.1)
        self.config.setdefault("diversification_bonus", 0.05)
        self.config.setdefault("adjustment_rate", 0.03)

        self.data_processor = DataProcessor()
        self.validator = ValidationUtils()

        # Correlation history
        self.correlation_matrix_history: List[np.ndarray] = []

    def adjust_weights(
        self,
        current_weights: Dict[str, float],
        performance_data: Dict[str, Any],
        feature_importance: Dict[str, float],
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Adjust weights based on feature correlations.

        Args:
            current_weights: Current feature weights
            performance_data: Performance metrics and analysis
            feature_importance: Feature importance scores
            market_conditions: Current market conditions

        Returns:
            Adjusted feature weights
        """
        try:
            # Validate inputs
            if not self.validator.validate_weight_dict(current_weights):
                logger.error("Invalid current weights")
                return current_weights

            # Get feature correlation matrix
            correlation_matrix = performance_data.get("feature_correlations")
            if correlation_matrix is None:
                logger.warning("No correlation data available, skipping adjustment")
                return current_weights

            # Store correlation history
            if isinstance(correlation_matrix, np.ndarray):
                self.correlation_matrix_history.append(correlation_matrix.copy())
                if len(self.correlation_matrix_history) > 10:  # Keep last 10
                    self.correlation_matrix_history = self.correlation_matrix_history[-10:]

            # Calculate correlation-based adjustments
            adjustments = self._calculate_correlation_adjustments(
                current_weights, correlation_matrix, feature_importance
            )

            # Apply adjustments
            adjusted_weights = {}
            for feature, current_weight in current_weights.items():
                adjustment = adjustments.get(feature, 0.0)
                new_weight = current_weight + adjustment

                # Ensure minimum weight
                new_weight = max(0.01, new_weight)

                adjusted_weights[feature] = new_weight

            # Normalize weights
            adjusted_weights = self._normalize_weights(adjusted_weights)

            self.adjustment_count += 1

            logger.info(f"Correlation-based adjustment completed. Features adjusted: {len(adjusted_weights)}")
            return adjusted_weights

        except Exception as e:
            logger.error(f"Failed to adjust weights: {e}")
            return current_weights

    def _calculate_correlation_adjustments(
        self,
        current_weights: Dict[str, float],
        correlation_matrix: np.ndarray,
        feature_importance: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate weight adjustments based on correlations.

        Args:
            current_weights: Current weights
            correlation_matrix: Feature correlation matrix
            feature_importance: Feature importance scores

        Returns:
            Weight adjustments for each feature
        """
        adjustments = {}
        features = list(current_weights.keys())

        # Ensure correlation matrix matches features
        if correlation_matrix.shape[0] != len(features):
            logger.error("Correlation matrix size doesn't match feature count")
            return adjustments

        for i, feature in enumerate(features):
            if i >= correlation_matrix.shape[0]:
                continue

            # Find highly correlated features
            correlated_features = []
            for j, other_feature in enumerate(features):
                if i != j and abs(correlation_matrix[i, j]) > self.config["correlation_threshold"]:
                    correlated_features.append((other_feature, correlation_matrix[i, j]))

            if not correlated_features:
                # No highly correlated features - small diversification bonus
                adjustments[feature] = self.config["diversification_bonus"] * current_weights[feature]
                continue

            # Calculate redundancy penalty
            total_correlation = sum(abs(corr) for _, corr in correlated_features)
            redundancy_penalty = self.config["redundancy_penalty"] * total_correlation

            # Weight penalty by relative importance
            feature_imp = feature_importance.get(feature, 0.0)
            avg_correlated_imp = np.mean([
                feature_importance.get(other, 0.0) for other, _ in correlated_features
            ])

            if feature_imp < avg_correlated_imp:
                # This feature is less important than correlated ones - reduce weight
                penalty_multiplier = 1.5
            else:
                # This feature is more important - reduce penalty
                penalty_multiplier = 0.5

            adjustments[feature] = -redundancy_penalty * penalty_multiplier * current_weights[feature]

        return adjustments

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to ensure they sum to 1.0.

        Args:
            weights: Weights to normalize

        Returns:
            Normalized weights
        """
        total_weight = sum(weights.values())

        if total_weight == 0:
            # Equal weights if all are zero
            normalized = {feature: 1.0 / len(weights) for feature in weights}
        else:
            # Normalize to sum to 1.0
            normalized = {feature: weight / total_weight for feature, weight in weights.items()}

        return normalized


class ReinforcementLearningStrategy(BaseAdjustmentStrategy):
    """
    Reinforcement learning-based weight adjustment strategy.

    Uses reinforcement learning principles to optimize feature weights
    based on reward signals from trading performance.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize reinforcement learning strategy.

        Args:
            config: Strategy configuration
        """
        super().__init__(config)

        # Default configuration
        self.config.setdefault("learning_rate", 0.01)
        self.config.setdefault("discount_factor", 0.95)
        self.config.setdefault("exploration_rate", 0.1)
        self.config.setdefault("reward_window", 50)
        self.config.setdefault("min_weight", 0.01)
        self.config.setdefault("max_weight", 1.0)

        self.data_processor = DataProcessor()
        self.validator = ValidationUtils()

        # RL state
        self.q_table: Dict[str, Dict[str, float]] = {}  # State-action values
        self.state_history: List[str] = []
        self.reward_history: List[float] = []
        self.action_history: List[Dict[str, float]] = []

        # Initialize random state
        np.random.seed(42)

    def adjust_weights(
        self,
        current_weights: Dict[str, float],
        performance_data: Dict[str, Any],
        feature_importance: Dict[str, float],
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Adjust weights using reinforcement learning.

        Args:
            current_weights: Current feature weights
            performance_data: Performance metrics and analysis
            feature_importance: Feature importance scores
            market_conditions: Current market conditions

        Returns:
            Adjusted feature weights
        """
        try:
            # Validate inputs
            if not self.validator.validate_weight_dict(current_weights):
                logger.error("Invalid current weights")
                return current_weights

            # Create state representation
            state = self._create_state_representation(
                current_weights, performance_data, market_conditions
            )

            # Calculate reward from performance
            reward = self._calculate_reward(performance_data)

            # Store experience
            self.state_history.append(state)
            self.reward_history.append(reward)
            self.action_history.append(current_weights.copy())

            # Maintain history size
            max_history = self.config["reward_window"] * 2
            if len(self.state_history) > max_history:
                self.state_history = self.state_history[-max_history:]
                self.reward_history = self.reward_history[-max_history:]
                self.action_history = self.action_history[-max_history:]

            # Update Q-table
            if len(self.state_history) >= 2:
                self._update_q_table()

            # Select action (weight adjustments)
            action = self._select_action(state, current_weights)

            # Apply action to get new weights
            new_weights = self._apply_action(current_weights, action)

            # Normalize and bound weights
            new_weights = self._normalize_and_bound_weights(new_weights)

            self.adjustment_count += 1

            logger.info(f"RL-based adjustment completed. Features adjusted: {len(new_weights)}")
            return new_weights

        except Exception as e:
            logger.error(f"Failed to adjust weights: {e}")
            return current_weights

    def _create_state_representation(
        self,
        weights: Dict[str, float],
        performance_data: Dict[str, Any],
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a discrete state representation for RL.

        Args:
            weights: Current weights
            performance_data: Performance data
            market_conditions: Market conditions

        Returns:
            State string representation
        """
        # Discretize weights into bins
        weight_bins = []
        for feature, weight in weights.items():
            if weight < 0.2:
                weight_bins.append("low")
            elif weight < 0.5:
                weight_bins.append("medium")
            else:
                weight_bins.append("high")

        # Performance state
        win_rate = performance_data.get("win_rate", 0.5)
        if win_rate < 0.4:
            perf_state = "poor"
        elif win_rate < 0.6:
            perf_state = "moderate"
        else:
            perf_state = "good"

        # Market conditions
        market_state = "neutral"
        if market_conditions:
            volatility = market_conditions.get("volatility", 0.5)
            if volatility > 0.7:
                market_state = "volatile"
            elif volatility < 0.3:
                market_state = "stable"

        # Combine into state string
        state = f"{'-'.join(weight_bins)}_{perf_state}_{market_state}"
        return state

    def _calculate_reward(self, performance_data: Dict[str, Any]) -> float:
        """
        Calculate reward from performance data.

        Args:
            performance_data: Performance metrics

        Returns:
            Reward value
        """
        win_rate = performance_data.get("win_rate", 0.5)
        total_return = performance_data.get("total_return", 0.0)
        max_drawdown = performance_data.get("max_drawdown", 0.0)

        # Reward components
        win_rate_reward = (win_rate - 0.5) * 2  # Scale to [-1, 1]
        return_reward = np.tanh(total_return * 10)  # Bound return reward
        drawdown_penalty = -abs(max_drawdown) * 2  # Penalty for drawdown

        # Combine rewards
        reward = (
            0.5 * win_rate_reward +
            0.3 * return_reward +
            0.2 * drawdown_penalty
        )

        return reward

    def _update_q_table(self) -> None:
        """
        Update Q-table using recent experiences.
        """
        if len(self.state_history) < 2 or len(self.reward_history) < 2:
            return

        # Get recent transition
        current_state = self.state_history[-2]
        next_state = self.state_history[-1]
        reward = self.reward_history[-1]
        action = self.action_history[-1]

        # Initialize Q-values if needed
        if current_state not in self.q_table:
            self.q_table[current_state] = {}

        if next_state not in self.q_table:
            self.q_table[next_state] = {}

        # Create action key
        action_key = str(sorted(action.items()))

        # Initialize action Q-value if needed
        if action_key not in self.q_table[current_state]:
            self.q_table[current_state][action_key] = 0.0

        # Get next state max Q-value
        next_max_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0

        # Q-learning update
        current_q = self.q_table[current_state][action_key]
        new_q = current_q + self.config["learning_rate"] * (
            reward + self.config["discount_factor"] * next_max_q - current_q
        )

        self.q_table[current_state][action_key] = new_q

    def _select_action(self, state: str, current_weights: Dict[str, float]) -> Dict[str, str]:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: Current state
            current_weights: Current weights

        Returns:
            Action dictionary (feature -> adjustment_type)
        """
        # Exploration vs exploitation
        if np.random.random() < self.config["exploration_rate"]:
            # Random action
            return self._random_action(current_weights)
        else:
            # Greedy action based on Q-table
            if state in self.q_table and self.q_table[state]:
                # Find best action
                best_action_key = max(self.q_table[state].items(), key=lambda x: x[1])[0]
                try:
                    best_action = eval(best_action_key)  # Convert string back to dict
                    return self._weights_to_action(best_action, current_weights)
                except:
                    pass

            # Fallback to random action
            return self._random_action(current_weights)

    def _random_action(self, current_weights: Dict[str, float]) -> Dict[str, str]:
        """
        Generate a random action.

        Args:
            current_weights: Current weights

        Returns:
            Random action dictionary
        """
        action_types = ["increase", "decrease", "maintain"]
        action = {}

        for feature in current_weights.keys():
            action[feature] = np.random.choice(action_types)

        return action

    def _weights_to_action(
        self,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float]
    ) -> Dict[str, str]:
        """
        Convert target weights to action types.

        Args:
            target_weights: Target weights
            current_weights: Current weights

        Returns:
            Action dictionary
        """
        action = {}

        for feature in current_weights.keys():
            current = current_weights[feature]
            target = target_weights.get(feature, current)

            if target > current + 0.05:
                action[feature] = "increase"
            elif target < current - 0.05:
                action[feature] = "decrease"
            else:
                action[feature] = "maintain"

        return action

    def _apply_action(
        self,
        current_weights: Dict[str, float],
        action: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Apply action to current weights.

        Args:
            current_weights: Current weights
            action: Action to apply

        Returns:
            New weights after action
        """
        new_weights = current_weights.copy()
        adjustment_size = 0.05  # Fixed adjustment size

        for feature, action_type in action.items():
            if action_type == "increase":
                new_weights[feature] = min(1.0, current_weights[feature] + adjustment_size)
            elif action_type == "decrease":
                new_weights[feature] = max(0.01, current_weights[feature] - adjustment_size)
            # maintain: keep current weight

        return new_weights

    def _normalize_and_bound_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize and bound weights.

        Args:
            weights: Weights to process

        Returns:
            Normalized and bounded weights
        """
        # Apply bounds
        bounded = {}
        for feature, weight in weights.items():
            bounded[feature] = max(self.config["min_weight"], min(self.config["max_weight"], weight))

        # Normalize to sum to 1.0
        total = sum(bounded.values())
        if total > 0:
            normalized = {feature: weight / total for feature, weight in bounded.items()}
        else:
            # Fallback to equal weights
            normalized = {feature: 1.0 / len(bounded) for feature in bounded}

        return normalized


class AdjustmentStrategyRegistry:
    """
    Registry for weight adjustment strategies.

    Provides a centralized way to register and retrieve adjustment strategies.
    """

    _strategies = {}

    @classmethod
    def register(cls, name: str, strategy_class: type) -> None:
        """
        Register an adjustment strategy.

        Args:
            name: Name of the strategy
            strategy_class: Strategy class to register
        """
        cls._strategies[name] = strategy_class
        logger.info(f"Registered adjustment strategy: {name}")

    @classmethod
    def get_strategy(cls, name: str, config: Optional[Dict[str, Any]] = None) -> WeightAdjustmentInterface:
        """
        Get an instance of a registered strategy.

        Args:
            name: Name of the strategy
            config: Configuration for the strategy

        Returns:
            Instance of the requested strategy

        Raises:
            ValueError: If strategy is not registered
        """
        if name not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ValueError(f"Unknown strategy '{name}'. Available: {available}")

        strategy_class = cls._strategies[name]
        return strategy_class(config)

    @classmethod
    def list_strategies(cls) -> List[str]:
        """
        List all registered strategies.

        Returns:
            List of strategy names
        """
        return list(cls._strategies.keys())

    @classmethod
    def clear_registry(cls) -> None:
        """
        Clear all registered strategies.
        """
        cls._strategies.clear()
        logger.info("Cleared adjustment strategy registry")


# Register default strategies
AdjustmentStrategyRegistry.register("performance_driven", PerformanceDrivenStrategy)
AdjustmentStrategyRegistry.register("correlation_based", CorrelationBasedStrategy)
AdjustmentStrategyRegistry.register("reinforcement_learning", ReinforcementLearningStrategy)