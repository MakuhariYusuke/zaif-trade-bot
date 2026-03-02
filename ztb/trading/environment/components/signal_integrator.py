"""
Signal Integrator Component.

This component integrates signal-based rewards into the main reward calculation.
Follows Single Responsibility Principle by focusing only on signal integration.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

from ztb.utils.logging_utils import get_logger

from .interfaces import ISignalIntegrator

if TYPE_CHECKING:
    from ztb.trading.strategies.action_signal_guide import (  # noqa: F401
        ActionSignalGuide,
        ActionSignalGuideConfig,
        GuidanceMode,
    )
    from ztb.trading.strategies.signal_reward_integrator import SignalRewardIntegrator  # noqa: F401

class SignalIntegrator(ISignalIntegrator):
    """
    Integrates signal-based rewards into the main reward calculation.

    This class encapsulates signal integration logic including:
    - Action signal guide initialization
    - Signal reward integration
    - Feature name management
    """

    def __init__(
        self,
        config: Any,  # EnvironmentConfig
        enabled: bool = False,
        guidance_level: str = "full",
        signal_bonus_weight: float = 0.1,
        signal_penalty_weight: float = 0.05,
        granville_weight: float = 1.2,
        dow_theory_weight: float = 1.5,
        heikin_ashi_weight: float = 1.0,
        enable_advanced_integration: bool = True,
    ):
        """
        Initialize SignalIntegrator.

        Args:
            config: Environment configuration
            enabled: Whether signal integration is enabled
            guidance_level: Signal guidance level
            signal_bonus_weight: Weight for signal bonuses
            signal_penalty_weight: Weight for signal penalties
            granville_weight: Weight for Granville signals
            dow_theory_weight: Weight for Dow Theory signals
            heikin_ashi_weight: Weight for Heikin-Ashi signals
            enable_advanced_integration: Whether to use advanced integration
        """
        self.config = config
        self.enabled = enabled
        self.logger = get_logger("ztb.trading.environment.signal_integrator")

        self.signal_guide: Any | None = None
        self.signal_integration: Any | None = None
        self._signal_guide_available = False

        if enabled:
            self._initialize_signal_guide(
                guidance_level,
                signal_bonus_weight,
                signal_penalty_weight,
                granville_weight,
                dow_theory_weight,
                heikin_ashi_weight,
                enable_advanced_integration,
            )

    def _initialize_signal_guide(
        self,
        guidance_level: str,
        signal_bonus_weight: float,
        signal_penalty_weight: float,
        granville_weight: float,
        dow_theory_weight: float,
        heikin_ashi_weight: float,
        enable_advanced_integration: bool,
    ):
        """Initialize the action signal guide and integration."""
        try:
            # Import heavy strategy modules only when signal integration is enabled.
            from ztb.trading.strategies.action_signal_guide import (
                ActionSignalGuide,
                ActionSignalGuideConfig,
                GuidanceMode,
            )
            from ztb.trading.strategies.signal_reward_integrator import (
                SignalRewardIntegrator,
            )

            # Initialize signal guide
            feature_names = getattr(self.config, "feature_names", None)

            # Convert string guidance_level to GuidanceMode enum
            guidance_map = {
                "full": GuidanceMode.FULL_GUIDANCE,
                "partial": GuidanceMode.PARTIAL_GUIDANCE,
                "minimal": GuidanceMode.MINIMAL_GUIDANCE,
                "fade_out": GuidanceMode.FADE_OUT,
                "none": GuidanceMode.NO_GUIDANCE,
                "strong": GuidanceMode.FULL_GUIDANCE,  # backward compatibility
            }

            guidance_enum = guidance_map.get(
                guidance_level.lower(), GuidanceMode.FULL_GUIDANCE
            )

            signal_guide_config = ActionSignalGuideConfig(
                guidance_level=guidance_enum,
                feature_names=feature_names,
            )
            self.signal_guide = ActionSignalGuide(config=signal_guide_config)

            # Initialize enhanced signal integration
            self.signal_integration = SignalRewardIntegrator(
                signal_guide=self.signal_guide,
                signal_bonus_weight=signal_bonus_weight,
                signal_penalty_weight=signal_penalty_weight,
                granville_weight=granville_weight,
                dow_theory_weight=dow_theory_weight,
                heikin_ashi_weight=heikin_ashi_weight,
                enable_advanced_integration=enable_advanced_integration,
            )

            self.logger.info(
                f"Initialized Enhanced Action Signal Guide with level: {guidance_level}"
            )
            self.logger.info(
                f"Advanced integration: enabled={enable_advanced_integration}, "
                f"weights: Granville={granville_weight}, Dow={dow_theory_weight}, "
                f"Heikin-Ashi={heikin_ashi_weight}"
            )
            self._signal_guide_available = True

        except Exception as e:
            self.logger.error(f"Failed to initialize signal guide: {e}")
            self.enabled = False

    def integrate_signal(
        self, reward: float, observation: np.ndarray | None, action: int, step: int
    ) -> float:
        """
        Apply signal integration to the reward if enabled.

        Args:
            reward: Base reward before signal integration
            observation: Current observation
            action: Action taken
            step: Current training step

        Returns:
            Modified reward with signal integration
        """
        # self.logger.debug(
        #     f"integrate_signal called: enabled={self.enabled}, action={action}, step={step}"
        # ) if step % 50 == 0 else None
        if not self.enabled or self.signal_integration is None:
            return reward

        if observation is None:
            return reward

        try:
            # set feature names if not already set
            if self.signal_guide and self.signal_guide.feature_names is None:
                # Try to get feature names from config first
                if hasattr(self.config, "feature_names") and self.config.feature_names:
                    self.signal_guide.set_feature_names(self.config.feature_names)
                    self.logger.info(
                        f"set feature_names from config: {len(self.config.feature_names)} features"
                    )
                # If not available in config, try to get from environment
                elif hasattr(self, "_env") and hasattr(self._env, "feature_names"):
                    self.signal_guide.set_feature_names(self._env.feature_names)
                    self.logger.info(
                        f"set feature_names from env: {len(self._env.feature_names)} features"
                    )
                # Last resort: try to get from observation builder
                elif (
                    hasattr(self, "_env")
                    and hasattr(self._env, "observation_builder")
                    and hasattr(self._env.observation_builder, "feature_names")
                ):
                    self.signal_guide.set_feature_names(
                        self._env.observation_builder.feature_names
                    )
                    self.logger.info(
                        f"set feature_names from observation_builder: {len(self._env.observation_builder.feature_names)} features"
                    )
                else:
                    self.logger.warning(
                        "Could not set feature_names for signal guide - signals will not work"
                    )

            # Apply signal integration using the unified integrator
            return self.signal_integration.integrate_signal_reward(
                reward, observation, action, step
            )

        except Exception as e:
            self.logger.warning(f"Signal integration failed: {e}")
            return reward
