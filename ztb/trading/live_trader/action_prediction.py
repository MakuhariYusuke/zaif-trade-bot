"""Action prediction implementation for live trading."""

from typing import TYPE_CHECKING, Any

import numpy as np

from ztb.trading.constants import ACTION_HOLD, ACTION_NAMES, normalize_action
from ztb.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from ztb.trading.live_trader.live_trader import LiveTrader


class ActionPrediction:
    """Handles action prediction using the trained model."""

    def __init__(self, live_trader: "LiveTrader") -> None:
        """Initialize action prediction with reference to live trader."""
        self.live_trader = live_trader
        self.logger = get_logger(__name__)

    def predict_action(self, features: np.ndarray[Any]) -> int:
        """Predict trading action using the model."""
        logger = self.logger
        try:
            # Handle different observation spaces for different algorithms
            if (
                hasattr(self.live_trader, "algorithm")
                and self.live_trader.algorithm == "sac"
            ):
                # SAC expects 5 features, take first 5 or pad if needed
                if len(features) >= 5:
                    obs_features = features[:5]
                else:
                    obs_features = np.pad(features, (0, 5 - len(features)), "constant")
                logger.debug(f"Using first 5 features for SAC: {obs_features}")
            else:
                # PPO uses all features, but in dry-run mode use first 5 to match model expectations
                if self.live_trader.dry_run:
                    if len(features) >= 5:
                        obs_features = features[:5]
                    else:
                        obs_features = np.pad(
                            features, (0, 5 - len(features)), "constant"
                        )
                    logger.debug(
                        f"Dry-run mode: using first 5 features for PPO: {obs_features}"
                    )
                else:
                    obs_features = features

            # Reshape for model input
            obs = obs_features.reshape(1, -1)

            if self.live_trader._is_maskable_ppo:
                # Update mask provider state
                self.live_trader.mask_provider.update_state(
                    current_position=self.live_trader.position,
                    position_entry_step=self.live_trader._position_entry_step,
                    current_step=self.live_trader._current_step,
                    forced_close_reason=None,
                )
                # Use action masking
                action_masks = self.live_trader.mask_provider.get_action_mask()
                action, _ = self.live_trader.model.predict(
                    obs, action_masks=action_masks
                )
            else:
                # Standard prediction
                action, _ = self.live_trader.model.predict(obs)

            logger.debug(
                f"Model prediction result: {action}, type: {type(action)}, shape: {getattr(action, 'shape', 'no shape')}"
            )

            # Handle different action formats and spaces
            if (
                hasattr(self.live_trader, "algorithm")
                and self.live_trader.algorithm == "sac"
            ):
                # Continuous action space - discretize to [0,1,2]
                if isinstance(action, (int, np.integer)):
                    action_val = float(action)
                elif isinstance(action, (float, np.floating)):
                    action_val = float(action)
                elif isinstance(action, np.ndarray):
                    if action.ndim == 0:
                        action_val = float(action.item())
                    elif action.ndim == 1 and len(action) == 1:
                        action_val = float(action[0])
                    elif action.ndim == 2 and action.shape == (1, 1):
                        # SAC often returns [[value]] format
                        action_val = float(action[0][0])
                        logger.debug(f"SAC action format [[{action_val}]] detected")
                    else:
                        logger.warning(f"Unexpected continuous action format: {action}")
                        action_val = 0.0
                else:
                    logger.warning(
                        f"Unknown continuous action type: {type(action)}, value: {action}"
                    )
                    action_val = 0.0

                logger.debug(f"Continuous action value: {action_val}")
                # Discretize continuous action to discrete action
                threshold = 0.1  # Small threshold to avoid noise
                if action_val > threshold:
                    final_action = 1  # BUY
                elif action_val < -threshold:
                    final_action = 2  # SELL
                else:
                    final_action = 0  # HOLD

                # Reduce log verbosity - only log significant discretization events
                if abs(action_val) > threshold:
                    logger.info(
                        f"SAC model output: {action_val:.4f} -> {ACTION_NAMES.get(final_action, 'UNKNOWN')}"
                    )
                else:
                    logger.debug(f"SAC model output: {action_val:.4f} -> HOLD")

            else:
                # Discrete action space
                if isinstance(action, (int, np.integer)):
                    final_action = int(action)
                elif isinstance(action, (float, np.floating)):
                    final_action = int(action)
                elif isinstance(action, np.ndarray):
                    if action.ndim == 0:
                        final_action = int(action.item())
                    elif action.ndim == 1:
                        if len(action) == 1:
                            final_action = int(action[0])
                        else:
                            # Probability distribution
                            logger.debug(
                                f"Treating as probability distribution: {action}"
                            )
                            final_action = int(np.argmax(action))
                    else:
                        logger.debug(
                            f"Multi-dimensional action array, flattening: {action}"
                        )
                        final_action = int(np.argmax(action.flatten()))
                else:
                    logger.warning(
                        f"Unknown discrete action type: {type(action)}, value: {action}"
                    )
                    try:
                        final_action = int(action)
                    except (ValueError, TypeError):
                        logger.error(
                            f"Cannot convert action {action} to int, using HOLD"
                        )
                        final_action = ACTION_HOLD

            logger.debug(f"Converted action: {final_action}")

            # Normalize discrete legacy (0/1/2) and continuous into internal ACTION_* values
            # Note: ACTION_SELL is -1 internally, but many upstream models/configs still emit 2.
            final_action = normalize_action(final_action)

            action = final_action
            logger.debug(f"Final validated action: {action}")

            self.live_trader._current_step += 1

            return action

        except Exception as e:
            logger = self.logger
            logger.error(f"Failed to predict action: {e}")
            return ACTION_HOLD  # Safe fallback
