"""Action prediction implementation for live trading."""

from typing import TYPE_CHECKING, Any

import numpy as np

from ztb.trading.constants import ACTION_HOLD, ACTION_NAMES, normalize_action
from ztb.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from ztb.trading.live_trader.live_trader import LiveTrader


def _resolve_expected_obs_dim(live_trader: "LiveTrader") -> int:
    """モデルの観測空間から期待次元数を解決する.

    優先順位:
      1. model.observation_space.shape[0]  (SB3 モデルから直接取得)
      2. live_trader.expected_features      (FeatureSchemaManager 経由)
      3. フォールバック無し — 呼出元で features をそのまま使う
    """
    model = getattr(live_trader, "model", None)
    if model is not None:
        obs_space = getattr(model, "observation_space", None)
        if obs_space is not None and hasattr(obs_space, "shape") and obs_space.shape:
            return int(obs_space.shape[0])

    expected = getattr(live_trader, "expected_features", None)
    if expected is not None and isinstance(expected, int) and expected > 0:
        return expected

    return 0  # 0 = 不明 → フォールバック (features をそのまま使う)


class ActionPrediction:
    """Handles action prediction using the trained model."""

    def __init__(self, live_trader: "LiveTrader") -> None:
        """Initialize action prediction with reference to live trader."""
        self.live_trader = live_trader
        self.logger = get_logger(__name__)
        self._expected_dim: int | None = None

    @property
    def expected_dim(self) -> int:
        """期待される観測次元数 (遅延解決・キャッシュ)."""
        if self._expected_dim is None:
            self._expected_dim = _resolve_expected_obs_dim(self.live_trader)
        return self._expected_dim

    def _prepare_observation(self, features: np.ndarray) -> np.ndarray:
        """モデルの期待次元に合わせて観測ベクトルを整形する.

        - expected_dim > 0: features を切り詰め or ゼロパディング
        - expected_dim == 0: features をそのまま使用 (次元不明時のフォールバック)
        """
        dim = self.expected_dim
        if dim <= 0:
            # 観測空間が不明 — そのまま使う (dry-run 等)
            return features

        n = len(features)
        if n == dim:
            return features
        elif n > dim:
            self.logger.debug(
                f"Truncating features {n} → {dim} to match model observation space"
            )
            return features[:dim]
        else:
            self.logger.debug(
                f"Padding features {n} → {dim} to match model observation space"
            )
            return np.pad(features, (0, dim - n), "constant")

    def predict_action(self, features: np.ndarray[Any]) -> int:
        """Predict trading action using the model."""
        logger = self.logger
        try:
            # モデルの observation_space / schema に基づいて次元を整合
            obs_features = self._prepare_observation(features)
            logger.debug(
                f"Observation prepared: input={len(features)} → output={len(obs_features)} "
                f"(expected_dim={self.expected_dim})"
            )

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
