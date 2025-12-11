"""Model loading implementation for live trading."""

from typing import TYPE_CHECKING, Union

import gymnasium as gym

if TYPE_CHECKING:
    try:
        from sb3_contrib import MaskablePPO  # type: ignore
    except Exception:
        MaskablePPO = None  # type: ignore
    try:
        from stable_baselines3 import PPO, SAC  # type: ignore
    except Exception:
        PPO = None  # type: ignore
        SAC = None  # type: ignore

from ztb.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from ztb.trading.live_trader.live_trader import LiveTrader


class ModelLoading:
    """Handles model loading and initialization for live trading."""

    def __init__(self, live_trader: "LiveTrader"):
        """Initialize model loading with reference to live trader."""
        self.live_trader = live_trader
        self.logger = get_logger(__name__)

    def load_model(self) -> "Union[PPO, MaskablePPO, SAC]":
        """Load the trained PPO, MaskablePPO, or SAC model.

        Bug #27 Fix: Now properly loads MaskablePPO models and uses
        ActionMaskProvider for action masking in production.

        Schema Integration: Load schema information for feature validation.
        """
        if not self.live_trader.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.live_trader.model_path}"
            )

        logger = self.logger

        # Determine algorithm from model path
        model_name = str(self.live_trader.model_path).lower()
        if "sac" in model_name:
            algorithm = "sac"
        else:
            algorithm = "ppo"

        logger.info(
            f"Loading {algorithm.upper()} model from {self.live_trader.model_path}"
        )

        if algorithm == "sac":
            # Lazy import SAC to avoid importing torch during module import time
            try:
                from stable_baselines3 import SAC as _SAC
            except Exception:
                _SAC = None

            if _SAC is None:
                raise ImportError("stable_baselines3.SAC is required but not available")

            model = _SAC.load(str(self.live_trader.model_path))
            logger.info("Model loaded as SAC")
            self.live_trader._is_maskable_ppo = False  # SAC doesn't use masks
            self.live_trader.algorithm = "sac"
        else:
            # Try loading as MaskablePPO first, fallback to PPO
            try:
                # Lazy import MaskablePPO in case sb3_contrib is present
                try:
                    from sb3_contrib import MaskablePPO as _MaskablePPO
                except Exception:
                    _MaskablePPO = None

                if _MaskablePPO is None:
                    raise ImportError(
                        "sb3_contrib.MaskablePPO is required but not available"
                    )

                model = _MaskablePPO.load(str(self.live_trader.model_path))
                logger.info("Model loaded as MaskablePPO with action masking support")
                self.live_trader._is_maskable_ppo = True
                self.live_trader.algorithm = "ppo"
            except Exception as e:
                logger.info(f"Not a MaskablePPO model ({e}), loading as standard PPO")
                try:
                    from stable_baselines3 import PPO as _PPO
                except Exception:
                    _PPO = None

                if _PPO is None:
                    raise ImportError(
                        "stable_baselines3.PPO is required but not available"
                    )

                model = _PPO.load(str(self.live_trader.model_path))
                logger.info("Model loaded as standard PPO (no action masking)")
                self.live_trader._is_maskable_ppo = False
                self.live_trader.algorithm = "ppo"

        # Log model spaces
        obs_space = model.observation_space
        action_space = model.action_space
        logger.info(f"Model observation space: {obs_space}")
        logger.info(f"Observation shape: {obs_space.shape}")
        logger.info(f"Model action space: {action_space}")
        logger.info(f"Action space type: {type(action_space)}")

        # Check if action space is continuous
        Box = gym.spaces.Box
        Discrete = gym.spaces.Discrete

        if isinstance(action_space, Box):
            self.live_trader.is_continuous_action = True
            logger.info("Detected continuous action space - will discretize actions")
        elif isinstance(action_space, Discrete):
            self.live_trader.is_continuous_action = False
            logger.info("Detected discrete action space")
        else:
            self.live_trader.is_continuous_action = False
            logger.warning(
                f"Unknown action space type: {type(action_space)} - assuming discrete"
            )

        # ========================================================================
        # Schema-based feature validation (Phase 3 Integration)
        # ========================================================================
        if not self.live_trader.dry_run:
            try:
                from ztb.trading.environment.schema_env_factory import (
                    create_env_from_model_path,
                )
                from ztb.training.core.feature_schema_manager import (
                    FeatureSchemaManager,
                )

                # Load model schema
                model_name = self.live_trader.model_path.stem
                schema_manager = FeatureSchemaManager(model_name)

                try:
                    metadata = schema_manager.load_schema()
                    logger.info(f"✅ Schema loaded for model: {model_name}")
                    logger.info(f"   Expected features: {metadata.num_features}")
                    logger.info(f"   Schema hash: {metadata.schema_hash}")
                    logger.info(f"   Created at: {metadata.created_at}")

                    # Store schema info for feature validation
                    self.live_trader.expected_features = metadata.num_features
                    self.live_trader.feature_names = metadata.feature_names
                    self.live_trader.model_schema_hash = metadata.schema_hash
                    self.live_trader.schema_available = True

                    logger.info("📋 Model feature requirements:")
                    logger.info(
                        f"   Total: {len(self.live_trader.feature_names)} features"
                    )
                    logger.info(f"   First 5: {self.live_trader.feature_names[:5]}")
                    logger.info(f"   Last 5: {self.live_trader.feature_names[-5:]}")

                except FileNotFoundError:
                    logger.warning(f"⚠️  Schema not found for model: {model_name}")
                    logger.warning(
                        f"   Schema file expected at: {self.live_trader.ztb_config.get_model_dir()}/schemas/{model_name}/"
                    )
                    logger.warning("   Falling back to legacy validation")
                    logger.warning(
                        "   Recommendation: Run migration if this is an old model"
                    )

                    self.live_trader.expected_features = None
                    self.live_trader.feature_names = None
                    self.live_trader.model_schema_hash = None
                    self.live_trader.schema_available = False

            except ImportError as e:
                logger.warning(f"Schema system not available: {e}")
                logger.warning("Using legacy feature validation")
                self.live_trader.expected_features = None
                self.live_trader.feature_names = None
                self.live_trader.model_schema_hash = None
                self.live_trader.schema_available = False
        else:
            # Dry-run mode: skip schema loading entirely
            logger.info("Dry-run mode: skipping schema loading")
            # Set expected features dynamically from feature set
            try:
                from ztb.features.feature_set_manager import get_feature_manager

                manager = get_feature_manager()
                self.live_trader.expected_features = manager.get_feature_count(
                    "curated"
                )
            except Exception:
                self.live_trader.expected_features = 78  # Fallback
            self.live_trader.feature_names = None
            self.live_trader.model_schema_hash = None
            self.live_trader.schema_available = False

        # Legacy feature validation (fallback)
        try:
            # Temporarily initialize price history for feature checking
            if not hasattr(self.live_trader, "price_history"):
                current_price = 1000000.0  # Dummy price for checking
                self.live_trader.price_history = [
                    current_price
                ] * self.live_trader.config["price_history_length"]

            # Skip feature validation during model loading - will be done after adapter initialization
            logger.info(
                "Feature validation deferred until after adapter initialization"
            )

        except Exception as e:
            logger.warning(f"Could not prepare for feature validation: {e}")

        # Send model loaded notification (skip in dry-run)
        if not self.live_trader.dry_run:
            self.live_trader._send_notification(
                "✅ Model Loaded Successfully",
                f"Model path: {self.live_trader.model_path}",
                "success",
            )

        return model
