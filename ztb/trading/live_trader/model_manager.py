"""
Model management for live trading bot.
"""
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO, SAC

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading and type detection."""

    def __init__(self) -> None:
        self.model: Optional[Union[PPO, MaskablePPO, SAC]] = None
        self._is_maskable_ppo: bool = False
        self._is_sac: bool = False
        self.expected_features: Optional[int] = None
        self.feature_names: Optional[list[str]] = None
        self.model_schema_hash: Optional[str] = None
        self.schema_available: bool = False

    def load_model(self, model_path: Path) -> Union[PPO, MaskablePPO, SAC]:
        """Load the trained model and detect its type.

        Args:
            model_path: Path to the model file

        Returns:
            Loaded model instance

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading model from {model_path}")

        # Try loading as MaskablePPO first, fallback to PPO, then SAC
        try:
            model = MaskablePPO.load(str(model_path))
            logger.info("Model loaded as MaskablePPO with action masking support")
            self._is_maskable_ppo = True
            self._is_sac = False
        except Exception as e:
            try:
                logger.info(f"Not a MaskablePPO model ({e}), trying standard PPO")
                model = PPO.load(str(model_path))
                logger.info("Model loaded as standard PPO (no action masking)")
                self._is_maskable_ppo = False
                self._is_sac = False
            except Exception as e2:
                logger.info(f"Not a PPO model ({e2}), trying SAC")
                model = SAC.load(str(model_path))
                logger.info("Model loaded as SAC")
                self._is_maskable_ppo = False
                self._is_sac = True

        self.model = model
        logger.info(
            f"_load_model completed: _is_maskable_ppo={self._is_maskable_ppo}, _is_sac={self._is_sac}"
        )
        return model

    def load_schema_info(self, model_name: str) -> None:
        """Load schema information for feature validation.

        Args:
            model_name: Name of the model
        """
        try:
            from ztb.training.core.feature_schema_manager import FeatureSchemaManager

            schema_manager = FeatureSchemaManager(model_name)

            try:
                metadata = schema_manager.load_schema()
                logger.info(f"✅ Schema loaded for model: {model_name}")
                logger.info(f"   Expected features: {metadata.num_features}")
                logger.info(f"   Schema hash: {metadata.schema_hash}")
                logger.info(f"   Created at: {metadata.created_at}")

                self.expected_features = metadata.num_features
                self.feature_names = metadata.feature_names
                self.model_schema_hash = metadata.schema_hash
                self.schema_available = True

                logger.info("📋 Model feature requirements:")
                logger.info(f"   Total: {len(self.feature_names)} features")
                logger.info(f"   First 5: {self.feature_names[:5]}")
                logger.info(f"   Last 5: {self.feature_names[-5:]}")

            except FileNotFoundError:
                logger.warning(f"⚠️  Schema not found for model: {model_name}")
                logger.warning("   Falling back to legacy validation")
                self._reset_schema_info()

        except ImportError as e:
            logger.warning(f"Schema system not available: {e}")
            logger.warning("Using legacy feature validation")
            self._reset_schema_info()

    def _reset_schema_info(self) -> None:
        """Reset schema information to None."""
        self.expected_features = None
        self.feature_names = None
        self.model_schema_hash = None
        self.schema_available = False

    @property
    def is_maskable_ppo(self) -> bool:
        """Check if model is MaskablePPO."""
        return self._is_maskable_ppo

    @property
    def is_sac(self) -> bool:
        """Check if model is SAC."""
        return self._is_sac

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information dictionary."""
        return {
            "is_maskable_ppo": self._is_maskable_ppo,
            "is_sac": self._is_sac,
            "expected_features": self.expected_features,
            "feature_names": self.feature_names,
            "model_schema_hash": self.model_schema_hash,
            "schema_available": self.schema_available,
        }
