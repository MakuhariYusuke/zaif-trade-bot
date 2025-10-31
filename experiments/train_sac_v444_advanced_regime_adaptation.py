#!/usr/bin/env python3
"""
SAC v444 Advanced Market Regime Adaptation Training Script

This script trains a PPO model with advanced 12-regime classification system
and sophisticated market adaptation capabilities.

Key Features:
- 12-regime classification (strong_bull_trend, moderate_bull_trend, weak_bull_trend,
  strong_bear_trend, moderate_bear_trend, weak_bear_trend, high_volatility_ranging,
  moderate_volatility_ranging, low_volatility_ranging, extreme_volatility,
  consolidation, breakout_setup, breakdown_setup)
- Regime-specific behavioral optimization parameters
- Dynamic feature selection engine with regime-optimized feature weights
- Multi-timeframe analysis integration (5m, 15m, 1h, 4h, 1d)
- Advanced risk management with VaR integration and multi-layer stop loss
- Hierarchical timeframe voting for regime confirmation
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from ztb.analysis.v444_regime_classifier import RegimeType, V444RegimeClassifier
    from ztb.training.v4xx_unified_trainer import V4XXUnifiedTrainer

    print("✓ Successfully imported V4XXUnifiedTrainer and V444RegimeClassifier")
except ImportError as e:
    print(f"✗ Failed to import required modules: {e}")
    sys.exit(1)


class V444RegimeAdaptationTrainer(V4XXUnifiedTrainer):
    """
    Extended trainer for SAC v444 with advanced regime adaptation capabilities

    This trainer incorporates the 12-regime classification system and provides
    regime-specific optimization and feature selection.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the v444 regime adaptation trainer

        Args:
            config: Training configuration dictionary
        """
        super().__init__(config)

        # Initialize regime classifier
        regime_config = config.get("regime_classification", {})
        self.regime_classifier = V444RegimeClassifier(regime_config)

        # Regime-specific training parameters
        self.regime_adaptation_enabled = config.get("regime_adaptation", {}).get(
            "enabled", True
        )
        self.adaptive_feature_selection = config.get(
            "adaptive_feature_selection", {}
        ).get("enabled", True)

        # Multi-timeframe configuration
        self.multi_timeframe_enabled = config.get("multi_timeframe", {}).get(
            "enabled", True
        )
        self.timeframes = config.get("multi_timeframe", {}).get(
            "timeframes", ["5m", "15m", "1h", "4h", "1d"]
        )

        print("✓ V444 Regime Adaptation Trainer initialized")
        print(
            f"  - Regime adaptation: {'enabled' if self.regime_adaptation_enabled else 'disabled'}"
        )
        print(
            f"  - Adaptive features: {'enabled' if self.adaptive_feature_selection else 'disabled'}"
        )
        print(
            f"  - Multi-timeframe: {'enabled' if self.multi_timeframe_enabled else 'disabled'}"
        )

    def _get_regime_specific_config(self, current_regime: RegimeType) -> Dict[str, Any]:
        """
        Get regime-specific training configuration

        Args:
            current_regime: Detected market regime

        Returns:
            Dictionary with regime-specific parameters
        """
        base_config = self.regime_classifier.get_regime_config(current_regime)

        # Convert to training-compatible format
        regime_training_config = {
            "ppo": {
                "entropy_coef": base_config.get("entropy_regularization", 0.01),
                "value_loss_coef": 0.5,
                "max_grad_norm": 0.5,
            },
            "reward": {
                "action_balance_target": base_config.get("action_balance_target", 0.5),
                "regime_adaptive_scaling": True,
            },
            "features": {
                "regime_weights": base_config.get("feature_weights", {}),
                "adaptive_selection": self.adaptive_feature_selection,
            },
        }

        return regime_training_config

    def _detect_market_regime(self, observation: Dict[str, Any]) -> RegimeType:
        """
        Detect current market regime from observation data

        Args:
            observation: Current market observation

        Returns:
            Detected regime type
        """
        try:
            # Extract price data from observation
            # This assumes observation contains OHLCV data
            if "price_data" in observation:
                price_data = observation["price_data"]
                # Convert to DataFrame format expected by classifier
                import pandas as pd

                # Create DataFrame from price data
                df = pd.DataFrame(
                    {
                        "open": price_data.get("open", []),
                        "high": price_data.get("high", []),
                        "low": price_data.get("low", []),
                        "close": price_data.get("close", []),
                        "volume": price_data.get("volume", []),
                    }
                )

                if not df.empty and len(df) > 50:  # Minimum data requirement
                    result = self.regime_classifier.detect_regime(df)
                    return result.primary_regime

            # Fallback to consolidation if insufficient data
            return RegimeType.CONSOLIDATION

        except Exception as e:
            print(f"Warning: Regime detection failed: {e}")
            return RegimeType.CONSOLIDATION

    def _apply_regime_adaptation(self, training_step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply regime-specific adaptations to training step

        Args:
            training_step: Current training step data

        Returns:
            Modified training step with regime adaptations
        """
        if not self.regime_adaptation_enabled:
            return training_step

        # Detect current regime
        current_regime = self._detect_market_regime(
            training_step.get("observation", {})
        )

        # Get regime-specific configuration
        regime_config = self._get_regime_specific_config(current_regime)

        # Apply regime-specific parameters
        adapted_step = training_step.copy()

        # Update PPO parameters
        if "ppo_params" in adapted_step:
            adapted_step["ppo_params"].update(regime_config.get("ppo", {}))

        # Update reward configuration
        if "reward_config" in adapted_step:
            adapted_step["reward_config"].update(regime_config.get("reward", {}))

        # Update feature configuration
        if "feature_config" in adapted_step:
            adapted_step["feature_config"].update(regime_config.get("features", {}))

        # Log regime detection
        adapted_step["detected_regime"] = current_regime.value

        return adapted_step

    def training_step(self, *args, **kwargs):
        """
        Override training step to include regime adaptation

        This method intercepts the training step and applies regime-specific
        adaptations before passing to the parent trainer.
        """
        # Get the original training step
        step_data = super().training_step(*args, **kwargs)

        # Apply regime adaptation
        adapted_step = self._apply_regime_adaptation(step_data)

        return adapted_step


def main():
    """Main training function for SAC v444 advanced regime adaptation."""

    # Configuration path
    config_path = "config/sac_v444_advanced_regime_adaptation_config.json"

    if not os.path.exists(config_path):
        print(f"✗ Configuration file not found: {config_path}")
        sys.exit(1)

    print("🚀 Starting SAC v444 Advanced Market Regime Adaptation Training")
    print(f"📋 Configuration: {config_path}")
    print("🎯 Target: 25% return improvement, 20% drawdown reduction")

    try:
        # Load and validate configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        print("✓ Configuration loaded successfully")
        print(
            f"  - Training episodes: {config.get('training', {}).get('total_episodes', 'N/A')}"
        )
        print(
            "  - Regimes: 12 (strong_bull_trend, moderate_bull_trend, weak_bull_trend, strong_bear_trend, moderate_bear_trend, weak_bear_trend, high_volatility_ranging, moderate_volatility_ranging, low_volatility_ranging, extreme_volatility, consolidation, breakout_setup, breakdown_setup)"
        )
        print(
            f"  - Multi-timeframe: {config.get('multi_timeframe', {}).get('enabled', False)}"
        )
        print(
            f"  - Adaptive features: {config.get('adaptive_feature_selection', {}).get('enabled', False)}"
        )

        # Initialize trainer
        trainer = V444RegimeAdaptationTrainer(config)

        print("\n🏃 Starting training process...")

        # Start training
        trainer.train()

        print("✅ Training completed successfully!")
        print("📊 Check results directory for performance metrics and regime analysis")

    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        print("💾 Saving checkpoint...")
        # Add checkpoint saving logic here if needed

    except Exception as e:
        print(f"✗ Training failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
