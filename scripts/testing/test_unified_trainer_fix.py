#!/usr/bin/env python3
"""
Test script to verify the unified_trainer balance penalty fix.
"""

import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_config_curriculum_stage():
    """Verify that config has correct curriculum_stage"""
    logger.info("=" * 80)
    logger.info("TEST 1: Verify config curriculum_stage")
    logger.info("=" * 80)

    config_files = [
        "config/sac_v444_3_balanced_penalty_scale_200.json",
        "config/sac_v444_4_balanced_penalty_scale_300.json",
        "config/sac_v444_5_balanced_penalty_scale_500.json",
    ]

    for config_file in config_files:
        if not Path(config_file).exists():
            logger.warning(f"Config not found: {config_file}")
            continue

        with open(config_file, "r") as f:
            config = json.load(f)

        curriculum_stage = (
            config.get("training", {})
            .get("curriculum_learning", {})
            .get("curriculum_stage", "NOT SET")
        )
        balance_penalty = (
            config.get("environment", {})
            .get("behavior_optimization", {})
            .get("balance_penalty", "NOT SET")
        )

        logger.info(f"\n{config_file}:")
        logger.info(f"  curriculum_stage: {curriculum_stage}")
        logger.info(f"  balance_penalty: {balance_penalty}")

        # Check if curriculum_stage is in the supported list
        supported_stages = (
            "forced_balance",
            "balanced_penalty",
            "balance_optimization",
            "balance_penalty",
        )
        if curriculum_stage in supported_stages:
            logger.info("  ✅ curriculum_stage is supported")
        else:
            logger.warning(
                f"  ⚠️  curriculum_stage '{curriculum_stage}' might not be recognized"
            )


def test_reward_calculator_logic():
    """Verify that reward_calculator supports the curriculum stage"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Verify reward_calculator supports curriculum stages")
    logger.info("=" * 80)

    try:
        import inspect

        from ztb.trading.environment.components.reward_calculator import (
            RewardCalculator,
        )

        # Get the source code
        source = inspect.getsource(RewardCalculator)

        # Check if the supported stages are in the source
        supported_stages = (
            "forced_balance",
            "balanced_penalty",
            "balance_optimization",
            "balance_penalty",
        )

        logger.info("\nChecking reward_calculator.py for supported curriculum stages:")
        for stage in supported_stages:
            if f'"{stage}"' in source or f"'{stage}'" in source:
                logger.info(f"  ✅ {stage} is referenced in RewardCalculator")
            else:
                logger.warning(f"  ❌ {stage} is NOT referenced in RewardCalculator")

        # Check the actual condition
        if "balance_penalty_enabled_stages" in source:
            logger.info(
                "  ✅ balance_penalty_enabled_stages tuple found in RewardCalculator"
            )
        elif "in balance_penalty_enabled_stages" in source:
            logger.info("  ✅ Support for multiple curriculum stages found")
        else:
            logger.warning("  ⚠️  No multi-stage support found in RewardCalculator")

    except ImportError as e:
        logger.warning(f"Could not import RewardCalculator: {e}")


def test_environment_config():
    """Verify that EnvironmentConfig supports curriculum_stage"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Verify EnvironmentConfig")
    logger.info("=" * 80)

    try:
        import inspect

        from ztb.training.environments.environment_config import EnvironmentConfig

        # Get the field definition
        fields = EnvironmentConfig.__dataclass_fields__

        if "curriculum_stage" in fields:
            field = fields["curriculum_stage"]
            logger.info("  ✅ curriculum_stage is defined in EnvironmentConfig")
            logger.info(f"     Type: {field.type}")
            logger.info(f"     Default: {field.default}")
        else:
            logger.warning("  ❌ curriculum_stage is NOT defined in EnvironmentConfig")

    except ImportError as e:
        logger.warning(f"Could not import EnvironmentConfig: {e}")


def test_config_manager():
    """Verify that ConfigManager correctly passes curriculum_stage"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Verify ConfigManager passes curriculum_stage")
    logger.info("=" * 80)

    try:
        import inspect

        from ztb.training.core.config_manager import ConfigManager

        # Get the source code
        source = inspect.getsource(ConfigManager)

        # Check if curriculum_stage is being set
        if "curriculum_stage" in source:
            logger.info("  ✅ curriculum_stage is handled in ConfigManager")

            # Check if it's being set from curriculum_learning
            if "curriculum_learning" in source and "curriculum_stage" in source:
                logger.info(
                    "  ✅ curriculum_stage is extracted from curriculum_learning"
                )
            else:
                logger.warning(
                    "  ⚠️  curriculum_stage extraction logic might be incomplete"
                )
        else:
            logger.warning("  ❌ curriculum_stage is NOT handled in ConfigManager")

    except ImportError as e:
        logger.warning(f"Could not import ConfigManager: {e}")


def main():
    """Run all tests"""
    logger.info("\n🔍 UNIFIED TRAINER BALANCE PENALTY FIX VERIFICATION\n")

    test_config_curriculum_stage()
    test_reward_calculator_logic()
    test_environment_config()
    test_config_manager()

    logger.info("\n" + "=" * 80)
    logger.info("✅ VERIFICATION COMPLETE")
    logger.info("=" * 80)
    logger.info("\nNext step: Run training to verify balance penalty is applied")
    logger.info(
        "  python quick_train_v444_configurable.py --config config/sac_v444_3_balanced_penalty_scale_200.json --verbose"
    )


if __name__ == "__main__":
    main()
