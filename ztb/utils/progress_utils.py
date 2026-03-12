#!/usr/bin/env python3
"""
Progress bar utilities for training visualization.
"""

import importlib.util
from typing import Any, Optional

STABLE_BASELINES3_AVAILABLE = importlib.util.find_spec("stable_baselines3") is not None

def configure_progress_bar(
    config: dict[str, Any],
    cli_override: bool | None = None,
    log: Any | None = None,
) -> bool:
    """
    Normalize progress bar settings and coordinate Stable-Baselines3 verbosity.

    Args:
        config: Mutable training configuration dictionary.
        cli_override: Optional explicit preference from CLI flags.
        log: Optional logger; defaults to module-level logger.

    Returns:
        bool: True when progress visuals should be enabled.
    """
    if config.get("_progress_configured"):
        return bool(config.get("progress_bar", False))

    logger_obj = log
    progress_preference: bool | None = cli_override

    legacy_top_level = config.pop("progress_bar", None)
    training_section = config.get("training")
    legacy_training = None
    if isinstance(training_section, dict):
        legacy_training = training_section.pop("progress_bar", None)

    if progress_preference is None and legacy_top_level is not None:
        progress_preference = bool(legacy_top_level)
    if progress_preference is None and legacy_training is not None:
        progress_preference = bool(legacy_training)

    ppo_config = config.setdefault("ppo", {})
    if not isinstance(ppo_config, dict):
        if logger_obj:
            logger_obj.warning(
                "PPO configuration expected to be a dict, but received %s. "
                "Disabling progress bar to avoid inconsistent state.",
                type(ppo_config),
            )
        config["progress_bar"] = False
        return False

    if progress_preference is None:
        progress_preference = bool(ppo_config.get("verbose", 0))

    use_progress_bar = bool(progress_preference)

    if STABLE_BASELINES3_AVAILABLE:
        desired_verbose = 1 if use_progress_bar else 0
        current_verbose = ppo_config.get("verbose")
        if current_verbose != desired_verbose:
            if logger_obj:
                logger_obj.info(
                    "Stable-Baselines3 detected; adjusting PPO verbose to %s for progress control.",
                    desired_verbose,
                )
        ppo_config["verbose"] = desired_verbose
    else:
        if logger_obj:
            logger_obj.info(
                "Stable-Baselines3 not available; %s fallback training progress bar.",
                "enabling" if use_progress_bar else "disabling",
            )
        if not use_progress_bar:
            ppo_config["verbose"] = 0

    config["progress_bar"] = use_progress_bar
    config["_progress_configured"] = True
    return use_progress_bar
