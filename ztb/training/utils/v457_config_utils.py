from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ztb.utils.config_loader import load_yaml_config


def load_config_dict(config_path: Path) -> dict[str, Any]:
    """Load a YAML config as a dict, returning {} for invalid or empty content."""
    config = load_yaml_config(config_path)
    if not isinstance(config, dict):
        return {}
    return config


def extract_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract training section as a dict."""
    training = config.get("training")
    if isinstance(training, Mapping):
        return dict(training)
    return {}


def validate_env_config(env_config: dict[str, Any]) -> None:
    """
    Doc04仕様: 環境設定の検証
    
    Args:
        env_config: training.environment設定
    
    Raises:
        ValueError: 設定が不正な場合（assertは使わない）
    """
    # entry_gate検証: training.environment配下に存在すべき
    if "entry_gate" not in env_config:
        raise ValueError(
            "Config error: 'entry_gate' must be under 'training.environment'. "
            "Please move entry_gate configuration to the correct location."
        )
    
    # entry_gate内容検証
    entry_gate = env_config["entry_gate"]
    if not isinstance(entry_gate, Mapping):
        raise ValueError(
            f"Config error: 'entry_gate' must be a mapping, got {type(entry_gate).__name__}"
        )
    
    # execution_model検証（存在する場合）
    if "execution_model" in env_config:
        exec_model = env_config["execution_model"]
        
        if not isinstance(exec_model, Mapping):
            raise ValueError(
                f"Config error: 'execution_model' must be a mapping, got {type(exec_model).__name__}"
            )
        
        # 必須フィールド検証
        required_fields = ["costs", "execution", "risk"]
        for field in required_fields:
            if field not in exec_model:
                raise ValueError(
                    f"Execution model missing required field: '{field}'. "
                    f"Required fields are: {required_fields}"
                )
        
        # costs内のslippage_model検証
        if "costs" in exec_model and isinstance(exec_model["costs"], Mapping):
            costs = exec_model["costs"]
            if "slippage_model" in costs:
                model = costs["slippage_model"]
                valid_models = ["fixed", "volume_based"]
                if model not in valid_models:
                    raise ValueError(
                        f"Invalid slippage_model: '{model}'. "
                        f"Must be one of {valid_models}"
                    )


def extract_env_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """
    Extract and validate training.environment section as a dict.
    
    Args:
        config: Full configuration mapping
    
    Returns:
        Environment configuration dict
    
    Raises:
        ValueError: If configuration is invalid
    """
    training = extract_training_config(config)
    env_config = training.get("environment")
    if isinstance(env_config, Mapping):
        env_dict = dict(env_config)
        # Doc04仕様: 検証実行
        validate_env_config(env_dict)
        return env_dict
    return {}


def extract_sac_params(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract training.sac_hyperparameters section as a dict."""
    training = extract_training_config(config)
    sac_params = training.get("sac_hyperparameters")
    if isinstance(sac_params, Mapping):
        return dict(sac_params)
    return {}


def extract_seed(config: Mapping[str, Any]) -> Optional[int]:
    """Extract training seed from config (training.seed or training.sac_hyperparameters.seed)."""
    training = extract_training_config(config)
    seed = None
    if isinstance(training, Mapping):
        seed = training.get("seed")
        if seed is None:
            sac_params = training.get("sac_hyperparameters")
            if isinstance(sac_params, Mapping):
                seed = sac_params.get("seed")
    if seed is None:
        return None
    try:
        return int(seed)
    except (TypeError, ValueError):
        return None
