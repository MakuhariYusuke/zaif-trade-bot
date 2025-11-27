"""
Typed config access helpers.

These helpers centralize runtime validations and type narrowing for
configuration values. Prefer using these functions instead of direct
indexing or dot navigation on dynamic ConfigDicts.
"""
from typing import Optional, cast

from ztb.types.common import (
    ConfigDict,
    ConfigValue,
    is_config_dict,
    is_numeric_config_value,
)
from ztb.utils.exceptions.custom_exceptions import ValidationError


def _get_by_path(config: ConfigDict, path: str) -> Optional[ConfigValue]:
    """Retrieve nested config value using dot notation.

    Returns None if any intermediate key doesn't exist or the structure
    isn't a mapping.
    """
    keys = path.split(".")
    # Start as `object` and narrow with `is_config_dict` TypeGuard
    current = cast(ConfigValue, config)
    for k in keys:
        if is_config_dict(current) and k in current:
            current = current[k]
        else:
            return None
    return cast(Optional[ConfigValue], current)


def get_numeric(
    config: ConfigDict, path: str, default: Optional[float] = None
) -> float:
    val = _get_by_path(config, path)
    if val is None:
        if default is None:
            raise ValidationError(f"Missing numeric config value for '{path}'")
        return float(default)
    if not is_numeric_config_value(val):
        raise ValidationError(f"Config '{path}' must be numeric, got {type(val)}")
    return float(val)


def get_string(config: ConfigDict, path: str, default: Optional[str] = None) -> str:
    val = _get_by_path(config, path)
    if val is None:
        if default is None:
            raise ValidationError(f"Missing string config value for '{path}'")
        return default
    if not isinstance(val, str):
        raise ValidationError(f"Config '{path}' must be a string, got {type(val)}")
    return val


def get_bool(config: ConfigDict, path: str, default: Optional[bool] = None) -> bool:
    val = _get_by_path(config, path)
    if val is None:
        if default is None:
            raise ValidationError(f"Missing bool config value for '{path}'")
        return default
    if not isinstance(val, bool):
        raise ValidationError(f"Config '{path}' must be a bool, got {type(val)}")
    return val


def get_int(config: ConfigDict, path: str, default: Optional[int] = None) -> int:
    val = _get_by_path(config, path)
    if val is None:
        if default is None:
            raise ValidationError(f"Missing int config value for '{path}'")
        return default
    if not isinstance(val, int):
        raise ValidationError(f"Config '{path}' must be an int, got {type(val)}")
    return val


def get_dict(config: ConfigDict, path: str) -> ConfigDict:
    val = _get_by_path(config, path)
    if val is None:
        return {}
    if not is_config_dict(val):
        raise ValidationError(f"Config '{path}' must be a dict, got {type(val)}")
    return val
