#!/usr/bin/env python3
"""Helpers for resolving environments and extracting metrics."""

from __future__ import annotations

import logging
from typing import Callable, Iterable, Optional

from ztb.types.common import ObjectMap

logger = logging.getLogger(__name__)

MetricCaster = Callable[[object], object]
MetricSpec = tuple[str, tuple[str, ...], Optional[MetricCaster]]

_BASE_METRIC_SPECS: tuple[MetricSpec, ...] = (
    ("final_balance", ("balance", "portfolio_value"), float),
    ("initial_balance", ("initial_balance", "initial_portfolio_value"), float),
    ("total_trades", ("total_trades",), int),
    ("gross_pnl", ("gross_pnl",), float),
    ("total_fees", ("total_fees",), float),
    ("total_slippage", ("total_slippage",), float),
    ("net_pnl", ("net_pnl", "realized_pnl", "total_pnl"), float),
)

_OPTIONAL_METRIC_SPECS: tuple[MetricSpec, ...] = (
    ("buy_count", ("buy_count",), int),
    ("sell_count", ("sell_count",), int),
    ("reward_scale", ("reward_scale",), float),
    ("reward_clip_min", ("reward_clip_min",), float),
    ("reward_clip_max", ("reward_clip_max",), float),
    ("realized_pnl", ("realized_pnl",), float),
)


def resolve_env(source: object) -> Optional[object]:
    """Resolve an environment from a trainer, model, or env-like object."""
    if source is None:
        return None

    if _looks_like_env(source):
        return source

    env = _safe_getattr(source, "env")
    if env is not None:
        return env

    model = _safe_getattr(source, "model")
    if model is not None:
        resolved = resolve_env(model)
        if resolved is not None:
            return resolved

    env = _call_get_env(source)
    if env is not None:
        return env

    alg_trainer = _safe_getattr(source, "algorithm_trainer")
    if alg_trainer is not None:
        resolved = resolve_env(alg_trainer)
        if resolved is not None:
            return resolved

    return None


def unwrap_env(env: object, max_depth: int = 10) -> Optional[object]:
    """Unwrap VecEnv/Monitor-style wrappers to reach the base environment."""
    if env is None:
        return None

    current = _unwrap_vec_env(env)
    for _ in range(max_depth):
        next_env = _next_wrapped_env(current)
        if next_env is None or next_env is current:
            break
        current = next_env
    return current


def extract_env_metrics(
    env: object, include_optional: bool = False
) -> dict[str, object]:
    """Extract balance and trade metrics from an environment."""
    metrics: dict[str, object] = {}
    if env is None:
        return metrics

    unwrapped = unwrap_env(env)
    if unwrapped is None:
        return metrics

    _populate_metric_specs(metrics, unwrapped, _BASE_METRIC_SPECS)

    if include_optional:
        _populate_metric_specs(metrics, unwrapped, _OPTIONAL_METRIC_SPECS)
        _set_sharpe_ratio(metrics, unwrapped)

    return metrics


def extract_trainer_env_metrics(
    trainer: object, include_optional: bool = False
) -> dict[str, object]:
    """Extract environment metrics from a trainer or algorithm wrapper."""
    env = resolve_env(trainer)
    return extract_env_metrics(env, include_optional=include_optional)


def compute_balance_roi(
    metrics: dict[str, object],
    final_key: str = "final_balance",
    initial_key: str = "initial_balance",
) -> Optional[float]:
    """Compute ROI percentage from metrics when available."""
    try:
        final_balance = float(metrics[final_key])
        initial_balance = float(metrics[initial_key])
    except Exception as e:
        logger.debug("metric extraction failed: %s", e)
        return None

    if initial_balance == 0:
        return None

    return (final_balance - initial_balance) / initial_balance * 100


def _safe_getattr(obj: object, name: str) -> Optional[object]:
    try:
        return getattr(obj, name)
    except Exception as e:
        logger.debug("metric extraction failed: %s", e)
        return None


def _call_get_env(obj: object) -> Optional[object]:
    try:
        getter = getattr(obj, "get_env", None)
    except Exception as e:
        logger.debug("metric extraction failed: %s", e)
        return None

    if callable(getter):
        try:
            return getter()
        except Exception as e:
            logger.debug("metric extraction failed: %s", e)
            return None
    return None


def _looks_like_env(obj: object) -> bool:
    try:
        if hasattr(obj, "step") and hasattr(obj, "reset"):
            return True
        if hasattr(obj, "envs") or hasattr(obj, "venv"):
            return True
    except Exception as e:
        logger.debug("metric extraction failed: %s", e)
        return False
    return False


def _unwrap_vec_env(env: object) -> object:
    try:
        envs = getattr(env, "envs", None)
        if envs:
            return envs[0]
    except Exception as e:
        logger.debug("metric extraction failed: %s", e)
        return env
    return env


def _next_wrapped_env(env: object) -> Optional[object]:
    try:
        if hasattr(env, "unwrapped"):
            unwrapped = getattr(env, "unwrapped")
            if unwrapped is not env:
                return unwrapped
        if hasattr(env, "env"):
            return getattr(env, "env")
        if hasattr(env, "venv"):
            return getattr(env, "venv")
        envs = getattr(env, "envs", None)
        if envs:
            return envs[0]
    except Exception as e:
        logger.debug("metric extraction failed: %s", e)
        return None
    return None


def _populate_metric_specs(
    metrics: dict[str, object], obj: object, specs: Iterable[MetricSpec]
) -> None:
    for key, attrs, caster in specs:
        _set_first_attr(metrics, key, obj, attrs, caster)


def _set_first_attr(
    metrics: dict[str, object],
    key: str,
    obj: object,
    attr_names: Iterable[str],
    caster: Optional[MetricCaster] = None,
) -> None:
    for attr in attr_names:
        try:
            value = getattr(obj, attr)
        except Exception as e:
            logger.debug("metric extraction failed: %s", e)
            continue
        if value is None:
            continue
        if caster is not None:
            try:
                value = caster(value)
            except Exception as e:
                logger.debug("metric extraction failed: %s", e)
                continue
        metrics[key] = value
        return


def _set_sharpe_ratio(metrics: dict[str, object], env: object) -> None:
    try:
        getter = getattr(env, "get_sharpe_ratio", None)
    except Exception as e:
        logger.debug("metric extraction failed: %s", e)
        getter = None
    if callable(getter):
        try:
            metrics["sharpe_ratio"] = float(getter())
            return
        except Exception as e:
            logger.debug("metric extraction failed: %s", e)
            return
    _set_first_attr(metrics, "sharpe_ratio", env, ("sharpe_ratio",), caster=float)


__all__ = [
    "compute_balance_roi",
    "extract_env_metrics",
    "extract_trainer_env_metrics",
    "resolve_env",
    "unwrap_env",
]
