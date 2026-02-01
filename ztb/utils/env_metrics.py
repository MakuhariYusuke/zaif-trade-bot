#!/usr/bin/env python3
"""Helpers for resolving environments and extracting metrics."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


def resolve_env(source: Any) -> Optional[Any]:
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


def unwrap_env(env: Any, max_depth: int = 10) -> Optional[Any]:
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
    env: Any, include_optional: bool = False
) -> Dict[str, Any]:
    """Extract balance and trade metrics from an environment."""
    metrics: Dict[str, Any] = {}
    if env is None:
        return metrics

    unwrapped = unwrap_env(env)
    if unwrapped is None:
        return metrics

    _set_first_attr(
        metrics,
        "final_balance",
        unwrapped,
        ("balance", "portfolio_value"),
        caster=float,
    )
    _set_first_attr(
        metrics,
        "initial_balance",
        unwrapped,
        ("initial_balance", "initial_portfolio_value"),
        caster=float,
    )
    _set_first_attr(
        metrics,
        "total_trades",
        unwrapped,
        ("total_trades",),
        caster=int,
    )
    _set_first_attr(
        metrics,
        "gross_pnl",
        unwrapped,
        ("gross_pnl",),
        caster=float,
    )
    _set_first_attr(
        metrics,
        "total_fees",
        unwrapped,
        ("total_fees",),
        caster=float,
    )
    _set_first_attr(
        metrics,
        "total_slippage",
        unwrapped,
        ("total_slippage",),
        caster=float,
    )
    _set_first_attr(
        metrics,
        "net_pnl",
        unwrapped,
        ("net_pnl", "realized_pnl", "total_pnl"),
        caster=float,
    )

    if include_optional:
        _set_first_attr(
            metrics,
            "buy_count",
            unwrapped,
            ("buy_count",),
            caster=int,
        )
        _set_first_attr(
            metrics,
            "sell_count",
            unwrapped,
            ("sell_count",),
            caster=int,
        )
        _set_sharpe_ratio(metrics, unwrapped)
        _set_first_attr(
            metrics,
            "reward_scale",
            unwrapped,
            ("reward_scale",),
            caster=float,
        )
        _set_first_attr(
            metrics,
            "reward_clip_min",
            unwrapped,
            ("reward_clip_min",),
            caster=float,
        )
        _set_first_attr(
            metrics,
            "reward_clip_max",
            unwrapped,
            ("reward_clip_max",),
            caster=float,
        )
        # P0: コスト分解メトリクスを追加（89# Phase 4.5）
        _set_first_attr(
            metrics,
            "gross_pnl",
            unwrapped,
            ("gross_pnl",),
            caster=float,
        )
        _set_first_attr(
            metrics,
            "net_pnl",
            unwrapped,
            ("net_pnl",),
            caster=float,
        )
        _set_first_attr(
            metrics,
            "total_fees",
            unwrapped,
            ("total_fees",),
            caster=float,
        )
        _set_first_attr(
            metrics,
            "total_slippage",
            unwrapped,
            ("total_slippage",),
            caster=float,
        )
        _set_first_attr(
            metrics,
            "realized_pnl",
            unwrapped,
            ("realized_pnl",),
            caster=float,
        )

    return metrics


def extract_trainer_env_metrics(
    trainer: Any, include_optional: bool = False
) -> Dict[str, Any]:
    """Extract environment metrics from a trainer or algorithm wrapper."""
    env = resolve_env(trainer)
    return extract_env_metrics(env, include_optional=include_optional)


def compute_balance_roi(
    metrics: Dict[str, Any],
    final_key: str = "final_balance",
    initial_key: str = "initial_balance",
) -> Optional[float]:
    """Compute ROI percentage from metrics when available."""
    try:
        final_balance = float(metrics[final_key])
        initial_balance = float(metrics[initial_key])
    except Exception:
        return None

    if initial_balance == 0:
        return None

    return (final_balance - initial_balance) / initial_balance * 100


def _safe_getattr(obj: Any, name: str) -> Optional[Any]:
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _call_get_env(obj: Any) -> Optional[Any]:
    try:
        getter = getattr(obj, "get_env", None)
    except Exception:
        return None

    if callable(getter):
        try:
            return getter()
        except Exception:
            return None
    return None


def _looks_like_env(obj: Any) -> bool:
    try:
        if hasattr(obj, "step") and hasattr(obj, "reset"):
            return True
        if hasattr(obj, "envs") or hasattr(obj, "venv"):
            return True
    except Exception:
        return False
    return False


def _unwrap_vec_env(env: Any) -> Any:
    try:
        envs = getattr(env, "envs", None)
        if envs:
            return envs[0]
    except Exception:
        return env
    return env


def _next_wrapped_env(env: Any) -> Optional[Any]:
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
    except Exception:
        return None
    return None


def _set_first_attr(
    metrics: Dict[str, Any],
    key: str,
    obj: Any,
    attr_names: Iterable[str],
    caster: Optional[type] = None,
) -> None:
    for attr in attr_names:
        if not hasattr(obj, attr):
            continue
        try:
            value = getattr(obj, attr)
        except Exception:
            continue
        if value is None:
            continue
        if caster is not None:
            try:
                value = caster(value)
            except Exception:
                continue
        metrics[key] = value
        return


def _set_sharpe_ratio(metrics: Dict[str, Any], env: Any) -> None:
    try:
        getter = getattr(env, "get_sharpe_ratio", None)
    except Exception:
        getter = None
    if callable(getter):
        try:
            metrics["sharpe_ratio"] = float(getter())
            return
        except Exception:
            return
    _set_first_attr(metrics, "sharpe_ratio", env, ("sharpe_ratio",), caster=float)


__all__ = [
    "compute_balance_roi",
    "extract_env_metrics",
    "extract_trainer_env_metrics",
    "resolve_env",
    "unwrap_env",
]
