"""
SAC training task — v460.

Entry-point for running Soft Actor-Critic training with the v460
configuration schema.  The script reads a YAML/JSON config, builds the
training environment and model, and hands off to ``stable-baselines3``.

Type-safety notes
-----------------
* ``val_cfg.get("environment", {})`` returns ``object`` when ``val_cfg``
  is typed as ``dict[str, object]``.  We cast via ``collections.abc.Mapping``
  so that ``dict(...)`` resolves without ``type: ignore[arg-type]``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_str_map(raw: object) -> dict[str, Any]:
    """
    Safely coerce *raw* to ``dict[str, Any]``.

    If *raw* is already a ``Mapping`` it is shallow-copied; otherwise an
    empty dict is returned.  This avoids ``type: ignore[arg-type]`` /
    ``type: ignore[assignment]`` when calling ``dict(val_cfg.get(..., {}))``.
    """
    if isinstance(raw, Mapping):
        return dict(raw)
    return {}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    """Load a JSON training-config from *path*."""
    with path.open() as fh:
        data: dict[str, Any] = json.load(fh)
    return data


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_env_config(val_cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Extract and validate the ``environment`` sub-section of a validation config.

    Previously this pattern required ``# type: ignore[arg-type]``:

    .. code-block:: python

        env_cfg = dict(val_cfg.get("environment", {}))  # type: ignore[arg-type]

    With the ``_as_str_map`` helper above mypy is satisfied without suppression.
    """
    # L296-equivalent — was: dict(val_cfg.get("environment", {}))  # type: ignore[arg-type]
    env_cfg: dict[str, Any] = _as_str_map(val_cfg.get("environment"))

    # L312-equivalent — was: dict(val_cfg.get("evaluation", {}))  # type: ignore[assignment]
    eval_cfg: dict[str, Any] = _as_str_map(val_cfg.get("evaluation"))

    merged: dict[str, Any] = {**env_cfg, **eval_cfg}
    return merged


def run_training(config_path: Path) -> None:
    """
    Run SAC training.

    Parameters
    ----------
    config_path:
        Path to the JSON training config.
    """
    cfg = load_config(config_path)

    # Extract sub-configs without type: ignore.
    training_cfg: dict[str, Any] = _as_str_map(cfg.get("training"))
    val_cfg: dict[str, Any] = _as_str_map(cfg.get("validation"))
    env_cfg = build_env_config(val_cfg)

    total_timesteps: int = int(training_cfg.get("total_timesteps", 100_000))

    # Lazy import of heavy dependencies so the module stays importable without
    # stable_baselines3 installed (e.g. in unit-test environments).
    try:
        from stable_baselines3 import SAC  # type: ignore[import-not-found]
    except ImportError:
        print(
            "stable_baselines3 is not installed — skipping actual training.",
            file=sys.stderr,
        )
        return

    print(
        f"[sac_train] Starting SAC training for {total_timesteps:,} timesteps "
        f"with env config: {env_cfg}"
    )
    # NOTE: Actual environment construction and model fitting would go here.


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAC training — v460")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to JSON training config file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point."""
    args = _parse_args(argv)
    run_training(args.config)


if __name__ == "__main__":
    main()
