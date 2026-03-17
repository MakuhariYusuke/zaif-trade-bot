"""Compatibility converter for legacy v4xx configuration formats."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import yaml

from ztb.utils.types import ConfigMap

def _as_map(value: object) -> ConfigMap:
    if isinstance(value, dict):
        return dict(value)
    return {}

class V4XXConfigConverter:
    """Normalize v4xx configs into a unified dict shape."""

    @staticmethod
    def load_and_convert_config(config_path: str) -> ConfigMap:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("Configuration root must be an object")
        return V4XXConfigConverter.convert_to_unified(dict(payload))

    @staticmethod
    def detect_config_version(config: Mapping[str, object]) -> str:
        version = config.get("version")
        if isinstance(version, str) and version:
            return version

        model_name: object | None = config.get("model_name")
        if not isinstance(model_name, str):
            training_map = _as_map(config.get("training"))
            model_name = training_map.get("model_name")
        if isinstance(model_name, str):
            lowered = model_name.lower()
            for token in ("v460", "v459", "v458", "v457", "v456", "v455", "v454"):
                if token in lowered:
                    return token
        return "v4xx"

    @staticmethod
    def convert_to_unified(config: Mapping[str, object]) -> ConfigMap:
        unified = dict(config)
        training = _as_map(unified.get("training"))
        unified["training"] = training

        if "algorithm" not in unified:
            algo = training.get("algorithm")
            if isinstance(algo, str):
                unified["algorithm"] = algo

        if "model_name" not in unified:
            model_name = training.get("model_name")
            if isinstance(model_name, str):
                unified["model_name"] = model_name

        environment = _as_map(training.get("environment"))
        if environment:
            env_cfg = _as_map(environment.get("config"))
            if env_cfg:
                environment["config"] = env_cfg
                training["environment"] = environment

        if "algorithm" not in unified:
            unified["algorithm"] = "sac"
        if "model_name" not in unified:
            unified["model_name"] = "v4xx_model"

        return unified
