from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import TypeAlias

import yaml

from scripts.v460.lib.fill_config import FillTestConfig

ConfigMapping: TypeAlias = dict[str, object]


def parse_yaml_mapping(yaml_text: str) -> ConfigMapping:
    """Parse YAML text and require a top-level mapping."""
    data = yaml.safe_load(yaml_text)
    if not isinstance(data, dict):
        raise TypeError("expected YAML mapping")
    return data


@lru_cache(maxsize=None)
def load_yaml_mapping(path: Path) -> dict[str, object]:
    """Load YAML from a file and require a top-level mapping."""
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"expected YAML mapping in {path}")
    return data


@lru_cache(maxsize=None)
def load_fill_test_config_from_text(yaml_text: str) -> FillTestConfig:
    """Build FillTestConfig from inline YAML text once and reuse it in tests."""
    return FillTestConfig.from_yaml(parse_yaml_mapping(yaml_text))


@lru_cache(maxsize=None)
def load_fill_test_config_from_path(path: Path) -> FillTestConfig:
    """Build FillTestConfig from a YAML file path once and reuse it in tests."""
    return FillTestConfig.from_yaml(load_yaml_mapping(path))


@lru_cache(maxsize=None)
def _load_fill_test_config_from_mapping_json(mapping_json: str) -> FillTestConfig:
    mapping = json.loads(mapping_json)
    if not isinstance(mapping, dict):
        raise TypeError("expected config mapping JSON object")
    return FillTestConfig.from_yaml(mapping)


def load_fill_test_config_from_mapping(mapping: ConfigMapping) -> FillTestConfig:
    """Build FillTestConfig from a JSON-like mapping once and reuse it in tests."""
    canonical_json = json.dumps(mapping, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _load_fill_test_config_from_mapping_json(canonical_json)


def clone_fill_test_config(config: FillTestConfig) -> FillTestConfig:
    """Return an isolated FillTestConfig copy for tests that may mutate it."""
    return copy.deepcopy(config)
