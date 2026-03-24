from __future__ import annotations

from pathlib import Path

from tests.unit.v460._yaml_test_helpers import (
    clone_fill_test_config,
    load_fill_test_config_from_mapping,
    load_fill_test_config_from_path,
    load_fill_test_config_from_text,
)


def test_load_fill_test_config_from_text_returns_cached_config_copies() -> None:
    base = load_fill_test_config_from_text(
        """
symbol: btc_jpy
order_quantity: 0.002
skip_gate:
  enabled: true
  mode: pnl
"""
    )
    copy_cfg = clone_fill_test_config(base)

    assert base.symbol == "btc_jpy"
    assert copy_cfg.order_quantity == 0.002
    assert copy_cfg is not base


def test_load_fill_test_config_from_path_caches_by_path(tmp_path: Path) -> None:
    yaml_path = tmp_path / "fill_test.yaml"
    yaml_path.write_text(
        "symbol: btc_jpy\norder_quantity: 0.003\n",
        encoding="utf-8",
    )

    first = load_fill_test_config_from_path(yaml_path)
    second = load_fill_test_config_from_path(yaml_path)

    assert first is second
    assert second.order_quantity == 0.003


def test_load_fill_test_config_from_mapping_caches_by_mapping_value() -> None:
    first = clone_fill_test_config(
        load_fill_test_config_from_mapping(
            {
                "micro_timeout": {
                    "enabled": True,
                    "wait_sec": 12.0,
                },
            }
        )
    )
    second = clone_fill_test_config(
        load_fill_test_config_from_mapping(
            {
                "micro_timeout": {
                    "wait_sec": 12.0,
                    "enabled": True,
                },
            }
        )
    )

    assert first.micro_timeout_enabled is True
    assert second.micro_timeout_enabled is True
    assert first.micro_timeout_wait_sec == second.micro_timeout_wait_sec == 12.0
