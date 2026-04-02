from __future__ import annotations

import pytest

from scripts.v460.lib.fill_config import FillTestConfig


def test_live_yaml_hour_params_match_703_plan(v460_fill_test_config_base: FillTestConfig) -> None:
    cfg = v460_fill_test_config_base
    assert cfg.skip_gate_hour_offsets[12] == pytest.approx(0.3)
    assert cfg.sell_hour_offset_boost[14] == pytest.approx(2.5)
    assert cfg.sell_hour_offset_boost[16] == pytest.approx(2.5)
    assert cfg.hour_ceiling_mult[12] == pytest.approx(1.5)
    assert cfg.hour_ceiling_mult[16] == pytest.approx(2.5)


def test_existing_hour_params_stay_intact(v460_fill_test_config_base: FillTestConfig) -> None:
    cfg = v460_fill_test_config_base
    assert cfg.sell_hour_offset_boost[2] == pytest.approx(5.0)
    assert cfg.sell_hour_offset_boost[4] == pytest.approx(5.0)
    assert cfg.hour_ceiling_mult[21] == pytest.approx(2.5)
