from __future__ import annotations

from scripts.v460.lib import param_adapter as shim_adapter
from ztb.trading.sizing import param_adapter as canonical_adapter


class TestParamAdapterCanonicalMigration:
    def test_shim_and_canonical_config_defaults_match(self) -> None:
        shim_cfg = shim_adapter.AdaptationConfig()
        canonical_cfg = canonical_adapter.AdaptationConfig()
        assert shim_cfg == canonical_cfg

    def test_shim_and_canonical_compute_match(self) -> None:
        shim_result = shim_adapter.compute_adaptation(
            fill_rate=0.7,
            as_ratio=0.1,
            sample_count=100,
            config=shim_adapter.AdaptationConfig(current_offset_ratio=0.05),
        )
        canonical_result = canonical_adapter.compute_adaptation(
            fill_rate=0.7,
            as_ratio=0.1,
            sample_count=100,
            config=canonical_adapter.AdaptationConfig(current_offset_ratio=0.05),
        )
        assert shim_result == canonical_result

    def test_clamp_offset_matches(self) -> None:
        assert shim_adapter.clamp_offset(0.5) == canonical_adapter.clamp_offset(0.5)
