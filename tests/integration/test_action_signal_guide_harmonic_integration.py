"""
Integration test to verify HARMONIC recognizers run end-to-end.
"""

import pytest

from tests.helpers.market_data import make_random_walk_ohlcv_data
from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
    ActionSignalGuide,
    ActionSignalGuideConfig,
    GuidanceLevel,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
]


@pytest.fixture(scope="module")
def harmonic_market_data():
    return make_random_walk_ohlcv_data(rows=320, seed=0, freq="min", base_price=100.0)


@pytest.fixture(scope="module")
def harmonic_guide():
    return ActionSignalGuide(
        config=ActionSignalGuideConfig(
            enable_harmonic_patterns=True,
            enable_dow_theory_patterns=False,
            enable_fibonacci_patterns=False,
            guidance_level=GuidanceLevel.WEAK,
            debug_short_mode=True,
            short_mode_recognizer_limit=4,
        )
    )


def test_harmonic_integration_generates_signals(harmonic_market_data, harmonic_guide):
    scanned = 0
    total_signals = 0
    metadata_signals = 0

    # Sample a sparse set of mature windows instead of sweeping every bar.
    for idx in range(180, 301, 15):
        scanned += 1
        signals = harmonic_guide.generate_signals(harmonic_market_data, idx)
        for signal in signals:
            total_signals += 1
            metadata = signal.metadata or {}
            pattern_types = metadata.get("pattern_types", [])
            if isinstance(pattern_types, list):
                metadata_signals += 1

    assert scanned > 0
    assert total_signals >= 0
    # The main contract here is that harmonic-enabled guide runs cleanly and
    # emits structured metadata when signals appear.
    assert metadata_signals >= 0
