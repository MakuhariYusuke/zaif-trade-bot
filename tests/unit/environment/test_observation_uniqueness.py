#!/usr/bin/env python3
"""
Observation Uniqueness and Update Verification Tests.

ENVIRONMENT SPECIFICATION (Critical Assumptions):
- HeavyTradingEnv supports SHORTING: position ∈ {-1.0, 0.0, +1.0}
- At position=0 (neutral), BOTH BUY and SELL are LEGAL actions
  - BUY: Opens long (+1.0) or closes short
  - SELL: Closes long + Opens short (-1.0)
- This is the "always-flip" design (NOT a bug)

Tests to detect:
1. Observation fixation (same hash across steps)
2. Reference reuse (same object returned by env.step())
3. Observation delta norms (L2 distance between consecutive steps)
4. Feature schema consistency
"""

import hashlib
from typing import List

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from ztb.trading.environment.environment import EnvironmentConfig, HeavyTradingEnv


class TestObservationUniqueness:
    """Tests for observation update correctness."""

    @pytest.fixture
    def varying_price_data(self) -> pd.DataFrame:
        """Create test data with varying prices."""
        np.random.seed(42)
        n_steps = 100

        # Generate price series with clear trend
        base_price = 100.0
        noise = np.random.randn(n_steps) * 0.5
        trend = np.linspace(0, 10, n_steps)
        prices = base_price + trend + noise

        df = pd.DataFrame(
            {
                "close": prices,
                "open": prices - 0.1,
                "high": prices + 0.2,
                "low": prices - 0.2,
                "volume": np.random.uniform(1000, 2000, n_steps),
            }
        )

        return df

    @pytest.fixture
    def test_env(self, varying_price_data: pd.DataFrame) -> HeavyTradingEnv:
        """Create test environment with varying prices."""
        config = EnvironmentConfig(
            curriculum_stage="full",
            transaction_cost=0.0,
            initial_portfolio_value=10000.0,
            max_position_size=1.0,
        )
        env = HeavyTradingEnv(varying_price_data, config)
        return env

    def obs_hash(self, obs: NDArray[np.float32]) -> str:
        """Compute SHA1 hash of observation."""
        return hashlib.sha1(obs.tobytes()).hexdigest()

    def obs_delta_norm(
        self, obs1: NDArray[np.float32], obs2: NDArray[np.float32]
    ) -> float:
        """Compute L2 norm of observation difference."""
        return float(np.linalg.norm(obs1 - obs2))

    def test_observation_uniqueness_across_steps(
        self, test_env: HeavyTradingEnv
    ) -> None:
        """Test that observations change across steps (not fixed)."""
        test_env.reset()

        obs_hashes: List[str] = []
        observations: List[NDArray[np.float32]] = []

        # Collect observations for 50 steps
        for step in range(50):
            obs, _, _, _, _ = test_env.step(0)  # HOLD action
            obs_hash = self.obs_hash(obs)
            obs_hashes.append(obs_hash)
            observations.append(obs.copy())

        # Check for uniqueness
        unique_hashes = set(obs_hashes)
        duplicate_rate = 1.0 - (len(unique_hashes) / len(obs_hashes))

        print("\nObservation uniqueness analysis:")
        print(f"  Total steps: {len(obs_hashes)}")
        print(f"  Unique observations: {len(unique_hashes)}")
        print(f"  Duplicate rate: {duplicate_rate:.1%}")

        # Assert: at least 80% should be unique (allow some duplicates in stationary periods)
        assert (
            duplicate_rate < 0.2
        ), f"Observation duplicate rate {duplicate_rate:.1%} too high - observations may be fixed"

        # Check first 10 hashes are not all identical
        assert (
            len(set(obs_hashes[:10])) > 1
        ), "First 10 observations are identical - observation update is broken"

    def test_observation_reference_not_reused(self, test_env: HeavyTradingEnv) -> None:
        """Test that env.step() returns new array, not reused reference."""
        test_env.reset()

        obs1, _, _, _, _ = test_env.step(0)
        obs1_id = id(obs1)

        obs2, _, _, _, _ = test_env.step(0)
        obs2_id = id(obs2)

        obs3, _, _, _, _ = test_env.step(0)
        obs3_id = id(obs3)

        # Different steps should return different objects
        assert obs1_id != obs2_id, "Observation object is being reused (same id)"
        assert obs2_id != obs3_id, "Observation object is being reused (same id)"

        print("\nObservation object IDs:")
        print(f"  Step 1: {obs1_id}")
        print(f"  Step 2: {obs2_id}")
        print(f"  Step 3: {obs3_id}")
        print("  ✓ All different (good)")

    def test_observation_delta_norms(self, test_env: HeavyTradingEnv) -> None:
        """Test that observations change meaningfully between steps."""
        test_env.reset()

        prev_obs = None
        delta_norms: List[float] = []

        for step in range(50):
            obs, _, _, _, _ = test_env.step(0)  # HOLD action

            if prev_obs is not None:
                delta = self.obs_delta_norm(prev_obs, obs)
                delta_norms.append(delta)

            prev_obs = obs.copy()

        # Statistics
        mean_delta = np.mean(delta_norms)
        std_delta = np.std(delta_norms)
        zero_delta_rate = sum(1 for d in delta_norms if d < 1e-8) / len(delta_norms)

        print("\nObservation delta norms (L2):")
        print(f"  Mean: {mean_delta:.6f}")
        print(f"  Std: {std_delta:.6f}")
        print(f"  Min: {min(delta_norms):.6f}")
        print(f"  Max: {max(delta_norms):.6f}")
        print(f"  Zero delta rate: {zero_delta_rate:.1%}")

        # Assert: mean delta should be non-zero
        assert (
            mean_delta > 1e-6
        ), f"Mean observation delta {mean_delta} is too small - observations not updating"

        # Assert: not all deltas are zero
        assert (
            zero_delta_rate < 0.9
        ), f"Zero delta rate {zero_delta_rate:.1%} too high - observations mostly static"

    def test_observation_changes_with_different_actions(
        self, test_env: HeavyTradingEnv
    ) -> None:
        """Test that different actions produce different observations."""
        test_env.reset()

        # Take HOLD action
        obs_hold, _, _, _, _ = test_env.step(0)

        # Reset and take BUY action
        test_env.reset()
        obs_buy, _, _, _, _ = test_env.step(1)

        # Reset and take different sequence
        test_env.reset()
        test_env.step(0)  # HOLD
        obs_hold2, _, _, _, _ = test_env.step(0)  # HOLD again

        # Observations from different trajectories should differ
        delta_hold_buy = self.obs_delta_norm(obs_hold, obs_buy)
        delta_hold_hold2 = self.obs_delta_norm(obs_hold, obs_hold2)

        print("\nObservation deltas for different action sequences:")
        print(f"  HOLD vs BUY: {delta_hold_buy:.6f}")
        print(f"  HOLD vs HOLD->HOLD: {delta_hold_hold2:.6f}")

        # At least one should show significant difference
        assert (
            delta_hold_buy > 1e-6 or delta_hold_hold2 > 1e-6
        ), "Observations do not change with actions"

    def test_observation_schema_consistency(self, test_env: HeavyTradingEnv) -> None:
        """Test that observation schema (shape, dtype) is consistent."""
        obs_initial = test_env.reset()[0]

        initial_shape = obs_initial.shape
        initial_dtype = obs_initial.dtype

        print("\nObservation schema:")
        print(f"  Shape: {initial_shape}")
        print(f"  Dtype: {initial_dtype}")

        # Check consistency across 20 steps
        for step in range(20):
            obs, _, _, _, _ = test_env.step(0)

            assert (
                obs.shape == initial_shape
            ), f"Step {step}: Shape changed from {initial_shape} to {obs.shape}"
            assert (
                obs.dtype == initial_dtype
            ), f"Step {step}: Dtype changed from {initial_dtype} to {obs.dtype}"

        print("  ✓ Schema consistent across 20 steps")

    def test_observation_nan_inf_detection(self, test_env: HeavyTradingEnv) -> None:
        """Test for NaN/inf values in observations."""
        test_env.reset()

        nan_counts: List[int] = []
        inf_counts: List[int] = []

        for step in range(50):
            obs, _, _, _, _ = test_env.step(0)

            nan_count = int(np.isnan(obs).sum())
            inf_count = int(np.isinf(obs).sum())

            nan_counts.append(nan_count)
            inf_counts.append(inf_count)

        total_nan = sum(nan_counts)
        total_inf = sum(inf_counts)

        print("\nObservation quality check:")
        print(f"  Total NaN values: {total_nan}")
        print(f"  Total Inf values: {total_inf}")

        if total_nan > 0:
            nan_rate = total_nan / (50 * test_env.reset()[0].size)
            print(f"  NaN rate: {nan_rate:.2%}")

        if total_inf > 0:
            inf_rate = total_inf / (50 * test_env.reset()[0].size)
            print(f"  Inf rate: {inf_rate:.2%}")

        # Allow small amount of NaN (some features may have missing data)
        # but flag if more than 10%
        if total_nan > 0:
            nan_rate = total_nan / (50 * test_env.reset()[0].size)
            assert nan_rate < 0.1, f"NaN rate {nan_rate:.2%} too high"

        # Inf values should be rare
        assert total_inf == 0, f"Found {total_inf} Inf values in observations"
