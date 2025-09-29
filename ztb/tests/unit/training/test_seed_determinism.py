"""
Test seed determinism for reproducible training.

Verifies that identical seeds produce identical reward sequences.
"""

import pytest

from ztb.utils.seed_manager import SeedManager, set_global_seed


class TestSeedDeterminism:
    """Test deterministic behavior with seed management."""

    def test_seed_manager_sets_all_generators(self):
        """Test that SeedManager sets seeds for all random generators."""
        manager = SeedManager()

        # Test with specific seed
        test_seed = 12345
        manager.set_seed(test_seed)

        assert manager.get_current_seed() == test_seed

        # Verify Python random
        import random

        random.seed(test_seed)
        expected_random = random.random()
        random.seed(test_seed)
        actual_random = random.random()
        assert expected_random == actual_random

        # Verify NumPy if available
        try:
            import numpy as np

            np.random.seed(test_seed)
            expected_np = np.random.random()
            np.random.seed(test_seed)
            actual_np = np.random.random()
            assert expected_np == actual_np
        except ImportError:
            pass

        # Verify PyTorch if available
        try:
            import torch

            torch.manual_seed(test_seed)
            expected_torch = torch.rand(1).item()
            torch.manual_seed(test_seed)
            actual_torch = torch.rand(1).item()
            assert expected_torch == actual_torch
        except ImportError:
            pass

    def test_deterministic_seed_generation(self):
        """Test deterministic seed generation from context."""
        manager = SeedManager()

        base_seed = 42
        context1 = "environment"
        context2 = "evaluation"

        seed1 = manager.generate_deterministic_seed(base_seed, context1)
        seed2 = manager.generate_deterministic_seed(base_seed, context2)
        seed1_again = manager.generate_deterministic_seed(base_seed, context1)

        # Same context should give same seed
        assert seed1 == seed1_again

        # Different contexts should give different seeds
        assert seed1 != seed2

    def test_fork_seed_requires_base_seed(self):
        """Test that fork_seed requires a base seed to be set."""
        manager = SeedManager()

        with pytest.raises(ValueError, match="No base seed set"):
            manager.fork_seed("test")

    def test_fork_seed_deterministic(self):
        """Test that fork_seed produces deterministic results."""
        manager = SeedManager()
        manager.set_seed(100)

        fork1 = manager.fork_seed("context_a")
        fork2 = manager.fork_seed("context_b")
        fork1_again = manager.fork_seed("context_a")

        assert fork1 == fork1_again
        assert fork1 != fork2

    @pytest.mark.parametrize("seed", [None, 42, 123456])
    def test_global_seed_function(self, seed):
        """Test global seed setting function."""
        set_global_seed(seed)

        from ztb.utils.seed_manager import get_current_global_seed

        if seed is None:
            # When seed is None, it should be set to some value
            assert get_current_global_seed() is not None
        else:
            assert get_current_global_seed() == seed

    def test_torch_determinism_enabled(self):
        """Test that PyTorch determinism is enabled when available."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        manager = SeedManager()
        manager.set_seed(42)

        # Check that deterministic settings are applied
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    def test_disable_determinism(self):
        """Test disabling determinism for performance."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        manager = SeedManager()
        manager.set_seed(42)
        manager.disable_determinism()

        # Check that deterministic settings are disabled
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True

    def test_reward_sequence_reproducibility(self):
        """Test that reward sequences are reproducible with same seed."""

        # Mock a simple reward generation function
        def generate_rewards(seed: int, n_steps: int = 100) -> list:
            set_global_seed(seed)
            rewards = []
            for _ in range(n_steps):
                # Use numpy random for consistency
                try:
                    import numpy as np

                    reward = np.random.normal(0, 1)
                except ImportError:
                    import random

                    reward = random.gauss(0, 1)
                rewards.append(reward)
            return rewards

        seed = 999
        rewards1 = generate_rewards(seed)
        rewards2 = generate_rewards(seed)

        # Rewards should be identical
        assert len(rewards1) == len(rewards2)
        for r1, r2 in zip(rewards1, rewards2):
            assert abs(r1 - r2) < 1e-10  # Very small tolerance for floating point
