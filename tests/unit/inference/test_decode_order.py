"""
Unit tests for action decoding with strict order and tiebreaker.

Test Coverage:
1. Decode order: mask → softmax(T) → argmax (NOT argmax → softmax)
2. Tiebreaker: HOLD with small margin → select top2
3. Illegal top2: fallback to top1
4. Temperature effects on probability distribution
5. Batch processing
6. Numerical stability
"""

import numpy as np
import pytest

try:
    import torch
    from ztb.inference.decode import (
        InferenceConfig,
        compute_legal_sell_rate,
        decode_action,
    )
except ImportError:
    pytest.skip("torch or ztb.inference modules not available", allow_module_level=True)
class TestDecodeOrder:
    """Test strict decode order: mask → softmax(T) → argmax."""

    def test_mask_before_softmax(self):
        """Test that mask is applied BEFORE softmax normalization."""
        # Logits: HOLD=1.0, BUY=2.0, SELL=3.0
        # Mask: HOLD=1, BUY=1, SELL=0 (SELL illegal)
        logits = np.array([1.0, 2.0, 3.0])
        mask = np.array([1, 1, 0])

        action, info = decode_action(
            logits, mask, InferenceConfig(temperature=1.0, enable_tiebreaker=False)
        )

        # Expected: SELL masked → only HOLD/BUY compete → BUY wins (higher logit)
        assert action == 1  # BUY

        # Verify probabilities: SELL should have ~0 probability
        assert info["probabilities"][2] < 1e-8, "SELL should have ~0 probability"

        # Verify HOLD + BUY probabilities sum to ~1
        assert np.isclose(
            info["probabilities"][0] + info["probabilities"][1], 1.0, atol=1e-6
        )

    def test_softmax_after_mask_not_before(self):
        """
        Test that softmax happens AFTER masking.

        If softmax happened before masking, high illegal logits would
        still contribute to normalization, distorting legal action probabilities.
        """
        # Scenario: SELL has very high logit but is illegal
        logits = np.array([1.0, 1.5, 100.0])  # SELL dominates
        mask = np.array([1, 1, 0])  # SELL illegal

        action, info = decode_action(
            logits, mask, InferenceConfig(temperature=1.0, enable_tiebreaker=False)
        )

        # If softmax before mask: SELL would dominate normalization
        # → HOLD/BUY would get tiny probabilities
        # If softmax after mask (CORRECT): SELL → -1e9 → exp(-1e9) ≈ 0
        # → HOLD/BUY compete normally

        # Verify: BUY should win (logit 1.5 > 1.0)
        assert action == 1

        # Verify: HOLD + BUY probabilities are reasonable (not distorted)
        assert info["probabilities"][0] > 0.2, "HOLD should have reasonable probability"
        assert info["probabilities"][1] > 0.4, "BUY should have higher probability"

    def test_temperature_scaling_affects_distribution(self):
        """Test that temperature scaling changes probability distribution."""
        logits = np.array([1.0, 2.0, 3.0])
        mask = np.array([1, 1, 1])  # All legal

        # Low temperature (0.1): more greedy, sharp distribution
        # Note: Temperature is clamped to [0.5, 1.5] range
        _, info_low = decode_action(
            logits, mask, InferenceConfig(temperature=0.1, enable_tiebreaker=False)
        )

        # High temperature (10.0): more uniform distribution
        # Note: Temperature is clamped to [0.5, 1.5] range
        _, info_high = decode_action(
            logits, mask, InferenceConfig(temperature=10.0, enable_tiebreaker=False)
        )

        # Low temp: SELL (highest logit) should have highest probability (adjusted for clamping)
        assert info_low["probabilities"][2] > 0.8  # Relaxed from 0.9 due to clamping

        # High temp: probabilities should be more uniform
        assert (
            info_high["probabilities"][2] < 0.6
        ), "High temp should reduce max probability"
        assert np.std(info_high["probabilities"]) < np.std(
            info_low["probabilities"]
        ), "High temp should reduce variance"


class TestTiebreaker:
    """Test tiebreaker logic: HOLD + small margin → select top2."""

    def test_tiebreaker_activates_with_small_margin(self):
        """Test tiebreaker activates when HOLD is top1 with small margin."""
        # HOLD=0.34, BUY=0.33, SELL=0.33 (after softmax)
        # Margin: 0.34 - 0.33 = 0.01 < tau (0.05)
        # Construct logits that produce this distribution
        logits = np.array([0.1, 0.0, 0.0])  # Close values
        mask = np.array([1, 1, 1])

        action, info = decode_action(
            logits,
            mask,
            InferenceConfig(
                temperature=1.0, tiebreaker_tau=0.05, enable_tiebreaker=True
            ),
        )

        # Tiebreaker should activate: select BUY (top2) instead of HOLD (top1)
        assert (
            info["tiebreaker_activated"] is True
        ), "Tiebreaker should activate with small margin"
        assert action == info["top2_actions"][1], "Should select top2 action"

    def test_tiebreaker_inactive_with_large_margin(self):
        """Test tiebreaker does NOT activate when margin is large."""
        # HOLD=0.8, BUY=0.1, SELL=0.1
        # Margin: 0.8 - 0.1 = 0.7 > tau (0.05)
        logits = np.array([5.0, 0.0, 0.0])  # HOLD dominates
        mask = np.array([1, 1, 1])

        action, info = decode_action(
            logits,
            mask,
            InferenceConfig(
                temperature=1.0, tiebreaker_tau=0.05, enable_tiebreaker=True
            ),
        )

        # Standard argmax: select HOLD (top1)
        assert info["tiebreaker_activated"] is False
        assert action == 0  # HOLD

    def test_tiebreaker_inactive_when_top1_not_hold(self):
        """Test tiebreaker only activates when top1 is HOLD."""
        # BUY=0.34, HOLD=0.33, SELL=0.33
        # Margin: 0.01 < tau, but top1 is BUY (not HOLD)
        logits = np.array([0.0, 0.1, 0.0])  # BUY top1
        mask = np.array([1, 1, 1])

        action, info = decode_action(
            logits,
            mask,
            InferenceConfig(
                temperature=1.0, tiebreaker_tau=0.05, enable_tiebreaker=True
            ),
        )

        # Tiebreaker should NOT activate (top1 != HOLD)
        assert info["tiebreaker_activated"] is False
        assert action == 1  # BUY (standard argmax)

    def test_tiebreaker_inactive_when_top2_illegal(self):
        """Test tiebreaker fallback to top1 when top2 is illegal."""
        # Create scenario where HOLD is top1, but top2 (SELL) is illegal
        # Need to make BUY (1) the top2, but make it illegal
        logits = np.array([0.1, 0.05, -5.0])  # HOLD > BUY >> SELL
        mask = np.array([1, 0, 1])  # BUY illegal, SELL legal but low prob

        action, info = decode_action(
            logits,
            mask,
            InferenceConfig(
                temperature=1.0, tiebreaker_tau=0.1, enable_tiebreaker=True
            ),
        )

        # Top1: HOLD, Top2: BUY (but illegal)
        # Tiebreaker should NOT activate because top2 is illegal
        # Should select HOLD (top1)
        assert info["tiebreaker_activated"] is False
        assert action == 0  # HOLD

    def test_tiebreaker_disabled_config(self):
        """Test tiebreaker can be disabled via config."""
        logits = np.array([0.1, 0.0, 0.0])  # Small margin
        mask = np.array([1, 1, 1])

        action, info = decode_action(
            logits,
            mask,
            InferenceConfig(
                temperature=1.0, tiebreaker_tau=0.05, enable_tiebreaker=False
            ),
        )

        # Tiebreaker disabled: always use standard argmax
        assert info["tiebreaker_activated"] is False


class TestBatchProcessing:
    """Test batch processing with multiple observations."""

    def test_batch_decode(self):
        """Test decoding batch of observations."""
        batch_size = 4
        logits = np.array(
            [
                [1.0, 2.0, 3.0],  # Obs 1: SELL wins
                [5.0, 0.0, 0.0],  # Obs 2: HOLD wins
                [0.0, 5.0, 0.0],  # Obs 3: BUY wins
                [0.0, 0.0, 5.0],  # Obs 4: SELL wins clearly
            ]
        )
        masks = np.array(
            [
                [1, 1, 1],  # All legal
                [1, 1, 0],  # SELL illegal
                [1, 1, 1],  # All legal
                [1, 1, 1],  # All legal
            ]
        )

        actions, info = decode_action(
            logits, masks, InferenceConfig(temperature=1.0, enable_tiebreaker=False)
        )

        # Verify actions
        assert actions[0] == 2  # SELL
        assert actions[1] == 0  # HOLD (SELL illegal)
        assert actions[2] == 1  # BUY
        assert actions[3] == 2  # SELL

        # Verify batch shapes
        assert info["probabilities"].shape == (batch_size, 3)
        assert info["top2_actions"].shape == (batch_size, 2)
        assert info["margin"].shape == (batch_size,)

    def test_batch_with_different_masks(self):
        """Test batch where each observation has different legal actions."""
        logits = np.array(
            [
                [1.0, 2.0, 3.0],  # SELL highest
                [1.0, 2.0, 3.0],  # SELL highest but illegal
                [1.0, 2.0, 3.0],  # BUY highest (SELL illegal)
            ]
        )
        masks = np.array(
            [
                [1, 1, 1],  # All legal → SELL
                [1, 1, 0],  # SELL illegal → BUY
                [1, 1, 0],  # SELL illegal → BUY
            ]
        )

        actions, _ = decode_action(
            logits, masks, InferenceConfig(temperature=1.0, enable_tiebreaker=False)
        )

        assert actions[0] == 2  # SELL (legal)
        assert actions[1] == 1  # BUY (SELL illegal)
        assert actions[2] == 1  # BUY (SELL illegal)


class TestNumericalStability:
    """Test numerical stability with extreme values."""

    def test_large_logits(self):
        """Test stability with very large logits."""
        logits = np.array([1000.0, 1001.0, 1002.0])
        mask = np.array([1, 1, 1])

        action, info = decode_action(
            logits, mask, InferenceConfig(temperature=1.0, enable_tiebreaker=False)
        )

        # Should handle large values without overflow
        assert action == 2  # SELL (highest)
        assert not np.any(np.isnan(info["probabilities"]))
        assert not np.any(np.isinf(info["probabilities"]))
        assert np.isclose(np.sum(info["probabilities"]), 1.0, atol=1e-6)

    def test_negative_logits(self):
        """Test stability with negative logits."""
        logits = np.array([-1000.0, -999.0, -998.0])
        mask = np.array([1, 1, 1])

        action, info = decode_action(
            logits, mask, InferenceConfig(temperature=1.0, enable_tiebreaker=False)
        )

        # Should handle negative values
        assert action == 2  # SELL (highest)
        assert np.isclose(np.sum(info["probabilities"]), 1.0, atol=1e-6)

    def test_torch_tensor_input(self):
        """Test compatibility with PyTorch tensors."""
        logits_torch = torch.tensor([1.0, 2.0, 3.0])
        mask_torch = torch.tensor([1, 1, 1])

        action, info = decode_action(
            logits_torch,
            mask_torch,
            InferenceConfig(temperature=1.0, enable_tiebreaker=False),
        )

        # Should convert and process correctly
        assert isinstance(action, int)
        assert isinstance(info["probabilities"], np.ndarray)
        assert action == 2  # SELL


class TestLegalSellRate:
    """Test legal SELL rate computation."""

    def test_legal_sell_rate_basic(self):
        """Test basic legal SELL rate computation."""
        # 10 steps, SELL legal in 8 steps, SELL chosen 2 times
        actions = np.array([0, 0, 2, 0, 1, 2, 0, 0, 0, 1])  # 2 SELLs
        legal_masks = np.array(
            [
                [1, 1, 1],  # Step 0: all legal
                [1, 1, 1],  # Step 1: all legal
                [1, 1, 1],  # Step 2: SELL chosen (legal)
                [1, 1, 0],  # Step 3: SELL illegal
                [1, 1, 0],  # Step 4: SELL illegal
                [1, 1, 1],  # Step 5: SELL chosen (legal)
                [1, 1, 1],  # Step 6: all legal
                [1, 1, 1],  # Step 7: all legal
                [1, 1, 1],  # Step 8: all legal
                [1, 1, 1],  # Step 9: all legal
            ]
        )

        stats = compute_legal_sell_rate(actions, legal_masks)

        assert stats["total_steps"] == 10
        assert stats["legal_sell_steps"] == 8  # SELL legal in 8 steps
        assert stats["sell_actions"] == 2  # SELL chosen 2 times
        assert np.isclose(stats["legal_sell_rate"], 2 / 8)  # 25%
        assert np.isclose(stats["overall_sell_rate"], 2 / 10)  # 20%

    def test_legal_sell_rate_no_legal_sells(self):
        """Test when SELL is never legal."""
        actions = np.array([0, 1, 0, 1, 0])
        legal_masks = np.array(
            [
                [1, 1, 0],
                [1, 1, 0],
                [1, 1, 0],
                [1, 1, 0],
                [1, 1, 0],
            ]
        )

        stats = compute_legal_sell_rate(actions, legal_masks)

        assert stats["legal_sell_steps"] == 0
        assert stats["sell_actions"] == 0
        assert stats["legal_sell_rate"] == 0.0

    def test_legal_sell_rate_all_sells(self):
        """Test when all actions are SELL and legal."""
        actions = np.array([2, 2, 2, 2, 2])
        legal_masks = np.array(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ]
        )

        stats = compute_legal_sell_rate(actions, legal_masks)

        assert stats["legal_sell_steps"] == 5
        assert stats["sell_actions"] == 5
        assert np.isclose(stats["legal_sell_rate"], 1.0)  # 100%


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_all_actions_illegal_raises_error(self):
        """Test fallback behavior when all actions are illegal."""
        import warnings
        
        logits = np.array([1.0, 2.0, 3.0])
        mask = np.array([0, 0, 0])  # All illegal

        # Should fall back to HOLD (action 0) with a warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            action, info = decode_action(logits, mask)
            
            # Check that warning was issued
            assert len(w) == 1
            assert "no legal actions" in str(w[0].message).lower()
            
            # Check fallback to HOLD
            assert action == 0

    def test_single_legal_action(self):
        """Test with only one legal action."""
        logits = np.array([1.0, 2.0, 3.0])
        mask = np.array([1, 0, 0])  # Only HOLD legal

        action, info = decode_action(
            logits, mask, InferenceConfig(temperature=1.0, enable_tiebreaker=False)
        )

        # Must select HOLD (only legal action)
        assert action == 0
        assert np.isclose(info["probabilities"][0], 1.0, atol=1e-6)
        assert info["probabilities"][1] < 1e-8
        assert info["probabilities"][2] < 1e-8

    def test_default_config(self):
        """Test that default config is applied when None."""
        logits = np.array([1.0, 2.0, 3.0])
        mask = np.array([1, 1, 1])

        action, info = decode_action(logits, mask, config=None)

        # Should use defaults: T=0.7, tau=0.05, tiebreaker=True
        assert isinstance(action, int)
        assert "probabilities" in info
        assert "tiebreaker_activated" in info
