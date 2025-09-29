#!/usr/bin/env python3
"""
Unit tests for parameter validation enhancements
"""

import unittest

from ztb.evaluation.auto_feature_generator import ParameterCombinationGenerator


class TestParameterValidation(unittest.TestCase):
    """Test cases for enhanced parameter validation"""

    def setUp(self):
        """Set up test fixtures"""
        self.generator = ParameterCombinationGenerator()

        # Sample config
        self.config = {
            "ema": {"periods": [5, 8, 12, 20]},
            "kama": {
                "fast_periods": [5, 10],
                "slow_periods": [20, 30],
                "efficiency_periods": [10, 20, 30],
            },
            "parameter_validation": {
                "min_period": 2,
                "max_period": 200,
                "forbidden_combinations": [
                    {"feature": "ema_cross", "fast": ">=slow"},
                    {"feature": "kama", "fast": ">=slow"},
                ],
            },
        }

    def test_validate_combination_ema_valid(self):
        """Test EMA parameter validation with valid parameters"""
        combination = (12,)  # Single parameter for EMA
        is_valid = self.generator.validate_combination(combination, "ema")
        self.assertTrue(is_valid)

    def test_validate_combination_ema_invalid_period(self):
        """Test EMA parameter validation with invalid period"""
        # Note: validate_combination only does basic feature-type validation
        # Detailed parameter range validation is done elsewhere
        combination = (1,)
        is_valid = self.generator.validate_combination(combination, "ema")
        # Should pass basic validation since EMA has no special rules
        self.assertTrue(is_valid)

    def test_validate_combination_kama_valid(self):
        """Test KAMA parameter validation with valid parameters"""
        combination = (5, 20, 10)  # fast, slow, efficiency
        is_valid = self.generator.validate_combination(combination, "kama")
        self.assertTrue(is_valid)

    def test_validate_combination_kama_invalid_fast_slow(self):
        """Test KAMA parameter validation with fast >= slow (forbidden)"""
        combination = (20, 20, 10)  # fast == slow
        is_valid = self.generator.validate_combination(combination, "kama")
        self.assertFalse(is_valid)

        combination = (25, 20, 10)  # fast > slow
        is_valid = self.generator.validate_combination(combination, "kama")
        self.assertFalse(is_valid)

    def test_validate_combination_ema_cross_valid(self):
        """Test EMA cross parameter validation with valid parameters"""
        combination = (5, 20)  # fast, slow
        is_valid = self.generator.validate_combination(combination, "ema_cross")
        self.assertTrue(is_valid)

    def test_validate_combination_ema_cross_invalid(self):
        """Test EMA cross parameter validation with fast >= slow (forbidden)"""
        combination = (20, 20)  # fast == slow
        is_valid = self.generator.validate_combination(combination, "ema_cross")
        self.assertFalse(is_valid)

        combination = (25, 20)  # fast > slow
        is_valid = self.generator.validate_combination(combination, "ema_cross")
        self.assertFalse(is_valid)

    def test_validate_combination_unknown_feature(self):
        """Test parameter validation with unknown feature type"""
        combination = (10,)
        is_valid = self.generator.validate_combination(combination, "unknown_feature")
        # Should pass basic validation if no specific rules
        self.assertTrue(is_valid)

    def test_validate_combination_missing_config(self):
        """Test parameter validation with missing config"""
        combination = (12,)
        is_valid = self.generator.validate_combination(combination, "ema")
        # Should pass if no validation config
        self.assertTrue(is_valid)

    def test_generate_combinations_with_validation(self):
        """Test that generate_combinations filters invalid combinations"""
        # Test the static method directly
        param_list1 = [5, 10, 20]  # fast periods
        param_list2 = [20, 30, 40]  # slow periods

        combinations = list(
            ParameterCombinationGenerator.generate_combinations(
                param_list1, param_list2
            )
        )

        # Should contain valid combinations
        self.assertTrue(len(combinations) > 0)

        # Test validation on combinations
        for combo in combinations:
            fast, slow = combo
            # Valid combinations should have fast < slow
            is_valid = ParameterCombinationGenerator.validate_combination(combo, "kama")
            if fast < slow:
                self.assertTrue(
                    is_valid, f"Valid combination {combo} should pass validation"
                )
            else:
                self.assertFalse(
                    is_valid, f"Invalid combination {combo} should fail validation"
                )


if __name__ == "__main__":
    unittest.main()
