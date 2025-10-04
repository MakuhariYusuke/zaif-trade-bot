"""
Tests for Promotion Engine functionality.
"""

from unittest.mock import patch

from ztb.evaluation.promotion import (
    DistributionCriterion,
    DurationCriterion,
    NumericCriterion,
    PromotionNotifier,
    PromotionResult,
    RatioCriterion,
    YamlPromotionEngine,
)


class TestNumericCriterion:
    """Test NumericCriterion class"""

    def test_greater_than_pass(self):
        criterion = NumericCriterion("sharpe_ratio", ">", 0.3, 0.5)
        result = criterion.evaluate({"sharpe_ratio": 0.5})
        assert result == (True, 0.5)

    def test_greater_than_fail(self):
        criterion = NumericCriterion("sharpe_ratio", ">", 0.3, 0.5)
        result = criterion.evaluate({"sharpe_ratio": 0.2})
        assert result == (False, 0.5 * (0.2 / 0.3))  # Partial score

    def test_less_than_pass(self):
        criterion = NumericCriterion("max_drawdown", "<", -0.15, 0.4)
        result = criterion.evaluate({"max_drawdown": -0.2})
        assert result == (True, 0.4)

    def test_missing_value(self):
        criterion = NumericCriterion("sharpe_ratio", ">", 0.3, 0.5)
        result = criterion.evaluate({})
        assert result == (False, 0.0)


class TestRatioCriterion:
    """Test RatioCriterion class"""

    def test_ratio_greater_than_pass(self):
        criterion = RatioCriterion("sortino_ratio", ">", 0.4, 0.5)
        result = criterion.evaluate({"sortino_ratio": 0.6})
        assert result == (True, 0.5)

    def test_ratio_greater_than_fail(self):
        criterion = RatioCriterion("calmar_ratio", ">", 0.5, 0.4)
        result = criterion.evaluate({"calmar_ratio": 0.3})
        assert result == (False, 0.4 * (0.3 / 0.5))  # Partial score

    def test_ratio_less_than_pass(self):
        criterion = RatioCriterion("sharpe_ratio", "<", 0.8, 0.3)
        result = criterion.evaluate({"sharpe_ratio": 0.6})
        assert result == (True, 0.3)


class TestDurationCriterion:
    """Test DurationCriterion class"""

    def test_duration_less_than_pass(self):
        criterion = DurationCriterion("max_drawdown_duration_days", "<", 30, 0.4)
        result = criterion.evaluate({"max_drawdown_duration_days": 20})
        assert result == (True, 0.4)

    def test_duration_less_than_fail(self):
        criterion = DurationCriterion("max_drawdown_duration_days", "<", 30, 0.4)
        result = criterion.evaluate({"max_drawdown_duration_days": 40})
        assert result == (False, 0.4 * (30 / 40))  # Partial score

    def test_duration_greater_than_pass(self):
        criterion = DurationCriterion("recovery_time", ">", 10, 0.5)
        result = criterion.evaluate({"recovery_time": 15})
        assert result == (True, 0.5)


class TestDistributionCriterion:
    """Test DistributionCriterion class"""

    def test_distribution_less_than_pass(self):
        criterion = DistributionCriterion("skew", "<", 1.0, 0.4)
        result = criterion.evaluate({"skew": 0.5})
        assert result == (True, 0.4)

    def test_distribution_less_than_fail(self):
        criterion = DistributionCriterion("kurtosis", "<", 1.0, 0.4)
        result = criterion.evaluate({"kurtosis": 2.0})
        assert result == (False, 0.4 * (1.0 / 2.0))  # Partial score

    def test_distribution_absolute_value(self):
        criterion = DistributionCriterion("skew", "<", 0.5, 0.5)
        result = criterion.evaluate({"skew": -0.8})  # | -0.8 | = 0.8 > 0.5, so fail
        assert result[0] == False


class TestYamlPromotionEngine:
    """Test YamlPromotionEngine class"""

    @patch("ztb.evaluation.promotion.Path")
    def test_load_config(self, mock_path):
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.__str__ = lambda x: "config/promotion_criteria.yaml"

        with patch("builtins.open") as mock_open:
            mock_file = mock_open.return_value.__enter__.return_value
            mock_file.read.return_value = """
categories:
  test_category:
    logic: "AND"
    criteria:
      - name: sharpe_ratio
        operator: ">"
        value: 0.3
        weight: 0.5
    required_score: 0.7
"""

            with patch("yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {
                    "categories": {
                        "test_category": {
                            "logic": "AND",
                            "criteria": [
                                {
                                    "name": "sharpe_ratio",
                                    "operator": ">",
                                    "value": 0.3,
                                    "weight": 0.5,
                                }
                            ],
                            "required_score": 0.7,
                        }
                    }
                }

                engine = YamlPromotionEngine()
                assert "test_category" in engine.config["categories"]

    def test_evaluate_promotion_pending_to_staging(self):
        """Test promotion from pending to staging"""
        config = {
            "categories": {
                "trend_features": {
                    "logic": "AND",
                    "criteria": [
                        {
                            "name": "sharpe_ratio",
                            "operator": ">",
                            "value": 0.3,
                            "weight": 1.0,
                        }
                    ],
                    "required_score": 0.8,
                }
            },
            "staging": {"min_samples_required": 1000},
        }

        engine = YamlPromotionEngine.__new__(YamlPromotionEngine)
        engine.config = config
        engine.criteria_cache = {}
        engine.notifier = PromotionNotifier({})

        # Test successful promotion
        feature_results = {"sharpe_ratio": 0.5, "sample_count": 1500}
        result, details = engine.evaluate_promotion(
            "test_feature", feature_results, "pending", "trend_features"
        )

        assert result == PromotionResult.PROMOTE
        assert details["achieved_score"] > details["required_score"]

    def test_evaluate_promotion_staging_to_verified(self):
        """Test promotion from staging to verified"""
        config = {
            "categories": {
                "trend_features": {
                    "logic": "AND",
                    "criteria": [
                        {
                            "name": "sharpe_ratio",
                            "operator": ">",
                            "value": 0.3,
                            "weight": 1.0,
                        }
                    ],
                    "required_score": 0.8,
                }
            },
            "staging": {"min_samples_required": 1000},
        }

        engine = YamlPromotionEngine.__new__(YamlPromotionEngine)
        engine.config = config
        engine.criteria_cache = {}
        engine.notifier = PromotionNotifier({})

        # Test successful promotion with sufficient samples
        feature_results = {"sharpe_ratio": 0.5, "sample_count": 1500}
        result, details = engine.evaluate_promotion(
            "test_feature", feature_results, "staging", "trend_features"
        )

        assert result == PromotionResult.PROMOTE
        assert details["staging_samples"] == 1500
        assert details["staging_min_samples"] == 1000

    def test_evaluate_promotion_insufficient_samples(self):
        """Test staging promotion blocked by insufficient samples"""
        config = {
            "categories": {
                "trend_features": {
                    "logic": "AND",
                    "criteria": [
                        {
                            "name": "sharpe_ratio",
                            "operator": ">",
                            "value": 0.3,
                            "weight": 1.0,
                        }
                    ],
                    "required_score": 0.8,
                }
            },
            "staging": {"min_samples_required": 2000},
        }

        engine = YamlPromotionEngine.__new__(YamlPromotionEngine)
        engine.config = config
        engine.criteria_cache = {}
        engine.notifier = PromotionNotifier({})

        # Test blocked by insufficient samples
        feature_results = {"sharpe_ratio": 0.5, "sample_count": 1500}
        result, details = engine.evaluate_promotion(
            "test_feature", feature_results, "staging", "trend_features"
        )

        assert result == PromotionResult.KEEP
        assert details["staging_samples"] == 1500
        assert details["staging_min_samples"] == 2000

    def test_evaluate_promotion_verified_to_demote(self):
        """Test demotion from verified"""
        config = {
            "categories": {
                "trend_features": {
                    "logic": "AND",
                    "criteria": [
                        {
                            "name": "sharpe_ratio",
                            "operator": ">",
                            "value": 0.3,
                            "weight": 1.0,
                        }
                    ],
                    "required_score": 0.8,
                }
            }
        }

        engine = YamlPromotionEngine.__new__(YamlPromotionEngine)
        engine.config = config
        engine.criteria_cache = {}
        engine.notifier = PromotionNotifier({})

        # Test demotion due to poor performance
        feature_results = {"sharpe_ratio": 0.1}
        result, details = engine.evaluate_promotion(
            "test_feature", feature_results, "verified", "trend_features"
        )

        assert result == PromotionResult.DEMOTE
        assert details["achieved_score"] < details["required_score"]

    def test_evaluate_promotion_or_logic(self):
        """Test OR logic promotion"""
        config = {
            "categories": {
                "test_category": {
                    "logic": "OR",
                    "criteria": [
                        {
                            "name": "sharpe_ratio",
                            "operator": ">",
                            "value": 0.5,
                            "weight": 0.6,
                        },
                        {
                            "name": "win_rate",
                            "operator": ">",
                            "value": 0.7,
                            "weight": 0.4,
                        },
                    ],
                    "required_score": 0.5,
                }
            }
        }

        engine = YamlPromotionEngine.__new__(YamlPromotionEngine)
        engine.config = config
        engine.criteria_cache = {}
        engine.notifier = PromotionNotifier({})

        # Test OR logic - only one criterion met
        feature_results = {"sharpe_ratio": 0.3, "win_rate": 0.8}
        result, details = engine.evaluate_promotion(
            "test_feature", feature_results, "pending", "test_category"
        )

        assert result == PromotionResult.PROMOTE
        assert details["logic"] == "OR"

    def test_evaluate_promotion_hard_requirements_pass(self):
        """Test promotion with hard requirements that pass"""
        config = {
            "categories": {
                "trend_features": {
                    "logic": "AND",
                    "criteria": [
                        {
                            "name": "sharpe_ratio",
                            "operator": ">",
                            "value": 0.3,
                            "weight": 1.0,
                            "type": "ratio",
                        }
                    ],
                    "hard_requirements": [
                        {
                            "name": "max_drawdown",
                            "operator": "<",
                            "value": -0.3,
                            "type": "numeric",
                        }
                    ],
                    "required_score": 0.8,
                }
            }
        }

        engine = YamlPromotionEngine.__new__(YamlPromotionEngine)
        engine.config = config
        engine.criteria_cache = {}
        engine.notifier = PromotionNotifier({})

        # Test successful promotion with hard requirements met
        feature_results = {"sharpe_ratio": 0.5, "max_drawdown": -0.4}
        result, details = engine.evaluate_promotion(
            "test_feature", feature_results, "pending", "trend_features"
        )

        assert result == PromotionResult.PROMOTE
        assert details["hard_requirements_passed"] == True
        assert len(details["hard_requirement_details"]) == 1

    def test_evaluate_promotion_hard_requirements_fail(self):
        """Test promotion blocked by hard requirements"""
        config = {
            "categories": {
                "trend_features": {
                    "logic": "AND",
                    "criteria": [
                        {
                            "name": "sharpe_ratio",
                            "operator": ">",
                            "value": 0.3,
                            "weight": 1.0,
                            "type": "ratio",
                        }
                    ],
                    "hard_requirements": [
                        {
                            "name": "max_drawdown",
                            "operator": "<",
                            "value": -0.3,
                            "type": "numeric",
                        }
                    ],
                    "required_score": 0.8,
                }
            }
        }

        engine = YamlPromotionEngine.__new__(YamlPromotionEngine)
        engine.config = config
        engine.criteria_cache = {}
        engine.notifier = PromotionNotifier({})

        # Test blocked by hard requirements
        feature_results = {
            "sharpe_ratio": 0.5,
            "max_drawdown": -0.2,
        }  # Hard requirement fails
        result, details = engine.evaluate_promotion(
            "test_feature", feature_results, "pending", "trend_features"
        )

        assert result == PromotionResult.KEEP
        assert details["hard_requirements_passed"] == False
        assert details["reason"] == "Hard requirements not met"
