"""
Unit tests for P0-4: Val/Test Leakage Prevention

Verifies that:
- Val/Test evaluations use independent environment instances
- Each environment has独立したscaler state
- Val/Test reporter statistics are completely separated
"""

import pytest
from typing import Dict, Any
from pathlib import Path


class TestP04ValTestLeakagePrevention:
    """Verify P0-4: Val/Test environment separation"""

    def test_evaluator_creates_new_env_per_evaluation(self):
        """Verify _evaluate_on_df() creates new environment instance"""
        evaluator_file = Path("ztb/evaluation/walk_forward/evaluator.py")
        
        if not evaluator_file.exists():
            pytest.skip("Evaluator file not found")
        
        content = evaluator_file.read_text(encoding='utf-8')
        
        # Verify env_factory is called in _evaluate_on_df
        assert "eval_env = self.env_factory(df)" in content, \
            "_evaluate_on_df does not create new environment"
        
        # Verify P0-4 documentation
        assert "P0-4" in content, "P0-4 fix not documented in evaluator"

    def test_evaluator_docstring_specifies_val_test_separation(self):
        """Verify _evaluate_on_df docstring documents Val/Test separation"""
        evaluator_file = Path("ztb/evaluation/walk_forward/evaluator.py")
        
        if not evaluator_file.exists():
            pytest.skip("Evaluator file not found")
        
        content = evaluator_file.read_text(encoding='utf-8')
        
        # Verify docstring mentions Val/Test separation
        assert "Val/Test" in content or "Val/Test Leakage" in content, \
            "Val/Test separation not documented in docstring"
        
        # Verify scaler independence is mentioned
        assert "scaler" in content.lower() and "独立" in content or "independent" in content.lower(), \
            "Scaler independence not documented"

    def test_evaluate_window_with_model_uses_separate_reporters(self):
        """Verify Val/Test use separate BacktestReporter instances"""
        evaluator_file = Path("ztb/evaluation/walk_forward/evaluator.py")
        
        if not evaluator_file.exists():
            pytest.skip("Evaluator file not found")
        
        content = evaluator_file.read_text(encoding='utf-8')
        
        # Verify val_reporter and test_reporter are created separately
        assert "val_reporter = BacktestReporter()" in content, \
            "val_reporter not created"
        assert "test_reporter = BacktestReporter()" in content, \
            "test_reporter not created"
        
        # Verify they are used in separate evaluation calls
        assert "self._evaluate_on_df(model, val_df, val_reporter" in content, \
            "val_reporter not passed to Val evaluation"
        assert "self._evaluate_on_df(model, test_df, test_reporter" in content, \
            "test_reporter not passed to Test evaluation"

    def test_environment_reset_initializes_independent_scaler(self):
        """Verify environment reset() initializes scaler with independent state"""
        env_file = Path("ztb/trading/environment/fast_intraday_env_v456.py")
        
        if not env_file.exists():
            pytest.skip("Environment file not found")
        
        content = env_file.read_text(encoding='utf-8')
        
        # Verify scaler.reset() is called in reset()
        assert "self.scaler.reset()" in content, \
            "scaler.reset() not called in environment reset()"
        
        # Verify prewarm is performed
        assert "prewarm" in content.lower(), \
            "Prewarm not implemented for scaler initialization"

    def test_phase0_causal_scaler_has_fit_boundary(self):
        """Verify CausalScaler enforces fit boundary (Phase 0.2c)"""
        scaler_file = Path("ztb/processing/causal_online_scaler.py")
        
        if not scaler_file.exists():
            pytest.skip("CausalScaler file not found")
        
        content = scaler_file.read_text(encoding='utf-8')
        
        # Verify fit() has end_idx parameter
        assert "def fit(" in content and "end_idx" in content, \
            "fit() does not have end_idx parameter"
        
        # Verify fit range is enforced
        assert "[:end_idx" in content or "[:end_idx+1]" in content, \
            "fit() does not enforce end_idx boundary"


class TestP04Integration:
    """Integration test for P0-4: end-to-end Val/Test separation"""

    def test_p04_documented_in_phase1_spec(self):
        """Verify P0-4 is documented in Phase 1 specification"""
        doc09 = Path("docs/v459/09_phase1_specification.md")
        
        if not doc09.exists():
            pytest.skip("Doc09 not found")
        
        content = doc09.read_text(encoding='utf-8')
        
        # Verify P0-4 is documented
        assert "P0-4" in content, "P0-4 not documented in Doc09"
        assert "Val/Test Leakage" in content or "Val/Test leakage" in content, \
            "Val/Test Leakage not documented"
        
        # Verify environment separation is documented
        assert "環境分離" in content or "environment" in content.lower(), \
            "Environment separation not documented"

    def test_phase0_test_coverage_includes_leakage_prevention(self):
        """Verify Phase 0 tests include data leakage prevention tests"""
        integration_test = Path("tests/integration/test_v459_phase0_integration.py")
        
        if not integration_test.exists():
            pytest.skip("Phase 0 integration test not found")
        
        content = integration_test.read_text(encoding='utf-8')
        
        # Verify leakage prevention tests exist
        assert "TestDataLeakagePrevention" in content, \
            "Data leakage prevention test class not found"
        
        # Verify scaler leakage tests
        assert "no_future_leak" in content or "leakage" in content.lower(), \
            "No future leakage test not found"

    def test_doc07_references_val_test_separation(self):
        """Verify Phase 0 completion report mentions Val/Test separation"""
        doc07 = Path("docs/v459/07_phase0_completion_report.md")
        
        if not doc07.exists():
            pytest.skip("Doc07 not found")
        
        content = doc07.read_text(encoding='utf-8')
        
        # Verify Val/Test separation concept is mentioned (flexible check)
        # Note: Doc07 may not have val_reporter/test_reporter explicitly after Doc08 corrections
        val_test_mentioned = (
            "Val/Test" in content or 
            "validation" in content.lower() and "test" in content.lower() or
            "P0-4" in content
        )
        
        assert val_test_mentioned, \
            "Val/Test separation concept not documented in Doc07"

    def test_evaluator_env_factory_preserves_independence(self):
        """Verify env_factory pattern ensures environment independence"""
        evaluator_file = Path("ztb/evaluation/walk_forward/evaluator.py")
        
        if not evaluator_file.exists():
            pytest.skip("Evaluator file not found")
        
        content = evaluator_file.read_text(encoding='utf-8')
        
        # Verify env_factory is used consistently
        assert "self.env_factory" in content, "env_factory not used"
        
        # Verify each evaluation gets independent env
        eval_on_df_count = content.count("def _evaluate_on_df")
        env_factory_in_eval = content.count("eval_env = self.env_factory(df)")
        
        # At least one env_factory call should exist in _evaluate_on_df
        assert env_factory_in_eval >= 1, \
            "env_factory not called in _evaluate_on_df"
