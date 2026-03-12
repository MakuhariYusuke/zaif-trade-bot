"""
Unit tests for P0-1 (Entry Gate Crash) and P0-2 (Entry Gate Config) completion verification.

Tests verify that Phase 0.2b/d implementations correctly address:
- P0-1: Entry Gate uses gate_result["should_enter"] (not should_block)
- P0-2: Entry Gate config is validated via validate_env_config()
"""

import pytest
from typing import Dict, Any

from ztb.training.utils.v457_config_utils import validate_env_config


class TestP02EntryGateConfig:
    """Verify P0-2: Entry Gate Config validation"""

    def test_validate_env_config_requires_entry_gate(self):
        """Verify validate_env_config raises error when entry_gate missing"""
        invalid_config: Dict[str, Any] = {}
        
        with pytest.raises(ValueError, match="entry_gate.*must be under"):
            validate_env_config(invalid_config)

    def test_validate_env_config_requires_entry_gate_dict(self):
        """Verify validate_env_config raises error when entry_gate is not dict"""
        invalid_config = {
            "entry_gate": "invalid_string",
        }
        
        with pytest.raises(ValueError, match="entry_gate.*must be a mapping"):
            validate_env_config(invalid_config)

    def test_validate_env_config_accepts_valid_config_minimal(self):
        """Verify validate_env_config accepts minimal valid config"""
        valid_config = {
            "entry_gate": {
                "enabled": True,
            },
        }
        
        # Should not raise
        validate_env_config(valid_config)

    def test_validate_env_config_validates_execution_model_if_present(self):
        """Verify execution_model validation when present"""
        config_with_exec_model = {
            "entry_gate": {
                "enabled": True,
            },
            "execution_model": {
                "costs": {
                    "slippage_model": "fixed",
                },
                "execution": {},
                "risk": {},
            },
        }
        
        # Should not raise
        validate_env_config(config_with_exec_model)

    def test_validate_env_config_rejects_invalid_slippage_model(self):
        """Verify invalid slippage_model is rejected"""
        invalid_config = {
            "entry_gate": {
                "enabled": True,
            },
            "execution_model": {
                "costs": {
                    "slippage_model": "invalid_model",
                },
                "execution": {},
                "risk": {},
            },
        }
        
        with pytest.raises(ValueError, match="Invalid slippage_model.*Must be one of"):
            validate_env_config(invalid_config)

    def test_validate_env_config_requires_execution_model_structure(self):
        """Verify execution_model requires costs/execution/risk"""
        invalid_config = {
            "entry_gate": {
                "enabled": True,
            },
            "execution_model": {
                "costs": {},
                # Missing 'execution' and 'risk'
            },
        }
        
        with pytest.raises(ValueError, match="Execution model missing required field"):
            validate_env_config(invalid_config)


class TestP01EntryGateImplementation:
    """Verify P0-1: Entry Gate implementation uses should_enter"""

    def test_entry_gate_uses_should_enter_key(self):
        """Verify fast_intraday_env_v456.py checks gate_result['should_enter']"""
        # Read the implementation file
        from pathlib import Path
        env_file = Path("ztb/trading/environment/fast_intraday_env_v456.py")
        
        if not env_file.exists():
            pytest.skip(f"Environment file not found: {env_file}")
        
        content = env_file.read_text(encoding='utf-8')
        
        # Verify 'should_enter' is used
        assert 'gate_result["should_enter"]' in content or "gate_result['should_enter']" in content, \
            "Implementation does not use gate_result['should_enter']"
        
        # Verify 'should_block' is NOT used (old API)
        assert 'gate_result["should_block"]' not in content and "gate_result['should_block']" not in content, \
            "Implementation still uses deprecated gate_result['should_block']"

    def test_entry_gate_config_loaded_from_env_config(self):
        """Verify entry_gate config is loaded from env_config"""
        from pathlib import Path
        env_file = Path("ztb/trading/environment/fast_intraday_env_v456.py")
        
        if not env_file.exists():
            pytest.skip(f"Environment file not found: {env_file}")
        
        content = env_file.read_text(encoding='utf-8')
        
        # Verify entry_gate is loaded from config
        assert 'entry_gate_config' in content, "entry_gate_config not found in implementation"
        assert 'env_config.get("entry_gate"' in content or 'config.get("entry_gate"' in content, \
            "entry_gate not loaded from env_config"


class TestP01P02Integration:
    """Integration verification for P0-1 and P0-2"""

    def test_p01_p02_documented_as_completed(self):
        """Verify P0-1 and P0-2 are documented as completed in Phase 0"""
        from pathlib import Path
        doc07 = Path("docs/v459/07_phase0_completion_report.md")
        
        if not doc07.exists():
            pytest.skip("Doc07 not found")
        
        content = doc07.read_text(encoding='utf-8')
        
        # Verify P0-1 is documented
        assert "P0-1" in content or "Entry Gate Crash" in content, \
            "P0-1 not documented in Phase 0 completion report"
        
        # Verify P0-2 is documented
        assert "P0-2" in content or "Entry Gate Config" in content, \
            "P0-2 not documented in Phase 0 completion report"
        
        # Verify Phase 0.2b (Entry Gate Safety) completed
        assert "Phase 0.2b" in content or "Entry Gate Safety" in content, \
            "Phase 0.2b not documented"

    def test_config_utils_has_validate_env_config(self):
        """Verify validate_env_config is exported from v457_config_utils"""
        from ztb.training.utils.v457_config_utils import validate_env_config
        
        assert callable(validate_env_config), "validate_env_config is not callable"
        
        # Test with valid config
        valid_config = {
            "entry_gate": {
                "enabled": True,
            },
        }
        validate_env_config(valid_config)  # Should not raise
