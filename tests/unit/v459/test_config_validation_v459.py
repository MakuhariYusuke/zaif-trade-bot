"""
Phase 0.2d: Config検証強化テスト

Doc04仕様に基づく検証:
- entry_gate配置チェック
- execution_model整合性検証
- ValueError使用（assertは使わない）
"""

import pytest
from ztb.training.utils.v457_config_utils import validate_env_config, extract_env_config


class TestConfigValidation:
    """Config検証テスト"""

    def test_validate_entry_gate_present(self):
        """entry_gateが存在する場合は正常"""
        config = {
            "entry_gate": {
                "enabled": True,
                "model_path": "test_model"
            }
        }
        # 例外が発生しないことを確認
        validate_env_config(config)

    def test_validate_entry_gate_missing(self):
        """entry_gateが無い場合はValueError"""
        config = {}
        with pytest.raises(ValueError, match="entry_gate.*must be under"):
            validate_env_config(config)

    def test_validate_entry_gate_not_mapping(self):
        """entry_gateが辞書でない場合はValueError"""
        config = {
            "entry_gate": "invalid_string"
        }
        with pytest.raises(ValueError, match="entry_gate.*must be a mapping"):
            validate_env_config(config)

    def test_validate_execution_model_full(self):
        """execution_modelが完全な場合は正常"""
        config = {
            "entry_gate": {"enabled": True},
            "execution_model": {
                "costs": {
                    "slippage_model": "fixed"
                },
                "execution": {
                    "order_type": "market"
                },
                "risk": {
                    "max_position": 1.0
                }
            }
        }
        validate_env_config(config)

    def test_validate_execution_model_not_mapping(self):
        """execution_modelが辞書でない場合はValueError"""
        config = {
            "entry_gate": {"enabled": True},
            "execution_model": "invalid_string"
        }
        with pytest.raises(ValueError, match="execution_model.*must be a mapping"):
            validate_env_config(config)

    def test_validate_execution_model_missing_costs(self):
        """costsフィールドが無い場合はValueError"""
        config = {
            "entry_gate": {"enabled": True},
            "execution_model": {
                "execution": {},
                "risk": {}
            }
        }
        with pytest.raises(ValueError, match="missing required field.*costs"):
            validate_env_config(config)

    def test_validate_execution_model_missing_execution(self):
        """executionフィールドが無い場合はValueError"""
        config = {
            "entry_gate": {"enabled": True},
            "execution_model": {
                "costs": {},
                "risk": {}
            }
        }
        with pytest.raises(ValueError, match="missing required field.*execution"):
            validate_env_config(config)

    def test_validate_execution_model_missing_risk(self):
        """riskフィールドが無い場合はValueError"""
        config = {
            "entry_gate": {"enabled": True},
            "execution_model": {
                "costs": {},
                "execution": {}
            }
        }
        with pytest.raises(ValueError, match="missing required field.*risk"):
            validate_env_config(config)

    def test_validate_slippage_model_fixed(self):
        """slippage_model='fixed'は正常"""
        config = {
            "entry_gate": {"enabled": True},
            "execution_model": {
                "costs": {"slippage_model": "fixed"},
                "execution": {},
                "risk": {}
            }
        }
        validate_env_config(config)

    def test_validate_slippage_model_volume_based(self):
        """slippage_model='volume_based'は正常"""
        config = {
            "entry_gate": {"enabled": True},
            "execution_model": {
                "costs": {"slippage_model": "volume_based"},
                "execution": {},
                "risk": {}
            }
        }
        validate_env_config(config)

    def test_validate_slippage_model_invalid(self):
        """slippage_modelが不正な値の場合はValueError"""
        config = {
            "entry_gate": {"enabled": True},
            "execution_model": {
                "costs": {"slippage_model": "invalid_model"},
                "execution": {},
                "risk": {}
            }
        }
        with pytest.raises(ValueError, match="Invalid slippage_model.*invalid_model"):
            validate_env_config(config)

    def test_validate_no_execution_model(self):
        """execution_modelが無い場合は正常（オプション扱い）"""
        config = {
            "entry_gate": {"enabled": True}
        }
        validate_env_config(config)

    def test_validate_no_slippage_model(self):
        """slippage_modelが無い場合は正常（オプション扱い）"""
        config = {
            "entry_gate": {"enabled": True},
            "execution_model": {
                "costs": {},
                "execution": {},
                "risk": {}
            }
        }
        validate_env_config(config)


class TestExtractEnvConfig:
    """extract_env_config()統合テスト"""

    def test_extract_valid_config(self):
        """正常なconfigの抽出"""
        config = {
            "training": {
                "environment": {
                    "entry_gate": {"enabled": True},
                    "execution_model": {
                        "costs": {"slippage_model": "fixed"},
                        "execution": {},
                        "risk": {}
                    }
                }
            }
        }
        result = extract_env_config(config)
        assert "entry_gate" in result
        assert "execution_model" in result

    def test_extract_invalid_config_raises(self):
        """不正なconfigはValueError"""
        config = {
            "training": {
                "environment": {
                    # entry_gateが無い
                    "execution_model": {
                        "costs": {},
                        "execution": {},
                        "risk": {}
                    }
                }
            }
        }
        with pytest.raises(ValueError, match="entry_gate"):
            extract_env_config(config)

    def test_extract_empty_training_returns_empty(self):
        """training.environmentが無い場合は空dict"""
        config = {
            "training": {}
        }
        result = extract_env_config(config)
        assert result == {}
