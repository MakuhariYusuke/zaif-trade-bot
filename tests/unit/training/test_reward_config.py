"""
Test Reward Config Schema and Loading

Phase 3 Day 3: Reward Config読み込みテスト
"""

import pytest
from pathlib import Path

from ztb.training.reward_config_schema import (
    RewardConfigSchema,
    load_reward_config,
    list_available_configs,
    compare_configs,
)
from ztb.trading.environment.utils.config import RewardSettings


def test_list_available_configs():
    """利用可能なConfigファイルのリストテスト"""
    configs = list_available_configs()
    assert len(configs) >= 3, "Should have at least 3 stage configs"
    
    # ファイル名確認
    config_names = [c.stem for c in configs]
    assert "stage1_basic" in config_names
    assert "stage2_extended" in config_names
    assert "stage3_advanced" in config_names


def test_load_stage1_config():
    """Stage 1 Config読み込みテスト"""
    config_path = Path("configs/rewards/stage1_basic.yaml")
    
    # Schema検証
    config_dict = RewardConfigSchema.load_and_validate(config_path)
    assert config_dict["name"] == "stage1_basic"
    assert config_dict["curriculum_stage"] == "simple"
    assert config_dict["balance_penalty"] == 0.1
    
    # RewardSettings変換
    settings = load_reward_config(config_path)
    assert isinstance(settings, RewardSettings)
    assert settings.curriculum_stage == "simple"
    assert settings.balance_penalty == 0.1
    assert settings.profit_weight == 1.0
    assert settings.risk_weight == 0.3


def test_load_stage2_config():
    """Stage 2 Config読み込みテスト"""
    config_path = Path("configs/rewards/stage2_extended.yaml")
    
    # Schema検証
    config_dict = RewardConfigSchema.load_and_validate(config_path)
    assert config_dict["name"] == "stage2_extended"
    assert config_dict["curriculum_stage"] == "trading_focused"
    assert config_dict["balance_penalty"] == 0.5
    
    # RewardSettings変換
    settings = load_reward_config(config_path)
    assert isinstance(settings, RewardSettings)
    assert settings.curriculum_stage == "trading_focused"
    assert settings.balance_penalty == 0.5
    assert settings.risk_weight == 0.7
    assert settings.dynamic_reward_shaping is not None
    assert settings.dynamic_reward_shaping.get("enabled") == True


def test_load_stage3_config():
    """Stage 3 Config読み込みテスト"""
    config_path = Path("configs/rewards/stage3_advanced.yaml")
    
    # Schema検証
    config_dict = RewardConfigSchema.load_and_validate(config_path)
    assert config_dict["name"] == "stage3_advanced"
    assert config_dict["curriculum_stage"] == "profit_optimized"
    assert config_dict["balance_penalty"] == 1.0
    assert config_dict["ultra_profit_multiplier"] == 2.5
    
    # RewardSettings変換
    settings = load_reward_config(config_path)
    assert isinstance(settings, RewardSettings)
    assert settings.curriculum_stage == "profit_optimized"
    assert settings.ultra_profit_multiplier == 2.5
    assert settings.ultra_risk_multiplier == 0.4
    assert settings.consistency_weight == 0.6
    
    # forced_balance設定確認
    assert "forced_balance_enabled" in settings.custom_reward_params
    assert settings.custom_reward_params["forced_balance_enabled"] == True


def test_config_progression():
    """Stage 1→2→3の設定進化テスト"""
    stage1 = load_reward_config("configs/rewards/stage1_basic.yaml")
    stage2 = load_reward_config("configs/rewards/stage2_extended.yaml")
    stage3 = load_reward_config("configs/rewards/stage3_advanced.yaml")
    
    # balance_penaltyの段階的増加
    assert stage1.balance_penalty < stage2.balance_penalty < stage3.balance_penalty
    
    # risk_weightの調整
    assert stage1.risk_weight < stage2.risk_weight
    
    # consistency_weightの増加
    assert stage1.consistency_weight < stage2.consistency_weight < stage3.consistency_weight
    
    # ultra_profit_multiplierの有効化
    assert stage1.ultra_profit_multiplier == 1.0
    assert stage2.ultra_profit_multiplier == 1.0
    assert stage3.ultra_profit_multiplier > 2.0


def test_compare_configs():
    """Config比較機能テスト"""
    comparison = compare_configs([
        "configs/rewards/stage1_basic.yaml",
        "configs/rewards/stage2_extended.yaml",
        "configs/rewards/stage3_advanced.yaml",
    ])
    
    assert len(comparison) == 3
    assert "stage1_basic" in comparison
    assert "stage2_extended" in comparison
    assert "stage3_advanced" in comparison
    
    # Stage 3がUltra Profitモード
    assert comparison["stage3_advanced"]["ultra_profit_multiplier"] > 2.0
    assert comparison["stage3_advanced"]["dynamic_shaping_enabled"] == True


def test_invalid_config_detection():
    """無効なConfig検出テスト"""
    # 存在しないファイル
    with pytest.raises(FileNotFoundError):
        load_reward_config("configs/rewards/nonexistent.yaml")


def test_stage1_metadata():
    """Stage 1メタデータテスト"""
    config_dict = RewardConfigSchema.load_and_validate("configs/rewards/stage1_basic.yaml")
    
    metadata = config_dict.get("metadata", {})
    assert metadata.get("phase") == "Phase 3 Day 3"
    assert metadata.get("purpose") is not None
    assert "expected_behavior" in metadata


def test_stage2_dynamic_shaping():
    """Stage 2動的シェーピング設定テスト"""
    config_dict = RewardConfigSchema.load_and_validate("configs/rewards/stage2_extended.yaml")
    
    drs = config_dict.get("dynamic_reward_shaping", {})
    assert drs.get("enabled") == True
    assert drs.get("market_regime_awareness") == True
    assert drs.get("volatility_adjusted_rewards") == True
    assert "regime_coefficients" in drs


def test_stage3_forced_balance():
    """Stage 3強制バランス設定テスト"""
    config_dict = RewardConfigSchema.load_and_validate("configs/rewards/stage3_advanced.yaml")
    
    fb = config_dict.get("forced_balance", {})
    assert fb.get("enabled") == True
    assert fb.get("min_actions") == 20
    assert "target_ratios" in fb
    assert fb["target_ratios"]["buy"] == 0.4
    assert fb["target_ratios"]["sell"] == 0.35
    assert fb["target_ratios"]["hold"] == 0.25


def test_config_schema_validation():
    """Schema検証機能テスト"""
    config_dict = {
        "name": "test_config",
        "description": "Test configuration",
        "curriculum_stage": "simple",
        "reward_scale": 100.0,
    }
    
    errors = RewardConfigSchema.validate(config_dict)
    assert len(errors) == 0, f"Valid config should have no errors, got: {errors}"


def test_config_schema_missing_required():
    """必須フィールド欠如検出テスト"""
    config_dict = {
        "name": "test_config",
        # description missing
        "curriculum_stage": "simple",
        "reward_scale": 100.0,
    }
    
    errors = RewardConfigSchema.validate(config_dict)
    assert len(errors) > 0
    assert any("description" in err for err in errors)


def test_config_schema_invalid_stage():
    """無効なcurriculum_stage検出テスト"""
    config_dict = {
        "name": "test_config",
        "description": "Test",
        "curriculum_stage": "invalid_stage",
        "reward_scale": 100.0,
    }
    
    errors = RewardConfigSchema.validate(config_dict)
    assert len(errors) > 0
    assert any("curriculum_stage" in err for err in errors)


def test_config_schema_value_constraints():
    """値範囲制約テスト"""
    config_dict = {
        "name": "test_config",
        "description": "Test",
        "curriculum_stage": "simple",
        "reward_scale": 100.0,
        "position_soft_cap": 1.5,  # 範囲外 (0.0-1.0)
    }
    
    errors = RewardConfigSchema.validate(config_dict)
    assert len(errors) > 0
    assert any("position_soft_cap" in err for err in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
