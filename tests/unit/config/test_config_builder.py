"""
Unit tests for ConfigBuilder.

Tests all configuration building methods with various scenarios.
"""


import pytest

from ztb.training.core.config_builder import ConfigBuilder


class TestConfigBuilderBasics:
    """ConfigBuilderの基本機能テスト"""

    def test_initialization(self):
        """初期化テスト"""
        config = {"algorithm": "ppo", "model_name": "test"}
        builder = ConfigBuilder(config)

        assert builder.config == config
        assert builder._config_manager is None

    def test_repr(self):
        """文字列表現テスト"""
        config = {"algorithm": "ppo", "model_name": "test_model"}
        builder = ConfigBuilder(config)

        result = repr(builder)
        assert "ppo" in result
        assert "test_model" in result
        assert "ConfigBuilder" in result


class TestGetConfigValue:
    """get_config_value()メソッドの詳細テスト"""

    def test_top_level_priority(self):
        """トップレベルが最優先"""
        config = {
            "learning_rate": 0.001,
            "ppo_hyperparameters": {"learning_rate": 0.002},
        }
        builder = ConfigBuilder(config)

        result = builder.get_config_value("learning_rate", ["ppo_hyperparameters"])
        assert result == 0.001  # トップレベルが優先

    def test_section_search(self):
        """セクション検索"""
        config = {"ppo_hyperparameters": {"learning_rate": 0.002}}
        builder = ConfigBuilder(config)

        result = builder.get_config_value("learning_rate", ["ppo_hyperparameters"])
        assert result == 0.002

    def test_section_priority_order(self):
        """セクション検索の優先順序"""
        config = {"ppo": {"learning_rate": 0.003}, "sac": {"learning_rate": 0.004}}
        builder = ConfigBuilder(config)

        # 最初のセクションが優先
        result = builder.get_config_value("learning_rate", ["ppo", "sac"])
        assert result == 0.003

    def test_default_fallback(self):
        """デフォルト値へのフォールバック"""
        config = {}
        builder = ConfigBuilder(config)

        result = builder.get_config_value("learning_rate", ["ppo"], default=0.0003)
        assert result == 0.0003

    def test_none_sections(self):
        """sectionsがNoneの場合"""
        config = {"learning_rate": 0.001}
        builder = ConfigBuilder(config)

        result = builder.get_config_value("learning_rate", sections=None)
        assert result == 0.001

    def test_missing_key_no_default(self):
        """キーが見つからずデフォルトもない場合"""
        config = {}
        builder = ConfigBuilder(config)

        result = builder.get_config_value("unknown_key")
        assert result is None


class TestMemoryOptimizationConfig:
    """get_memory_optimization_config()テスト"""

    def test_with_values(self):
        """値が設定されている場合"""
        config = {"data_rows_limit": 10000, "max_features": 50}
        builder = ConfigBuilder(config)

        result = builder.get_memory_optimization_config()
        assert result["data_rows_limit"] == 10000
        assert result["max_features"] == 50

    def test_without_values(self):
        """値が設定されていない場合"""
        config = {}
        builder = ConfigBuilder(config)

        result = builder.get_memory_optimization_config()
        assert result["data_rows_limit"] is None
        assert result["max_features"] is None


class TestEnvironmentConfig:
    """get_environment_config()テスト"""

    def test_with_environment_section(self):
        """environmentセクションから取得"""
        config = {
            "environment": {
                "max_position_size": 2.0,
                "initial_balance": 500000,
                "transaction_cost": 0.001,
                "reward_scaling": 2.0,
            }
        }
        builder = ConfigBuilder(config)

        result = builder.get_environment_config()
        assert result["max_position_size"] == 2.0
        assert result["initial_balance"] == 500000
        assert result["transaction_cost"] == 0.001
        assert result["reward_scaling"] == 2.0

    def test_with_defaults(self):
        """デフォルト値が使われる場合"""
        config = {}
        builder = ConfigBuilder(config)

        result = builder.get_environment_config()
        # DEFAULT_PPO_CONFIGのデフォルト値が使われる
        assert "max_position_size" in result
        assert "initial_balance" in result
        assert "transaction_cost" in result
        assert "reward_scaling" in result


class TestPPOCoreConfig:
    """get_ppo_core_config()テスト"""

    def test_complete_config(self):
        """完全な設定"""
        config = {
            "ppo_hyperparameters": {
                "learning_rate": 0.007503,
                "n_steps": 2048,
                "batch_size": 256,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
            }
        }
        builder = ConfigBuilder(config)

        result = builder.get_ppo_core_config()
        assert result["learning_rate"] == 0.007503
        assert result["n_steps"] == 2048
        assert result["batch_size"] == 256
        assert result["ent_coef"] == 0.01

    def test_partial_config_with_defaults(self):
        """一部の設定のみ（デフォルト値使用）"""
        config = {"ppo_hyperparameters": {"learning_rate": 0.001}}
        builder = ConfigBuilder(config)

        result = builder.get_ppo_core_config()
        assert result["learning_rate"] == 0.001
        # 他はデフォルト値
        assert result["n_steps"] is not None
        assert result["batch_size"] is not None

    def test_optional_parameters(self):
        """オプションパラメータ"""
        config = {"ppo_hyperparameters": {"clip_range_vf": 0.3, "target_kl": 0.01}}
        builder = ConfigBuilder(config)

        result = builder.get_ppo_core_config()
        assert result["clip_range_vf"] == 0.3
        assert result["target_kl"] == 0.01


class TestSACCoreConfig:
    """get_sac_core_config()テスト"""

    def test_sac_config(self):
        """SAC設定の取得"""
        config = {
            "sac_hyperparameters": {
                "learning_rate": 0.0003,
                "buffer_size": 100000,
                "batch_size": 512,
                "tau": 0.01,
                "gamma": 0.95,
                "ent_coef": "auto",
                "target_entropy": "auto",
            }
        }
        builder = ConfigBuilder(config)

        result = builder.get_sac_core_config()
        assert result["learning_rate"] == 0.0003
        assert result["buffer_size"] == 100000
        assert result["batch_size"] == 512
        assert result["tau"] == 0.01
        assert result["ent_coef"] == "auto"
        assert result["target_entropy"] == "auto"

    def test_sac_defaults(self):
        """SACデフォルト値"""
        config = {}
        builder = ConfigBuilder(config)

        result = builder.get_sac_core_config()
        # Stable-Baselines3準拠のデフォルト
        assert result["learning_rate"] == 3e-4
        assert result["buffer_size"] == 50000
        assert result["ent_coef"] == "auto"
        assert result["target_entropy"] == "auto"


class TestFeatureConfig:
    """get_feature_config()テスト"""

    def test_with_values(self):
        """値が設定されている場合"""
        config = {
            "feature_set": "curated",
            "custom_features": ["rsi", "macd"],
            "feature_config_path": "/path/to/config",
            "max_features": 60,
        }
        builder = ConfigBuilder(config)

        result = builder.get_feature_config()
        assert result["feature_set"] == "curated"
        assert result["custom_features"] == ["rsi", "macd"]
        assert result["feature_config_path"] == "/path/to/config"
        assert result["max_features"] == 60

    def test_defaults(self):
        """デフォルト値"""
        config = {}
        builder = ConfigBuilder(config)

        result = builder.get_feature_config()
        assert result["feature_set"] == "curated"
        assert result["custom_features"] is None
        assert result["feature_config_path"] is None
        assert result["max_features"] is None


class TestConfigManagerIntegration:
    """ConfigManager統合テスト"""

    def test_config_manager_lazy_initialization(self):
        """ConfigManagerの遅延初期化"""
        config = {"algorithm": "ppo"}
        builder = ConfigBuilder(config)

        # 最初はNone
        assert builder._config_manager is None

        # アクセス時に初期化
        manager = builder.config_manager
        assert manager is not None

        # 2回目は同じインスタンス
        manager2 = builder.config_manager
        assert manager is manager2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
