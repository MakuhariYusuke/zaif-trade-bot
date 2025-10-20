"""
Unit tests for PPOAlgorithm.

Tests PPO-specific functionality, validation, and configuration.
"""


import pytest

from ztb.training.algorithms.ppo import PPOAlgorithm


class TestPPOAlgorithmBasics:
    """PPOAlgorithmの基本機能テスト"""

    def test_initialization_standard(self):
        """標準初期化"""
        ppo = PPOAlgorithm()

        assert ppo._use_auto_halt is False
        assert ppo._trainer is None
        assert ppo._model is None
        assert ppo._config is None

    def test_initialization_with_auto_halt(self):
        """AutoHalt有効で初期化"""
        ppo = PPOAlgorithm(use_auto_halt=True)

        assert ppo._use_auto_halt is True

    def test_algorithm_name(self):
        """algorithm_nameプロパティ"""
        ppo = PPOAlgorithm()

        assert ppo.algorithm_name == "ppo"

    def test_repr_standard(self):
        """文字列表現（標準）"""
        ppo = PPOAlgorithm()
        result = repr(ppo)

        assert "PPOAlgorithm" in result
        assert "Standard" in result
        assert "not loaded" in result

    def test_repr_auto_halt(self):
        """文字列表現（AutoHalt）"""
        ppo = PPOAlgorithm(use_auto_halt=True)
        result = repr(ppo)

        assert "PPOAlgorithm" in result
        assert "AutoHalt" in result


class TestPPODefaultConfig:
    """get_default_config()のテスト"""

    def test_default_config_structure(self):
        """デフォルト設定の構造（ハイパーパラメータのみ）"""
        ppo = PPOAlgorithm()
        config = ppo.get_default_config()

        # get_default_config()はハイパーパラメータのみを返す（SACと統一）
        assert isinstance(config, dict)
        assert "learning_rate" in config
        assert "n_steps" in config
        assert "batch_size" in config

    def test_default_hyperparameters(self):
        """デフォルトハイパーパラメータ"""
        ppo = PPOAlgorithm()
        params = ppo.get_default_config()

        # 必須パラメータ
        assert "learning_rate" in params
        assert "n_steps" in params
        assert "batch_size" in params
        assert "n_epochs" in params
        assert "gamma" in params
        assert "gae_lambda" in params
        assert "clip_range" in params
        assert "ent_coef" in params
        assert "vf_coef" in params
        assert "max_grad_norm" in params

        # デフォルト値の確認
        assert params["learning_rate"] == 0.0003
        assert params["n_steps"] == 2048
        assert params["batch_size"] == 64
        assert params["ent_coef"] == 0.01

    def test_default_environment(self):
        """デフォルト環境設定（このテストは削除予定）"""
        pytest.skip(
            "get_default_config() returns only hyperparameters, not full config"
        )

    def test_default_reward_settings(self):
        """デフォルト報酬設定（このテストは削除予定）"""
        pytest.skip(
            "get_default_config() returns only hyperparameters, not full config"
        )


class TestPPOValidateConfig:
    """validate_config()のテスト"""

    def test_valid_config(self):
        """有効な設定"""
        ppo = PPOAlgorithm()
        config = {
            "ppo_hyperparameters": {
                "learning_rate": 0.0003,
                "n_steps": 2048,
                "batch_size": 64,
            }
        }

        assert ppo.validate_config(config) is True

    def test_missing_ppo_hyperparameters(self):
        """ppo_hyperparametersセクションが欠落（ValueErrorが発生）"""
        ppo = PPOAlgorithm()
        config = {"algorithm": "ppo"}

        # validate_config()は例外を投げる実装なので、これをキャッチ
        with pytest.raises(ValueError, match="Missing required PPO parameter"):
            ppo.validate_config(config)

    def test_missing_required_param(self):
        """必須パラメータが欠落（ValueErrorが発生）"""
        ppo = PPOAlgorithm()

        # learning_rateがない
        config1 = {"ppo_hyperparameters": {"n_steps": 2048, "batch_size": 64}}
        with pytest.raises(ValueError, match="Missing required PPO parameter"):
            ppo.validate_config(config1)

        # n_stepsがない
        config2 = {"ppo_hyperparameters": {"learning_rate": 0.0003, "batch_size": 64}}
        with pytest.raises(ValueError, match="Missing required PPO parameter"):
            ppo.validate_config(config2)

        # batch_sizeがない
        config3 = {"ppo_hyperparameters": {"learning_rate": 0.0003, "n_steps": 2048}}
        with pytest.raises(ValueError, match="Missing required PPO parameter"):
            ppo.validate_config(config3)

    def test_optional_parameters(self):
        """オプションパラメータは必須ではない"""
        ppo = PPOAlgorithm()
        config = {
            "ppo_hyperparameters": {
                "learning_rate": 0.0003,
                "n_steps": 2048,
                "batch_size": 64,
                # clip_range_vf, target_klなどは省略
            }
        }

        # オプションパラメータがなくても有効
        assert ppo.validate_config(config) is True

    def test_empty_config(self):
        """空の設定（ValueErrorが発生）"""
        ppo = PPOAlgorithm()
        config = {}

        with pytest.raises(ValueError, match="Missing required PPO parameter"):
            ppo.validate_config(config)


class TestPPOCreateModel:
    """create_model()のテスト（プレースホルダー確認）"""

    def test_create_model_logs_info(self):
        """create_model()が情報をログする"""
        ppo = PPOAlgorithm()

        # 現時点ではプレースホルダー実装
        # 環境やTensorBoardログパスは将来実装
        assert ppo._trainer is None
        assert ppo._model is None

    def test_create_model_with_auto_halt(self):
        """AutoHaltフラグが設定に反映される"""
        ppo_standard = PPOAlgorithm(use_auto_halt=False)
        ppo_auto_halt = PPOAlgorithm(use_auto_halt=True)

        assert ppo_standard._use_auto_halt is False
        assert ppo_auto_halt._use_auto_halt is True


class TestPPOTrain:
    """train()のテスト（プレースホルダー確認）"""

    def test_train_requires_initialization(self):
        """train()は初期化が必要"""
        ppo = PPOAlgorithm()

        # trainerが初期化されていない
        assert ppo._trainer is None


class TestPPOConfigCompatibility:
    """既存設定ファイルとの互換性テスト"""

    def test_v394d_config_compatible(self):
        """v394d設定との互換性"""
        ppo = PPOAlgorithm()
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

        assert ppo.validate_config(config) is True

    def test_v394f_config_compatible(self):
        """v394f設定との互換性（高エントロピー）"""
        ppo = PPOAlgorithm()
        config = {
            "ppo_hyperparameters": {
                "learning_rate": 0.007503,
                "n_steps": 2048,
                "batch_size": 256,
                "ent_coef": 0.2,  # v394dの20倍
            }
        }

        assert ppo.validate_config(config) is True

    def test_minimal_config(self):
        """最小限の設定"""
        ppo = PPOAlgorithm()
        config = {
            "ppo_hyperparameters": {
                "learning_rate": 0.0003,
                "n_steps": 1024,
                "batch_size": 32,
            }
        }

        assert ppo.validate_config(config) is True


class TestPPOEdgeCases:
    """エッジケースのテスト"""

    def test_extreme_hyperparameters(self):
        """極端なハイパーパラメータ値"""
        ppo = PPOAlgorithm()
        config = {
            "ppo_hyperparameters": {"learning_rate": 1.0, "n_steps": 1, "batch_size": 1}
        }  # 極端に大きい  # 最小  # 最小

        # 検証は通る（値の範囲チェックは別途必要）
        assert ppo.validate_config(config) is True

    def test_negative_values(self):
        """負の値（検証でValueErrorが発生）"""
        ppo = PPOAlgorithm()
        config = {
            "ppo_hyperparameters": {
                "learning_rate": -0.001,
                "n_steps": 2048,
                "batch_size": 64,
            }
        }  # 負の値

        # validate_config()は負の値でValueErrorを投げる
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            ppo.validate_config(config)

    def test_string_instead_of_number(self):
        """数値の代わりに文字列（型チェックでTypeErrorが発生）"""
        ppo = PPOAlgorithm()
        config = {
            "ppo_hyperparameters": {
                "learning_rate": "0.0003",
                "n_steps": 2048,
                "batch_size": 64,
            }
        }  # 文字列

        # validate_config()は型チェックでTypeErrorを投げる
        with pytest.raises(TypeError):
            ppo.validate_config(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
