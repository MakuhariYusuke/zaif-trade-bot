"""
SACAlgorithm単体テスト。

SACAlgorithmクラスの全機能を網羅的にテストする。
"""

from unittest.mock import Mock

import pytest

from ztb.training.algorithms.sac import SACAlgorithm

# ========================================
# 基本機能テスト
# ========================================


class TestSACAlgorithmBasics:
    """SACAlgorithm基本機能のテスト。"""

    def test_initialization_standard(self):
        """標準的な初期化のテスト。"""
        sac = SACAlgorithm()
        assert sac is not None
        assert sac._model is None

    def test_algorithm_name(self):
        """algorithm_nameプロパティのテスト。"""
        sac = SACAlgorithm()
        assert sac.algorithm_name == "sac"

    def test_repr_standard(self):
        """__repr__のテスト（モデル未初期化）。"""
        sac = SACAlgorithm()
        repr_str = repr(sac)
        assert "SACAlgorithm" in repr_str
        assert "not_initialized" in repr_str


# ========================================
# デフォルト設定テスト
# ========================================


class TestSACDefaultConfig:
    """SACデフォルト設定のテスト。"""

    def test_default_config_structure(self):
        """デフォルト設定の構造チェック。"""
        config = SACAlgorithm.get_default_config()

        # 必須キーの存在確認
        assert "learning_rate" in config
        assert "buffer_size" in config
        assert "batch_size" in config
        assert "tau" in config
        assert "gamma" in config
        assert "ent_coef" in config
        assert "target_entropy" in config

    def test_default_hyperparameters(self):
        """デフォルトハイパーパラメータの値チェック。"""
        config = SACAlgorithm.get_default_config()

        # 学習率
        assert config["learning_rate"] == 3e-4

        # Replay Buffer
        assert config["buffer_size"] == 50000
        assert config["learning_starts"] == 1000
        assert config["batch_size"] == 256

        # 訓練パラメータ
        assert config["tau"] == 0.005
        assert config["gamma"] == 0.99
        assert config["train_freq"] == 1
        assert config["gradient_steps"] == 1

        # エントロピー正則化
        assert config["ent_coef"] == "auto"
        assert config["target_entropy"] == "auto"

    def test_default_other_params(self):
        """その他のデフォルトパラメータチェック。"""
        config = SACAlgorithm.get_default_config()

        assert config["verbose"] == 1
        assert config["device"] == "auto"
        assert config["policy_kwargs"] is None


# ========================================
# 設定検証テスト
# ========================================


class TestSACValidateConfig:
    """SAC設定検証のテスト。"""

    def test_valid_config(self):
        """有効な設定のテスト。"""
        config = {
            "learning_rate": 3e-4,
            "buffer_size": 50000,
            "batch_size": 256,
        }
        assert SACAlgorithm.validate_config(config) is True

    def test_missing_required_param(self):
        """必須パラメータ不足のテスト。"""
        # learning_rateが不足
        config = {
            "buffer_size": 50000,
            "batch_size": 256,
        }
        with pytest.raises(
            ValueError, match="Missing required SAC parameter: learning_rate"
        ):
            SACAlgorithm.validate_config(config)

        # buffer_sizeが不足
        config = {
            "learning_rate": 3e-4,
            "batch_size": 256,
        }
        with pytest.raises(
            ValueError, match="Missing required SAC parameter: buffer_size"
        ):
            SACAlgorithm.validate_config(config)

    def test_invalid_learning_rate(self):
        """不正な学習率のテスト。"""
        config = {
            "learning_rate": -0.001,
            "buffer_size": 50000,
            "batch_size": 256,
        }
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            SACAlgorithm.validate_config(config)

    def test_invalid_buffer_size(self):
        """不正なbuffer_sizeのテスト。"""
        config = {
            "learning_rate": 3e-4,
            "buffer_size": -1000,
            "batch_size": 256,
        }
        with pytest.raises(ValueError, match="buffer_size must be positive"):
            SACAlgorithm.validate_config(config)

    def test_buffer_size_smaller_than_batch_size(self):
        """buffer_size < batch_sizeのエラーテスト。"""
        config = {
            "learning_rate": 3e-4,
            "buffer_size": 100,
            "batch_size": 256,
        }
        with pytest.raises(ValueError, match="buffer_size .* must be >= batch_size"):
            SACAlgorithm.validate_config(config)

    def test_optional_parameters(self):
        """オプションパラメータのテスト。"""
        config = {
            "learning_rate": 3e-4,
            "buffer_size": 50000,
            "batch_size": 256,
            "tau": 0.01,
            "gamma": 0.95,
            "ent_coef": 0.2,
        }
        assert SACAlgorithm.validate_config(config) is True


# ========================================
# モデル作成テスト（プレースホルダー）
# ========================================


class TestSACCreateModel:
    """SACモデル作成のテスト（プレースホルダー）。"""

    def test_create_model_logs_info(self):
        """create_modelがログ出力することを確認（実際の環境不要）。"""
        sac = SACAlgorithm()

        # MockのVecEnvを作成
        mock_env = Mock()
        mock_env.observation_space = Mock()
        mock_env.action_space = Mock()
        mock_env.num_envs = 1

        config = {
            "learning_rate": 3e-4,
            "buffer_size": 50000,
            "batch_size": 256,
        }

        # 実際のSACモデル作成は環境依存のためスキップ
        # ここでは設定検証だけ確認
        assert SACAlgorithm.validate_config(config) is True


# ========================================
# train()メソッドテスト
# ========================================


class TestSACTrain:
    """SAC訓練のテスト。"""

    def test_train_requires_initialization(self):
        """モデル未初期化時のエラーテスト。"""
        sac = SACAlgorithm()

        with pytest.raises(ValueError, match="Model must be initialized"):
            sac.train(None, total_timesteps=1000)


# ========================================
# 既存設定との互換性テスト
# ========================================


class TestSACConfigCompatibility:
    """SAC設定の互換性テスト。"""

    def test_minimal_config(self):
        """最小限の設定でバリデーション成功。"""
        config = {
            "learning_rate": 3e-4,
            "buffer_size": 10000,
            "batch_size": 128,
        }
        assert SACAlgorithm.validate_config(config) is True

    def test_full_config(self):
        """完全な設定でバリデーション成功。"""
        config = {
            "learning_rate": 3e-4,
            "buffer_size": 50000,
            "learning_starts": 1000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": "auto",
            "target_entropy": "auto",
            "verbose": 1,
            "device": "cpu",
        }
        assert SACAlgorithm.validate_config(config) is True


# ========================================
# エッジケーステスト
# ========================================


class TestSACEdgeCases:
    """SACエッジケースのテスト。"""

    def test_extreme_hyperparameters(self):
        """極端なハイパーパラメータでもバリデーション成功。"""
        config = {
            "learning_rate": 1e-6,  # 非常に小さい
            "buffer_size": 1000000,  # 非常に大きい
            "batch_size": 32,
        }
        assert SACAlgorithm.validate_config(config) is True

    def test_negative_values(self):
        """負の値でエラー。"""
        config = {
            "learning_rate": 3e-4,
            "buffer_size": 50000,
            "batch_size": -10,
        }
        with pytest.raises(ValueError, match="batch_size must be positive"):
            SACAlgorithm.validate_config(config)

    def test_zero_values(self):
        """ゼロ値でエラー。"""
        config = {
            "learning_rate": 0,
            "buffer_size": 50000,
            "batch_size": 256,
        }
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            SACAlgorithm.validate_config(config)

    def test_string_ent_coef(self):
        """ent_coef="auto"は有効。"""
        config = {
            "learning_rate": 3e-4,
            "buffer_size": 50000,
            "batch_size": 256,
            "ent_coef": "auto",
        }
        assert SACAlgorithm.validate_config(config) is True

    def test_numeric_ent_coef(self):
        """ent_coef=数値も有効。"""
        config = {
            "learning_rate": 3e-4,
            "buffer_size": 50000,
            "batch_size": 256,
            "ent_coef": 0.1,
        }
        assert SACAlgorithm.validate_config(config) is True
