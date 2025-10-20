"""
Unit tests for SAC algorithm transfer learning functionality.
"""

from unittest.mock import Mock, patch

import pytest

from ztb.training.algorithms.sac.sac_algorithm import SACAlgorithm


class TestSACTransferLearning:
    """Test cases for SAC transfer learning functionality."""

    def test_transfer_learning_config_validation(self):
        """Test transfer learning configuration validation."""
        algorithm = SACAlgorithm()

        # 有効な転移学習設定
        config = algorithm.get_default_config()
        config.update(
            {
                "transfer_learning_enabled": True,
                "pretrained_model_path": "/path/to/model.zip",
                "freeze_layers": 2,
                "fine_tune_learning_rate": 1e-4,
            }
        )
        assert algorithm.validate_config(config)

        # 無効な設定：pretrained_model_pathなし
        config_invalid = algorithm.get_default_config()
        config_invalid["transfer_learning_enabled"] = True
        with pytest.raises(ValueError, match="pretrained_model_path is not specified"):
            algorithm.validate_config(config_invalid)

        # 無効な設定：負のfreeze_layers
        config_invalid = algorithm.get_default_config()
        config_invalid.update(
            {
                "transfer_learning_enabled": True,
                "pretrained_model_path": "/path/to/model.zip",
                "freeze_layers": -1,
            }
        )
        with pytest.raises(ValueError, match="freeze_layers must be non-negative"):
            algorithm.validate_config(config_invalid)

        # 無効な設定：MLPで過大なfreeze_layers
        config_invalid = algorithm.get_default_config()
        config_invalid.update(
            {
                "transfer_learning_enabled": True,
                "pretrained_model_path": "/path/to/model.zip",
                "freeze_layers": 20,  # MLPの最大層数を超える
            }
        )
        with pytest.raises(ValueError, match="freeze_layers .* too large for MLP"):
            algorithm.validate_config(config_invalid)

        # 無効な設定：LSTM/Transformerでfreeze_layers > 1.0
        config_invalid = algorithm.get_default_config()
        config_invalid.update(
            {
                "transfer_learning_enabled": True,
                "pretrained_model_path": "/path/to/model.zip",
                "network_type": "lstm",
                "freeze_layers": 1.5,  # 1.0を超える
            }
        )
        with pytest.raises(ValueError, match="freeze_layers .* must be <= 1.0"):
            algorithm.validate_config(config_invalid)

    @patch("ztb.training.algorithms.sac.sac_algorithm.SAC.load")
    def test_apply_transfer_learning_mlp(self, mock_load):
        """Test transfer learning application for MLP network."""
        # モックモデルの設定
        mock_pretrained = Mock()
        mock_pretrained.device = "cpu"
        mock_pretrained.policy = Mock()
        mock_load.return_value = mock_pretrained

        algorithm = SACAlgorithm()
        model = Mock()
        model.device = "cpu"
        policy = Mock()

        # ActorとCriticのネットワークをモック
        actor_net = Mock()
        actor_net.children.return_value = [Mock(), Mock()]  # 2層
        policy.actor = actor_net

        critic_net = Mock()
        critic_net.children.return_value = [Mock(), Mock()]  # 2層
        policy.critic = critic_net

        model.policy = policy

        config = {
            "transfer_learning_enabled": True,
            "pretrained_model_path": "/path/to/model.zip",
            "network_type": "mlp",
            "freeze_layers": 2,
            "fine_tune_learning_rate": 1e-4,
        }

        # 転移学習の適用
        algorithm._apply_transfer_learning(model, config)

        # 事前学習済みモデルの読み込みが呼ばれたことを確認
        mock_load.assert_called_once_with("/path/to/model.zip", device="cpu")

    @patch("ztb.training.algorithms.sac.sac_algorithm.SAC.load")
    def test_apply_transfer_learning_lstm_validation(self, mock_load):
        """Test transfer learning validation for LSTM network."""
        # LSTMポリシーのモック
        from ztb.training.models.advanced_networks import LSTMPolicy

        mock_pretrained = Mock()
        mock_pretrained.device = "cpu"
        mock_pretrained.policy = Mock(spec=LSTMPolicy)
        mock_load.return_value = mock_pretrained

        algorithm = SACAlgorithm()
        model = Mock()
        model.device = "cpu"
        model.policy = Mock(spec=LSTMPolicy)

        config = {
            "transfer_learning_enabled": True,
            "pretrained_model_path": "/path/to/model.zip",
            "network_type": "lstm",
            "freeze_layers": 0.5,  # 50%凍結
        }

        # 転移学習の適用（検証成功）
        algorithm._apply_transfer_learning(model, config)

    @patch("ztb.training.algorithms.sac.sac_algorithm.SAC.load")
    def test_apply_transfer_learning_network_mismatch(self, mock_load):
        """Test transfer learning with network type mismatch."""
        # MLPポリシーのモック（LSTMが必要な場合）
        mock_pretrained = Mock()
        mock_pretrained.device = "cpu"
        mock_pretrained.policy = Mock()  # MLPポリシー
        mock_load.return_value = mock_pretrained

        algorithm = SACAlgorithm()
        model = Mock()
        model.device = "cpu"
        model.policy = Mock()

        config = {
            "transfer_learning_enabled": True,
            "pretrained_model_path": "/path/to/model.zip",
            "network_type": "lstm",  # LSTMが必要
        }

        # ネットワークタイプの不一致でエラー
        with pytest.raises(ValueError, match="Network type mismatch"):
            algorithm._apply_transfer_learning(model, config)

    @patch("torch.nn.Linear")
    def test_freeze_mlp_layers(self, mock_linear_class):
        """Test MLP layer freezing functionality."""
        # isinstanceチェックをモック
        mock_linear_class.__instancecheck__ = lambda self, obj: obj in [
            layer1,
            layer2,
            layer3,
        ]

        algorithm = SACAlgorithm()
        model = Mock()
        policy = Mock()

        # ActorとCriticのモックネットワーク
        actor_net = Mock()
        critic_net = Mock()

        # Linear層のモック
        layer1 = Mock()
        layer1.parameters.return_value = [Mock()]
        layer2 = Mock()
        layer2.parameters.return_value = [Mock()]
        layer3 = Mock()  # 凍結対象外
        layer3.parameters.return_value = [Mock()]

        # modules()が層を返すようにモック
        actor_net.modules.return_value = [actor_net, layer1, layer2, layer3]
        critic_net.modules.return_value = [critic_net, layer1, layer2]

        policy.actor = actor_net
        policy.critic = critic_net
        model.policy = policy

        # 2層を凍結
        algorithm._freeze_mlp_layers(model, 2)

        # 最初の2つのLinear層のパラメータが凍結されたことを確認
        layer1.parameters.return_value[0].requires_grad_.assert_called_with(False)
        layer2.parameters.return_value[0].requires_grad_.assert_called_with(False)
        # 3番目の層は凍結対象外
        layer3.parameters.return_value[0].requires_grad_.assert_not_called()

    def test_freeze_advanced_layers_lstm(self):
        """Test LSTM layer freezing functionality."""
        algorithm = SACAlgorithm()
        model = Mock()
        policy = Mock()
        extractor = Mock()

        # LSTM層のモック
        lstm_net = Mock()
        lstm_layer1 = Mock()
        lstm_layer1.parameters.return_value = [Mock()]
        lstm_layer2 = Mock()
        lstm_layer2.parameters.return_value = [Mock()]
        lstm_layer3 = Mock()
        lstm_layer3.parameters.return_value = [Mock()]

        # children()が層のリストを返すようにモック
        lstm_net.children.return_value = [lstm_layer1, lstm_layer2, lstm_layer3]
        extractor.lstm = lstm_net
        policy.features_extractor = extractor
        model.policy = policy

        # 50%を凍結（1層）
        algorithm._freeze_layers(model, 0.5, {"network_type": "lstm"})

        # 最初の層のみ凍結
        lstm_layer1.parameters.return_value[0].requires_grad_.assert_called_with(False)
        lstm_layer2.parameters.return_value[0].requires_grad_.assert_not_called()
        lstm_layer3.parameters.return_value[0].requires_grad_.assert_not_called()

    def test_set_fine_tune_learning_rate(self):
        """Test fine-tuning learning rate setting."""
        algorithm = SACAlgorithm()
        model = Mock()

        # オプティマイザのモック
        optimizer = Mock()
        param_group = {"lr": 1e-3}
        optimizer.param_groups = [param_group]
        model.policy_optimizer = optimizer

        # 学習率を設定
        algorithm._set_fine_tune_learning_rate(model, 1e-4)

        # 学習率が更新されたことを確認
        assert param_group["lr"] == 1e-4

    @patch("ztb.training.algorithms.sac.sac_algorithm.SAC.load")
    def test_transfer_learning_disabled_by_default(self, mock_load):
        """Test that transfer learning is disabled by default."""
        algorithm = SACAlgorithm()
        config = algorithm.get_default_config()

        # デフォルトで無効
        assert config["transfer_learning_enabled"] is False
        assert config["pretrained_model_path"] is None

        # 転移学習が無効な場合、loadが呼ばれない
        model = Mock()
        algorithm._apply_transfer_learning(model, config)
        mock_load.assert_not_called()
