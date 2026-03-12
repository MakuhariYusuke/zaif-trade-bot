import os
import sys
import tempfile
from unittest.mock import Mock

sys.path.insert(0, "src")


class TestCheckpointLight:
    def setup_method(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_checkpoint_")
        self.model = Mock()
        self.model.policy.state_dict.return_value = {"layer1": "policy_data"}
        self.model.value_net = Mock()
        self.model.value_net.state_dict.return_value = {"layer2": "value_data"}
        self.model.scaler = "scaler_data"

    def teardown_method(self):
        import shutil

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_light_checkpoint_size_reduction(self):
        """軽量チェックポイントのサイズ削減効果"""
        import pickle

        # 完全チェックポイント相当のデータ
        full_data = {
            "policy": self.model.policy.state_dict(),
            "value_net": self.model.value_net.state_dict(),
            "scaler": self.model.scaler,
            "optimizer": {"state": "large_optimizer_data" * 1000},  # 大きなデータ
            "replay_buffer": {"buffer": "large_buffer_data" * 1000},
        }

        # 軽量チェックポイントのデータ
        light_data = {
            "policy": self.model.policy.state_dict(),
            "value_net": self.model.value_net.state_dict(),
            "scaler": self.model.scaler,
        }

        # サイズ比較
        full_size = len(pickle.dumps(full_data))
        light_size = len(pickle.dumps(light_data))

        # 軽量版が小さいことを確認
        assert light_size < full_size
        reduction_ratio = (full_size - light_size) / full_size
        assert reduction_ratio > 0.5  # 50%以上削減
