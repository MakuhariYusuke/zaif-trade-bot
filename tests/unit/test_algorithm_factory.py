"""
Unit tests for AlgorithmFactory.

Tests registration, creation, error handling, and utility methods.
"""

import pytest
from typing import Dict, Any

from ztb.training.algorithms.algorithm_factory import AlgorithmFactory
from ztb.training.algorithms.base_algorithm import BaseRLAlgorithm
from ztb.training.algorithms.ppo import PPOAlgorithm


class DummyAlgorithm(BaseRLAlgorithm):
    """テスト用のダミーアルゴリズム"""
    
    @property
    def algorithm_name(self) -> str:
        return "dummy"
    
    def create_model(self, env, config, tensorboard_log=None):
        return None
    
    def train(self, model, total_timesteps, callback=None, **kwargs):
        return model
    
    def get_default_config(self) -> Dict[str, Any]:
        return {"test": "config"}


class InvalidAlgorithm:
    """BaseRLAlgorithmを継承していない不正なクラス"""
    pass


class TestAlgorithmFactoryRegistration:
    """アルゴリズム登録のテスト"""
    
    def test_register_valid_algorithm(self):
        """有効なアルゴリズムの登録"""
        # PPOは既に登録されているのでダミーを使用
        AlgorithmFactory.register("test_dummy", DummyAlgorithm)
        
        assert AlgorithmFactory.is_registered("test_dummy")
        
        # クリーンアップ
        AlgorithmFactory.unregister("test_dummy")
    
    def test_register_invalid_algorithm(self):
        """無効なアルゴリズムの登録（エラー）"""
        with pytest.raises(TypeError) as exc_info:
            AlgorithmFactory.register("invalid", InvalidAlgorithm)
        
        assert "must be a subclass of BaseRLAlgorithm" in str(exc_info.value)
    
    def test_register_case_insensitive(self):
        """大文字小文字を区別しない登録"""
        AlgorithmFactory.register("TEST_CASE", DummyAlgorithm)
        
        assert AlgorithmFactory.is_registered("test_case")
        assert AlgorithmFactory.is_registered("TEST_CASE")
        
        # クリーンアップ
        AlgorithmFactory.unregister("test_case")


class TestAlgorithmFactoryCreation:
    """アルゴリズム作成のテスト"""
    
    def test_create_ppo(self):
        """PPOアルゴリズムの作成"""
        ppo = AlgorithmFactory.create("ppo")
        
        assert isinstance(ppo, PPOAlgorithm)
        assert ppo.algorithm_name == "ppo"
    
    def test_create_sac(self):
        """SACアルゴリズムの作成"""
        from ztb.training.algorithms.sac import SACAlgorithm
        
        sac = AlgorithmFactory.create("sac")
        
        assert isinstance(sac, SACAlgorithm)
        assert sac.algorithm_name == "sac"
    
    def test_create_case_insensitive(self):
        """大文字小文字を区別しない作成"""
        ppo_lower = AlgorithmFactory.create("ppo")
        ppo_upper = AlgorithmFactory.create("PPO")
        ppo_mixed = AlgorithmFactory.create("PpO")
        
        assert ppo_lower.algorithm_name == "ppo"
        assert ppo_upper.algorithm_name == "ppo"
        assert ppo_mixed.algorithm_name == "ppo"
    
    def test_create_unknown_algorithm(self):
        """未登録のアルゴリズム作成（エラー）"""
        with pytest.raises(ValueError) as exc_info:
            AlgorithmFactory.create("unknown_algo")
        
        assert "Unknown algorithm: 'unknown_algo'" in str(exc_info.value)
        assert "Available algorithms:" in str(exc_info.value)
    
    def test_create_with_kwargs(self):
        """コンストラクタ引数付きで作成"""
        # PPOAlgorithmはuse_auto_haltを受け取る
        ppo = AlgorithmFactory.create("ppo", use_auto_halt=True)
        
        assert isinstance(ppo, PPOAlgorithm)
        # use_auto_haltフラグが設定されているか確認
        assert ppo._use_auto_halt is True


class TestAlgorithmFactoryUtilities:
    """ユーティリティメソッドのテスト"""
    
    def test_list_algorithms(self):
        """登録済みアルゴリズムのリスト取得"""
        algorithms = AlgorithmFactory.list_algorithms()
        
        assert isinstance(algorithms, list)
        assert "ppo" in algorithms
        assert "sac" in algorithms
        assert algorithms == sorted(algorithms)  # アルファベット順
    
    def test_is_registered_true(self):
        """登録済みアルゴリズムの確認"""
        assert AlgorithmFactory.is_registered("ppo") is True
        assert AlgorithmFactory.is_registered("sac") is True
    
    def test_is_registered_false(self):
        """未登録アルゴリズムの確認"""
        assert AlgorithmFactory.is_registered("td3") is False
        assert AlgorithmFactory.is_registered("unknown") is False
    
    def test_is_registered_case_insensitive(self):
        """大文字小文字を区別しない確認"""
        assert AlgorithmFactory.is_registered("ppo") is True
        assert AlgorithmFactory.is_registered("PPO") is True
        assert AlgorithmFactory.is_registered("PpO") is True
    
    def test_get_info(self):
        """アルゴリズム情報の取得"""
        info = AlgorithmFactory.get_info()
        
        assert isinstance(info, dict)
        assert "count" in info
        assert "algorithms" in info
        assert "registry" in info
        
        assert info["count"] >= 1  # 少なくともPPO
        assert "ppo" in info["algorithms"]
        assert "ppo" in info["registry"]
        assert info["registry"]["ppo"] == "PPOAlgorithm"
    
    def test_unregister_existing(self):
        """既存アルゴリズムの登録解除"""
        # テスト用アルゴリズムを登録
        AlgorithmFactory.register("temp_test", DummyAlgorithm)
        assert AlgorithmFactory.is_registered("temp_test")
        
        # 登録解除
        result = AlgorithmFactory.unregister("temp_test")
        assert result is True
        assert not AlgorithmFactory.is_registered("temp_test")
    
    def test_unregister_nonexistent(self):
        """存在しないアルゴリズムの登録解除"""
        result = AlgorithmFactory.unregister("nonexistent_algo")
        assert result is False
    
    def test_unregister_case_insensitive(self):
        """大文字小文字を区別しない登録解除"""
        AlgorithmFactory.register("TEMP_CASE", DummyAlgorithm)
        
        result = AlgorithmFactory.unregister("temp_case")
        assert result is True
        assert not AlgorithmFactory.is_registered("TEMP_CASE")


class TestAlgorithmFactoryMultipleAlgorithms:
    """複数アルゴリズムの同時管理テスト"""
    
    def test_multiple_registrations(self):
        """複数のアルゴリズムを登録"""
        # テスト用アルゴリズムを複数登録
        AlgorithmFactory.register("algo1", DummyAlgorithm)
        AlgorithmFactory.register("algo2", DummyAlgorithm)
        AlgorithmFactory.register("algo3", DummyAlgorithm)
        
        algorithms = AlgorithmFactory.list_algorithms()
        assert "algo1" in algorithms
        assert "algo2" in algorithms
        assert "algo3" in algorithms
        
        # クリーンアップ
        AlgorithmFactory.unregister("algo1")
        AlgorithmFactory.unregister("algo2")
        AlgorithmFactory.unregister("algo3")
    
    def test_registry_isolation(self):
        """レジストリの独立性"""
        # 新しいアルゴリズムを登録
        AlgorithmFactory.register("isolated_test", DummyAlgorithm)
        
        info_before = AlgorithmFactory.get_info()
        count_before = info_before["count"]
        
        # 別のアルゴリズムを作成（レジストリに影響しない）
        algo = AlgorithmFactory.create("ppo")
        
        info_after = AlgorithmFactory.get_info()
        count_after = info_after["count"]
        
        assert count_before == count_after
        
        # クリーンアップ
        AlgorithmFactory.unregister("isolated_test")


class TestAlgorithmFactoryEdgeCases:
    """エッジケースのテスト"""
    
    def test_empty_algorithm_name(self):
        """空のアルゴリズム名"""
        with pytest.raises(ValueError):
            AlgorithmFactory.create("")
    
    def test_none_algorithm_name(self):
        """Noneのアルゴリズム名"""
        with pytest.raises((ValueError, AttributeError)):
            AlgorithmFactory.create(None)
    
    def test_overwrite_registration(self):
        """同じ名前で再登録（上書き）"""
        class DummyV1(BaseRLAlgorithm):
            @property
            def algorithm_name(self):
                return "overwrite_test"
            def create_model(self, *args, **kwargs):
                return "v1"
            def train(self, *args, **kwargs):
                return None
            def get_default_config(self):
                return {"version": 1}
        
        class DummyV2(BaseRLAlgorithm):
            @property
            def algorithm_name(self):
                return "overwrite_test"
            def create_model(self, *args, **kwargs):
                return "v2"
            def train(self, *args, **kwargs):
                return None
            def get_default_config(self):
                return {"version": 2}
        
        # V1を登録
        AlgorithmFactory.register("overwrite_test", DummyV1)
        algo_v1 = AlgorithmFactory.create("overwrite_test")
        assert algo_v1.get_default_config()["version"] == 1
        
        # V2で上書き
        AlgorithmFactory.register("overwrite_test", DummyV2)
        algo_v2 = AlgorithmFactory.create("overwrite_test")
        assert algo_v2.get_default_config()["version"] == 2
        
        # クリーンアップ
        AlgorithmFactory.unregister("overwrite_test")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
