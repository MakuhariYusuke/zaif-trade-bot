"""
アルゴリズム切り替え統合テスト。

PPOとSACをAlgorithmFactory経由で切り替えられることを検証する。
実際の訓練環境は使用せず、モックで統合性を確認する。
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from ztb.training.algorithms import AlgorithmFactory
from ztb.training.algorithms.ppo import PPOAlgorithm
from ztb.training.algorithms.sac import SACAlgorithm


# ========================================
# アルゴリズム登録確認テスト
# ========================================

class TestAlgorithmRegistration:
    """アルゴリズムが正しく登録されているかテスト。"""
    
    def test_ppo_is_registered(self):
        """PPOが登録されていることを確認。"""
        assert AlgorithmFactory.is_registered("ppo")
        assert AlgorithmFactory.is_registered("PPO")
    
    def test_sac_is_registered(self):
        """SACが登録されていることを確認。"""
        assert AlgorithmFactory.is_registered("sac")
        assert AlgorithmFactory.is_registered("SAC")
    
    def test_list_all_algorithms(self):
        """全アルゴリズムのリストを取得。"""
        algorithms = AlgorithmFactory.list_algorithms()
        
        assert "ppo" in algorithms
        assert "sac" in algorithms
        assert len(algorithms) >= 2
    
    def test_get_factory_info(self):
        """ファクトリー情報を取得。"""
        info = AlgorithmFactory.get_info()
        
        assert info["count"] >= 2
        assert "ppo" in info["algorithms"]
        assert "sac" in info["algorithms"]
        assert info["registry"]["ppo"] == "PPOAlgorithm"
        assert info["registry"]["sac"] == "SACAlgorithm"


# ========================================
# アルゴリズム作成テスト
# ========================================

class TestAlgorithmCreation:
    """アルゴリズムの作成テスト。"""
    
    def test_create_ppo_instance(self):
        """PPOインスタンスを作成。"""
        ppo = AlgorithmFactory.create("ppo")
        
        assert isinstance(ppo, PPOAlgorithm)
        assert ppo.algorithm_name == "ppo"
    
    def test_create_sac_instance(self):
        """SACインスタンスを作成。"""
        sac = AlgorithmFactory.create("sac")
        
        assert isinstance(sac, SACAlgorithm)
        assert sac.algorithm_name == "sac"
    
    def test_create_ppo_with_auto_halt(self):
        """PPOをAutoHalt付きで作成。"""
        ppo = AlgorithmFactory.create("ppo", use_auto_halt=True)
        
        assert isinstance(ppo, PPOAlgorithm)
        assert ppo._use_auto_halt is True
    
    def test_create_multiple_instances(self):
        """複数のインスタンスを独立して作成。"""
        ppo1 = AlgorithmFactory.create("ppo")
        ppo2 = AlgorithmFactory.create("ppo")
        sac1 = AlgorithmFactory.create("sac")
        
        # 各インスタンスが独立していることを確認
        assert ppo1 is not ppo2
        assert ppo1 is not sac1
        assert isinstance(ppo1, PPOAlgorithm)
        assert isinstance(ppo2, PPOAlgorithm)
        assert isinstance(sac1, SACAlgorithm)


# ========================================
# 設定検証テスト
# ========================================

class TestConfigValidation:
    """設定検証の統合テスト。"""
    
    def test_ppo_valid_config(self):
        """PPO有効設定の検証。"""
        ppo = AlgorithmFactory.create("ppo")
        config = ppo.get_default_config()
        
        # デフォルト設定が有効であることを確認
        assert PPOAlgorithm.validate_config(config) is True
    
    def test_sac_valid_config(self):
        """SAC有効設定の検証。"""
        sac = AlgorithmFactory.create("sac")
        config = sac.get_default_config()
        
        # デフォルト設定が有効であることを確認
        assert SACAlgorithm.validate_config(config) is True
    
    def test_ppo_invalid_config(self):
        """PPO無効設定でエラー。"""
        config = {
            "learning_rate": -0.001,  # 負の値
            "n_steps": 2048,
            "batch_size": 64,
        }
        
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            PPOAlgorithm.validate_config(config)
    
    def test_sac_invalid_config(self):
        """SAC無効設定でエラー。"""
        config = {
            "learning_rate": 3e-4,
            "buffer_size": 100,  # batch_sizeより小さい
            "batch_size": 256,
        }
        
        with pytest.raises(ValueError, match="buffer_size .* must be >= batch_size"):
            SACAlgorithm.validate_config(config)


# ========================================
# アルゴリズム切り替えシナリオ
# ========================================

class TestAlgorithmSwitchingScenarios:
    """アルゴリズム切り替えの実践的なシナリオテスト。"""
    
    def test_switch_from_ppo_to_sac(self):
        """PPOからSACへの切り替え。"""
        # PPOで開始
        config_v394 = {
            "algorithm": "ppo",
            "ppo_hyperparameters": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
            }
        }
        
        ppo = AlgorithmFactory.create(config_v394["algorithm"])
        assert ppo.algorithm_name == "ppo"
        
        # SACに切り替え
        config_v395 = {
            "algorithm": "sac",
            "sac_hyperparameters": {
                "learning_rate": 3e-4,
                "buffer_size": 50000,
                "batch_size": 256,
            }
        }
        
        sac = AlgorithmFactory.create(config_v395["algorithm"])
        assert sac.algorithm_name == "sac"
        
        # 両方が独立していることを確認
        assert ppo is not sac
        assert isinstance(ppo, PPOAlgorithm)
        assert isinstance(sac, SACAlgorithm)
    
    def test_config_isolation(self):
        """各アルゴリズムの設定が独立していることを確認。"""
        # PPO設定
        ppo_config = {
            "learning_rate": 1e-3,
            "n_steps": 1024,
            "batch_size": 32,
        }
        
        # SAC設定
        sac_config = {
            "learning_rate": 3e-4,
            "buffer_size": 100000,
            "batch_size": 512,
        }
        
        # 両方の設定が有効であることを確認
        assert PPOAlgorithm.validate_config(ppo_config) is True
        assert SACAlgorithm.validate_config(sac_config) is True
        
        # 設定が混在しないことを確認
        ppo = AlgorithmFactory.create("ppo")
        sac = AlgorithmFactory.create("sac")
        
        ppo_defaults = ppo.get_default_config()
        sac_defaults = sac.get_default_config()
        
        # PPOにはn_stepsがある
        assert "n_steps" in ppo_defaults
        # SACにはbuffer_sizeがある
        assert "buffer_size" in sac_defaults
        # 互いに持たない設定
        assert "buffer_size" not in ppo_defaults
        assert "n_steps" not in sac_defaults


# ========================================
# デフォルト設定比較テスト
# ========================================

class TestDefaultConfigComparison:
    """PPOとSACのデフォルト設定を比較。"""
    
    def test_learning_rate_defaults(self):
        """learning_rateのデフォルト値比較。"""
        ppo = AlgorithmFactory.create("ppo")
        sac = AlgorithmFactory.create("sac")
        
        ppo_config = ppo.get_default_config()
        sac_config = sac.get_default_config()
        
        # 両方とも3e-4
        assert ppo_config["learning_rate"] == 3e-4
        assert sac_config["learning_rate"] == 3e-4
    
    def test_gamma_defaults(self):
        """gammaのデフォルト値比較。"""
        ppo = AlgorithmFactory.create("ppo")
        sac = AlgorithmFactory.create("sac")
        
        ppo_config = ppo.get_default_config()
        sac_config = sac.get_default_config()
        
        # 両方とも0.99
        assert ppo_config["gamma"] == 0.99
        assert sac_config["gamma"] == 0.99
    
    def test_ppo_specific_defaults(self):
        """PPO固有のデフォルト設定。"""
        ppo = AlgorithmFactory.create("ppo")
        config = ppo.get_default_config()
        
        # PPO固有
        assert config["n_steps"] == 2048
        assert config["batch_size"] == 64
        assert config["n_epochs"] == 10
        assert config["gae_lambda"] == 0.95
        assert config["clip_range"] == 0.2
    
    def test_sac_specific_defaults(self):
        """SAC固有のデフォルト設定。"""
        sac = AlgorithmFactory.create("sac")
        config = sac.get_default_config()
        
        # SAC固有
        assert config["buffer_size"] == 50000
        assert config["batch_size"] == 256
        assert config["tau"] == 0.005
        assert config["ent_coef"] == "auto"
        assert config["target_entropy"] == "auto"


# ========================================
# エラーハンドリングテスト
# ========================================

class TestErrorHandling:
    """エラーハンドリングのテスト。"""
    
    def test_unknown_algorithm_error(self):
        """未知のアルゴリズムでエラー。"""
        with pytest.raises(ValueError, match="Unknown algorithm: 'td3'"):
            AlgorithmFactory.create("td3")
    
    def test_empty_algorithm_name(self):
        """空のアルゴリズム名でエラー。"""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            AlgorithmFactory.create("")
    
    def test_invalid_ppo_hyperparameters(self):
        """PPO無効ハイパーパラメータ。"""
        config = {
            "learning_rate": 3e-4,
            "n_steps": 0,  # 無効
            "batch_size": 64,
        }
        
        with pytest.raises(ValueError):
            PPOAlgorithm.validate_config(config)
    
    def test_invalid_sac_hyperparameters(self):
        """SAC無効ハイパーパラメータ。"""
        config = {
            "learning_rate": 3e-4,
            "buffer_size": -1000,  # 無効
            "batch_size": 256,
        }
        
        with pytest.raises(ValueError):
            SACAlgorithm.validate_config(config)


# ========================================
# 実践的な統合シナリオ
# ========================================

class TestPracticalIntegrationScenarios:
    """実践的な統合シナリオ。"""
    
    def test_v394_to_v395_migration(self):
        """v394（PPO）からv395（SAC）への移行シナリオ。"""
        # v394: PPO訓練
        v394_config = {
            "algorithm": "ppo",
            "ppo_hyperparameters": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "ent_coef": 0.2,  # 20倍でもHOLD 89%
            }
        }
        
        ppo = AlgorithmFactory.create(v394_config["algorithm"])
        assert ppo.algorithm_name == "ppo"
        
        # v395: SACに移行（探索性能改善を期待）
        v395_config = {
            "algorithm": "sac",
            "sac_hyperparameters": {
                "learning_rate": 3e-4,
                "buffer_size": 50000,
                "batch_size": 256,
                "ent_coef": "auto",  # 自動調整
                "target_entropy": "auto",
            }
        }
        
        sac = AlgorithmFactory.create(v395_config["algorithm"])
        assert sac.algorithm_name == "sac"
        
        # 両方が有効な設定であることを確認
        assert PPOAlgorithm.validate_config(v394_config["ppo_hyperparameters"]) is True
        assert SACAlgorithm.validate_config(v395_config["sac_hyperparameters"]) is True
    
    def test_batch_algorithm_creation(self):
        """複数アルゴリズムを一括作成。"""
        algorithms = ["ppo", "sac"]
        instances = []
        
        for algo_name in algorithms:
            algo = AlgorithmFactory.create(algo_name)
            instances.append(algo)
        
        # 全て正しく作成されたか確認
        assert len(instances) == 2
        assert isinstance(instances[0], PPOAlgorithm)
        assert isinstance(instances[1], SACAlgorithm)
    
    def test_config_file_simulation(self):
        """設定ファイルからの読み込みシミュレーション。"""
        # JSON設定ファイルのシミュレーション
        config_files = [
            {
                "version": "v394d",
                "algorithm": "ppo",
                "ppo_hyperparameters": {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64}
            },
            {
                "version": "v395a",
                "algorithm": "sac",
                "sac_hyperparameters": {"learning_rate": 3e-4, "buffer_size": 50000, "batch_size": 256}
            }
        ]
        
        results = []
        for config in config_files:
            algo = AlgorithmFactory.create(config["algorithm"])
            results.append({
                "version": config["version"],
                "algorithm": algo.algorithm_name,
                "instance": algo
            })
        
        # 両バージョンが正しく処理されたか確認
        assert results[0]["algorithm"] == "ppo"
        assert results[1]["algorithm"] == "sac"
        assert isinstance(results[0]["instance"], PPOAlgorithm)
        assert isinstance(results[1]["instance"], SACAlgorithm)
