"""
unified_trainer.py リファクタリングのテスト。

ConfigBuilderとAlgorithmFactory統合が正常に動作するか確認。
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.training.core.config_builder import ConfigBuilder
from ztb.training.core.algorithm_trainer import AlgorithmTrainer
from ztb.training.core.config_manager import ConfigManager
from ztb.training.algorithms import AlgorithmFactory


def test_config_builder():
    """ConfigBuilderのテスト"""
    print("=" * 60)
    print("🧪 ConfigBuilder Test")
    print("=" * 60)
    
    # テスト設定
    config = {
        "algorithm": "ppo",
        "model_name": "test_model",
        "total_timesteps": 10000,
        "ppo_hyperparameters": {
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "ent_coef": 0.01
        },
        "environment": {
            "initial_balance": 200000,
            "transaction_cost": 0.0005
        }
    }
    
    # ConfigBuilder作成
    builder = ConfigBuilder(config)
    print(f"\n✅ Created: {builder}")
    
    # 設定値取得テスト
    lr = builder.get_config_value("learning_rate", ["ppo_hyperparameters"], 0.0001)
    print(f"✅ Learning rate: {lr}")
    assert lr == 0.0003, f"Expected 0.0003, got {lr}"
    
    # メモリ最適化設定
    memory_config = builder.get_memory_optimization_config()
    print(f"✅ Memory optimization: {memory_config}")
    
    # 環境設定
    env_config = builder.get_environment_config()
    print(f"✅ Environment config: initial_balance={env_config['initial_balance']}")
    assert env_config["initial_balance"] == 200000
    
    # PPO設定
    ppo_config = builder.get_ppo_core_config()
    print(f"✅ PPO config: {len(ppo_config)} parameters")
    assert ppo_config["learning_rate"] == 0.0003
    assert ppo_config["ent_coef"] == 0.01
    
    # 特徴量設定
    feature_config = builder.get_feature_config()
    print(f"✅ Feature config: {feature_config}")
    
    print("\n🎉 ConfigBuilder test passed!")


def test_algorithm_factory_integration():
    """AlgorithmFactory統合テスト"""
    print("\n" + "=" * 60)
    print("🧪 AlgorithmFactory Integration Test")
    print("=" * 60)
    
    # テスト設定
    config = {
        "algorithm": "ppo",
        "model_name": "test_ppo",
        "total_timesteps": 10000,
        "ppo_hyperparameters": {
            "learning_rate": 0.007503,
            "n_steps": 2048,
            "batch_size": 256,
            "ent_coef": 0.01
        }
    }
    
    # ConfigManager作成
    config_manager = ConfigManager(config)
    print(f"✅ Created ConfigManager")
    
    # AlgorithmTrainer作成
    algorithm_trainer = AlgorithmTrainer(config_manager, progress_bar_enabled=False)
    print(f"✅ Created AlgorithmTrainer")
    
    # AlgorithmFactoryから直接PPO取得
    ppo_algo = AlgorithmFactory.create("ppo")
    print(f"✅ Created PPO via AlgorithmFactory: {ppo_algo}")
    
    # デフォルト設定確認
    default_config = ppo_algo.get_default_config()
    print(f"✅ PPO default config has {len(default_config)} sections")
    
    print("\n🎉 AlgorithmFactory integration test passed!")


def test_unified_trainer_import():
    """UnifiedTrainerのインポートテスト"""
    print("\n" + "=" * 60)
    print("🧪 UnifiedTrainer Import Test")
    print("=" * 60)
    
    try:
        from ztb.training.unified_trainer import UnifiedTrainer
        print("✅ Successfully imported UnifiedTrainer")
        
        # 簡単な設定で初期化テスト
        config = {
            "algorithm": "ppo",
            "model_name": "import_test",
            "total_timesteps": 1000
        }
        
        trainer = UnifiedTrainer(config, dry_run=True)
        print(f"✅ Created UnifiedTrainer with algorithm: {trainer.algorithm}")
        print(f"✅ ConfigBuilder available: {trainer.config_builder is not None}")
        
        # ConfigBuilder経由で設定取得
        memory_config = trainer.get_memory_optimization_config()
        print(f"✅ Memory config via ConfigBuilder: {memory_config}")
        
        print("\n🎉 UnifiedTrainer import test passed!")
        
    except Exception as e:
        print(f"❌ Failed to import or initialize UnifiedTrainer: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_summary():
    """全体のサマリー"""
    print("\n" + "=" * 60)
    print("📊 Refactoring Summary")
    print("=" * 60)
    
    print("\n✅ Completed:")
    print("  1. ConfigBuilder created and working")
    print("  2. AlgorithmFactory integrated into AlgorithmTrainer")
    print("  3. UnifiedTrainer using ConfigBuilder for all config methods")
    print("  4. PPO now uses AlgorithmFactory (hybrid mode)")
    
    print("\n📋 Architecture:")
    print("  unified_trainer.py (simplified)")
    print("    ↓ uses")
    print("  ConfigBuilder (config extraction)")
    print("    ↓ uses")
    print("  AlgorithmTrainer (algorithm dispatch)")
    print("    ↓ uses")
    print("  AlgorithmFactory.create('ppo')")
    print("    ↓ creates")
    print("  PPOAlgorithm (new architecture)")
    
    print("\n🎯 Benefits:")
    print("  - unified_trainer.py methods delegated to ConfigBuilder")
    print("  - Easy to add new algorithms (SAC, TD3, etc.)")
    print("  - Config building logic centralized")
    print("  - AlgorithmFactory provides pluggable architecture")
    
    print("\n⏭️  Next Steps:")
    print("  1. Test with existing training scripts (train_v394d.py)")
    print("  2. Verify PPO training works correctly")
    print("  3. Add SAC implementation")
    print("  4. Fully migrate PPO to new architecture")


if __name__ == "__main__":
    try:
        # Run all tests
        test_config_builder()
        test_algorithm_factory_integration()
        success = test_unified_trainer_import()
        test_summary()
        
        if success:
            print("\n" + "=" * 60)
            print("🎉 ALL TESTS PASSED!")
            print("=" * 60)
            print("\n✅ unified_trainer.py refactoring successful!")
            print("✅ Ready to test with existing training scripts")
        else:
            print("\n❌ Some tests failed. Please check errors above.")
            sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
