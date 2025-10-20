"""
アルゴリズムファクトリーのテストスクリプト。

新しいアルゴリズム差し替え機能が正しく動作するか確認する。
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.training.algorithms import AlgorithmFactory, PPOAlgorithm


def test_algorithm_factory():
    """アルゴリズムファクトリーの基本機能をテスト"""

    print("=" * 60)
    print("🧪 Algorithm Factory Test")
    print("=" * 60)

    # Test 1: 利用可能なアルゴリズムのリスト
    print("\n[Test 1] List available algorithms")
    algorithms = AlgorithmFactory.list_algorithms()
    print(f"✅ Available algorithms: {algorithms}")
    assert "ppo" in algorithms, "PPO should be registered"

    # Test 2: アルゴリズム情報の取得
    print("\n[Test 2] Get algorithm info")
    info = AlgorithmFactory.get_info()
    print(f"✅ Algorithm count: {info['count']}")
    print(f"✅ Registry: {info['registry']}")

    # Test 3: PPOアルゴリズムの作成
    print("\n[Test 3] Create PPO algorithm")
    ppo = AlgorithmFactory.create("ppo")
    print(f"✅ Created: {ppo}")
    print(f"✅ Algorithm name: {ppo.algorithm_name}")
    assert ppo.algorithm_name == "ppo", "Algorithm name should be 'ppo'"

    # Test 4: デフォルト設定の取得
    print("\n[Test 4] Get default config")
    config = ppo.get_default_config()
    print(f"✅ Default config keys: {list(config.keys())}")
    print("✅ PPO hyperparameters:")
    for key, value in config["ppo_hyperparameters"].items():
        print(f"   - {key}: {value}")

    # Test 5: 設定の検証
    print("\n[Test 5] Validate config")
    is_valid = ppo.validate_config(config)
    print(f"✅ Config validation: {'PASSED' if is_valid else 'FAILED'}")
    assert is_valid, "Default config should be valid"

    # Test 6: 不正なアルゴリズム名
    print("\n[Test 6] Try to create unknown algorithm")
    try:
        AlgorithmFactory.create("unknown")
        print("❌ Should have raised ValueError")
        assert False
    except ValueError as e:
        print(f"✅ Correctly raised ValueError: {e}")

    # Test 7: 大文字小文字を区別しない
    print("\n[Test 7] Case-insensitive algorithm name")
    ppo_upper = AlgorithmFactory.create("PPO")
    print(f"✅ Created with 'PPO': {ppo_upper.algorithm_name}")
    assert ppo_upper.algorithm_name == "ppo", "Should normalize to lowercase"

    # Test 8: 登録確認
    print("\n[Test 8] Check registration")
    is_registered = AlgorithmFactory.is_registered("ppo")
    print(f"✅ PPO is registered: {is_registered}")
    assert is_registered, "PPO should be registered"

    is_registered_unknown = AlgorithmFactory.is_registered("sac")
    print(f"✅ SAC is registered: {is_registered_unknown}")
    assert not is_registered_unknown, "SAC should not be registered yet"

    print("\n" + "=" * 60)
    print("🎉 All tests passed!")
    print("=" * 60)
    print("\n📋 Summary:")
    print("   - PPO algorithm is properly registered")
    print("   - Factory can create PPO instances")
    print("   - Default config is valid")
    print("   - Error handling works correctly")
    print("\n✅ Ready to integrate with unified_trainer.py")


def test_ppo_algorithm_directly():
    """PPOAlgorithmクラスを直接テスト"""

    print("\n" + "=" * 60)
    print("🧪 PPO Algorithm Direct Test")
    print("=" * 60)

    # Test 1: PPOAlgorithm直接作成
    print("\n[Test 1] Create PPOAlgorithm directly")
    ppo = PPOAlgorithm()
    print(f"✅ Created: {ppo}")

    # Test 2: AutoHalt版
    print("\n[Test 2] Create PPOAlgorithm with auto_halt")
    ppo_auto_halt = PPOAlgorithm(use_auto_halt=True)
    print(f"✅ Created: {ppo_auto_halt}")

    # Test 3: デフォルト設定
    print("\n[Test 3] Get default config")
    config = ppo.get_default_config()
    print(f"✅ Algorithm: {config['algorithm']}")
    print(f"✅ Learning rate: {config['ppo_hyperparameters']['learning_rate']}")
    print(f"✅ Entropy coefficient: {config['ppo_hyperparameters']['ent_coef']}")

    print("\n✅ PPO Algorithm direct test passed!")


if __name__ == "__main__":
    try:
        test_algorithm_factory()
        test_ppo_algorithm_directly()

        print("\n" + "=" * 60)
        print("🚀 Next Steps:")
        print("=" * 60)
        print("1. ✅ Algorithm factory is working")
        print("2. ⏭️  Update unified_trainer.py to use AlgorithmFactory")
        print("3. ⏭️  Test with existing training scripts")
        print("4. ⏭️  Add SAC implementation")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
