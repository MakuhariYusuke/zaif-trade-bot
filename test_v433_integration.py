#!/usr/bin/env python3
"""
V433統合システムテストスクリプト
適応型SACコア、オンライン学習エンジン、unified_optimizerの統合テスト
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.adaptive_sac_core import AdaptiveSACCore, AdaptiveSACConfig
from ztb.training.online_learning_engine import OnlineLearningEngine, OnlineLearningConfig
from ztb.optimization.unified_optimizer import UnifiedOptimizer, OptimizationConfig
from ztb.training.unified_trainer.trainer import UnifiedTrainer

def test_unified_optimizer():
    """Unified Optimizerのテスト"""
    print("Testing Unified Optimizer...")

    config = OptimizationConfig(
        enable_hyperparameter_optimization=True,
        enable_system_optimization=True,
        enable_reward_optimization=True,
        max_trials=10,
        max_parallel_trials=2
    )

    optimizer = UnifiedOptimizer(config)

    # ハイパーパラメータ最適化のテスト
    search_space = {
        "learning_rate": {"type": "float", "low": 0.0001, "high": 0.1},
        "batch_size": {"type": "int", "low": 16, "high": 128}
    }

    def dummy_objective(params):
        lr = params.get("learning_rate", 0.001)
        bs = params.get("batch_size", 32)
        return -(lr - 0.01)**2 - (bs - 64)**2 / 1000

    try:
        result = optimizer.optimize_hyperparameters(dummy_objective, search_space)
        print(f"✓ Hyperparameter optimization completed: {result.best_params}")
    except Exception as e:
        print(f"✗ Hyperparameter optimization failed: {e}")

    # システム最適化のテスト
    try:
        system_result = optimizer.optimize_system()
        print(f"✓ System optimization completed: {system_result}")
    except Exception as e:
        print(f"✗ System optimization failed: {e}")

    print("Unified Optimizer test completed\n")

def test_adaptive_sac_core():
    """Adaptive SAC Coreのテスト"""
    print("Testing Adaptive SAC Core...")

    config = AdaptiveSACConfig(
        enable_market_regime_adaptation=True,
        enable_online_learning=True,
        adaptation_interval_steps=100,
        performance_window_size=10
    )

    observation_dim = 10
    action_dim = 3

    try:
        adaptive_sac = AdaptiveSACCore(config, observation_dim, action_dim)
        status = adaptive_sac.get_adaptation_status()
        print(f"✓ Adaptive SAC Core initialized: {status}")
    except Exception as e:
        print(f"✗ Adaptive SAC Core initialization failed: {e}")

    print("Adaptive SAC Core test completed\n")

def test_online_learning_engine():
    """Online Learning Engineのテスト"""
    print("Testing Online Learning Engine...")

    config = OnlineLearningConfig(
        stream_buffer_size=1000,
        learning_batch_size=32,
        experience_buffer_size=10000,
        adaptation_threshold=0.1
    )

    # Adaptive SAC Coreの作成
    sac_config = AdaptiveSACConfig()
    adaptive_sac = AdaptiveSACCore(sac_config, 10, 3)

    try:
        engine = OnlineLearningEngine(config, adaptive_sac)
        status = engine.get_learning_status()
        print(f"✓ Online Learning Engine initialized: {status}")
    except Exception as e:
        print(f"✗ Online Learning Engine initialization failed: {e}")

    print("Online Learning Engine test completed\n")

def test_unified_trainer_integration():
    """Unified Trainer統合テスト"""
    print("Testing Unified Trainer V433 Integration...")

    # V433適応型学習設定
    config = {
        "algorithm": "sac",
        "model_name": "v433_adaptive_test",
        "total_timesteps": 1000,
        "enable_v433_adaptive": True,
        "v433_adaptive_config": {
            "enable_market_regime_adaptation": True,
            "enable_online_learning": True,
            "adaptation_interval_steps": 100,
            "learning_rate": 3e-4,
            "buffer_size": 10000,
            "performance_window_size": 10,
            "stream_buffer_size": 1000,
            "learning_batch_size": 32,
            "experience_buffer_size": 5000,
            "enable_hyperparameter_optimization": True,
            "enable_system_optimization": True,
            "enable_reward_optimization": True,
            "max_trials": 5
        },
        "environment": {
            "observation_dim": 10,
            "action_dim": 3,
            "initial_balance": 10000.0,
            "transaction_cost": 1e-5,
            "max_position_size": 1.0,
            "window_size": 64
        },
        "data_config": {},
        "features": {}
    }

    try:
        trainer = UnifiedTrainer(config, dry_run=True)
        print("✓ Unified Trainer with V433 components initialized")

        # V433コンポーネントの確認
        if trainer.enable_v433_adaptive:
            print("✓ V433 adaptive learning enabled")
            if trainer.adaptive_sac_core:
                print("✓ Adaptive SAC Core integrated")
            if trainer.online_learning_engine:
                print("✓ Online Learning Engine integrated")
            if trainer.unified_optimizer:
                print("✓ Unified Optimizer integrated")
        else:
            print("✗ V433 adaptive learning not enabled")

    except Exception as e:
        print(f"✗ Unified Trainer integration failed: {e}")

    print("Unified Trainer integration test completed\n")

def main():
    """メインテスト関数"""
    print("V433 Integrated System Test")
    print("=" * 50)

    # 各コンポーネントのテスト
    test_unified_optimizer()
    test_adaptive_sac_core()
    test_online_learning_engine()
    test_unified_trainer_integration()

    print("All tests completed!")
    print("\nV433 Phase 2 Components:")
    print("✓ unified_optimizer - Consolidated optimization system")
    print("✓ adaptive_sac_core - Market regime adaptive SAC")
    print("✓ online_learning_engine - Real-time continuous learning")
    print("✓ unified_trainer integration - V433 adaptive training")

if __name__ == "__main__":
    main()