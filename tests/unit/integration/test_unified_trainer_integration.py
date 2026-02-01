#!/usr/bin/env python3
"""
Integration test for V4XXUnifiedTrainer with enhanced optimizer features
"""

import json
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.training.v4xx_unified_trainer import V4XXUnifiedTrainer


def create_test_config():
    """テスト用の設定ファイルを作成"""
    config = {
        "algorithm": "sac",
        "model_name": "test_sac_v442",
        "training": {
            "model_name": "test_sac_v442",
            "algorithm": "sac",
            "total_timesteps": 1000,
            "learning_rate": 0.001,
            "batch_size": 64,
            "buffer_size": 10000,
            "sac_hyperparameters": {
                "learning_rate": 0.0003,
                "buffer_size": 1000000,
                "learning_starts": 1000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "ent_coef": 0.01,
                "target_update_interval": 1,
                "target_entropy": -2.0,
            },
        },
        "optimizer_features": {
            "max_history": 500,
            "enable_normalization": True,
            "normalization_method": "robust",
            "outlier_threshold": 1.5,
        },
        "environment": {
            "name": "test_env",
            "observation_space": "continuous",
            "action_space": "continuous",
        },
        "model": {"policy": "MlpPolicy", "learning_rate": 0.001},
    }
    return config


def test_unified_trainer_integration():
    """統合トレーナーのテスト"""
    print("Testing V4XXUnifiedTrainer integration with enhanced optimizer features...")

    # 一時設定ファイルを作成
    config = create_test_config()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f, indent=2)
        config_path = f.name

    try:
        # トレーナーを初期化
        trainer = V4XXUnifiedTrainer(config_path)

        # 設定が正しく読み込まれたか確認
        assert trainer.config is not None
        assert "optimizer_features" in trainer.config
        print("✓ Configuration loaded successfully")

        # Optimizer trackerが正しく初期化されたか確認
        assert trainer.optimizer_tracker is not None
        assert hasattr(trainer.optimizer_tracker, "enable_normalization")
        assert trainer.optimizer_tracker.enable_normalization == True
        assert trainer.optimizer_tracker.normalization_method == "robust"
        print("✓ Optimizer tracker initialized with config settings")

        # アルゴリズムトレーナーを作成
        trainer.initialize_trainer()
        assert trainer.trainer is not None
        print("✓ Algorithm trainer initialized")

        # Optimizer trackerがトレーナーに渡されたか確認
        assert hasattr(trainer.trainer, "optimizer_tracker")
        assert trainer.trainer.optimizer_tracker is not None
        print("✓ Optimizer tracker passed to algorithm trainer")

        # 特徴量ベクトルをテスト
        features = trainer.optimizer_tracker.get_feature_vector()
        assert len(features) == 11  # 11個の特徴量
        assert all(isinstance(v, (int, float)) for v in features.values())
        print(f"✓ Feature vector generated with {len(features)} features")

        # 統計的機能をテスト
        correlations = trainer.optimizer_tracker.compute_feature_correlations()
        importance = trainer.optimizer_tracker.compute_feature_importance()
        assert isinstance(correlations, dict)
        assert isinstance(importance, dict)
        print("✓ Statistical analysis functions working")

        # デバッグ情報をテスト
        features_debug = trainer.optimizer_tracker.get_feature_vector(
            include_debug_info=True
        )
        assert "_debug_info" in features_debug
        debug_info = features_debug["_debug_info"]
        assert "update_count" in debug_info
        assert "history_lengths" in debug_info
        print("✓ Debug information available")

        print("🎉 V4XXUnifiedTrainer integration test passed!")

    finally:
        # 一時ファイルを削除
        Path(config_path).unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        test_unified_trainer_integration()
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
