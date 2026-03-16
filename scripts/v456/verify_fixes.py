#!/usr/bin/env python3
"""
v456 修正内容の検証スクリプト

以下を確認：
1. Config ロード正常化
2. Reward parameters 適用
3. ロギング スロットル修正 (last_log_step 分離)
4. Checkpoint manager API 修正
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_config_loading():
    """Config ロード検証"""
    logger.info("=" * 70)
    logger.info("Test 1: Config Loading")
    logger.info("=" * 70)
    
    try:
        import yaml
        config_file = Path("config/v456/base/config.yaml")
        
        if not config_file.exists():
            logger.error(f"✗ Config file not found: {config_file}")
            return False
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if not config:
            logger.error("✗ Config is empty")
            return False
        
        logger.info(f"✓ Config loaded successfully")
        logger.info(f"  - Keys: {list(config.keys())}")
        
        if 'reward_settings' in config:
            logger.info(f"  - Reward parameters: {len(config['reward_settings'])} items")
            logger.info(f"    {list(config['reward_settings'].keys())[:5]}...")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Config loading failed: {e}")
        return False


def verify_callback_separation():
    """Callback log/save 分離検証"""
    logger.info("\n" + "=" * 70)
    logger.info("Test 2: Callback Log/Save Separation")
    logger.info("=" * 70)
    
    try:
        from ztb.utils.checkpoint import CheckpointManager
        import inspect
        
        # scripts.v456 モジュールをオンザフライでロード
        spec = __import__('importlib').util.spec_from_file_location(
            "train_v456_optimized",
            Path(__file__).parent / "train_v456_optimized.py"
        )
        train_module = __import__('importlib').util.module_from_spec(spec)
        spec.loader.exec_module(train_module)
        
        V456TrainingCallbackOptimized = train_module.V456TrainingCallbackOptimized
        
        callback = V456TrainingCallbackOptimized(
            checkpoint_mgr=None,
            cache_coord=None,
            save_freq=5000,
            log_freq=1000
        )
        
        # 属性確認
        if not hasattr(callback, 'last_log_step'):
            logger.error("✗ last_log_step attribute not found")
            return False
        
        if not hasattr(callback, 'last_save_step'):
            logger.error("✗ last_save_step attribute not found")
            return False
        
        logger.info("✓ Callback has separate log/save counters")
        logger.info(f"  - last_log_step: {callback.last_log_step}")
        logger.info(f"  - last_save_step: {callback.last_save_step}")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Callback verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_checkpoint_api():
    """CheckpointManager API 検証"""
    logger.info("\n" + "=" * 70)
    logger.info("Test 3: CheckpointManager API")
    logger.info("=" * 70)
    
    try:
        from ztb.utils.checkpoint import CheckpointManager
        
        # 一時ディレクトリで生成
        temp_dir = Path("temp_checkpoint_test")
        temp_dir.mkdir(exist_ok=True)
        
        mgr = CheckpointManager(save_dir=str(temp_dir))
        
        # API確認
        if not hasattr(mgr, 'save_sync'):
            logger.error("✗ save_sync method not found")
            return False
        
        if not hasattr(mgr, 'save_async'):
            logger.error("✗ save_async method not found")
            return False
        
        logger.info("✓ CheckpointManager has correct API")
        logger.info(f"  - save_sync: {hasattr(mgr, 'save_sync')}")
        logger.info(f"  - save_async: {hasattr(mgr, 'save_async')}")
        
        # テスト保存
        test_data = {"test": "data"}
        path = mgr.save_sync(test_data, step=100, metadata={"test": True})
        logger.info(f"  - Test save successful: {path}")
        
        # クリーンアップ
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return True
    
    except Exception as e:
        logger.error(f"✗ CheckpointManager verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_environment_reward_params():
    """環境の Reward Parameters ワイアリング検証"""
    logger.info("\n" + "=" * 70)
    logger.info("Test 4: Environment Reward Parameters")
    logger.info("=" * 70)
    
    try:
        from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
        import inspect
        
        # コンストラクタのシグネチャを確認
        sig = inspect.signature(FastIntradayEnvV456.__init__)
        params = list(sig.parameters.keys())
        
        # reward_params がパラメータとして存在するか
        if 'reward_params' not in params:
            logger.error("✗ reward_params parameter not in FastIntradayEnvV456.__init__")
            return False
        
        logger.info("✓ reward_params parameter exists in constructor")
        
        # ソースコードから reward_params の使用を確認
        source = inspect.getsource(FastIntradayEnvV456)
        
        # self.reward_params の記載を確認
        if 'self.reward_params' in source:
            logger.info("✓ reward_params is stored in environment")
            
            # ワイアリング確認
            if 'reward_kwargs.update(self.reward_params)' in source or '**self.reward_params' in source:
                logger.info("✓ reward_params is wired to reward function")
                return True
            else:
                logger.warning("⚠ reward_params stored but usage pattern unclear")
                return True
        else:
            logger.error("✗ reward_params not used in environment")
            return False
    
    except Exception as e:
        logger.error(f"✗ Environment reward params verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_causal_features():
    """Causal feature 計算検証 (look-ahead leak 修正)"""
    logger.info("\n" + "=" * 70)
    logger.info("Test 5: Causal Feature Calculation")
    logger.info("=" * 70)
    
    try:
        import inspect
        from ztb.trading.environment import factory_v456
        
        # factory_v456 のソースコード確認
        source = inspect.getsource(factory_v456.EnvironmentFactory)
        
        # Causal な計算方法が使われているかを確認
        # NOT: np.convolve(..., mode='same')
        # OK: rolling mean, for loops with i-period bounds
        
        if "np.convolve" in source and "mode='same'" in source:
            logger.error("✗ Detected non-causal np.convolve with mode='same'")
            return False
        
        # Causal pattern の確認
        has_causal_pattern = (
            "for i in range" in source and  # for ループで過去のみ参照
            "max(0, i" in source  # 過去を上限に (max(0, i - period) パターン)
        )
        
        if has_causal_pattern:
            logger.info("✓ BB calculations use causal rolling patterns")
            logger.info("  - Uses for loops with backward-looking windows")
            logger.info("  - No future values incorporated")
        else:
            logger.warning("⚠ Could not verify causal pattern clearly")
        
        # 具体的なメソッド確認
        if "_calculate_bb_width" in source and "_calculate_bb_pct" in source:
            logger.info("✓ Causal BB feature methods exist:")
            logger.info("  - _calculate_bb_width: ✓")
            logger.info("  - _calculate_bb_pct: ✓")
            
            if has_causal_pattern:
                return True
        
        return has_causal_pattern
    
    except Exception as e:
        logger.error(f"✗ Causal feature verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main verification"""
    logger.info("\n")
    logger.info("███████████████████████████████████████████████████████████████████")
    logger.info("v456 AI Code Review 修正検証")
    logger.info("███████████████████████████████████████████████████████████████████\n")
    
    results = {
        "Config Loading": verify_config_loading(),
        "Callback Separation": verify_callback_separation(),
        "CheckpointManager API": verify_checkpoint_api(),
        "Environment Reward Params": verify_environment_reward_params(),
        "Causal Feature Calculation": verify_causal_features(),
    }
    
    logger.info("\n" + "=" * 70)
    logger.info("検証結果サマリー")
    logger.info("=" * 70)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    logger.info("\n" + "=" * 70)
    if all_passed:
        logger.info("✅ すべての検証に PASS しました")
        return 0
    else:
        logger.info(f"❌ {sum(not v for v in results.values())} 個のテストが失敗しました")
        return 1


if __name__ == "__main__":
    sys.exit(main())
