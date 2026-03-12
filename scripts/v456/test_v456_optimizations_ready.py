#!/usr/bin/env python3
"""
v456 Training Validation & Performance Test

Phase 1-3 最適化の統合検証と実行可能性確認
"""
import logging
import sys
from pathlib import Path
import time
import json

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from ztb.utils.checkpoint import CheckpointManager
from ztb.utils.cache_coordination import CacheCoordinator
from ztb.utils.error_utils import safe_operation
from ztb.optimization.parallel.window_evaluator import ParallelWindowEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_phase_1a_checkpoint():
    """Phase 1-A: Checkpoint Manager テスト"""
    logger.info("=" * 60)
    logger.info("Testing Phase 1-A: Checkpoint Manager")
    logger.info("=" * 60)
    
    checkpoint_dir = Path("models/v456/test_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_mgr = CheckpointManager(
        save_dir=str(checkpoint_dir),
        compress="zstd",
        keep_last=3,
    )
    logger.info(f"✓ CheckpointManager initialized with zstd compression")
    
    # ダミーデータで save テスト
    test_data = {
        "model_state": {"weights": [1.0, 2.0, 3.0]},
        "config": {"lr": 0.0003, "gamma": 0.99},
    }
    
    for step in [1000, 5000, 10000]:
        try:
            path = checkpoint_mgr.save_sync(
                test_data,
                step=step,
                metadata={"step": step, "reward": 100.0}
            )
            logger.info(f"✓ Checkpoint saved at step {step}: {path}")
        except Exception as e:
            logger.error(f"❌ Checkpoint save failed: {e}")
            return False
    
    logger.info("✅ Phase 1-A PASS\n")
    return True


def test_phase_1b_error_handling():
    """Phase 1-B: Unified error handling"""
    logger.info("=" * 60)
    logger.info("Testing Phase 1-B: Error Handling")
    logger.info("=" * 60)
    
    # Success case
    def success_op():
        return 42
    
    result = safe_operation(success_op, default_result=0, collect_errors=False)
    if result == 42:
        logger.info(f"✓ Success case: {result}")
    else:
        logger.error(f"❌ Success case failed")
        return False
    
    # Error case
    def error_op():
        raise ValueError("Test error")
    
    result = safe_operation(error_op, default_result=999, collect_errors=False)
    if result == 999:
        logger.info(f"✓ Error handling case: returned default value")
    else:
        logger.error(f"❌ Error handling failed")
        return False
    
    logger.info("✅ Phase 1-B PASS\n")
    return True


def test_phase_3_cache():
    """Phase 3: Cache Coordination"""
    logger.info("=" * 60)
    logger.info("Testing Phase 3: Cache Coordination")
    logger.info("=" * 60)
    
    cache_coord = CacheCoordinator(
        max_items=100,
        ttl_seconds=3600,
    )
    logger.info(f"✓ CacheCoordinator initialized: LRU+TTL")
    
    # Put/Get test
    cache_coord.put("test_key_1", {"value": 100})
    cache_coord.put("test_key_2", {"value": 200})
    
    val1 = cache_coord.get("test_key_1")
    val2 = cache_coord.get("test_key_1")  # Hit
    val3 = cache_coord.get("test_key_3")  # Miss
    
    stats = cache_coord.get_stats()
    logger.info(f"✓ Cache operations completed")
    logger.info(f"  - Stats: {stats}")
    
    if stats['hit_rate'] > 0:
        logger.info(f"✅ Phase 3 PASS\n")
        return True
    else:
        logger.error(f"❌ Phase 3 FAIL")
        return False


def test_phase_2_parallel():
    """Phase 2: Parallel Evaluation"""
    logger.info("=" * 60)
    logger.info("Testing Phase 2: Parallel Window Evaluator")
    logger.info("=" * 60)
    
    try:
        evaluator = ParallelWindowEvaluator(
            num_workers=4,
            enable_error_collection=True,
            enable_caching=False,
        )
        logger.info(f"✓ ParallelWindowEvaluator initialized with 4 workers")
        logger.info(f"✅ Phase 2 PASS\n")
        return True
    except Exception as e:
        logger.error(f"❌ Phase 2 FAIL: {e}")
        return False


def main():
    """メイン統合テスト"""
    logger.info("\n" + "=" * 60)
    logger.info("v456 Phase 1-3 Integration Test")
    logger.info("=" * 60 + "\n")
    
    start_time = time.time()
    
    results = {
        "Phase 1-B (Error Handling)": test_phase_1b_error_handling(),
        "Phase 1-A (Checkpoint)": test_phase_1a_checkpoint(),
        "Phase 2 (Parallel Evaluation)": test_phase_2_parallel(),
        "Phase 3 (Cache Coordination)": test_phase_3_cache(),
    }
    
    elapsed = time.time() - start_time
    
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    all_passed = True
    for phase, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{phase}: {status}")
        if not passed:
            all_passed = False
    
    logger.info(f"\nElapsed time: {elapsed:.2f}s")
    
    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED - v456 optimizations ready for training!")
        return 0
    else:
        logger.error("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
