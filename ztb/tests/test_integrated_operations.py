#!/usr/bin/env python3
"""
統合運用管理システムテスト
Integrated Operations Management System Test
"""

import logging
import os
import sys
import time

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from ztb.adaptation.monitoring.config import MonitoringConfig, ScalabilityConfig
from ztb.adaptation.online_learning.config import OnlineLearningConfig
from ztb.adaptation.operations.config import OperationsConfig
from ztb.adaptation.safety.config import SafetyConfig


# SACConfigの代わりに直接設定を作成
class MockSACConfig:
    def __init__(self):
        self.monitoring = MonitoringConfig()
        self.safety = SafetyConfig()
        self.online_learning = OnlineLearningConfig()
        self.operations = OperationsConfig()
        self.scalability = ScalabilityConfig()


from ztb.adaptation.operations.config import IntegratedOperationsConfig
from ztb.adaptation.operations.manager import IntegratedOperationsManager

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_integrated_operations():
    """統合運用管理テスト"""
    logger.info("Starting Integrated Operations Management Test")

    try:
        # 設定の初期化
        sac_config = MockSACConfig()
        operations_config = IntegratedOperationsConfig(
            integrated_operations_enabled=True,
            monitoring_enabled=True,
            safety_enabled=True,
            scalability_enabled=True,
            online_learning_enabled=False,  # テストでは無効
            health_check_interval_seconds=5,  # テスト用に短く
            component_sync_interval_seconds=10,
        )

        # 統合マネージャーの初期化
        manager = IntegratedOperationsManager(sac_config, operations_config)

        # システム起動テスト
        logger.info("Testing system startup...")
        success = manager.start_all_systems()
        if not success:
            logger.error("Failed to start integrated systems")
            return False

        # 起動待機
        time.sleep(2)

        # ステータス取得テスト
        logger.info("Testing system status retrieval...")
        status = manager.get_system_status()
        logger.info(f"System status: {status}")

        # アラート概要取得テスト
        logger.info("Testing alerts summary...")
        alerts = manager.get_alerts_summary()
        logger.info(f"Alerts summary: {alerts}")

        # ヘルスチェック待機
        logger.info("Waiting for health checks...")
        time.sleep(15)

        # 最終ステータス確認
        final_status = manager.get_system_status()
        logger.info(f"Final system status: {final_status}")

        # システム停止テスト
        logger.info("Testing system shutdown...")
        manager.stop_all_systems()

        logger.info("Integrated Operations Management Test completed successfully")
        return True

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return False


def test_configuration_validation():
    """設定検証テスト"""
    logger.info("Testing configuration validation...")

    try:
        # 有効な設定
        valid_config = IntegratedOperationsConfig()
        logger.info("Valid configuration created successfully")

        # 無効な設定（負の値）
        try:
            invalid_config = IntegratedOperationsConfig(
                health_check_interval_seconds=-1
            )
            logger.error("Should have failed with negative interval")
            return False
        except ValueError:
            logger.info("Correctly rejected negative interval")

        # 無効な設定（ゼロ値）
        try:
            invalid_config = IntegratedOperationsConfig(critical_error_threshold=0)
            logger.error("Should have failed with zero threshold")
            return False
        except ValueError:
            logger.info("Correctly rejected zero threshold")

        logger.info("Configuration validation test passed")
        return True

    except Exception as e:
        logger.error(f"Configuration validation test failed: {e}")
        return False


def main():
    """メイン実行関数"""
    logger.info("=== Integrated Operations Management System Test ===")

    # 設定検証テスト
    if not test_configuration_validation():
        logger.error("Configuration validation test failed")
        return 1

    # 統合運用テスト
    if not test_integrated_operations():
        logger.error("Integrated operations test failed")
        return 1

    logger.info("All tests passed successfully!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
