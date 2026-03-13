#!/usr/bin/env python3
"""
V433 Phase 5 Integration Test Runner
"""

import asyncio
import os
import sys

# パス追加 - プロジェクトルートを追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tests.test_v433_phase5_integration import TestV433Phase5Integration


async def run_integration_tests(specific_test=None):
    """統合テスト実行"""
    print("Starting V433 Phase 5 Integration Tests...")
    print("=" * 50)

    test_instance = TestV433Phase5Integration()

    # セットアップ
    try:
        await test_instance.asyncSetUp()
        print("✓ Test setup completed")
    except Exception as e:
        print(f"✗ Test setup failed: {e}")
        return False

    # テスト実行
    test_results = {}

    test_methods = [
        ("test_paper_trading_integration", "Paper Trading Integration"),
        ("test_parallel_running_integration", "Parallel Running Integration"),
        ("test_gradual_rollout_integration", "Gradual Rollout Integration"),
        ("test_monitoring_integration", "Monitoring Integration"),
        ("test_emergency_control_integration", "Emergency Control Integration"),
        ("test_failure_recovery_integration", "Failure Recovery Integration"),
        ("test_performance_under_load", "Performance Under Load"),
        ("test_full_system_integration", "Full System Integration"),
    ]

    if specific_test:
        test_methods = [(m, d) for m, d in test_methods if m == specific_test]

    for method_name, display_name in test_methods:
        try:
            method = getattr(test_instance, method_name)
            await method()
            print(f"✓ {display_name}: PASSED")
            test_results[method_name] = True
        except Exception as e:
            print(f"✗ {display_name}: FAILED - {e}")
            test_results[method_name] = False

    # 完全システム統合テスト
    try:
        await test_instance.test_full_system_integration()
        print("✓ Full System Integration: PASSED")
        test_results["test_full_system_integration"] = True
    except Exception as e:
        print(f"✗ Full System Integration: FAILED - {e}")
        test_results["test_full_system_integration"] = False

    # クリーンアップ
    try:
        await test_instance.asyncTearDown()
        print("✓ Test cleanup completed")
    except Exception as e:
        print(f"✗ Test cleanup failed: {e}")

    # 結果集計
    print("\n" + "=" * 50)
    print("INTEGRATION TEST RESULTS")
    print("=" * 50)

    total_tests = len(test_results)
    passed_tests = sum(test_results.values())

    for method_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        display_name = method_name.replace("test_", "").replace("_", " ").title()
        print(f"{display_name}: {status}")

    print(f"\nSUMMARY: {passed_tests}/{total_tests} tests passed")

    success_rate = (passed_tests / total_tests) * 100
    print(f"Success Rate: {success_rate:.1f}%")
    if success_rate >= 80:
        print("🎉 INTEGRATION TESTS SUCCESSFUL!")
        return True
    else:
        print("❌ INTEGRATION TESTS FAILED - Check failures above")
        return False


if __name__ == "__main__":
    specific_test = sys.argv[1] if len(sys.argv) > 1 else None
    success = asyncio.run(run_integration_tests(specific_test))
    sys.exit(0 if success else 1)
