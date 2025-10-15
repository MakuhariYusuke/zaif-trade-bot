"""
全単体テストを実行し、カバレッジレポートを生成。

Usage:
    python run_unit_tests.py
    python run_unit_tests.py --coverage
    python run_unit_tests.py --verbose
"""

import sys
import subprocess
from pathlib import Path

# プロジェクトルート
project_root = Path(__file__).parent
tests_dir = project_root / "tests" / "unit"


def run_tests(with_coverage=False, verbose=False):
    """単体テストを実行"""
    
    # pytestがインストールされているか確認
    try:
        import pytest
        print("✅ pytest is installed")
    except ImportError:
        print("❌ pytest is not installed")
        print("Installing pytest...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "pytest-cov"])
        import pytest
    
    # テスト引数を構築
    args = [
        str(tests_dir),
        "-v" if verbose else "",
        "--tb=short",
    ]
    
    if with_coverage:
        # カバレッジ付きで実行
        try:
            import pytest_cov
            print("✅ pytest-cov is installed")
        except ImportError:
            print("Installing pytest-cov...")
            subprocess.run([sys.executable, "-m", "pip", "install", "pytest-cov"])
        
        args.extend([
            "--cov=ztb.training.algorithms",
            "--cov=ztb.training.core.config_builder",
            "--cov=ztb.training.core.algorithm_trainer",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
        ])
    
    # 空文字列を削除
    args = [arg for arg in args if arg]
    
    print("\n" + "=" * 60)
    print("🧪 Running Unit Tests")
    print("=" * 60)
    print(f"Test directory: {tests_dir}")
    print(f"Coverage: {'enabled' if with_coverage else 'disabled'}")
    print(f"Verbose: {'enabled' if verbose else 'disabled'}")
    print("=" * 60 + "\n")
    
    # pytest実行
    exit_code = pytest.main(args)
    
    if exit_code == 0:
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        
        if with_coverage:
            print("\n📊 Coverage report generated:")
            print(f"   - Terminal: see above")
            print(f"   - HTML: {project_root / 'htmlcov' / 'index.html'}")
    else:
        print("\n" + "=" * 60)
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
    
    return exit_code


def print_test_summary():
    """テストファイルのサマリーを表示"""
    print("\n" + "=" * 60)
    print("📋 Test Files Summary")
    print("=" * 60)
    
    test_files = list(tests_dir.glob("test_*.py"))
    
    for test_file in sorted(test_files):
        # ファイル内のテストクラス数を数える
        content = test_file.read_text(encoding="utf-8")
        class_count = content.count("class Test")
        test_count = content.count("def test_")
        
        print(f"\n{test_file.name}:")
        print(f"  - Test classes: {class_count}")
        print(f"  - Test methods: {test_count}")
    
    print(f"\n Total test files: {len(test_files)}")
    print("=" * 60)


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run unit tests")
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Run with coverage report"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--summary", "-s",
        action="store_true",
        help="Show test files summary"
    )
    
    args = parser.parse_args()
    
    if args.summary:
        print_test_summary()
        return 0
    
    exit_code = run_tests(
        with_coverage=args.coverage,
        verbose=args.verbose
    )
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
