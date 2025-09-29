#!/usr/bin/env python3
"""
Go/No-Go Check Script for Zaif Trade Bot

Automatically evaluates system readiness for production deployment.
Checks various metrics and provides GO/NO-GO/CONDITIONAL recommendations.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import psutil
import time


class GoNoGoChecker:
    """Automated Go/No-Go evaluation system."""

    def __init__(self, workspace_root: Path):
        self.root = workspace_root
        self.results: Dict[str, Any] = {}
        self.checks: List[Dict[str, Any]] = []

    def add_check(self, name: str, description: str, check_func, critical: bool = True):
        """Add a check to the evaluation suite."""
        self.checks.append({
            'name': name,
            'description': description,
            'func': check_func,
            'critical': critical,
            'result': None,
            'details': None
        })

    def run_checks(self) -> bool:
        """Run all checks and return overall GO status."""
        print("🔍 Running Go/No-Go checks...\n")

        all_passed = True
        for check in self.checks:
            print(f"Checking: {check['name']}")
            try:
                result, details = check['func']()
                check['result'] = result
                check['details'] = details

                status = "✅ PASS" if result else "❌ FAIL"
                print(f"  {status}: {details}")

                if not result and check['critical']:
                    all_passed = False

            except Exception as e:
                check['result'] = False
                check['details'] = f"Check failed with error: {str(e)}"
                print(f"  ❌ ERROR: {check['details']}")
                if check['critical']:
                    all_passed = False

        print(f"\n{'='*50}")
        overall = "GO" if all_passed else "NO-GO"
        print(f"Overall Result: {overall}")
        print(f"{'='*50}")

        return all_passed

    def generate_report(self) -> Dict[str, Any]:
        """Generate detailed JSON report."""
        return {
            'timestamp': time.time(),
            'overall_result': 'GO' if all(c['result'] for c in self.checks if c['critical']) else 'NO-GO',
            'checks': [
                {
                    'name': c['name'],
                    'description': c['description'],
                    'result': c['result'],
                    'critical': c['critical'],
                    'details': c['details']
                } for c in self.checks
            ]
        }


def check_test_pipeline(checker: GoNoGoChecker) -> tuple[bool, str]:
    """Check if CI/CD pipeline tests pass."""
    try:
        # Run unit tests
        result = subprocess.run(
            ['npm', 'run', '-s', 'test:unit'],
            cwd=checker.root,
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode != 0:
            return False, f"Unit tests failed: {result.stderr[:200]}"

        # Run integration tests
        result = subprocess.run(
            ['npm', 'run', '-s', 'test:int-fast'],
            cwd=checker.root,
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode != 0:
            return False, f"Integration tests failed: {result.stderr[:200]}"

        return True, "All tests passed"
    except subprocess.TimeoutExpired:
        return False, "Test execution timed out"
    except Exception as e:
        return False, f"Test execution error: {str(e)}"


def check_dependencies(checker: GoNoGoChecker) -> tuple[bool, str]:
    """Check if all dependencies are installed."""
    try:
        # Check Python dependencies
        result = subprocess.run(
            [sys.executable, '-c', 'import ztb; print("OK")'],
            cwd=checker.root,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return False, f"Python imports failed: {result.stderr}"

        # Check Node dependencies
        if not (checker.root / 'node_modules').exists():
            return False, "Node modules not installed"

        return True, "All dependencies available"
    except Exception as e:
        return False, f"Dependency check error: {str(e)}"


def check_configuration(checker: GoNoGoChecker) -> tuple[bool, str]:
    """Check configuration files."""
    env_file = checker.root / '.env'
    if not env_file.exists():
        return False, ".env file missing"

    # Basic checks (don't validate secrets)
    with open(env_file) as f:
        content = f.read()
        if 'ZAIF_API_KEY' not in content or content.strip() == '':
            return False, ".env file incomplete"

    return True, "Configuration files present"


def check_performance(checker: GoNoGoChecker) -> tuple[bool, str]:
    """Check system performance metrics."""
    try:
        # Memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 80:
            return False, f"High memory usage: {memory.percent}%"

        # CPU usage (sample)
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 80:
            return False, f"High CPU usage: {cpu_percent}%"

        return True, f"Performance OK (Memory: {memory.percent}%, CPU: {cpu_percent}%)"
    except Exception as e:
        return False, f"Performance check error: {str(e)}"


def check_data_quality(checker: GoNoGoChecker) -> tuple[bool, str]:
    """Check data quality metrics."""
    try:
        stats_file = checker.root / 'stats.json'
        if not stats_file.exists():
            return False, "No recent stats file found"

        with open(stats_file) as f:
            stats = json.load(f)

        # Check for recent data
        if 'date' not in stats:
            return False, "Invalid stats format"

        # Check for excessive errors
        total_trades = sum(item['stats'].get('trades', 0) for item in stats.get('data', []))
        if total_trades == 0:
            return False, "No trading activity detected"

        return True, f"Data quality OK (Total trades: {total_trades})"
    except Exception as e:
        return False, f"Data quality check error: {str(e)}"


def check_security(checker: GoNoGoChecker) -> tuple[bool, str]:
    """Check security posture."""
    try:
        secrets_file = checker.root / '.secrets.baseline'
        if not secrets_file.exists():
            return False, "Secrets baseline not found"

        # Check if detect-secrets has been run recently
        mtime = os.path.getmtime(secrets_file)
        if time.time() - mtime > 86400:  # 24 hours
            return False, "Secrets scan outdated"

        return True, "Security checks passed"
    except Exception as e:
        return False, f"Security check error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='Go/No-Go Checker for Zaif Trade Bot')
    parser.add_argument('--report', action='store_true', help='Generate detailed JSON report')
    args = parser.parse_args()

    workspace_root = Path(__file__).parent.parent
    checker = GoNoGoChecker(workspace_root)

    # Add all checks
    checker.add_check(
        "CI/CD Pipeline",
        "Unit and integration tests pass",
        lambda: check_test_pipeline(checker)
    )
    checker.add_check(
        "Dependencies",
        "All required packages installed",
        lambda: check_dependencies(checker)
    )
    checker.add_check(
        "Configuration",
        "Environment and config files valid",
        lambda: check_configuration(checker)
    )
    checker.add_check(
        "Performance",
        "System resources within limits",
        lambda: check_performance(checker)
    )
    checker.add_check(
        "Data Quality",
        "Trading data and stats valid",
        lambda: check_data_quality(checker)
    )
    checker.add_check(
        "Security",
        "Secrets scanning completed",
        lambda: check_security(checker),
        critical=False  # Warning only
    )

    # Run checks
    success = checker.run_checks()

    if args.report:
        report = checker.generate_report()
        print("\n📊 Detailed Report:")
        print(json.dumps(report, indent=2, ensure_ascii=False))

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()