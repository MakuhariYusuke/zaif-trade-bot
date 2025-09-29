#!/usr/bin/env python3
"""
Release Preparation Orchestrator

Automates pre-release checks and artifact preparation.
Runs go/no-go checks, canary tests, and creates release bundles.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import psutil


def run_command(cmd: list[str], cwd: Optional[Path] = None, timeout: int = 300) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"


def check_go_no_go() -> bool:
    """Run basic go/no-go checks."""
    print("🔍 Running go/no-go checks...")

    checks = [
        ("mypy", ["python", "-m", "mypy", "ztb/"]),
        ("flake8", ["python", "-m", "flake8", "ztb/"]),
        ("tests", ["python", "-m", "pytest", "ztb/tests/unit/", "-x", "--tb=short"]),
    ]

    for name, cmd in checks:
        print(f"  Checking {name}...")
        exit_code, stdout, stderr = run_command(cmd)
        if exit_code != 0:
            print(f"❌ {name} check failed:")
            print(stderr)
            return False
        print(f"✅ {name} passed")

    return True


def run_canary_test(duration_minutes: int = 3) -> tuple[bool, Dict[str, Any]]:
    """Run short canary test with fault injection."""
    print(f"🧪 Running canary test ({duration_minutes} minutes)...")

    # Run paper trader with canary mode
    cmd = [
        "python", "-m", "ztb.live.paper_trader",
        "--mode", "replay",
        "--policy", "sma_fast_slow",
        f"--duration-minutes", str(duration_minutes),
        "--enable-risk",
        "--risk-profile", "balanced",
        "--canary-mode",  # Hypothetical flag
    ]

    start_time = time.time()
    exit_code, stdout, stderr = run_command(cmd, timeout=duration_minutes * 60 + 60)

    canary_results = {
        "duration_seconds": time.time() - start_time,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "success": exit_code == 0,
    }

    if exit_code == 0:
        print("✅ Canary test passed")
        return True, canary_results
    else:
        print("❌ Canary test failed:")
        print(stderr)
        return False, canary_results


def update_executive_summary(canary_results: Dict[str, Any]) -> bool:
    """Update executive summary with latest results."""
    print("📊 Updating executive summary...")

    summary_path = Path("executive_summary.md")
    if not summary_path.exists():
        print("⚠️  executive_summary.md not found, skipping update")
        return True

    # Read current summary
    content = summary_path.read_text()

    # Add canary results
    timestamp = datetime.now().isoformat()
    canary_section = f"""
## Latest Canary Results
- **Timestamp**: {timestamp}
- **Duration**: {canary_results['duration_seconds']:.1f}s
- **Status**: {'✅ Passed' if canary_results['success'] else '❌ Failed'}

"""

    # Insert after title
    lines = content.split('\n')
    if len(lines) > 1:
        lines.insert(1, canary_section)
        new_content = '\n'.join(lines)
        summary_path.write_text(new_content)
        print("✅ Executive summary updated")
        return True
    else:
        print("⚠️  Could not update executive summary")
        return False


def create_release_bundle(output_dir: Path, canary_results: Dict[str, Any]) -> Path:
    """Create release preparation bundle."""
    print("📦 Creating release bundle...")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    bundle_name = f"release_prep_{timestamp}"
    bundle_dir = output_dir / bundle_name

    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Collect artifacts
    artifacts = {
        "canary_results.json": canary_results,
        "run_metadata.json": {
            "timestamp": timestamp,
            "type": "release_prep",
            "canary_success": canary_results["success"],
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "cpu_count": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024**3),
            }
        }
    }

    # Copy key files
    files_to_copy = [
        "executive_summary.md",
        "executive_summary.png",
        "CHANGELOG.md",
        "requirements.txt",
        "pyrightconfig.json",
        "mypy.ini",
    ]

    for file_path in files_to_copy:
        src = Path(file_path)
        if src.exists():
            shutil.copy2(src, bundle_dir / src.name)

    # Copy results directories if they exist
    result_dirs = ["results", "artifacts", "reports"]
    for result_dir in result_dirs:
        src_dir = Path(result_dir)
        if src_dir.exists():
            dst_dir = bundle_dir / result_dir
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)

    # Write artifacts
    for filename, data in artifacts.items():
        with open(bundle_dir / filename, 'w') as f:
            json.dump(data, f, indent=2)

    # Create zip archive
    zip_path = output_dir / f"{bundle_name}.zip"
    shutil.make_archive(str(zip_path.with_suffix('')), 'zip', bundle_dir)

    print(f"✅ Release bundle created: {zip_path}")
    return zip_path


def main():
    """Main release preparation workflow."""
    parser = argparse.ArgumentParser(description="Release preparation orchestrator")
    parser.add_argument("--canary-duration", type=int, default=3,
                       help="Canary test duration in minutes")
    parser.add_argument("--output-dir", default="artifacts",
                       help="Output directory for release bundle")
    parser.add_argument("--skip-canary", action="store_true",
                       help="Skip canary test")
    parser.add_argument("--force", action="store_true",
                       help="Force release prep even if checks fail")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 Starting release preparation...")

    # Step 1: Go/no-go checks
    if not check_go_no_go():
        if not args.force:
            print("❌ Go/no-go checks failed. Use --force to continue.")
            sys.exit(1)
        else:
            print("⚠️  Continuing despite failed checks (--force)")

    # Step 2: Canary test
    canary_success = True
    canary_results = {}

    if not args.skip_canary:
        canary_success, canary_results = run_canary_test(args.canary_duration)
        if not canary_success and not args.force:
            print("❌ Canary test failed. Use --force to continue.")
            sys.exit(1)
    else:
        print("⏭️  Skipping canary test")
        canary_results = {"skipped": True, "success": True}

    # Step 3: Update executive summary
    if not update_executive_summary(canary_results):
        print("⚠️  Failed to update executive summary")

    # Step 4: Create release bundle
    zip_path = create_release_bundle(output_dir, canary_results)

    # Final status
    overall_success = check_go_no_go() and canary_success

    if overall_success:
        print("🎉 Release preparation completed successfully!")
        print(f"📦 Bundle: {zip_path}")
        sys.exit(0)
    else:
        print("⚠️  Release preparation completed with warnings")
        print(f"📦 Bundle: {zip_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()