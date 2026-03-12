#!/usr/bin/env python3
"""
Phase 4.5 P1: バックグラウンド実行用ラッパー

PowerShellのStart-Jobで実行し、KeyboardInterrupt問題を回避。
"""

import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent

def main():
    script_path = project_root / "scripts" / "v459" / "run_phase45_p1.py"
    log_file = project_root / "results" / "phase45_p1_baseline" / "p1_execution.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting P1 experiment in background...")
    print(f"Script: {script_path}")
    print(f"Log: {log_file}")
    
    # シンプルにPython実行（バックグラウンドではない）
    with open(log_file, 'w') as f:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(project_root),
        )
    
    print(f"Exit code: {result.returncode}")
    print(f"Check log: {log_file}")

if __name__ == "__main__":
    main()
