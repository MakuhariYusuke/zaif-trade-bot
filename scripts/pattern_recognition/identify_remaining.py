#!/usr/bin/env python3
"""
Identify remaining files that need refactoring
"""

import subprocess

# 既にリファクタリングしたファイル
refactored_files = [
    "ztb/analysis/analyze_backtest_detailed.py",
    "ztb/analysis/advanced_sac_v434_1_analysis.py",
    "ztb/analysis/auto_feature_generator.py",
    "ztb/analysis/backtest_model.py",
    "ztb/analysis/backtest_sac_v423b.py",
    "ztb/analysis/backtest_v381_v384.py",
]

# sharpe_ratioを使用しているファイルを取得
result = subprocess.run(
    ["git", "grep", "-l", "sharpe_ratio", "--", "*.py"],
    capture_output=True,
    text=True,
    cwd=".",
)

if result.returncode == 0:
    all_files = [line.strip() for line in result.stdout.split("\n") if line.strip()]

    remaining_files = [
        f for f in all_files if f not in refactored_files and "ztb/analysis/" in f
    ]

    print(f"残りの分析ファイル数: {len(remaining_files)}")
    for i, f in enumerate(remaining_files[:15]):  # 最初の15個を表示
        print(f"{i+1}. {f}")
    if len(remaining_files) > 15:
        print(f"... 他 {len(remaining_files) - 15} ファイル")

    # 優先度の高いファイルを特定（backtest_で始まるファイル）
    priority_files = [f for f in remaining_files if "backtest" in f.lower()]
    print(f"\n優先度高（backtest関連）: {len(priority_files)} ファイル")
    for f in priority_files[:5]:
        print(f"  - {f}")
else:
    print("Error running git grep")
