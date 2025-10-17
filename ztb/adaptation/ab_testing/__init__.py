"""
A/B Testing Framework Module
新旧モデルの比較評価と自動ロールバックシステム

Features:
- Traffic splitting: 確率的/時間ベース/条件ベース分割
- Statistical testing: t-test, Mann-Whitney, confidence intervals
- Auto-rollback: パフォーマンス低下時の自動切り戻し
- Multi-armed bandit: 最適モデル選択の自動化
"""

from .framework import ABTestingFramework
from .config import ABTestingConfig
from .types import TestVariant, TestResult

__all__ = [
    "ABTestingFramework",
    "ABTestingConfig",
    "TestVariant",
    "TestResult",
]