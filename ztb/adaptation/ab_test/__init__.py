"""
A/B Testing Framework for Automated Model Adaptation
SAC v421 Improvement Plan - Automated Model Adaptation

This module provides a comprehensive A/B testing framework designed for:
- Statistical validation of model improvements
- Memory-efficient streaming data processing
- Parallel processing for performance optimization
- Risk management with automatic rollback capabilities
- Real-time traffic management and gradual deployment

Key Components:
- types.py: Type definitions and data structures
- config.py: Configuration management with performance optimizations
- analyzer.py: Streaming statistical analysis engine
- executor.py: Test execution engine with memory monitoring
- selector.py: Model selection and rollback logic

Usage:
    from ztb.adaptation.ab_test import ABTestExecutor, ABTestConfig

    config = ABTestConfig()
    executor = ABTestExecutor(config)
    # Run A/B tests with automatic model selection
"""

from .analyzer import ABTestAnalyzer
from .config import ABTestConfig
from .executor import ABTestExecutor
from .selector import ModelSelector, TrafficManager
from .types import (
    ABTestConfiguration,
    ABTestMetrics,
    ABTestResult,
    ABTestResultSummary,
    ABTestState,
    ABTestVariant,
    RiskAssessment,
    StatisticalResult,
    StreamingStatistics,
)

__all__ = [
    # Types
    "ABTestVariant",
    "ABTestConfiguration",
    "ABTestState",
    "ABTestMetrics",
    "ABTestResult",
    "ABTestResultSummary",
    "StatisticalResult",
    "StreamingStatistics",
    "RiskAssessment",
    # Configuration
    "ABTestConfig",
    # Core Components
    "ABTestAnalyzer",
    "ABTestExecutor",
    "ModelSelector",
    "TrafficManager",
]

__version__ = "1.0.0"
__author__ = "SAC v421 Improvement Plan"
