#!/usr/bin/env python3
"""
System utilities for environment and hardware management.
"""

import importlib
import os
from typing import Any, Dict, Optional

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def check_library_availability(library_name: str, feature_name: str) -> bool:
    """
    ライブラリの利用可能性をチェックし、結果をログ出力

    Args:
        library_name: インポートするライブラリ名
        feature_name: 機能の説明名

    Returns:
        bool: ライブラリが利用可能かどうか
    """
    try:
        importlib.import_module(library_name)
        return True
    except ImportError:
        logger.warning(f"{library_name} not available. {feature_name} will be disabled.")
        return False


def safe_import(library_name: str, feature_name: str) -> Optional[Any]:
    """
    安全なライブラリインポートを実行

    Args:
        library_name: インポートするライブラリ名
        feature_name: 機能の説明名

    Returns:
        インポートされたモジュール、またはNone
    """
    try:
        return importlib.import_module(library_name)
    except ImportError:
        logger.warning(f"{library_name} not available. {feature_name} will be disabled.")
        return None


def create_library_flags() -> Dict[str, bool]:
    """
    一般的なライブラリの利用可能性フラグを作成

    Returns:
        Dict[str, bool]: ライブラリ名をキー、利用可能性を値とする辞書
    """
    libraries = {
        'optuna': 'Bayesian optimization',
        'tqdm': 'Progress bars',
        'psutil': 'System monitoring',
        'scipy': 'Statistical functions',
        'sklearn': 'Machine learning',
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'matplotlib': 'Plotting',
        'seaborn': 'Statistical visualization'
    }

    flags = {}
    for lib_name, feature_desc in libraries.items():
        flags[f"{lib_name.upper()}_AVAILABLE"] = check_library_availability(lib_name, feature_desc)

    return flags


def configure_pytorch_environment(cuda_optimizations: bool = True) -> None:
    """
    Configure PyTorch environment variables for optimal performance.

    Args:
        cuda_optimizations: Whether to enable CUDA optimizations
    """
    # Basic PyTorch settings
    os.environ["PYTORCH_DISABLE_TORCH_DYNAMO"] = "1"

    if cuda_optimizations:
        # CUDA optimizations
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    else:
        # Disable CUDA to reduce memory usage
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    # Threading optimization
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def get_system_info() -> Dict[str, Any]:
    """Get basic system information."""
    return {
        "cuda_available": os.environ.get("CUDA_VISIBLE_DEVICES", "") != "",
        "pytorch_dynamo_disabled": os.environ.get("PYTORCH_DISABLE_TORCH_DYNAMO")
        == "1",
        "memory_optimized": "PYTORCH_CUDA_ALLOC_CONF" in os.environ,
    }
