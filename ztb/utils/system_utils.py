#!/usr/bin/env python3
"""
System utilities for environment and hardware management.
"""

import os
from typing import Any, Dict


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
