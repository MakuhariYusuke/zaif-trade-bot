#!/usr/bin/env python3
"""
Distributed training utilities.
"""

import os
from typing import Any

def get_distributed_info() -> dict[str, Any]:
    """
    Get distributed training information.

    Returns:
        Dictionary containing distributed training info
    """
    info = {
        "is_distributed": False,
        "world_size": 1,
        "rank": 0,
        "local_rank": 0,
        "master_addr": "localhost",
        "master_port": "12345",
    }

    # Check for distributed environment variables
    if "WORLD_SIZE" in os.environ:
        info["is_distributed"] = True
        info["world_size"] = int(os.environ.get("WORLD_SIZE", 1))
        info["rank"] = int(os.environ.get("RANK", 0))
        info["local_rank"] = int(os.environ.get("LOCAL_RANK", 0))
        info["master_addr"] = os.environ.get("MASTER_ADDR", "localhost")
        info["master_port"] = os.environ.get("MASTER_PORT", "12345")

    return info

def is_master_process() -> bool:
    """
    Check if current process is the master process.

    Returns:
        True if master process, False otherwise
    """
    info = get_distributed_info()
    return info["rank"] == 0

def get_device_count() -> int:
    """
    Get number of available devices.

    Returns:
        Number of devices
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
        else:
            return 1  # CPU only
    except ImportError:
        return 1

def setup_distributed_training(
    backend: str = "nccl", init_method: str = "env://"
) -> Any | None:
    """
    Setup distributed training.

    Args:
        backend: Backend for distributed training
        init_method: Initialization method

    Returns:
        Process group if distributed, None otherwise
    """
    info = get_distributed_info()

    if not info["is_distributed"]:
        return None

    try:
        import torch.distributed as dist

        # Initialize the process group
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=info["world_size"],
            rank=info["rank"],
        )

        return dist.group.WORLD

    except ImportError:
        return None
