"""
Distributed Training Utilities for SAC Training

This module provides distributed training capabilities for SAC using PyTorch's
DDP (DistributedDataParallel) and DataParallel for multi-GPU/multi-node training.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Dict, List, Optional, Callable, Any, Union
import logging
from contextlib import contextmanager
import socket
import time

logger = logging.getLogger(__name__)


class DistributedTrainingConfig:
    """
    Configuration for distributed training setup.
    """

    def __init__(
        self,
        world_size: int = 1,
        rank: int = 0,
        master_addr: str = "127.0.0.1",
        master_port: str = "12345",
        backend: str = "gloo",  # 'gloo' for CPU, 'nccl' for GPU
        init_method: Optional[str] = None,
        timeout: int = 1800,  # 30 minutes
    ):
        """
        Initialize distributed training configuration.

        Args:
            world_size: Total number of processes
            rank: Rank of current process
            master_addr: Master node address
            master_port: Master node port
            backend: Backend for distributed training ('gloo' or 'nccl')
            init_method: Initialization method
            timeout: Timeout for distributed operations
        """
        self.world_size = world_size
        self.rank = rank
        self.master_addr = master_addr
        self.master_port = master_port
        self.backend = backend
        self.init_method = init_method
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> 'DistributedTrainingConfig':
        """Create config from environment variables."""
        return cls(
            world_size=int(os.environ.get('WORLD_SIZE', 1)),
            rank=int(os.environ.get('RANK', 0)),
            master_addr=os.environ.get('MASTER_ADDR', '127.0.0.1'),
            master_port=os.environ.get('MASTER_PORT', '12345'),
            backend=os.environ.get('DIST_BACKEND', 'gloo'),
            init_method=os.environ.get('INIT_METHOD', None),
        )

    def to_env(self) -> Dict[str, str]:
        """Convert config to environment variables."""
        return {
            'WORLD_SIZE': str(self.world_size),
            'RANK': str(self.rank),
            'MASTER_ADDR': self.master_addr,
            'MASTER_PORT': self.master_port,
            'DIST_BACKEND': self.backend,
        }


class DistributedTrainer:
    """
    Distributed training wrapper for SAC models.
    """

    def __init__(
        self,
        model: nn.Module,
        config: DistributedTrainingConfig,
        device: Optional[torch.device] = None,
        find_unused_parameters: bool = False,
    ):
        """
        Initialize distributed trainer.

        Args:
            model: Neural network model to distribute
            config: Distributed training configuration
            device: Target device (auto-detected if None)
            find_unused_parameters: Whether to find unused parameters in backward pass
        """
        self.config = config
        self.is_distributed = config.world_size > 1
        self.is_master = config.rank == 0

        # Setup device
        if device is None:
            if torch.cuda.is_available() and config.backend == 'nccl':
                self.device = torch.device(f'cuda:{config.rank % torch.cuda.device_count()}')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        # Move model to device
        self.model = model.to(self.device)

        # Setup distributed training
        if self.is_distributed:
            if config.backend == 'nccl' and not torch.cuda.is_available():
                logger.warning("NCCL backend requested but CUDA not available. Falling back to Gloo.")
                config.backend = 'gloo'

            # Only setup DDP if distributed training is initialized
            if dist.is_initialized():
                self.model = nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=[self.device.index] if self.device.type == 'cuda' else None,
                    output_device=self.device.index if self.device.type == 'cuda' else None,
                    find_unused_parameters=find_unused_parameters,
                )
                logger.info(f"Initialized DDP with rank {config.rank}/{config.world_size} on {self.device}")
            else:
                logger.warning("Distributed training not initialized, using single device training")
                self.is_distributed = False
        else:
            # Single GPU/CPU training
            if torch.cuda.device_count() > 1 and config.backend == 'nccl':
                self.model = nn.DataParallel(self.model)
                logger.info(f"Initialized DataParallel with {torch.cuda.device_count()} GPUs")
            else:
                logger.info(f"Initialized single device training on {self.device}")

        self.training_stats = {}

    def get_model(self) -> nn.Module:
        """Get the underlying model (unwrap DDP if needed)."""
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self.model.module
        return self.model

    def save_checkpoint(
        self,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        **kwargs
    ):
        """Save distributed checkpoint (only master process)."""
        if not self.is_master:
            return

        checkpoint = {
            'model_state_dict': self.get_model().state_dict(),
            'epoch': epoch,
            'distributed_config': {
                'world_size': self.config.world_size,
                'rank': self.config.rank,
                'backend': self.config.backend,
            },
            **kwargs
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load distributed checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.get_model().load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint

    def all_reduce(self, tensor: torch.Tensor, op: str = "SUM") -> torch.Tensor:
        """Perform all-reduce operation across processes."""
        if not self.is_distributed:
            return tensor

        op_map = {
            "SUM": dist.ReduceOp.SUM,
            "MEAN": dist.ReduceOp.SUM,  # Will divide by world_size
            "MAX": dist.ReduceOp.MAX,
            "MIN": dist.ReduceOp.MIN,
        }

        dist.all_reduce(tensor, op=op_map[op])

        if op == "MEAN":
            tensor.div_(self.config.world_size)

        return tensor

    def barrier(self):
        """Synchronization barrier across all processes."""
        if self.is_distributed:
            dist.barrier()

    def get_world_size(self) -> int:
        """Get total number of processes."""
        return self.config.world_size if self.is_distributed else 1

    def get_rank(self) -> int:
        """Get rank of current process."""
        return self.config.rank if self.is_distributed else 0

    def is_master_process(self) -> bool:
        """Check if current process is master."""
        return self.is_master


def setup_distributed_training(config: DistributedTrainingConfig):
    """
    Setup distributed training environment.

    Args:
        config: Distributed training configuration

    Returns:
        True if setup successful
    """
    if config.world_size <= 1:
        logger.info("Single process training - no distributed setup needed")
        return True

    try:
        # Set environment variables
        env_vars = config.to_env()
        for key, value in env_vars.items():
            os.environ[key] = value

        # Initialize process group
        if config.init_method:
            dist.init_process_group(
                backend=config.backend,
                init_method=config.init_method,
                world_size=config.world_size,
                rank=config.rank,
                timeout=torch.distributed.Timeout(seconds=config.timeout) if hasattr(torch.distributed, 'Timeout') else config.timeout
            )
        else:
            dist.init_process_group(
                backend=config.backend,
                world_size=config.world_size,
                rank=config.rank,
                timeout=torch.distributed.Timeout(seconds=config.timeout) if hasattr(torch.distributed, 'Timeout') else config.timeout
            )

        logger.info(f"Initialized distributed training: rank {config.rank}/{config.world_size}")
        return True

    except Exception as e:
        logger.error(f"Failed to setup distributed training: {e}")
        return False


def cleanup_distributed_training():
    """Cleanup distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Cleaned up distributed training")


@contextmanager
def distributed_context(config: DistributedTrainingConfig):
    """
    Context manager for distributed training.

    Usage:
        with distributed_context(config) as trainer:
            # Use trainer for distributed training
            pass
    """
    try:
        success = setup_distributed_training(config)
        if not success:
            raise RuntimeError("Failed to setup distributed training")

        yield config

    finally:
        cleanup_distributed_training()


def find_free_port() -> str:
    """Find a free port for distributed training."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
        return str(port)


def launch_distributed_training(
    world_size: int,
    train_fn: Callable,
    args: Optional[tuple] = None,
    backend: str = "gloo",
    master_addr: str = "127.0.0.1",
    master_port: Optional[str] = None,
) -> List[mp.Process]:
    """
    Launch distributed training processes.

    Args:
        world_size: Number of processes
        train_fn: Training function to run
        args: Arguments for training function
        backend: Backend for distributed training
        master_addr: Master address
        master_port: Master port (auto-assigned if None)

    Returns:
        List of process handles
    """
    if master_port is None:
        master_port = find_free_port()

    processes = []

    for rank in range(world_size):
        config = DistributedTrainingConfig(
            world_size=world_size,
            rank=rank,
            master_addr=master_addr,
            master_port=master_port,
            backend=backend,
        )

        # Set environment variables for this process
        env = os.environ.copy()
        env.update(config.to_env())

        p = mp.Process(
            target=train_fn,
            args=(config, *(args or ())),
            env=env
        )
        p.start()
        processes.append(p)

        # Small delay to avoid port conflicts
        time.sleep(0.1)

    return processes


def wait_for_processes(processes: List[mp.Process]):
    """Wait for distributed training processes to complete."""
    for p in processes:
        p.join()

    # Check for errors
    for i, p in enumerate(processes):
        if p.exitcode != 0:
            logger.error(f"Process {i} exited with code {p.exitcode}")


# Utility functions for distributed training
def reduce_loss(loss: torch.Tensor, config: DistributedTrainingConfig) -> torch.Tensor:
    """
    Reduce loss across all processes.

    Args:
        loss: Local loss tensor
        config: Distributed training configuration

    Returns:
        Reduced loss tensor
    """
    if config.world_size <= 1:
        return loss

    reduced_loss = loss.clone()
    dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
    reduced_loss.div_(config.world_size)

    return reduced_loss


def gather_tensor(tensor: torch.Tensor, config: DistributedTrainingConfig) -> List[torch.Tensor]:
    """
    Gather tensor from all processes.

    Args:
        tensor: Local tensor
        config: Distributed training configuration

    Returns:
        List of tensors from all processes
    """
    if config.world_size <= 1:
        return [tensor]

    # Gather tensors
    gathered_list = [torch.zeros_like(tensor) for _ in range(config.world_size)]
    dist.all_gather(gathered_list, tensor)

    return gathered_list


def broadcast_tensor(tensor: torch.Tensor, src_rank: int, config: DistributedTrainingConfig):
    """
    Broadcast tensor from source rank to all processes.

    Args:
        tensor: Tensor to broadcast
        src_rank: Source rank
        config: Distributed training configuration
    """
    if config.world_size <= 1:
        return

    dist.broadcast(tensor, src=src_rank)


def get_distributed_info() -> Dict[str, Any]:
    """Get information about distributed training setup."""
    if not dist.is_initialized():
        return {
            'is_distributed': False,
            'world_size': 1,
            'rank': 0,
            'backend': None,
        }

    return {
        'is_distributed': True,
        'world_size': dist.get_world_size(),
        'rank': dist.get_rank(),
        'backend': dist.get_backend(),
    }