#!/usr/bin/env python3
"""
Refactored Unified Trainer implementation with enhanced UI and modularity.
"""

import logging
from typing import Any, Dict, Optional, List
import time
import torch
import torch.nn as nn

# Try to import federated learning and mixed precision dependencies
try:
    import opacus
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False

try:
    from torch.amp import GradScaler
    AMP_AVAILABLE = True
except ImportError:
    try:
        from torch.cuda.amp import GradScaler
        AMP_AVAILABLE = True
    except ImportError:
        AMP_AVAILABLE = False

from ztb.training.unified_trainer.algorithms import create_algorithm_trainer
from ztb.training.unified_trainer.config_manager import ConfigManager
from ztb.training.unified_trainer.reporting import TrainingReporter
from ztb.training.unified_trainer.ui import TrainingUI
from ztb.utils.logging_utils import get_logger

# Import optimization utilities
from ztb.utils.memory_utils import MemoryTracker, optimize_array_dtype, temporary_array, memory_efficient_processing
from ztb.utils.performance_profiler import PerformanceProfiler
from ztb.utils.cache_utils import TTLCache
from ztb.utils.parallel_experiments import ParallelExperimentConfig

# Import system optimizer
from ztb.optimization.system_optimizer import SystemOptimizer, MemoryOptimizer, PerformanceOptimizer

# Import distributed training utilities
from ztb.training.distributed.distributed_training import (
    DistributedTrainingConfig,
    DistributedTrainer,
    setup_distributed_training,
    cleanup_distributed_training,
)

# Import quantization and compression utilities
from ztb.training.quantization.quantizer import SACQuantizer, QuantizationPipeline
from ztb.training.distillation.dist
from ztb.training.compression.compressor import CompositeCompressor


class UnifiedTrainer:
    """
    Refactored Unified training interface with enhanced UI and modularity.

    WORK ASSIGNMENT:
    ---------------
    - PPO Algorithm: @trading-team - Standard RL training, evaluation, logging
    - Base ML Algorithm: @ml-research-team - Custom experiments, prototyping
    - Iterative Algorithm: @production-team - Long-running training, monitoring
    """

    def __init__(
        self,
        config: Dict[str, Any],
        force: bool = False,
        dry_run: bool = False,
        enable_streaming: bool = False,
        stream_batch_size: int = 256,
        max_features: Optional[int] = None,
        total_timesteps: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        enable_distributed: bool = False,
        world_size: int = 1,
        distributed_backend: str = "gloo",
    ):
        """
        Initialize UnifiedTrainer.

        Args:
            config: Training configuration dictionary
            force: Force execution without prompts
            dry_run: Validate without executing
            enable_streaming: Enable streaming data pipeline
            stream_batch_size: Batch size for streaming
            max_features: Maximum number of features
            total_timesteps: Override total_timesteps from config (for quick validation runs)
            gradient_accumulation_steps: Number of steps to accumulate gradients
            enable_distributed: Enable distributed training
            world_size: Number of processes for distributed training
            distributed_backend: Backend for distributed training ('gloo' or 'nccl')
        """
        # Store configuration
        self.config = config
        self.force = force
        self.dry_run = dry_run
        self.enable_streaming = enable_streaming
        self.stream_batch_size = stream_batch_size
        self.max_features = max_features
        self.total_timesteps = total_timesteps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.enable_distributed = enable_distributed
        self.world_size = world_size
        self.distributed_backend = distributed_backend

        # Initialize components
        self.logger = get_logger(__name__)
        self.ui = TrainingUI(self.logger)
        self.config_manager = ConfigManager(self.logger)
        self.reporter = TrainingReporter(self.logger)

        # Algorithm trainer (created during run)
        self.algorithm_trainer = None

        # Federated Learning components
        self.federated_clients = []
        self.global_model_state = None

        # Distributed Training components
        self.distributed_config = None
        self.distributed_trainer = None

        # Mixed Precision Training components
        self.grad_scaler = None
        if AMP_AVAILABLE and config.get('enable_mixed_precision', False):
            try:
                self.grad_scaler = GradScaler()
            except Exception as e:
                self.logger.warning(f"Failed to initialize GradScaler: {e}")
                self.grad_scaler = None

        # Initialize optimization utilities
        self.memory_tracker = MemoryTracker()
        self.performance_profiler = PerformanceProfiler()
        self.feature_cache = TTLCache(ttl_seconds=300)  # 5 minute TTL for feature computations

        # Initialize system optimizer for comprehensive optimizations
        self.system_optimizer = SystemOptimizer(
            enable_memory_tracking=config.get('enable_memory_tracking', True),
            enable_performance_profiling=config.get('enable_performance_profiling', True),
            enable_io_caching=config.get('enable_io_caching', True),
            memory_threshold_mb=config.get('memory_threshold_mb', 100.0),
            cache_ttl_seconds=config.get('cache_ttl_seconds', 300),
            gc_interval_steps=config.get('gc_interval_steps', 100),
        )

        # Parallel config will be initialized when needed for parallel experiments
        self.parallel_config = None

        # Training results
        self.training_success = False
        self.training_stats = {}
        self.training_report = {}

    def run(self) -> bool:
        """
        Execute training based on configured algorithm.

        Returns:
            bool: True if training completed successfully
        """
        try:
            # Display header
            algorithm = self.config.get('algorithm', 'unknown')
            config_name = self.config.get('model_name', 'unnamed')
            self.ui.print_header(algorithm, config_name)

            # Display configuration summary
            self.ui.print_config_summary(self.config)

            # Validate configuration
            if not self._validate_configuration():
                return False

            # Handle dry run
            if self.dry_run:
                self.ui.print_info("Dry run mode: validation completed successfully")
                return True

            # Execute training
            return self._execute_training()

        except Exception as e:
            self.ui.print_error(f"Training execution failed: {e}")
            self.logger.error(f"Training execution failed: {e}", exc_info=True)
            return False

    def _apply_system_optimizations(self) -> None:
        """Apply system-level optimizations to the training setup."""
        try:
            # Optimize model memory usage
            if hasattr(self.algorithm_trainer, 'model'):
                self.algorithm_trainer.model = self.system_optimizer.optimize_model_memory(
                    self.algorithm_trainer.model
                )

            # Optimize dataloader if available
            if hasattr(self.algorithm_trainer, 'dataloader'):
                self.algorithm_trainer.dataloader = self.system_optimizer.optimize_dataloader(
                    self.algorithm_trainer.dataloader
                )

            # Enable performance optimizations
            PerformanceOptimizer.enable_torch_optimizations()
            PerformanceOptimizer.optimize_numpy_operations()

            # Log optimization status
            system_stats = self.system_optimizer.get_system_stats()
            self.logger.info(f"System optimizations applied: {system_stats}")

        except Exception as e:
            self.logger.warning(f"Failed to apply some system optimizations: {e}")

    def _validate_configuration(self) -> bool:
        """Validate configuration using enhanced validator."""
        self.logger.info("Validating configuration...")

        # Use the algorithm trainer's validation if available
        algorithm = self.config.get('algorithm', '').lower()

        try:
            # Create algorithm trainer for validation
            trainer = create_algorithm_trainer(algorithm, self.config, self.logger)

            # Validate using trainer
            is_valid = trainer.validate_config()

            if is_valid:
                self.ui.print_success("Configuration validation passed")
                return True
            else:
                self.ui.print_error("Configuration validation failed")
                return False

        except ValueError as e:
            self.ui.print_error(f"Invalid algorithm: {e}")
            return False
        except Exception as e:
            self.ui.print_error(f"Configuration validation error: {e}")
            return False

    def _execute_training(self) -> bool:
        """Execute the actual training."""
        algorithm = self.config.get('algorithm', '').lower()

        try:
            # Override total_timesteps from command line if provided
            if self.total_timesteps is not None:
                self.config['total_timesteps'] = self.total_timesteps
                self.logger.info(f"Overriding total_timesteps from command line: {self.total_timesteps:,}")

            # Check for distributed training
            if self.enable_distributed and self.world_size > 1:
                self.logger.info(f"Distributed training enabled with {self.world_size} processes")
                if not self._setup_distributed_training():
                    self.ui.print_error("Failed to setup distributed training")
                    return False

            # Check for federated learning
            if self.config.get('enable_federated', False):
                self.logger.info("Federated learning enabled")
                if not self._setup_federated_learning():
                    self.ui.print_error("Failed to setup federated learning")
                    return False

            # Check for mixed precision training
            if self.config.get('enable_mixed_precision', False):
                self.logger.info("Mixed precision training enabled")
                if not self._setup_mixed_precision():
                    self.ui.print_error("Failed to setup mixed precision training")
                    return False

            # Create algorithm trainer
            self.logger.info(f"Creating {algorithm.upper()} trainer...")
            self.algorithm_trainer = create_algorithm_trainer(
                algorithm, 
                self.config, 
                self.logger, 
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                system_optimizer=self.system_optimizer
            )

            # Apply system optimizations before training
            self.logger.info("Applying system-level optimizations...")
            self._apply_system_optimizations()

            # Start training UI
            self.ui.start_training()

            # Initialize optimization tracking
            self.logger.info("Initializing performance optimization tracking...")
            self.memory_tracker.__enter__()
            start_time = time.time()

            # Execute training (federated or regular)
            self.logger.info(f"Starting {algorithm.upper()} training...")
            if self.config.get('enable_federated', False):
                success = self._execute_federated_training()
            else:
                success = self.algorithm_trainer.train()

            # Stop optimization tracking and collect metrics
            training_time = time.time() - start_time
            self.memory_tracker.__exit__(None, None, None)
            memory_stats = f"Training completed in {training_time:.2f} seconds"
            perf_report = f"Total training time: {training_time:.2f}s"

            # Log optimization metrics
            self.logger.info("Training performance metrics:")
            self.logger.info(f"Memory usage: {memory_stats}")
            self.logger.info(f"Performance profile: {perf_report}")

            # Log system optimization statistics
            system_stats = self.system_optimizer.get_system_stats()
            self.logger.info("System optimization statistics:")
            for key, value in system_stats.items():
                self.logger.info(f"  {key}: {value}")

            # Get training statistics
            if success and hasattr(self.algorithm_trainer, 'get_training_stats'):
                self.training_stats = self.algorithm_trainer.get_training_stats()
                # Add optimization metrics to training stats
                self.training_stats['optimization'] = {
                    'memory_stats': memory_stats,
                    'performance_profile': perf_report,
                    'parallel_processing_enabled': self.parallel_config is not None,
                    'cache_size': len(self.feature_cache.cache) if hasattr(self.feature_cache, 'cache') else 0,
                    'data_optimization_applied': True
                }            # Display completion
            self.ui.print_training_complete(success, self.training_stats if success else None)

            # Generate and save training report
            if success:
                self.training_report = self.reporter.generate_report(
                    self.config, self.training_stats, success
                )
                report_path = self.reporter.save_report(self.training_report)
                self.reporter.print_summary(self.training_report)

                if report_path:
                    self.ui.print_success(f"Training report saved to: {report_path}")

            self.training_success = success
            return success

        except Exception as e:
            self.ui.print_error(f"Training execution failed: {e}")
            self.logger.error(f"Training execution failed: {e}", exc_info=True)
            return False

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return self.training_stats.copy()

    def get_training_report(self) -> Dict[str, Any]:
        """Get complete training report."""
        return self.training_report.copy()

    def is_training_complete(self) -> bool:
        """Check if training completed successfully."""
        return self.training_success

    def _setup_federated_learning(self) -> bool:
        """
        Setup federated learning components.

        Returns:
            bool: True if setup successful
        """
        try:
            if not OPACUS_AVAILABLE:
                self.logger.warning("Opacus not available. Federated learning will work without differential privacy.")
            
            num_clients = self.config.get('num_clients', 3)
            self.federated_clients = []
            
            # Initialize federated clients (simplified - in real implementation, 
            # each client would have its own data and trainer)
            for i in range(num_clients):
                client_config = self.config.copy()
                client_config['client_id'] = i
                self.federated_clients.append(client_config)
            
            self.logger.info(f"Initialized {num_clients} federated clients")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup federated learning: {e}")
            return False

    def _setup_mixed_precision(self) -> bool:
        """
        Setup mixed precision training components.

        Returns:
            bool: True if setup successful
        """
        try:
            if not AMP_AVAILABLE:
                self.logger.error("PyTorch AMP not available. Mixed precision training requires PyTorch >= 1.6")
                return False
            
            if not torch.cuda.is_available():
                self.logger.warning("CUDA not available. Mixed precision training may not provide benefits on CPU.")
            
            # GradScaler is already initialized in __init__
            self.logger.info("Mixed precision training setup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup mixed precision training: {e}")
            return False

    def _execute_federated_training(self) -> bool:
        """
        Execute federated learning training.

        Returns:
            bool: True if training successful
        """
        try:
            num_rounds = self.config.get('federated_rounds', 10)
            client_fraction = self.config.get('client_fraction', 1.0)
            
            self.logger.info(f"Starting federated training with {num_rounds} rounds")
            
            # Initialize global model
            global_success = self.algorithm_trainer.train()
            if not global_success:
                self.logger.error("Initial global model training failed")
                return False
            
            # Get initial global model state
            if hasattr(self.algorithm_trainer, 'get_model_state'):
                self.global_model_state = self.algorithm_trainer.get_model_state()
            
            # Federated learning rounds
            for round_num in range(num_rounds):
                self.logger.info(f"Federated round {round_num + 1}/{num_rounds}")
                
                # Select participating clients
                num_participants = max(1, int(len(self.federated_clients) * client_fraction))
                participating_clients = self.federated_clients[:num_participants]
                
                client_updates = []
                
                # Client training (simplified - in real implementation, 
                # each client would train on their local data)
                for client_config in participating_clients:
                    self.logger.debug(f"Training client {client_config['client_id']}")
                    
                    # Create client trainer with global model state
                    client_trainer = create_algorithm_trainer(
                        self.config.get('algorithm', '').lower(), 
                        client_config, 
                        self.logger
                    )
                    
                    # Load global model state
                    if hasattr(client_trainer, 'set_model_state') and self.global_model_state:
                        client_trainer.set_model_state(self.global_model_state)
                    
                    # Client training (reduced timesteps for federated learning)
                    client_timesteps = self.config.get('total_timesteps', 10000) // 10
                    client_config['total_timesteps'] = client_timesteps
                    
                    client_success = client_trainer.train()
                    if client_success and hasattr(client_trainer, 'get_model_state'):
                        client_updates.append(client_trainer.get_model_state())
                
                # Aggregate client updates (Federated Averaging)
                if client_updates:
                    self.global_model_state = self._federated_average(client_updates)
                    
                    # Update global model
                    if hasattr(self.algorithm_trainer, 'set_model_state'):
                        self.algorithm_trainer.set_model_state(self.global_model_state)
                
                self.ui.print_info(f"Completed federated round {round_num + 1}/{num_rounds}")
            
            # Final global model training/refinement
            self.logger.info("Performing final global model refinement")
            final_success = self.algorithm_trainer.train()
            
            if final_success and hasattr(self.algorithm_trainer, 'get_training_stats'):
                self.training_stats = self.algorithm_trainer.get_training_stats()
            
            return final_success
            
        except Exception as e:
            self.logger.error(f"Federated training failed: {e}")
            return False

    def _federated_average(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform federated averaging of client model updates.

        Args:
            client_updates: List of client model states

        Returns:
            Averaged global model state
        """
        try:
            if not client_updates:
                return {}
            
            # Simple averaging (in real implementation, this would be more sophisticated)
            averaged_state = {}
            num_clients = len(client_updates)
            
            # Get all parameter keys from first client
            param_keys = client_updates[0].keys()
            
            for key in param_keys:
                if key in client_updates[0]:
                    # Average parameters across clients
                    param_sum = None
                    for client_state in client_updates:
                        if key in client_state:
                            if param_sum is None:
                                param_sum = client_state[key].clone()
                            else:
                                param_sum += client_state[key]
                    
                    if param_sum is not None:
                        averaged_state[key] = param_sum / num_clients
            
            return averaged_state
            
        except Exception as e:
            self.logger.error(f"Federated averaging failed: {e}")
            return {}

    def _apply_mixed_precision(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Apply mixed precision training to loss computation.

        Args:
            loss: Computed loss tensor

        Returns:
            Scaled loss for backward pass
        """
        if self.grad_scaler is not None and self.config.get('enable_mixed_precision', False):
            # Scale loss for mixed precision training
            return self.grad_scaler.scale(loss)
        return loss

    def _step_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Perform optimizer step with mixed precision support.

        Args:
            optimizer: PyTorch optimizer
        """
        if self.grad_scaler is not None and self.config.get('enable_mixed_precision', False):
            # Unscale gradients and step optimizer
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()
        else:
            optimizer.step()

    def _setup_distributed_training(self) -> bool:
        """
        Setup distributed training environment.

        Returns:
            bool: True if setup successful
        """
        try:
            # Create distributed training configuration
            self.distributed_config = DistributedTrainingConfig.from_env()
            self.distributed_config.world_size = self.world_size
            self.distributed_config.backend = self.distributed_backend

            # Setup distributed training
            success = setup_distributed_training(self.distributed_config)
            if not success:
                return False

            self.logger.info(f"Distributed training setup complete: rank {self.distributed_config.rank}/{self.distributed_config.world_size}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to setup distributed training: {e}")
            return False