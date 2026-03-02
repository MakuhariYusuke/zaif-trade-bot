"""
SAC v421 Model Compression Module

This module provides comprehensive model compression techniques including
pruning, low-rank approximation, and composite compression pipelines.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from scipy.linalg import svd

logger = logging.getLogger(__name__)

class SACPruner:
    """
    Advanced pruning techniques for SAC models.
    """

    def __init__(self, pruning_config: dict | None = None):
        """
        Initialize the SAC pruner.

        Args:
            pruning_config: Configuration dictionary for pruning
        """
        self.config = pruning_config or self._get_default_config()

    def _get_default_config(self) -> dict:
        """Get default pruning configuration."""
        return {
            "method": "l1_unstructured",  # 'l1_unstructured', 'l2_unstructured', 'structured'
            "amount": 0.2,  # Pruning amount (0.0-1.0)
            "target_modules": [nn.Linear, nn.Conv1d, nn.Conv2d],
            "skip_modules": [],
            "pruning_schedule": "one_shot",  # 'one_shot', 'iterative', 'gradual'
            "iterations": 10,  # For iterative pruning
            "retraining_epochs": 5,  # Epochs to retrain after pruning
        }

    def analyze_pruning_opportunities(self, model: nn.Module) -> dict:
        """
        Analyze model for pruning opportunities.

        Args:
            model: Model to analyze

        Returns:
            Analysis results dictionary
        """
        analysis = {
            "total_parameters": 0,
            "prunable_parameters": 0,
            "prunable_modules": [],
            "estimated_pruning_ratio": 0.0,
            "parameter_importance": {},
        }

        for name, module in model.named_modules():
            if not list(module.children()):  # Leaf module
                if self._is_prunable_module(module):
                    param_count = sum(p.numel() for p in module.parameters())
                    analysis["prunable_parameters"] += param_count
                    analysis["prunable_modules"].append(
                        {
                            "name": name,
                            "type": type(module).__name__,
                            "parameters": param_count,
                        }
                    )

                    # Calculate parameter importance (L1 norm)
                    importance = 0.0
                    for param_name, param in module.named_parameters():
                        if "weight" in param_name:
                            importance += torch.norm(param, p=1).item()

                    analysis["parameter_importance"][name] = importance

                total_params = sum(p.numel() for p in module.parameters())
                analysis["total_parameters"] += total_params

        if analysis["total_parameters"] > 0:
            analysis["estimated_pruning_ratio"] = (
                analysis["prunable_parameters"] / analysis["total_parameters"]
            )

        return analysis

    def _is_prunable_module(self, module: nn.Module) -> bool:
        """Check if a module is prunable."""
        return (
            type(module) in self.config["target_modules"]
            and type(module) not in self.config["skip_modules"]
        )

    def apply_pruning(self, model: nn.Module) -> tuple[nn.Module, dict]:
        """
        Apply pruning to the model.

        Args:
            model: Model to prune

        Returns:
            tuple of (pruned_model, pruning_stats)
        """
        logger.info("Starting model pruning...")

        # Analyze model first
        analysis = self.analyze_pruning_opportunities(model)

        if analysis["prunable_parameters"] == 0:
            logger.warning("No prunable modules found, returning original model")
            return model, {"status": "skipped", "reason": "no_prunable_modules"}

        try:
            if self.config["pruning_schedule"] == "one_shot":
                pruned_model = self._apply_one_shot_pruning(model)
            elif self.config["pruning_schedule"] == "iterative":
                pruned_model = self._apply_iterative_pruning(model)
            elif self.config["pruning_schedule"] == "gradual":
                pruned_model = self._apply_gradual_pruning(model)
            else:
                raise ValueError(
                    f"Unknown pruning schedule: {self.config['pruning_schedule']}"
                )

            # Collect pruning statistics
            stats = self._collect_pruning_stats(model, pruned_model, analysis)

            logger.info(
                f"Pruning completed. Removed {stats['parameters_removed']} parameters"
            )
            return pruned_model, stats

        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            return model, {"status": "failed", "error": str(e)}

    def _apply_one_shot_pruning(self, model: nn.Module) -> nn.Module:
        """Apply one-shot pruning."""
        for name, module in model.named_modules():
            if self._is_prunable_module(module):
                # Apply L1 unstructured pruning
                prune.l1_unstructured(
                    module, name="weight", amount=self.config["amount"]
                )

                # Make pruning permanent
                prune.remove(module, "weight")

        return model

    def _apply_iterative_pruning(self, model: nn.Module) -> nn.Module:
        """Apply iterative pruning with retraining."""
        # This would require integration with training loop
        # For now, implement as one-shot
        logger.warning("Iterative pruning not fully implemented, using one-shot")
        return self._apply_one_shot_pruning(model)

    def _apply_gradual_pruning(self, model: nn.Module) -> nn.Module:
        """Apply gradual pruning."""
        # This would require integration with training loop
        # For now, implement as one-shot
        logger.warning("Gradual pruning not fully implemented, using one-shot")
        return self._apply_one_shot_pruning(model)

    def _collect_pruning_stats(
        self, original_model: nn.Module, pruned_model: nn.Module, analysis: dict
    ) -> dict:
        """Collect comprehensive pruning statistics."""
        original_params = sum(p.numel() for p in original_model.parameters())
        pruned_params = sum(p.numel() for p in pruned_model.parameters())

        stats = {
            "status": "success",
            "original_parameters": original_params,
            "pruned_parameters": pruned_params,
            "parameters_removed": original_params - pruned_params,
            "compression_ratio": pruned_params / original_params
            if original_params > 0
            else 0,
            "pruning_method": self.config["method"],
            "pruning_amount": self.config["amount"],
            "pruned_modules": len(analysis["prunable_modules"]),
        }

        return stats

class LowRankApproximator:
    """
    Low-rank approximation techniques for model compression.
    """

    def __init__(self, lra_config: dict | None = None):
        """
        Initialize the low-rank approximator.

        Args:
            lra_config: Configuration dictionary for low-rank approximation
        """
        self.config = lra_config or self._get_default_config()

    def _get_default_config(self) -> dict:
        """Get default low-rank approximation configuration."""
        return {
            "method": "svd",  # 'svd', 'tucker'
            "rank_ratio": 0.5,  # Rank ratio to keep (0.0-1.0)
            "target_modules": [nn.Linear],
            "auto_rank_selection": True,
            "rank_selection_threshold": 0.95,  # Cumulative energy threshold
        }

    def analyze_low_rank_opportunities(self, model: nn.Module) -> dict:
        """
        Analyze model for low-rank approximation opportunities.

        Args:
            model: Model to analyze

        Returns:
            Analysis results dictionary
        """
        analysis = {
            "compressible_modules": [],
            "estimated_compression": 0.0,
            "rank_distributions": {},
        }

        for name, module in model.named_modules():
            if self._is_lra_module(module):
                if hasattr(module, "weight") and module.weight is not None:
                    weight = module.weight.data
                    if weight.dim() >= 2:
                        # Calculate approximate rank using SVD
                        try:
                            U, s, Vt = svd(weight.cpu().numpy(), full_matrices=False)
                            cumulative_energy = np.cumsum(s**2) / np.sum(s**2)

                            # Find effective rank
                            effective_rank = (
                                np.searchsorted(
                                    cumulative_energy,
                                    self.config["rank_selection_threshold"],
                                )
                                + 1
                            )
                            max_rank = min(weight.shape[-2:])

                            compression_ratio = effective_rank / max_rank

                            analysis["compressible_modules"].append(
                                {
                                    "name": name,
                                    "shape": list(weight.shape),
                                    "effective_rank": effective_rank,
                                    "max_rank": max_rank,
                                    "compression_ratio": compression_ratio,
                                }
                            )

                            analysis["rank_distributions"][name] = {
                                "singular_values": s[
                                    :20
                                ].tolist(),  # Top 20 singular values
                                "cumulative_energy": cumulative_energy[:20].tolist(),
                            }

                        except Exception as e:
                            logger.warning(f"Failed to analyze {name}: {e}")

        # Calculate overall compression estimate
        if analysis["compressible_modules"]:
            avg_compression = np.mean(
                [m["compression_ratio"] for m in analysis["compressible_modules"]]
            )
            analysis["estimated_compression"] = avg_compression

        return analysis

    def _is_lra_module(self, module: nn.Module) -> bool:
        """Check if a module is suitable for low-rank approximation."""
        return type(module) in self.config["target_modules"] and hasattr(
            module, "weight"
        )

    def apply_low_rank_approximation(self, model: nn.Module) -> tuple[nn.Module, dict]:
        """
        Apply low-rank approximation to the model.

        Args:
            model: Model to compress

        Returns:
            tuple of (compressed_model, compression_stats)
        """
        logger.info("Starting low-rank approximation...")

        # Analyze model first
        analysis = self.analyze_low_rank_opportunities(model)

        if not analysis["compressible_modules"]:
            logger.warning("No compressible modules found, returning original model")
            return model, {"status": "skipped", "reason": "no_compressible_modules"}

        try:
            compressed_model = self._apply_svd_compression(model, analysis)

            # Collect compression statistics
            stats = self._collect_lra_stats(model, compressed_model, analysis)

            logger.info(
                f"Low-rank approximation completed. Compression: {stats['compression_ratio']:.2%}"
            )
            return compressed_model, stats

        except Exception as e:
            logger.error(f"Low-rank approximation failed: {e}")
            return model, {"status": "failed", "error": str(e)}

    def _apply_svd_compression(self, model: nn.Module, analysis: dict) -> nn.Module:
        """Apply SVD-based low-rank approximation."""
        for module_info in analysis["compressible_modules"]:
            name = module_info["name"]
            target_rank = module_info["effective_rank"]

            # Get the module
            module = dict(model.named_modules())[name]

            if hasattr(module, "weight") and module.weight is not None:
                weight = module.weight.data

                # Apply SVD decomposition
                U, s, Vt = svd(weight.cpu().numpy(), full_matrices=False)

                # Truncate to target rank
                U_trunc = U[:, :target_rank]
                s_trunc = s[:target_rank]
                Vt_trunc = Vt[:target_rank, :]

                # Reconstruct approximated weight
                approximated = U_trunc @ np.diag(s_trunc) @ Vt_trunc

                # Update module weight
                module.weight.data = torch.from_numpy(approximated).to(
                    weight.device, weight.dtype
                )

        return model

    def _collect_lra_stats(
        self, original_model: nn.Module, compressed_model: nn.Module, analysis: dict
    ) -> dict:
        """Collect low-rank approximation statistics."""
        original_params = sum(p.numel() for p in original_model.parameters())
        compressed_params = sum(p.numel() for p in compressed_model.parameters())

        stats = {
            "status": "success",
            "method": self.config["method"],
            "original_parameters": original_params,
            "compressed_parameters": compressed_params,
            "compression_ratio": compressed_params / original_params
            if original_params > 0
            else 0,
            "compressed_modules": len(analysis["compressible_modules"]),
            "rank_ratio": self.config["rank_ratio"],
        }

        return stats

class CompositeCompressor:
    """
    Composite compression pipeline combining multiple techniques.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.pruner = SACPruner(self.config.get("pruning", {}))
        self.lra = LowRankApproximator(self.config.get("lra", {}))

    def run_compression_pipeline(self, model: nn.Module) -> dict:
        """
        Run composite compression pipeline.

        Args:
            model: Model to compress

        Returns:
            Compression results dictionary
        """
        results = {
            "success": False,
            "original_model": model,
            "compressed_model": None,
            "compression_stats": {},
            "pipeline_steps": [],
            "recommendations": [],
        }

        try:
            current_model = model
            total_stats = {
                "original_parameters": sum(p.numel() for p in model.parameters()),
                "final_parameters": 0,
                "total_compression_ratio": 1.0,
                "steps_applied": [],
            }

            # Step 1: Pruning
            if self.config.get("enable_pruning", True):
                logger.info("Step 1: Applying pruning...")
                pruned_model, prune_stats = self.pruner.apply_pruning(current_model)

                if prune_stats.get("status") == "success":
                    current_model = pruned_model
                    total_stats["steps_applied"].append("pruning")
                    results["pipeline_steps"].append(
                        {"step": "pruning", "stats": prune_stats}
                    )
                else:
                    results["recommendations"].append(
                        "Pruning failed, skipping to next step"
                    )

            # Step 2: Low-rank approximation
            if self.config.get("enable_lra", True):
                logger.info("Step 2: Applying low-rank approximation...")
                lra_model, lra_stats = self.lra.apply_low_rank_approximation(
                    current_model
                )

                if lra_stats.get("status") == "success":
                    current_model = lra_model
                    total_stats["steps_applied"].append("lra")
                    results["pipeline_steps"].append(
                        {"step": "low_rank_approximation", "stats": lra_stats}
                    )
                else:
                    results["recommendations"].append("Low-rank approximation failed")

            # Final statistics
            total_stats["final_parameters"] = sum(
                p.numel() for p in current_model.parameters()
            )
            if total_stats["original_parameters"] > 0:
                total_stats["total_compression_ratio"] = (
                    total_stats["final_parameters"] / total_stats["original_parameters"]
                )

            results.update(
                {
                    "success": True,
                    "compressed_model": current_model,
                    "compression_stats": total_stats,
                }
            )

            logger.info(
                f"Composite compression completed. "
                f"Total compression ratio: {total_stats['total_compression_ratio']:.2%}"
            )

        except Exception as e:
            logger.error(f"Composite compression failed: {e}")
            results["error"] = str(e)

        return results
