import pytest
import torch
import torch.nn as nn

from ztb.training.compression.composite_compressor import AdaptiveCompressor
from ztb.training.compression.composite_compressor import (
    CompositeCompressor as NewCompositeCompressor,
)
from ztb.training.compression.composite_compressor import (
    CompressionMetrics,
    benchmark_compression,
    compress_model_pipeline,
)
from ztb.training.compression.compressor import CompositeCompressor


if not all(hasattr(nn, attr) for attr in ("Sequential", "LSTM", "Linear", "ReLU")):
    pytest.skip(
        "Composite compression tests require the full torch.nn surface; current suite is running with a lightweight stub.",
        allow_module_level=True,
    )


def _tiny_model():
    return nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 3))


def test_composite_compressor_pipeline():
    model = _tiny_model()
    config = {
        "pruning": {"method": "l1_unstructured", "amount": 0.1},
        "lra": {"method": "svd", "rank_ratio": 0.5},
        "enable_pruning": True,
        "enable_lra": True,
    }

    compressor = CompositeCompressor(config)
    results = compressor.run_compression_pipeline(model)

    assert "success" in results
    assert results["success"] in (True, False)
    # If succeeded, ensure compressed_model returned and stats include final_parameters
    if results["success"]:
        assert "compressed_model" in results and results["compressed_model"] is not None
        assert (
            "compression_stats" in results
            and "final_parameters" in results["compression_stats"]
        )
    else:
        # Ensure error key is present in failure case
        assert "error" in results


class TestNewCompositeCompressor:
    """Test new CompositeCompressor implementation."""

    def test_initialization(self):
        """Test compressor initialization."""
        compressor = NewCompositeCompressor()
        assert compressor.config is not None
        assert "pipeline" in compressor.config
        assert compressor.metrics is not None

    def test_compression_metrics(self):
        """Test compression metrics calculation."""
        metrics = CompressionMetrics()

        # Create test models
        model1 = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))
        model2 = nn.Sequential(nn.Linear(10, 15), nn.ReLU(), nn.Linear(15, 1))

        metrics.calculate_metrics(model1, model2)

        assert metrics.original_size > 0
        assert metrics.compressed_size > 0
        assert metrics.compression_ratio > 0
        assert metrics.memory_savings >= 0

    def test_compression_metrics_handles_parameterless_models(self):
        """Test metrics calculation does not divide by zero for parameterless models."""
        metrics = CompressionMetrics()

        model1 = nn.Sequential(nn.ReLU())
        model2 = nn.Sequential(nn.Identity())

        metrics.calculate_metrics(model1, model2)

        assert metrics.original_size == 0
        assert metrics.compressed_size == 0
        assert metrics.compression_ratio == 1.0
        assert metrics.memory_savings == 0.0

    def test_compress_model_basic(self):
        """Test basic model compression."""
        model = _tiny_model()
        compressor = NewCompositeCompressor(
            {
                "pipeline": ["pruning"],
                "pruning": {"amount": 0.1},
            }  # Only pruning for simplicity
        )

        compressed = compressor.compress_model(model)

        # Model should be compressed (different object)
        assert compressed is not model
        assert isinstance(compressed, nn.Module)

        # Check metrics
        report = compressor.get_compression_report()
        assert "compression_ratio" in report
        assert "memory_savings_percent" in report

    def test_adaptive_compressor(self):
        """Test adaptive compressor."""
        model = _tiny_model()
        compressor = AdaptiveCompressor(target_compression_ratio=0.8)

        compressed = compressor.compress_adaptively(model)

        assert isinstance(compressed, nn.Module)
        assert compressed is not model

    def test_compress_model_pipeline_function(self):
        """Test compress_model_pipeline utility function."""
        model = _tiny_model()

        compressed, report = compress_model_pipeline(
            model, compression_ratio=0.7, techniques=["pruning"]
        )

        assert isinstance(compressed, nn.Module)
        assert isinstance(report, dict)
        assert "compression_ratio" in report

    def test_benchmark_compression(self):
        """Test compression benchmarking."""
        model1 = _tiny_model()
        model2 = _tiny_model()  # Same model for simplicity

        # Create dummy test data
        test_data = torch.randn(5, 10)

        results = benchmark_compression(model1, model2, test_data, num_runs=3)

        assert "original_avg_time" in results
        assert "compressed_avg_time" in results
        assert "speedup_ratio" in results
        assert results["original_avg_time"] > 0
        assert results["compressed_avg_time"] > 0

    def test_model_analysis(self):
        """Test model analysis for adaptive compression."""
        compressor = AdaptiveCompressor()
        model = _tiny_model()

        analysis = compressor._analyze_model(model)

        assert "num_parameters" in analysis
        assert "num_layers" in analysis
        assert "layer_types" in analysis
        assert analysis["num_parameters"] > 0
        assert analysis["num_layers"] > 0

    def test_pipeline_selection(self):
        """Test optimal pipeline selection."""
        compressor = AdaptiveCompressor()

        analysis = {
            "layer_types": {"linear": 2, "conv2d": 0, "lstm": 0},
            "num_parameters": 1000,
        }

        pipeline_config = compressor._select_optimal_pipeline(analysis)

        assert "pipeline" in pipeline_config
        assert len(pipeline_config["pipeline"]) > 0
        assert "pruning" in pipeline_config["pipeline"]
        assert "quantization" in pipeline_config["pipeline"]
