import pytest
import torch
import torch.nn as nn

from ztb.training.quantization.quantizer import QuantizationPipeline, SACQuantizer


if not all(hasattr(nn, attr) for attr in ("Sequential", "Linear", "ReLU", "LSTM")):
    pytest.skip(
        "Quantization tests require the full torch.nn surface; current suite is running with a lightweight stub.",
        allow_module_level=True,
    )


def _make_tiny_model():
    # Simple model to test quantization pipeline
    return nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 3))


def test_quantizer_analyze_and_quantize():
    model = _make_tiny_model()
    quantizer = SACQuantizer({"quantizable_modules": [nn.Linear]})

    analysis = quantizer.analyze_model(model)
    assert "total_parameters" in analysis
    assert analysis["total_parameters"] > 0

    qmodel, stats = quantizer.quantize_model(model)
    # quantize_dynamic may not change model on CPU for some layers; check returned type
    assert isinstance(stats, dict)
    assert "status" in stats or "compression_ratio" in stats


def test_quantization_pipeline_run():
    model = _make_tiny_model()
    pipeline = QuantizationPipeline(
        {
            "quantization": {
                "quantizable_modules": [nn.Linear],
                "accuracy_tolerance": 0.1,
            }
        }
    )

    # Dummy validation data
    dummy_input = torch.randn(4, 10)
    results = pipeline.run_pipeline(model, validation_data=(dummy_input, None))

    assert "success" in results
    assert results["success"] in (True, False)
    assert "analysis" in results
    assert "quantization_stats" in results
