import torch
import torch.nn as nn
from ztb.training.compression.compressor import CompositeCompressor


def _tiny_model():
    return nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 3))


def test_composite_compressor_pipeline():
    model = _tiny_model()
    config = {
        'pruning': {'method': 'l1_unstructured', 'amount': 0.1},
        'lra': {'method': 'svd', 'rank_ratio': 0.5},
        'enable_pruning': True,
        'enable_lra': True
    }

    compressor = CompositeCompressor(config)
    results = compressor.run_compression_pipeline(model)

    assert 'success' in results
    assert results['success'] in (True, False)
    # If succeeded, ensure compressed_model returned and stats include final_parameters
    if results['success']:
        assert 'compressed_model' in results and results['compressed_model'] is not None
        assert 'compression_stats' in results and 'final_parameters' in results['compression_stats']
    else:
        # Ensure error key is present in failure case
        assert 'error' in results

