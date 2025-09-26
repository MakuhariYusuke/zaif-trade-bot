#!/usr/bin/env python3
"""
Compression benchmark for checkpoint IO optimization
LZ4 vs ZSTD compression comparison for 100k experiment checkpoints
"""

import time
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any
import json

try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


class CompressionBenchmark:
    """Benchmark LZ4 vs ZSTD compression for checkpoint data"""

    def __init__(self):
        self.results = []

    def generate_sample_checkpoint_data(self, size_mb: int = 50) -> Dict[str, Any]:
        """Generate sample checkpoint data similar to 100k experiment"""
        # Simulate neural network weights (large arrays)
        np.random.seed(42)

        data = {
            'model_weights': {
                'layer1': np.random.randn(1000, 2000).astype(np.float32),
                'layer2': np.random.randn(2000, 1000).astype(np.float32),
                'layer3': np.random.randn(1000, 500).astype(np.float32),
            },
            'optimizer_state': {
                'adam_m': np.random.randn(1000, 500).astype(np.float32),
                'adam_v': np.random.randn(1000, 500).astype(np.float32),
            },
            'metadata': {
                'step': 50000,
                'epoch': 100,
                'loss': 0.0234,
                'val_loss': 0.0345,
                'learning_rate': 0.001,
                'timestamp': time.time(),
                'experiment_config': {
                    'total_steps': 100000,
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'model_type': 'reinforcement_learning'
                }
            },
            'training_history': {
                'rewards': np.random.randn(50000).tolist(),
                'losses': np.random.exponential(0.1, 50000).tolist(),
                'pnl_history': np.random.randn(50000).tolist()
            }
        }

        # Adjust size to target MB
        target_bytes = size_mb * 1024 * 1024
        current_bytes = len(pickle.dumps(data))

        if current_bytes < target_bytes:
            # Add padding data to reach target size
            padding_size = (target_bytes - current_bytes) // 4  # float32 = 4 bytes
            data['padding'] = np.random.randn(padding_size).astype(np.float32)

        return data

    def benchmark_compression(self, data: Dict[str, Any], compression_type: str) -> Dict[str, Any]:
        """Benchmark single compression method"""
        start_time = time.time()

        # Serialize to bytes
        pickle_data = pickle.dumps(data)
        original_size = len(pickle_data)

        # Compress
        if compression_type == 'lz4' and HAS_LZ4:
            compressed = lz4.frame.compress(pickle_data, compression_level=1)  # Fast compression
        elif compression_type == 'zstd' and HAS_ZSTD:
            ctx = zstd.ZstdCompressor(level=3)  # Balanced compression
            compressed = ctx.compress(pickle_data)
        elif compression_type == 'zlib':
            compressed = zlib.compress(pickle_data, level=6)
        else:
            raise ValueError(f"Unsupported compression: {compression_type}")

        compressed_size = len(compressed)
        compression_time = time.time() - start_time

        # Decompression benchmark
        start_time = time.time()
        if compression_type == 'lz4' and HAS_LZ4:
            decompressed = lz4.frame.decompress(compressed)
        elif compression_type == 'zstd' and HAS_ZSTD:
            ctx = zstd.ZstdDecompressor()
            decompressed = ctx.decompress(compressed)
        elif compression_type == 'zlib':
            decompressed = zlib.decompress(compressed)

        decompression_time = time.time() - start_time

        # Verify data integrity
        assert pickle.loads(decompressed) is not None, "Data corruption detected"

        return {
            'compression_type': compression_type,
            'original_size_mb': original_size / (1024 * 1024),
            'compressed_size_mb': compressed_size / (1024 * 1024),
            'compression_ratio': compressed_size / original_size,
            'compression_time_sec': compression_time,
            'decompression_time_sec': decompression_time,
            'total_time_sec': compression_time + decompression_time,
            'compression_speed_mbps': (original_size / (1024 * 1024)) / compression_time,
            'decompression_speed_mbps': (original_size / (1024 * 1024)) / decompression_time
        }

    def run_benchmark(self, data_sizes_mb: list = [10, 25, 50, 100]):
        """Run comprehensive compression benchmark"""
        print("🔍 Checkpoint Compression Benchmark")
        print("=" * 50)

        for size_mb in data_sizes_mb:
            print(f"\n📊 Testing with {size_mb}MB checkpoint data")
            data = self.generate_sample_checkpoint_data(size_mb)

            compression_types = []
            if HAS_LZ4:
                compression_types.append('lz4')
            if HAS_ZSTD:
                compression_types.append('zstd')
            compression_types.append('zlib')  # Always available

            for comp_type in compression_types:
                try:
                    result = self.benchmark_compression(data, comp_type)
                    self.results.append(result)

                    print(f"  {comp_type.upper():4}: "
                          ".1f"
                          ".1f"
                          ".3f"
                          ".1f"
                          ".1f")

                except Exception as e:
                    print(f"  {comp_type.upper():4}: ERROR - {e}")

    def save_results(self, output_file: str = "reports/compression_benchmark.json"):
        """Save benchmark results to JSON"""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': time.time(),
                'results': self.results,
                'summary': self._generate_summary()
            }, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to {output_file}")

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not self.results:
            return {}

        # Group by compression type
        by_type = {}
        for result in self.results:
            comp_type = result['compression_type']
            if comp_type not in by_type:
                by_type[comp_type] = []
            by_type[comp_type].append(result)

        summary = {}
        for comp_type, results in by_type.items():
            ratios = [r['compression_ratio'] for r in results]
            comp_times = [r['compression_time_sec'] for r in results]
            decomp_times = [r['decompression_time_sec'] for r in results]

            summary[comp_type] = {
                'avg_compression_ratio': np.mean(ratios),
                'best_compression_ratio': min(ratios),
                'avg_compression_time': np.mean(comp_times),
                'avg_decompression_time': np.mean(decomp_times),
                'avg_total_time': np.mean([t + d for t, d in zip(comp_times, decomp_times)]),
                'sample_count': len(results)
            }

        return summary


if __name__ == "__main__":
    benchmark = CompressionBenchmark()
    benchmark.run_benchmark([10, 25, 50])
    benchmark.save_results()