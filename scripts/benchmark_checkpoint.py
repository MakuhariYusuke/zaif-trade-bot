#!/usr/bin/env python3
"""
Checkpoint performance benchmark for Zaif Trade Bot.

Measures model checkpoint save/load performance with soft performance gates.
"""

import time
import tempfile
import os
import pickle
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path


@dataclass
class CheckpointResult:
    """Result of a checkpoint benchmark."""
    name: str
    duration_seconds: float
    throughput_mb_per_second: float
    file_size_mb: float
    passed_gates: bool
    gate_violations: List[str]


class CheckpointBenchmark:
    """Benchmark for checkpoint save/load operations."""

    def __init__(self):
        self.perf_gates = {
            'max_save_time_seconds': 30,     # Maximum time to save checkpoint
            'max_load_time_seconds': 15,     # Maximum time to load checkpoint
            'min_save_throughput_mb_s': 10,  # Minimum save throughput
            'min_load_throughput_mb_s': 20,  # Minimum load throughput
            'max_checkpoint_size_mb': 1000,  # Maximum checkpoint size
        }

    def create_test_model(self, size: str = 'medium') -> Dict[str, Any]:
        """Create a test model/checkpoint data structure."""
        if size == 'small':
            # Small model: ~10MB
            data = {
                'weights': np.random.normal(0, 1, (1000, 1000)).astype(np.float32),
                'biases': np.random.normal(0, 0.1, 1000).astype(np.float32),
                'metadata': {'layers': 3, 'activation': 'relu'},
                'optimizer_state': {'lr': 0.001, 'momentum': np.random.normal(0, 0.01, 1000)}
            }
        elif size == 'medium':
            # Medium model: ~50MB
            data = {
                'weights': np.random.normal(0, 1, (5000, 2000)).astype(np.float32),
                'biases': np.random.normal(0, 0.1, 2000).astype(np.float32),
                'metadata': {'layers': 5, 'activation': 'relu', 'dropout': 0.2},
                'optimizer_state': {
                    'lr': 0.001,
                    'beta1': 0.9,
                    'beta2': 0.999,
                    'momentum': np.random.normal(0, 0.01, (5000, 2000))
                },
                'training_history': pd.DataFrame({
                    'epoch': range(100),
                    'loss': np.random.exponential(0.1, 100),
                    'val_loss': np.random.exponential(0.15, 100)
                })
            }
        else:  # large
            # Large model: ~200MB
            data = {
                'weights': np.random.normal(0, 1, (10000, 5000)).astype(np.float32),
                'biases': np.random.normal(0, 0.1, 5000).astype(np.float32),
                'metadata': {'layers': 8, 'activation': 'relu', 'batch_norm': True},
                'optimizer_state': {
                    'lr': 0.0001,
                    'beta1': 0.9,
                    'beta2': 0.999,
                    'momentum': np.random.normal(0, 0.01, (10000, 5000)),
                    'velocity': np.random.normal(0, 0.01, (10000, 5000))
                },
                'training_history': pd.DataFrame({
                    'epoch': range(500),
                    'loss': np.random.exponential(0.05, 500),
                    'val_loss': np.random.exponential(0.08, 500),
                    'accuracy': np.random.beta(2, 1, 500)
                }),
                'feature_importance': np.random.exponential(1, 1000)
            }

        return data

    def get_data_size_mb(self, data: Dict[str, Any]) -> float:
        """Estimate size of data in MB."""
        # Rough estimation
        total_elements = 0
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                total_elements += value.size
            elif isinstance(value, pd.DataFrame):
                total_elements += value.memory_usage(deep=True).sum()
            elif isinstance(value, dict):
                # Rough estimate for nested dicts
                total_elements += len(str(value)) * 2  # Rough char to bytes

        # Assume 4 bytes per float32 element, 8 bytes per float64
        estimated_bytes = total_elements * 4
        return estimated_bytes / (1024 * 1024)

    def benchmark_pickle_save(self, data: Dict[str, Any], file_path: Path) -> CheckpointResult:
        """Benchmark pickle-based checkpoint saving."""
        print(f"Benchmarking pickle save to {file_path}...")

        start_time = time.time()
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        duration = time.time() - start_time
        file_size = file_path.stat().st_size / (1024 * 1024)  # MB
        throughput = file_size / duration

        violations = []
        if duration > self.perf_gates['max_save_time_seconds']:
            violations.append(".2f")
        if throughput < self.perf_gates['min_save_throughput_mb_s']:
            violations.append(".2f")
        if file_size > self.perf_gates['max_checkpoint_size_mb']:
            violations.append(".2f")

        return CheckpointResult(
            name="pickle_save",
            duration_seconds=duration,
            throughput_mb_per_second=throughput,
            file_size_mb=file_size,
            passed_gates=len(violations) == 0,
            gate_violations=violations
        )

    def benchmark_pickle_load(self, file_path: Path) -> CheckpointResult:
        """Benchmark pickle-based checkpoint loading."""
        print(f"Benchmarking pickle load from {file_path}...")

        start_time = time.time()
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        duration = time.time() - start_time
        file_size = file_path.stat().st_size / (1024 * 1024)  # MB
        throughput = file_size / duration

        violations = []
        if duration > self.perf_gates['max_load_time_seconds']:
            violations.append(".2f")
        if throughput < self.perf_gates['min_load_throughput_mb_s']:
            violations.append(".2f")

        return CheckpointResult(
            name="pickle_load",
            duration_seconds=duration,
            throughput_mb_per_second=throughput,
            file_size_mb=file_size,
            passed_gates=len(violations) == 0,
            gate_violations=violations
        )

    def benchmark_json_save(self, data: Dict[str, Any], file_path: Path) -> CheckpointResult:
        """Benchmark JSON-based checkpoint saving (metadata only)."""
        print(f"Benchmarking JSON save to {file_path}...")

        # Convert numpy arrays to lists for JSON serialization
        json_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            elif isinstance(value, pd.DataFrame):
                json_data[key] = value.to_dict('records')
            else:
                json_data[key] = value

        start_time = time.time()
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        duration = time.time() - start_time
        file_size = file_path.stat().st_size / (1024 * 1024)  # MB
        throughput = file_size / duration

        violations = []
        if duration > self.perf_gates['max_save_time_seconds']:
            violations.append(".2f")

        return CheckpointResult(
            name="json_save",
            duration_seconds=duration,
            throughput_mb_per_second=throughput,
            file_size_mb=file_size,
            passed_gates=len(violations) == 0,
            gate_violations=violations
        )

    def benchmark_numpy_save(self, data: Dict[str, Any], file_path: Path) -> CheckpointResult:
        """Benchmark numpy .npz save."""
        print(f"Benchmarking numpy save to {file_path}...")

        # Extract arrays for .npz
        arrays = {}
        metadata = {}

        for key, value in data.items():
            if isinstance(value, np.ndarray):
                arrays[key] = value
            else:
                metadata[key] = value

        start_time = time.time()
        np.savez(file_path, **arrays)

        # Save metadata separately
        metadata_file = file_path.with_suffix('.metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        duration = time.time() - start_time
        file_size = (file_path.stat().st_size + metadata_file.stat().st_size) / (1024 * 1024)
        throughput = file_size / duration

        violations = []
        if duration > self.perf_gates['max_save_time_seconds']:
            violations.append(".2f")
        if throughput < self.perf_gates['min_save_throughput_mb_s']:
            violations.append(".2f")

        return CheckpointResult(
            name="numpy_save",
            duration_seconds=duration,
            throughput_mb_per_second=throughput,
            file_size_mb=file_size,
            passed_gates=len(violations) == 0,
            gate_violations=violations
        )

    def print_result(self, result: CheckpointResult):
        """Print checkpoint benchmark result."""
        status = "✓ PASS" if result.passed_gates else "✗ FAIL"

        print(f"\n{result.name.upper()} BENCHMARK - {status}")
        print(f"Duration: {result.duration_seconds:.2f}s")
        print(f"Throughput: {result.throughput_mb_per_second:.1f} MB/s")
        print(f"File Size: {result.file_size_mb:.1f} MB")

        if result.gate_violations:
            print("Gate Violations:")
            for violation in result.gate_violations:
                print(f"  - {violation}")

    def run_checkpoint_benchmarks(self, model_size: str = 'medium') -> List[CheckpointResult]:
        """Run all checkpoint benchmarks."""
        print(f"Running Checkpoint Performance Benchmarks ({model_size} model)...")
        print("=" * 60)

        results = []

        # Create test data
        test_data = self.create_test_model(model_size)
        estimated_size = self.get_data_size_mb(test_data)
        print(f"Test model estimated size: {estimated_size:.1f} MB")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Pickle benchmarks
            pickle_file = temp_path / 'checkpoint.pkl'
            save_result = self.benchmark_pickle_save(test_data, pickle_file)
            self.print_result(save_result)
            results.append(save_result)

            load_result = self.benchmark_pickle_load(pickle_file)
            self.print_result(load_result)
            results.append(load_result)

            # JSON benchmark (metadata only)
            json_file = temp_path / 'checkpoint.json'
            json_result = self.benchmark_json_save(test_data, json_file)
            self.print_result(json_result)
            results.append(json_result)

            # Numpy benchmark
            npz_file = temp_path / 'checkpoint.npz'
            numpy_result = self.benchmark_numpy_save(test_data, npz_file)
            self.print_result(numpy_result)
            results.append(numpy_result)

        # Summary
        passed = sum(1 for r in results if r.passed_gates)
        total = len(results)

        print(f"\n{'='*60}")
        print(f"SUMMARY: {passed}/{total} checkpoint benchmarks passed performance gates")

        if passed < total:
            print("⚠️  Some checkpoint benchmarks failed performance gates.")
            print("   Consider optimizing checkpoint format or storage.")
        else:
            print("✅ All checkpoint benchmarks passed performance gates!")

        return results


def main():
    """Main benchmark runner."""
    import argparse

    parser = argparse.ArgumentParser(description='Run checkpoint performance benchmarks')
    parser.add_argument('--model-size', choices=['small', 'medium', 'large'],
                       default='medium', help='Size of test model')

    args = parser.parse_args()

    benchmark = CheckpointBenchmark()
    results = benchmark.run_checkpoint_benchmarks(args.model_size)

    # Exit with code based on results
    all_passed = all(r.passed_gates for r in results)
    exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()