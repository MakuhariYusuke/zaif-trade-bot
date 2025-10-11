#!/usr/bin/env python3
"""
Memory leak monitoring overhead benchmark
Measure performance impact of gc.get_objects() calls in 100k experiments
"""

import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import psutil


class MemoryMonitoringBenchmark:
    """Benchmark memory monitoring overhead in experiments"""

    def __init__(self):
        self.results = []

    def simulate_experiment_step(self, step_num: int) -> Dict[str, Any]:
        """Simulate a single experiment step with memory allocation"""
        # Simulate memory allocation patterns similar to RL experiments
        if step_num % 1000 == 0:
            # Large allocation every 1000 steps (like model updates)
            large_data = np.random.randn(10000, 100).astype(np.float32)
        elif step_num % 100 == 0:
            # Medium allocation every 100 steps
            medium_data = np.random.randn(1000, 100).astype(np.float32)
        else:
            # Small allocation for most steps
            small_data = np.random.randn(100, 10).astype(np.float32)

        # Simulate some processing
        time.sleep(0.001)  # 1ms processing time

        return {
            "step": step_num,
            "reward": np.random.normal(0, 1),
            "memory_mb": psutil.Process().memory_info().rss / (1024 * 1024),
        }

    def benchmark_memory_monitoring(
        self, total_steps: int = 10000, monitoring_interval: int = 10
    ) -> Dict[str, Any]:
        """Benchmark memory monitoring overhead"""

        print(
            f"🔍 Benchmarking memory monitoring (steps: {total_steps}, interval: {monitoring_interval})"
        )

        # Force garbage collection before starting
        gc.collect()
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        memory_history = []
        gc_stats = []
        object_counts = []
        step_times = []

        for step in range(total_steps):
            step_start = time.time()

            # Simulate experiment step
            step_result = self.simulate_experiment_step(step)

            # Memory monitoring (similar to ResourceMonitor.log_resources)
            if step % monitoring_interval == 0:
                current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                memory_history.append(current_memory)

                # Measure time for GC operations
                gc_start = time.time()
                gc_counts = gc.get_count()
                gc_statistics = gc.get_stats()
                current_objects = len(gc.get_objects())
                gc_time = time.time() - gc_start

                gc_stats.append(gc_counts)
                object_counts.append(current_objects)

                # Store GC timing
                step_times.append(
                    {
                        "step": step,
                        "gc_time_sec": gc_time,
                        "total_step_time_sec": time.time() - step_start,
                    }
                )

        total_time = time.time() - start_time
        end_memory = psutil.Process().memory_info().rss

        # Calculate statistics
        gc_times = [t["gc_time_sec"] for t in step_times]
        step_times_total = [t["total_step_time_sec"] for t in step_times]

        result = {
            "total_steps": total_steps,
            "monitoring_interval": monitoring_interval,
            "total_time_sec": total_time,
            "avg_step_time_sec": total_time / total_steps,
            "memory_increase_mb": (end_memory - start_memory) / (1024 * 1024),
            "gc_operations_count": len(step_times),
            "avg_gc_time_sec": np.mean(gc_times) if gc_times else 0,
            "max_gc_time_sec": max(gc_times) if gc_times else 0,
            "gc_time_percentage": (sum(gc_times) / total_time) * 100,
            "avg_step_time_with_monitoring": (
                np.mean(step_times_total) if step_times_total else 0
            ),
            "memory_history": memory_history[-10:],  # Last 10 measurements
            "object_counts": object_counts[-10:] if object_counts else [],
        }

        self.results.append(result)
        return result

    def run_comprehensive_benchmark(self):
        """Run benchmarks with different monitoring frequencies"""
        print("🔍 Memory Monitoring Overhead Benchmark")
        print("=" * 50)

        # Test different monitoring intervals
        intervals = [1, 5, 10, 50, 100, 300]  # steps between monitoring
        total_steps = 5000

        for interval in intervals:
            result = self.benchmark_memory_monitoring(total_steps, interval)

            print(f"\n📊 Interval: {interval} steps")
            print(f"  Total time: {result['total_time_sec']:.2f}s")
            print(f"  Avg step time: {result['avg_step_time_sec'] * 1000:.2f}ms")
            print(f"  GC time %: {result['gc_time_percentage']:.2f}%")
            print(f"  Avg GC time: {result['avg_gc_time_sec'] * 1000:.2f}ms")
            print(f"  Memory increase: {result['memory_increase_mb']:.1f}MB")

    def analyze_dynamic_threshold(self, memory_history: List[float]) -> Dict[str, Any]:
        """Analyze dynamic threshold calculation for memory leak detection"""
        if len(memory_history) < 10:
            return {"error": "Insufficient data for analysis"}

        # Calculate moving statistics
        window_size = min(10, len(memory_history))
        recent_memory = memory_history[-window_size:]

        mean_memory = np.mean(recent_memory)
        std_memory = np.std(recent_memory)

        # Dynamic threshold: mean + 2*std (95% confidence)
        dynamic_threshold = mean_memory + 2 * std_memory

        # Fixed threshold (50% increase from initial)
        initial_memory = memory_history[0]
        fixed_threshold = initial_memory * 1.5

        # Trend analysis
        if len(memory_history) >= 20:
            early_mean = np.mean(memory_history[:10])
            late_mean = np.mean(memory_history[-10:])
            trend_increase = ((late_mean / early_mean) - 1) * 100
        else:
            trend_increase = 0

        return {
            "mean_memory_mb": mean_memory,
            "std_memory_mb": std_memory,
            "dynamic_threshold_mb": dynamic_threshold,
            "fixed_threshold_mb": fixed_threshold,
            "threshold_ratio": (
                dynamic_threshold / fixed_threshold if fixed_threshold > 0 else 0
            ),
            "trend_increase_percent": trend_increase,
            "recommendation": (
                "dynamic" if dynamic_threshold < fixed_threshold else "fixed"
            ),
        }

    def save_results(
        self, output_file: str = "reports/memory_monitoring_benchmark.json"
    ):
        """Save benchmark results"""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        # Add dynamic threshold analysis
        if self.results:
            latest_result = self.results[-1]
            if "memory_history" in latest_result and latest_result["memory_history"]:
                threshold_analysis = self.analyze_dynamic_threshold(
                    latest_result["memory_history"]
                )
                latest_result["threshold_analysis"] = threshold_analysis

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": time.time(),
                    "benchmark_results": self.results,
                    "recommendations": {
                        "optimal_monitoring_interval": self._find_optimal_interval(),
                        "threshold_method": "dynamic" if self.results else "unknown",
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"\n💾 Results saved to {output_file}")

    def _find_optimal_interval(self) -> int:
        """Find optimal monitoring interval based on overhead"""
        if not self.results:
            return 10  # Default

        # Find interval with <1% GC overhead and reasonable monitoring frequency
        candidates = [
            r
            for r in self.results
            if r["gc_time_percentage"] < 1.0 and r["monitoring_interval"] <= 100
        ]

        if candidates:
            # Prefer more frequent monitoring within acceptable overhead
            return min(candidates, key=lambda x: x["monitoring_interval"])[
                "monitoring_interval"
            ]

        return 10  # Fallback


if __name__ == "__main__":
    benchmark = MemoryMonitoringBenchmark()
    benchmark.run_comprehensive_benchmark()
    benchmark.save_results()
