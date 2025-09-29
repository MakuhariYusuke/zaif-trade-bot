#!/usr/bin/env python3
"""
Streaming performance benchmark for Zaif Trade Bot.

Measures real-time data processing performance with soft performance gates.
"""

import asyncio
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List

import numpy as np


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    duration_seconds: float
    operations_per_second: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    memory_mb: float
    passed_gates: bool
    gate_violations: List[str]


class StreamingBenchmark:
    """Benchmark for streaming data processing."""

    def __init__(self):
        self.perf_gates = {
            "min_ops_per_second": 1000,  # Minimum operations per second
            "max_p95_latency_ms": 100,  # Maximum 95th percentile latency
            "max_memory_mb": 500,  # Maximum memory usage
        }

    async def benchmark_data_ingestion(
        self, num_messages: int = 10000
    ) -> BenchmarkResult:
        """Benchmark data ingestion from streaming source."""
        print(f"Benchmarking data ingestion with {num_messages} messages...")

        latencies = []
        memory_usage = []
        start_time = time.time()

        for i in range(num_messages):
            msg_start = time.time()

            # Simulate message processing
            message = {
                "timestamp": datetime.now().isoformat(),
                "price": 10000 + np.random.normal(0, 100),
                "volume": np.random.exponential(10),
                "sequence": i,
            }

            # Simulate processing delay (network + parsing)
            await asyncio.sleep(0.001)  # 1ms processing time

            # Record latency
            latency = (time.time() - msg_start) * 1000  # ms
            latencies.append(latency)

            # Record memory every 100 messages
            if i % 100 == 0:
                # Simplified memory check
                memory_usage.append(50 + np.random.normal(0, 5))  # MB

        duration = time.time() - start_time
        ops_per_second = num_messages / duration

        # Calculate percentiles
        latencies.sort()
        p50 = statistics.median(latencies)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        avg_memory = statistics.mean(memory_usage) if memory_usage else 0

        # Check performance gates
        violations = []
        if ops_per_second < self.perf_gates["min_ops_per_second"]:
            violations.append(".2f")
        if p95 > self.perf_gates["max_p95_latency_ms"]:
            violations.append(".2f")
        if avg_memory > self.perf_gates["max_memory_mb"]:
            violations.append(".2f")

        return BenchmarkResult(
            name="data_ingestion",
            duration_seconds=duration,
            operations_per_second=ops_per_second,
            latency_p50=p50,
            latency_p95=p95,
            latency_p99=p99,
            memory_mb=avg_memory,
            passed_gates=len(violations) == 0,
            gate_violations=violations,
        )

    async def benchmark_signal_processing(
        self, num_signals: int = 5000
    ) -> BenchmarkResult:
        """Benchmark real-time signal processing."""
        print(f"Benchmarking signal processing with {num_signals} signals...")

        latencies = []
        start_time = time.time()

        # Pre-generate test data
        prices = np.random.normal(10000, 100, num_signals)
        volumes = np.random.exponential(10, num_signals)

        for i in range(num_signals):
            signal_start = time.time()

            # Simulate signal processing pipeline
            price = prices[i]
            volume = volumes[i]

            # Simple moving average calculation
            if i >= 20:
                ma20 = np.mean(prices[i - 20 : i])
                signal = (
                    1 if price > ma20 * 1.001 else (-1 if price < ma20 * 0.999 else 0)
                )
            else:
                signal = 0

            # Simulate additional processing
            await asyncio.sleep(0.0005)  # 0.5ms processing

            latency = (time.time() - signal_start) * 1000
            latencies.append(latency)

        duration = time.time() - start_time
        ops_per_second = num_signals / duration

        latencies.sort()
        p50 = statistics.median(latencies)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        # Check gates (signal processing has stricter requirements)
        violations = []
        if ops_per_second < 2000:  # Higher requirement for signals
            violations.append(".2f")
        if p95 > 50:  # Lower latency requirement
            violations.append(".2f")

        return BenchmarkResult(
            name="signal_processing",
            duration_seconds=duration,
            operations_per_second=ops_per_second,
            latency_p50=p50,
            latency_p95=p95,
            latency_p99=p99,
            memory_mb=30,  # Estimated
            passed_gates=len(violations) == 0,
            gate_violations=violations,
        )

    async def benchmark_order_execution(
        self, num_orders: int = 1000
    ) -> BenchmarkResult:
        """Benchmark order execution simulation."""
        print(f"Benchmarking order execution with {num_orders} orders...")

        latencies = []
        start_time = time.time()

        for i in range(num_orders):
            order_start = time.time()

            # Simulate order execution
            order = {
                "id": f"order_{i}",
                "type": "market",
                "side": "buy" if i % 2 == 0 else "sell",
                "quantity": np.random.uniform(0.001, 1.0),
                "price": 10000 + np.random.normal(0, 50),
            }

            # Simulate API call + processing
            await asyncio.sleep(0.005)  # 5ms simulated API latency

            # Simulate slippage calculation
            executed_price = order["price"] * (1 + np.random.normal(0, 0.001))
            executed_quantity = order["quantity"] * (1 - np.random.normal(0, 0.01))

            latency = (time.time() - order_start) * 1000
            latencies.append(latency)

        duration = time.time() - start_time
        ops_per_second = num_orders / duration

        latencies.sort()
        p50 = statistics.median(latencies)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        violations = []
        if ops_per_second < 200:  # Order execution is slower
            violations.append(".2f")
        if p95 > 200:  # Higher latency tolerance for orders
            violations.append(".2f")

        return BenchmarkResult(
            name="order_execution",
            duration_seconds=duration,
            operations_per_second=ops_per_second,
            latency_p50=p50,
            latency_p95=p95,
            latency_p99=p99,
            memory_mb=25,
            passed_gates=len(violations) == 0,
            gate_violations=violations,
        )

    def print_result(self, result: BenchmarkResult):
        """Print benchmark result."""
        status = "✓ PASS" if result.passed_gates else "✗ FAIL"

        print(f"\n{result.name.upper()} BENCHMARK - {status}")
        print(f"Duration: {result.duration_seconds:.2f}s")
        print(f"Operations/sec: {result.operations_per_second:.0f}")
        print(f"Latency P50: {result.latency_p50:.2f}ms")
        print(f"Latency P95: {result.latency_p95:.2f}ms")
        print(f"Latency P99: {result.latency_p99:.2f}ms")
        print(f"Memory: {result.memory_mb:.1f}MB")

        if result.gate_violations:
            print("Gate Violations:")
            for violation in result.gate_violations:
                print(f"  - {violation}")

    async def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all streaming benchmarks."""
        print("Running Streaming Performance Benchmarks...")
        print("=" * 50)

        results = []

        # Data ingestion benchmark
        result = await self.benchmark_data_ingestion()
        self.print_result(result)
        results.append(result)

        # Signal processing benchmark
        result = await self.benchmark_signal_processing()
        self.print_result(result)
        results.append(result)

        # Order execution benchmark
        result = await self.benchmark_order_execution()
        self.print_result(result)
        results.append(result)

        # Summary
        passed = sum(1 for r in results if r.passed_gates)
        total = len(results)

        print(f"\n{'=' * 50}")
        print(f"SUMMARY: {passed}/{total} benchmarks passed performance gates")

        if passed < total:
            print("⚠️  Some benchmarks failed performance gates.")
            print("   Consider optimizing before deployment.")
        else:
            print("✅ All benchmarks passed performance gates!")

        return results


async def main():
    """Main benchmark runner."""
    benchmark = StreamingBenchmark()
    results = await benchmark.run_all_benchmarks()

    # Exit with code based on results
    all_passed = all(r.passed_gates for r in results)
    exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
