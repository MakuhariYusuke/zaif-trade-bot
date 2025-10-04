#!/usr/bin/env python3
"""
bench_io.py
I/O benchmarking script for Parquet compression, chunking, and column projection.
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ztb.cache.parquet_io import read_parquet, write_parquet


def generate_test_data(n_rows: int = 100000) -> pd.DataFrame:
    """Generate test DataFrame similar to trading data"""
    np.random.seed(42)

    data = {
        "timestamp": pd.date_range("2023-01-01", periods=n_rows, freq="1min"),
        "open": np.random.uniform(100, 200, n_rows),
        "high": np.random.uniform(100, 200, n_rows),
        "low": np.random.uniform(100, 200, n_rows),
        "close": np.random.uniform(100, 200, n_rows),
        "volume": np.random.uniform(1000, 10000, n_rows),
        "rsi": np.random.uniform(0, 100, n_rows),
        "macd": np.random.normal(0, 10, n_rows),
        "bb_upper": np.random.uniform(100, 200, n_rows),
        "bb_lower": np.random.uniform(100, 200, n_rows),
    }

    # Ensure high >= close >= low, etc.
    for i in range(n_rows):
        data["high"][i] = max(data["open"][i], data["high"][i], data["close"][i])  # type: ignore
        data["low"][i] = min(data["open"][i], data["low"][i], data["close"][i])  # type: ignore

    return pd.DataFrame(data)


def measure_io_performance(
    df: pd.DataFrame, config: Dict[str, Any], temp_dir: Path
) -> Dict[str, Any]:
    """Measure I/O performance for given config"""
    process = psutil.Process(os.getpid())

    # Write benchmark
    write_start = time.time()
    mem_before_write = process.memory_info().rss
    cpu_before_write = psutil.cpu_percent(interval=None)

    parquet_path = (
        temp_dir
        / f"test_{config['parquet']['compression']}_{config['parquet']['row_group_size']}.parquet"
    )
    write_parquet(df, parquet_path, config)

    write_time = time.time() - write_start
    mem_after_write = process.memory_info().rss
    cpu_after_write = psutil.cpu_percent(interval=None)

    # Read benchmark
    read_start = time.time()
    mem_before_read = process.memory_info().rss
    cpu_before_read = psutil.cpu_percent(interval=None)

    df_read = read_parquet(parquet_path, config)

    read_time = time.time() - read_start
    mem_after_read = process.memory_info().rss
    cpu_after_read = psutil.cpu_percent(interval=None)

    # Calculate metrics
    file_size = parquet_path.stat().st_size / (1024 * 1024)  # MB

    return {
        "compression": config["parquet"]["compression"],
        "row_group_size": config["parquet"]["row_group_size"],
        "use_columns": len(config["parquet"].get("use_columns", [])) > 0,
        "write_time_sec": write_time,
        "read_time_sec": read_time,
        "file_size_mb": file_size,
        "write_throughput_mbps": (file_size / write_time) if write_time > 0 else 0,
        "read_throughput_mbps": (file_size / read_time) if read_time > 0 else 0,
        "write_cpu_percent": cpu_after_write,
        "read_cpu_percent": cpu_after_read,
        "write_mem_mb": (mem_after_write - mem_before_write) / (1024 * 1024),
        "read_mem_mb": (mem_after_read - mem_before_read) / (1024 * 1024),
    }


def run_benchmark(
    n_rows: int = 100000, output_dir: Path = Path("reports")
) -> pd.DataFrame:
    """Run comprehensive I/O benchmark"""
    output_dir.mkdir(exist_ok=True)

    # Generate test data
    df = generate_test_data(n_rows)
    print(f"Generated test data: {len(df)} rows, {len(df.columns)} columns")

    # Benchmark configurations
    configs = []

    # Compression types
    compressions = ["snappy", "lz4"]
    try:
        import pyarrow as pa

        if "ZSTD" in pa.Codec.list_codecs():
            compressions.append("zstd")
    except Exception:
        pass

    # Row group sizes
    row_group_sizes = [20000, 50000, 100000]

    # Column projection: all vs subset
    column_configs = [
        [],  # All columns
        ["timestamp", "close", "volume", "rsi"],  # Essential columns
    ]

    for compression in compressions:
        for row_group_size in row_group_sizes:
            for use_columns in column_configs:
                config = {
                    "parquet": {
                        "compression": compression,
                        "row_group_size": row_group_size,
                        "use_columns": use_columns,
                        "engine": "pyarrow",
                    },
                    "limits": {"peak_memory_mb": 1200},
                }
                configs.append(config)

    # Run benchmarks
    results = []
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for i, config in enumerate(configs):
            print(f"Benchmark {i + 1}/{len(configs)}: {config['parquet']}")
            try:
                result = measure_io_performance(df, config, temp_path)
                results.append(result)
            except Exception as e:
                print(f"Error in benchmark: {e}")
                continue

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    csv_path = output_dir / "io_bench_results.csv"
    results_df.to_csv(csv_path, index=False)

    # Generate plots
    generate_plots(results_df, output_dir)

    return results_df


def generate_plots(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate performance comparison plots"""
    # Group by configuration for top 3
    grouped = (
        results_df.groupby(["compression", "row_group_size", "use_columns"])
        .agg(
            {
                "write_throughput_mbps": "mean",
                "read_throughput_mbps": "mean",
                "write_cpu_percent": "mean",
                "read_cpu_percent": "mean",
                "write_mem_mb": "mean",
                "read_mem_mb": "mean",
            }
        )
        .reset_index()
    )

    # Sort by read throughput (primary metric)
    top_configs = grouped.nlargest(3, "read_throughput_mbps")

    # Plot throughput
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Write throughput
    axes[0, 0].bar(range(len(top_configs)), top_configs["write_throughput_mbps"])
    axes[0, 0].set_title("Write Throughput (MB/s)")
    axes[0, 0].set_xticks(range(len(top_configs)))
    axes[0, 0].set_xticklabels(
        [
            f"{row['compression']}\n{row['row_group_size']}\n{'subset' if row['use_columns'] else 'all'}"
            for _, row in top_configs.iterrows()
        ],
        fontsize=8,
    )

    # Read throughput
    axes[0, 1].bar(range(len(top_configs)), top_configs["read_throughput_mbps"])
    axes[0, 1].set_title("Read Throughput (MB/s)")
    axes[0, 1].set_xticks(range(len(top_configs)))
    axes[0, 1].set_xticklabels(
        [
            f"{row['compression']}\n{row['row_group_size']}\n{'subset' if row['use_columns'] else 'all'}"
            for _, row in top_configs.iterrows()
        ],
        fontsize=8,
    )

    # CPU usage
    axes[1, 0].bar(
        range(len(top_configs)),
        top_configs["write_cpu_percent"],
        label="Write",
        alpha=0.7,
    )
    axes[1, 0].bar(
        range(len(top_configs)),
        top_configs["read_cpu_percent"],
        label="Read",
        alpha=0.7,
    )
    axes[1, 0].set_title("CPU Usage (%)")
    axes[1, 0].legend()

    # Memory usage
    axes[1, 1].bar(
        range(len(top_configs)), top_configs["write_mem_mb"], label="Write", alpha=0.7
    )
    axes[1, 1].bar(
        range(len(top_configs)), top_configs["read_mem_mb"], label="Read", alpha=0.7
    )
    axes[1, 1].set_title("Memory Usage (MB)")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "io_bench_top3.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Top 3 configurations saved to {output_dir / 'io_bench_top3.png'}")


def generate_weekly_report_section(results: pd.DataFrame) -> str:
    """
    Generate markdown section for weekly report

    Args:
        results: Benchmark results DataFrame

    Returns:
        Markdown content for weekly report
    """
    if results.empty:
        return "## I/O Performance Benchmark\n\nNo benchmark results available.\n"

    # Get top 3 configurations
    top3 = results.nlargest(3, "read_throughput_mbps")

    report_lines = []
    report_lines.append("## I/O Performance Benchmark")
    report_lines.append(
        f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*"
    )
    report_lines.append("")

    # Summary statistics
    report_lines.append("### Performance Summary")
    report_lines.append(
        f"- **Best Read Throughput**: {results['read_throughput_mbps'].max():.1f} MB/s"
    )
    report_lines.append(
        f"- **Best Write Throughput**: {results['write_throughput_mbps'].max():.1f} MB/s"
    )
    report_lines.append(
        f"- **Lowest Memory Usage**: {results[['write_mem_mb', 'read_mem_mb']].min().min():.1f} MB"
    )
    report_lines.append(f"- **Configurations Tested**: {len(results)}")
    report_lines.append("")

    # Top 3 configurations table
    report_lines.append("### Top 3 I/O Configurations")
    report_lines.append(
        "| Rank | Compression | Row Group | Columns | Read MB/s | Write MB/s | Memory MB |"
    )
    report_lines.append(
        "|------|-------------|-----------|---------|-----------|------------|-----------|"
    )

    for i, (_, row) in enumerate(top3.iterrows(), 1):
        columns_desc = "All" if not row["use_columns"] else "Essential"
        total_mem = row["write_mem_mb"] + row["read_mem_mb"]

        report_lines.append(
            f"| {i} | {row['compression']} | {row['row_group_size'] // 1000}K | {columns_desc} | "
            f"{row['read_throughput_mbps']:.1f} | {row['write_throughput_mbps']:.1f} | {total_mem:.1f} |"
        )

    report_lines.append("")

    # Recommended configuration
    best_config = top3.iloc[0]
    report_lines.append("### Recommended Configuration")
    report_lines.append(f"**Optimal for Raspberry Pi environments:**")
    report_lines.append(f"- Compression: `{best_config['compression']}`")
    report_lines.append(f"- Row Group Size: `{best_config['row_group_size']:,}`")
    columns_rec = (
        "Use column projection (essential columns only)"
        if best_config["use_columns"]
        else "Read all columns"
    )
    report_lines.append(f"- Column Strategy: {columns_rec}")
    report_lines.append(
        f"- Expected Performance: {best_config['read_throughput_mbps']:.1f} MB/s read, {best_config['write_throughput_mbps']:.1f} MB/s write"
    )
    report_lines.append("")

    # Performance insights
    report_lines.append("### Performance Insights")

    # Compression analysis
    compression_perf = (
        results.groupby("compression")["read_throughput_mbps"]
        .mean()
        .sort_values(ascending=False)
    )
    best_compression = compression_perf.index[0]
    report_lines.append(
        f"- **Best compression for speed**: {best_compression} ({compression_perf.iloc[0]:.1f} MB/s avg)"
    )

    # Memory usage analysis
    if "use_columns" in results.columns:
        proj_results = results[results["use_columns"].astype(str) != "[]"]
        no_proj_results = results[results["use_columns"].astype(str) == "[]"]

        if not proj_results.empty and not no_proj_results.empty:
            proj_mem = proj_results[["read_mem_mb", "write_mem_mb"]].mean().mean()
            no_proj_mem = no_proj_results[["read_mem_mb", "write_mem_mb"]].mean().mean()
            mem_savings = (no_proj_mem - proj_mem) / no_proj_mem * 100

            report_lines.append(
                f"- **Column projection savings**: {mem_savings:.1f}% memory reduction"
            )

    report_lines.append("")
    return "\n".join(report_lines)


def update_weekly_report(
    benchmark_results: pd.DataFrame, weekly_report_path: str = "weekly_report.md"
) -> None:
    """
    Update weekly report with I/O benchmark results

    Args:
        benchmark_results: Results from I/O benchmarking
        weekly_report_path: Path to weekly report file
    """
    # Generate benchmark section
    benchmark_section = generate_weekly_report_section(benchmark_results)

    # Read existing weekly report
    if os.path.exists(weekly_report_path):
        with open(weekly_report_path, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = "# Weekly Feature Evaluation Report\n\n"

    # Remove existing I/O benchmark section if present
    io_section_start = content.find("## I/O Performance Benchmark")
    if io_section_start != -1:
        # Find the next section or end of file
        next_section = content.find("\n## ", io_section_start + 1)
        if next_section != -1:
            content = content[:io_section_start] + content[next_section:]
        else:
            content = content[:io_section_start]

    # Add new benchmark section before "Notes" section or at the end
    if "## Notes" in content:
        insert_pos = content.find("## Notes")
        updated_content = (
            content[:insert_pos] + benchmark_section + "\n" + content[insert_pos:]
        )
    else:
        updated_content = content.rstrip() + "\n\n" + benchmark_section

    # Write updated report
    with open(weekly_report_path, "w", encoding="utf-8") as f:
        f.write(updated_content)

    print(f"Weekly report updated with I/O benchmark results: {weekly_report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="I/O Benchmarking Tool")
    parser.add_argument(
        "--rows", "-r", type=int, default=100000, help="Number of test rows"
    )
    parser.add_argument("--output", "-o", default="reports", help="Output directory")
    parser.add_argument(
        "--update-weekly", action="store_true", help="Update weekly report with results"
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    print("Starting I/O benchmark...")
    results = run_benchmark(args.rows, output_dir)

    # Print top 3
    top3 = results.nlargest(3, "read_throughput_mbps")
    print("\nTop 3 configurations by read throughput:")
    for _, row in top3.iterrows():
        columns_desc = "All" if not row["use_columns"] else "Essential"
        print(
            f"  {row['compression']}/{row['row_group_size'] // 1000}K/{columns_desc}: {row['read_throughput_mbps']:.1f} MB/s read"
        )

    # Update weekly report if requested
    if args.update_weekly:
        update_weekly_report(results)
        print(f"\nBenchmark section added to weekly report")


if __name__ == "__main__":
    main()  # type: ignore
