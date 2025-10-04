#!/usr/bin/env python3
"""
Compression benchmark for cache optimization
キャッシュ最適化のための圧縮ベンチマーク
"""

import pickle
import time
import zlib

import lz4.frame
import numpy as np
import pandas as pd
import zstandard as zstd


def create_test_data(size_mb: int = 10) -> pd.DataFrame:
    """テストデータ作成（DataFrame）"""
    n_rows = size_mb * 1024 * 1024 // 100  # 概算
    data = {
        "timestamp": pd.date_range("2020-01-01", periods=n_rows, freq="1min"),
        "open": np.random.uniform(100, 200, n_rows),
        "high": np.random.uniform(100, 200, n_rows),
        "low": np.random.uniform(100, 200, n_rows),
        "close": np.random.uniform(100, 200, n_rows),
        "volume": np.random.uniform(0, 1000, n_rows),
        "rsi": np.random.uniform(0, 100, n_rows),
        "macd": np.random.uniform(-50, 50, n_rows),
        "bb_upper": np.random.uniform(100, 200, n_rows),
        "bb_lower": np.random.uniform(100, 200, n_rows),
    }
    return pd.DataFrame(data)


def benchmark_compressor(
    name: str,
    compressor_func: callable,
    decompressor_func: callable,
    data: bytes,
    iterations: int = 5,
) -> dict:
    """圧縮/展開ベンチマーク"""
    original_size = len(data)

    # 圧縮ベンチマーク
    compress_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        compressed = compressor_func(data)
        compress_times.append(time.perf_counter() - start)

    compressed_size = len(compressed)
    compression_ratio = (1 - compressed_size / original_size) * 100

    # 展開ベンチマーク
    decompress_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        decompressed = decompressor_func(compressed)
        decompress_times.append(time.perf_counter() - start)

    # 整合性チェック
    assert decompressed == data, f"Data corruption in {name}"

    return {
        "compressor": name,
        "original_size_mb": original_size / (1024 * 1024),
        "compressed_size_mb": compressed_size / (1024 * 1024),
        "compression_ratio": compression_ratio,
        "compress_time_avg": np.mean(compress_times) * 1000,  # ms
        "decompress_time_avg": np.mean(decompress_times) * 1000,  # ms
        "compress_speed_mbs": original_size / (1024 * 1024) / np.mean(compress_times),
        "decompress_speed_mbs": original_size
        / (1024 * 1024)
        / np.mean(decompress_times),
    }


def main():
    print("🧪 Compression Benchmark for Trading Data Cache")
    print("=" * 60)

    # テストデータ作成
    test_data = create_test_data(size_mb=5)  # 5MBのテストデータ
    pickled_data = pickle.dumps(test_data, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Test data size: {len(pickled_data) / (1024 * 1024):.2f} MB")
    print()

    # 圧縮方式の定義
    compressors = {
        "zlib-1": (lambda d: zlib.compress(d, 1), zlib.decompress),
        "zlib-6": (lambda d: zlib.compress(d, 6), zlib.decompress),
        "zlib-9": (lambda d: zlib.compress(d, 9), zlib.decompress),
        "zstd-1": (
            lambda d: zstd.ZstdCompressor(level=1).compress(d),
            lambda c: zstd.ZstdDecompressor().decompress(c),
        ),
        "zstd-3": (
            lambda d: zstd.ZstdCompressor(level=3).compress(d),
            lambda c: zstd.ZstdDecompressor().decompress(c),
        ),
        "zstd-10": (
            lambda d: zstd.ZstdCompressor(level=10).compress(d),
            lambda c: zstd.ZstdDecompressor().decompress(c),
        ),
        "lz4": (lz4.frame.compress, lz4.frame.decompress),
    }

    results = []
    for name, (comp_func, decomp_func) in compressors.items():
        print(f"Testing {name}...")
        try:
            result = benchmark_compressor(name, comp_func, decomp_func, pickled_data)
            results.append(result)
            print(".2f")
        except Exception as e:
            print(f"  ❌ Failed: {e}")

    print("\n📊 Results Summary:")
    print("-" * 100)
    print(
        f"{'Compressor':<10} {'Ratio%':<8} {'Comp MB/s':<10} {'Decomp MB/s':<12} {'Comp Time(ms)':<12} {'Decomp Time(ms)':<14}"
    )
    print("-" * 100)

    for r in sorted(results, key=lambda x: x["decompress_time_avg"]):
        print(
            f"{r['compressor']:<10} {r['compression_ratio']:<8.1f} {r['compress_speed_mbs']:<10.1f} {r['decompress_speed_mbs']:<12.1f} {r['compress_time_avg']:<12.2f} {r['decompress_time_avg']:<14.2f}"
        )

    # 推奨設定
    print("\n🎯 Recommendations for Trading Cache:")
    print("- Fast training (high iteration): Use lz4 or zstd-1")
    print("- Balanced (normal training): Use zstd-3 or zlib-6")
    print("- Best compression (disk space): Use zstd-10 or zlib-9")


if __name__ == "__main__":
    main()
