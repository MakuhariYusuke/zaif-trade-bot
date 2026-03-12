"""Monitor memory usage during training"""
import sys
import time

import psutil

from ztb.trading.environment.constants import BYTES_PER_MB

def monitor_memory_usage(pid: int | None = None, duration: int = 60) -> dict:
    """Monitor memory usage for a process.

    Args:
        pid: Process ID to monitor (None for current process)
        duration: Duration to monitor in seconds

    Returns:
        Dictionary with memory usage statistics
    """
    if pid is None:
        process = psutil.Process()
    else:
        process = psutil.Process(pid)

    print("Monitoring memory usage...")
    print(f"{'Time':>8} {'RSS (MB)':>12} {'VMS (MB)':>12} {'% MEM':>8}")
    print("-" * 50)

    start_time = time.time()
    memory_stats = []

    try:
        while time.time() - start_time < duration:
            mem_info = process.memory_info()
            mem_percent = process.memory_percent()
            elapsed = time.time() - start_time

            stats = {
                "time": elapsed,
                "rss_mb": mem_info.rss / BYTES_PER_MB,
                "vms_mb": mem_info.vms / BYTES_PER_MB,
                "mem_percent": mem_percent,
            }
            memory_stats.append(stats)

            print(
                f"{elapsed:8.1f} {stats['rss_mb']:12.1f} {stats['vms_mb']:12.1f} {stats['mem_percent']:8.1f}%"
            )
            sys.stdout.flush()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

    return {
        "process_id": process.pid,
        "duration": time.time() - start_time,
        "memory_stats": memory_stats,
        "peak_rss_mb": max(s["rss_mb"] for s in memory_stats) if memory_stats else 0,
        "peak_vms_mb": max(s["vms_mb"] for s in memory_stats) if memory_stats else 0,
        "avg_mem_percent": sum(s["mem_percent"] for s in memory_stats)
        / len(memory_stats)
        if memory_stats
        else 0,
    }

if __name__ == "__main__":
    monitor_memory_usage()
