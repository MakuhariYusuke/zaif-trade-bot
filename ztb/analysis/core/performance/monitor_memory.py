"""Monitor memory usage during training"""
import sys
import time

import psutil

process = psutil.Process()

print("Monitoring memory usage (Ctrl+C to stop)...")
print(f"{'Time':>8} {'RSS (MB)':>12} {'VMS (MB)':>12} {'% MEM':>8}")
print("-" * 50)

try:
    start_time = time.time()
    while True:
        mem_info = process.memory_info()
        mem_percent = process.memory_percent()
        elapsed = time.time() - start_time

        print(
            f"{elapsed:8.1f} {mem_info.rss / 1024 / 1024:12.1f} {mem_info.vms / 1024 / 1024:12.1f} {mem_percent:8.1f}%"
        )
        sys.stdout.flush()
        time.sleep(5)
except KeyboardInterrupt:
    print("\nMonitoring stopped.")
