#!/usr/bin/env python3
"""Monitor memory usage of running training process."""

import psutil
import time
from datetime import datetime

def monitor_training_process(pid: int, duration_seconds: int = 60):
    """Monitor memory usage of training process."""
    try:
        process = psutil.Process(pid)
        print(f"Monitoring PID {pid} for {duration_seconds} seconds...")
        print(f"{'Time':<12} {'RSS (MB)':<12} {'VMS (MB)':<12} {'CPU %':<8}")
        print("-" * 50)
        
        start_time = time.time()
        max_memory = 0
        
        while time.time() - start_time < duration_seconds:
            try:
                mem_info = process.memory_info()
                rss_mb = mem_info.rss / (1024 * 1024)
                vms_mb = mem_info.vms / (1024 * 1024)
                cpu_percent = process.cpu_percent(interval=1.0)
                
                max_memory = max(max_memory, rss_mb)
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"{timestamp:<12} {rss_mb:<12.2f} {vms_mb:<12.2f} {cpu_percent:<8.1f}")
                
                time.sleep(2)
            except psutil.NoSuchProcess:
                print(f"\n⚠️  Process {pid} terminated")
                break
            except Exception as e:
                print(f"Error: {e}")
                break
        
        print("-" * 50)
        print(f"Maximum memory usage: {max_memory:.2f} MB")
        
        # Check if under 500MB limit
        if max_memory < 500:
            print(f"✅ Memory usage stayed under 500MB limit!")
        else:
            print(f"⚠️  Memory usage exceeded 500MB limit")
            
    except psutil.NoSuchProcess:
        print(f"❌ Process {pid} not found")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python monitor_training_memory.py <PID> [duration_seconds]")
        sys.exit(1)
    
    pid = int(sys.argv[1])
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    
    monitor_training_process(pid, duration)
