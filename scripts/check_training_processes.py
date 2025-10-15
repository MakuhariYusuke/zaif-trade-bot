import psutil

# Find all Python processes
print("All Python processes:")
for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        if 'python' in proc.info['name'].lower():
            cmdline = proc.cmdline()
            print(f"PID {proc.pid}: {' '.join(cmdline[:4]) if cmdline else 'no cmdline'}")
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue

# Check for training processes specifically
print("\nChecking for training processes:")
training_found = False
for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        if 'python' in proc.info['name'].lower():
            cmdline = proc.cmdline()
            if cmdline and any('train_sac' in arg for arg in cmdline):
                training_found = True
                print(f"Found training process: PID {proc.pid}")
                print(f"  Full command: {' '.join(cmdline)}")
                print(f"  CPU %: {proc.cpu_percent(interval=1):.1f}")
                print(f"  Memory MB: {proc.memory_info().rss / 1024 / 1024:.1f}")
                print(f"  Status: {proc.status()}")
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue

if not training_found:
    print("No training processes found.")