import os

import psutil

# 現在のプロセスを確認
current_pid = os.getpid()
print(f"Current process PID: {current_pid}")

# Pythonプロセスを探す
python_processes = []
for proc in psutil.process_iter(["pid", "name", "cmdline"]):
    try:
        if proc.info["name"] and "python" in proc.info["name"].lower():
            cmdline = proc.info["cmdline"]
            if cmdline and len(cmdline) > 1:
                if "unified_trainer" in " ".join(cmdline):
                    python_processes.append(proc.info)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue

if python_processes:
    print("Found unified_trainer processes:")
    for proc in python_processes:
        print(f'  PID: {proc["pid"]}, Command: {" ".join(proc["cmdline"])[:100]}...')
else:
    print("No unified_trainer processes found")
