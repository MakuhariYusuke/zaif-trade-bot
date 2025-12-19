import time


class _SimpleMemoryMonitor:
    def get_memory_stats(self):
        return {"current_mb": 0, "peak_mb": 0}

    def force_cleanup(self):
        return None


class WorkerPoolStub:
    def __init__(self, num_workers=1, config=None):
        self.num_workers = num_workers

    def start_pool(self):
        return True

    def stop_pool(self):
        return None

    def submit_task(self, task_type, task_data, callback=None):
        # Simulate per-task processing time inversely proportional to worker count
        processing_delay = 0.01 / max(1, getattr(self, "num_workers", 1))
        time.sleep(processing_delay)
        if callback:
            try:
                callback({"task_type": task_type, "task_data": task_data, "result": True})
            except Exception:
                pass
        return f"{task_type}_{int(time.time()*1000)}"


class DistributedTrainingManager:
    def __init__(self, config=None):
        self.config = config
        self.coordinator = None
        self.memory_monitor = _SimpleMemoryMonitor()
        self.worker_pool = WorkerPoolStub(getattr(config, "num_workers", 1), config)
        self.is_initialized = False
        self.training_active = False

    def initialize(self, *args, **kwargs):
        self.is_initialized = True
        return True

    def shutdown(self):
        self.is_initialized = False
        return None

    def start_distributed_training(self, training_config):
        if not self.is_initialized:
            return False
        self.training_active = True
        return True

    def stop_distributed_training(self):
        self.training_active = False
        return None

    def submit_training_task(self, task_type, task_data, callback=None):
        return self.worker_pool.submit_task(task_type, task_data, callback)

    def get_training_status(self):
        return {
            "training_active": self.training_active,
            "memory_status": self.memory_monitor.get_memory_stats(),
        }
