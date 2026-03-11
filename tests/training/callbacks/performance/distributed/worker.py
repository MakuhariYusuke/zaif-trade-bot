import time


class WorkerPool:
    def __init__(self, *args, config=None):
        # Support WorkerPool(num_workers, config) or WorkerPool(config=config)
        if len(args) >= 1 and isinstance(args[0], int):
            self.n_workers = args[0]
        else:
            self.n_workers = 1
        self.config = config
        self._running = False

    def start_pool(self):
        self._running = True
        return True

    def stop_pool(self):
        self._running = False
        return None

    def submit_task(self, task_type, task_data, callback=None):
        # Synchronously execute the callback with a mock result
        result = {"task_type": task_type, "task_data": task_data, "result": True}
        if callback:
            try:
                callback(result)
            except Exception:
                pass

        return f"{task_type}_{int(time.time() * 1000)}"

    def get_pool_status(self):
        return {"pool_running": self._running, "num_workers": self.n_workers}
