import time
from typing import Dict


class DistributedWorker:
    def __init__(self, worker_id: int, config):
        self.worker_id = worker_id
        self.config = config
        self.is_running = False
        self.status = "idle"
        self.stats = {"tasks_executed": 0}
        # Simple process-like object for tests
        class _Proc:
            def __init__(self, owner):
                self._owner = owner

            def is_alive(self):
                return bool(self._owner.is_running)

        self.process = _Proc(self)

    def start(self):
        self.is_running = True
        self.status = "running"
        return True

    def stop(self):
        self.is_running = False
        self.status = "stopped"
        return True

    def send_task(self, task_name: str, data: dict, timeout: float = 5.0):
        # simple synchronous emulation
        time.sleep(0.01)
        self.stats["tasks_executed"] += 1
        return {"completed": True, "validation_loss": 0.0, "validation_accuracy": 1.0}

    def get_status(self):
        return {
            "stats": self.stats,
            "status": self.status,
            "is_running": self.is_running,
            "process_alive": self.process.is_alive(),
        }


class WorkerPool:
    def __init__(self, num_workers: int, config):
        self.num_workers = num_workers
        self.config = config
        self._running = False
        self.workers: Dict[int, DistributedWorker] = {}

    def start_pool(self):
        self._running = True
        for i in range(self.num_workers):
            self.workers[i] = DistributedWorker(i, self.config)
            self.workers[i].start()
        return True

    def stop_pool(self):
        for w in list(self.workers.values()):
            w.stop()
        self.workers = {}
        self._running = False

    def submit_task(self, task_name: str, data: dict, callback=None):
        wid = next(iter(self.workers))
        res = self.workers[wid].send_task(task_name, data)
        if callback:
            callback(res)
        return 1

    def get_pool_status(self):
        return {"pool_running": self._running, "num_workers": len(self.workers), "worker_details": {i: w.get_status() for i, w in self.workers.items()}}
