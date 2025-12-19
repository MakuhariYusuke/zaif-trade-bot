from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import Dict
from datetime import datetime
import pickle
import threading
import time


@dataclass
class DistributedConfig:
    enable_distributed: bool = True
    num_workers: int = 1
    sync_interval: float = 1.0
    heartbeat_interval: float = 2.0
    max_queue_size: int = 10


class Message:
    def __init__(self, msg_type: str, sender_id: int, data: dict, timestamp: datetime = None):
        self.msg_type = msg_type
        self.sender_id = sender_id
        self.data = data
        self.timestamp = timestamp or datetime.now()

    def to_bytes(self) -> bytes:
        msg = {
            "msg_type": self.msg_type,
            "sender_id": self.sender_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }
        return pickle.dumps(msg)

    @classmethod
    def from_bytes(cls, b: bytes) -> "Message":
        obj = pickle.loads(b)
        ts = datetime.fromisoformat(obj.get("timestamp"))
        return cls(obj.get("msg_type"), obj.get("sender_id"), obj.get("data"), ts)


@dataclass
class WorkerInfo:
    worker_id: int
    host: str
    port: int
    status: str = "idle"
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metrics: dict = field(default_factory=dict)


class DistributedCoordinator:
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.is_master = False
        self.workers: Dict[int, WorkerInfo] = {}
        self.message_queue = Queue()
        self.stats = {"messages_sent": 0, "messages_received": 0, "tasks_distributed": 0}

    def register_worker(self, worker: WorkerInfo) -> bool:
        if worker.worker_id in self.workers:
            return False
        self.workers[worker.worker_id] = worker
        return True

    def unregister_worker(self, worker_id: int) -> bool:
        if worker_id in self.workers:
            del self.workers[worker_id]
            return True
        return False

    def distribute_task(self, task_data: dict, worker_id: int = None):
        # Simple round-robin / pick first
        if not self.workers:
            return None
        wid = worker_id if worker_id is not None else next(iter(self.workers))
        self.stats["tasks_distributed"] = self.stats.get("tasks_distributed", 0) + 1
        return wid

    def start_coordination(self):
        # Start a background thread to process messages from the queue
        if hasattr(self, "_running") and self._running:
            return None

        self._running = True
        self._thread = threading.Thread(target=self._process_messages, daemon=True)
        self._thread.start()
        return None

    def stop_coordination(self):
        self._running = False
        if hasattr(self, "_thread") and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        return None

    def _process_messages(self):
        while getattr(self, "_running", False):
            try:
                msg = self.message_queue.get(timeout=0.1)
            except Empty:
                continue

            # Simple message handling
            if getattr(msg, "msg_type", None) == "heartbeat":
                wid = getattr(msg, "sender_id", None)
                if wid in self.workers:
                    self.workers[wid].status = "idle"
                    self.workers[wid].last_heartbeat = msg.timestamp
            elif getattr(msg, "msg_type", None) == "metrics":
                # aggregate or update metrics
                wid = getattr(msg, "sender_id", None)
                if wid in self.workers:
                    self.workers[wid].metrics.update(getattr(msg, "data", {}))

    def aggregate_metrics(self, worker_metrics: Dict[int, Dict[str, object]]) -> Dict[str, object]:
        if not worker_metrics:
            return {}

        aggregated = {}
        keys = set()
        for m in worker_metrics.values():
            keys.update(m.keys())

        for k in keys:
            vals = [m[k] for m in worker_metrics.values() if k in m]
            if not vals:
                continue
            if all(isinstance(v, (int, float)) for v in vals):
                mean_val = sum(vals) / len(vals)
                aggregated[k] = {
                    "mean": round(mean_val, 2),
                    "min": min(vals),
                    "max": max(vals),
                    "sum": sum(vals),
                    "count": len(vals),
                }
            else:
                aggregated[k] = {"values": vals, "count": len(vals)}

        return aggregated
