class DistributedConfig:
    def __init__(self, *args, **kwargs):
        self.config = kwargs

class DistributedCoordinator:
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.workers = {}

    def start(self):
        self._running = True
        return True

    def stop(self):
        self._running = False
        return True

    # Backwards-compatible methods used by performance tests
    def register_worker(self, worker_info):
        wid = getattr(worker_info, "worker_id", None)
        if wid is None:
            return False
        self.workers[wid] = worker_info
        return True

    def stop_coordination(self):
        return self.stop()

    def aggregate_metrics(self, worker_metrics):
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
                aggregated[k] = {"mean": round(mean_val, 2), "min": min(vals), "max": max(vals), "sum": sum(vals), "count": len(vals)}
            else:
                aggregated[k] = {"values": vals, "count": len(vals)}
        return aggregated
