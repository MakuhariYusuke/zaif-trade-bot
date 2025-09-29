"""Observability helpers providing structured logging, metrics, correlation IDs, and artifact export."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def generate_correlation_id() -> str:
    """Return a random correlation identifier."""
    return uuid.uuid4().hex


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    return logging.getLogger(name)


class JsonLogFormatter(logging.Formatter):
    """Formatter that emits records as one-line JSON objects."""

    def __init__(self, correlation_id: str) -> None:
        super().__init__()
        self._correlation_id = correlation_id

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'correlation_id': self._correlation_id,
        }
        if record.exc_info:
            payload['exception'] = self.formatException(record.exc_info)
        extra = getattr(record, 'extra', None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)


@dataclass
class ObservabilityClient:
    """Coordinates structured logging, metrics emission, and artifact export."""

    logger: logging.Logger
    metrics_path: Path
    artifacts_dir: Path
    correlation_id: str
    _metrics_lock: threading.Lock = field(default_factory=threading.Lock)

    def log_event(self, event: str, payload: Optional[Dict[str, Any]] = None, level: int = logging.INFO) -> None:
        data = payload.copy() if payload else {}
        data['event'] = event
        data['correlation_id'] = self.correlation_id
        self.logger.log(level, '', extra={'extra': data})

    def record_metrics(self, metrics: Dict[str, Any]) -> None:
        with self._metrics_lock:
            lines = []
            for key, value in metrics.items():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                lines.append(f"{key}{{correlation_id=\"{self.correlation_id}\"}} {numeric}")
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self.metrics_path.write_text("\n".join(lines) + "\n", encoding='utf-8')

    def export_artifact(self, name: str, data: Any) -> Path:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = self.artifacts_dir / f"{name}.json"
        wrapped = {
            'correlation_id': self.correlation_id,
            'timestamp': datetime.utcnow().isoformat(),
            'payload': data,
        }
        with artifact_path.open('w', encoding='utf-8') as fh:
            json.dump(wrapped, fh, indent=2, ensure_ascii=False)
        return artifact_path

    def close(self) -> None:
        handlers = list(self.logger.handlers)
        for handler in handlers:
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)


def setup_observability(name: str, base_dir: Path, correlation_id: Optional[str] = None) -> ObservabilityClient:
    correlation = correlation_id or generate_correlation_id()
    base_dir.mkdir(parents=True, exist_ok=True)
    log_path = base_dir / f"{name}.log.jsonl"
    metrics_path = base_dir / f"{name}.prom"
    artifacts_dir = base_dir / 'artifacts'

    logger = logging.getLogger(f"observability.{name}.{correlation}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding='utf-8')
    handler.setFormatter(JsonLogFormatter(correlation))
    logger.handlers = [handler]

    return ObservabilityClient(
        logger=logger,
        metrics_path=metrics_path,
        artifacts_dir=artifacts_dir,
        correlation_id=correlation,
    )
