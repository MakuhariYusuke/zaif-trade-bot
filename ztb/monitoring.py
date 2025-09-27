"""
Prometheus metrics exporter for trading bot monitoring.

Exposes metrics for data pipeline, job execution, and trading performance.
"""
import logging
import time
from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)

class PrometheusExporter:
    """
    Prometheus metrics exporter for comprehensive monitoring.
    """

    def __init__(self, port: int = 8000):
        self.port = port

        # Data pipeline metrics
        self.data_fetch_total = Counter(
            'ztb_data_fetch_total',
            'Total number of data fetch operations',
            ['status']
        )
        self.data_fetch_duration = Histogram(
            'ztb_data_fetch_duration_seconds',
            'Time spent fetching data',
            ['source']
        )
        self.data_integrity_checks = Counter(
            'ztb_data_integrity_checks_total',
            'Total number of data integrity checks',
            ['result']
        )

        # Job execution metrics
        self.job_executions_total = Counter(
            'ztb_job_executions_total',
            'Total number of ML job executions',
            ['status']
        )
        self.job_execution_duration = Histogram(
            'ztb_job_execution_duration_seconds',
            'Time spent executing ML jobs'
        )
        self.active_jobs = Gauge(
            'ztb_active_jobs',
            'Number of currently active jobs'
        )

        # Trading metrics
        self.trading_orders_total = Counter(
            'ztb_trading_orders_total',
            'Total number of trading orders',
            ['side', 'status']
        )
        self.portfolio_balance = Gauge(
            'ztb_portfolio_balance',
            'Current portfolio balance'
        )
        self.portfolio_pnl = Gauge(
            'ztb_portfolio_pnl',
            'Current portfolio P&L'
        )
        self.risk_drawdown = Gauge(
            'ztb_risk_drawdown_percent',
            'Current drawdown percentage'
        )

        # Quality gates and drift monitoring
        self.data_drift_score = Gauge(
            'ztb_data_drift_score',
            'Data drift detection score (0-1, higher = more drift)',
            ['feature_type']
        )
        self.model_performance_drift = Gauge(
            'ztb_model_performance_drift',
            'Model performance drift indicator',
            ['metric_type']
        )
        self.quality_gates_passed = Counter(
            'ztb_quality_gates_passed_total',
            'Total number of quality gates passed',
            ['gate_type']
        )
        self.quality_gates_failed = Counter(
            'ztb_quality_gates_failed_total',
            'Total number of quality gates failed',
            ['gate_type', 'reason']
        )
        self.alerts_triggered = Counter(
            'ztb_alerts_triggered_total',
            'Total number of alerts triggered',
            ['alert_type', 'severity']
        )

        # System metrics
        self.system_errors_total = Counter(
            'ztb_system_errors_total',
            'Total number of system errors',
            ['component']
        )

    def start_server(self) -> None:
        """Start Prometheus metrics server"""
        try:
            start_http_server(self.port)
            logger.info(f"Prometheus metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")

    # Data pipeline metrics
    def record_data_fetch(self, status: str, duration: float, source: str = "binance") -> None:
        """Record data fetch operation"""
        self.data_fetch_total.labels(status=status).inc()
        self.data_fetch_duration.labels(source=source).observe(duration)

    def record_integrity_check(self, result: str) -> None:
        """Record data integrity check result"""
        self.data_integrity_checks.labels(result=result).inc()

    # Job execution metrics
    def record_job_start(self) -> None:
        """Record job start"""
        self.active_jobs.inc()

    def record_job_completion(self, status: str, duration: float) -> None:
        """Record job completion"""
        self.active_jobs.dec()
        self.job_executions_total.labels(status=status).inc()
        self.job_execution_duration.observe(duration)

    # Trading metrics
    def record_order(self, side: str, status: str) -> None:
        """Record trading order"""
        self.trading_orders_total.labels(side=side, status=status).inc()

    def update_portfolio_metrics(self, balance: float, pnl: float, drawdown: float) -> None:
        """Update portfolio metrics"""
        self.portfolio_balance.set(balance)
        self.portfolio_pnl.set(pnl)
        self.risk_drawdown.set(drawdown)

    # System metrics
    def record_error(self, component: str) -> None:
        """Record system error"""
        self.system_errors_total.labels(component=component).inc()

    # Quality gates and drift monitoring
    def record_data_drift(self, feature_type: str, drift_score: float) -> None:
        """Record data drift detection score"""
        self.data_drift_score.labels(feature_type=feature_type).set(drift_score)

    def record_model_performance_drift(self, metric_type: str, drift_value: float) -> None:
        """Record model performance drift"""
        self.model_performance_drift.labels(metric_type=metric_type).set(drift_value)

    def record_quality_gate_result(self, gate_type: str, passed: bool, reason: str = "") -> None:
        """Record quality gate result"""
        if passed:
            self.quality_gates_passed.labels(gate_type=gate_type).inc()
        else:
            self.quality_gates_failed.labels(gate_type=gate_type, reason=reason).inc()

    def trigger_alert(self, alert_type: str, severity: str, message: str = "") -> None:
        """Trigger an alert"""
        self.alerts_triggered.labels(alert_type=alert_type, severity=severity).inc()
        logger.warning(f"Alert triggered: {alert_type} ({severity}) - {message}")

        # Send notification if available
        try:
            from .notifications import send_notification
            send_notification(
                title=f"Alert: {alert_type}",
                message=f"Severity: {severity}\n{message}",
                priority="high" if severity == "critical" else "normal"
            )
        except ImportError:
            pass  # Notifications not available


class ResourceMonitor:
    """
    Resource monitor for disk and memory usage.
    """

    def __init__(self, disk_threshold_gb: float = 10.0, mem_threshold_gb: float = 1.0):
        self.disk_threshold_gb = disk_threshold_gb
        self.mem_threshold_gb = mem_threshold_gb
        self.last_disk_alert = 0.0
        self.last_mem_alert = 0.0
        self.alert_cooldown = 300  # 5 minutes

        # Prometheus metrics
        self.disk_free_gb = Gauge('ztb_disk_free_gb', 'Free disk space in GB')
        self.mem_free_gb = Gauge('ztb_mem_free_gb', 'Free memory in GB')

    def check_resources(self) -> None:
        """Check disk and memory resources and send alerts if thresholds exceeded"""
        try:
            import psutil
        except ImportError:
            logger.warning("psutil not available, skipping resource monitoring")
            return

        current_time = time.time()

        # Check disk space
        disk = psutil.disk_usage('/')
        free_disk_gb = disk.free / (1024**3)
        self.disk_free_gb.set(free_disk_gb)

        if free_disk_gb < self.disk_threshold_gb:
            if current_time - self.last_disk_alert > self.alert_cooldown:
                self._send_disk_alert(free_disk_gb)
                self.last_disk_alert = current_time

        # Check memory
        mem = psutil.virtual_memory()
        free_mem_gb = mem.available / (1024**3)
        self.mem_free_gb.set(free_mem_gb)

        if free_mem_gb < self.mem_threshold_gb:
            if current_time - self.last_mem_alert > self.alert_cooldown:
                self._send_mem_alert(free_mem_gb)
                self.last_mem_alert = current_time

    def _send_disk_alert(self, free_gb: float) -> None:
        """Send disk space alert"""
        message = f"⚠️ Low disk space: {free_gb:.1f}GB remaining (threshold: {self.disk_threshold_gb}GB)"
        logger.warning(message)

        try:
            from .notifications import send_notification
            send_notification(
                title="Disk Space Alert",
                message=message,
                priority="high"
            )
        except ImportError:
            pass

    def _send_mem_alert(self, free_gb: float) -> None:
        """Send memory alert"""
        message = f"⚠️ Low memory: {free_gb:.1f}GB remaining (threshold: {self.mem_threshold_gb}GB)"
        logger.warning(message)

        try:
            from .notifications import send_notification
            send_notification(
                title="Memory Alert",
                message=message,
                priority="high"
            )
        except ImportError:
            pass


# Global exporter instance
exporter = PrometheusExporter()

def get_exporter() -> PrometheusExporter:
    """Get global Prometheus exporter instance"""
    return exporter

# Global resource monitor instance
resource_monitor = ResourceMonitor()

def get_resource_monitor() -> ResourceMonitor:
    """Get global resource monitor instance"""
    return resource_monitor