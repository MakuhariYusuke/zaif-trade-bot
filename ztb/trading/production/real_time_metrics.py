"""
V433 Phase 5: Production Monitoring Layer - Real-time Metrics

運用環境のリアルタイム指標収集と分析を行う。
"""

import asyncio
import logging
import statistics
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Awaitable, Callable

import psutil

from ztb.io.json_io import write_json
from ztb.trading.environment.constants import BYTES_PER_MB
from ztb.trading.production.state_persistence import (
    read_state_payload,
    write_state_payload,
)

class MetricType(Enum):
    """指標タイプ"""

    GAUGE = "gauge"  # ゲージ（現在の値）
    COUNTER = "counter"  # カウンター（累積値）
    HISTOGRAM = "histogram"  # ヒストグラム（分布）
    SUMMARY = "summary"  # サマリー（統計量）

class MetricSource(Enum):
    """指標ソース"""

    SYSTEM = "system"  # システム指標
    APPLICATION = "application"  # アプリケーションメトリクス
    BUSINESS = "business"  # ビジネス指標
    EXTERNAL = "external"  # 外部サービス

@dataclass
class MetricValue:
    """指標値"""

    name: str
    value: float
    timestamp: datetime
    labels: dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

@dataclass
class MetricSeries:
    """指標時系列"""

    name: str
    metric_type: MetricType
    source: MetricSource
    description: str
    unit: str
    values: list[MetricValue] = field(default_factory=list)
    last_updated: datetime | None = None

@dataclass
class MetricAggregation:
    """指標集計"""

    name: str
    count: int
    sum: float
    avg: float
    min: float
    max: float
    std: float
    p50: float  # 中央値
    p95: float  # 95パーセンタイル
    p99: float  # 99パーセンタイル
    period_start: datetime
    period_end: datetime

class RealTimeMetrics:
    """
    リアルタイム指標収集・分析システム

    システム、アプリケーション、ビジネス指標をリアルタイムで収集し、
    集計・分析を行う。
    """

    def __init__(
        self,
        collection_interval_seconds: int = 10,
        retention_period_hours: int = 24,
        max_series_per_metric: int = 1000,
    ):
        """
        初期化

        Args:
            collection_interval_seconds: 収集間隔（秒）
            retention_period_hours: 保持期間（時間）
            max_series_per_metric: 指標ごとの最大系列数
        """
        self.collection_interval_seconds = collection_interval_seconds
        self.retention_period_hours = retention_period_hours
        self.max_series_per_metric = max_series_per_metric

        # 指標管理
        self.metric_series: dict[str, MetricSeries] = {}
        self.metric_aggregations: dict[str, list[MetricAggregation]] = {}

        # 収集設定
        self.enabled_sources = set([MetricSource.SYSTEM, MetricSource.APPLICATION])
        self.custom_collectors: dict[str, Callable[[], list[MetricValue]]] = {}

        # 収集制御
        self.collection_active = False
        self.collection_thread: threading.Thread | None = None
        self.last_collection = datetime.now()

        # コールバック
        self.metric_callbacks: list[Callable[[MetricValue], Awaitable[None]]] = []
        self.aggregation_callbacks: list[
            Callable[[MetricAggregation], Awaitable[None]]
        ] = []

        # ロギング
        self.logger = logging.getLogger(__name__)

        # デフォルト指標初期化
        self._initialize_default_metrics()

        self.logger.info("Real-time Metrics initialized")

    def _initialize_default_metrics(self) -> None:
        """デフォルト指標初期化"""
        default_metrics = [
            # システム指標
            (
                "cpu_usage_percent",
                MetricType.GAUGE,
                MetricSource.SYSTEM,
                "CPU使用率",
                "%",
            ),
            (
                "memory_usage_percent",
                MetricType.GAUGE,
                MetricSource.SYSTEM,
                "メモリ使用率",
                "%",
            ),
            (
                "disk_usage_percent",
                MetricType.GAUGE,
                MetricSource.SYSTEM,
                "ディスク使用率",
                "%",
            ),
            (
                "network_bytes_sent",
                MetricType.COUNTER,
                MetricSource.SYSTEM,
                "ネットワーク送信バイト",
                "bytes",
            ),
            (
                "network_bytes_recv",
                MetricType.COUNTER,
                MetricSource.SYSTEM,
                "ネットワーク受信バイト",
                "bytes",
            ),
            # アプリケーション指標
            (
                "active_connections",
                MetricType.GAUGE,
                MetricSource.APPLICATION,
                "アクティブ接続数",
                "count",
            ),
            (
                "request_rate",
                MetricType.GAUGE,
                MetricSource.APPLICATION,
                "リクエストレート",
                "req/s",
            ),
            (
                "response_time_avg",
                MetricType.GAUGE,
                MetricSource.APPLICATION,
                "平均レスポンスタイム",
                "ms",
            ),
            (
                "error_rate",
                MetricType.GAUGE,
                MetricSource.APPLICATION,
                "エラーレート",
                "%",
            ),
            # ビジネス指標
            (
                "orders_processed",
                MetricType.COUNTER,
                MetricSource.BUSINESS,
                "処理済み注文数",
                "count",
            ),
            (
                "profit_loss",
                MetricType.GAUGE,
                MetricSource.BUSINESS,
                "損益",
                "currency",
            ),
            ("win_rate", MetricType.GAUGE, MetricSource.BUSINESS, "勝率", "%"),
            (
                "sharpe_ratio",
                MetricType.GAUGE,
                MetricSource.BUSINESS,
                "シャープレシオ",
                "ratio",
            ),
        ]

        for name, mtype, source, desc, unit in default_metrics:
            self.register_metric(name, mtype, source, desc, unit)

    def register_metric(
        self,
        name: str,
        metric_type: MetricType,
        source: MetricSource,
        description: str,
        unit: str,
    ) -> None:
        """
        指標登録

        Args:
            name: 指標名
            metric_type: 指標タイプ
            source: 指標ソース
            description: 説明
            unit: 単位
        """
        if name in self.metric_series:
            self.logger.warning(f"Metric {name} already registered")
            return

        series = MetricSeries(
            name=name,
            metric_type=metric_type,
            source=source,
            description=description,
            unit=unit,
        )

        self.metric_series[name] = series
        self.metric_aggregations[name] = []

        self.logger.info(f"Metric registered: {name} ({metric_type.value})")

    def unregister_metric(self, name: str) -> None:
        """
        指標登録解除

        Args:
            name: 指標名
        """
        if name in self.metric_series:
            del self.metric_series[name]
            del self.metric_aggregations[name]
            self.logger.info(f"Metric unregistered: {name}")

    def record_metric(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """
        指標値記録

        Args:
            name: 指標名
            value: 値
            labels: ラベル
        """
        if name not in self.metric_series:
            self.logger.warning(f"Unknown metric: {name}")
            return

        series = self.metric_series[name]
        labels = labels or {}

        metric_value = MetricValue(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels,
            metric_type=series.metric_type,
        )

        series.values.append(metric_value)
        series.last_updated = datetime.now()

        # 系列サイズ制限
        if len(series.values) > self.max_series_per_metric:
            series.values = series.values[-self.max_series_per_metric :]

        # コールバック実行
        for callback in self.metric_callbacks:
            try:
                asyncio.create_task(callback(metric_value))
            except Exception as e:
                self.logger.error(f"Metric callback error: {e}")

    def add_custom_collector(
        self, collector_name: str, collector_func: Callable[[], list[MetricValue]]
    ) -> None:
        """
        カスタムコレクター追加

        Args:
            collector_name: コレクター名
            collector_func: コレクター関数
        """
        self.custom_collectors[collector_name] = collector_func
        self.logger.info(f"Custom collector added: {collector_name}")

    def remove_custom_collector(self, collector_name: str) -> None:
        """
        カスタムコレクター削除

        Args:
            collector_name: コレクター名
        """
        if collector_name in self.custom_collectors:
            del self.custom_collectors[collector_name]
            self.logger.info(f"Custom collector removed: {collector_name}")

    def collect_system_metrics(self) -> list[MetricValue]:
        """
        システム指標収集

        Returns:
            list[MetricValue]: 収集された指標値
        """
        metrics = []

        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(
                MetricValue("cpu_usage_percent", cpu_percent, datetime.now())
            )

            # メモリ使用率
            memory = psutil.virtual_memory()
            metrics.append(
                MetricValue("memory_usage_percent", memory.percent, datetime.now())
            )

            # ディスク使用率
            disk = psutil.disk_usage("/")
            metrics.append(
                MetricValue("disk_usage_percent", disk.percent, datetime.now())
            )

            # ネットワークI/O
            net = psutil.net_io_counters()
            metrics.append(
                MetricValue("network_bytes_sent", net.bytes_sent, datetime.now())
            )
            metrics.append(
                MetricValue("network_bytes_recv", net.bytes_recv, datetime.now())
            )

        except Exception as e:
            self.logger.error(f"System metrics collection error: {e}")

        return metrics

    def collect_application_metrics(self) -> list[MetricValue]:
        """
        アプリケーションメトリクス収集

        Returns:
            list[MetricValue]: 収集された指標値
        """
        metrics = []

        try:
            # プロセス情報
            process = psutil.Process()
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()

            metrics.append(
                MetricValue("process_cpu_percent", cpu_percent, datetime.now())
            )
            metrics.append(
                MetricValue(
                    "process_memory_mb", memory_info.rss / BYTES_PER_MB, datetime.now()
                )
            )

            # スレッド数
            thread_count = process.num_threads()
            metrics.append(MetricValue("thread_count", thread_count, datetime.now()))

        except Exception as e:
            self.logger.error(f"Application metrics collection error: {e}")

        return metrics

    def collect_business_metrics(self) -> list[MetricValue]:
        """
        ビジネス指標収集

        Returns:
            list[MetricValue]: 収集された指標値
        """
        # 実際の実装ではビジネスロジックから指標を取得
        # ここではシミュレーション
        metrics = []

        try:
            # シミュレーション指標
            import random

            orders_processed = random.randint(10, 100)
            profit_loss = random.uniform(-1000, 1000)
            win_rate = random.uniform(45, 55)
            sharpe_ratio = random.uniform(0.1, 0.5)

            metrics.append(
                MetricValue("orders_processed", orders_processed, datetime.now())
            )
            metrics.append(MetricValue("profit_loss", profit_loss, datetime.now()))
            metrics.append(MetricValue("win_rate", win_rate, datetime.now()))
            metrics.append(MetricValue("sharpe_ratio", sharpe_ratio, datetime.now()))

        except Exception as e:
            self.logger.error(f"Business metrics collection error: {e}")

        return metrics

    def collect_all_metrics(self) -> None:
        """全指標収集"""
        all_metrics = []

        # システム指標
        if MetricSource.SYSTEM in self.enabled_sources:
            all_metrics.extend(self.collect_system_metrics())

        # アプリケーションメトリクス
        if MetricSource.APPLICATION in self.enabled_sources:
            all_metrics.extend(self.collect_application_metrics())

        # ビジネス指標
        if MetricSource.BUSINESS in self.enabled_sources:
            all_metrics.extend(self.collect_business_metrics())

        # カスタムコレクター
        for collector_func in self.custom_collectors.values():
            try:
                custom_metrics = collector_func()
                all_metrics.extend(custom_metrics)
            except Exception as e:
                self.logger.error(f"Custom collector error: {e}")

        # 指標記録
        for metric in all_metrics:
            self.record_metric(metric.name, metric.value, metric.labels)

        self.last_collection = datetime.now()

    def get_metric_values(
        self,
        name: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
    ) -> list[MetricValue]:
        """
        指標値取得

        Args:
            name: 指標名
            start_time: 開始時間
            end_time: 終了時間
            limit: 取得件数制限

        Returns:
            list[MetricValue]: 指標値リスト
        """
        if name not in self.metric_series:
            return []

        values = self.metric_series[name].values

        # 時間フィルタ
        if start_time:
            values = [v for v in values if v.timestamp >= start_time]
        if end_time:
            values = [v for v in values if v.timestamp <= end_time]

        # ソート（最新順）
        values.sort(key=lambda v: v.timestamp, reverse=True)

        if limit:
            values = values[:limit]

        return values

    def get_metric_aggregation(
        self, name: str, period_minutes: int = 60
    ) -> MetricAggregation | None:
        """
        指標集計取得

        Args:
            name: 指標名
            period_minutes: 集計期間（分）

        Returns:
            MetricAggregation | None: 指標集計
        """
        if name not in self.metric_series:
            return None

        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=period_minutes)

        values = self.get_metric_values(name, start_time, end_time)
        if not values:
            return None

        numeric_values = [v.value for v in values]

        try:
            aggregation = MetricAggregation(
                name=name,
                count=len(numeric_values),
                sum=sum(numeric_values),
                avg=statistics.mean(numeric_values),
                min=min(numeric_values),
                max=max(numeric_values),
                std=statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0,
                p50=statistics.median(numeric_values),
                p95=self._percentile(numeric_values, 95),
                p99=self._percentile(numeric_values, 99),
                period_start=start_time,
                period_end=end_time,
            )

            # 集計履歴保存
            self.metric_aggregations[name].append(aggregation)

            # 履歴サイズ制限（最新100件）
            if len(self.metric_aggregations[name]) > 100:
                self.metric_aggregations[name] = self.metric_aggregations[name][-100:]

            # コールバック実行
            for callback in self.aggregation_callbacks:
                try:
                    asyncio.create_task(callback(aggregation))
                except Exception as e:
                    self.logger.error(f"Aggregation callback error: {e}")

            return aggregation

        except Exception as e:
            self.logger.error(f"Metric aggregation error for {name}: {e}")
            return None

    def _percentile(self, values: list[float], percentile: float) -> float:
        """
        パーセンタイル計算

        Args:
            values: 値リスト
            percentile: パーセンタイル

        Returns:
            float: パーセンタイル値
        """
        if not values:
            return 0.0

        values_sorted = sorted(values)
        k = (len(values_sorted) - 1) * (percentile / 100.0)
        f = int(k)
        c = k - f

        if f + 1 < len(values_sorted):
            return values_sorted[f] + c * (values_sorted[f + 1] - values_sorted[f])
        else:
            return values_sorted[f]

    def get_all_metrics_summary(self) -> dict[str, Any]:
        """
        全指標要約取得

        Returns:
            dict[str, Any]: 指標要約
        """
        summary = {
            "total_metrics": len(self.metric_series),
            "last_collection": self.last_collection.isoformat(),
            "collection_active": self.collection_active,
            "metrics": {},
        }

        for name, series in self.metric_series.items():
            latest_value = series.values[-1] if series.values else None

            summary["metrics"][name] = {
                "type": series.metric_type.value,
                "source": series.source.value,
                "description": series.description,
                "unit": series.unit,
                "last_updated": series.last_updated.isoformat()
                if series.last_updated
                else None,
                "total_values": len(series.values),
                "latest_value": latest_value.value if latest_value else None,
                "latest_timestamp": latest_value.timestamp.isoformat()
                if latest_value
                else None,
            }

        return summary

    def export_metrics(self, filepath: str, format: str = "json") -> None:
        """
        指標エクスポート

        Args:
            filepath: エクスポートファイルパス
            format: エクスポート形式
        """
        if format == "json":
            data = {"export_time": datetime.now().isoformat(), "metrics": {}}

            for name, series in self.metric_series.items():
                data["metrics"][name] = {
                    "type": series.metric_type.value,
                    "source": series.source.value,
                    "description": series.description,
                    "unit": series.unit,
                    "values": [
                        {
                            "value": v.value,
                            "timestamp": v.timestamp.isoformat(),
                            "labels": v.labels,
                        }
                        for v in series.values[-1000:]  # 最新1000件
                    ],
                }

            write_json(filepath, data, indent=2, ensure_ascii=False)

        self.logger.info(f"Metrics exported to {filepath}")

    def start_collection(self) -> None:
        """指標収集開始"""
        if self.collection_active:
            return

        self.collection_active = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop, daemon=True
        )
        self.collection_thread.start()

        self.logger.info("Metrics collection started")

    def stop_collection(self) -> None:
        """指標収集停止"""
        self.collection_active = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)

        self.logger.info("Metrics collection stopped")

    def _collection_loop(self) -> None:
        """収集ループ"""
        while self.collection_active:
            try:
                start_time = time.time()
                self.collect_all_metrics()
                elapsed = time.time() - start_time

                # 収集間隔調整
                sleep_time = max(0, self.collection_interval_seconds - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                self.logger.error(f"Collection loop error: {e}")
                time.sleep(10)

    def cleanup_old_data(self) -> None:
        """古いデータクリーンアップ"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_period_hours)

        for series in self.metric_series.values():
            series.values = [v for v in series.values if v.timestamp >= cutoff_time]

        for name in self.metric_aggregations:
            self.metric_aggregations[name] = [
                agg
                for agg in self.metric_aggregations[name]
                if agg.period_end >= cutoff_time
            ]

        self.logger.info("Old metrics data cleaned up")

    def add_metric_callback(
        self, callback: Callable[[MetricValue], Awaitable[None]]
    ) -> None:
        """
        指標コールバック追加

        Args:
            callback: コールバック関数
        """
        self.metric_callbacks.append(callback)

    def add_aggregation_callback(
        self, callback: Callable[[MetricAggregation], Awaitable[None]]
    ) -> None:
        """
        集計コールバック追加

        Args:
            callback: コールバック関数
        """
        self.aggregation_callbacks.append(callback)

    def save_state(self, filepath: str) -> None:
        """
        状態保存

        Args:
            filepath: 保存ファイルパス
        """
        state = {
            "collection_interval_seconds": self.collection_interval_seconds,
            "retention_period_hours": self.retention_period_hours,
            "max_series_per_metric": self.max_series_per_metric,
            "enabled_sources": [s.value for s in self.enabled_sources],
            "last_collection": self.last_collection.isoformat(),
            "metric_series": {
                name: {
                    "name": series.name,
                    "metric_type": series.metric_type.value,
                    "source": series.source.value,
                    "description": series.description,
                    "unit": series.unit,
                    "last_updated": series.last_updated.isoformat()
                    if series.last_updated
                    else None,
                    "values": [
                        {
                            "name": v.name,
                            "value": v.value,
                            "timestamp": v.timestamp.isoformat(),
                            "labels": v.labels,
                            "metric_type": v.metric_type.value,
                        }
                        for v in series.values[-500:]  # 最新500件
                    ],
                }
                for name, series in self.metric_series.items()
            },
        }

        write_state_payload(filepath, state)

        self.logger.info(f"Metrics state saved to {filepath}")

    def load_state(self, filepath: str) -> bool:
        """
        状態読み込み

        Args:
            filepath: 読み込みファイルパス

        Returns:
            bool: 読み込み成功フラグ
        """
        try:
            state = read_state_payload(filepath)

            self.collection_interval_seconds = state["collection_interval_seconds"]
            self.retention_period_hours = state["retention_period_hours"]
            self.max_series_per_metric = state["max_series_per_metric"]
            self.enabled_sources = set(
                MetricSource(s) for s in state["enabled_sources"]
            )
            self.last_collection = datetime.fromisoformat(state["last_collection"])

            # 指標系列復元
            self.metric_series = {}
            self.metric_aggregations = {}

            for name, series_data in state.get("metric_series", {}).items():
                series = MetricSeries(
                    name=series_data["name"],
                    metric_type=MetricType(series_data["metric_type"]),
                    source=MetricSource(series_data["source"]),
                    description=series_data["description"],
                    unit=series_data["unit"],
                    last_updated=datetime.fromisoformat(series_data["last_updated"])
                    if series_data["last_updated"]
                    else None,
                )

                # 値復元
                for v_data in series_data.get("values", []):
                    value = MetricValue(
                        name=v_data["name"],
                        value=v_data["value"],
                        timestamp=datetime.fromisoformat(v_data["timestamp"]),
                        labels=v_data["labels"],
                        metric_type=MetricType(v_data["metric_type"]),
                    )
                    series.values.append(value)

                self.metric_series[name] = series
                self.metric_aggregations[name] = []

            self.logger.info(f"Metrics state loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load metrics state: {e}")
            return False
