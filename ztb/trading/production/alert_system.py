"""
V433 Phase 5: Production Monitoring Layer - Alert System

リアルタイム指標に基づいてアラートを発行し、運用チームに通知する。
"""

import asyncio
import json
import logging
import smtplib
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

import requests

from ztb.trading.production.state_persistence import (
    read_state_payload,
    write_state_payload,
)
from ztb.types.alert_types import AlertLevel, AlertStatus


class NotificationChannel(Enum):
    """通知チャネル"""

    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    LOG = "log"


@dataclass
class AlertRule:
    """アラートルール"""

    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # '>', '<', '>=', '<=', '==', '!='
    threshold: float
    level: AlertLevel
    enabled: bool = True
    cooldown_minutes: int = 5
    auto_resolve: bool = True
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class ProductionAlert:
    """アラート"""

    alert_id: str
    rule_id: str
    level: AlertLevel
    status: AlertStatus
    title: str
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    notification_sent: bool = False


@dataclass
class NotificationConfig:
    """通知設定"""

    channel: NotificationChannel
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)


class AlertSystem:
    """
    アラートシステム

    リアルタイム指標を監視し、条件に基づいてアラートを発行。
    複数の通知チャネルをサポート。
    """

    def __init__(
        self, alert_history_size: int = 1000, default_cooldown_minutes: int = 5
    ):
        """
        初期化

        Args:
            alert_history_size: アラート履歴サイズ
            default_cooldown_minutes: デフォルトクールダウン時間（分）
        """
        self.alert_history_size = alert_history_size
        self.default_cooldown_minutes = default_cooldown_minutes

        # アラートルール管理
        self.alert_rules: Dict[str, AlertRule] = {}

        # アラート管理
        self.active_alerts: Dict[str, ProductionAlert] = {}
        self.alert_history: List[ProductionAlert] = []

        # アラート抑制
        self.alert_cooldowns: Dict[str, datetime] = {}

        # 通知設定
        self.notification_configs: Dict[NotificationChannel, NotificationConfig] = {}
        self._initialize_default_notifications()

        # モニタリング
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None

        # コールバック
        self.alert_callbacks: List[Callable[[ProductionAlert], Awaitable[None]]] = []

        # ロギング
        self.logger = logging.getLogger(__name__)

        # デフォルトルール初期化
        self._initialize_default_rules()

        self.logger.info("Alert System initialized")

    def _initialize_default_rules(self) -> None:
        """デフォルトアラートルール初期化"""
        default_rules = [
            AlertRule(
                "cpu_high",
                "High CPU Usage",
                "CPU使用率が高くなっています",
                "cpu_usage_percent",
                ">",
                85.0,
                AlertLevel.WARNING,
            ),
            AlertRule(
                "cpu_critical",
                "Critical CPU Usage",
                "CPU使用率が危険なレベルです",
                "cpu_usage_percent",
                ">",
                95.0,
                AlertLevel.CRITICAL,
            ),
            AlertRule(
                "memory_high",
                "High Memory Usage",
                "メモリ使用率が高くなっています",
                "memory_usage_percent",
                ">",
                90.0,
                AlertLevel.WARNING,
            ),
            AlertRule(
                "memory_critical",
                "Critical Memory Usage",
                "メモリ使用率が危険なレベルです",
                "memory_usage_percent",
                ">",
                95.0,
                AlertLevel.CRITICAL,
            ),
            AlertRule(
                "disk_high",
                "High Disk Usage",
                "ディスク使用率が高くなっています",
                "disk_usage_percent",
                ">",
                85.0,
                AlertLevel.WARNING,
            ),
            AlertRule(
                "error_rate_high",
                "High Error Rate",
                "エラーレートが高くなっています",
                "error_rate",
                ">",
                5.0,
                AlertLevel.CRITICAL,
            ),
            AlertRule(
                "response_time_high",
                "High Response Time",
                "レスポンスタイムが高くなっています",
                "response_time_avg",
                ">",
                5000.0,
                AlertLevel.WARNING,
            ),
            AlertRule(
                "win_rate_low",
                "Low Win Rate",
                "勝率が低下しています",
                "win_rate",
                "<",
                40.0,
                AlertLevel.WARNING,
            ),
            AlertRule(
                "sharpe_ratio_low",
                "Low Sharpe Ratio",
                "シャープレシオが低下しています",
                "sharpe_ratio",
                "<",
                0.1,
                AlertLevel.CRITICAL,
            ),
        ]

        for rule in default_rules:
            self.add_alert_rule(rule)

    def _initialize_default_notifications(self) -> None:
        """デフォルト通知設定初期化"""
        # ログ通知（常に有効）
        self.notification_configs[NotificationChannel.LOG] = NotificationConfig(
            channel=NotificationChannel.LOG, enabled=True
        )

    def add_alert_rule(self, rule: AlertRule) -> None:
        """
        アラートルール追加

        Args:
            rule: アラートルール
        """
        self.alert_rules[rule.rule_id] = rule
        self.logger.info(f"Alert rule added: {rule.rule_id} - {rule.name}")

    def remove_alert_rule(self, rule_id: str) -> None:
        """
        アラートルール削除

        Args:
            rule_id: アラートルールID
        """
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            self.logger.info(f"Alert rule removed: {rule_id}")

    def update_alert_rule(self, rule: AlertRule) -> None:
        """
        アラートルール更新

        Args:
            rule: アラートルール
        """
        if rule.rule_id in self.alert_rules:
            self.alert_rules[rule.rule_id] = rule
            self.logger.info(f"Alert rule updated: {rule.rule_id}")

    def enable_alert_rule(self, rule_id: str) -> None:
        """
        アラートルール有効化

        Args:
            rule_id: アラートルールID
        """
        if rule_id in self.alert_rules:
            self.alert_rules[rule_id].enabled = True
            self.logger.info(f"Alert rule enabled: {rule_id}")

    def disable_alert_rule(self, rule_id: str) -> None:
        """
        アラートルール無効化

        Args:
            rule_id: アラートルールID
        """
        if rule_id in self.alert_rules:
            self.alert_rules[rule_id].enabled = False
            self.logger.info(f"Alert rule disabled: {rule_id}")

    def check_metric_against_rules(
        self,
        metric_name: str,
        metric_value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> List[ProductionAlert]:
        """
        指標値に対するルールチェック

        Args:
            metric_name: 指標名
            metric_value: 指標値
            labels: ラベル

        Returns:
            List[ProductionAlert]: 生成されたアラートリスト
        """
        alerts = []
        labels = labels or {}

        for rule in self.alert_rules.values():
            if not rule.enabled or rule.metric_name != metric_name:
                continue

            # クールダウンチェック
            cooldown_key = f"{rule.rule_id}:{metric_name}"
            if cooldown_key in self.alert_cooldowns:
                if datetime.now() < self.alert_cooldowns[cooldown_key]:
                    continue

            # 条件評価
            if self._evaluate_condition(metric_value, rule.condition, rule.threshold):
                alert = self._create_alert(rule, metric_value, labels)
                alerts.append(alert)

                # クールダウン設定
                self.alert_cooldowns[cooldown_key] = datetime.now() + timedelta(
                    minutes=rule.cooldown_minutes
                )

        return alerts

    def _evaluate_condition(
        self, value: float, condition: str, threshold: float
    ) -> bool:
        """
        条件評価

        Args:
            value: 値
            condition: 条件
            threshold: 閾値

        Returns:
            bool: 条件成立フラグ
        """
        if condition == ">":
            return value > threshold
        elif condition == "<":
            return value < threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return value == threshold
        elif condition == "!=":
            return value != threshold
        else:
            return False

    def _create_alert(
        self, rule: AlertRule, metric_value: float, labels: Dict[str, str]
    ) -> ProductionAlert:
        """
        アラート作成

        Args:
            rule: アラートルール
            metric_value: 指標値
            labels: ラベル

        Returns:
            ProductionAlert: アラート
        """
        alert_id = f"ALERT_{rule.rule_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        title = f"[{rule.level.value.upper()}] {rule.name}"
        message = self._generate_alert_message(rule, metric_value)

        alert = ProductionAlert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            level=rule.level,
            status=AlertStatus.ACTIVE,
            title=title,
            message=message,
            metric_name=rule.metric_name,
            metric_value=metric_value,
            threshold=rule.threshold,
            timestamp=datetime.now(),
            labels={**rule.labels, **labels},
        )

        return alert

    def _generate_alert_message(self, rule: AlertRule, metric_value: float) -> str:
        """
        アラートメッセージ生成

        Args:
            rule: アラートルール
            metric_value: 指標値

        Returns:
            str: アラートメッセージ
        """
        condition_desc = {
            ">": "above",
            "<": "below",
            ">=": "at or above",
            "<=": "at or below",
            "==": "equal to",
            "!=": "not equal to",
        }.get(rule.condition, rule.condition)

        return (
            f"{rule.description}\n"
            f"Metric: {rule.metric_name}\n"
            f"Current Value: {metric_value:.2f}\n"
            f"Threshold: {rule.threshold:.2f} ({condition_desc})"
        )

    def process_alerts(self, alerts: List[ProductionAlert]) -> None:
        """
        アラート処理

        Args:
            alerts: アラートリスト
        """
        for alert in alerts:
            # アラート履歴保存
            self.alert_history.append(alert)

            # 履歴サイズ制限
            if len(self.alert_history) > self.alert_history_size:
                self.alert_history = self.alert_history[-self.alert_history_size :]

            # アクティブアラート管理
            alert_key = f"{alert.rule_id}:{alert.metric_name}"
            self.active_alerts[alert_key] = alert

            # 通知送信
            asyncio.create_task(self._send_alert_notifications(alert))

            # コールバック実行
            for callback in self.alert_callbacks:
                try:
                    asyncio.create_task(callback(alert))  # type: ignore
                except Exception as e:
                    self.logger.error(f"Alert callback error: {e}")

            self.logger.warning(f"Alert generated: {alert.title}")

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        アラート確認

        Args:
            alert_id: アラートID
            acknowledged_by: 確認者

        Returns:
            bool: 確認成功フラグ
        """
        for alert in self.alert_history:
            if alert.alert_id == alert_id and alert.status == AlertStatus.ACTIVE:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now()
                alert.acknowledged_by = acknowledged_by

                # アクティブアラートから削除
                alert_key = f"{alert.rule_id}:{alert.metric_name}"
                if alert_key in self.active_alerts:
                    del self.active_alerts[alert_key]

                self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                return True

        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """
        アラート解決

        Args:
            alert_id: アラートID

        Returns:
            bool: 解決成功フラグ
        """
        for alert in self.alert_history:
            if alert.alert_id == alert_id and alert.status in [
                AlertStatus.ACTIVE,
                AlertStatus.ACKNOWLEDGED,
            ]:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()

                # アクティブアラートから削除
                alert_key = f"{alert.rule_id}:{alert.metric_name}"
                if alert_key in self.active_alerts:
                    del self.active_alerts[alert_key]

                self.logger.info(f"Alert resolved: {alert_id}")
                return True

        return False

    async def _send_alert_notifications(self, alert: ProductionAlert) -> None:
        """
        アラート通知送信

        Args:
            alert: アラート
        """
        if alert.notification_sent:
            return

        for channel, config in self.notification_configs.items():
            if not config.enabled:
                continue

            try:
                if channel == NotificationChannel.LOG:
                    await self._send_log_notification(alert)
                elif channel == NotificationChannel.EMAIL:
                    await self._send_email_notification(alert, config)
                elif channel == NotificationChannel.SLACK:
                    await self._send_slack_notification(alert, config)
                elif channel == NotificationChannel.WEBHOOK:
                    await self._send_webhook_notification(alert, config)
                elif channel == NotificationChannel.SMS:
                    await self._send_sms_notification(alert, config)

            except Exception as e:
                self.logger.error(f"Failed to send {channel.value} notification: {e}")

        alert.notification_sent = True

    async def _send_log_notification(self, alert: ProductionAlert) -> None:
        """
        ログ通知送信

        Args:
            alert: アラート
        """
        log_message = (
            f"ALERT {alert.level.value.upper()}: {alert.title} - {alert.message}"
        )
        self.logger.warning(log_message)

    async def _send_email_notification(
        self, alert: ProductionAlert, config: NotificationConfig
    ) -> None:
        """
        メール通知送信

        Args:
            alert: アラート
            config: 通知設定
        """
        smtp_server = config.config.get("smtp_server")
        smtp_port = config.config.get("smtp_port", 587)
        username = config.config.get("username")
        password = config.config.get("password")
        from_addr = config.config.get("from_addr")
        to_addrs = config.config.get("to_addrs", [])

        if not all([smtp_server, username, password, from_addr, to_addrs]):
            self.logger.warning("Email notification not configured properly")
            return

        assert smtp_server is not None
        assert username is not None
        assert password is not None
        assert from_addr is not None

        msg = MIMEMultipart()
        msg["From"] = from_addr
        msg["To"] = ", ".join(to_addrs)
        msg["Subject"] = alert.title

        body = f"""
{alert.message}

Timestamp: {alert.timestamp.isoformat()}
Alert ID: {alert.alert_id}
Rule ID: {alert.rule_id}
Status: {alert.status.value}

Labels: {json.dumps(alert.labels, indent=2)}
        """

        msg.attach(MIMEText(body, "plain"))

        try:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            text = msg.as_string()
            server.sendmail(from_addr, to_addrs, text)
            server.quit()

            self.logger.info(f"Email notification sent for alert {alert.alert_id}")

        except Exception as e:
            self.logger.error(f"Email notification failed: {e}")

    async def _send_slack_notification(
        self, alert: ProductionAlert, config: NotificationConfig
    ) -> None:
        """
        Slack通知送信

        Args:
            alert: アラート
            config: 通知設定
        """
        webhook_url = config.config.get("webhook_url")
        if not webhook_url:
            self.logger.warning("Slack webhook URL not configured")
            return

        color_map = {
            AlertLevel.INFO: "good",
            AlertLevel.WARNING: "warning",
            AlertLevel.CRITICAL: "#ff0000",
        }

        payload = {
            "attachments": [
                {
                    "color": color_map.get(alert.level, "warning"),
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {"title": "Alert ID", "value": alert.alert_id, "short": True},
                        {
                            "title": "Timestamp",
                            "value": alert.timestamp.isoformat(),
                            "short": True,
                        },
                        {"title": "Status", "value": alert.status.value, "short": True},
                    ],
                }
            ]
        }

        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            self.logger.info(f"Slack notification sent for alert {alert.alert_id}")

        except Exception as e:
            self.logger.error(f"Slack notification failed: {e}")

    async def _send_webhook_notification(
        self, alert: ProductionAlert, config: NotificationConfig
    ) -> None:
        """
        Webhook通知送信

        Args:
            alert: アラート
            config: 通知設定
        """
        webhook_url = config.config.get("webhook_url")
        if not webhook_url:
            self.logger.warning("Webhook URL not configured")
            return

        payload = {
            "alert_id": alert.alert_id,
            "rule_id": alert.rule_id,
            "level": alert.level.value,
            "status": alert.status.value,
            "title": alert.title,
            "message": alert.message,
            "metric_name": alert.metric_name,
            "metric_value": alert.metric_value,
            "threshold": alert.threshold,
            "timestamp": alert.timestamp.isoformat(),
            "labels": alert.labels,
        }

        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            self.logger.info(f"Webhook notification sent for alert {alert.alert_id}")

        except Exception as e:
            self.logger.error(f"Webhook notification failed: {e}")

    async def _send_sms_notification(
        self, alert: ProductionAlert, config: NotificationConfig
    ) -> None:
        """
        SMS通知送信

        Args:
            alert: アラート
            config: 通知設定
        """
        # SMS通知の実装（実際のSMSサービスと連携）
        self.logger.info(
            f"SMS notification for alert {alert.alert_id} (not implemented)"
        )

    def configure_notification(
        self,
        channel: NotificationChannel,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        通知設定

        Args:
            channel: 通知チャネル
            enabled: 有効フラグ
            config: 設定
        """
        self.notification_configs[channel] = NotificationConfig(
            channel=channel, enabled=enabled, config=config or {}
        )

        self.logger.info(
            f"Notification configured: {channel.value} ({'enabled' if enabled else 'disabled'})"
        )

    def get_active_alerts(
        self, level: Optional[AlertLevel] = None
    ) -> List[ProductionAlert]:
        """
        アクティブアラート取得

        Args:
            level: アラートレベルフィルタ

        Returns:
            List[Alert]: アクティブアラート
        """
        alerts = list(self.active_alerts.values())

        if level:
            alerts = [a for a in alerts if a.level == level]

        return alerts

    def get_alert_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[AlertLevel] = None,
        status: Optional[AlertStatus] = None,
        limit: Optional[int] = None,
    ) -> List[ProductionAlert]:
        """
        アラート履歴取得

        Args:
            start_time: 開始時間
            end_time: 終了時間
            level: アラートレベル
            status: アラートステータス
            limit: 取得件数制限

        Returns:
            List[Alert]: アラート履歴
        """
        alerts = self.alert_history

        if start_time:
            alerts = [a for a in alerts if a.timestamp >= start_time]
        if end_time:
            alerts = [a for a in alerts if a.timestamp <= end_time]
        if level:
            alerts = [a for a in alerts if a.level == level]
        if status:
            alerts = [a for a in alerts if a.status == status]

        # 最新順ソート
        alerts.sort(key=lambda a: a.timestamp, reverse=True)

        if limit:
            alerts = alerts[:limit]

        return alerts

    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        アラート要約取得

        Args:
            hours: 集計期間（時間）

        Returns:
            Dict[str, Any]: アラート要約
        """
        period_start = datetime.now() - timedelta(hours=hours)
        recent_alerts = [a for a in self.alert_history if a.timestamp >= period_start]

        summary = {
            "period_hours": hours,
            "total_alerts": len(recent_alerts),
            "active_alerts": len(self.active_alerts),
            "by_level": {
                "info": len([a for a in recent_alerts if a.level == AlertLevel.INFO]),
                "warning": len(
                    [a for a in recent_alerts if a.level == AlertLevel.WARNING]
                ),
                "critical": len(
                    [a for a in recent_alerts if a.level == AlertLevel.CRITICAL]
                ),
            },
            "by_status": {
                "active": len(
                    [a for a in recent_alerts if a.status == AlertStatus.ACTIVE]
                ),
                "acknowledged": len(
                    [a for a in recent_alerts if a.status == AlertStatus.ACKNOWLEDGED]
                ),
                "resolved": len(
                    [a for a in recent_alerts if a.status == AlertStatus.RESOLVED]
                ),
                "expired": len(
                    [a for a in recent_alerts if a.status == AlertStatus.EXPIRED]
                ),
            },
        }

        return summary

    def start_monitoring(self) -> None:
        """モニタリング開始"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

        self.logger.info("Alert monitoring started")

    def stop_monitoring(self) -> None:
        """モニタリング停止"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

        self.logger.info("Alert monitoring stopped")

    def _monitoring_loop(self) -> None:
        """モニタリングループ"""
        while self.monitoring_active:
            try:
                # 自動解決チェック
                self._check_auto_resolve_alerts()

                # 期限切れアラートチェック
                self._check_expired_alerts()

                time.sleep(60)  # 1分間隔

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)

    def _check_auto_resolve_alerts(self) -> None:
        """自動解決アラートチェック"""
        for alert_key, alert in list(self.active_alerts.items()):
            rule = self.alert_rules.get(alert.rule_id)
            if rule and rule.auto_resolve:
                # 自動解決ロジック（実際の実装では指標値再チェック）
                # ここでは簡易的に一定時間後に解決
                if datetime.now() - alert.timestamp > timedelta(minutes=10):
                    self.resolve_alert(alert.alert_id)

    def _check_expired_alerts(self) -> None:
        """期限切れアラートチェック"""
        # 24時間以上アクティブのアラートを期限切れとする
        cutoff_time = datetime.now() - timedelta(hours=24)

        for alert_key, alert in list(self.active_alerts.items()):
            if alert.timestamp < cutoff_time:
                alert.status = AlertStatus.EXPIRED
                del self.active_alerts[alert_key]
                self.logger.warning(f"Alert expired: {alert.alert_id}")

    def add_alert_callback(
        self, callback: Callable[[ProductionAlert], Awaitable[None]]
    ) -> None:
        """
        アラートコールバック追加

        Args:
            callback: コールバック関数
        """
        self.alert_callbacks.append(callback)

    def save_state(self, filepath: str) -> None:
        """
        状態保存

        Args:
            filepath: 保存ファイルパス
        """
        state = {
            "alert_history_size": self.alert_history_size,
            "default_cooldown_minutes": self.default_cooldown_minutes,
            "alert_rules": {
                rule_id: {
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "description": rule.description,
                    "metric_name": rule.metric_name,
                    "condition": rule.condition,
                    "threshold": rule.threshold,
                    "level": rule.level.value,
                    "enabled": rule.enabled,
                    "cooldown_minutes": rule.cooldown_minutes,
                    "auto_resolve": rule.auto_resolve,
                    "labels": rule.labels,
                }
                for rule_id, rule in self.alert_rules.items()
            },
            "alert_history": [
                {
                    "alert_id": a.alert_id,
                    "rule_id": a.rule_id,
                    "level": a.level.value,
                    "status": a.status.value,
                    "title": a.title,
                    "message": a.message,
                    "metric_name": a.metric_name,
                    "metric_value": a.metric_value,
                    "threshold": a.threshold,
                    "timestamp": a.timestamp.isoformat(),
                    "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None,
                    "acknowledged_at": a.acknowledged_at.isoformat()
                    if a.acknowledged_at
                    else None,
                    "acknowledged_by": a.acknowledged_by,
                    "labels": a.labels,
                    "notification_sent": a.notification_sent,
                }
                for a in self.alert_history[-500:]  # 最新500件
            ],
            "notification_configs": {
                channel.value: {
                    "channel": config.channel.value,
                    "enabled": config.enabled,
                    "config": config.config,
                }
                for channel, config in self.notification_configs.items()
            },
        }

        write_state_payload(filepath, state)

        self.logger.info(f"Alert system state saved to {filepath}")

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

            self.alert_history_size = state["alert_history_size"]
            self.default_cooldown_minutes = state["default_cooldown_minutes"]

            # アラートルール復元
            self.alert_rules = {}
            for rule_id, rule_data in state.get("alert_rules", {}).items():
                rule = AlertRule(
                    rule_id=rule_data["rule_id"],
                    name=rule_data["name"],
                    description=rule_data["description"],
                    metric_name=rule_data["metric_name"],
                    condition=rule_data["condition"],
                    threshold=rule_data["threshold"],
                    level=AlertLevel(rule_data["level"]),
                    enabled=rule_data["enabled"],
                    cooldown_minutes=rule_data["cooldown_minutes"],
                    auto_resolve=rule_data["auto_resolve"],
                    labels=rule_data["labels"],
                )
                self.alert_rules[rule_id] = rule

            # アラート履歴復元
            self.alert_history = []
            for a_data in state.get("alert_history", []):
                alert = ProductionAlert(
                    alert_id=a_data["alert_id"],
                    rule_id=a_data["rule_id"],
                    level=AlertLevel(a_data["level"]),
                    status=AlertStatus(a_data["status"]),
                    title=a_data["title"],
                    message=a_data["message"],
                    metric_name=a_data["metric_name"],
                    metric_value=a_data["metric_value"],
                    threshold=a_data["threshold"],
                    timestamp=datetime.fromisoformat(a_data["timestamp"]),
                    resolved_at=datetime.fromisoformat(a_data["resolved_at"])
                    if a_data["resolved_at"]
                    else None,
                    acknowledged_at=datetime.fromisoformat(a_data["acknowledged_at"])
                    if a_data["acknowledged_at"]
                    else None,
                    acknowledged_by=a_data["acknowledged_by"],
                    labels=a_data["labels"],
                    notification_sent=a_data["notification_sent"],
                )
                self.alert_history.append(alert)

            # アクティブアラート再構築
            self.active_alerts = {}
            for alert in self.alert_history:
                if alert.status == AlertStatus.ACTIVE:
                    alert_key = f"{alert.rule_id}:{alert.metric_name}"
                    self.active_alerts[alert_key] = alert

            # 通知設定復元
            self.notification_configs = {}
            for channel_str, config_data in state.get(
                "notification_configs", {}
            ).items():
                channel = NotificationChannel(channel_str)
                config = NotificationConfig(
                    channel=channel,
                    enabled=config_data["enabled"],
                    config=config_data["config"],
                )
                self.notification_configs[channel] = config

            self.logger.info(f"Alert system state loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load alert system state: {e}")
            return False
