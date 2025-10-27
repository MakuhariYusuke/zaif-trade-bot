"""
LoggerManager: Unified logging for experiments.

Provides simultaneous output to console, logfile, and Discord.
Commonizes log_experiment_start, log_experiment_end, and log_error.

Usage:
    from ztb.utils.logger import LoggerManager

    logger = LoggerManager()
    logger.log_experiment_start("experiment_name", config)
    # ... experiment ...
    logger.log_experiment_end(results)
"""

import json
import logging
import os
import queue
import threading
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

from ztb.types.common import ConfigDict


class AsyncNotifier:
    """非同期Discord通知クラス（キュー + 集約送信）"""

    def __init__(
        self,
        logger_manager: "LoggerManager",
        flush_sec: int = 300,
        experiment_type: str = "standard",
    ):
        self.logger_manager = logger_manager
        self.q: queue.Queue[Any] = queue.Queue()
        # 実験タイプに応じた通知間隔調整
        if experiment_type == "100k":
            # 100k実験では通知頻度を下げる
            self.flush_sec = max(flush_sec, 1800)  # 最低30分
            self.heartbeat_interval = 3600  # 1時間ごとのハートビート
        else:
            self.flush_sec = flush_sec
            self.heartbeat_interval = 1800  # 30分ごとのハートビート
        self.buf: List[str] = []
        self.experiment_type = experiment_type
        self.last_heartbeat = time.time()
        # メトリクス追跡用
        self.metrics_callback: Optional[Callable[[], Any]] = None
        threading.Thread(target=self._loop, daemon=True).start()

    def set_metrics_callback(self, callback_func: Callable[[], Any]) -> None:
        """メトリクス取得コールバックを設定"""
        self.metrics_callback = callback_func

    def enqueue(self, msg: str) -> None:
        """通常メッセージをキューに追加"""
        self.q.put(("info", msg))

    def error(self, msg: str) -> None:
        """エラーメッセージを即時送信"""
        self.q.put(("error", msg))

    def flush(self) -> None:
        """バッファを即時送信"""
        if self.buf:
            body = "\n".join(self.buf)
            self.buf.clear()
            self.logger_manager.send_custom_notification(
                "📣 Training Update", body, color=0x00AAFF
            )

    def _loop(self) -> None:
        last = time.time()
        while True:
            try:
                kind, msg = self.q.get(timeout=1)
                if kind == "error":
                    self.logger_manager.send_custom_notification(
                        "🚨 Error", msg, color=0xFF0000
                    )
                else:
                    self.buf.append(f"- {msg}")
                # 定期的に集約送信
                if time.time() - last >= self.flush_sec and self.buf:
                    body = "\n".join(self.buf)
                    self.buf.clear()
                    last = time.time()
                    self.logger_manager.send_custom_notification(
                        "📣 Training Update", body, color=0x00AAFF
                    )
            except queue.Empty:
                # タイムアウト時も集約送信チェック
                current_time = time.time()
                if current_time - last >= self.flush_sec and self.buf:
                    body = "\n".join(self.buf)
                    self.buf.clear()
                    last = current_time
                    self.logger_manager.send_custom_notification(
                        "📣 Training Update", body, color=0x00AAFF
                    )

                # ハートビート通知（100k実験のみ）
                if (
                    self.experiment_type == "100k"
                    and current_time - self.last_heartbeat >= self.heartbeat_interval
                ):
                    # メトリクス取得
                    metrics_info = ""
                    if self.metrics_callback:
                        try:
                            metrics = self.metrics_callback()
                            if metrics:
                                metrics_info = (
                                    f"\n💾 Memory: {metrics.get('memory_mb', 'N/A')}MB | "
                                    f"🧠 Objects: {metrics.get('object_count', 'N/A')} | "
                                    f"⚡ Reward: {metrics.get('avg_reward', 'N/A'):.3f}"
                                )
                        except Exception as e:
                            metrics_info = f"\n⚠️ Metrics error: {e}"

                    self.logger_manager.send_custom_notification(
                        "💓 Heartbeat",
                        f"100k experiment still running (last update: {time.strftime('%H:%M:%S', time.localtime(last))})"
                        f"{metrics_info}",
                        color=0x00FF00,
                    )
                    self.last_heartbeat = current_time


class LoggerManager:
    """Unified logging manager for experiments"""

    def __init__(
        self,
        discord_webhook: Optional[str] = None,
        log_file: Optional[str] = None,
        experiment_id: Optional[str] = None,
        test_mode: bool = False,
        enable_async: bool = True,
        experiment_type: str = "standard",
    ):
        # Load webhook from environment if not provided
        if not discord_webhook and not os.getenv("DISCORD_WEBHOOK"):
            self._load_webhook_from_env()

        self.discord_webhook = discord_webhook or os.getenv("DISCORD_WEBHOOK")
        self.log_file = log_file
        self.experiment_id = experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.test_mode = test_mode
        self.experiment_type = experiment_type
        self.session_start_time: Optional[datetime] = None
        self.session_id: Optional[str] = None

        self.jsonl_log_path = Path("logs") / f"run-{self.experiment_id}.jsonl"
        self.jsonl_log_path.parent.mkdir(exist_ok=True)

        # Setup standard logging
        self.logger = logging.getLogger(f"ztb.{self.experiment_id}")
        self.logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler with rotation
        if log_file:
            file_handler = RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Async notifier
        self.async_notifier = (
            AsyncNotifier(self, experiment_type=self.experiment_type)
            if enable_async
            else None
        )

    def _mask_secrets(self, message: str) -> str:
        """Mask sensitive information in log messages"""
        import re

        # Patterns to mask
        patterns = [
            (
                r'(?i)(api_key|apikey|key)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',
                r"\1: ***MASKED***",
            ),
            (
                r'(?i)(secret|token|webhook)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',
                r"\1: ***MASKED***",
            ),
            (
                r'(?i)(password|passwd)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{8,})["\']?',
                r"\1: ***MASKED***",
            ),
            # Discord webhook URLs
            (
                r"https://discord\.com/api/webhooks/[0-9]+/[a-zA-Z0-9_-]+",
                "https://discord.com/api/webhooks/***MASKED***/***MASKED***",
            ),
            # Generic long alphanumeric strings that look like keys
            (r"\b[a-zA-Z0-9_-]{32,}\b", "***MASKED***"),
        ]

        for pattern, replacement in patterns:
            message = re.sub(pattern, replacement, message)

        return message

    def _load_webhook_from_env(self) -> None:
        """Load Discord webhook from .env file"""
        # Search for .env file in parent directories
        search_path = Path(__file__).parent
        for _ in range(5):  # Max 5 levels up
            candidate = search_path / ".env"
            if candidate.exists():
                with open(candidate, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, value = line.split("=", 1)
                            if key == "DISCORD_WEBHOOK":
                                os.environ[key] = value
                                return
                return
            search_path = search_path.parent

    def _log_to_jsonl(self, event: str, data: Dict[str, Any]) -> None:
        """Log structured event to jsonl file"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "experiment_id": self.experiment_id,
            "event": event,
            **data,
        }
        with open(self.jsonl_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def _send_discord(
        self, message: str, embed_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send message to Discord with exponential backoff retry"""
        if not self.discord_webhook:
            self.logger.warning("No webhook URL configured, skipping notification")
            return False

        # Mask secrets in message
        message = self._mask_secrets(message)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                payload: Dict[str, Any] = {
                    "content": message,
                    "embeds": [embed_data] if embed_data else [],
                    "username": "Trading RL Bot",
                    "avatar_url": "https://i.imgur.com/4M34hi2.png",
                }

                response = requests.post(
                    self.discord_webhook,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )

                if response.status_code in (200, 204):
                    self.logger.info("Discord notification sent successfully")
                    return True
                else:
                    self.logger.error(
                        f"Failed to send Discord notification: {response.status_code}, {response.text}"
                    )
                    return False

            except Exception as e:
                wait_time = 2**attempt  # Exponential backoff
                self.logger.warning(
                    f"Discord notification attempt {attempt + 1} failed: {e}, retrying in {wait_time}s"
                )
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    self.logger.error(
                        f"Discord notification failed after {max_retries} attempts"
                    )
                    return False
        return False

    def start_session(
        self,
        session_type: str = "training",
        config_name: str = "default",
        prefix: str = "",
    ) -> str:
        """セッション開始通知"""
        self.session_start_time = datetime.now()
        self.session_id = self.session_start_time.strftime("%Y%m%d-%H%M%S")

        icon = "🧪" if self.test_mode else "🚀"
        title = f"{icon} **{session_type.title()} Session Started**"
        if prefix:
            title = f"{prefix} {title}"

        content = f"{title}\n\n**Session ID:** `{self.session_id}`\n**Config:** `{config_name}`\n**Start Time:** {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')} JST\n**Session Type:** {session_type.title()}"

        color = 0x00FF00  # Green for start
        self._send_discord(content, {"color": color})
        return str(self.session_id)

    def end_session(
        self,
        results: Dict[str, Any],
        session_type: str = "training",
        notify_on_winrate_below_threshold: bool = True,
    ) -> None:
        """セッション終了通知 with detailed results analysis

        Args:
            results: 結果データ
            session_type: セッションタイプ
            notify_on_winrate_below_threshold: 勝率が閾値を下回った場合のみ通知するか（デフォルト: True）
        """
        if not self.session_start_time:
            self.logger.warning("Session not started, cannot send end notification")
            return

        end_time = datetime.now()
        duration = end_time - self.session_start_time

        # 結果データの解析
        reward_stats = results.get("reward_stats", {})
        pnl_stats = results.get("pnl_stats", {})
        trading_stats = results.get("trading_stats", {})

        # 勝率の計算
        total_trades = trading_stats.get("total_trades", 0)
        winning_trades = trading_stats.get("winning_trades", 0)
        win_rate_percent = (
            (winning_trades / total_trades * 100) if total_trades > 0 else 0
        )
        profit_factor = trading_stats.get("profit_factor", 0)

        # 設定ファイルから閾値読み込み (オプション)
        win_rate_threshold = 50.0  # Default threshold

        # 損益単位の決定（BUY主体=BTC, SELL主体=JPY）
        buy_ratio = trading_stats.get("buy_ratio", 0)
        sell_ratio = trading_stats.get("sell_ratio", 0)
        pnl_unit = "BTC" if buy_ratio > sell_ratio else "JPY"
        total_pnl = pnl_stats.get("mean_total_pnl", 0)

        content = f"{'🧪' if self.test_mode else '✅'} **{session_type.title()} Session Completed**\n\n**Session ID:** `{self.session_id}`\n**Duration:** `{str(duration).split('.')[0]}`\n**End Time:** {end_time.strftime('%Y-%m-%d %H:%M:%S')} JST\n\n"

        # 数値系をコードブロックにまとめる
        content += "```\n"
        content += f"Win Rate: {win_rate_percent:.1f}% {'(参考)' if win_rate_percent < win_rate_threshold else ''}\n"
        content += f"Profit Factor: {profit_factor:.2f}\n"
        content += f"Total Trades: {total_trades}\n"
        content += f"Total PnL: {total_pnl:,.2f} {pnl_unit}\n"
        content += f"Max Drawdown: {pnl_stats.get('max_drawdown', 0):.4f}\n"
        content += f"Avg Hold Time: {trading_stats.get('mean_trades_per_episode', 0):.1f} trades/episode\n"
        content += f"Total Fees: {total_trades * 0.001:,.4f} BTC\n"
        content += "Max Position Size: 1.0\n"
        content += "Risk Reduction Triggers: 0\n"
        content += "```"

        # 色設定：勝率が閾値下回りなら黄、それ以外は緑
        color = 0xFFFF00 if win_rate_percent < win_rate_threshold else 0x00FF00

        # 勝率が閾値を下回った場合のみ通知 (オプションで無効化可能)
        should_notify = (
            win_rate_percent < win_rate_threshold
            if notify_on_winrate_below_threshold
            else True
        )
        if should_notify:
            self._send_discord(content, {"color": color})
        else:
            self.logger.info(
                f"Win rate {win_rate_percent:.1f}% is above threshold {win_rate_threshold}%, skipping notification"
            )
            self.logger.info(
                f"Win rate {win_rate_percent:.1f}% is above threshold {win_rate_threshold}%, skipping notification"
            )

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message"""
        self.logger.info(message)
        self._log_to_jsonl("info", {"message": message, **kwargs})

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message"""
        self.logger.warning(message)
        self._log_to_jsonl("warning", {"message": message, **kwargs})

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message"""
        self.logger.error(message)
        self._log_to_jsonl("error", {"message": message, **kwargs})

    def log_experiment_start(
        self, experiment_name: str, config: ConfigDict
    ) -> None:
        """Log experiment start"""
        # Use session management for consistency
        session_id = self.start_session("experiment", experiment_name)
        self.logger.info(
            f"🚀 Experiment '{experiment_name}' started (ID: {self.experiment_id})"
        )
        self._log_to_jsonl(
            "experiment_start",
            {
                "experiment_name": experiment_name,
                "config": config,
                "session_id": session_id,
            },
        )

    def log_experiment_end(self, results: Dict[str, Any]) -> None:
        """Log experiment end"""
        # Use session management for detailed results
        self.end_session(results, "experiment")
        self.logger.info(f"✅ Experiment completed (ID: {self.experiment_id})")
        self._log_to_jsonl("experiment_end", {"results": results})

    def log_error(self, error_message: str, details: Optional[str] = None) -> None:
        """Log error with notification"""
        self.send_error_notification(error_message, details)
        self.logger.error(
            f"❌ Error: {error_message}" + (f" - {details}" if details else "")
        )
        self._log_to_jsonl(
            "error", {"error_message": error_message, "details": details}
        )

    def log_heartbeat(self, step: int, mem_gb: float, steps_per_sec: float) -> None:
        """Log periodic heartbeat with metrics"""
        message = (
            f"💓 Heartbeat: Step {step}, Mem {mem_gb:.1f}GB, {steps_per_sec:.1f} it/s"
        )
        self.logger.info(message)

    def send_custom_notification(
        self,
        title: str,
        message: str,
        color: int = 0x0099FF,
        fields: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """カスタム通知"""
        content = f"**{title}**\n\n{message}"
        if fields:
            for field in fields:
                name = field.get("name", "")
                value = field.get("value", "")
                content += f"\n\n**{name}:** {value}"

        embed = {"color": color}
        self._send_discord(content, embed)

    def enqueue_notification(self, message: str) -> None:
        """非同期通知をキューに追加"""
        if self.async_notifier:
            self.async_notifier.enqueue(message)

    def send_error_notification(
        self, error_message: str, error_details: Optional[str] = None
    ) -> None:
        """致命的エラー時の即時通知"""
        error_time = datetime.now()

        icon = "🧪" if self.test_mode else "🚨"
        title = f"{icon} **Critical Error Alert**"

        content = f"{title}\n\n**Session ID:** `{self.session_id or 'Unknown'}`\n**Error:** {error_message}\n**Error Time:** {error_time.strftime('%Y-%m-%d %H:%M:%S')} JST"
        if error_details:
            # Truncate long error details to 1000 characters for Discord notification
            content += f"\n\n**Error Details:**\n```\n{error_details[:1000]}\n```"

        self._send_discord(content, {"color": 0xFF0000})  # Red
