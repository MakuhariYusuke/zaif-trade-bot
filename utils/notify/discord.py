# Discord Notifications Module for Trading RL
# 取引RLプロジェクトのDiscord通知モジュール

import os
import sys
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
import queue
import threading
import time

# ローカルモジュールのインポート
sys.path.append(str(Path(__file__).parent.parent))
from pathlib import Path


class AsyncNotifier:
    """非同期Discord通知クラス（キュー + 集約送信）"""
    
    def __init__(self, notifier, flush_sec=300):
        self.n = notifier
        self.q = queue.Queue()
        self.flush_sec = flush_sec
        self.buf = []
        threading.Thread(target=self._loop, daemon=True).start()

    def enqueue(self, msg):
        """通常メッセージをキューに追加"""
        self.q.put(("info", msg))
    
    def error(self, msg):
        """エラーメッセージを即時送信"""
        self.q.put(("error", msg))

    def _loop(self):
        last = time.time()
        while True:
            try:
                kind, msg = self.q.get(timeout=1)
                if kind == "error":
                    self.n.send_custom_notification("🚨 Error", msg, color=0xFF0000)
                else:
                    self.buf.append(f"- {msg}")
                # 定期的に集約送信
                if time.time() - last >= self.flush_sec and self.buf:
                    body = "\n".join(self.buf)
                    self.buf.clear()
                    last = time.time()
                    self.n.send_custom_notification("📣 Training Update", body, color=0x00AAFF)
            except queue.Empty:
                # タイムアウト時も集約送信チェック
                if time.time() - last >= self.flush_sec and self.buf:
                    body = "\n".join(self.buf)
                    self.buf.clear()
                    last = time.time()
                    self.n.send_custom_notification("📣 Training Update", body, color=0x00AAFF)


class DiscordNotifier:
    """Discord通知クラス"""

    def __init__(self, webhook_url: Optional[str] = None, test_mode: bool = False):
        """初期化"""
        self.test_mode = test_mode
        # 環境変数が設定されていない場合、.envファイルから読み込み
        if not os.getenv('DISCORD_WEBHOOK'):
            env_path = Path(__file__).parent.parent.parent.parent / '.env'
            # .envファイルを現在のディレクトリから親ディレクトリに向かって探索
            env_path = None
            search_path = Path(__file__).parent
            for _ in range(5):  # 最大5階層まで探索
                candidate = search_path / '.env'
                if candidate.exists():
                    env_path = candidate
                    break
                search_path = search_path.parent
            if env_path:
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            if key == 'DISCORD_WEBHOOK':
                                os.environ[key] = value
                                break
        self.webhook_url = webhook_url or os.getenv('DISCORD_WEBHOOK')
        self.session_start_time = None
        self.session_id = None

        if not self.webhook_url:
            logging.warning("DISCORD_WEBHOOK not found in environment variables")

    def _send_notification(self, content: str, color: int = 0x0099ff) -> bool:
        """Discordに通知を送信"""
        if not self.webhook_url:
            logging.warning("No webhook URL configured, skipping notification")
            return False

        try:
            payload = {
                "content": content,
                "embeds": [{"color": color}],
                "username": "Trading RL Bot",
                "avatar_url": "https://i.imgur.com/4M34hi2.png"  # Trading bot avatar
            }

            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )

            if response.status_code in (200, 204):
                logging.info("Discord notification sent successfully")
                return True
            else:
                logging.error(f"Failed to send Discord notification: {response.status_code}, {response.text}")
                return False

        except Exception as e:
            logging.error(f"Error sending Discord notification: {e}")
            return False

    def start_session(self, session_type: str = "training", config_name: str = "default") -> str:
        """セッション開始通知"""
        self.session_start_time = datetime.now()
        self.session_id = self.session_start_time.strftime('%Y%m%d-%H%M%S')

        icon = "🧪" if self.test_mode else "🚀"
        title = f"{icon} **{session_type.title()} Session Started**"

        content = f"{title}\n\n**Session ID:** `{self.session_id}`\n**Config:** `{config_name}`\n**Start Time:** {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')} JST\n**Session Type:** {session_type.title()}"

        color = 0x00ff00  # Green for start
        self._send_notification(content, color)
        return self.session_id

    def end_session(self, results: Dict[str, Any], session_type: str = "training") -> None:
        """セッション終了通知"""
        if not self.session_start_time:
            logging.warning("Session not started, cannot send end notification")
            return

        end_time = datetime.now()
        duration = end_time - self.session_start_time

        # 結果データの解析
        reward_stats = results.get('reward_stats', {})
        pnl_stats = results.get('pnl_stats', {})
        trading_stats = results.get('trading_stats', {})

        # 勝率の計算
        total_trades = trading_stats.get('total_trades', 0)
        winning_trades = trading_stats.get('winning_trades', 0)
        win_rate_percent = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        profit_factor = trading_stats.get('profit_factor', 0)

        # 設定ファイルから閾値読み込み
        config_path = Path("../rl_config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            win_rate_threshold = config.get('notifications', {}).get('win_rate_threshold_percent', 50.0)
        else:
            win_rate_threshold = 50.0

        # 閾値下回りチェック
        should_notify = win_rate_percent < win_rate_threshold

        # 損益単位の決定（BUY主体=BTC, SELL主体=JPY）
        buy_ratio = trading_stats.get('buy_ratio', 0)
        sell_ratio = trading_stats.get('sell_ratio', 0)
        pnl_unit = "BTC" if buy_ratio > sell_ratio else "JPY"
        total_pnl = pnl_stats.get('mean_total_pnl', 0)

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
        content += f"Max Position Size: 1.0\n"
        content += f"Risk Reduction Triggers: 0\n"
        content += "```"

        # 色設定：勝率が閾値下回りなら黄、それ以外は緑
        color = 0xffff00 if win_rate_percent < win_rate_threshold else 0x00ff00

        # 勝率が閾値を下回った場合のみ通知
        if should_notify:
            self._send_notification(content, color)
        else:
            logging.info(f"Win rate {win_rate_percent:.1f}% is above threshold {win_rate_threshold}%, skipping notification")

    def send_error_notification(self, error_message: str, error_details: Optional[str] = None) -> None:
        """致命的エラー時の即時通知"""
        error_time = datetime.now()

        icon = "🧪" if self.test_mode else "🚨"
        title = f"{icon} **Critical Error Alert**"

        content = f"{title}\n\n**Session ID:** `{self.session_id or 'Unknown'}`\n**Error:** {error_message}\n**Error Time:** {error_time.strftime('%Y-%m-%d %H:%M:%S')} JST"
        if error_details:
            content += f"\n\n**Error Details:**\n```\n{error_details[:1000]}\n```"

        self._send_notification(content, 0xff0000)  # Red

    def send_custom_notification(self, title: str, message: str, color: int = 0x0099ff,
                               fields: Optional[List[Dict[str, Union[str, bool]]]] = None) -> None:
        """カスタム通知"""
        content = f"**{title}**\n\n{message}"
        if fields:
            for field in fields:
                name = field.get('name', '')
                value = field.get('value', '')
                content += f"\n\n**{name}:** {value}"

        self._send_notification(content, color)


# DiscordNotifierインスタンスのファクトリ関数
def get_notifier(webhook_url: Optional[str] = None) -> DiscordNotifier:
    return DiscordNotifier(webhook_url)


def notify_session_start(session_type: str = "training", config_name: str = "default", notifier: Optional[DiscordNotifier] = None) -> str:
    """セッション開始を通知"""
    notifier = notifier or get_notifier()
    return notifier.start_session(session_type, config_name)


def notify_session_end(results: Dict[str, Any], session_type: str = "training", notifier: Optional[DiscordNotifier] = None) -> None:
    """セッション終了を通知"""
    notifier = notifier or get_notifier()
    notifier.end_session(results, session_type)


def notify_error(error_message: str, error_details: Optional[str] = None, notifier: Optional[DiscordNotifier] = None) -> None:
    """エラーを通知"""
    notifier = notifier or get_notifier()
    notifier.send_error_notification(error_message, error_details)


def notify_custom(title: str, message: str, color: int = 0x0099ff,
                 fields: Optional[List[Dict[str, Union[str, bool]]]] = None,
                 notifier: Optional[DiscordNotifier] = None) -> None:
    """カスタム通知"""
    notifier = notifier or get_notifier()
    notifier.send_custom_notification(title, message, color, fields)


# Dry-runテスト関数
def test_notifications() -> None:
    """通知機能のテスト（Dry-run）"""
    print("🧪 Testing Discord Notifications...")

    # テスト用のモックデータ
    mock_results = {
        'reward_stats': {
            'mean_total_reward': 1000.0,
            'std_total_reward': 100.0
        },
        'pnl_stats': {
            'mean_total_pnl': 500.0,
            'max_drawdown': 0.05
        },
        'trading_stats': {
            'total_trades': 50,
            'mean_trades_per_episode': 5.0,
            'buy_ratio': 0.6,
            'sell_ratio': 0.4
        }
    }

    # テスト用インスタンス
    test_notifier = get_notifier()

    # セッション開始テスト
    session_id = notify_session_start("test", "test_config", notifier=test_notifier)
    print(f"✅ Session start notification sent (ID: {session_id})")

    # カスタム通知テスト
    notify_custom(
        "🧪 Test Notification",
        "This is a test notification from Trading RL Bot",
        0x00ff00,
        [
            {"name": "Test Field 1", "value": "Value 1", "inline": True},
            {"name": "Test Field 2", "value": "Value 2", "inline": True}
        ],
        notifier=test_notifier
    )
    print("✅ Custom notification sent")

    # セッション終了テスト
    notify_session_end(mock_results, "test", notifier=test_notifier)
    print("✅ Session end notification sent")

    # エラーテスト
    notify_error("Test Error", "This is a test error message", notifier=test_notifier)
    print("✅ Error notification sent")

    print("🎉 All notification tests completed!")


if __name__ == "__main__":
    # コマンドラインからテスト実行
    test_notifications()