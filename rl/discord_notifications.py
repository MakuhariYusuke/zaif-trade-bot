# Discord Notifications Module for Trading RL
# 取引RLプロジェクトのDiscord通知モジュール

import os
import sys
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import logging
from pathlib import Path

# ローカルモジュールのインポート
sys.path.append(str(Path(__file__).parent.parent))
from pathlib import Path


class DiscordNotifier:
    """Discord通知クラス"""

    def __init__(self, webhook_url: Optional[str] = None):
        """初期化"""
        self.webhook_url = webhook_url or os.getenv('DISCORD_WEBHOOK')
        self.session_start_time = None
        self.session_id = None

        if not self.webhook_url:
            logging.warning("DISCORD_WEBHOOK not found in environment variables")

    def _send_notification(self, embed: Dict[str, Any]) -> bool:
        """Discordに通知を送信"""
        if not self.webhook_url:
            logging.warning("No webhook URL configured, skipping notification")
            return False

        try:
            payload = {
                "embeds": [embed],
                "username": "Trading RL Bot",
                "avatar_url": "https://i.imgur.com/4M34hi2.png"  # Trading bot avatar
            }

            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )

            if response.status_code == 204:
                logging.info("Discord notification sent successfully")
                return True
            else:
                logging.error(f"Failed to send Discord notification: {response.status_code}")
                return False

        except Exception as e:
            logging.error(f"Error sending Discord notification: {e}")
            return False

    def start_session(self, session_type: str = "training", config_name: str = "default") -> str:
        """セッション開始通知"""
        self.session_start_time = datetime.now()
        self.session_id = self.session_start_time.strftime('%Y%m%d-%H%M')

        embed = {
            "title": f"🚀 {session_type.title()} Session Started",
            "description": f"**Session ID:** `{self.session_id}`\n**Config:** `{config_name}`",
            "color": 0x00ff00,  # Green
            "fields": [
                {
                    "name": "⏰ Start Time (JST)",
                    "value": self.session_start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "inline": True
                },
                {
                    "name": "📊 Session Type",
                    "value": session_type.title(),
                    "inline": True
                },
                {
                    "name": "⚙️ Configuration",
                    "value": config_name,
                    "inline": True
                }
            ],
            "timestamp": self.session_start_time.isoformat(),
            "footer": {
                "text": "Trading RL Bot"
            }
        }

        self._send_notification(embed)
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
        # 簡易的な勝率計算（実際のロジックは取引結果に基づく）
        win_rate = pnl_stats.get('mean_total_pnl', 0) > 0

        # 損益単位の決定（BUY主体=BTC, SELL主体=JPY）
        buy_ratio = trading_stats.get('buy_ratio', 0)
        sell_ratio = trading_stats.get('sell_ratio', 0)
        pnl_unit = "BTC" if buy_ratio > sell_ratio else "JPY"
        total_pnl = pnl_stats.get('mean_total_pnl', 0)

        embed = {
            "title": f"✅ {session_type.title()} Session Completed",
            "description": f"**Session ID:** `{self.session_id}`\n**Duration:** `{str(duration).split('.')[0]}`",
            "color": 0x0000ff,  # Blue
            "fields": [
                {
                    "name": "⏰ End Time (JST)",
                    "value": end_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "inline": True
                },
                {
                    "name": "⏱️ Duration",
                    "value": str(duration).split('.')[0],
                    "inline": True
                },
                {
                    "name": "🎯 Win Rate",
                    "value": f"{'✅' if win_rate else '❌'} ({'Win' if win_rate else 'Loss'})",
                    "inline": True
                },
                {
                    "name": "📊 Total Trades",
                    "value": str(total_trades),
                    "inline": True
                },
                {
                    "name": "💰 Total PnL",
                    "value": f"{total_pnl:,.2f} {pnl_unit}",
                    "inline": True
                },
                {
                    "name": "📉 Max Drawdown",
                    "value": f"{pnl_stats.get('max_drawdown', 0):.4f}",
                    "inline": True
                },
                {
                    "name": "⏳ Avg Hold Time",
                    "value": f"{trading_stats.get('mean_trades_per_episode', 0):.1f} trades/episode",
                    "inline": True
                },
                {
                    "name": "💵 Total Fees",
                    "value": f"{total_trades * 0.001:,.4f} BTC",  # 仮定の取引手数料
                    "inline": True
                },
                {
                    "name": "📈 Max Position Size",
                    "value": "1.0",  # 仮定値
                    "inline": True
                },
                {
                    "name": "🛡️ Risk Reduction Triggers",
                    "value": "0",  # 仮定値
                    "inline": True
                }
            ],
            "timestamp": end_time.isoformat(),
            "footer": {
                "text": f"Trading RL Bot - {session_type.title()} Complete"
            }
        }

        self._send_notification(embed)

    def send_error_notification(self, error_message: str, error_details: Optional[str] = None) -> None:
        """致命的エラー時の即時通知"""
        error_time = datetime.now()

        embed = {
            "title": "🚨 Critical Error Alert",
            "description": f"**Session ID:** `{self.session_id or 'Unknown'}`\n**Error:** {error_message}",
            "color": 0xff0000,  # Red
            "fields": [
                {
                    "name": "⏰ Error Time (JST)",
                    "value": error_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "inline": False
                }
            ],
            "timestamp": error_time.isoformat(),
            "footer": {
                "text": "Trading RL Bot - Error Alert"
            }
        }

        if error_details:
            embed["fields"].append({
                "name": "📋 Error Details",
                "value": error_details[:1000],  # Discordの文字数制限対策
                "inline": False
            })

        self._send_notification(embed)

    def send_custom_notification(self, title: str, message: str, color: int = 0x0099ff,
                               fields: Optional[List[Dict[str, Union[str, bool]]]] = None) -> None:
        """カスタム通知"""
        notification_time = datetime.now()

        embed = {
            "title": title,
            "description": message,
            "color": color,
            "timestamp": notification_time.isoformat(),
            "footer": {
                "text": "Trading RL Bot"
            }
        }

        if fields:
            embed["fields"] = fields

        self._send_notification(embed)


# グローバルインスタンス
notifier = DiscordNotifier()


def notify_session_start(session_type: str = "training", config_name: str = "default") -> str:
    """セッション開始を通知"""
    return notifier.start_session(session_type, config_name)


def notify_session_end(results: Dict[str, Any], session_type: str = "training") -> None:
    """セッション終了を通知"""
    notifier.end_session(results, session_type)


def notify_error(error_message: str, error_details: Optional[str] = None) -> None:
    """エラーを通知"""
    notifier.send_error_notification(error_message, error_details)


def notify_custom(title: str, message: str, color: int = 0x0099ff,
                 fields: Optional[List[Dict[str, Union[str, bool]]]] = None) -> None:
    """カスタム通知"""
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

    # セッション開始テスト
    session_id = notify_session_start("test", "test_config")
    print(f"✅ Session start notification sent (ID: {session_id})")

    # カスタム通知テスト
    notify_custom(
        "🧪 Test Notification",
        "This is a test notification from Trading RL Bot",
        0x00ff00,
        [
            {"name": "Test Field 1", "value": "Value 1", "inline": True},
            {"name": "Test Field 2", "value": "Value 2", "inline": True}
        ]
    )
    print("✅ Custom notification sent")

    # セッション終了テスト
    notify_session_end(mock_results, "test")
    print("✅ Session end notification sent")

    # エラーテスト
    notify_error("Test Error", "This is a test error message")
    print("✅ Error notification sent")

    print("🎉 All notification tests completed!")


if __name__ == "__main__":
    # コマンドラインからテスト実行
    test_notifications()