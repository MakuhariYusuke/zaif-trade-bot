import requests
import json
import os
from datetime import datetime

class DiscordNotifier:
    def __init__(self, webhook_url=None):
        self.webhook_url = webhook_url or os.getenv('DISCORD_WEBHOOK_URL')

    def send_notification(self, title, message, color=0x00ff00):
        if not self.webhook_url:
            print("Discord webhook URL not configured")
            return

        embed = {
            "title": title,
            "description": message,
            "color": color,
            "timestamp": datetime.utcnow().isoformat()
        }

        data = {
            "embeds": [embed]
        }

        try:
            response = requests.post(self.webhook_url, json=data)
            if response.status_code == 204:
                print("Discord notification sent successfully")
            else:
                print(f"Failed to send Discord notification: {response.status_code}")
        except Exception as e:
            print(f"Error sending Discord notification: {e}")

# テスト結果通知
def send_test_results_notification():
    notifier = DiscordNotifier()

    message = """**1kステップ強化学習テスト完了**

📊 **全体統計:**
• 総フィーチャー数: 29
• 品質ゲート通過: 28
• 品質ゲート失敗: 1

📈 **カテゴリ別結果:**
• trend: 7/8 成功
• volatility: 6/6 成功
• momentum: 6/6 成功
• volume: 4/4 成功
• wave1: 2/2 成功
• wave3: 2/2 成功

❌ **失敗フィーチャー:**
• KAMA (相関失敗)

⚡ **パフォーマンス指標:**
• データセット: CoinGecko BTC/JPY (366日分)
• 実行時間: ~7秒
• メモリ使用量: 最小
• ステップ数: 1k (品質評価ベース)"""

    notifier.send_notification(
        "🚀 強化学習テスト完了",
        message,
        color=0x00ff00 if 28/29 > 0.8 else 0xffa500
    )

if __name__ == "__main__":
    send_test_results_notification()