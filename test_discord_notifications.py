#!/usr/bin/env python3
# Discord Notifications Test Script
# Discord通知機能テストスクリプト

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent))

# Discord通知モジュールのインポート
from rl.notify.discord.discord_notifications import DiscordNotifier

def load_env_file():
    """環境変数ファイルを読み込み"""
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✅ .env file loaded")
    else:
        print("⚠️ .env file not found")

def create_mock_results() -> Dict[str, Any]:
    """テスト用のモック結果データを作成"""
    return {
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
            'winning_trades': 25,
            'profit_factor': 1.2,
            'mean_trades_per_episode': 5.0,
            'buy_ratio': 0.6,
            'sell_ratio': 0.4
        }
    }

def test_discord_notifications():
    """Discord通知機能をテスト"""
    print("🧪 Starting Discord Notifications Test...")
    print("=" * 50)

    # 環境変数読み込み
    load_env_file()

    # Webhook URL確認
    webhook_url = os.getenv('DISCORD_WEBHOOK')
    if webhook_url:
        print(f"✅ DISCORD_WEBHOOK found: {webhook_url[:50]}...")
    else:
        print("❌ DISCORD_WEBHOOK not found in environment")
        return

    # DiscordNotifierインスタンス作成（環境変数読み込み後）
    notifier = DiscordNotifier()

    # モックデータ作成
    mock_results = create_mock_results()

    try:
        # 1. セッション開始通知
        print("\n1️⃣ Testing notify_session_start...")
        session_id = notifier.start_session("test", "test_config")
        print(f"✅ Session start notification sent (ID: {session_id})")

        # 2. カスタム通知
        print("\n2️⃣ Testing notify_custom...")
        notifier.send_custom_notification(
            "🧪 Test Notification",
            "This is a test message from Trading RL Bot",
            0x00ff00
        )
        print("✅ Custom notification sent")

        # 3. セッション終了通知
        print("\n3️⃣ Testing notify_session_end...")
        notifier.end_session(mock_results, "test")
        print("✅ Session end notification sent")

        # 4. エラー通知
        print("\n4️⃣ Testing notify_error...")
        notifier.send_error_notification("Test Error", "This is a test error message")
        print("✅ Error notification sent")

        print("\n" + "=" * 50)
        print("🎉 All Discord notification tests completed successfully!")
        print("📱 Please check your Discord channel for the notifications.")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_discord_notifications()