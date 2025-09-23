#!/usr/bin/env python3
# Test Script for Enhanced Logging and Notifications
# 強化されたログ・通知機能テストスクリプト

import os
import sys
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

# Discord通知モジュールのインポート
from rl.notify.discord.discord_notifications import DiscordNotifier

def test_enhanced_logging():
    """強化されたログ機能をテスト"""
    print("🧪 Testing Enhanced Logging and Notifications...")

    # DiscordNotifierインスタンス作成
    notifier = DiscordNotifier()

    # セッション開始
    session_id = notifier.start_session("test_enhanced", "logging_test")
    print(f"✅ Session started: {session_id}")

    # カスタム通知
    notifier.send_custom_notification(
        "🔧 Enhanced Logging Test",
        "Testing improved logging and notification system",
        color=0x00ff00
    )
    print("✅ Custom notification sent")

    # 強制例外テスト
    try:
        raise ValueError("Test exception for logging verification")
    except Exception as e:
        logging.exception(f"Test exception occurred: {e}")
        notifier.send_error_notification("Test Error", f"Exception: {str(e)}")
        print("✅ Exception logged and notified")

    # セッション終了
    mock_results = {
        'reward_stats': {'mean_total_reward': 1000.0},
        'pnl_stats': {'mean_total_pnl': 500.0, 'max_drawdown': 0.05},
        'trading_stats': {
            'total_trades': 100,
            'winning_trades': 55,
            'profit_factor': 1.2,
            'mean_trades_per_episode': 5.0,
            'buy_ratio': 0.6,
            'sell_ratio': 0.4
        }
    }

    notifier.end_session(mock_results, "test_enhanced")
    print("✅ Session ended")

    print("🎉 All enhanced logging tests completed!")
    print("📋 Check logs/ directory for session log files")
    print("📱 Check Discord for notifications")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    test_enhanced_logging()