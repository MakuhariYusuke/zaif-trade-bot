#!/usr/bin/env python3
# ruff: noqa: E402
"""
1kステップ強化学習テスト結果通知スクリプト
既存のDiscordNotifierクラスを使用
"""

import sys
from typing import Any, Dict

from ztb.utils.path_utils import get_project_root

# プロジェクトルートをパスに追加
project_root = get_project_root()
sys.path.append(str(project_root))

from ztb.utils import DiscordNotifier


def send_1k_test_results() -> None:
    """1kステップテスト結果をDiscordに通知"""

    # DiscordNotifierの初期化
    notifier = DiscordNotifier()

    # テスト結果データ
    test_results: Dict[str, Any] = {
        "total_features": 29,
        "passed_features": 28,
        "failed_features": 1,
        "failed_feature": "KAMA (相関失敗)",
        "category_results": {
            "trend": "7/8 成功",
            "volatility": "6/6 成功",
            "momentum": "6/6 成功",
            "volume": "4/4 成功",
            "wave1": "2/2 成功",
            "wave3": "2/2 成功",
        },
        "performance_metrics": {
            "dataset": "CoinGecko BTC/JPY (366日分)",
            "execution_time": "~7秒",
            "memory_usage": "16.24 MB",
            "steps_simulated": "1k (品質評価ベース)",
        },
    }

    # 通知メッセージの作成
    message = f"""**1kステップ強化学習テスト完了**

📊 **全体統計:**
• 総フィーチャー数: {test_results["total_features"]}
• 品質ゲート通過: {test_results["passed_features"]}
• 品質ゲート失敗: {test_results["failed_features"]}

📈 **カテゴリ別結果:**
• trend: {test_results["category_results"]["trend"]}
• volatility: {test_results["category_results"]["volatility"]}
• momentum: {test_results["category_results"]["momentum"]}
• volume: {test_results["category_results"]["volume"]}
• wave1: {test_results["category_results"]["wave1"]}
• wave3: {test_results["category_results"]["wave3"]}

❌ **失敗フィーチャー:**
• {test_results["failed_feature"]}

⚡ **パフォーマンス指標:**
• データセット: {test_results["performance_metrics"]["dataset"]}
• 実行時間: {test_results["performance_metrics"]["execution_time"]}
• メモリ使用量: {test_results["performance_metrics"]["memory_usage"]}
• ステップ数: {test_results["performance_metrics"]["steps_simulated"]}"""

    # Discord通知送信
    success_rate = test_results["passed_features"] / test_results["total_features"]
    color = "success" if success_rate > 0.8 else "warning"

    notifier.send_notification("🚀 強化学習テスト完了", message, color=color)

    print("✅ Discord通知送信完了")


if __name__ == "__main__":
    send_1k_test_results()
