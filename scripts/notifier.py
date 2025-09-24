#!/usr/bin/env python3
"""
notifier.py
Slack/Discord通知機能
"""

import os
import json
import requests
from typing import Dict, Optional
from pathlib import Path


def send_summary(summary: Dict, channel: str = "slack") -> bool:
    """
    評価サマリーを通知

    Args:
        summary: レポート生成時に作成するdict
            {
                "total_features": int,
                "valid_results": int,
                "experimental_count": int,
                "re_evaluate_count": int,
                "monitor_count": int,
                "maintain_count": int,
                "success_rate": float
            }
        channel: "slack" or "discord"

    Returns:
        bool: 送信成功時True
    """
    if channel == "slack":
        return _send_slack(summary)
    elif channel == "discord":
        return _send_discord(summary)
    else:
        print(f"Unsupported channel: {channel}")
        return False


def _send_slack(summary: Dict) -> bool:
    """Slack通知"""
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("SLACK_WEBHOOK_URL not set")
        return False

    # Slackメッセージ作成
    message = {
        "text": "Weekly Feature Evaluation Report",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "📊 Weekly Feature Evaluation Report"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Total Features:*\n{summary['total_features']}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Valid Results:*\n{summary['valid_results']}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Experimental:*\n{summary['experimental_count']}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Success Rate:*\n{summary['success_rate']:.1f}%"
                    }
                ]
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*🔄 Re-evaluate:*\n{summary['re_evaluate_count']}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*👀 Monitor:*\n{summary['monitor_count']}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*✅ Maintain:*\n{summary['maintain_count']}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Report:*\n<weekly_report.md|View Details>"
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(webhook_url, json=message)
        response.raise_for_status()
        print("Slack notification sent successfully")
        return True
    except Exception as e:
        print(f"Failed to send Slack notification: {e}")
        return False


def _send_discord(summary: Dict) -> bool:
    """Discord通知"""
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("DISCORD_WEBHOOK_URL not set")
        return False

    # Discordメッセージ作成
    embed = {
        "title": "📊 Weekly Feature Evaluation Report",
        "color": 0x00ff00,
        "fields": [
            {
                "name": "📈 Total Features",
                "value": str(summary["total_features"]),
                "inline": True
            },
            {
                "name": "✅ Valid Results",
                "value": str(summary["valid_results"]),
                "inline": True
            },
            {
                "name": "🧪 Experimental",
                "value": str(summary["experimental_count"]),
                "inline": True
            },
            {
                "name": "📊 Success Rate",
                "value": f"{summary['success_rate']:.1f}%",
                "inline": True
            },
            {
                "name": "🔄 Re-evaluate",
                "value": str(summary["re_evaluate_count"]),
                "inline": True
            },
            {
                "name": "👀 Monitor",
                "value": str(summary["monitor_count"]),
                "inline": True
            },
            {
                "name": "✅ Maintain",
                "value": str(summary["maintain_count"]),
                "inline": True
            }
        ],
        "footer": {
            "text": "View full report: weekly_report.md"
        }
    }

    message = {"embeds": [embed]}

    try:
        response = requests.post(webhook_url, json=message)
        response.raise_for_status()
        print("Discord notification sent successfully")
        return True
    except Exception as e:
        print(f"Failed to send Discord notification: {e}")
        return False


def create_summary_dict(ablation_results: Dict, config: Dict) -> Dict:
    """レポート生成時に使用するサマリdict作成"""
    total_features = len(ablation_results["ablation_results"])
    valid_results = ablation_results.get("summary", {}).get("valid_results", 0)
    experimental_count = len(ablation_results.get("experimental_features", []))

    # ステータス別カウント
    re_evaluate_count = 0
    monitor_count = 0
    maintain_count = 0

    for result in ablation_results["ablation_results"].values():
        status = _get_status(result, config)
        if status == "**Re-evaluate**":
            re_evaluate_count += 1
        elif status == "Monitor":
            monitor_count += 1
        elif status == "Maintain":
            maintain_count += 1

    success_rate = (valid_results / total_features * 100) if total_features > 0 else 0

    return {
        "total_features": total_features,
        "valid_results": valid_results,
        "experimental_count": experimental_count,
        "re_evaluate_count": re_evaluate_count,
        "monitor_count": monitor_count,
        "maintain_count": maintain_count,
        "success_rate": success_rate
    }


def _get_status(feature_result: Dict, config: Dict) -> str:
    """特徴量のステータス判定（should_re_evaluateの簡易版）"""
    ds = feature_result.get('delta_sharpe', {})
    if not ds:
        return "Insufficient Data"

    mean = ds.get('mean', 0)
    ci_low = ds.get('ci95', [0, 0])[0]

    thresholds = config.get('thresholds', {})
    re_evaluate_threshold = thresholds.get('re_evaluate', 0.05)
    monitor_threshold = thresholds.get('monitor', 0.01)

    if mean > re_evaluate_threshold and ci_low > 0:
        return "**Re-evaluate**"
    elif mean > monitor_threshold:
        return "Monitor"
    else:
        return "Maintain"