#!/usr/bin/env python3
"""
Extract reward correlation data from latest training logs
最新のトレーニングログから報酬相関データを抽出
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_reward_data_from_logs(log_content: str) -> pd.DataFrame:
    """ログから報酬データを抽出"""
    # 正規表現パターンでログを解析
    pattern = r"Backtest optimization: portfolio_delta=([-\d.]+), portfolio_pct_change=([-\d.]+), portfolio_reward=([-\d.]+), position_bonus=([-\d.]+), action_penalty=([-\d.]+), total_reward=([-\d.]+)"

    matches = re.findall(pattern, log_content)

    if not matches:
        print("No reward data found in logs")
        return pd.DataFrame()

    data = []
    for match in matches:
        portfolio_delta = float(match[0])
        portfolio_pct_change = float(match[1])
        portfolio_reward = float(match[2])
        position_bonus = float(match[3])
        action_penalty = float(match[4])
        total_reward = float(match[5])

        data.append(
            {
                "portfolio_delta": portfolio_delta,
                "portfolio_pct_change": portfolio_pct_change,
                "portfolio_reward": portfolio_reward,
                "position_bonus": position_bonus,
                "action_penalty": action_penalty,
                "total_reward": total_reward,
            }
        )

    return pd.DataFrame(data)


def analyze_correlation(df: pd.DataFrame) -> dict:
    """相関分析を実行"""
    if df.empty:
        return {}

    # portfolio_pct_changeとtotal_rewardの相関
    pct_change_corr = np.corrcoef(df["portfolio_pct_change"], df["total_reward"])[0, 1]

    # portfolio_deltaとtotal_rewardの相関
    delta_corr = np.corrcoef(df["portfolio_delta"], df["total_reward"])[0, 1]

    # 符号一致率
    pct_change_signs = np.sign(df["portfolio_pct_change"])
    reward_signs = np.sign(df["total_reward"])
    sign_agreement = np.mean(pct_change_signs == reward_signs)

    analysis = {
        "total_samples": len(df),
        "pct_change_reward_correlation": pct_change_corr,
        "delta_reward_correlation": delta_corr,
        "sign_agreement_rate": sign_agreement,
        "reward_range": (df["total_reward"].min(), df["total_reward"].max()),
        "pct_change_range": (
            df["portfolio_pct_change"].min(),
            df["portfolio_pct_change"].max(),
        ),
        "portfolio_delta_range": (
            df["portfolio_delta"].min(),
            df["portfolio_delta"].max(),
        ),
    }

    return analysis


def main():
    """メイン関数"""
    print("=== Extracting Reward Correlation from Latest Training Logs ===")

    # 最新のログファイルを見つける（仮定）
    # 実際にはログファイルのパスを指定
    log_file = Path("training_log.txt")

    if not log_file.exists():
        print(
            "Log file not found. Please save the training output to 'training_log.txt'"
        )
        return

    with open(log_file, "r", encoding="utf-8") as f:
        log_content = f.read()

    # ログからデータを抽出
    df = extract_reward_data_from_logs(log_content)

    if df.empty:
        print("No data extracted from logs")
        return

    print(f"Extracted {len(df)} reward samples")

    # 相関分析
    analysis = analyze_correlation(df)

    print("\n=== Correlation Analysis Results ===")
    print(f"Total samples: {analysis['total_samples']}")
    print(
        f"Pearson correlation (pct_change vs total_reward): {analysis['pct_change_reward_correlation']:.6f}"
    )
    print(
        f"Pearson correlation (delta vs total_reward): {analysis['delta_reward_correlation']:.6f}"
    )
    print(f"Sign agreement rate: {analysis['sign_agreement_rate']:.4f}")
    print(f"Reward range: {analysis['reward_range']}")
    print(f"Pct change range: {analysis['pct_change_range']}")
    print(f"Portfolio delta range: {analysis['portfolio_delta_range']}")

    # データを保存
    df.to_csv("latest_training_rewards.csv", index=False)
    print("\nData saved to 'latest_training_rewards.csv'")

    # 散布図を作成
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.scatter(df["portfolio_pct_change"], df["total_reward"], alpha=0.6)
    plt.xlabel("Portfolio % Change")
    plt.ylabel("Total Reward")
    plt.title(".4f")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.scatter(df["portfolio_delta"], df["total_reward"], alpha=0.6)
    plt.xlabel("Portfolio Delta")
    plt.ylabel("Total Reward")
    plt.title(".4f")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.hist(df["total_reward"], bins=50, alpha=0.7)
    plt.xlabel("Total Reward")
    plt.ylabel("Frequency")
    plt.title("Reward Distribution")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("latest_training_correlation.png", dpi=150, bbox_inches="tight")
    print("Plot saved to 'latest_training_correlation.png'")


if __name__ == "__main__":
    main()
