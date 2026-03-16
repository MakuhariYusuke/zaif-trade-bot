#!/usr/bin/env python3
"""
SELLアクション発生ゼロの根本原因調査スクリプト
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import json

import pandas as pd

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL


def test_environment_sell_execution():
    """環境がSELLアクションを正しく実行するかテスト"""
    print("=== 環境SELLアクション実行テスト ===")

    try:
        # 設定ファイル読み込み
        config_path = "config/v445/sac_v445.3_strong_selling_optimized.json"
        with open(config_path, "r") as f:
            config_data = json.load(f)

        # 環境設定 - training.environmentを使用
        env_config_dict = config_data["training"]["environment"]

        # データ読み込み
        data_path = config_data["training"]["data_config"]["data_path"]
        df = pd.read_csv(data_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        print(f"データ読み込み完了: {len(df)}行")
        print(f"データ期間: {df['timestamp'].min()} - {df['timestamp'].max()}")

        # 環境初期化を試行
        try:
            from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
            from ztb.trading.environment.utils.config import EnvironmentConfig

            env_config = EnvironmentConfig.from_dict(env_config_dict)
            env = HeavyTradingEnv(
                config=env_config,
                df=df,
                use_observation_normalization=True,
                use_reward_normalization=False,
            )

            obs, info = env.reset()
            print("✅ 環境初期化成功")
            print(f"初期ポジション: {env.position}")
            print(f"初期ポートフォリオ: {env.portfolio_value}")

            # BUYアクション実行
            print("\n--- BUYアクション実行 ---")
            obs, reward, terminated, truncated, info = env.step(ACTION_BUY)
            print(f"BUY後ポジション: {env.position}")
            print(f"BUY後ポートフォリオ: {env.portfolio_value}")
            print(f"BUY報酬: {reward}")

            # SELLアクション実行
            print("\n--- SELLアクション実行 ---")
            obs, reward, terminated, truncated, info = env.step(ACTION_SELL)
            print(f"SELL後ポジション: {env.position}")
            print(f"SELL後ポートフォリオ: {env.portfolio_value}")
            print(f"SELL報酬: {reward}")

            env.close()
            print("✅ 環境SELLアクション実行テスト完了")

        except Exception as e:
            print(f"❌ 環境初期化/実行エラー: {e}")
            import traceback

            traceback.print_exc()

    except Exception as e:
        print(f"❌ 設定ファイル読み込みエラー: {e}")
        import traceback

        traceback.print_exc()


def analyze_backtest_results():
    """バックテスト結果を分析してSELLアクションの有無を確認"""
    print("\n=== バックテスト結果分析 ===")

    try:
        # バックテスト結果ファイルを探す
        backtest_files = [
            "backtest_results_sac_v444.json",
            "backtest_results_sac_v444_2.json",
            "backtest_results.json",
        ]

        for filename in backtest_files:
            filepath = f"backtest_results/{filename}"
            if os.path.exists(filepath):
                print(f"\n--- {filename} の分析 ---")
                with open(filepath, "r") as f:
                    results = json.load(f)

                # アクション分布を確認
                if "action_distribution" in results:
                    action_dist = results["action_distribution"]
                    print(f"アクション分布: {action_dist}")

                    total_actions = sum(action_dist.values())
                    sell_count = action_dist.get("SELL", 0)
                    sell_percentage = (
                        (sell_count / total_actions * 100) if total_actions > 0 else 0
                    )

                    print(f"SELLアクション数: {sell_count} ({sell_percentage:.1f}%)")

                    if sell_count == 0:
                        print(
                            "🚨 このバックテストではSELLアクションが1回も発生していません"
                        )
                    else:
                        print("✅ SELLアクションが正常に発生しています")

                # 取引履歴を確認
                if "trades" in results:
                    trades = results["trades"]
                    sell_trades = [t for t in trades if t.get("action") == "SELL"]
                    print(f"SELL取引数: {len(sell_trades)}")

                    if len(sell_trades) == 0:
                        print("🚨 SELL取引が1件もありません")

    except Exception as e:
        print(f"❌ バックテスト結果分析エラー: {e}")
        import traceback

        traceback.print_exc()


def test_reward_calculation():
    """報酬計算の詳細分析"""
    print("\n=== 報酬計算詳細分析 ===")

    try:
        # 設定ファイル読み込み
        config_path = "config/v445/sac_v445.3_strong_selling_optimized.json"
        with open(config_path, "r") as f:
            config_data = json.load(f)

        # 環境設定 - training.environmentを使用
        env_config_dict = config_data["training"]["environment"]

        # データ読み込み
        data_path = config_data["training"]["data_config"]["data_path"]
        df = pd.read_csv(data_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # 環境初期化
        from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
        from ztb.trading.environment.utils.config import EnvironmentConfig

        env_config = EnvironmentConfig.from_dict(env_config_dict)

        env = HeavyTradingEnv(
            config=env_config,
            df=df,
            use_observation_normalization=True,
            use_reward_normalization=False,
        )

        obs, info = env.reset()

        # BUYアクションの報酬計算
        print("--- BUYアクション報酬計算 ---")
        obs, reward, terminated, truncated, info = env.step(ACTION_BUY)
        reward_components = getattr(
            env.reward_calculator, "_last_reward_components", {}
        )
        print(f"BUY報酬: {reward}")
        print(f"報酬コンポーネント: {reward_components}")

        # SELLアクションの報酬計算
        print("\n--- SELLアクション報酬計算 ---")
        obs, reward, terminated, truncated, info = env.step(ACTION_SELL)
        reward_components = getattr(
            env.reward_calculator, "_last_reward_components", {}
        )
        print(f"SELL報酬: {reward}")
        print(f"報酬コンポーネント: {reward_components}")

        # HOLDアクションの報酬計算（比較用）
        print("\n--- HOLDアクション報酬計算 ---")
        obs, reward, terminated, truncated, info = env.step(ACTION_HOLD)
        reward_components = getattr(
            env.reward_calculator, "_last_reward_components", {}
        )
        print(f"HOLD報酬: {reward}")
        print(f"報酬コンポーネント: {reward_components}")

        env.close()
        print("✅ 報酬計算詳細分析完了")

    except Exception as e:
        print(f"❌ 報酬計算詳細分析エラー: {e}")
        import traceback

        traceback.print_exc()


def check_reward_function():
    """報酬関数のSELLアクション設定を確認"""
    print("\n=== 報酬関数設定確認 ===")

    try:
        config_path = "config/v445/sac_v445.3_strong_selling_optimized.json"
        with open(config_path, "r") as f:
            config_data = json.load(f)

        env_config = config_data["training"]["environment"]

        # 報酬関連設定を確認
        reward_settings = env_config.get("reward_settings", {})
        action_bonuses = reward_settings.get("action_bonuses", {})
        print(f"action_bonuses: {action_bonuses}")

        # sell_action_bonusを確認
        sell_bonus = action_bonuses.get("sell_action_bonus", 0)
        print(f"sell_action_bonus: {sell_bonus}")

        if sell_bonus <= 0:
            print("⚠️ sell_action_bonusが0以下です")
        else:
            print("✅ sell_action_bonusが設定されています")

        # buy_action_bonusも確認
        buy_bonus = action_bonuses.get("buy_action_bonus", 0)
        print(f"buy_action_bonus: {buy_bonus}")

        # behavior_optimizationも確認
        behavior_opt = env_config.get("behavior_optimization", {})
        print(f"behavior_optimization: {behavior_opt}")

    except Exception as e:
        print(f"❌ 報酬関数設定確認エラー: {e}")
        import traceback

        traceback.print_exc()


def main():
    """メイン実行関数"""
    print("🔍 SELLアクション発生ゼロの根本原因調査")
    print("=" * 50)

    try:
        test_environment_sell_execution()
        test_reward_calculation()
        analyze_backtest_results()
        check_reward_function()

        print("\n" + "=" * 50)
        print("🎯 調査完了")
        print("📋 次のステップ:")
        print("1. 環境がSELLアクションを正しく実行できることを確認")
        print("2. バックテスト結果でSELLアクションが発生しているか確認")
        print("3. 報酬関数でSELLアクションが適切に報酬されているか確認")
        print("4. 必要に応じて報酬関数の根本的修正を実施")

    except Exception as e:
        print(f"❌ 調査中にエラーが発生: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
