# PnL計算ロジック検証スクリプト
# 1M学習前確認項目の検証

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from ztb.trading.environment.environment import HeavyTradingEnv


def test_pnl_calculation():
    """PnL計算ロジックの検証"""
    print("=== 1. PnL計算ロジック検証 ===")

    # サンプルデータを読み込み
    data_path = Path("../data/features/2025/04/sample_04.parquet")
    df = pd.read_parquet(data_path)
    print(f"データ読み込み完了: {len(df)}行 x {len(df.columns)}列")

    # 環境設定
    config = {
        "reward_scaling": 1.0,
        "transaction_cost": 0.001,
        "max_position_size": 1.0,
        "risk_free_rate": 0.0,
    }

    # 環境の作成
    env = HeavyTradingEnv(df, config)
    print("環境作成完了")

    # サンプルエピソードの実行
    print("\n--- サンプルエピソード実行 ---")
    obs, info = env.reset()
    print(f"初期状態: position={env.position}, entry_price={env.entry_price}")

    # ステップバイステップでPnL計算を検証
    for step in range(min(10, len(df))):  # 最初の10ステップのみ
        # ランダムアクション（テスト用）
        action = np.random.choice([0, 1, 2])  # hold, buy, sell

        print(f"\nステップ {step}:")
        print(f"  アクション: {action} ({['hold', 'buy', 'sell'][action]})")
        print(f"  現在価格: {df.loc[env.current_step, 'price']:.2f}")
        print(f"  ポジション（実行前）: {env.position}")

        # ステップ実行前のPnL計算
        if env.position != 0:
            try:
                current_price_val = df.loc[env.current_step, "price"]
                entry_price_val = env.entry_price

                # 安全な型変換
                if isinstance(current_price_val, (int, float)):
                    current_price = float(current_price_val)
                else:
                    current_price = 0.0

                if isinstance(entry_price_val, (int, float)):
                    entry_price = float(entry_price_val)
                else:
                    entry_price = 0.0

                price_change = current_price - entry_price
                basic_pnl = float(env.position) * price_change
                transaction_cost = (
                    abs(float(env.position)) * entry_price * config["transaction_cost"]
                )
                total_pnl = basic_pnl - transaction_cost

                print(f"  価格変化: {price_change:.2f}")
                print(f"  基本PnL: {basic_pnl:.4f}")
                print(f"  取引コスト: {transaction_cost:.4f}")
                print(f"  合計PnL: {total_pnl:.4f}")
            except Exception as e:
                print(f"  PnL計算エラー: {e}")
                print(
                    f"  current_price: {df.loc[env.current_step, 'price']} (type: {type(df.loc[env.current_step, 'price'])})"
                )
                print(
                    f"  entry_price: {env.entry_price} (type: {type(env.entry_price)})"
                )

        # アクション実行
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"  ポジション（実行後）: {env.position}")
        print(f"  エントリー価格: {env.entry_price}")
        print(f"  リワード: {reward:.6f}")
        print(f"  累積PnL: {env.total_pnl:.4f}")

        if terminated or truncated:
            break

    print("\n最終状態:")
    print(f"  総取引回数: {env.trades_count}")
    print(f"  最終ポジション: {env.position}")
    print(f"  最終累積PnL: {env.total_pnl:.4f}")


def test_pnl_unit_logic():
    """損益単位の切り替えロジック検証"""
    print("\n=== 2. 損益単位切り替えロジック検証 ===")

    # 買い主体のシナリオ
    buy_trades = 7
    sell_trades = 3
    total_trades = buy_trades + sell_trades
    buy_ratio = buy_trades / total_trades
    sell_ratio = sell_trades / total_trades

    pnl_unit = "BTC" if buy_ratio > sell_ratio else "JPY"

    print(f"買い取引: {buy_trades}回 ({buy_ratio:.1%})")
    print(f"売り取引: {sell_trades}回 ({sell_ratio:.1%})")
    print(f"判定結果: {pnl_unit}単位 (買い主体={buy_ratio > sell_ratio})")

    # 売り主体のシナリオ
    buy_trades = 3
    sell_trades = 7
    total_trades = buy_trades + sell_trades
    buy_ratio = buy_trades / total_trades
    sell_ratio = sell_trades / total_trades

    pnl_unit = "BTC" if buy_ratio > sell_ratio else "JPY"

    print(f"\n買い取引: {buy_trades}回 ({buy_ratio:.1%})")
    print(f"売り取引: {sell_trades}回 ({sell_ratio:.1%})")
    print(f"判定結果: {pnl_unit}単位 (買い主体={buy_ratio > sell_ratio})")


def test_data_quality():
    """データ品質チェック"""
    print("\n=== 3. データ品質チェック ===")

    data_path = Path("../data/features/2025/04/sample_04.parquet")
    df = pd.read_parquet(data_path)

    print(f"データ形状: {df.shape}")
    print(f"データ期間: {df['ts'].min()} から {df['ts'].max()}")

    # 欠損値チェック
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    print("\n欠損値チェック:")
    print(f"  総欠損値: {total_nulls}")
    if total_nulls > 0:
        print("  列別欠損値:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"    {col}: {count}")

    # 外れ値チェック（価格の基本統計）
    price_stats = df["price"].describe()
    print("\n価格統計:")
    print(f"  平均: {price_stats['mean']:.2f}")
    print(f"  標準偏差: {price_stats['std']:.2f}")
    print(f"  最小: {price_stats['min']:.2f}")
    print(f"  最大: {price_stats['max']:.2f}")

    # 特徴量ごとの分布サマリー
    print("\n特徴量分布サマリー:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:5]:  # 最初の5つの数値列のみ
        if col != "ts":  # タイムスタンプ以外
            stats = df[col].describe()
            print(f"  {col}:")
            print(f"    平均: {stats['mean']:.2f}")
            print(f"    標準偏差: {stats['std']:.2f}")
            print(f"    歪度: {stats['50%']:.2f}")


def main():
    """メイン実行関数"""
    print("🔍 1M学習前確認項目検証開始")
    print("=" * 50)

    try:
        test_pnl_calculation()
        test_pnl_unit_logic()
        test_data_quality()

        print("\n" + "=" * 50)
        print("✅ 検証完了")

    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
