"""
Phase 2 実市場データバックテスト

BTC/JPY実市場データを使用してPhase 1 + Phase 2最適化の性能を評価します。
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ztb.trading.backtest.adapters import ActionSignalGuideAdapter
from ztb.utils.performance_profiler import PerformanceProfiler


def load_btc_jpy_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """BTC/JPYデータを読み込み"""
    if file_path is None:
        # デフォルトのデータファイルを探す
        possible_paths = [
            "data/yahoo_finance/btc_jpy_1h_converted.csv",
            "data/btc_jpy_1h.csv",
            "btc_jpy_data.csv",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break

    if file_path is None or not os.path.exists(file_path):
        print(
            f"データファイルが見つからないため、サンプルデータを生成します: {file_path}"
        )
        return create_sample_btc_data()

    try:
        df = pd.read_csv(file_path)
        print(f"データを読み込みました: {file_path}, 行数: {len(df)}")

        # データの前処理
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
        elif "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)

        # 必要なカラムの確認と変換
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = []

        for col in required_cols:
            if col not in df.columns:
                # 大文字で探す
                upper_col = col.capitalize()
                if upper_col in df.columns:
                    df[col] = df[upper_col]
                else:
                    missing_cols.append(col)

        if missing_cols:
            print(f"必要なカラムが不足しています: {missing_cols}")
            print("サンプルデータを生成します")
            return create_sample_btc_data()

        # データ型の確認
        df = df.astype(
            {
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "volume": "float64",
            }
        )

        return df

    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        print("サンプルデータを生成します")
        return create_sample_btc_data()


def create_sample_btc_data(num_days: int = 365) -> pd.DataFrame:
    """BTC/JPYのサンプルデータを生成"""
    print(f"BTC/JPYサンプルデータを生成します: {num_days}日分")

    np.random.seed(42)  # 再現性のために

    # 2023年の日付範囲
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(num_days * 24)]

    # BTC価格の現実的な動きをシミュレート
    base_price = 3000000  # 300万円スタート

    # トレンドとボラティリティの変化を考慮
    prices = [base_price]
    trend_changes = [0, 100, 200, 300]  # トレンド変化ポイント

    for i in range(1, len(dates)):
        # 基本リターン（時間軸）
        base_return = 0.0001  # 0.01% per hour

        # トレンド成分
        trend_multiplier = 1.0
        for change_point in trend_changes:
            if i > change_point * 24:  # 日数から時間に変換
                trend_multiplier *= 0.9  # 徐々に弱まるトレンド

        # ボラティリティ（価格に応じて変化）
        volatility = 0.03 * (
            prices[-1] / base_price
        )  # 価格が上がるとボラティリティも上昇

        # ランダム成分
        random_return = np.random.normal(0, volatility)

        # 総リターン
        total_return = base_return * trend_multiplier + random_return

        new_price = prices[-1] * (1 + total_return)
        prices.append(max(new_price, 100000))  # 最低価格設定

    # OHLCVデータ作成
    data = []
    for i, price in enumerate(prices):
        # 1時間足のOHLC生成
        volatility_factor = np.random.uniform(0.005, 0.02)  # 0.5%-2%の範囲内変動
        high = price * (1 + volatility_factor)
        low = price * (1 - volatility_factor)
        open_price = prices[i - 1] if i > 0 else price

        # より現実的なvolume（価格と変動に応じて）
        base_volume = 1000000  # 100万単位
        volume_multiplier = (
            1 + (abs(price - open_price) / price) * 5
        )  # 変動が大きいほどvolume増加
        volume = base_volume * volume_multiplier * np.random.uniform(0.5, 1.5)

        data.append(
            {
                "timestamp": dates[i],
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "volume": volume,
            }
        )

    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)

    print(f"サンプルデータ生成完了: {len(df)}行")
    return df


def run_backtest(
    data: pd.DataFrame,
    adapter: ActionSignalGuideAdapter,
    initial_balance: float = 1000000,
) -> Dict[str, Any]:
    """バックテスト実行"""
    print(f"\nバックテスト開始: 初期残高 {initial_balance:,.0f}円")
    print(f"データ期間: {data.index[0]} から {data.index[-1]}")
    print(f"データ行数: {len(data)}")

    balance = initial_balance
    positions = []  # ポジション履歴
    trades = []  # トレード履歴
    current_position = None

    # パフォーマンス追跡
    peak_balance = balance
    max_drawdown = 0
    total_trades = 0
    winning_trades = 0

    for i in range(50, len(data), 5):  # 5時間ごとにチェック（処理速度考慮）
        current_data = data.iloc[: i + 1].copy()  # DataFrameとして確実に扱う
        current_price = current_data["close"].iloc[-1]

        # シグナル生成
        signal = adapter.generate_signal(current_data, 1 if current_position else 0)

        # デバッグ: シグナル内容確認
        if i % 1000 == 0:  # 1000回ごとにデバッグ出力
            print(f"Signal at index {i}: {signal}")

        # ポジション管理 - シグナル品質向上
        action = signal.get("action", signal.get("direction", 0))
        confidence = signal.get("confidence", 0.5)

        if isinstance(action, (int, float)):
            # direction値の場合の変換 - 閾値緩和
            if action > 0.05 and current_position is None:  # 0.1 → 0.05
                action = "buy"
            elif action < -0.05 and current_position is not None:  # -0.1 → -0.05
                action = "sell"
            else:
                action = "hold"
        elif isinstance(action, str):
            # actionが文字列の場合、コンフィデンスでフィルタリング
            if confidence < 0.6:  # コンフィデンスが低い場合はhold
                action = "hold"

        # 追加のシグナルフィルタリング
        if action in ["buy", "sell"]:
            # 市場ボラティリティチェック
            if len(current_data) > 20:
                recent_volatility = current_data["close"].pct_change().std() * np.sqrt(
                    24
                )  # 日次ボラティリティ
                if recent_volatility > 0.1:  # 過度なボラティリティ時はhold
                    action = "hold"
                    print(
                        f"High volatility detected ({recent_volatility:.3f}), holding position"
                    )

        if action == "buy" and current_position is None:
            # ロングポジションオープン
            position_size = (balance * 0.1) / current_price  # 残高の10%をBTCで購入
            if position_size > 0:
                current_position = {
                    "type": "long",
                    "entry_price": current_price,
                    "size": position_size,
                    "entry_time": current_data.index[-1],
                    "stop_loss": signal.get("stop_loss"),
                    "take_profit": signal.get("take_profit"),
                }
                balance -= position_size * current_price
                positions.append(current_position.copy())
                total_trades += 1

        elif action == "sell" and current_position is not None:
            # ポジションクローズ
            exit_price = current_price
            entry_price = current_position["entry_price"]
            position_size = current_position["size"]

            # P&L計算
            if current_position["type"] == "long":
                pnl = (exit_price - entry_price) * position_size
            else:
                pnl = (entry_price - exit_price) * position_size

            balance += position_size * exit_price + pnl

            # トレード記録
            trade = {
                "entry_time": current_position["entry_time"],
                "exit_time": current_data.index[-1],
                "type": current_position["type"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "size": position_size,
                "pnl": pnl,
                "return_pct": pnl / (entry_price * position_size),
            }
            trades.append(trade)

            if pnl > 0:
                winning_trades += 1

            current_position = None

        # ドローダウン計算
        if balance > peak_balance:
            peak_balance = balance
        current_drawdown = (peak_balance - balance) / peak_balance
        max_drawdown = max(max_drawdown, current_drawdown)

    # 最終結果計算
    total_return = balance - initial_balance
    total_return_pct = total_return / initial_balance

    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # 年率リターン計算（簡易版）
    total_days = (data.index[-1] - data.index[0]).total_seconds() / (60 * 60 * 24)
    annual_return = (
        (balance / initial_balance) ** (365 / total_days) - 1 if total_days > 0 else 0
    )

    # Sharpe Ratio計算（単純化）
    if trades:
        returns = [t["return_pct"] for t in trades]
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
    else:
        sharpe_ratio = 0

    results = {
        "initial_balance": initial_balance,
        "final_balance": balance,
        "total_return": total_return,
        "total_return_pct": total_return_pct,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "trades": trades,
        "positions": positions,
    }

    return results


def analyze_backtest_results(results: Dict[str, Any]):
    """バックテスト結果の分析"""
    print("\n" + "=" * 60)
    print("バックテスト結果分析")
    print("=" * 60)

    print(f"初期残高: {results['initial_balance']:,.0f}円")
    print(f"最終残高: {results['final_balance']:,.0f}円")
    print(f"総リターン: {results['total_return_pct']*100:+.2f}%")
    print(f"年率リターン: {results['annual_return']*100:.1f}%")
    print(f"勝率: {results['win_rate']:.1%}")
    print(f"最大ドローダウン: {results['max_drawdown']:.1%}")
    print(f"総トレード数: {results['total_trades']}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")

    # 月次リターン分析
    if results["trades"]:
        trades_df = pd.DataFrame(results["trades"])
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
        trades_df.set_index("entry_time", inplace=True)

        monthly_returns = trades_df.groupby(pd.Grouper(freq="ME"))["return_pct"].sum()

        print("\n月次リターン統計:")
        print(f"  平均月次リターン: {monthly_returns.mean():.2%}")
        print(f"  月次リターン標準偏差: {monthly_returns.std():.2%}")
        print(f"  最高月次リターン: {monthly_returns.max():.2%}")
        print(f"  最悪月次リターン: {monthly_returns.min():.2%}")

    # パフォーマンス評価
    print("\nパフォーマンス評価:")
    if results["sharpe_ratio"] > 1.0:
        print("  ✓ Sharpe Ratio > 1.0: 良好なリスク調整リターン")
    elif results["sharpe_ratio"] > 0.5:
        print("  △ Sharpe Ratio > 0.5: 許容可能なリスク調整リターン")
    else:
        print("  ✗ Sharpe Ratio ≦ 0.5: リスク調整リターンが不十分")

    if results["max_drawdown"] < 0.15:
        print("  ✓ 最大ドローダウン < 15%: リスク管理が機能")
    else:
        print("  ✗ 最大ドローダウン ≧ 15%: リスク管理の見直しが必要")

    if results["win_rate"] > 0.4:
        print("  ✓ 勝率 > 40%: シグナル品質が一定レベル")
    else:
        print("  △ 勝率 ≦ 40%: シグナル品質の改善が必要")


def main():
    """メイン実行関数"""
    print("Phase 2 実市場データバックテスト")
    print("=" * 80)

    # パフォーマンスプロファイリング
    profiler = PerformanceProfiler()

    with profiler.profile_context("Complete Backtest"):
        # データ読み込み
        data = load_btc_jpy_data()

        # アダプタ初期化
        print("\nActionSignalGuideAdapter初期化...")
        adapter = ActionSignalGuideAdapter()

        # バックテスト実行
        results = run_backtest(data, adapter)

        # 結果分析
        analyze_backtest_results(results)

    print("\n" + "=" * 80)
    print("Phase 2バックテスト完了")
    print("=" * 80)

    # 改善提案
    print("\n改善提案:")
    print("1. パラメータ最適化: リスク管理パラメータのチューニング")
    print("2. シグナル品質向上: 偽陽性削減と真陽性増加")
    print("3. ポートフォリオ最適化: 複数シグナルの相関考慮")
    print("4. リアルタイム適応: 市場環境変化への対応強化")


if __name__ == "__main__":
    main()
