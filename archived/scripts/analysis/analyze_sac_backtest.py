import json


def analyze_action_streaks(discrete_actions):
    """連続アクションのストリークを分析"""
    if not discrete_actions:
        return {}

    buy_streaks = []
    sell_streaks = []
    current_buy_streak = 0
    current_sell_streak = 0

    for action in discrete_actions:
        if action == 1:  # BUY
            current_buy_streak += 1
            if current_sell_streak > 0:
                sell_streaks.append(current_sell_streak)
                current_sell_streak = 0
        elif action == 2:  # SELL
            current_sell_streak += 1
            if current_buy_streak > 0:
                buy_streaks.append(current_buy_streak)
                current_buy_streak = 0
        else:  # HOLD
            if current_buy_streak > 0:
                buy_streaks.append(current_buy_streak)
                current_buy_streak = 0
            if current_sell_streak > 0:
                sell_streaks.append(current_sell_streak)
                current_sell_streak = 0

    # Add final streaks if any
    if current_buy_streak > 0:
        buy_streaks.append(current_buy_streak)
    if current_sell_streak > 0:
        sell_streaks.append(current_sell_streak)

    return {
        "buy_streaks": buy_streaks,
        "sell_streaks": sell_streaks,
        "max_buy_streak": max(buy_streaks) if buy_streaks else 0,
        "max_sell_streak": max(sell_streaks) if sell_streaks else 0,
        "avg_buy_streak": sum(buy_streaks) / len(buy_streaks) if buy_streaks else 0.0,
        "avg_sell_streak": sum(sell_streaks) / len(sell_streaks)
        if sell_streaks
        else 0.0,
        "total_buy_streak_count": len(buy_streaks),
        "total_sell_streak_count": len(sell_streaks),
    }


def main():
    with open("results/backtest_v420_hold_relaxed.json", "r") as f:
        data = json.load(f)

    print("=== SAC v420 Hold Relaxed バックテスト詳細分析 ===")
    print(f'総ステップ数: {data["total_steps"]}')
    print(f'初期ポートフォリオ: {data["initial_portfolio"]:,.0f} JPY')
    print(f'最終ポートフォリオ: {data["final_portfolio"]:,.0f} JPY')
    print(f'総リターン: {data["total_return_pct"]:.2f}%')
    print(f'総取引数: {data["total_trades"]}')
    print(f'総PnL: {data["total_pnl"]:,.0f} JPY')
    print(f'勝率: {data["win_rate"]:.1f}%')
    print(f'平均取引PnL: {data["avg_trade_pnl"]:.2f} JPY')
    print()

    print("=== 離散アクション分布 ===")
    actions = data["action_distribution"]
    total_actions = sum(actions.values())
    for action, count in actions.items():
        pct = count / total_actions * 100 if total_actions > 0 else 0
        action_name = {0: "HOLD", 1: "BUY", 2: "SELL"}[int(action)]
        print(f"  {action_name}: {count}回 ({pct:.1f}%)")
    print()

    if "continuous_action_stats" in data and data["continuous_action_stats"]:
        stats = data["continuous_action_stats"]
        print("=== 連続アクション値統計 ===")
        print(f'平均値: {stats["continuous_action_mean"]:.3f}')
        print(f'標準偏差: {stats["continuous_action_std"]:.3f}')
        print(f'最小値: {stats["continuous_action_min"]:.3f}')
        print(f'最大値: {stats["continuous_action_max"]:.3f}')
        print(f'中央値: {stats["continuous_action_median"]:.3f}')
        print(f'第1四分位数: {stats["continuous_action_q25"]:.3f}')
        print(f'第3四分位数: {stats["continuous_action_q75"]:.3f}')
        print()

        print("=== 連続アクション値ヒストグラム (主要な分布) ===")
        hist = stats["continuous_action_histogram"]
        bins = hist["bins"]
        counts = hist["counts"]
        percentages = hist["percentages"]

        # 主要なビン（割合が1%以上）のものだけ表示
        for i, (bin_range, count, pct) in enumerate(zip(bins, counts, percentages)):
            if pct >= 1.0:  # 1%以上のものだけ表示
                print(f"  {bin_range}: {count}回 ({pct:.1f}%)")
        print()

        # アクションストリーク分析
        if "action_streaks" in stats:
            streaks = stats["action_streaks"]
            print("=== 連続アクションストリーク分析 ===")
            print("BUYストリーク:")
            print(f'  最大連続BUY数: {streaks["max_buy_streak"]}回')
            print(f'  平均連続BUY数: {streaks["avg_buy_streak"]:.1f}回')
            print(f'  BUYストリーク総数: {streaks["total_buy_streak_count"]}回')
            print("SELLストリーク:")
            print(f'  最大連続SELL数: {streaks["max_sell_streak"]}回')
            print(f'  平均連続SELL数: {streaks["avg_sell_streak"]:.1f}回')
            print(f'  SELLストリーク総数: {streaks["total_sell_streak_count"]}回')
            print()

            # ストリーク長の分布を表示（主要なもの）
            print("=== BUYストリーク長分布 (上位5件) ===")
            buy_streak_counts = {}
            for streak in streaks["buy_streaks"]:
                buy_streak_counts[streak] = buy_streak_counts.get(streak, 0) + 1

            sorted_buy_streaks = sorted(
                buy_streak_counts.items(), key=lambda x: x[1], reverse=True
            )
            for length, count in sorted_buy_streaks[:5]:
                print(f"  {length}回連続BUY: {count}回発生")
            print()

            print("=== SELLストリーク長分布 (上位5件) ===")
            sell_streak_counts = {}
            for streak in streaks["sell_streaks"]:
                sell_streak_counts[streak] = sell_streak_counts.get(streak, 0) + 1

            sorted_sell_streaks = sorted(
                sell_streak_counts.items(), key=lambda x: x[1], reverse=True
            )
            for length, count in sorted_sell_streaks[:5]:
                print(f"  {length}回連続SELL: {count}回発生")
        else:
            print("アクションストリーク情報がありません。")
    else:
        print("連続アクション統計情報がありません。")


if __name__ == "__main__":
    main()
