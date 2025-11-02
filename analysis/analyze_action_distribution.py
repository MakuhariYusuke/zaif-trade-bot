import os

# ログファイルから行動分布を分析
import sys
from collections import defaultdict

log_file = sys.argv[1] if len(sys.argv) > 1 else "logs/training.log"
if os.path.exists(log_file):
    action_counts_first = defaultdict(int)
    action_counts_second = defaultdict(int)
    total_first = 0
    total_second = 0

    step_idx = 0
    # Read raw bytes and try multiple decodings (logs may be UTF-16/UTF-8)
    with open(log_file, "rb") as f:
        raw = f.read()
    text = None
    for enc in ("utf-8", "utf-16", "utf-16-le", "utf-16-be", "latin-1"):
        try:
            text = raw.decode(enc)
            if (
                "SAC continuous action:" in text
                or "sac continuous action:" in text.lower()
            ):
                break
        except Exception:
            text = None
    if text is None:
        # fallback: replace undecodable parts
        text = raw.decode("utf-8", errors="replace")

    for line in text.splitlines():
        if "SAC continuous action:" in line or "sac continuous action:" in line.lower():
            if "discrete action:" not in line and "discrete action" not in line:
                continue
                step_idx += 1
                # extract discrete action by searching for 'discrete action:' token
                try:
                    # find 'discrete action:' and parse the integer that follows
                    da_idx = line.index("discrete action:")
                    after = line[da_idx + len("discrete action:") :]
                    # after may look like ' 1 (BUY)'
                    discrete_str = after.strip().split()[0]
                    discrete_action = int(float(discrete_str))
                except Exception:
                    # fallback: try to parse by splitting
                    parts = line.split()
                    discrete_action = None
                    for i, p in enumerate(parts):
                        if p.startswith("discrete") and i + 2 < len(parts):
                            try:
                                discrete_action = int(float(parts[i + 2]))
                                break
                            except Exception:
                                continue
                    if discrete_action is None:
                        continue

                if step_idx <= 500:
                    action_counts_first[discrete_action] += 1
                    total_first += 1
                else:
                    action_counts_second[discrete_action] += 1
                    total_second += 1

    print("=== 行動分布分析 ===")
    print(f"最初の500ステップ (総数: {total_first}):")
    for action, count in sorted(action_counts_first.items()):
        pct = (count / total_first * 100) if total_first > 0 else 0
        action_name = (
            ["HOLD", "BUY", "SELL"][action] if action < 3 else f"ACTION_{action}"
        )
        print(f"  {action_name}: {count} ({pct:.1f}%)")

    print(f"\n後の500ステップ (総数: {total_second}):")
    for action, count in sorted(action_counts_second.items()):
        pct = (count / total_second * 100) if total_second > 0 else 0
        action_name = (
            ["HOLD", "BUY", "SELL"][action] if action < 3 else f"ACTION_{action}"
        )
        print(f"  {action_name}: {count} ({pct:.1f}%)")

    # 改善度の計算
    if total_first > 0 and total_second > 0:
        buy_first = action_counts_first.get(1, 0) / total_first
        buy_second = action_counts_second.get(1, 0) / total_second
        sell_first = action_counts_first.get(2, 0) / total_first
        sell_second = action_counts_second.get(2, 0) / total_second

        print("\n=== 改善分析 ===")
        print(
            f"BUY行動の変化: {buy_first:.3f} → {buy_second:.3f} (差: {buy_second - buy_first:.3f})"
        )
        print(
            f"SELL行動の変化: {sell_first:.3f} → {sell_second:.3f} (差: {sell_second - sell_first:.3f})"
        )

        if abs(buy_second - buy_first) > 0.05 or abs(sell_second - sell_first) > 0.05:
            print("✅ 行動分布に有意な変化が見られます")
        else:
            print("⚠️ 行動分布の変化が小さいか、一定です")
else:
    print("ログファイルが見つかりません")
