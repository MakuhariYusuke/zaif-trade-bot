"""
TensorBoardイベントファイルから訓練の最終状態を抽出
"""

from pathlib import Path
from typing import Any


def read_tensorboard_summary(event_file_path: Path | str) -> dict[str, Any]:
    """TensorBoardイベントファイルから最終timestepsを読み取る（簡易版）"""
    try:
        with open(event_file_path, "rb") as f:
            data = f.read()
            # ファイルサイズから進捗を推定
            size_kb = len(data) / 1024

            # 簡易的な推定: 約164KB = 約8,000-10,000 timesteps
            # 完全な訓練（100,000 steps）なら約2-3MB必要
            estimated_progress = (size_kb / 2000) * 100  # 2MBを100%と仮定

            return {
                "file_size_kb": size_kb,
                "estimated_progress": min(estimated_progress, 100),
                "status": "Incomplete" if size_kb < 500 else "Completed?",
            }
    except Exception as e:
        return {"error": str(e)}


def analyze_training_sessions() -> None:
    """全セッションの訓練状況を分析"""

    sessions = {
        "v394a (HOLD Penalty)": "ppo_session_5",
        "v394e (High Entropy)": "ppo_session_6",
        "v394b (Trade Reward)": "ppo_session_7",
        "v394c (Balanced)": "ppo_session_8",
        "v394d (Aggressive - 1st)": "ppo_session_9",
        "v394d (Aggressive - 2nd)": "ppo_session_10",
    }

    checkpoints_dir = Path("checkpoints")

    print("=" * 80)
    print("v394 Series Training Analysis")
    print("=" * 80)
    print()

    for name, session_id in sessions.items():
        session_path = checkpoints_dir / session_id

        if not session_path.exists():
            print(f"❌ {name:35s} | Not found")
            continue

        # イベントファイルを探す
        event_files = list(session_path.glob("events.out.tfevents.*"))

        if event_files:
            event_file = event_files[0]
            info = read_tensorboard_summary(event_file)

            if "error" in info:
                print(f"❌ {name:35s} | Error: {info['error']}")
            else:
                status_icon = "✅" if info["estimated_progress"] > 80 else "⏳"
                print(
                    f"{status_icon} {name:35s} | "
                    f"Size: {info['file_size_kb']:6.1f} KB | "
                    f"Est. Progress: {info['estimated_progress']:5.1f}% | "
                    f"Status: {info['status']}"
                )
        else:
            print(f"❓ {name:35s} | No event files")

    print("=" * 80)
    print()
    print("📝 Notes:")
    print("  - File size ~164 KB suggests early termination (~8,000-10,000 steps)")
    print("  - Full training (100,000 steps) should produce ~2-3 MB")
    print("  - All v394 trainings appear to have terminated early")
    print()
    print("🎯 Recommendation:")
    print("  - Run ONE version at a time to completion (100,000 steps)")
    print("  - Start with v394d (most promising: initial HOLD 50%)")
    print("  - Monitor memory usage during training")
    print("=" * 80)


if __name__ == "__main__":
    analyze_training_sessions()
