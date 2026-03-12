"""
v394シリーズの訓練完了確認とモデル移動
"""

import shutil
from pathlib import Path

def check_training_completion() -> None:
    """各v394バージョンの訓練完了状況を確認"""

    sessions = {
        "v394e (High Entropy)": "ppo_session_6",
        "v394b (Trade Reward)": "ppo_session_7",
        "v394c (Balanced)": "ppo_session_8",
        "v394d (Aggressive)": "ppo_session_9",
        "v394d (Aggressive - retry)": "ppo_session_10",
    }

    checkpoints_dir = Path("checkpoints")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("v394 Series Training Completion Check")
    print("=" * 80)
    print()

    for name, session_id in sessions.items():
        session_path = checkpoints_dir / session_id

        if not session_path.exists():
            print(f"❌ {name:35s} | Session: {session_id:15s} | Not found")
            continue

        # best_model.zipを探す
        best_model = session_path / "best_model.zip"
        final_model = session_path / "final_model.zip"

        model_found = None
        if best_model.exists():
            model_found = best_model
        elif final_model.exists():
            model_found = final_model

        if model_found:
            size_mb = model_found.stat().st_size / (1024 * 1024)
            print(
                f"✅ {name:35s} | Session: {session_id:15s} | Model: {model_found.name} ({size_mb:.1f} MB)"
            )

            # モデル名を推定
            version_name = None
            if "v394e" in name or session_id == "ppo_session_6":
                version_name = "ppo_v394e_high_entropy"
            elif "v394b" in name or session_id == "ppo_session_7":
                version_name = "ppo_v394b_trade_reward"
            elif "v394c" in name or session_id == "ppo_session_8":
                version_name = "ppo_v394c_balanced"
            elif "v394d" in name and (
                "session_9" in session_id or "session_10" in session_id
            ):
                version_name = "ppo_v394d_aggressive"

            if version_name:
                # モデルディレクトリ作成
                model_dest_dir = models_dir / version_name
                model_dest_dir.mkdir(parents=True, exist_ok=True)

                # モデルをコピー
                dest_path = model_dest_dir / "best_model.zip"
                if not dest_path.exists():
                    shutil.copy2(model_found, dest_path)
                    print(f"   → Copied to: {dest_path}")
                else:
                    print(f"   → Already exists: {dest_path}")
        else:
            # TensorBoardイベントファイルを確認
            event_files = list(session_path.glob("events.out.tfevents.*"))
            if event_files:
                total_size = sum(f.stat().st_size for f in event_files) / 1024
                print(
                    f"⏳ {name:35s} | Session: {session_id:15s} | Training... ({len(event_files)} events, {total_size:.1f} KB)"
                )
            else:
                print(f"❓ {name:35s} | Session: {session_id:15s} | Unknown status")

    print("=" * 80)

if __name__ == "__main__":
    check_training_completion()
