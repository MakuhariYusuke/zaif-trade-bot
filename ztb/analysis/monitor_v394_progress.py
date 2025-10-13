"""
v394シリーズの訓練進捗モニタリング
各セッションのログディレクトリから最新のAction分布を取得
"""

import os
import re
from pathlib import Path
from datetime import datetime


def parse_tensorboard_logs(session_dir: Path) -> dict[str, str]:
    """TensorBoardログから最新のメトリクスを取得（簡易版）"""
    # 実装は複雑なので、ここでは手動確認を推奨
    return {}


def monitor_training_progress() -> None:
    """訓練進捗を表示"""
    checkpoints_dir = Path("checkpoints")
    
    sessions = {
        "v394b (Trade Reward)": "ppo_session_7",
        "v394c (Balanced)": "ppo_session_8", 
        "v394d (Aggressive)": "ppo_session_9",
        "v394e (High Entropy)": "ppo_session_6",
    }
    
    print("="*80)
    print("v394 Series Training Progress")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    for name, session_id in sessions.items():
        session_path = checkpoints_dir / session_id
        if session_path.exists():
            # ログディレクトリのサイズで進捗を推定
            total_size = sum(f.stat().st_size for f in session_path.rglob('*') if f.is_file())
            print(f"✅ {name:30s} | Session: {session_id:15s} | Size: {total_size/1024:.1f} KB")
        else:
            print(f"⏸️  {name:30s} | Session: {session_id:15s} | Not started")
    
    print("="*80)
    print()
    print("📊 To view detailed metrics, use TensorBoard:")
    print("   tensorboard --logdir checkpoints")
    print()
    print("🔍 Or check terminal outputs for Action distribution:")
    print("   - Look for 'pan_action_counts' in training logs")
    print("   - Format: [HOLD, BUY, SELL]")
    print("="*80)


if __name__ == "__main__":
    monitor_training_progress()
