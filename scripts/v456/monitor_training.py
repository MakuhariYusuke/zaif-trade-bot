#!/usr/bin/env python3
"""
v456 訓練進捗ダッシュボード

リアルタイム訓練進捗を表示
"""

import sys
import time
from pathlib import Path
from typing import Optional

def get_training_metrics(log_file: str = "training_50k_log.txt") -> Optional[dict]:
    """訓練メトリクスを抽出"""
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        if not lines:
            return None
        
        # 最新のマイルストーン行を見つける
        for line in reversed(lines):
            if "Milestone" in line and "Avg Reward" in line:
                try:
                    parts = line.split("|")
                    step_str = parts[0].split()[-2].replace(",", "")
                    reward_str = parts[1].split(":")[-1].strip()
                    episodes_str = parts[2].split(":")[-1].strip()
                    
                    return {
                        'step': int(step_str),
                        'avg_reward': float(reward_str),
                        'episodes': int(episodes_str)
                    }
                except (ValueError, IndexError):
                    continue
        
        return None
    except FileNotFoundError:
        return None


def format_duration(seconds: float) -> str:
    """秒を人間が読みやすい形式に変換"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def draw_progress_bar(current: int, total: int, width: int = 40) -> str:
    """プログレスバーを描画"""
    percentage = current / total if total > 0 else 0
    filled = int(width * percentage)
    
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {percentage*100:.1f}%"


def clear_screen():
    """画面をクリア"""
    print("\033[2J\033[H", end="")


def display_dashboard(elapsed_time: float):
    """ダッシュボードを表示"""
    metrics = get_training_metrics()
    
    if metrics is None:
        print("⏳ Waiting for training to start...")
        print(f"   Elapsed: {format_duration(elapsed_time)}")
        return
    
    step = metrics['step']
    reward = metrics['avg_reward']
    episodes = metrics['episodes']
    target_steps = 50000
    
    # 速度計算
    speed_steps_per_sec = step / elapsed_time if elapsed_time > 0 else 0
    
    # 残り時間推定
    remaining_steps = target_steps - step
    estimated_remaining_seconds = remaining_steps / speed_steps_per_sec if speed_steps_per_sec > 0 else 0
    
    # 出力
    print("╔" + "═" * 68 + "╗")
    print("║" + "  v456 TRAINING PROGRESS DASHBOARD".ljust(69) + "║")
    print("╠" + "═" * 68 + "╣")
    
    # メトリクス
    print(f"║ Step Progress:  {step:>7,} / {target_steps:,}".ljust(69) + "║")
    print(f"║ {draw_progress_bar(step, target_steps)}".ljust(69) + "║")
    print("║".ljust(69) + "║")
    
    # 報酬
    print(f"║ Avg Reward:     {reward:>12.4f}".ljust(69) + "║")
    print(f"║ Total Episodes: {episodes:>12}".ljust(69) + "║")
    print("║".ljust(69) + "║")
    
    # 速度と時間
    print(f"║ Speed:          {speed_steps_per_sec:>10.2f} steps/sec".ljust(69) + "║")
    print(f"║ Elapsed Time:   {format_duration(elapsed_time):>20}".ljust(69) + "║")
    print(f"║ Est. Remaining: {format_duration(estimated_remaining_seconds):>20}".ljust(69) + "║")
    print(f"║ Est. Total:     {format_duration(elapsed_time + estimated_remaining_seconds):>20}".ljust(69) + "║")
    print("║".ljust(69) + "║")
    
    # ステータス
    completion_pct = (step / target_steps * 100) if target_steps > 0 else 0
    
    if step >= target_steps:
        status = "✅ COMPLETED"
    elif completion_pct >= 90:
        status = "🔥 ALMOST THERE"
    elif completion_pct >= 50:
        status = "⚡ HALFWAY THERE"
    elif completion_pct >= 25:
        status = "🚀 IN PROGRESS"
    else:
        status = "⏳ STARTING"
    
    print(f"║ Status:         {status:<52}".ljust(69) + "║")
    print("╚" + "═" * 68 + "╝")


def main():
    """メイン処理"""
    print("v456 Training Progress Dashboard")
    print("Press Ctrl+C to stop monitoring\n")
    
    start_time = time.time()
    
    try:
        while True:
            clear_screen()
            elapsed = time.time() - start_time
            display_dashboard(elapsed)
            
            # 1秒待機
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        return


if __name__ == "__main__":
    main()
