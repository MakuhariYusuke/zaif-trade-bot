#!/usr/bin/env python3
"""
v456 訓練完了待機・自動分析スクリプト

訓練が完了するまで待機し、完了後に自動的に分析を実行
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_training_progress(log_file: str = "training_50k_log.txt") -> tuple[Optional[int], Optional[int]]:
    """
    訓練ログの最新進捗を確認
    
    Returns:
        (last_step, total_steps): 最新ステップ数と目標ステップ数
    """
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        if not lines:
            return None, None
        
        # 最新のマイルストーン行を見つける
        last_milestone = None
        for line in reversed(lines):
            if "Milestone" in line and "Avg Reward" in line:
                try:
                    parts = line.split("|")
                    step_str = parts[0].split()[-2].replace(",", "")
                    last_step = int(step_str)
                    last_milestone = last_step
                    break
                except (ValueError, IndexError):
                    continue
        
        return last_milestone, 50000
    
    except FileNotFoundError:
        return None, None


def wait_for_training_completion(
    log_file: str = "training_50k_log.txt",
    check_interval: int = 30,
    timeout_minutes: int = 60
) -> bool:
    """
    訓練完了を待機
    
    Args:
        log_file: 訓練ログファイル
        check_interval: チェック間隔（秒）
        timeout_minutes: タイムアウト時間（分）
    
    Returns:
        bool: 訓練が完了したか
    """
    logger.info("=" * 70)
    logger.info("Waiting for v456 training completion...")
    logger.info("=" * 70)
    
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    
    last_progress = None
    stalled_count = 0
    
    while True:
        current_step, target_steps = check_training_progress(log_file)
        
        if current_step is None:
            logger.warning(f"Log file not accessible: {log_file}")
            time.sleep(check_interval)
            continue
        
        progress_pct = (current_step / target_steps * 100) if target_steps else 0
        
        logger.info(f"Progress: {current_step:,} / {target_steps:,} steps ({progress_pct:.1f}%)")
        
        # 完了判定
        if current_step >= target_steps:
            logger.info("✅ Training completed!")
            return True
        
        # スタール検出（進捗なし）
        if last_progress == current_step:
            stalled_count += 1
            if stalled_count > 5:  # 2.5分間進捗なし
                logger.warning("Training appears to be stalled")
                stalled_count = 0
        else:
            stalled_count = 0
        
        last_progress = current_step
        
        # タイムアウト判定
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            logger.error(f"Training timeout after {timeout_minutes} minutes")
            return False
        
        time.sleep(check_interval)


def run_analysis_pipeline():
    """分析パイプラインを実行"""
    logger.info("=" * 70)
    logger.info("Running analysis pipeline...")
    logger.info("=" * 70)
    
    scripts = [
        ("analyze_v456_training.py", "v456 訓練分析"),
        ("compare_v456_vs_v455.py", "v456 vs v455 比較"),
    ]
    
    for script_name, description in scripts:
        script_path = Path(__file__).parent / script_name
        
        if not script_path.exists():
            logger.warning(f"Script not found: {script_path}")
            continue
        
        logger.info(f"\n▶️  Running: {description} ({script_name})")
        logger.info("-" * 70)
        
        try:
            result = subprocess.run(
                ["python", str(script_path)],
                capture_output=False,
                timeout=600  # 10分タイムアウト
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {description} completed")
            else:
                logger.error(f"❌ {description} failed with code {result.returncode}")
        
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {description} timed out")
        except Exception as e:
            logger.error(f"❌ Error running {description}: {e}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Analysis pipeline completed!")
    logger.info("=" * 70)


def main():
    """メイン処理"""
    # 訓練完了を待機
    training_completed = wait_for_training_completion(
        log_file="training_50k_log.txt",
        check_interval=30,  # 30秒ごとにチェック
        timeout_minutes=60  # 60分でタイムアウト
    )
    
    if training_completed:
        # 分析パイプラインを実行
        run_analysis_pipeline()
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ Complete workflow finished!")
        logger.info("=" * 70)
        logger.info("\nOutput files:")
        logger.info("  - analysis_results/v456/v456_training_analysis.png")
        logger.info("  - analysis_results/v456/v456_training_analysis_report.md")
        logger.info("  - analysis_results/v456/v456_training_analysis.json")
        logger.info("  - analysis_results/comparison/v456_vs_v455_comparison.png")
        logger.info("  - analysis_results/comparison/v456_vs_v455_report.md")
        logger.info("  - analysis_results/comparison/comparison_results.json")
    else:
        logger.error("❌ Training did not complete successfully")


if __name__ == "__main__":
    main()
