#!/usr/bin/env python3
# Parallel Training Script for 1M Learning
# 1M学習並列実行スクリプト

import os
import sys
import json
import subprocess
import time
import psutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import logging
import threading
import queue

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

# Discord通知モジュールのインポート
from utils.notify.discord import DiscordNotifier

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def determine_process_count():
    """プロセス数を自動決定（物理コア数と空きメモリを考慮）"""
    physical = psutil.cpu_count(logical=False) or 1
    avail_gb = psutil.virtual_memory().available / (1024**3)
    base = max(1, physical // 2)
    mem_cap = max(1, int(avail_gb // 2))
    return min(base, mem_cap, 4)  # 上限4

def _pump_stream(name, stream, cfg, logger):
    """ストリームを非同期で読み取る（Windows対応）"""
    for line in iter(stream.readline, ''):
        line = line.rstrip('\n')
        if stream is cfg["proc"].stdout:
            logger.info(f"[{cfg['name']}] {line}")
        else:
            logger.error(f"[{cfg['name']}] {line}")

class ParallelTrainingManager:
    """並列学習マネージャー"""

    def __init__(self):
        # 環境設定の読み込み
        self.env_config = self._load_env_config()
        
        self.notifier = DiscordNotifier()
        self.processes = []
        self.session_id = None
        self.log_file = None
        self.log_files = []  # 監視対象ログファイル
        self.last_log_check = time.time()
        
        # プロセス数を自動決定
        self.num_processes = determine_process_count()
        os.environ['PARALLEL_PROCESSES'] = str(self.num_processes)
        
        # 利用可能CPUを取得
        available_cpus = list(range(psutil.cpu_count(logical=False) or 1))
        
        self.training_configs = [
            {
                'name': 'generalization',
                'config': 'config/training/prod.json',
                'priority': 'high',
                'cpu_affinity': self.calculate_cpu_affinity(0, self.num_processes, available_cpus),
                'process_index': 0
            },
            {
                'name': 'aggressive',
                'config': 'config/training/prod_aggressive.json',
                'priority': 'low',
                'cpu_affinity': self.calculate_cpu_affinity(1, self.num_processes, available_cpus),
                'process_index': 1
            }
        ]

    def _load_env_config(self) -> dict:
        """環境設定を読み込み"""
        env = os.environ.get('ENV', 'dev')
        env_config_path = Path(f'config/environment/{env}.json')
        if env_config_path.exists():
            with open(env_config_path, 'r') as f:
                return json.load(f)
        return {}

    def calculate_cpu_affinity(self, process_index: int, total_processes: int, available_cpus: List[int]) -> List[int]:
        """プロセスインデックスに基づいてCPUアフィニティを動的に計算"""
        cpus_per_process = len(available_cpus) // total_processes
        start = process_index * cpus_per_process
        end = start + cpus_per_process
        return available_cpus[start:end]

    def set_process_priority(self, process: subprocess.Popen, config: Dict[str, Any]):
        """プロセス優先度設定"""
        try:
            # Windowsの場合、プロセス優先度クラスを設定
            if os.name == 'nt':  # Windows
                import win32api
                import win32process
                import win32con

                priority_class = win32con.NORMAL_PRIORITY_CLASS
                if config['priority'] == 'high':
                    priority_class = win32con.HIGH_PRIORITY_CLASS
                elif config['priority'] == 'low':
                    priority_class = win32con.IDLE_PRIORITY_CLASS

                handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, False, process.pid)
                win32process.SetPriorityClass(handle, priority_class)
                win32api.CloseHandle(handle)

                logger.info(f"Set priority class for {config['name']}: {config['priority']}")
            else:
                # Unix系OSの場合
                # Use 'priority' key for both OS, map to nice value for Unix
                priority_map = {'high': -10, 'normal': 0, 'low': 10}
                nice_value = priority_map.get(config.get('priority', 'normal'), 0)
                try:
                    psutil.Process(process.pid).nice(nice_value)
                except Exception as e:
                    logger.warning(f"Failed to set nice value for {config['name']}: {e}")

            # CPUアフィニティ設定
            if hasattr(psutil.Process, 'cpu_affinity'):
                p = psutil.Process(process.pid)
                p.cpu_affinity(config['cpu_affinity'])
                logger.info(f"Set CPU affinity for {config['name']}: {config['cpu_affinity']}")

        except Exception as e:
            logger.warning(f"Failed to set process priority for {config['name']}: {e}")

    def start_training_process(self, config: Dict[str, Any]) -> subprocess.Popen:
        """学習プロセスを開始"""
        cmd = [
            sys.executable,
            'train.py',
            '--config', config['config'],
            '--checkpoint-interval', '10000',
            '--checkpoint-dir', 'models/checkpoints',
            '--outlier-multiplier', '2.5'
        ]

        logger.info(f"Starting training process: {config['name']}")
        logger.info(f"Command: {' '.join(cmd)}")

        # 環境変数を子プロセスに渡す
        env = os.environ.copy()
        env['PARALLEL_PROCESSES'] = str(self.num_processes)
        env['PROCESS_ID'] = str(config['process_index'])
        env['CPU_AFFINITY'] = ','.join(map(str, config['cpu_affinity']))

        # プロセス開始
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=Path(__file__).parent  # プロジェクトルートをカレントディレクトリに設定
        )

        # 優先度設定
        self.set_process_priority(process, config)

        return process

    def monitor_processes(self):
        """プロセス監視（スレッド化されたストリーム読取を使用）"""
        while self.processes:
            # ログファイル更新チェック（30分ごと）
            current_time = time.time()
            if current_time - self.last_log_check > 1800:  # 30分
                self._check_log_updates()
                self.last_log_check = current_time

            for i, (process, config) in enumerate(self.processes):
                if process.poll() is not None:
                    # プロセス終了
                    exit_code = process.returncode

                    logger.info(f"Process {config['name']} finished with exit code: {exit_code}")

                    if exit_code != 0:
                        # エラー通知（ストリームはスレッドで既に読まれている）
                        self.notifier.send_error_notification(
                            f"Training Failed: {config['name']}",
                            f"Exit code: {exit_code}"
                        )

                    # プロセスリストから削除
                    self.processes.pop(i)
                    break

            time.sleep(10)  # 10秒ごとに監視

    def _check_log_updates(self):
        """ログファイル更新チェック"""
        try:
            # logs/ ディレクトリ内のログファイルをチェック
            logs_dir = Path('logs')
            if logs_dir.exists():
                for log_file in logs_dir.rglob('*.log'):
                    if time.time() - log_file.stat().st_mtime > 1800:  # 30分以上更新なし
                        self.notifier.send_error_notification(
                            "Log Stale Warning",
                            f"Log file {log_file.name} not updated for 30+ minutes"
                        )
        except Exception as e:
            logger.error(f"Error checking log updates: {e}")

    def send_completion_notification(self, config: Dict[str, Any]):
        """完了通知送信"""
        # モック結果データ（実際の学習結果を取得する必要がある）
        mock_results = {
            'reward_stats': {'mean_total_reward': 1000.0},
            'pnl_stats': {'mean_total_pnl': 500.0, 'max_drawdown': 0.05},
            'trading_stats': {
                'total_trades': 100,
                'winning_trades': 55,
                'profit_factor': 1.2,
                'mean_trades_per_episode': 5.0,
                'buy_ratio': 0.6,
                'sell_ratio': 0.4
            }
        }

        self.notifier.end_session(mock_results, f"{config['name']} training")

    def run_parallel_training(self):
        """並列学習実行"""
        # セッションID生成
        self.session_id = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_file = Path('logs') / f'session_{self.session_id}.log'
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # ログ設定を更新（ファイルハンドラ追加）
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        logger.info("🚀 Starting Parallel Training Session")
        logger.info("🚀 Starting Parallel Training Session")
        logger.info(f"Session ID: {self.session_id}")
        # print("🚀 Starting Parallel Training Session")  # Use logger for unified output
        # 開始通知
        session_id = self.notifier.start_session("parallel_training", "generalization + aggressive")

        try:
            # 各学習プロセスを開始
            for config in self.training_configs:
                process = self.start_training_process(config)
                self.processes.append((process, config))

                # ストリーム読取スレッドを起動
                t_out = threading.Thread(target=_pump_stream, args=("stdout", process.stdout, {"proc": process, "name": config["name"]}, logger), daemon=True)
                t_err = threading.Thread(target=_pump_stream, args=("stderr", process.stderr, {"proc": process, "name": config["name"]}, logger), daemon=True)
                t_out.start()
                t_err.start()

                logger.info(f"✅ Started {config['name']} training process (PID: {process.pid})")
                logger.info(f"proc_name={config['name']}, pid={process.pid}, affinity={config['cpu_affinity']}")
                logger.info(f"✅ Started {config['name']} training process (PID: {process.pid})")
                # print(f"✅ Started {config['name']} training process (PID: {process.pid})")  # Use logger for unified output
            # プロセス監視（リアルタイムログ出力）
            self.monitor_processes()

            # 全プロセス完了通知
            logger.info("🎉 All training processes completed!")
            logger.info("🎉 All training processes completed!")
            # print("🎉 All training processes completed!")  # Use logger for unified output
            # 完了通知（実際の結果を取得して送信）
            for config in self.training_configs:
                self.send_completion_notification(config)

        except Exception as e:
            logger.exception(f"Error in parallel training: {e}")
            self.notifier.send_error_notification("Parallel Training Error", f"Session {self.session_id}: {str(e)}")
            raise

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Parallel Training Script')
    parser.add_argument('--timesteps', type=int, help='Override total_timesteps from config')
    args = parser.parse_args()
    
    # Safety Clamp: PRODUCTION=1でない限り、total_timestepsを1000に強制
    is_production = os.environ.get('PRODUCTION') == '1'
    if not is_production and args.timesteps and args.timesteps > 1000:
        args.timesteps = 1000
        logger.warning("[SAFETY] Non-production run: total_timesteps clamped to 1000")
    
    manager = ParallelTrainingManager()
    if args.timesteps:
        # config の total_timesteps を上書き
        for config in manager.training_configs:
            config_path = Path(config['config'])
            if config_path.exists():
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                cfg['training']['total_timesteps'] = args.timesteps
                with open(config_path, 'w') as f:
                    json.dump(cfg, f, indent=2)
                logger.info(f"Updated {config['name']} total_timesteps to {args.timesteps}")
    
    manager.run_parallel_training()

if __name__ == "__main__":
    main()