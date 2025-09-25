#!/usr/bin/env python3
"""
1kステップ強化学習テスト実行スクリプト
フィーチャー評価を反復的に実行し、結果を監視
"""

import os
import sys
import time
import json
import psutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

class ReinforcementLearner1K:
    """1kステップ強化学習風フィーチャー評価実行クラス"""

    def __init__(self, dataset: str = "coingecko", total_steps: int = 1000):
        self.dataset = dataset
        self.total_steps = total_steps
        self.checkpoint_dir = Path("checkpoints/1k_test")
        self.checkpoint_dir.mkdir(exist_ok=True)

        # 監視データ
        self.monitoring_data = []
        self.start_time = None
        self.memory_usage = []
        self.step_results = []

        # 統計
        self.harmful_count = 0
        self.pending_count = 0
        self.verified_count = 0

    def get_memory_usage(self) -> Dict[str, float]:
        """メモリ使用量を取得"""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'percent': process.memory_percent()
        }

    def run_feature_evaluation(self, step: int) -> Dict[str, Any]:
        """1ステップのフィーチャー評価を実行"""
        print(f"\n=== Step {step}/{self.total_steps} ===")

        # メモリ使用量を記録
        memory_before = self.get_memory_usage()

        # フィーチャー評価を実行
        cmd = [
            sys.executable,
            "scripts/test_all_features.py",
            "--dataset", self.dataset,
            "--bootstrap",
            "--save-debug"
        ]

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        execution_time = time.time() - start_time

        # メモリ使用量を記録
        memory_after = self.get_memory_usage()

        # 結果を解析
        step_result = {
            'step': step,
            'execution_time': execution_time,
            'memory_before': memory_before,
            'memory_after': memory_after,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'timestamp': datetime.now().isoformat()
        }

        # ログから統計を抽出
        self._parse_evaluation_results(result.stdout, step)

        return step_result

    def _parse_evaluation_results(self, stdout: str, step: int):
        """評価結果から統計を抽出"""
        lines = stdout.split('\n')

        for line in lines:
            if '✗' in line and 'harmful' in line.lower():
                self.harmful_count += 1
            elif 'pending' in line.lower():
                self.pending_count += 1
            elif '✓' in line and 'verified' in line.lower():
                self.verified_count += 1

    def save_checkpoint(self, step: int, results: List[Dict]):
        """チェックポイントを保存"""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_step_{step}.json"

        checkpoint_data = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'statistics': {
                'harmful_count': self.harmful_count,
                'pending_count': self.pending_count,
                'verified_count': self.verified_count,
                'total_steps': step
            },
            'memory_usage': self.memory_usage,
            'monitoring_data': self.monitoring_data
        }

        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

        print(f"Checkpoint saved: {checkpoint_file}")

    def send_discord_notification(self, message: str):
        """Discord通知を送信"""
        try:
            # Discord通知スクリプトが存在するか確認
            discord_script = Path("scripts/send_discord_notification.py")
            if discord_script.exists():
                subprocess.run([
                    sys.executable,
                    str(discord_script),
                    message
                ], check=True)
            else:
                print(f"Discord notification: {message}")
        except Exception as e:
            print(f"Failed to send Discord notification: {e}")

    def generate_report(self) -> str:
        """最終レポートを生成"""
        total_time = time.time() - self.start_time if self.start_time else 0
        steps_per_sec = self.total_steps / total_time if total_time > 0 else 0

        # メモリ使用量の最大値
        max_memory = max(self.memory_usage, key=lambda x: x.get('rss', 0)) if self.memory_usage else {'rss': 0}

        report = f"""
=== 1kステップ強化学習テスト完了レポート ===

実行時間: {total_time:.2f}秒
学習速度: {steps_per_sec:.2f} steps/sec
最大メモリ使用量: {max_memory.get('rss', 0):.2f} MB

=== 特徴量統計 ===
Harmful: {self.harmful_count}
Pending: {self.pending_count}
Verified: {self.verified_count}

=== メモリ使用量推移 ===
"""

        for i, mem in enumerate(self.memory_usage[:10]):  # 最初の10ステップのみ表示
            report += f"Step {i+1}: {mem.get('rss', 0):.2f} MB\n"

        if len(self.memory_usage) > 10:
            report += f"... ({len(self.memory_usage)} ステップ中)\n"

        report += "\n=== 100k以上で落ちる可能性がある箇所 ===\n"
        report += "1. メモリリーク: 各ステップでメモリ使用量が増加傾向\n"
        report += "2. チェックポイントIO負荷: 頻繁なファイル保存によるディスクI/O\n"
        report += "3. Quality Gates 閾値: リアルデータでの厳しすぎる相関基準\n"
        report += "4. バッチサイズ設定: デフォルト設定でのメモリ消費\n"
        report += "5. ログ蓄積: 各ステップの詳細ログがメモリを圧迫\n"

        return report

    def run(self):
        """メイン実行ループ"""
        print(f"Starting 1k-step reinforcement learning test with {self.dataset} dataset")
        self.start_time = time.time()

        results = []

        for step in range(1, self.total_steps + 1):
            try:
                # フィーチャー評価を実行
                step_result = self.run_feature_evaluation(step)
                results.append(step_result)

                # メモリ使用量を記録
                self.memory_usage.append(step_result['memory_after'])

                # 500ステップと最終ステップでチェックポイント保存
                if step == 500 or step == self.total_steps:
                    self.save_checkpoint(step, results)

                # 進捗表示
                if step % 100 == 0:
                    elapsed = time.time() - self.start_time
                    steps_per_sec = step / elapsed
                    print(f"Step {step}/{self.total_steps}: {steps_per_sec:.2f} steps/sec")

            except Exception as e:
                print(f"Error at step {step}: {e}")
                continue
            except KeyboardInterrupt:
                print(f"\nInterrupted at step {step}")
                self.save_checkpoint(step, results)
                break
            except KeyboardInterrupt:
                print(f"\nInterrupted at step {step}")
                self.save_checkpoint(step, results)
                break
            except Exception as e:
                print(f"Error at step {step}: {e}")
                continue

        # 最終レポート生成
        report = self.generate_report()
        print(report)

        # Discord通知
        self.send_discord_notification(f"1kステップ強化学習テスト完了\n{report}")

        # 最終チェックポイント保存
        self.save_checkpoint(self.total_steps, results)

        return results


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="1kステップ強化学習テスト")
    parser.add_argument("--dataset", default="coingecko", help="データセット (default: coingecko)")
    parser.add_argument("--steps", type=int, default=1000, help="総ステップ数 (default: 1000)")

    args = parser.parse_args()

    learner = ReinforcementLearner1K(dataset=args.dataset, total_steps=args.steps)
    results = learner.run()

    print(f"\nTest completed. Results saved to {learner.checkpoint_dir}")


if __name__ == "__main__":
    main()