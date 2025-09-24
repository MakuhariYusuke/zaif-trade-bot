#!/usr/bin/env python3
# Profile Training Script for Heavy Trading RL Project
# 重特徴量取引RLプロジェクトのプロファイリング実行スクリプト

import os
import sys
import cProfile
import pstats
from pathlib import Path
import torch

# ローカルモジュールのインポート
sys.path.append(str(Path(__file__).parent.parent))

from src.trading.ppo_trainer import PPOTrainer

def profile_training():
    """トレーニングをプロファイル"""
    # 設定
    config = {
        'training': {
            'total_timesteps': 1000,
            'batch_size': 32,
            'n_steps': 1024,
            'gamma': 0.99,
            'learning_rate': 3e-4,
            'ent_coef': 0.01,
            'clip_range': 0.2,
            'n_epochs': 10,
            'gae_lambda': 0.95,
            'max_grad_norm': 0.5,
            'vf_coef': 0.5,
            'verbose': 0,
            'seed': 42,
        },
        'paths': {
            'log_dir': './logs/',
            'model_dir': './models/',
            'tensorboard_log': './tensorboard/',
        }
    }

    # データパス（サンプルデータを使用）
    data_path = 'generate_sample_data.py'  # 存在チェック
    if not Path(data_path).exists():
        print("Error: Sample data not found. Run generate_sample_data.py first.")
        return

    # プロファイラ開始
    profiler = cProfile.Profile()
    profiler.enable()

    # torch profiler
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:

        # トレーニング実行
        trainer = PPOTrainer(data_path, config)
        model = trainer.train()

    profiler.disable()

    # 結果出力
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # 上位20関数

    # ファイル保存
    with open('profile_results.txt', 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats('cumulative')
        stats.print_stats(20)

    print("Profile results saved to profile_results.txt")

if __name__ == '__main__':
    profile_training()