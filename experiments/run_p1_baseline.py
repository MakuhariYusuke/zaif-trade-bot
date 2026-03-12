#!/usr/bin/env python3
"""
P1: 基準モデル作成 - PnLのみ報酬で基準を確立

89#に基づき、ペナルティなしのPnLのみ報酬で基準モデルを作成し、
「取引自体が利益か損失か」を判定する。
"""

from __future__ import annotations

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from experiments.p0_measurement_setup import EnvironmentMetrics, extract_environment_metrics
from ztb.rl.sac.sac_trainer import SACTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """実験設定"""
    experiment_id: str
    description: str
    reward_params: dict[str, Any]
    env_params: dict[str, Any]
    training_steps: int = 25000
    eval_episodes: int = 10
    seeds: list[int] | None = None
    
    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 123]


def create_p1_experiments() -> list[ExperimentConfig]:
    """P1実験設定を生成"""
    
    base_env_params = {
        "data_dir": str(project_root / "data"),
        "initial_balance": 100000.0,
        "max_steps": 10000,
        "transaction_cost": 0.001,  # 0.1%
    }
    
    experiments = [
        # P1-1: PnLのみ（ペナルティ全無効）
        ExperimentConfig(
            experiment_id="P1-1",
            description="PnLのみ（ペナルティ全無効）",
            reward_params={
                "alpha": 0.0,              # position change penalty OFF
                "beta": 0.0,               # holding time penalty OFF
                "gamma": 0.0,              # inventory risk OFF
                "fee_penalty_weight": 0.0, # extra fee penalty OFF
                "edge_penalty_rate": 0.0,  # edge penalty OFF
                "vol_floor_penalty": 0.0,  # vol floor penalty OFF
                "hold_ramp": 0.0,          # time decay OFF
            },
            env_params=base_env_params,
        ),
        
        # P1-2: PnL - 基本コスト（fee+slipのみ自然控除）
        ExperimentConfig(
            experiment_id="P1-2",
            description="PnL - 基本コスト（fee+slip自然控除）",
            reward_params={
                "alpha": 0.0,
                "beta": 0.0,
                "gamma": 0.0,
                "fee_penalty_weight": 0.0,  # 追加ペナルティなし（基本控除は残る）
                "edge_penalty_rate": 0.0,
                "vol_floor_penalty": 0.0,
                "hold_ramp": 0.0,
            },
            env_params=base_env_params,
        ),
        
        # P1-3: 現行設定（参考）
        ExperimentConfig(
            experiment_id="P1-3",
            description="現行設定（Day11再現）",
            reward_params={
                # デフォルト値を使用（明示的に指定しない）
            },
            env_params=base_env_params,
        ),
        
        # P1-4: コストゼロ環境でPnLのみ
        ExperimentConfig(
            experiment_id="P1-4",
            description="コストゼロ環境でPnLのみ",
            reward_params={
                "alpha": 0.0,
                "beta": 0.0,
                "gamma": 0.0,
                "fee_penalty_weight": 0.0,
                "edge_penalty_rate": 0.0,
                "vol_floor_penalty": 0.0,
                "hold_ramp": 0.0,
            },
            env_params={
                **base_env_params,
                "transaction_cost": 0.0,  # コストゼロ
            },
        ),
    ]
    
    return experiments


def run_single_experiment(
    config: ExperimentConfig,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """単一の実験を実行"""
    
    experiment_name = f"{config.experiment_id}_seed{seed}"
    logger.info(f"\n{'='*60}")
    logger.info(f"実験開始: {experiment_name}")
    logger.info(f"説明: {config.description}")
    logger.info(f"報酬パラメータ: {config.reward_params}")
    logger.info(f"{'='*60}")
    
    # 実験ディレクトリ
    exp_dir = output_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # トレーナー作成
    trainer = SACTrainer(
        output_dir=str(exp_dir),
        env_params=config.env_params,
        reward_params=config.reward_params,
        seed=seed,
    )
    
    # 学習前メトリクス
    logger.info("学習前評価...")
    pre_metrics = evaluate_and_extract_metrics(trainer, n_episodes=3)
    logger.info(f"学習前 Balance ROI: {pre_metrics['balance_roi']:.2f}%")
    
    # 学習実行
    logger.info(f"\n学習開始（{config.training_steps} steps）...")
    trainer.train(total_timesteps=config.training_steps)
    
    # 学習後メトリクス
    logger.info("\n学習後評価...")
    post_metrics = evaluate_and_extract_metrics(trainer, n_episodes=config.eval_episodes)
    
    # 結果サマリー
    result = {
        "experiment_id": config.experiment_id,
        "description": config.description,
        "seed": seed,
        "training_steps": config.training_steps,
        "pre_training": pre_metrics,
        "post_training": post_metrics,
        "improvement": {
            "balance_roi_diff": post_metrics["balance_roi"] - pre_metrics["balance_roi"],
            "gross_roi_diff": post_metrics["gross_roi"] - pre_metrics["gross_roi"],
        },
        "config": {
            "reward_params": config.reward_params,
            "env_params": config.env_params,
        },
    }
    
    # 結果出力
    logger.info("\n" + "=" * 60)
    logger.info(f"実験完了: {experiment_name}")
    logger.info(f"Gross ROI: {post_metrics['gross_roi']:+.2f}%")
    logger.info(f"Net ROI:   {post_metrics['net_roi']:+.2f}%")
    logger.info(f"Balance ROI: {post_metrics['balance_roi']:+.2f}%")
    logger.info(f"Cost Ratio: {post_metrics['cost_ratio']:.2f}%")
    logger.info(f"Total Trades: {post_metrics['total_trades']}")
    logger.info("=" * 60)
    
    # 結果保存
    result_file = exp_dir / "result.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    trainer.cleanup()
    
    return result


def evaluate_and_extract_metrics(
    trainer: SACTrainer,
    n_episodes: int = 10,
) -> dict[str, float]:
    """評価を実行しメトリクスを抽出"""
    
    # 評価環境を取得
    eval_env = trainer.eval_env
    model = trainer.model
    
    all_metrics = []
    
    for ep in range(n_episodes):
        obs = eval_env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            
            # VecEnvの場合
            if isinstance(done, np.ndarray):
                done = done[0]
        
        # エピソード終了後のメトリクス
        metrics = extract_environment_metrics(eval_env)
        all_metrics.append(metrics.to_dict())
    
    # 平均計算
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if m[key] is not None]
        if values:
            avg_metrics[key] = float(np.mean(values))
            avg_metrics[f"{key}_std"] = float(np.std(values))
    
    return avg_metrics


def run_p1_experiments() -> None:
    """P1実験群を実行"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "experiments" / "p1_baseline" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("P1: 基準モデル作成実験")
    logger.info(f"出力ディレクトリ: {output_dir}")
    logger.info("=" * 70)
    
    experiments = create_p1_experiments()
    all_results = []
    
    for config in experiments:
        for seed in config.seeds:
            try:
                result = run_single_experiment(config, seed, output_dir)
                all_results.append(result)
            except Exception as e:
                logger.error(f"実験失敗: {config.experiment_id} seed={seed}: {e}")
                import traceback
                traceback.print_exc()
    
    # 総合結果
    print_summary(all_results)
    
    # 結果保存
    summary_file = output_dir / "all_results.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n結果保存完了: {summary_file}")


def print_summary(results: list[dict[str, Any]]) -> None:
    """結果サマリーを出力"""
    
    logger.info("\n" + "=" * 90)
    logger.info("P1 実験結果サマリー")
    logger.info("=" * 90)
    
    # 実験IDごとに集約
    from collections import defaultdict
    by_experiment = defaultdict(list)
    for r in results:
        by_experiment[r["experiment_id"]].append(r)
    
    logger.info(f"{'ID':>6} | {'Description':<35} | {'Gross ROI':>10} | {'Net ROI':>10} | {'Balance ROI':>12} | {'Cost':>6} | {'Trades':>7}")
    logger.info("-" * 100)
    
    for exp_id in sorted(by_experiment.keys()):
        exp_results = by_experiment[exp_id]
        
        # 平均計算
        gross_rois = [r["post_training"]["gross_roi"] for r in exp_results]
        net_rois = [r["post_training"]["net_roi"] for r in exp_results]
        balance_rois = [r["post_training"]["balance_roi"] for r in exp_results]
        cost_ratios = [r["post_training"]["cost_ratio"] for r in exp_results]
        trades = [r["post_training"]["total_trades"] for r in exp_results]
        
        desc = exp_results[0]["description"][:35]
        
        logger.info(
            f"{exp_id:>6} | {desc:<35} | "
            f"{np.mean(gross_rois):>+9.2f}% | "
            f"{np.mean(net_rois):>+9.2f}% | "
            f"{np.mean(balance_rois):>+11.2f}% | "
            f"{np.mean(cost_ratios):>5.2f}% | "
            f"{int(np.mean(trades)):>7}"
        )
    
    logger.info("=" * 90)
    
    # 判断基準
    logger.info("\n判断基準:")
    if by_experiment.get("P1-1"):
        p1_1_roi = np.mean([r["post_training"]["balance_roi"] for r in by_experiment["P1-1"]])
        if p1_1_roi > 0:
            logger.info(f"✅ P1-1 (PnLのみ) > 0%: 取引自体は利益。コスト/ペナルティ調整で改善可能")
        else:
            logger.info(f"⚠️ P1-1 (PnLのみ) < 0%: 取引戦略自体が損失。学習設計見直し必要")


if __name__ == "__main__":
    run_p1_experiments()
