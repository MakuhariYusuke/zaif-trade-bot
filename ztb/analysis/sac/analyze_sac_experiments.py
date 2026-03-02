#!/usr/bin/env python3
"""
SAC v434.2 実験結果分析スクリプト
複数実験の結果を分析し、最適な設定を特定
"""

import os
import re
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from ztb.analysis.common.plot_utils import save_plot
from ztb.io.json_io import read_json, write_json

def load_experiment_results(results_path: str) -> list[dict[str, Any]]:
    """実験結果を読み込み"""
    return read_json(results_path)

def extract_evaluation_reward(stdout: str) -> float:
    """stdoutから評価平均報酬を抽出"""
    # "評価完了 - 平均報酬: -8.78 (5エピソード)" のようなパターンを検索
    pattern = r"評価完了 - 平均報酬: ([+-]?\d*\.?\d+)"
    match = re.search(pattern, stdout)
    if match:
        return float(match.group(1))
    return 0.0

def analyze_hyperparameter_impact(results: list[dict[str, Any]]) -> dict[str, Any]:
    """ハイパーパラメータの影響を分析"""
    successful_results = [r for r in results if r.get("status") == "success"]

    if not successful_results:
        return {"error": "成功した実験がありません"}

    # 各実験に評価報酬を追加
    for exp in successful_results:
        exp["evaluation_avg_reward"] = extract_evaluation_reward(exp.get("stdout", ""))

    # 各ハイパーパラメータとパフォーマンスの相関を分析
    analysis = {
        "learning_rate_analysis": {},
        "batch_size_analysis": {},
        "gamma_analysis": {},
        "tau_analysis": {},
        "total_experiments": len(successful_results),
    }

    # 学習率の分析
    lr_groups = {}
    for exp in successful_results:
        lr = exp["hyperparams"]["learning_rate"]
        reward = exp["evaluation_avg_reward"]

        if lr not in lr_groups:
            lr_groups[lr] = []
        lr_groups[lr].append(reward)

    analysis["learning_rate_analysis"] = {
        lr: {
            "mean_reward": sum(rewards) / len(rewards),
            "count": len(rewards),
            "rewards": rewards,
        }
        for lr, rewards in lr_groups.items()
    }

    # バッチサイズの分析
    bs_groups = {}
    for exp in successful_results:
        bs = exp["hyperparams"]["batch_size"]
        reward = exp["evaluation_avg_reward"]

        if bs not in bs_groups:
            bs_groups[bs] = []
        bs_groups[bs].append(reward)

    analysis["batch_size_analysis"] = {
        bs: {
            "mean_reward": sum(rewards) / len(rewards),
            "count": len(rewards),
            "rewards": rewards,
        }
        for bs, rewards in bs_groups.items()
    }

    # 割引率の分析
    gamma_groups = {}
    for exp in successful_results:
        gamma = exp["hyperparams"]["gamma"]
        reward = exp["evaluation_avg_reward"]

        if gamma not in gamma_groups:
            gamma_groups[gamma] = []
        gamma_groups[gamma].append(reward)

    analysis["gamma_analysis"] = {
        gamma: {
            "mean_reward": sum(rewards) / len(rewards),
            "count": len(rewards),
            "rewards": rewards,
        }
        for gamma, rewards in gamma_groups.items()
    }

    # ターゲット更新率の分析
    tau_groups = {}
    for exp in successful_results:
        tau = exp["hyperparams"]["tau"]
        reward = exp["evaluation_avg_reward"]

        if tau not in tau_groups:
            tau_groups[tau] = []
        tau_groups[tau].append(reward)

    analysis["tau_analysis"] = {
        tau: {
            "mean_reward": sum(rewards) / len(rewards),
            "count": len(rewards),
            "rewards": rewards,
        }
        for tau, rewards in tau_groups.items()
    }

    return analysis

def find_best_hyperparameters(analysis: dict[str, Any]) -> dict[str, Any]:
    """最適なハイパーパラメータを特定"""
    recommendations = {}

    # 各パラメータで最も良い値を見つける
    if analysis.get("learning_rate_analysis"):
        best_lr = max(
            analysis["learning_rate_analysis"].items(),
            key=lambda x: x[1]["mean_reward"],
        )
        recommendations["best_learning_rate"] = {
            "value": best_lr[0],
            "mean_reward": best_lr[1]["mean_reward"],
            "confidence": best_lr[1]["count"],
        }

    if analysis.get("batch_size_analysis"):
        best_bs = max(
            analysis["batch_size_analysis"].items(), key=lambda x: x[1]["mean_reward"]
        )
        recommendations["best_batch_size"] = {
            "value": best_bs[0],
            "mean_reward": best_bs[1]["mean_reward"],
            "confidence": best_bs[1]["count"],
        }

    if analysis.get("gamma_analysis"):
        best_gamma = max(
            analysis["gamma_analysis"].items(), key=lambda x: x[1]["mean_reward"]
        )
        recommendations["best_gamma"] = {
            "value": best_gamma[0],
            "mean_reward": best_gamma[1]["mean_reward"],
            "confidence": best_gamma[1]["count"],
        }

    if analysis.get("tau_analysis"):
        best_tau = max(
            analysis["tau_analysis"].items(), key=lambda x: x[1]["mean_reward"]
        )
        recommendations["best_tau"] = {
            "value": best_tau[0],
            "mean_reward": best_tau[1]["mean_reward"],
            "confidence": best_tau[1]["count"],
        }

    return recommendations

def generate_analysis_report(results: list[dict[str, Any]], output_path: str):
    """分析レポート生成"""
    successful_results = [r for r in results if r.get("status") == "success"]

    if not successful_results:
        print("成功した実験がありません。分析をスキップします。")
        return

    # ハイパーパラメータ影響分析
    analysis = analyze_hyperparameter_impact(results)

    # 最適設定特定
    recommendations = find_best_hyperparameters(analysis)

    # 実験結果をDataFrameに変換
    experiment_data = []
    for exp in successful_results:
        backtest = exp.get("backtest_result", {})

        # stdoutから評価報酬を抽出
        stdout = exp.get("stdout", "")
        evaluation_reward = 0.0
        import re

        match = re.search(r"評価完了 - 平均報酬: (-?\d+\.?\d*)", stdout)
        if match:
            evaluation_reward = float(match.group(1))

        row = {
            "version": exp["version"],
            "description": exp["description"],
            "learning_rate": exp["hyperparams"]["learning_rate"],
            "batch_size": exp["hyperparams"]["batch_size"],
            "gamma": exp["hyperparams"]["gamma"],
            "tau": exp["hyperparams"]["tau"],
            "evaluation_avg_reward": evaluation_reward,
            "total_reward": backtest.get("total_reward", 0),
            "avg_episode_reward": backtest.get("avg_episode_reward", 0),
            "portfolio_return_pct": backtest.get("portfolio_return_pct", 0),
            "sharpe_ratio": backtest.get("sharpe_ratio", 0),
            "max_drawdown": backtest.get("max_drawdown", 0),
            "final_portfolio_value": backtest.get("final_portfolio_value", 0),
            "total_trades": backtest.get("total_trades", 0),
            "win_rate": backtest.get("win_rate", 0),
            "evaluation_episodes": backtest.get("evaluation_episodes", 0),
        }
        experiment_data.append(row)

    df = pd.DataFrame(experiment_data)

    # レポート作成
    report = {
        "summary": {
            "total_experiments": len(results),
            "successful_experiments": len(successful_results),
            "success_rate": len(successful_results) / len(results) * 100
            if results
            else 0,
        },
        "hyperparameter_analysis": analysis,
        "recommendations": recommendations,
        "experiment_results": experiment_data,
        "best_overall_experiment": None,
    }

    # 総合的に最も良い実験を特定（評価平均報酬が最も高いもの）
    if successful_results:
        best_overall = max(
            successful_results,
            key=lambda x: extract_evaluation_reward(x.get("stdout", "")),
        )
        best_reward = extract_evaluation_reward(best_overall.get("stdout", ""))
        backtest = best_overall.get("backtest_result", {})
        report["best_overall_experiment"] = {
            "version": best_overall["version"],
            "description": best_overall["description"],
            "hyperparams": best_overall["hyperparams"],
            "evaluation_avg_reward": best_reward,
            "avg_episode_reward": backtest.get("avg_episode_reward", 0),
            "portfolio_return_pct": backtest.get("portfolio_return_pct", 0),
            "backtest_result": backtest,
        }

    # CSV形式でも保存
    if not df.empty:
        csv_path = output_path.replace(".json", ".csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"実験結果CSV保存: {csv_path}")

    # レポート保存
    write_json(output_path, report, indent=2, ensure_ascii=False)

    print(f"分析レポート生成: {output_path}")

    # 結果表示
    print("\n=== 実験分析結果 ===")
    print(f"総実験数: {len(results)}")
    print(f"成功実験数: {len(successful_results)}")
    print(
        f"成功率: {len(successful_results) / len(results) * 100:.1f}%"
        if results
        else "0%"
    )

    if report["best_overall_experiment"]:
        best = report["best_overall_experiment"]
        print(f"\n最適実験: {best['version']} - {best['description']}")
        print(f"ハイパーパラメータ: {best['hyperparams']}")
        print(f"評価平均報酬: {best['evaluation_avg_reward']:.2f}")
        print(f"平均エピソード報酬: {best['avg_episode_reward']:.2f}")
        print(f"ポートフォリオリターン: {best['portfolio_return_pct']:.2f}%")
        backtest = best["backtest_result"]
        if backtest:
            print("バックテスト結果:")
            print(f"  - シャープレシオ: {backtest.get('sharpe_ratio', 0):.2f}")
            print(f"  - 最大ドローダウン: {backtest.get('max_drawdown', 0):.2f}%")
            print(f"  - 勝率: {backtest.get('win_rate', 0):.1%}")
            print(f"  - 評価エピソード数: {backtest.get('evaluation_episodes', 0)}")

    print("\n推奨ハイパーパラメータ:")
    for param, rec in recommendations.items():
        print(
            f"  - {param}: {rec['value']} (平均報酬: {rec['mean_reward']:.2f}, 実験数: {rec['confidence']})"
        )

def create_visualizations(
    results: list[dict[str, Any]], output_dir: str = "experiment_plots"
):
    """結果の可視化"""
    successful_results = [r for r in results if r.get("status") == "success"]

    if not successful_results:
        print("可視化対象のデータがありません")
        return

    # 各実験に評価報酬を追加
    for exp in successful_results:
        exp["evaluation_avg_reward"] = extract_evaluation_reward(exp.get("stdout", ""))

    os.makedirs(output_dir, exist_ok=True)

    # ハイパーパラメータ vs パフォーマンスの散布図
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("SAC v434.2 ハイパーパラメータ vs 評価平均報酬", fontsize=16)

    # 学習率 vs 評価平均報酬
    learning_rates = [exp["hyperparams"]["learning_rate"] for exp in successful_results]
    eval_rewards = [exp["evaluation_avg_reward"] for exp in successful_results]
    axes[0, 0].scatter(learning_rates, eval_rewards, alpha=0.6)
    axes[0, 0].set_xlabel("Learning Rate")
    axes[0, 0].set_ylabel("Evaluation Average Reward")
    axes[0, 0].set_title("Learning Rate vs Evaluation Reward")
    axes[0, 0].grid(True, alpha=0.3)

    # バッチサイズ vs 評価平均報酬
    batch_sizes = [exp["hyperparams"]["batch_size"] for exp in successful_results]
    axes[0, 1].scatter(batch_sizes, eval_rewards, alpha=0.6)
    axes[0, 1].set_xlabel("Batch Size")
    axes[0, 1].set_ylabel("Evaluation Average Reward")
    axes[0, 1].set_title("Batch Size vs Evaluation Reward")
    axes[0, 1].grid(True, alpha=0.3)

    # 割引率 vs 評価平均報酬
    gammas = [exp["hyperparams"]["gamma"] for exp in successful_results]
    axes[1, 0].scatter(gammas, eval_rewards, alpha=0.6)
    axes[1, 0].set_xlabel("Gamma (Discount Rate)")
    axes[1, 0].set_ylabel("Evaluation Average Reward")
    axes[1, 0].set_title("Gamma vs Evaluation Reward")
    axes[1, 0].grid(True, alpha=0.3)

    # ターゲット更新率 vs 評価平均報酬
    taus = [exp["hyperparams"]["tau"] for exp in successful_results]
    axes[1, 1].scatter(taus, eval_rewards, alpha=0.6)
    axes[1, 1].set_xlabel("Tau (Target Update Rate)")
    axes[1, 1].set_ylabel("Evaluation Average Reward")
    axes[1, 1].set_title("Tau vs Evaluation Reward")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "hyperparameter_analysis.png")
    save_plot(plot_path)
    plt.close()

    print(f"ハイパーパラメータ分析プロット保存: {plot_path}")

    # 評価報酬の分布
    plt.figure(figsize=(8, 6))
    plt.hist(eval_rewards, bins=10, alpha=0.7, edgecolor="black")
    plt.xlabel("Evaluation Average Reward")
    plt.ylabel("Frequency")
    plt.title("Evaluation Reward Distribution")
    plt.grid(True, alpha=0.3)

    reward_dist_plot_path = os.path.join(output_dir, "reward_distribution.png")
    save_plot(reward_dist_plot_path)
    plt.close()

    print(f"報酬分布プロット保存: {reward_dist_plot_path}")

def main():
    """メイン関数"""
    results_path = "sac_v434_2_experiment_results.json"

    if not os.path.exists(results_path):
        print(f"実験結果ファイルが見つかりません: {results_path}")
        print("まず run_sac_experiments.py を実行してください。")
        return 1

    print("=== SAC v434.2 実験結果分析開始 ===")

    # 結果読み込み
    results = load_experiment_results(results_path)

    # 分析レポート生成
    analysis_path = "sac_v434_2_analysis_report.json"
    generate_analysis_report(results, analysis_path)

    # 可視化
    create_visualizations(results)

    print("\n=== 分析完了 ===")
    return 0

if __name__ == "__main__":
    exit(main())
