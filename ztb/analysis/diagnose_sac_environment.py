"""
SAC環境診断スクリプト

Critic Loss爆発とent_coef上昇の根本原因を特定するため、
環境の詳細な動作を診断します。

診断項目:
1. 報酬分布（平均、分散、範囲、異常値）
2. 行動空間（連続→離散変換の妥当性）
3. 観測値スケール（各特徴量の統計）
4. エピソード動作（長さ、done条件）
5. Q値推定（訓練済みモデル使用時）
"""

import json
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Any

# プロジェクトルート追加
sys.path.insert(0, str(Path(__file__).parent))

from ztb.training.core.config_builder import ConfigBuilder
from ztb.training.algorithms.algorithm_factory import AlgorithmFactory
from ztb.utils.data_utils import load_csv_data_optimized


class EnvironmentDiagnostics:
    """環境診断クラス"""
    
    def __init__(self, config_path: str) -> None:
        """
        Args:
            config_path: 設定ファイルパス
        """
        self.config_path = config_path
        self.env = None
        self.diagnostics = {
            "config": config_path,
            "episodes": [],
            "summary": {}
        }
    
    def setup_environment(self) -> None:
        """環境セットアップ"""
        print(f"\n{'='*80}")
        print(f"環境診断開始: {self.config_path}")
        print(f"{'='*80}\n")
        
        # 設定ファイル読み込み
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # データセットパスとその他の設定を取得
        dataset_path = config_dict.get('data_path', 'btc_jpy_yahoo_real_dataset.csv')
        initial_balance = config_dict.get('initial_balance', 100000)
        algorithm = config_dict.get('algorithm', 'sac')
        
        # データ読み込み
        print(f"データ読み込み: {dataset_path}")
        df = load_csv_data_optimized(dataset_path)
        print(f"  データ件数: {len(df)}")
        
        # ConfigBuilderを使って統一設定を構築
        builder = ConfigBuilder(config_dict)
        unified_config = builder.build_unified_config()
        
        # AlgorithmFactoryを使って環境作成
        factory = AlgorithmFactory(
            algorithm=algorithm,
            config=unified_config,
            df=df
        )
        
        # 環境取得
        self.env = factory.create_environment()
        
        print(f"✓ 環境作成完了")
        print(f"  - Algorithm: {algorithm}")
        print(f"  - Action Space: {self.env.action_space}")
        print(f"  - Observation Space: {self.env.observation_space}")
        print(f"  - Initial Balance: {initial_balance:,.0f}")
        
    def run_episode(self, episode_num: int, num_steps: int = 100) -> Dict[str, Any]:
        """
        エピソード実行と診断
        
        Args:
            episode_num: エピソード番号
            num_steps: 最大ステップ数
            
        Returns:
            エピソード診断結果
        """
        print(f"\n--- Episode {episode_num} ---")
        
        obs = self.env.reset()
        
        episode_data = {
            "episode": episode_num,
            "steps": [],
            "observations": [],
            "actions_continuous": [],
            "actions_discrete": [],
            "rewards": [],
            "total_steps": 0,
            "done_reason": None
        }
        
        for step in range(num_steps):
            # ランダム行動（連続空間）
            action_continuous = self.env.action_space.sample()
            
            # 連続→離散変換（環境内部と同じロジック）
            from ztb.trading.environment.constants import continuous_to_discrete_action
            continuous_value = float(action_continuous[0])
            discrete_action = continuous_to_discrete_action(continuous_value)
            
            # Step実行
            next_obs, reward, done, info = self.env.step(action_continuous)
            
            # データ記録
            episode_data["observations"].append(obs.copy())
            episode_data["actions_continuous"].append(action_continuous.copy())
            episode_data["actions_discrete"].append(discrete_action)
            episode_data["rewards"].append(reward)
            
            # ステップ詳細
            step_detail = {
                "step": step,
                "action_continuous": action_continuous.tolist(),
                "reward": reward,
                "done": done,
                "portfolio_value": info.get("portfolio_value", 0),
                "position": info.get("position", 0),
            }
            episode_data["steps"].append(step_detail)
            
            # 進捗表示（10ステップごと）
            if (step + 1) % 10 == 0:
                print(f"  Step {step+1}/{num_steps} | "
                      f"Reward: {reward:+.4f} | "
                      f"Portfolio: {info.get('portfolio_value', 0):,.0f}")
            
            obs = next_obs
            
            if done:
                episode_data["done_reason"] = info.get("done_reason", "unknown")
                print(f"  Episode終了 at step {step+1}: {episode_data['done_reason']}")
                break
        
        episode_data["total_steps"] = len(episode_data["steps"])
        
        return episode_data
    
    def analyze_episode(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        エピソードデータ分析
        
        Args:
            episode_data: エピソードデータ
            
        Returns:
            分析結果
        """
        rewards = np.array(episode_data["rewards"])
        observations = np.array(episode_data["observations"])
        actions_continuous = np.array(episode_data["actions_continuous"])
        
        analysis = {
            "episode": episode_data["episode"],
            "total_steps": episode_data["total_steps"],
            "done_reason": episode_data["done_reason"],
            
            # 報酬統計
            "rewards": {
                "mean": float(np.mean(rewards)),
                "std": float(np.std(rewards)),
                "min": float(np.min(rewards)),
                "max": float(np.max(rewards)),
                "sum": float(np.sum(rewards)),
                "median": float(np.median(rewards)),
                "percentile_5": float(np.percentile(rewards, 5)),
                "percentile_95": float(np.percentile(rewards, 95)),
                "num_positive": int(np.sum(rewards > 0)),
                "num_negative": int(np.sum(rewards < 0)),
                "num_zero": int(np.sum(rewards == 0)),
            },
            
            # 観測値統計
            "observations": {
                "shape": observations.shape,
                "mean": observations.mean(axis=0).tolist(),
                "std": observations.std(axis=0).tolist(),
                "min": observations.min(axis=0).tolist(),
                "max": observations.max(axis=0).tolist(),
            },
            
            # 行動統計（連続）
            "actions_continuous": {
                "mean": actions_continuous.mean(axis=0).tolist(),
                "std": actions_continuous.std(axis=0).tolist(),
                "min": actions_continuous.min(axis=0).tolist(),
                "max": actions_continuous.max(axis=0).tolist(),
            },
        }
        
        # 離散行動統計
        if episode_data["actions_discrete"]:
            actions_discrete = np.array(episode_data["actions_discrete"])
            unique, counts = np.unique(actions_discrete, return_counts=True)
            analysis["actions_discrete"] = {
                "distribution": {int(u): int(c) for u, c in zip(unique, counts)},
                "unique_actions": len(unique),
            }
        
        return analysis
    
    def print_analysis(self, analysis: Dict[str, Any]) -> None:
        """分析結果を見やすく表示"""
        print(f"\n{'='*80}")
        print(f"Episode {analysis['episode']} 分析結果")
        print(f"{'='*80}")
        
        # 基本情報
        print(f"\n【基本情報】")
        print(f"  Total Steps: {analysis['total_steps']}")
        print(f"  Done Reason: {analysis['done_reason']}")
        
        # 報酬統計
        print(f"\n【報酬統計】")
        r = analysis["rewards"]
        print(f"  平均: {r['mean']:+.6f}")
        print(f"  標準偏差: {r['std']:.6f}")
        print(f"  範囲: [{r['min']:+.6f}, {r['max']:+.6f}]")
        print(f"  中央値: {r['median']:+.6f}")
        print(f"  5%tile: {r['percentile_5']:+.6f}")
        print(f"  95%tile: {r['percentile_95']:+.6f}")
        print(f"  合計: {r['sum']:+.6f}")
        print(f"  正の報酬: {r['num_positive']} ({r['num_positive']/len(r)*100:.1f}%)")
        print(f"  負の報酬: {r['num_negative']} ({r['num_negative']/len(r)*100:.1f}%)")
        print(f"  ゼロ: {r['num_zero']} ({r['num_zero']/len(r)*100:.1f}%)")
        
        # 観測値統計
        print(f"\n【観測値統計】")
        obs = analysis["observations"]
        print(f"  Shape: {obs['shape']}")
        print(f"  平均範囲: [{min(obs['mean']):.4f}, {max(obs['mean']):.4f}]")
        print(f"  標準偏差範囲: [{min(obs['std']):.4f}, {max(obs['std']):.4f}]")
        print(f"  最小値範囲: [{min(obs['min']):.4f}, {max(obs['min']):.4f}]")
        print(f"  最大値範囲: [{min(obs['max']):.4f}, {max(obs['max']):.4f}]")
        
        # 行動統計（連続）
        print(f"\n【連続行動統計】")
        act_cont = analysis["actions_continuous"]
        print(f"  平均: {act_cont['mean']}")
        print(f"  標準偏差: {act_cont['std']}")
        print(f"  範囲: [{act_cont['min']}, {act_cont['max']}]")
        
        # 行動統計（離散）
        if "actions_discrete" in analysis:
            print(f"\n【離散行動統計】")
            act_disc = analysis["actions_discrete"]
            print(f"  ユニーク行動数: {act_disc['unique_actions']}")
            print(f"  分布:")
            total = sum(act_disc["distribution"].values())
            for action, count in sorted(act_disc["distribution"].items()):
                print(f"    Action {action}: {count} ({count/total*100:.1f}%)")
    
    def generate_summary(self, all_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        全エピソードのサマリー生成
        
        Args:
            all_analyses: 全エピソード分析結果
            
        Returns:
            サマリー
        """
        # 報酬統計を集約
        all_rewards_mean = [a["rewards"]["mean"] for a in all_analyses]
        all_rewards_std = [a["rewards"]["std"] for a in all_analyses]
        all_rewards_min = [a["rewards"]["min"] for a in all_analyses]
        all_rewards_max = [a["rewards"]["max"] for a in all_analyses]
        
        summary = {
            "num_episodes": len(all_analyses),
            "total_steps": sum(a["total_steps"] for a in all_analyses),
            
            "rewards": {
                "mean_across_episodes": {
                    "mean": float(np.mean(all_rewards_mean)),
                    "std": float(np.std(all_rewards_mean)),
                    "min": float(np.min(all_rewards_mean)),
                    "max": float(np.max(all_rewards_mean)),
                },
                "std_across_episodes": {
                    "mean": float(np.mean(all_rewards_std)),
                    "std": float(np.std(all_rewards_std)),
                    "min": float(np.min(all_rewards_std)),
                    "max": float(np.max(all_rewards_std)),
                },
                "range_across_episodes": {
                    "min": float(np.min(all_rewards_min)),
                    "max": float(np.max(all_rewards_max)),
                },
            },
            
            # 問題検出
            "issues": []
        }
        
        # 問題検出ロジック
        avg_reward_mean = summary["rewards"]["mean_across_episodes"]["mean"]
        avg_reward_std = summary["rewards"]["std_across_episodes"]["mean"]
        reward_range_max = summary["rewards"]["range_across_episodes"]["max"]
        reward_range_min = summary["rewards"]["range_across_episodes"]["min"]
        
        # 報酬の分散が大きすぎる
        if avg_reward_std > 1.0:
            summary["issues"].append({
                "type": "high_reward_variance",
                "severity": "high",
                "value": avg_reward_std,
                "threshold": 1.0,
                "description": "報酬の標準偏差が大きすぎます。Critic Lossが爆発する原因の可能性。"
            })
        
        # 報酬の範囲が広すぎる
        if reward_range_max > 10.0 or reward_range_min < -10.0:
            summary["issues"].append({
                "type": "wide_reward_range",
                "severity": "high",
                "value": [reward_range_min, reward_range_max],
                "threshold": [-10.0, 10.0],
                "description": "報酬の範囲が設定されたクリッピング範囲を超えています。"
            })
        
        # 報酬がゼロに偏りすぎ
        avg_zero_pct = np.mean([a["rewards"]["num_zero"] / a["total_steps"] 
                                for a in all_analyses])
        if avg_zero_pct > 0.5:
            summary["issues"].append({
                "type": "too_many_zero_rewards",
                "severity": "medium",
                "value": avg_zero_pct,
                "threshold": 0.5,
                "description": "報酬がゼロに偏りすぎています。学習シグナルが不足している可能性。"
            })
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]) -> None:
        """サマリー表示"""
        print(f"\n{'='*80}")
        print(f"全体サマリー")
        print(f"{'='*80}")
        
        print(f"\n【集計】")
        print(f"  総エピソード数: {summary['num_episodes']}")
        print(f"  総ステップ数: {summary['total_steps']}")
        
        print(f"\n【報酬統計（全エピソード平均）】")
        r_mean = summary["rewards"]["mean_across_episodes"]
        print(f"  平均の平均: {r_mean['mean']:+.6f}")
        print(f"  平均の範囲: [{r_mean['min']:+.6f}, {r_mean['max']:+.6f}]")
        
        r_std = summary["rewards"]["std_across_episodes"]
        print(f"  標準偏差の平均: {r_std['mean']:.6f}")
        print(f"  標準偏差の範囲: [{r_std['min']:.6f}, {r_std['max']:.6f}]")
        
        r_range = summary["rewards"]["range_across_episodes"]
        print(f"  全体の範囲: [{r_range['min']:+.6f}, {r_range['max']:+.6f}]")
        
        # 問題検出
        if summary["issues"]:
            print(f"\n【⚠️ 検出された問題】")
            for i, issue in enumerate(summary["issues"], 1):
                print(f"\n  問題 {i}: {issue['type']} (重要度: {issue['severity']})")
                print(f"    値: {issue['value']}")
                print(f"    閾値: {issue['threshold']}")
                print(f"    説明: {issue['description']}")
        else:
            print(f"\n【✓ 重大な問題は検出されませんでした】")
    
    def save_diagnostics(self, output_path: str = "sac_environment_diagnostics.json") -> None:
        """診断結果を保存"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.diagnostics, f, indent=2, ensure_ascii=False)
        print(f"\n診断結果を保存: {output_path}")
    
    def run(self, num_episodes: int = 3, num_steps_per_episode: int = 100) -> None:
        """
        診断実行
        
        Args:
            num_episodes: 実行エピソード数
            num_steps_per_episode: エピソードあたりのステップ数
        """
        self.setup_environment()
        
        all_analyses = []
        
        for ep in range(1, num_episodes + 1):
            # エピソード実行
            episode_data = self.run_episode(ep, num_steps_per_episode)
            
            # 分析
            analysis = self.analyze_episode(episode_data)
            all_analyses.append(analysis)
            
            # 結果表示
            self.print_analysis(analysis)
            
            # 診断データに追加
            self.diagnostics["episodes"].append({
                "episode_data": episode_data,
                "analysis": analysis
            })
        
        # サマリー生成
        summary = self.generate_summary(all_analyses)
        self.diagnostics["summary"] = summary
        
        # サマリー表示
        self.print_summary(summary)
        
        # 保存
        self.save_diagnostics()
        
        print(f"\n{'='*80}")
        print(f"環境診断完了")
        print(f"{'='*80}\n")


def main() -> None:
    """メイン実行"""
    # v395g設定を使用（最新バージョン）
    config_path = "configs/sac_v395g_micro_reward.json"
    
    diagnostics = EnvironmentDiagnostics(config_path)
    
    # 3エピソード、各100ステップで診断
    diagnostics.run(num_episodes=3, num_steps_per_episode=100)


if __name__ == "__main__":
    main()
