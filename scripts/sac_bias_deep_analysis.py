#!/usr/bin/env python3
"""
Deep Analysis Script for SAC SELL Bias Investigation

Analyzes SAC model's action distribution, reward patterns, and curriculum learning stages
to identify the root cause of SELL bias.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
import json

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.trading.environment.components.reward_calculator import RewardCalculator
from ztb.utils.logging_utils import get_logger

class SACBiasAnalyzer:
    """Analyzes SAC model for SELL bias root causes."""

    def __init__(self, model_path: str, data_path: str, config_path: str):
        # Use the corrected model
        self.model_path = "checkpoints/sac_corrected_v1.zip"  # Use corrected model
        self.data_path = data_path
        self.config_path = config_path
        self.logger = get_logger(__name__)

        # Load components
        self.model = self._load_model()
        self.data = self._load_data()
        self.config = self._load_config()
        self.env = self._create_environment()
        self.reward_calculator = self._create_reward_calculator()

        # Analysis results
        self.action_distributions = defaultdict(Counter)
        self.reward_distributions = defaultdict(list)
        self.stage_analysis = defaultdict(dict)

    def _load_model(self) -> SAC:
        """Load the trained SAC model."""
        self.logger.info(f"Loading model from {self.model_path}")
        model = SAC.load(self.model_path)
        self.logger.info("Model loaded successfully")
        return model

    def _load_data(self) -> pd.DataFrame:
        """Load BTC/JPY data."""
        self.logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        self.logger.info(f"Loaded {len(df)} data points")
        return df

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        self.logger.info(f"Loading config from {self.config_path}")
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.logger.info("Config loaded successfully")
        return config

    def _create_environment(self) -> HeavyTradingEnv:
        """Create trading environment."""
        env_config = EnvironmentConfig.from_dict(self.config)
        env = HeavyTradingEnv(df=self.data, config=env_config)
        return env

    def _create_reward_calculator(self) -> RewardCalculator:
        """Create reward calculator."""
        env_config = EnvironmentConfig.from_dict(self.config)
        reward_settings = self.config.get('reward_settings', {})
        calculator = RewardCalculator(
            config=env_config,
            reward_settings=reward_settings,
            initial_portfolio_value=200000.0
        )
        return calculator

    def analyze_action_distribution(self, n_samples: int = 10000) -> Dict[str, Any]:
        """Analyze SAC model's action distribution across different scenarios."""
        self.logger.info(f"Analyzing action distribution with {n_samples} samples")

        results = {
            'continuous_actions': [],
            'discrete_actions': [],
            'action_frequencies': Counter(),
            'stage_distributions': defaultdict(Counter),
            'reward_patterns': defaultdict(list)
        }

        # Reset environment
        obs, info = self.env.reset()

        for i in range(n_samples):
            # Get action from model
            action_continuous, _ = self.model.predict(obs, deterministic=True)
            action_value = float(action_continuous[0])

            # Convert to discrete action using current thresholds
            buy_threshold = self.config.get('reward_settings', {}).get('action_threshold_buy', 0.05)
            sell_threshold = self.config.get('reward_settings', {}).get('action_threshold_sell', -0.3)

            if action_value > buy_threshold:
                discrete_action = 1  # BUY
            elif action_value < sell_threshold:
                discrete_action = 2  # SELL
            else:
                discrete_action = 0  # HOLD

            # Calculate reward for this action
            reward = self.reward_calculator.calculate_reward(
                action=discrete_action,
                current_price=self.env._resolve_price(),
                position=self.env.position,
                portfolio_value=self.env.portfolio_value,
                atr=self.env._resolve_atr(),
                transaction_cost=0.0,
                reward_scaling=self.config.get('reward_settings', {}).get('reward_scale', 8000.0),
                pnl=0.0,  # Simplified for analysis
                old_position=self.env.position,
                step=i,
                observation=obs,
                reward_history=[],
                portfolio_value_history=[]
            )

            # Record results
            results['continuous_actions'].append(action_value)
            results['discrete_actions'].append(discrete_action)
            results['action_frequencies'][discrete_action] += 1
            results['reward_patterns'][discrete_action].append(reward)

            # Get curriculum stage
            stage = getattr(self.env.config, 'curriculum_stage', 'default')
            results['stage_distributions'][stage][discrete_action] += 1

            # Take step in environment
            obs, reward, terminated, truncated, info = self.env.step(discrete_action)

            if terminated or truncated:
                obs, info = self.env.reset()

            if i % 1000 == 0:
                self.logger.info(f"Processed {i}/{n_samples} samples")

        return results

    def analyze_reward_bias(self, n_episodes: int = 100) -> Dict[str, Any]:
        """Analyze reward bias across different curriculum stages."""
        self.logger.info(f"Analyzing reward bias with {n_episodes} episodes")

        results = {
            'stage_rewards': defaultdict(lambda: defaultdict(list)),
            'action_rewards': defaultdict(lambda: defaultdict(list)),
            'pnl_distribution': defaultdict(list),
            'stage_transitions': []
        }

        for episode in range(n_episodes):
            obs, info = self.env.reset()
            episode_rewards = []
            episode_actions = []

            # Track initial stage
            initial_stage = getattr(self.env.config, 'curriculum_stage', 'default')
            current_stage = initial_stage

            done = False
            step = 0

            while not done and step < 1000:  # Limit episode length
                # Get action
                action_continuous, _ = self.model.predict(obs, deterministic=True)
                action_value = float(action_continuous[0])

                # Convert to discrete
                buy_threshold = self.config.get('reward_settings', {}).get('action_threshold_buy', 0.05)
                sell_threshold = self.config.get('reward_settings', {}).get('action_threshold_sell', -0.3)

                if action_value > buy_threshold:
                    discrete_action = 1
                elif action_value < sell_threshold:
                    discrete_action = 2
                else:
                    discrete_action = 0

                # Calculate reward
                reward = self.reward_calculator.calculate_reward(
                    action=discrete_action,
                    current_price=self.env._resolve_price(),
                    position=self.env.position,
                    portfolio_value=self.env.portfolio_value,
                    atr=self.env._resolve_atr(),
                    transaction_cost=0.0,
                    reward_scaling=self.config.get('reward_settings', {}).get('reward_scale', 8000.0),
                    pnl=self.env.total_pnl - (episode_rewards[-1] if episode_rewards else 0),
                    old_position=self.env.position,
                    step=step,
                    observation=obs,
                    reward_history=episode_rewards,
                    portfolio_value_history=[]
                )

                # Record results
                stage = getattr(self.env.config, 'curriculum_stage', 'default')
                results['stage_rewards'][stage][discrete_action].append(reward)
                results['action_rewards'][discrete_action][stage].append(reward)
                results['pnl_distribution'][stage].append(self.env.total_pnl)

                episode_rewards.append(reward)
                episode_actions.append(discrete_action)

                # Check stage transition
                if stage != current_stage:
                    results['stage_transitions'].append({
                        'from': current_stage,
                        'to': stage,
                        'step': step,
                        'episode': episode
                    })
                    current_stage = stage

                # Step environment
                obs, actual_reward, terminated, truncated, info = self.env.step(discrete_action)
                done = terminated or truncated
                step += 1

            if episode % 10 == 0:
                self.logger.info(f"Completed episode {episode}/{n_episodes}")

        return results

    def generate_report(self, action_analysis: Dict, reward_analysis: Dict) -> str:
        """Generate comprehensive analysis report."""
        report = []
        report.append("=" * 80)
        report.append("SAC SELL BIAS ROOT CAUSE ANALYSIS REPORT")
        report.append("=" * 80)

        # Action Distribution Analysis
        report.append("\n1. ACTION DISTRIBUTION ANALYSIS")
        report.append("-" * 40)

        total_actions = sum(action_analysis['action_frequencies'].values())
        report.append(f"Total samples analyzed: {total_actions}")

        for action, count in action_analysis['action_frequencies'].items():
            percentage = (count / total_actions) * 100
            action_name = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}[action]
            report.append(f"{action_name}: {count} ({percentage:.1f}%)")

        # Check for bias
        buy_count = action_analysis['action_frequencies'][1]
        sell_count = action_analysis['action_frequencies'][2]
        if sell_count > buy_count * 2:
            report.append("⚠️  SEVERE SELL BIAS DETECTED")
        elif sell_count > buy_count * 1.5:
            report.append("⚠️  MODERATE SELL BIAS DETECTED")

        # Continuous Action Analysis
        continuous_actions = np.array(action_analysis['continuous_actions'])
        report.append("\nContinuous Action Statistics:")
        report.append(f"  Mean: {continuous_actions.mean():.4f}")
        report.append(f"  Std: {continuous_actions.std():.4f}")
        report.append(f"  Min: {continuous_actions.min():.4f}")
        report.append(f"  Max: {continuous_actions.max():.4f}")
        report.append(f"  Median: {np.median(continuous_actions):.4f}")

        # Stage Analysis
        report.append("\n2. CURRICULUM STAGE ANALYSIS")
        report.append("-" * 40)

        for stage, distributions in action_analysis['stage_distributions'].items():
            report.append(f"\nStage: {stage}")
            stage_total = sum(distributions.values())
            for action, count in distributions.items():
                action_name = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}[action]
                percentage = (count / stage_total) * 100
                report.append(f"  {action_name}: {count} ({percentage:.1f}%)")

        # Reward Analysis
        report.append("\n3. REWARD PATTERN ANALYSIS")
        report.append("-" * 40)

        for action, rewards in action_analysis['reward_patterns'].items():
            if rewards:
                rewards_array = np.array(rewards)
                action_name = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}[action]
                report.append(f"\n{action_name} Action Rewards:")
                report.append(f"  Mean: {rewards_array.mean():.4f}")
                report.append(f"  Std: {rewards_array.std():.4f}")
                report.append(f"  Min: {rewards_array.min():.4f}")
                report.append(f"  Max: {rewards_array.max():.4f}")
                report.append(f"  Samples: {len(rewards)}")

        # Reward Bias Analysis
        report.append("\n4. REWARD BIAS ANALYSIS")
        report.append("-" * 40)

        for stage, action_rewards in reward_analysis['stage_rewards'].items():
            report.append(f"\nStage: {stage}")
            for action, rewards in action_rewards.items():
                if rewards:
                    action_name = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}[action]
                    avg_reward = np.mean(rewards)
                    report.append(f"  {action_name}: {avg_reward:.4f} (n={len(rewards)})")

        # Root Cause Analysis
        report.append("\n5. ROOT CAUSE HYPOTHESES")
        report.append("-" * 40)

        # Analyze continuous action distribution
        mean_action = continuous_actions.mean()
        if mean_action < -0.1:
            report.append("⚠️  SAC outputs predominantly negative actions (SELL-biased)")
        elif mean_action > 0.1:
            report.append("⚠️  SAC outputs predominantly positive actions (BUY-biased)")
        else:
            report.append("✓ SAC action distribution appears balanced")

        # Analyze reward patterns
        buy_rewards = action_analysis['reward_patterns'][1]
        sell_rewards = action_analysis['reward_patterns'][2]

        if buy_rewards and sell_rewards:
            buy_avg = np.mean(buy_rewards)
            sell_avg = np.mean(sell_rewards)

            if sell_avg > buy_avg * 1.2:
                report.append("⚠️  SELL actions receive significantly higher rewards")
            elif buy_avg > sell_avg * 1.2:
                report.append("⚠️  BUY actions receive significantly higher rewards")
            else:
                report.append("✓ BUY/SELL reward balance appears reasonable")

        # Threshold analysis
        buy_threshold = self.config.get('reward_settings', {}).get('action_threshold_buy', 0.05)
        sell_threshold = self.config.get('reward_settings', {}).get('action_threshold_sell', -0.3)

        actions_above_buy = sum(1 for x in continuous_actions if x > buy_threshold)
        actions_below_sell = sum(1 for x in continuous_actions if x < sell_threshold)

        buy_ratio = actions_above_buy / len(continuous_actions)
        sell_ratio = actions_below_sell / len(continuous_actions)

        report.append(f"\nThreshold Analysis:")
        report.append(f"  BUY threshold: {buy_threshold} (coverage: {buy_ratio:.1%})")
        report.append(f"  SELL threshold: {sell_threshold} (coverage: {sell_ratio:.1%})")

        if sell_ratio > buy_ratio * 2:
            report.append("⚠️  SELL threshold allows much more coverage than BUY threshold")

        report.append("\n" + "=" * 80)
        return "\n".join(report)

    def run_full_analysis(self) -> str:
        """Run complete analysis and generate report."""
        self.logger.info("Starting comprehensive SAC bias analysis")

        # Run action distribution analysis
        action_results = self.analyze_action_distribution(n_samples=5000)

        # Run reward bias analysis
        reward_results = self.analyze_reward_bias(n_episodes=50)

        # Generate report
        report = self.generate_report(action_results, reward_results)

        # Save detailed results
        output_file = "results/sac_bias_analysis_detailed.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'action_analysis': action_results,
                'reward_analysis': reward_results,
                'config': self.config
            }, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Detailed results saved to {output_file}")

        return report

def main():
    # Configuration
    model_path = "checkpoints/sac_session/sac_v404_extreme_win_rate_final.zip"
    data_path = "btc_jpy_real_dataset.csv"
    config_path = "config/sac_v404_config.json"

    try:
        analyzer = SACBiasAnalyzer(model_path, data_path, config_path)
        report = analyzer.run_full_analysis()

        print(report)

        # Save report to file
        with open("results/sac_bias_analysis_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)

        print("\nAnalysis completed successfully!")
        print("Report saved to: results/sac_bias_analysis_report.txt")
        print("Detailed results saved to: results/sac_bias_analysis_detailed.json")

    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()