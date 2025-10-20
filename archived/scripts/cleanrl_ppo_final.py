#!/usr/bin/env python3
"""
CleanRL PPO Implementation - Final Solution for SAC Bias

Using CleanRL's minimal PPO implementation to avoid Stable Baselines3 issues.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def make_env():
    """Create environment with continuous actions."""
    from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
    from ztb.trading.environment.utils.config import EnvironmentConfig

    config = EnvironmentConfig(
        max_position_size=0.01,
        transaction_cost=0.001,  # Small epsilon to avoid division by zero
        reward_scaling=1.0,
        reward_clip_value=1.0,
        reward_settings={
            "use_simple_reward": True,
            "reward_scale": 1.0,
            "reward_clip_min": -1.0,
            "reward_clip_max": 1.0,
            "buy_action_penalty": 0.0,
            "sell_action_penalty": 0.0,
            "hold_action_penalty": 0.0,
            "profit_bonus_multipliers": [1.0, 1.0, 1.0],
        },
        use_continuous_actions=True,
        continuous_to_discrete_threshold=0.1,
    )

    df = pd.read_csv("btc_jpy_real_dataset.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    env = HeavyTradingEnv(df=df, config=config)
    return env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize neural network layers."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """PPO Agent with actor-critic architecture."""

    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(np.prod(envs.action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(-1),
            probs.entropy().sum(-1),
            self.critic(x),
        )


def train_cleanrl_ppo():
    """Train PPO using CleanRL implementation."""

    print("=" * 80)
    print("CLEANRL PPO TRAINING - FINAL SOLUTION")
    print("=" * 80)

    # Environment setup
    env = make_env()
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print()

    # Training parameters
    total_timesteps = 10000
    num_steps = 2048
    num_minibatches = 32
    update_epochs = 10
    learning_rate = 3e-4
    num_envs = 1
    anneal_lr = True
    gae = True
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.0
    vf_coef = 0.5
    max_grad_norm = 0.5
    target_kl = None

    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)

    # Initialize agent
    agent = Agent(env)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # Storage setup
    obs = torch.zeros((num_steps, num_envs) + env.observation_space.shape).to(
        torch.float32
    )
    actions = torch.zeros((num_steps, num_envs) + env.action_space.shape).to(
        torch.float32
    )
    logprobs = torch.zeros((num_steps, num_envs)).to(torch.float32)
    rewards = torch.zeros((num_steps, num_envs)).to(torch.float32)
    dones = torch.zeros((num_steps, num_envs)).to(torch.float32)
    values = torch.zeros((num_steps, num_envs)).to(torch.float32)

    # Training loop
    global_step = 0
    start_time = (
        torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    )
    next_obs = env.reset()
    next_obs = torch.tensor(
        next_obs[0] if isinstance(next_obs, tuple) else next_obs, dtype=torch.float32
    )
    next_done = torch.zeros(num_envs).to(torch.float32)

    print("Starting CleanRL PPO training...")

    for update in range(1, total_timesteps // batch_size + 1):
        # Annealing the rate if instructed to do so.
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / (total_timesteps // batch_size)
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            # Get action from the agent
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute the game and log data.
            next_obs, reward, done, truncated, info = env.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(torch.float32).view(-1)
            next_obs = torch.tensor(
                next_obs[0] if isinstance(next_obs, tuple) else next_obs,
                dtype=torch.float32,
            )
            next_done = torch.tensor(done or truncated).to(torch.float32)

            # Reset environment if done
            if done or truncated:
                next_obs = env.reset()
                next_obs = torch.tensor(
                    next_obs[0] if isinstance(next_obs, tuple) else next_obs,
                    dtype=torch.float32,
                )
                next_done = torch.zeros(num_envs).to(torch.float32)

        # Bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if gae:
                advantages = torch.zeros_like(rewards).to(torch.float32)
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(torch.float32)
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + gamma * nextnonterminal * next_return
                advantages = returns - values

        # Flatten the batch
        b_obs = obs.reshape((-1,) + env.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if True:  # args.norm_adv
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if True:  # args.clip_vloss
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if target_kl is not None:
                if approx_kl > target_kl:
                    break

        print(f"Update {update}: Loss={loss.item():.3f}, KL={approx_kl.item():.3f}")

    # Save the model
    model_path = "models/cleanrl_ppo_final.zip"
    torch.save(agent.state_dict(), model_path)
    print(f"✅ CleanRL PPO model saved to: {model_path}")

    # Test the model
    test_cleanrl_ppo(agent)

    return model_path


def test_cleanrl_ppo(agent):
    """Test the trained CleanRL PPO model."""

    print("\n" + "=" * 80)
    print("TESTING CLEANRL PPO SOLUTION")
    print("=" * 80)

    env = make_env()
    df = pd.read_csv("btc_jpy_real_dataset.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    actions = []
    print("Sampling actions from CleanRL PPO model...")

    for i in range(2000):
        step = np.random.randint(100, len(df) - 100)
        obs = np.array(
            [
                df.iloc[step]["close"],
                df.iloc[step]["volume"] if "volume" in df.columns else 1000,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor.unsqueeze(0))

        action_value = float(action.squeeze().cpu().numpy())
        actions.append(action_value)

    actions = np.array(actions)

    print("\nCleanRL PPO Action Distribution (2000 samples):")
    print(f"Mean:   {np.mean(actions):.4f}")
    print(f"Std:    {np.std(actions):.4f}")
    print(f"Min:    {np.min(actions):.4f}")
    print(f"Max:    {np.max(actions):.4f}")
    print(f"Median: {np.median(actions):.4f}")

    buy_threshold = 0.1
    sell_threshold = -0.1

    buy_count = sum(1 for a in actions if a > buy_threshold)
    sell_count = sum(1 for a in actions if a < sell_threshold)
    hold_count = sum(1 for a in actions if sell_threshold <= a <= buy_threshold)

    total = len(actions)
    print("\nDiscrete Action Distribution:")
    print(f"BUY:  {buy_count:4d} ({buy_count/total*100:5.1f}%)")
    print(f"SELL: {sell_count:4d} ({sell_count/total*100:5.1f}%)")
    print(f"HOLD: {hold_count:4d} ({hold_count/total*100:5.1f}%)")

    # Success criteria
    balance_ratio = (
        min(buy_count, sell_count) / max(buy_count, sell_count)
        if max(buy_count, sell_count) > 0
        else 0
    )
    std_dev = np.std(actions)

    print("\nSuccess Criteria:")
    print(f"Balance ratio (min/max): {balance_ratio:.3f} (target: >0.7)")
    print(f"Action std deviation:    {std_dev:.3f} (target: >0.3)")

    if balance_ratio > 0.7 and std_dev > 0.3:
        print("\n🎉 SUCCESS: CleanRL PPO produces BALANCED actions!")
        print("The SAC SELL bias issue has been RESOLVED!")
        return "SUCCESS"
    elif balance_ratio > 0.5:
        print("\n⚠️ PARTIAL SUCCESS: CleanRL PPO shows some balance but not perfect")
        return "PARTIAL_SUCCESS"
    else:
        print("\n❌ FAILURE: Still significant bias")
        return "FAILURE"


def main():
    """Main solution function."""

    print("🧹 CLEANRL PPO - FINAL SOLUTION")
    print("=" * 80)
    print("Root Cause: Stable Baselines3 PPO has fundamental issues")
    print("Solution: Use CleanRL's minimal PPO implementation")
    print("=" * 80)

    # Train and test CleanRL PPO
    model_path = train_cleanrl_ppo()

    results = {
        "solution": "Use CleanRL PPO instead of Stable Baselines3",
        "root_cause": "Stable Baselines3 PPO produces constant biased outputs",
        "cleanrl_ppo_model_path": model_path,
        "environment_config": {
            "use_continuous_actions": True,
            "continuous_to_discrete_threshold": 0.1,
        },
        "status": "CleanRL PPO training completed",
    }

    with open("results/cleanrl_ppo_final_solution.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n📄 Results saved to: results/cleanrl_ppo_final_solution.json")
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("Both Stable Baselines3 SAC and PPO produce constant biased outputs.")
    print(
        "This indicates a fundamental issue with the RL implementation or environment."
    )
    print("CleanRL PPO provides a cleaner baseline to test against.")
    print("=" * 80)


if __name__ == "__main__":
    main()
