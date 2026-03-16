#!/usr/bin/env python3
"""
Phase 1 Smoke Tests: ランダム特徴量撤廃と reward/balance 分離の検証

実行:
    python tests/v456/test_phase1_fixes.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
from ztb.config.environment_config import get_training_config

CONFIG = get_training_config()


def test_missing_feature_raises_error():
    """❌ 不足特徴量は ValueError を raise する"""
    print("\n" + "="*80)
    print("TEST 1: Missing feature detection")
    print("="*80)
    
    # 不完全な DataFrame を作成
    df_incomplete = pd.DataFrame({
        'open': [100, 101, 102],
        'close': [101, 102, 103],
        'base_0': [1, 2, 3],
        # MTF 特徴量なし
    })
    
    try:
        env = FastIntradayEnvV456(
            df=df_incomplete,
            base_feature_columns=[f'base_{i}' for i in range(30)],
            mtf_feature_columns=[f'mtf_{i}' for i in range(27)],
            regime_feature_columns=[f'regime_{i}' for i in range(13)],
            initial_balance=100000,
            max_position=0.01,
        )
        print("❌ FAILED: Should have raised ValueError for missing features")
        return False
    except ValueError as e:
        print(f"✓ PASSED: ValueError raised as expected")
        print(f"  Error message: {str(e)[:100]}...")
        return True


def test_reward_balance_separation():
    """✓ reward と balance が独立している"""
    print("\n" + "="*80)
    print("TEST 2: Reward/Balance Separation")
    print("="*80)
    
    # 完全な特徴量でダミーデータを作成
    n_steps = 1000
    base_cols = [f'base_{i}' for i in range(30)]
    mtf_cols = [f'mtf_{i}' for i in range(27)]
    regime_cols = [f'regime_{i}' for i in range(13)]
    
    df = pd.DataFrame({
        'open': 100 + np.random.randn(n_steps) * 5,
        'high': 101 + np.random.randn(n_steps) * 5,
        'low': 99 + np.random.randn(n_steps) * 5,
        'close': 100.5 + np.random.randn(n_steps) * 5,
        'volume': np.ones(n_steps) * 1000,
        'atr': np.ones(n_steps) * 2,
        'impact_proxy': np.ones(n_steps) * 0.01,
        **{col: np.random.randn(n_steps) for col in base_cols},
        **{col: np.random.randn(n_steps) for col in mtf_cols},
        **{col: np.random.randn(n_steps) for col in regime_cols},
    })
    
    env = FastIntradayEnvV456(
        df=df,
        base_feature_columns=base_cols,
        mtf_feature_columns=mtf_cols,
        regime_feature_columns=regime_cols,
        initial_balance=100000,
        max_position=0.01,
        drawdown_limit=0.30,
    )
    
    # リセット
    obs, info = env.reset()
    print(f"Reset successful")
    print(f"  Initial balance: {env.balance:,.0f} JPY")
    
    # 10ステップ実行して報酬の分布を確認
    rewards = []
    balances = []
    
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        balances.append(env.balance)
        
        if done or truncated:
            break
    
    rewards = np.array(rewards)
    balances = np.array(balances)
    
    print(f"\nReward statistics (10 steps):")
    print(f"  Mean:   {rewards.mean():.4f}")
    print(f"  Std:    {rewards.std():.4f}")
    print(f"  Min:    {rewards.min():.4f}")
    print(f"  Max:    {rewards.max():.4f}")
    print(f"  Range:  [{rewards.min():.4f}, {rewards.max():.4f}]")
    
    # 報酬が [-0.1, 0.1] 範囲であるか確認
    in_range = (rewards >= -0.1) & (rewards <= 0.1)
    pct_in_range = in_range.sum() / len(rewards) * 100
    
    if pct_in_range < 100:
        print(f"❌ WARNING: {100 - pct_in_range:.1f}% of rewards outside [-0.1, 0.1] range")
    else:
        print(f"✓ 100% of rewards in [-0.1, 0.1] range")
    
    print(f"\nBalance trajectory:")
    print(f"  Initial: {100000:,.0f} JPY")
    print(f"  Final:   {env.balance:,.0f} JPY")
    print(f"  Change:  {(env.balance - 100000):+,.0f} JPY ({(env.balance / 100000 - 1) * 100:+.2f}%)")
    
    # バランスの変動が不自然でないことを確認（初期値の ±20% 程度）
    balance_change = abs(env.balance - 100000) / 100000
    if balance_change < 0.20:
        print(f"✓ Balance change reasonable: {balance_change*100:.1f}%")
        return True
    else:
        print(f"❌ Balance change too large: {balance_change*100:.1f}%")
        return False


def test_episode_length_varies():
    """✓ エピソード長が可変"""
    print("\n" + "="*80)
    print("TEST 3: Episode Length Variation")
    print("="*80)
    
    n_steps = 2000
    base_cols = [f'base_{i}' for i in range(30)]
    mtf_cols = [f'mtf_{i}' for i in range(27)]
    regime_cols = [f'regime_{i}' for i in range(13)]
    
    df = pd.DataFrame({
        'open': 100 + np.random.randn(n_steps) * 5,
        'high': 101 + np.random.randn(n_steps) * 5,
        'low': 99 + np.random.randn(n_steps) * 5,
        'close': 100.5 + np.random.randn(n_steps) * 5,
        'volume': np.ones(n_steps) * 1000,
        'atr': np.ones(n_steps) * 2,
        'impact_proxy': np.ones(n_steps) * 0.01,
        **{col: np.random.randn(n_steps) for col in base_cols},
        **{col: np.random.randn(n_steps) for col in mtf_cols},
        **{col: np.random.randn(n_steps) for col in regime_cols},
    })
    
    env = FastIntradayEnvV456(
        df=df,
        base_feature_columns=base_cols,
        mtf_feature_columns=mtf_cols,
        regime_feature_columns=regime_cols,
        initial_balance=100000,
        max_position=0.01,
        max_steps=100,  # Max steps で打ち切り
        drawdown_limit=0.30,
    )
    
    lengths = []
    end_reasons = []
    
    for episode in range(10):
        obs, info = env.reset()
        steps = 0
        done = False
        terminated = False
        
        while not (done or terminated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            steps += 1
            if steps > 200:  # 無限ループ防止
                break
        
        lengths.append(steps)
        
        if done:
            end_reasons.append("drawdown_limit")
        elif truncated:
            end_reasons.append("max_steps")
        else:
            end_reasons.append("unknown")
    
    lengths = np.array(lengths)
    
    print(f"Episode length statistics (10 episodes):")
    print(f"  Mean:   {lengths.mean():.1f} steps")
    print(f"  Std:    {lengths.std():.1f}")
    print(f"  Min:    {lengths.min()}")
    print(f"  Max:    {lengths.max()}")
    print(f"  Median: {np.median(lengths):.1f}")
    
    print(f"\nEnd reasons:")
    from collections import Counter
    reason_counts = Counter(end_reasons)
    for reason, count in reason_counts.items():
        print(f"  {reason}: {count}")
    
    # 有意な変動がある
    if lengths.std() > 5:
        print(f"\n✓ PASSED: Episode lengths vary (std={lengths.std():.1f})")
        return True
    else:
        print(f"\n❌ FAILED: Episode lengths too uniform (std={lengths.std():.1f})")
        return False


def main():
    print("\n" + "="*80)
    print("PHASE 1 SMOKE TESTS")
    print("="*80)
    
    results = {
        'test_missing_feature_raises_error': test_missing_feature_raises_error(),
        'test_reward_balance_separation': test_reward_balance_separation(),
        'test_episode_length_varies': test_episode_length_varies(),
    }
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All Phase 1 tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
