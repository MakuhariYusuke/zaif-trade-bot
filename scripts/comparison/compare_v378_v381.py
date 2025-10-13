#!/usr/bin/env python3
"""
Compare v378 vs v381_revised training results
"""
import json
import os
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

def extract_metrics(log_dir):
    """Extract key metrics from TensorBoard logs"""
    if not Path(log_dir).exists():
        print(f"Warning: {log_dir} not found")
        return None
    
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()
    
    metrics = {}
    
    # Extract reward progression
    if 'rollout/ep_rew_mean' in ea.Tags()['scalars']:
        rewards = ea.Scalars('rollout/ep_rew_mean')
        metrics['rewards'] = [{'step': r.step, 'value': r.value} for r in rewards]
        metrics['final_reward'] = rewards[-1].value if rewards else None
        metrics['best_reward'] = max(r.value for r in rewards) if rewards else None
        metrics['reward_trend_last10'] = [r.value for r in rewards[-10:]] if len(rewards) >= 10 else []
    
    # Extract explained variance
    if 'train/explained_variance' in ea.Tags()['scalars']:
        evs = ea.Scalars('train/explained_variance')
        metrics['explained_variance'] = [{'step': e.step, 'value': e.value} for e in evs]
        metrics['final_ev'] = evs[-1].value if evs else None
        metrics['avg_ev_last10'] = np.mean([e.value for e in evs[-10:]]) if len(evs) >= 10 else None
    
    # Extract action counts
    if 'train/pan_action_counts' in ea.Tags()['tensors']:
        # This is a tensor, need special handling
        pass
    
    # Extract policy gradient loss
    if 'train/policy_gradient_loss' in ea.Tags()['scalars']:
        pg_losses = ea.Scalars('train/policy_gradient_loss')
        metrics['final_pg_loss'] = pg_losses[-1].value if pg_losses else None
    
    # Extract value loss
    if 'train/value_loss' in ea.Tags()['scalars']:
        v_losses = ea.Scalars('train/value_loss')
        metrics['final_value_loss'] = v_losses[-1].value if v_losses else None
    
    # Extract approx KL
    if 'train/approx_kl' in ea.Tags()['scalars']:
        kls = ea.Scalars('train/approx_kl')
        metrics['final_approx_kl'] = kls[-1].value if kls else None
        metrics['avg_approx_kl'] = np.mean([k.value for k in kls]) if kls else None
    
    return metrics

def main() -> None:
    log_dirs = {
        'v378_scale': 'runs/ppo_reward_v378_scale',
        'v381_revised': 'runs/ppo_reward_v381_revised_profit_focused'
    }
    
    results = {}
    
    for name, log_dir in log_dirs.items():
        print(f"\n{'='*60}")
        print(f"Extracting metrics from: {name}")
        print(f"{'='*60}")
        metrics = extract_metrics(log_dir)
        if metrics:
            results[name] = metrics
            print(f"✅ Final Reward: {metrics.get('final_reward', 'N/A'):.2f}")
            print(f"✅ Best Reward: {metrics.get('best_reward', 'N/A'):.2f}")
            print(f"✅ Final Explained Variance: {metrics.get('final_ev', 'N/A'):.4f}")
            print(f"✅ Avg EV (last 10): {metrics.get('avg_ev_last10', 'N/A'):.4f}")
            print(f"✅ Final Policy Gradient Loss: {metrics.get('final_pg_loss', 'N/A'):.4f}")
            print(f"✅ Final Value Loss: {metrics.get('final_value_loss', 'N/A'):.2f}")
            print(f"✅ Final Approx KL: {metrics.get('final_approx_kl', 'N/A'):.4f}")
            print(f"✅ Avg Approx KL: {metrics.get('avg_approx_kl', 'N/A'):.4f}")
    
    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    if 'v378_scale' in results and 'v381_revised' in results:
        v378 = results['v378_scale']
        v381 = results['v381_revised']
        
        reward_improvement = ((v381['final_reward'] - v378['final_reward']) / abs(v378['final_reward'])) * 100
        ev_improvement = ((v381['final_ev'] - v378['final_ev']) / max(abs(v378['final_ev']), 0.01)) * 100
        
        print(f"\n📊 Reward Comparison:")
        print(f"  v378:  {v378['final_reward']:.2f} (best: {v378['best_reward']:.2f})")
        print(f"  v381:  {v381['final_reward']:.2f} (best: {v381['best_reward']:.2f})")
        print(f"  Improvement: {reward_improvement:+.1f}%")
        
        print(f"\n📊 Explained Variance:")
        print(f"  v378:  {v378['final_ev']:.4f} (avg last10: {v378['avg_ev_last10']:.4f})")
        print(f"  v381:  {v381['final_ev']:.4f} (avg last10: {v381['avg_ev_last10']:.4f})")
        print(f"  Improvement: {ev_improvement:+.1f}%")
        
        print(f"\n📊 Training Stability:")
        print(f"  v378 Approx KL:  {v378['avg_approx_kl']:.4f}")
        print(f"  v381 Approx KL:  {v381['avg_approx_kl']:.4f}")
        
        print(f"\n📊 Loss Comparison:")
        print(f"  v378 PG Loss:    {v378['final_pg_loss']:.4f}")
        print(f"  v381 PG Loss:    {v381['final_pg_loss']:.4f}")
        print(f"  v378 Value Loss: {v378['final_value_loss']:.2f}")
        print(f"  v381 Value Loss: {v381['final_value_loss']:.2f}")
    
    # Save results to JSON
    output_path = 'v378_vs_v381_comparison.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Detailed results saved to: {output_path}")

if __name__ == '__main__':
    main()
