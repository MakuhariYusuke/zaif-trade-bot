#!/usr/bin/env python3
"""
Analyze training logs for entropy, value_loss, policy_loss
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def analyze_training_logs():
    """Analyze training logs from tensorboard"""

    # Find latest training log
    tensorboard_dir = Path("tensorboard")
    subdirs = [d for d in tensorboard_dir.iterdir() if d.is_dir()]

    # Sort by modification time (newest first)
    subdirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    latest_log_dir = None
    for subdir in subdirs:
        if subdir.name.startswith("PPO_") or subdir.name.startswith("tb_"):
            latest_log_dir = subdir
            break

    if not latest_log_dir:
        print("No training logs found in tensorboard directory")
        # Try to find any subdirectory
        if subdirs:
            latest_log_dir = subdirs[0]
            print(f"Using latest directory: {latest_log_dir}")
        else:
            print("No tensorboard directories found")
            return

    print(f"Analyzing logs from: {latest_log_dir}")

    # Load tensorboard logs
    event_acc = EventAccumulator(str(latest_log_dir))
    event_acc.Reload()

    # Available scalars
    scalars = event_acc.Tags()['scalars']
    print(f"Available scalars: {scalars}")

    # Ensure scalars is a list
    if not isinstance(scalars, list):
        print(f"Unexpected scalars type: {type(scalars)}")
        return

    # Look for training-related metrics with different naming patterns
    training_tags = [
        'train/entropy_loss', 'train/value_loss', 'train/policy_gradient_loss', 'train/approx_kl',
        'train/entropy', 'train/value', 'train/policy', 'train/kl',
        'loss/entropy', 'loss/value', 'loss/policy', 'loss/kl',
        'entropy', 'value_loss', 'policy_loss', 'kl_div'
    ]

    # Extract key metrics
    metrics = {}

    for tag in training_tags:
        if tag in scalars:
            events = event_acc.Scalars(tag)
            metrics[tag] = {
                'steps': [e.step for e in events],
                'values': [e.value for e in events]
            }
            print(f"{tag}: {len(events)} data points")
        else:
            print(f"{tag}: not found")

    # If no training metrics found, show all available metrics
    if not metrics:
        print("\nNo standard training metrics found. Available metrics:")
        for tag in scalars:
            events = event_acc.Scalars(tag)
            print(f"  {tag}: {len(events)} data points")
            if len(events) > 0:
                print(f"    Sample values: {events[0].value:.6f} -> {events[-1].value:.6f}")

    # Plot available metrics
    if metrics:
        plt.figure(figsize=(12, 8))
        plot_idx = 1

        for tag, data in metrics.items():
            if plot_idx > 4:
                break
            plt.subplot(2, 2, plot_idx)
            plt.plot(data['steps'], data['values'], alpha=0.7)
            plt.title(tag)
            plt.xlabel('Training Steps')
            plt.ylabel('Value')
            plt.grid(True, alpha=0.3)
            plot_idx += 1

        plt.tight_layout()
        plt.savefig('training_logs_analysis.png', dpi=150, bbox_inches='tight')
        print(f"Training logs analysis saved to: training_logs_analysis.png")
        # plt.show()  # Commented out to prevent blocking in console environment

        # Summary statistics
        print("\n=== Training Metrics Summary ===")
        for tag, data in metrics.items():
            if data['values']:
                values = np.array(data['values'])
                print(f"{tag}:")
                print(f"  Initial: {values[0]:.6f}")
                print(f"  Final: {values[-1]:.6f}")
                print(f"  Mean: {np.mean(values):.6f}")
                print(f"  Std: {np.std(values):.6f}")
                print(f"  Min: {np.min(values):.6f}")
                print(f"  Max: {np.max(values):.6f}")

    # Check for entropy-related metrics
    entropy_found = False
    for tag in metrics.keys():
        if 'entropy' in tag.lower():
            entropy_found = True
            final_entropy = np.mean(metrics[tag]['values'][-10:]) if len(metrics[tag]['values']) >= 10 else np.mean(metrics[tag]['values'])
            print(".4f")
            if abs(final_entropy) < 0.1:
                print("⚠️  WARNING: Entropy is very close to zero - possible collapse!")
            elif abs(final_entropy) < 0.5:
                print("⚠️  WARNING: Entropy is relatively low - may indicate limited exploration")
            else:
                print("✅ Entropy looks healthy")

            print("\n=== Entropy Coefficient Recommendations ===")
            if abs(final_entropy) < 0.1:
                print("Current ent_coef is too low. Try increasing to 0.1-0.2 for more exploration.")
            elif abs(final_entropy) > 1.5:
                print("Current ent_coef might be too high. Try decreasing to 0.01-0.05 for more exploitation.")
            else:
                print("Current ent_coef seems reasonable.")
            break

    if not entropy_found:
        print("No entropy-related metrics found in the logs.")

if __name__ == "__main__":
    analyze_training_logs()