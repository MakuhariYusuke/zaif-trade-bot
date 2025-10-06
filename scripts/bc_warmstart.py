#!/usr/bin/env python3
"""
Behavioral Cloning (BC) Warmstart for Policy Head.

Trains policy head only with simple rule-based SELL labels to
ensure SELL logits are active from the start.

Usage:
    python scripts/bc_warmstart.py \
        --data ml-dataset-mirrored.csv \
        --model models/ppo_base.zip \
        --output models/bc_init_policy.zip \
        --steps 10000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sb3_contrib import MaskablePPO

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class BCDataset(Dataset):
    """Dataset for behavioral cloning."""
    
    def __init__(self, features: np.ndarray, actions: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.actions = torch.LongTensor(actions)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.actions[idx]


def create_rule_based_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Create simple rule-based labels for BC warmstart.
    
    Rule: trend_ratio < 1.0 AND RSI < 40 → SELL
          trend_ratio > 1.0 AND RSI > 60 → BUY
          Otherwise → HOLD
    
    Args:
        df: Input dataframe
        
    Returns:
        Array of actions (0=HOLD, 1=BUY, 2=SELL)
    """
    actions = np.zeros(len(df), dtype=int)  # Default: HOLD
    
    # Try to use existing features
    has_trend = "trend_ratio" in df.columns
    has_rsi = "rsi_14" in df.columns or "rsi" in df.columns
    
    if has_trend and has_rsi:
        trend = df["trend_ratio"].values
        rsi = df.get("rsi_14", df.get("rsi")).values
        
        # SELL condition
        sell_mask = (trend < 1.0) & (rsi < 40)
        actions[sell_mask] = 2
        
        # BUY condition
        buy_mask = (trend > 1.0) & (rsi > 60)
        actions[buy_mask] = 1
    
    elif "action" in df.columns:
        # Fallback: use existing labels (e.g., from mirror augmentation)
        actions = df["action"].values
    
    return actions


def bc_warmstart(
    model: MaskablePPO,
    train_loader: DataLoader,
    steps: int = 10000,
    lr: float = 5e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Perform BC warmstart on policy head only.
    
    Args:
        model: PPO model
        train_loader: Training data loader
        steps: Number of training steps
        lr: Learning rate
        device: Device to use
    """
    # Freeze value head and feature extractor
    for param in model.policy.value_net.parameters():
        param.requires_grad = False
    
    for param in model.policy.features_extractor.parameters():
        param.requires_grad = False
    
    # Only train action head
    optimizer = optim.Adam(model.policy.action_net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.policy.to(device)
    model.policy.train()
    
    step = 0
    losses = []
    
    print("\nStarting BC warmstart...")
    print(f"Device: {device}")
    print(f"Target steps: {steps}")
    print()
    
    while step < steps:
        for features, actions in train_loader:
            if step >= steps:
                break
            
            features = features.to(device)
            actions = actions.to(device)
            
            # Forward pass
            with torch.no_grad():
                features_extracted = model.policy.features_extractor(features)
                if model.policy.share_features_extractor:
                    latent_pi, _ = model.policy.mlp_extractor(features_extracted)
                else:
                    latent_pi = model.policy.mlp_extractor.forward_actor(features_extracted)
            
            # Action logits (only this has grad)
            logits = model.policy.action_net(latent_pi)
            
            # Cross-entropy loss
            loss = criterion(logits, actions)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            step += 1
            
            if step % 1000 == 0:
                avg_loss = np.mean(losses[-1000:])
                print(f"Step {step}/{steps}, Loss: {avg_loss:.4f}")
    
    # Unfreeze all parameters
    for param in model.policy.parameters():
        param.requires_grad = True
    
    print(f"\n✅ BC warmstart complete. Final loss: {np.mean(losses[-100:]):.4f}")


def verify_sell_logits(
    model: MaskablePPO,
    test_features: np.ndarray,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Verify SELL logits are competitive after BC.
    
    Args:
        model: Trained model
        test_features: Test features
        device: Device
    """
    model.policy.to(device)
    model.policy.eval()
    
    with torch.no_grad():
        features_tensor = torch.FloatTensor(test_features).to(device)
        features_extracted = model.policy.features_extractor(features_tensor)
        
        if model.policy.share_features_extractor:
            latent_pi, _ = model.policy.mlp_extractor(features_extracted)
        else:
            latent_pi = model.policy.mlp_extractor.forward_actor(features_extracted)
        
        logits = model.policy.action_net(latent_pi).cpu().numpy()
    
    # Compute mean logits per action
    mean_logits = np.mean(logits, axis=0)
    
    print("\nLogit verification:")
    print(f"  Mean logits: HOLD={mean_logits[0]:.3f}, BUY={mean_logits[1]:.3f}, SELL={mean_logits[2]:.3f}")
    
    # Check if SELL is competitive (within 0.1 of max)
    max_logit = np.max(mean_logits)
    sell_logit = mean_logits[2]
    
    is_competitive = (max_logit - sell_logit) <= 0.1
    
    if is_competitive:
        print(f"  ✅ SELL logit is competitive (gap: {max_logit - sell_logit:.3f} ≤ 0.1)")
    else:
        print(f"  ⚠️  SELL logit may be weak (gap: {max_logit - sell_logit:.3f} > 0.1)")
    
    return is_competitive


def main():
    parser = argparse.ArgumentParser(description="BC warmstart for policy head")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Training data CSV (preferably mirror-augmented)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Base PPO model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for BC-initialized model",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="BC training steps (default: 10000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size (default: 256)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate (default: 5e-4)",
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("BC Warmstart for Policy Head")
    print("="*60)
    print(f"Data: {args.data}")
    print(f"Base model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print()
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv(args.data)
    print(f"  Size: {len(df)} rows")
    
    # Create rule-based labels
    print("\nCreating rule-based labels...")
    actions = create_rule_based_labels(df)
    
    action_counts = np.bincount(actions)
    print(f"  Label distribution:")
    print(f"    HOLD: {action_counts[0]} ({action_counts[0]/len(actions)*100:.1f}%)")
    print(f"    BUY: {action_counts[1]} ({action_counts[1]/len(actions)*100:.1f}%)")
    print(f"    SELL: {action_counts[2]} ({action_counts[2]/len(actions)*100:.1f}%)")
    
    # Prepare features (exclude action column if present)
    feature_cols = [col for col in df.columns if col != "action"]
    features = df[feature_cols].values
    
    print(f"\n  Features: {features.shape[1]} columns")
    
    # Create dataset and loader
    dataset = BCDataset(features, actions)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Load model
    print("\nLoading base model...")
    model = MaskablePPO.load(args.model)
    print("  ✅ Model loaded")
    
    # BC warmstart
    bc_warmstart(
        model=model,
        train_loader=train_loader,
        steps=args.steps,
        lr=args.lr,
    )
    
    # Verify SELL logits
    test_indices = np.random.choice(len(features), size=min(1000, len(features)), replace=False)
    verify_sell_logits(model, features[test_indices])
    
    # Save
    print(f"\nSaving BC-initialized model...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"✅ Saved to: {args.output}")
    print()
    print("Ready for PPO training with SELL-aware initialization.")


if __name__ == "__main__":
    main()
