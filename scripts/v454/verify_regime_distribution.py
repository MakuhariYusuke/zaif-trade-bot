import os
import sys
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from collections import Counter

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from ztb.analysis.market_regime_classifier import MarketRegimeClassifier

def verify_regime_distribution():
    # Load config
    config_path = project_root / "config" / "v454" / "sac_v454_phaseC_config.json"
    print(f"Loading config from {config_path}...")
    with open(config_path, "r") as f:
        full_config = json.load(f)
    
    # Extract regime classifier config
    # The structure in config is environment -> advanced_market_regime -> regime_classifier_config
    regime_config = full_config.get("environment", {}).get("advanced_market_regime", {}).get("regime_classifier_config", {})
    
    print("Regime Classifier Config:")
    print(json.dumps(regime_config, indent=2))
    
    # Load data
    data_path = project_root / "data" / "btc_jpy_1m_v454.csv"
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Ensure datetime index if needed, though classifier uses integer index
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    
    print("Initializing MarketRegimeClassifier...")
    classifier = MarketRegimeClassifier(regime_config)
    
    print("Detecting regimes (analyzing all steps)...")
    
    regime_counts = Counter()
    
    # Analyze the whole dataset or a significant portion
    # To be fast but accurate, let's analyze every step but maybe skip the very beginning (warmup)
    start_idx = 200 # Warmup
    
    # Use a step size of 1 to get exact distribution, or larger for speed
    # 17000 rows is small enough to do all
    step_size = 1 
    indices = range(start_idx, len(df), step_size)
    
    print(f"Analyzing {len(indices)} steps...")
    
    for i in tqdm(indices):
        try:
            # detect_regime expects the dataframe and the current integer index
            result = classifier.detect_regime(df, i)
            if result:
                regime_counts[result.primary_regime.value] += 1
        except Exception as e:
            # print(f"Error at index {i}: {e}")
            pass
            
    total_steps = sum(regime_counts.values())
    print(f"\nTotal classified steps: {total_steps}")
    
    print("\nRegime Distribution:")
    print(f"{'Regime':<30} | {'Count':<10} | {'Share':<10}")
    print("-" * 56)
    
    sorted_regimes = sorted(regime_counts.items(), key=lambda x: x[1], reverse=True)
    
    for regime, count in sorted_regimes:
        share = (count / total_steps) * 100 if total_steps > 0 else 0
        print(f"{regime:<30} | {count:<10} | {share:.2f}%")
        
    # Save results
    output_path = project_root / "backtest_results" / "v454_regime_distribution_phaseC_verification.json"
    os.makedirs(output_path.parent, exist_ok=True)
    
    results = {
        "total_steps": total_steps,
        "distribution": {k: v for k, v in sorted_regimes},
        "config": regime_config
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    verify_regime_distribution()
