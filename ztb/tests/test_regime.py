import numpy as np
import pandas as pd
from regime_evaluation import RegimeEvaluator

# Load sample data
df = pd.read_csv("ml-dataset-enhanced.csv", nrows=10000)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")

# Initialize evaluator
evaluator = RegimeEvaluator()

# Classify regimes
regime_labels, regime_counts = evaluator.classify_market_regime(df)
print("Regime counts:", regime_counts)

# For demo, create dummy actions (mostly SELL to simulate bias)
np.random.seed(42)
actions = np.random.choice(
    [0, 1, 2], size=len(regime_labels), p=[0.1, 0.05, 0.85]
)  # SELL bias

# Analyze
result = evaluator.analyze_regime_performance(
    {"test": "dummy"}, df, regime_labels, {"test": actions}
)

# Print action distributions
for regime in ["trend", "range", "high_vol", "low_vol"]:
    metrics = result.regime_metrics["test"][regime]
    print(f"{regime}: {metrics.action_distribution}")
