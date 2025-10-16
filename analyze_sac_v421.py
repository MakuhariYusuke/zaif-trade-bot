import sys
sys.path.append('.')
from ztb.analysis.analyze_backtest import analyze_backtest
from ztb.training.unified_trainer.configs.sac_v421_balanced_trading_config import config

# Run backtest analysis for SAC v421
results = analyze_backtest(
    model_path='models/sac_v421_balanced_trading.zip',
    config=config,
    output_dir='results/sac_v421_analysis'
)
print('Analysis completed successfully!')