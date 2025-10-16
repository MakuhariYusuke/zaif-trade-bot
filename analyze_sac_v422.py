import sys
import json
sys.path.append('.')
from ztb.analysis.analyze_backtest import BacktestAnalyzer

# Load config from JSON
with open('config/sac_v422_balanced_trading_config.json', 'r') as f:
    config = json.load(f)

# Run backtest analysis for SAC v422
analyzer = BacktestAnalyzer('results/sac_v422_backtest.json')
report = analyzer.generate_comprehensive_report()
print(report)
print('Analysis completed successfully!')