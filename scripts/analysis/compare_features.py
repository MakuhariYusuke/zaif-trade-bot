import pandas as pd
from ztb.features.sac_v427_feature_engineering import SACv427FeatureEngineer

# Load data
df = pd.read_csv('data/btc_jpy_real_dataset.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# Generate v427 features with full set
engineer_v427 = SACv427FeatureEngineer()
features_v427_full = engineer_v427.generate_v427_features(df, feature_set='full')
print('v427 full features:', len(features_v427_full.columns))

# Generate v427 features with no_harmful set (should exclude dividends and stock_splits)
features_v427_no_harmful = engineer_v427.generate_v427_features(df, feature_set='no_harmful')
print('v427 no_harmful features:', len(features_v427_no_harmful.columns))

# Check if dividends and stock_splits are in the data
print('Has dividends:', 'dividends' in df.columns)
print('Has stock_splits:', 'stock splits' in df.columns)